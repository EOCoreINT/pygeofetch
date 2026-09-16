"""
Classification — supervised (Random Forest) and unsupervised (K-means)
pixel classification for multi-band rasters.

Fills a real, verified gap: pygeofetch had no image classification
capability at all before this module -- every spectral index and
processing tool assumed the user would classify results with a
separate library. This plugs directly into the existing raster I/O
conventions (_safe_read_band/_safe_write_band, BigTIFF-safe output)
so a classification result is immediately usable with
PostProcessor.vectorize() and zonal_stats(), which already expected
"a classification raster" as real input.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pygeofetch.processing.base import (
    ProcessingResult,
    _require_numpy,
    _require_rasterio,
    _safe_write_band,
    _timed,
    clip_output_if_requested,
)

logger = logging.getLogger(__name__)


def _require_sklearn():
    try:
        import sklearn  # noqa: F401

        return sklearn
    except ImportError:
        msg = 'scikit-learn is required for classification: pip install "pygeofetch[ml]"'
        raise ImportError(msg) from None


def _require_geopandas():
    try:
        import geopandas as gpd

        return gpd
    except ImportError:
        msg = 'geopandas is required for training-sample extraction: pip install "pygeofetch[geo]"'
        raise ImportError(msg) from None


def _require_joblib():
    try:
        import joblib

        return joblib
    except ImportError:
        msg = 'joblib is required to save/load trained models: pip install "pygeofetch[ml]"'
        raise ImportError(msg) from None


def _read_stack(inputs: list, ref_shape=None):
    """
    Real, minimal multi-band stack reader for classification -- reads
    every input band fully (classification needs the whole array to
    fit K-means/Random Forest; chunking only applies to predict(),
    once a model already exists). Returns (stack, profile) where
    stack has shape (n_bands, height, width).
    """
    rasterio = _require_rasterio()
    np = _require_numpy()

    bands = []
    profile = None
    for p in inputs:
        with rasterio.open(p) as src:
            data = src.read(1).astype(np.float32)
            if profile is None:
                profile = src.profile.copy()
                ref_shape = data.shape
            elif data.shape != ref_shape:
                msg = (
                    f"classification: input grids do not match "
                    f"({p}: {data.shape} vs expected {ref_shape}) -- "
                    f"align inputs to the same real grid first "
                    f"(e.g. via Preprocessor.reproject/resample)."
                )
                raise ValueError(msg)
            bands.append(data)
    return np.stack(bands, axis=0), profile


class Classifier:
    """
    Pixel classification for multi-band raster stacks.

    Example (unsupervised)::

        client.classify.kmeans(inputs=[b02, b03, b04, b08], n_clusters=5,
                                output="clusters.tif")

    Example (supervised)::

        client.classify.train(
            inputs=[b02, b03, b04, b08],
            training_data="training_samples.geojson", class_field="land_cover",
            output_model="model.joblib",
        )
        client.classify.predict(
            inputs=[b02, b03, b04, b08], model="model.joblib",
            output="land_cover_map.tif",
        )
    """

    @_timed
    def kmeans(
        self,
        inputs: list,
        n_clusters: int = 5,
        output: str | None = None,
        random_state: int = 42,
        bbox=None,
        geometry=None,
        geometry_crs: str = "EPSG:4326",
    ) -> ProcessingResult:
        """
        Unsupervised K-means clustering across a real multi-band stack.
        No training data needed -- groups pixels by spectral similarity
        into n_clusters classes (arbitrary integer labels; cluster 0
        has no inherent meaning like "water" until you inspect it).

        Args:
            inputs:     Single-band raster paths, all sharing the same
                        real grid.
            n_clusters: Number of real clusters to fit.
            output:     Output path.
            bbox, geometry, geometry_crs: Optional post-clip, same
                        handling as Preprocessor.clip().
        """
        sklearn = _require_sklearn()
        np = _require_numpy()
        from sklearn.cluster import KMeans

        stack, profile = _read_stack(inputs)
        n_bands, height, width = stack.shape

        pixels = stack.reshape(n_bands, -1).T  # (n_pixels, n_bands)
        valid_mask = np.all(np.isfinite(pixels), axis=1)

        labels = np.full(pixels.shape[0], -1, dtype=np.int32)
        if valid_mask.any():
            km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels[valid_mask] = km.fit_predict(pixels[valid_mask])

        result_arr = labels.reshape(height, width).astype(np.float32)
        result_arr[result_arr == -1] = np.nan

        out_path = Path(output) if output else Path(inputs[0]).with_name(
            Path(inputs[0]).stem + "_kmeans.tif"
        )
        _safe_write_band(result_arr, profile, out_path, nodata=-9999.0)
        clip_output_if_requested(out_path, bbox=bbox, geometry=geometry, geometry_crs=geometry_crs)

        logger.info("K-means (%d clusters) → %s", n_clusters, out_path.name)
        return ProcessingResult(
            success=True, operation="kmeans", output_path=out_path,
            input_path=Path(inputs[0]),
            metadata={"n_clusters": n_clusters, "valid_pixel_fraction": float(valid_mask.mean())},
        )

    @_timed
    def train(
        self,
        inputs: list,
        training_data: str | Path,
        class_field: str,
        output_model: str | Path,
        model_type: str = "random_forest",
        n_estimators: int = 100,
        test_size: float = 0.2,
        random_state: int = 42,
        **model_kwargs: Any,
    ) -> ProcessingResult:
        """
        Train a real supervised classifier from labeled vector training
        samples (points or polygons) and a multi-band raster stack.

        Args:
            inputs:        Single-band raster paths, all the same grid.
            training_data: Path to a real vector file (GeoJSON,
                           Shapefile, etc.) with labeled training
                           geometries -- points sample one pixel each;
                           polygons sample every real pixel their
                           footprint covers.
            class_field:   Attribute column holding the real class
                           label (string or integer).
            output_model:  Where to save the trained model (joblib).
            model_type:    "random_forest" (default) or "svm".
            n_estimators:  Real Random Forest tree count (ignored for SVM).
            test_size:     Fraction of samples held out for a real,
                           honest accuracy report -- not used to train.

        Returns a ProcessingResult with real accuracy metrics in
        ``.metadata`` (never hidden or assumed -- computed on the
        held-out test split every time).
        """
        sklearn = _require_sklearn()
        np = _require_numpy()
        gpd = _require_geopandas()
        joblib = _require_joblib()
        rasterio = _require_rasterio()
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report

        stack, profile = _read_stack(inputs)
        n_bands, height, width = stack.shape
        transform = profile["transform"]

        samples_gdf = gpd.read_file(training_data)
        if class_field not in samples_gdf.columns:
            msg = (
                f"train: class_field {class_field!r} not found in "
                f"{Path(training_data).name}'s real columns: "
                f"{list(samples_gdf.columns)}"
            )
            raise ValueError(msg)

        raster_crs = profile.get("crs")
        if raster_crs is not None and samples_gdf.crs is not None and str(samples_gdf.crs) != str(raster_crs):
            samples_gdf = samples_gdf.to_crs(raster_crs)

        X, y = [], []
        for _, row in samples_gdf.iterrows():
            geom = row.geometry
            label = row[class_field]
            if geom.geom_type == "Point":
                col, row_idx = ~transform * (geom.x, geom.y)
                row_idx, col = int(row_idx), int(col)
                if 0 <= row_idx < height and 0 <= col < width:
                    pixel = stack[:, row_idx, col]
                    if np.all(np.isfinite(pixel)):
                        X.append(pixel)
                        y.append(label)
            else:
                # Real, direct approach: rasterize this one polygon onto
                # the stack's own real grid, then take every pixel it covers.
                from rasterio.features import geometry_mask
                poly_mask = ~geometry_mask([geom.__geo_interface__], out_shape=(height, width),
                                            transform=transform, invert=False)
                rows_idx, cols_idx = np.where(poly_mask)
                for r_i, c_i in zip(rows_idx, cols_idx):
                    pixel = stack[:, r_i, c_i]
                    if np.all(np.isfinite(pixel)):
                        X.append(pixel)
                        y.append(label)

        if len(X) < 10:
            msg = (
                f"train: only {len(X)} real, valid training pixels extracted "
                f"from {Path(training_data).name} -- too few to train a "
                f"meaningful model. Check that the training geometries "
                f"actually overlap the raster's real extent and CRS."
            )
            raise ValueError(msg)

        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y)) > 1 else None,
        )

        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, **model_kwargs)
        elif model_type == "svm":
            model = SVC(**model_kwargs)
        else:
            msg = f"train: unknown model_type {model_type!r} -- use 'random_forest' or 'svm'"
            raise ValueError(msg)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        output_model = Path(output_model)
        output_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "n_bands": n_bands, "model_type": model_type}, output_model)

        logger.info(
            "Trained %s on %d real samples (%d held out) -- accuracy=%.3f",
            model_type, len(X_train), len(X_test), accuracy,
        )
        return ProcessingResult(
            success=True, operation="train", output_path=output_model,
            input_path=Path(training_data),
            metadata={
                "model_type": model_type, "n_train_samples": len(X_train),
                "n_test_samples": len(X_test), "test_accuracy": accuracy,
                "classification_report": report,
            },
        )

    @_timed
    def predict(
        self,
        inputs: list,
        model: str | Path,
        output: str | None = None,
        bbox=None,
        geometry=None,
        geometry_crs: str = "EPSG:4326",
    ) -> ProcessingResult:
        """
        Apply a previously-trained model (from .train()) to a real
        multi-band raster stack, producing a classification map.

        Raises a clear error if the model expects a different number
        of bands than `inputs` provides, rather than silently running
        sklearn's own less-specific shape-mismatch error.
        """
        joblib = _require_joblib()
        np = _require_numpy()

        saved = joblib.load(model)
        clf, expected_bands = saved["model"], saved["n_bands"]

        stack, profile = _read_stack(inputs)
        n_bands, height, width = stack.shape
        if n_bands != expected_bands:
            msg = (
                f"predict: model was trained on {expected_bands} real bands, "
                f"but {n_bands} were provided in `inputs` -- pass the same "
                f"real bands, in the same order, used for training."
            )
            raise ValueError(msg)

        pixels = stack.reshape(n_bands, -1).T
        valid_mask = np.all(np.isfinite(pixels), axis=1)

        predictions = np.full(pixels.shape[0], np.nan, dtype=np.float32)
        if valid_mask.any():
            preds = clf.predict(pixels[valid_mask])
            try:
                predictions[valid_mask] = preds.astype(np.float32)
            except (TypeError, ValueError):
                # Real, honest string-label case: encode to real integer
                # codes rather than silently failing to write a float raster.
                classes = sorted(set(preds))
                code_map = {c: i for i, c in enumerate(classes)}
                predictions[valid_mask] = np.array([code_map[p] for p in preds], dtype=np.float32)

        result_arr = predictions.reshape(height, width)

        out_path = Path(output) if output else Path(inputs[0]).with_name(
            Path(inputs[0]).stem + "_classified.tif"
        )
        _safe_write_band(result_arr, profile, out_path, nodata=-9999.0)
        clip_output_if_requested(out_path, bbox=bbox, geometry=geometry, geometry_crs=geometry_crs)

        logger.info("Prediction → %s", out_path.name)
        return ProcessingResult(
            success=True, operation="predict", output_path=out_path,
            input_path=Path(inputs[0]),
            metadata={"valid_pixel_fraction": float(valid_mask.mean())},
        )
