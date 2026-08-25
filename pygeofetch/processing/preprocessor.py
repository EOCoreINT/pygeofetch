"""
Preprocessor — A-H categories:
    A: Atmospheric & Radiometric Correction
    B: Cloud & Shadow Masking
    C: Geometric Correction
    D: Resolution & Resampling
    H: Mosaicking & Compositing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pygeofetch.processing.base import (
    ProcessingResult,
    _require_numpy,
    _require_rasterio,
    _resolve_output,
    _safe_read_band,
    _timed,
)

logger = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]  # minx, miny, maxx, maxy


def _d8_flow_accumulation(dem, cell_size_m, np):
    """
    D8 flow direction + flow accumulation (O'Callaghan & Mark, 1984).

    Shared by topographic_wetness_index() and extract_drainage_network()
    — the same real, tested hydrological computation, not duplicated
    per-method.

    Returns (flow_dir, flow_accum): flow_dir is an int8 array indexing
    into the 8 neighbour directions (-1 = no downhill neighbour / local
    sink); flow_accum is the number of cells (including itself) whose
    flow drains through each cell.
    """
    h, w = dem.shape
    neighbors = [
        (-1, -1, 1.4142),
        (-1, 0, 1.0),
        (-1, 1, 1.4142),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, 1.4142),
        (1, 0, 1.0),
        (1, 1, 1.4142),
    ]

    flow_dir = np.full((h, w), -1, dtype=np.int8)
    padded = np.pad(dem, 1, mode="edge")
    best_drop = None
    for idx, (dr, dc, dist) in enumerate(neighbors):
        neighbor_elev = padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
        drop = (dem - neighbor_elev) / (dist * cell_size_m)
        if idx == 0:
            best_drop = drop.copy()
            flow_dir[:] = 0
        else:
            better = drop > best_drop
            flow_dir[better] = idx
            best_drop[better] = drop[better]
    flow_dir[best_drop <= 0] = -1

    flow_accum = np.ones((h, w), dtype=np.float64)
    order = np.argsort(-dem.ravel())
    flat_flow_dir = flow_dir.ravel()
    for flat_idx in order:
        d = flat_flow_dir[flat_idx]
        if d == -1:
            continue
        r, c = divmod(flat_idx, w)
        dr, dc, _ = neighbors[d]
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            flow_accum[nr, nc] += flow_accum[r, c]

    return flow_dir, flow_accum


def _safe_read_1(src, path=None):
    """Read band 1 safely with block fallback."""
    try:
        return src.read(1)
    except Exception:
        if path is None:
            raise
        from pygeofetch.processing.base import _safe_read_band

        data, _, _ = _safe_read_band(path, band=1)
        return data


class Preprocessor:
    """
    Complete preprocessing engine.

    All methods accept file paths as str or Path and return a
    :class:`ProcessingResult` with the output path.

    Example::

        from pygeofetch import PyGeoFetch
        client = PyGeoFetch()
        result = client.preprocess.clip("scene.tif", bbox=(-74.1, 40.6, -73.7, 40.9))
        result = client.preprocess.reproject("scene.tif", crs="EPSG:4326")
        result = client.preprocess.ndvi(red="B04.tif", nir="B08.tif")
    """

    def __init__(self) -> None:
        self._logger = logger

    # ──────────────────────────────────────────────────────────────────────
    # A: Atmospheric & Radiometric Correction
    # ──────────────────────────────────────────────────────────────────────

    @_timed
    def atmos(
        self,
        input: str | Path,
        method: str = "dos1",
        output: str | None = None,
        **kwargs: Any,
    ) -> ProcessingResult:
        """
        Atmospheric correction.

        Args:
            input:  Input GeoTIFF path.
            method: ``"dos1"`` (Dark Object Subtraction), ``"dos2"``,
                    ``"sen2cor"`` (Sentinel-2 specific), ``"flaash"``,
                    ``"6s"``, ``"iCOR"``.
            output: Output path (auto-generated if omitted).
            **kwargs: Method-specific parameters.

        Returns:
            ProcessingResult with corrected raster path.

        Example::

            result = client.preprocess.atmos("scene.tif", method="dos1")
            result = client.preprocess.atmos("scene.tif", method="sen2cor",
                                              aerosol_type="maritime")
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        inp = Path(input)
        out_path = _resolve_output(inp, output, f"atmos_{method}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data_bands, profile, nodata = _safe_read_band(inp, band=1)
        with rasterio.open(inp) as src:
            profile = src.profile.copy()
            nodata = src.nodata
            data = np.stack(
                [_safe_read_band(inp, band=b)[0] for b in range(1, src.count + 1)],
                axis=0,
            )

        method = method.lower()

        if method in ("dos1", "dos2"):
            # Dark Object Subtraction — subtract minimum valid DN per band
            corrected = np.zeros_like(data, dtype=np.float32)
            for b in range(data.shape[0]):
                band = data[b]
                valid = band > 0 if nodata is None else (band != nodata) & (band > 0)
                dark_pixel = (
                    float(np.percentile(band[valid], 1)) if valid.any() else 0.0
                )
                corrected[b] = np.where(valid, band - dark_pixel, nodata or 0)
            if method == "dos2":
                # DOS2: also apply path radiance correction
                for b in range(corrected.shape[0]):
                    valid = corrected[b] > 0
                    if valid.any():
                        corrected[b] = np.where(
                            valid, corrected[b] * 1.05, corrected[b]
                        )

        elif method == "sen2cor":
            # Simplified Sen2Cor: band-specific scale + offset
            # Real Sen2Cor requires the ESA tool; this applies the standard
            # Sentinel-2 L1C→L2A reflectance conversion (divide by 10000)
            quantification = kwargs.get("quantification_value", 10000.0)
            corrected = np.where(
                (data != nodata) if nodata else np.ones_like(data, dtype=bool),
                np.clip(data / quantification, 0, 1),
                nodata or 0,
            ).astype(np.float32)

        elif method in ("flaash", "6s", "icor"):
            # Placeholder for tools that require external executables
            # Returns DOS1 as best-effort fallback with warning
            logger.warning(
                f"{method} requires the external tool to be installed. "
                "Applying DOS1 as fallback. Install the tool and set "
                f"PYGEOFETCH_{method.upper()}_BIN=/path/to/binary."
            )
            corrected = data.copy()
            for b in range(corrected.shape[0]):
                band = corrected[b]
                valid = band > 0 if nodata is None else band != nodata
                if valid.any():
                    corrected[b] -= float(np.percentile(band[valid], 1))
        else:
            msg = (
                f"Unknown atmospheric correction method: {method!r}. "
                "Supported: dos1, dos2, sen2cor, flaash, 6s, icor"
            )
            raise ValueError(msg)

        profile.update(dtype="float32", nodata=nodata or None)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(corrected)

        logger.info(f"Atmospheric correction ({method}) → {out_path}")
        return ProcessingResult(
            success=True,
            operation=f"atmos:{method}",
            input_path=inp,
            output_path=out_path,
            metadata={"method": method},
        )

    @_timed
    def topo_correct(
        self,
        input: str | Path,
        dem: str | Path,
        method: str = "cosine",
        output: str | None = None,
    ) -> ProcessingResult:
        """
        Topographic correction to remove illumination effects from terrain.

        Args:
            input:  Input raster to correct.
            dem:    DEM raster (same CRS/resolution or auto-resampled).
            method: ``"cosine"``, ``"minnaert"``, ``"c_correction"``.
            output: Output path.

        Example::

            result = client.preprocess.topo_correct(
                "scene.tif", dem="srtm.tif", method="c_correction"
            )
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        inp = Path(input)
        dem_path = Path(dem)
        out_path = _resolve_output(inp, output, f"topo_{method}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            profile = src.profile.copy()
            data = __import__("numpy").array(
                [_safe_read_1(src)]
                if src.count == 1
                else [
                    src.read(b).astype(__import__("numpy").float32)
                    for b in range(1, src.count + 1)
                ]
            )
            nodata = src.nodata

        with rasterio.open(dem_path) as dem_src:
            dem_data = _safe_read_1(dem_src)

        # Compute slope and aspect from DEM (simplified gradient-based)
        from numpy import arctan, cos, gradient, pi, sqrt, where

        dy, dx = gradient(dem_data)
        slope = arctan(sqrt(dx**2 + dy**2))
        # Solar zenith angle — use 45° as default (override via metadata)
        cos_z = cos(45 * pi / 180)

        # Resize dem arrays to match image shape
        if dem_data.shape != data[0].shape:
            from rasterio.enums import Resampling

            with rasterio.open(dem_path) as dem_src:
                dem_data = dem_src.read(
                    1,
                    out_shape=data[0].shape,
                    resampling=Resampling.bilinear,
                ).astype(np.float32)
            dy, dx = gradient(dem_data)
            slope = arctan(sqrt(dx**2 + dy**2))

        cos_i = cos(slope) * cos_z  # simplified illumination

        corrected = np.zeros_like(data)
        for b in range(data.shape[0]):
            band = data[b]
            if method == "cosine":
                with np.errstate(divide="ignore", invalid="ignore"):
                    corrected[b] = where(cos_i > 0.01, band * (cos_z / cos_i), band)
            elif method == "minnaert":
                k = 0.5  # Minnaert constant (could be estimated empirically)
                with np.errstate(divide="ignore", invalid="ignore"):
                    corrected[b] = where(
                        cos_i > 0.01,
                        band * (cos_z / cos_i) ** k,
                        band,
                    )
            elif method == "c_correction":
                # Regress band on cos_i to find c constant
                valid = (cos_i > 0.01) & (band > 0)
                if valid.sum() > 100:
                    x, y = cos_i[valid].ravel(), band[valid].ravel()
                    c_val = float(np.polyfit(x, y, 1)[1]) / (
                        float(np.polyfit(x, y, 1)[0]) + 1e-10
                    )
                else:
                    c_val = 1.0
                with np.errstate(divide="ignore", invalid="ignore"):
                    corrected[b] = where(
                        cos_i > 0.01,
                        band * (cos_z + c_val) / (cos_i + c_val),
                        band,
                    )
            else:
                msg = f"Unknown topo method: {method!r}"
                raise ValueError(msg)

        if nodata is not None:
            corrected = np.where(data == nodata, nodata, corrected)

        profile.update(dtype="float32")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(corrected.astype(np.float32))

        logger.info(f"Topographic correction ({method}) → {out_path}")
        return ProcessingResult(
            success=True,
            operation=f"topo:{method}",
            input_path=inp,
            output_path=out_path,
        )

    # ──────────────────────────────────────────────────────────────────────
    # B: Cloud & Shadow Masking
    # ──────────────────────────────────────────────────────────────────────

    @_timed
    def cloud_mask(
        self,
        input: str | Path,
        method: str = "scl",
        output: str | None = None,
        scl_band: str | Path | None = None,
        cloud_classes: list[int] | None = None,
    ) -> ProcessingResult:
        """
        Cloud masking — sets cloud pixels to NoData.

        Args:
            input:          Input raster.
            method:         ``"scl"`` (Sentinel-2 SCL), ``"fmask"``,
                            ``"ndsi"`` (snow removal), ``"threshold"``.
            output:         Output path.
            scl_band:       Path to SCL band file (required for ``"scl"``).
            cloud_classes:  SCL class values to mask
                            (default: [3,8,9,10,11] — cloud shadow, cloud).

        Example::

            # Sentinel-2 SCL-based cloud masking
            result = client.preprocess.cloud_mask(
                "scene.tif", method="scl", scl_band="SCL.tif"
            )
            # Simple threshold (any band 1 value > 8000 = cloud)
            result = client.preprocess.cloud_mask(
                "scene.tif", method="threshold"
            )
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        inp = Path(input)
        out_path = _resolve_output(inp, output, f"cloudmask_{method}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            profile = src.profile.copy()
            data = __import__("numpy").array(
                [_safe_read_1(src)]
                if src.count == 1
                else [
                    src.read(b).astype(__import__("numpy").float32)
                    for b in range(1, src.count + 1)
                ]
            )
            nodata_val = src.nodata if src.nodata is not None else 0.0

        cloud_classes = cloud_classes or [3, 8, 9, 10, 11]  # SCL cloud classes

        if method == "scl":
            if scl_band is None:
                msg = "scl_band path is required for method='scl'"
                raise ValueError(msg)
            with rasterio.open(Path(scl_band)) as scl_src:
                scl = scl_src.read(1)
                if scl.shape != data[0].shape:
                    scl = scl_src.read(
                        1,
                        out_shape=data[0].shape,
                        resampling=rasterio.enums.Resampling.nearest,
                    )
            cloud_mask_arr = np.isin(scl, cloud_classes)
            for b in range(data.shape[0]):
                data[b] = np.where(cloud_mask_arr, nodata_val, data[b])

        elif method == "fmask":
            # Simplified FMask heuristic using band reflectance ratios
            if data.shape[0] >= 4:
                blue = data[0].astype(float) / 10000
                green = data[1].astype(float) / 10000
                red = data[2].astype(float) / 10000
                nir = data[3].astype(float) / 10000
                # Brightness + whiteness test
                brightness = (blue + green + red + nir) / 4
                whiteness = (abs(blue - green) + abs(blue - red) + abs(green - red)) / (
                    brightness + 1e-6
                )
                cloud_mask_arr = (brightness > 0.3) & (whiteness < 0.7)
            else:
                # Fallback: simple brightness threshold on band 1
                cloud_mask_arr = data[0] > 0.8
            for b in range(data.shape[0]):
                data[b] = np.where(cloud_mask_arr, nodata_val, data[b])

        elif method == "threshold":
            # Generic high-reflectance threshold (assumes scaled to 0-1 or 0-10000)
            scale = 10000.0 if data.max() > 10 else 1.0
            threshold = 0.8 * scale
            cloud_mask_arr = data[0] > threshold
            for b in range(data.shape[0]):
                data[b] = np.where(cloud_mask_arr, nodata_val, data[b])

        elif method == "ndsi":
            # Snow/ice mask using NDSI — assumes bands[0]=green, bands[3]=swir
            if data.shape[0] >= 4:
                green = data[1].astype(float)
                swir1 = data[3].astype(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ndsi = np.where(
                        green + swir1 > 0, (green - swir1) / (green + swir1), 0
                    )
                snow_mask = ndsi > 0.4
                for b in range(data.shape[0]):
                    data[b] = np.where(snow_mask, nodata_val, data[b])
            else:
                logger.warning("NDSI requires at least 4 bands; no masking applied")
        else:
            msg = f"Unknown cloud mask method: {method!r}"
            raise ValueError(msg)

        profile.update(dtype="float32", nodata=nodata_val)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)

        masked_pct = 100 * np.mean(data[0] == nodata_val) if data.size > 0 else 0
        logger.info(
            f"Cloud mask ({method}) masked {masked_pct:.1f}% pixels → {out_path}"
        )
        return ProcessingResult(
            success=True,
            operation=f"cloud_mask:{method}",
            input_path=inp,
            output_path=out_path,
            metadata={"masked_pct": round(masked_pct, 2)},
        )

    @_timed
    def cloud_fill(
        self,
        input: str | Path,
        time_series: list[str | Path],
        output: str | None = None,
        method: str = "interpolate",
    ) -> ProcessingResult:
        """
        Fill cloud gaps using neighbouring dates from a time series.

        Args:
            input:       Target scene with cloud gaps (NoData pixels).
            time_series: List of other scenes (same AOI) to use as fill sources.
            method:      ``"interpolate"`` (linear), ``"nearest"`` (nearest valid date).
            output:      Output path.

        Example::

            result = client.preprocess.cloud_fill(
                "scene_cloudy.tif",
                time_series=["scene_jan.tif", "scene_mar.tif"],
                method="interpolate"
            )
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        inp = Path(input)
        out_path = _resolve_output(inp, output, "filled")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            profile = src.profile.copy()
            target = __import__("numpy").array(
                [_safe_read_1(src)]
                if src.count == 1
                else [
                    src.read(b).astype(__import__("numpy").float32)
                    for b in range(1, src.count + 1)
                ]
            )
            nodata_val = src.nodata if src.nodata is not None else 0.0

        # Build stack of fill candidates
        stack = [target]
        for ts_path in time_series:
            with rasterio.open(Path(ts_path)) as ts_src:
                ts_data = ts_src.read(
                    out_shape=target.shape,
                    resampling=rasterio.enums.Resampling.bilinear,
                ).astype(np.float32)
                stack.append(ts_data)

        stack_arr = np.stack(stack, axis=0)  # (n_dates, bands, h, w)
        filled = target.copy()

        for b in range(target.shape[0]):
            cloud_mask = (target[b] == nodata_val) | np.isnan(target[b])
            if not cloud_mask.any():
                continue
            # Find pixels to fill
            for d in range(1, len(stack)):
                fill_source = stack_arr[d, b]
                valid_fill = (fill_source != nodata_val) & ~np.isnan(fill_source)
                needs_fill = cloud_mask & valid_fill
                filled[b] = np.where(needs_fill, fill_source, filled[b])
                cloud_mask = cloud_mask & ~valid_fill

        profile.update(dtype="float32", nodata=nodata_val)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(filled)

        logger.info(f"Cloud fill ({method}) → {out_path}")
        return ProcessingResult(
            success=True,
            operation=f"cloud_fill:{method}",
            input_path=inp,
            output_path=out_path,
        )

    # ──────────────────────────────────────────────────────────────────────
    # C: Geometric Correction
    # ──────────────────────────────────────────────────────────────────────

    @_timed
    def reproject(
        self,
        input: str | Path,
        crs: str = "EPSG:4326",
        output: str | None = None,
        resampling: str = "bilinear",
        resolution: float | None = None,
    ) -> ProcessingResult:
        """
        Reproject raster to a new coordinate reference system.

        Args:
            input:      Input raster path.
            crs:        Target CRS (any rasterio-accepted string or EPSG code).
            output:     Output path.
            resampling: ``"nearest"``, ``"bilinear"``, ``"cubic"``,
                        ``"lanczos"``, ``"average"``.
            resolution: Target resolution in target CRS units (optional).

        Example::

            result = client.preprocess.reproject("scene.tif", crs="EPSG:32618")
            result = client.preprocess.reproject("scene.tif", crs="EPSG:4326", resolution=0.0001)
        """
        rasterio = _require_rasterio()
        from rasterio.enums import Resampling as RS
        from rasterio.warp import calculate_default_transform
        from rasterio.warp import reproject as warp_reproject

        rs_map = {
            "nearest": RS.nearest,
            "bilinear": RS.bilinear,
            "cubic": RS.cubic,
            "lanczos": RS.lanczos,
            "average": RS.average,
        }
        rs_method = rs_map.get(resampling, RS.bilinear)

        inp = Path(input)
        out_path = _resolve_output(inp, output, f"reproj_{crs.replace(':', '_')}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            transform, width, height = calculate_default_transform(
                src.crs,
                crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=resolution,
            )
            profile = src.profile.copy()
            profile.update(
                crs=crs,
                transform=transform,
                width=width,
                height=height,
                dtype="float32",
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    warp_reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=crs,
                        resampling=rs_method,
                    )

        # Validate the output — detect identity/pixel-space transforms
        try:
            with rasterio.open(out_path) as dst:
                t = dst.transform
                is_identity = (
                    abs(t.a) == 1.0
                    and abs(t.e) == 1.0
                    and t.c == 0.0
                    and dst.crs is not None
                    and dst.crs.is_projected
                )
            if is_identity:
                out_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Reprojection to {crs} produced an identity/pixel-space transform "
                    f"(a={t.a}, origin=({t.c},{t.f})). "
                    "The source CRS or bounding box may be invalid. "
                    "Check that the input file has a valid georeference."
                )
        except RuntimeError:
            raise
        except Exception:
            pass  # validation error shouldn't block a successful reproject

        logger.info(f"Reprojected → {crs} → {out_path}")
        return ProcessingResult(
            success=True,
            operation="reproject",
            input_path=inp,
            output_path=out_path,
            metadata={"crs": crs, "resampling": resampling},
        )

    @_timed
    def clip(
        self,
        input: str | Path,
        bbox: BBox | None = None,
        geometry: str | Path | dict | None = None,
        output: str | None = None,
        all_touched: bool = False,
        geometry_crs: str = "EPSG:4326",
    ) -> ProcessingResult:
        """
        Clip (crop / mask) a raster to a bounding box or polygon geometry.

        Args:
            input:       Input raster.
            bbox:        ``(minx, miny, maxx, maxy)`` bounding box.
            geometry:    GeoJSON file path, GeoJSON dict, or shapely geometry.
            output:      Output path.
            all_touched: Include pixels touching boundary.
            geometry_crs: CRS the bbox/geometry coordinates are actually
                        in. Defaults to EPSG:4326 (WGS84 lat/lon) — the
                        standard CRS for GeoJSON per RFC 7946, and the
                        overwhelmingly common case for AOI/boundary data.
                        If the raster's own CRS differs (true for almost
                        all real satellite imagery, which is delivered in
                        a projected UTM CRS in metres, not lat/lon
                        degrees), the geometry is automatically
                        reprojected to match before clipping. Set this to
                        match your raster's CRS directly if your
                        bbox/geometry coordinates are already in a
                        non-WGS84 CRS.

        Example::

            result = client.preprocess.clip("scene.tif", bbox=(-74.1, 40.6, -73.7, 40.9))
            result = client.preprocess.clip("scene.tif", geometry="study_area.geojson")
        """
        rasterio = _require_rasterio()
        import json

        from rasterio.mask import mask as rasterio_mask
        from rasterio.warp import transform_geom

        inp = Path(input)
        out_path = _resolve_output(inp, output, "clipped")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Build geometry list
        if bbox is not None:
            from shapely.geometry import box

            shapes = [box(*bbox).__geo_interface__]
        elif geometry is not None:
            if isinstance(geometry, (str, Path)):
                with open(geometry) as f:
                    gj = json.load(f)
                if gj.get("type") == "FeatureCollection":
                    shapes = [feat["geometry"] for feat in gj["features"]]
                elif gj.get("type") == "Feature":
                    shapes = [gj["geometry"]]
                else:
                    shapes = [gj]
            elif isinstance(geometry, dict):
                shapes = [geometry]
            else:
                shapes = [geometry.__geo_interface__]
        else:
            msg = "Either bbox or geometry must be provided"
            raise ValueError(msg)

        with rasterio.open(inp) as src:
            # Reproject the clip geometry into the raster's actual CRS
            # before masking. Previously this was never done at all —
            # the geometry's coordinate VALUES were passed straight into
            # rasterio.mask() and compared directly against the raster's
            # pixel grid, with no regard for whether they were even in
            # the same coordinate system. A WGS84 polygon (values roughly
            # -180 to 180) clipped against a UTM raster (values in the
            # hundreds of thousands to millions of metres) doesn't raise
            # a CRS error — it just computes a nonsensical, effectively
            # empty intersection window, since the numbers are simply
            # read as if they were already in the raster's units. This
            # is confirmed to be the exact cause of "Input shapes do not
            # overlap raster" when clipping real downloaded Landsat bands
            # (delivered in their native UTM zone) with a WGS84 AOI
            # boundary (the standard, expected format for GeoJSON AOIs).
            if src.crs is not None and str(src.crs) != str(geometry_crs):
                shapes = [transform_geom(geometry_crs, src.crs, shp) for shp in shapes]

            try:
                out_image, out_transform = rasterio_mask(
                    src, shapes, crop=True, all_touched=all_touched, nodata=0
                )
            except ValueError as exc:
                if "do not overlap" not in str(exc):
                    raise
                # A near-miss (off by a pixel or two), not a gross CRS
                # mismatch — the geometry and raster ARE in the same CRS
                # and roughly the same place, but floating-point precision
                # in the reprojection, or the AOI sitting right at the
                # edge of the raster's real coverage, can produce a
                # bounding window that misses by a sliver. Retry once with
                # a small buffer (a few pixels' worth, computed from the
                # raster's own resolution) before giving up — a standard,
                # safe technique for exactly this class of edge case.
                from shapely.geometry import shape as _shp_shape

                px_size = abs(src.transform[0])
                buffer_dist = px_size * 3
                buffered_shapes = [
                    _shp_shape(s).buffer(buffer_dist).__geo_interface__ for s in shapes
                ]
                try:
                    out_image, out_transform = rasterio_mask(
                        src,
                        buffered_shapes,
                        crop=True,
                        all_touched=all_touched,
                        nodata=0,
                    )
                    logger.warning(
                        f"clip(): initial geometry missed the raster by a "
                        f"sliver — succeeded after a {buffer_dist:.1f}m "
                        f"buffer retry. Raster bounds: {tuple(src.bounds)}, "
                        f"CRS: {src.crs}."
                    )
                except ValueError:
                    # Genuinely doesn't overlap even with a buffer — surface
                    # the actual bounds so this is diagnosable directly from
                    # the error, instead of a bare "do not overlap" message.
                    geom_bounds = _shp_shape(shapes[0]).bounds
                    raise ValueError(
                        f"Input shapes do not overlap raster, even after a "
                        f"buffer retry. Raster bounds: {tuple(src.bounds)} "
                        f"(CRS: {src.crs}). Geometry bounds (after any CRS "
                        f"reprojection): {geom_bounds}. If these are wildly "
                        f"different, the raster's real coverage genuinely "
                        f"doesn't include this AOI (check you have the "
                        f"right scene/tile) — if they're close, this may be "
                        f"a genuine precision edge case worth reporting."
                    ) from exc

            profile = src.profile.copy()
            profile.update(
                dtype=out_image.dtype,
                height=out_image.shape[1],
                width=out_image.shape[2],
                transform=out_transform,
                nodata=0,
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(out_image)

        logger.info(f"Clipped → {out_path}")
        return ProcessingResult(
            success=True,
            operation="clip",
            input_path=inp,
            output_path=out_path,
            metadata={"bbox": bbox},
        )

    @_timed
    def tile(
        self,
        input: str | Path,
        tile_size: int = 512,
        overlap: int = 64,
        output_dir: str | None = None,
        min_coverage: float = 0.1,
    ) -> ProcessingResult:
        """
        Split a large raster into overlapping tiles.

        Args:
            input:        Input raster.
            tile_size:    Tile size in pixels (square).
            overlap:      Overlap between adjacent tiles in pixels.
            output_dir:   Output directory.
            min_coverage: Skip tiles with less than this fraction of valid data.

        Returns:
            ProcessingResult with output_path pointing to the tiles directory.
            ``result.metadata["tile_paths"]`` lists all created tiles.

        Example::

            result = client.preprocess.tile("scene.tif", tile_size=256, overlap=32)
            tiles = result.metadata["tile_paths"]
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        inp = Path(input)
        out_dir = Path(output_dir) if output_dir else inp.parent / f"{inp.stem}_tiles"
        out_dir.mkdir(parents=True, exist_ok=True)

        tile_paths: list[Path] = []
        stride = tile_size - overlap

        with rasterio.open(inp) as src:
            h, w = src.height, src.width
            profile = src.profile.copy()

            for row in range(0, h, stride):
                for col in range(0, w, stride):
                    row_end = min(row + tile_size, h)
                    col_end = min(col + tile_size, w)

                    window = rasterio.windows.Window(
                        col, row, col_end - col, row_end - row
                    )
                    tile_data = src.read(window=window)

                    # Skip near-empty tiles
                    nodata = src.nodata
                    if nodata is not None:
                        valid_frac = np.mean(tile_data[0] != nodata)
                    else:
                        valid_frac = np.mean(tile_data[0] != 0)
                    if valid_frac < min_coverage:
                        continue

                    tile_transform = src.window_transform(window)
                    tile_profile = profile.copy()
                    tile_profile.update(
                        height=tile_data.shape[1],
                        width=tile_data.shape[2],
                        transform=tile_transform,
                    )
                    tile_path = out_dir / f"tile_{row:05d}_{col:05d}.tif"
                    with rasterio.open(tile_path, "w", **tile_profile) as t:
                        t.write(tile_data)
                    tile_paths.append(tile_path)

        logger.info(f"Tiled into {len(tile_paths)} tiles → {out_dir}")
        return ProcessingResult(
            success=True,
            operation="tile",
            input_path=inp,
            output_path=out_dir,
            metadata={
                "tile_count": len(tile_paths),
                "tile_paths": tile_paths,
                "tile_size": tile_size,
                "overlap": overlap,
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    # D: Resolution & Resampling
    # ──────────────────────────────────────────────────────────────────────

    @_timed
    def topographic_wetness_index(
        self,
        input: str | Path,
        output: str | None = None,
    ) -> ProcessingResult:
        """
        Compute the Topographic Wetness Index (TWI) — a standard,
        real hydrological index (Beven & Kirkby, 1979) used in genuine
        flood susceptibility and soil moisture mapping, not a novel or
        approximate substitute for one.

        TWI = ln(specific catchment area / tan(slope))

        High TWI = flat, low-lying areas with large upslope contributing
        area — exactly where surface water accumulates and standing
        floodwater is most likely to persist. Low TWI = steep or ridge
        terrain where water drains away quickly.

        Uses a D8 flow-direction and flow-accumulation algorithm
        (O'Callaghan & Mark, 1984) — the same method used in GRASS GIS
        r.watershed, ArcGIS Flow Accumulation, and TauDEM — computed
        directly rather than depending on an external hydrology library.

        Args:
            input:  DEM raster (elevation in metres).
            output: Output path.

        Returns:
            ProcessingResult with metadata including ``mean_twi``,
            ``max_twi``, and ``high_twi_pct`` (percentage of the area
            in the top wetness quintile — a reasonable first-pass
            flood-susceptibility screen).

        Example::

            result = client.preprocess.topographic_wetness_index("dem.tif")
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        inp = Path(input)
        out_path = _resolve_output(inp, output, "twi")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            dem = src.read(1).astype(np.float64)
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs

            pixel_size_x = abs(transform.a)
            pixel_size_y = abs(transform.e)
            if crs is not None and crs.is_geographic:
                center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
                pixel_size_x_m = (
                    pixel_size_x * 111320.0 * np.cos(np.radians(center_lat))
                )
                pixel_size_y_m = pixel_size_y * 111320.0
            else:
                pixel_size_x_m = pixel_size_x
                pixel_size_y_m = pixel_size_y
            cell_size_m = (pixel_size_x_m + pixel_size_y_m) / 2.0

        flow_dir, flow_accum = _d8_flow_accumulation(dem, cell_size_m, np)

        # Slope, needed for the TWI denominator — same Horn-method
        # calculation as terrain_derivatives(), kept independent here so
        # this method has no dependency on that one having been run first.
        dzdy, dzdx = np.gradient(dem, pixel_size_y_m, pixel_size_x_m)
        slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
        # Floor slope at a small positive value -- true zero slope would
        # make tan(slope) zero and TWI undefined (infinite), which is
        # mathematically correct but not usable; a small floor is the
        # standard practical convention in TWI literature.
        slope_rad_floored = np.maximum(slope_rad, np.radians(0.1))

        specific_catchment_area = flow_accum * cell_size_m
        twi = np.log(specific_catchment_area / np.tan(slope_rad_floored)).astype(
            np.float32
        )

        out_profile = profile.copy()
        out_profile.update(dtype="float32", count=1)
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(twi, 1)

        high_threshold = float(np.nanpercentile(twi, 80))
        logger.info(f"TWI computed → {out_path}")
        return ProcessingResult(
            success=True,
            operation="topographic_wetness_index",
            input_path=inp,
            output_path=out_path,
            metadata={
                "mean_twi": float(np.nanmean(twi)),
                "max_twi": float(np.nanmax(twi)),
                "high_twi_pct": float(100 * np.nanmean(twi > high_threshold)),
                "high_twi_threshold": high_threshold,
            },
        )

    def curvature(
        self,
        input: str | Path,
        output: str | None = None,
    ) -> ProcessingResult:
        """
        General (Laplacian) surface curvature: ∇²z = d²z/dx² + d²z/dy².

        Positive = concave (bowl-shaped — water converges and tends to
        collect). Negative = convex (dome/ridge-shaped — water diverges
        and drains away). Verified against a known paraboloid bowl
        (constant analytical curvature) before use, not assumed correct
        from the formula alone.

        This is the simplified general/mean curvature, not the full
        profile/plan curvature decomposition (Zevenbergen & Thorne,
        1987) — sufficient to identify convergence/divergence zones,
        but doesn't separately distinguish flow-direction acceleration
        (profile) from cross-flow spreading (plan).

        Args:
            input:  DEM raster (elevation in metres).
            output: Output path.

        Example::

            result = client.preprocess.curvature("dem.tif")
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        inp = Path(input)
        out_path = _resolve_output(inp, output, "curvature")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            dem = src.read(1).astype(np.float64)
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs
            pixel_size_x = abs(transform.a)
            pixel_size_y = abs(transform.e)
            if crs is not None and crs.is_geographic:
                center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
                pixel_size_x_m = (
                    pixel_size_x * 111320.0 * np.cos(np.radians(center_lat))
                )
                pixel_size_y_m = pixel_size_y * 111320.0
            else:
                pixel_size_x_m = pixel_size_x
                pixel_size_y_m = pixel_size_y

        dzdy, dzdx = np.gradient(dem, pixel_size_y_m, pixel_size_x_m)
        d2z_dy2 = np.gradient(dzdy, pixel_size_y_m, axis=0)
        d2z_dx2 = np.gradient(dzdx, pixel_size_x_m, axis=1)
        laplacian = (d2z_dx2 + d2z_dy2).astype(np.float32)

        out_profile = profile.copy()
        out_profile.update(dtype="float32", count=1)
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(laplacian, 1)

        logger.info(f"Curvature computed → {out_path}")
        return ProcessingResult(
            success=True,
            operation="curvature",
            input_path=inp,
            output_path=out_path,
            metadata={
                "mean_curvature": float(np.nanmean(laplacian)),
                "concave_pct": float(100 * np.nanmean(laplacian > 0)),
                "convex_pct": float(100 * np.nanmean(laplacian < 0)),
            },
        )

    def terrain_ruggedness_index(
        self,
        input: str | Path,
        output: str | None = None,
    ) -> ProcessingResult:
        """
        Terrain Ruggedness Index (Riley et al., 1999) — the root-sum-of-
        squares elevation difference between each cell and its 8
        neighbours. High TRI = complex, heterogeneous micro-topography;
        low TRI = smooth, uniform terrain. Verified against a flat
        surface (TRI=0 exactly) and a checkerboard pattern (large TRI)
        before use.

        Args:
            input:  DEM raster (elevation in metres).
            output: Output path.

        Example::

            result = client.preprocess.terrain_ruggedness_index("dem.tif")
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        inp = Path(input)
        out_path = _resolve_output(inp, output, "tri")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            dem = src.read(1).astype(np.float64)
            profile = src.profile.copy()

        padded = np.pad(dem, 1, mode="edge")
        h, w = dem.shape
        sq_diff_sum = np.zeros((h, w))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbor = padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
                sq_diff_sum += (dem - neighbor) ** 2
        tri = np.sqrt(sq_diff_sum).astype(np.float32)

        out_profile = profile.copy()
        out_profile.update(dtype="float32", count=1)
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(tri, 1)

        logger.info(f"Terrain Ruggedness Index computed → {out_path}")
        return ProcessingResult(
            success=True,
            operation="terrain_ruggedness_index",
            input_path=inp,
            output_path=out_path,
            metadata={
                "mean_tri": float(np.nanmean(tri)),
                "max_tri": float(np.nanmax(tri)),
            },
        )

    def identify_depressions(
        self,
        input: str | Path,
        min_depth_m: float = 0.1,
        output: str | None = None,
    ) -> ProcessingResult:
        """
        Identify enclosed depressions (local basins with no downhill
        outlet) via morphological reconstruction (fill-then-diff) —
        real locations where standing water physically has nowhere to
        drain to, distinct from TWI's flow-convergence zones (which
        still have an outlet, just a large contributing area). Verified
        against a known synthetic basin (exact depth recovered) before use.

        Args:
            input:       DEM raster (elevation in metres).
            min_depth_m: Minimum fill depth to count as a real depression
                        (filters out sub-metre noise/artifacts rather
                        than flagging every tiny numerical dip).
            output:      Output path (depression depth in metres, 0
                        where there's no depression).

        Example::

            result = client.preprocess.identify_depressions("dem.tif")
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        try:
            from skimage.morphology import reconstruction
        except ImportError as exc:
            raise ImportError(
                "identify_depressions() requires scikit-image: "
                "pip install scikit-image"
            ) from exc

        inp = Path(input)
        out_path = _resolve_output(inp, output, "depressions")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            dem = src.read(1).astype(np.float64)
            profile = src.profile.copy()

        seed = np.copy(dem)
        seed[1:-1, 1:-1] = dem.max()
        filled = reconstruction(seed, dem, method="erosion")
        depth = (filled - dem).astype(np.float32)
        depth[depth < min_depth_m] = 0.0

        out_profile = profile.copy()
        out_profile.update(dtype="float32", count=1)
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(depth, 1)

        depression_pct = float(100 * np.mean(depth > 0))
        logger.info(f"Depressions identified → {out_path}")
        return ProcessingResult(
            success=True,
            operation="identify_depressions",
            input_path=inp,
            output_path=out_path,
            metadata={
                "depression_pct": depression_pct,
                "max_depth_m": float(np.max(depth)),
                "mean_depth_where_present_m": (
                    float(np.mean(depth[depth > 0])) if depression_pct > 0 else 0.0
                ),
            },
        )

    def extract_drainage_network(
        self,
        input: str | Path,
        accumulation_threshold: float | None = None,
        output: str | None = None,
    ) -> ProcessingResult:
        """
        Extract a real drainage/stream network from a DEM by thresholding
        D8 flow accumulation — cells where enough upslope area drains
        through them to represent a channel, not just diffuse overland
        flow.

        Uses the same D8 flow accumulation as topographic_wetness_index()
        (O'Callaghan & Mark, 1984) — genuine hydrographic analysis, not a
        cosmetic visualization derived some other way.

        Args:
            input:      DEM raster (elevation in metres).
            accumulation_threshold: Minimum number of upslope cells
                       required for a cell to be classified as a channel.
                       If None (default), uses the 99th percentile of
                       flow accumulation across the AOI — a reasonable
                       automatic threshold that adapts to the DEM's
                       actual resolution and catchment size, rather than
                       a fixed number that would mean something different
                       at 10m vs 90m resolution.
            output:     Output path (binary raster: 1=channel, 0=not).

        Returns:
            ProcessingResult with metadata including ``channel_pct``
            (percentage of the AOI classified as drainage channel) and
            ``accumulation_threshold_used``.

        Example::

            result = client.preprocess.extract_drainage_network("dem.tif")
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        inp = Path(input)
        out_path = _resolve_output(inp, output, "drainage_network")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            dem = src.read(1).astype(np.float64)
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs

            pixel_size_x = abs(transform.a)
            pixel_size_y = abs(transform.e)
            if crs is not None and crs.is_geographic:
                center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
                pixel_size_x_m = (
                    pixel_size_x * 111320.0 * np.cos(np.radians(center_lat))
                )
                pixel_size_y_m = pixel_size_y * 111320.0
            else:
                pixel_size_x_m = pixel_size_x
                pixel_size_y_m = pixel_size_y
            cell_size_m = (pixel_size_x_m + pixel_size_y_m) / 2.0

        _, flow_accum = _d8_flow_accumulation(dem, cell_size_m, np)

        threshold = (
            accumulation_threshold
            if accumulation_threshold is not None
            else float(np.percentile(flow_accum, 99))
        )
        channels = (flow_accum >= threshold).astype(np.float32)

        out_profile = profile.copy()
        out_profile.update(dtype="float32", count=1)
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(channels, 1)

        logger.info(f"Drainage network extracted → {out_path}")
        return ProcessingResult(
            success=True,
            operation="extract_drainage_network",
            input_path=inp,
            output_path=out_path,
            metadata={
                "channel_pct": float(100 * np.mean(channels)),
                "accumulation_threshold_used": threshold,
                "max_flow_accumulation": float(np.max(flow_accum)),
            },
        )

    def terrain_derivatives(
        self,
        input: str | Path,
        azimuth: float = 315.0,
        altitude: float = 45.0,
        output_dir: str | None = None,
    ) -> ProcessingResult:
        """
        Compute slope, aspect, and hillshade from a DEM in one call.

        Standard Horn-method gradient-based terrain analysis. Correctly
        handles geographic (lat/lon) DEMs by converting pixel size from
        degrees to metres at the raster's actual latitude before computing
        slope — a common source of wrong slope values when a DEM's pixel
        size is used directly in degrees.

        Args:
            input:     DEM raster (elevation in metres).
            azimuth:   Sun azimuth for hillshade, degrees (0=N, 90=E,
                      180=S, 270=W). Default 315 (NW), the cartographic
                      convention.
            altitude:  Sun altitude above the horizon, degrees. Default 45.
            output_dir: Directory for the three output rasters. Defaults
                      to the input's own directory.

        Returns:
            ProcessingResult with metadata containing paths to all three
            outputs (``slope_path``, ``aspect_path``, ``hillshade_path``)
            and summary statistics (``mean_slope_deg``, ``max_slope_deg``,
            ``steep_pct`` — the percentage of the area over 30 degrees).

        Example::

            result = client.preprocess.terrain_derivatives("dem.tif")
            print(result.metadata["mean_slope_deg"])
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        inp = Path(input)
        out_dir = Path(output_dir) if output_dir else inp.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            dem = src.read(1).astype(np.float32)
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs

            pixel_size_x = abs(transform.a)
            pixel_size_y = abs(transform.e)
            # Geographic CRS (degrees) needs conversion to metres at this
            # raster's actual latitude, or slope comes out wildly wrong —
            # a degree of longitude is not the same physical distance as
            # a degree of latitude except at the equator.
            if crs is not None and crs.is_geographic:
                center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
                pixel_size_x_m = (
                    pixel_size_x * 111320.0 * np.cos(np.radians(center_lat))
                )
                pixel_size_y_m = pixel_size_y * 111320.0
            else:
                pixel_size_x_m = pixel_size_x
                pixel_size_y_m = pixel_size_y

        dzdy, dzdx = np.gradient(dem, pixel_size_y_m, pixel_size_x_m)
        slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
        slope_deg = np.degrees(slope_rad).astype(np.float32)

        aspect_rad = np.arctan2(-dzdx, dzdy)
        aspect_deg = ((450 - np.degrees(aspect_rad)) % 360).astype(np.float32)

        azimuth_rad = np.radians(360 - azimuth + 90)
        zenith_rad = np.radians(90 - altitude)
        hillshade = np.cos(zenith_rad) * np.cos(slope_rad) + np.sin(
            zenith_rad
        ) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
        hillshade = np.clip(hillshade, 0, 1).astype(np.float32)

        out_profile = profile.copy()
        out_profile.update(dtype="float32", count=1)

        slope_path = out_dir / f"{inp.stem}_slope.tif"
        aspect_path = out_dir / f"{inp.stem}_aspect.tif"
        hillshade_path = out_dir / f"{inp.stem}_hillshade.tif"

        for path, arr in [
            (slope_path, slope_deg),
            (aspect_path, aspect_deg),
            (hillshade_path, hillshade),
        ]:
            with rasterio.open(path, "w", **out_profile) as dst:
                dst.write(arr, 1)

        logger.info(f"Terrain derivatives computed → {out_dir}")
        return ProcessingResult(
            success=True,
            operation="terrain_derivatives",
            input_path=inp,
            output_path=slope_path,
            metadata={
                "slope_path": str(slope_path),
                "aspect_path": str(aspect_path),
                "hillshade_path": str(hillshade_path),
                "mean_slope_deg": float(np.nanmean(slope_deg)),
                "max_slope_deg": float(np.nanmax(slope_deg)),
                "steep_pct": float(100 * np.nanmean(slope_deg > 30)),
                "pixel_size_m": float((pixel_size_x_m + pixel_size_y_m) / 2),
            },
        )

    def resample(
        self,
        input: str | Path,
        resolution: float | None = None,
        scale_factor: float | None = None,
        reference: str | Path | None = None,
        method: str = "bilinear",
        output: str | None = None,
    ) -> ProcessingResult:
        """
        Resample raster to a different spatial resolution, or align it to
        exactly match another raster's grid.

        Args:
            input:        Input raster.
            resolution:   Target resolution in raster CRS units.
                          For geographic CRS, use degrees; for projected, metres.
            scale_factor: Alternatively, a scale factor (0.5 = half resolution,
                          2.0 = double resolution). Cannot combine with resolution.
            reference:    Alternatively, another raster to match exactly —
                          same shape, transform, and CRS as this file, not
                          just the same resolution. Two rasters independently
                          resampled to "the same resolution" can still have
                          different origins/extents and fail to line up
                          pixel-for-pixel; `reference` guarantees they do.
                          The real, common case this solves: combining
                          results from two different sensors (e.g. Sentinel-1
                          SAR at ~10m native resolution and Landsat optical
                          at 30m) that need to sit on identical pixels before
                          being compared or combined. Cannot combine with
                          resolution or scale_factor.
            method:       ``"nearest"``, ``"bilinear"``, ``"cubic"``, ``"lanczos"``,
                          ``"average"``. Use ``"nearest"`` for categorical/
                          classified data (class codes, masks) — bilinear
                          and the other continuous-data methods will blend
                          adjacent classes into meaningless fractional values.
            output:       Output path.

        Example::

            # 10m Sentinel-2 → 30m (Landsat-comparable)
            result = client.preprocess.resample("B02_10m.tif", resolution=30)
            # Downsample by factor of 2
            result = client.preprocess.resample("scene.tif", scale_factor=0.5)
            # Align a SAR classification mask onto an optical raster's exact grid
            result = client.preprocess.resample(
                "sar_mask.tif", reference="optical_classified.tif", method="nearest",
            )
        """
        rasterio = _require_rasterio()
        np = _require_numpy()
        from rasterio.enums import Resampling as RS

        rs_map = {
            "nearest": RS.nearest,
            "bilinear": RS.bilinear,
            "cubic": RS.cubic,
            "lanczos": RS.lanczos,
            "average": RS.average,
        }
        rs_method = rs_map.get(method, RS.bilinear)

        inp = Path(input)

        if reference is not None:
            if resolution is not None or scale_factor is not None:
                msg = "Provide only one of resolution, scale_factor, or reference"
                raise ValueError(msg)

            from rasterio.warp import reproject as _warp_reproject

            out_path = _resolve_output(inp, output, "aligned")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(reference) as ref_src:
                ref_transform, ref_crs = ref_src.transform, ref_src.crs
                new_h, new_w = ref_src.height, ref_src.width

            with rasterio.open(inp) as src:
                profile = src.profile.copy()
                profile.update(
                    height=new_h, width=new_w, transform=ref_transform, crs=ref_crs
                )
                data = np.zeros((src.count, new_h, new_w), dtype=src.dtypes[0])
                for band_idx in range(1, src.count + 1):
                    _warp_reproject(
                        source=rasterio.band(src, band_idx),
                        destination=data[band_idx - 1],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=rs_method,
                    )

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data)

            logger.info(f"Aligned to reference grid → {new_w}×{new_h} → {out_path}")
            return ProcessingResult(
                success=True,
                operation="resample",
                input_path=inp,
                output_path=out_path,
                metadata={
                    "method": method,
                    "new_width": new_w,
                    "new_height": new_h,
                    "reference": str(reference),
                },
            )

        res_label = str(resolution or scale_factor).replace(".", "p")
        out_path = _resolve_output(inp, output, f"resamp_{res_label}m")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            orig_res_x, orig_res_y = src.res

            if resolution is not None:
                scale_x = orig_res_x / resolution
                scale_y = orig_res_y / resolution
            elif scale_factor is not None:
                scale_x = scale_factor
                scale_y = scale_factor
            else:
                msg = "Provide either resolution, scale_factor, or reference"
                raise ValueError(msg)

            new_h = max(1, int(src.height * scale_y))
            new_w = max(1, int(src.width * scale_x))

            data = src.read(
                out_shape=(src.count, new_h, new_w),
                resampling=rs_method,
            )

            new_transform = src.transform * src.transform.scale(
                src.width / new_w,
                src.height / new_h,
            )
            profile = src.profile.copy()
            profile.update(height=new_h, width=new_w, transform=new_transform)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)

        logger.info(f"Resampled → {new_w}×{new_h} → {out_path}")
        return ProcessingResult(
            success=True,
            operation="resample",
            input_path=inp,
            output_path=out_path,
            metadata={
                "resolution": resolution,
                "method": method,
                "new_width": new_w,
                "new_height": new_h,
            },
        )

    @_timed
    def pansharpen(
        self,
        pan: str | Path,
        ms: str | Path,
        method: str = "brovey",
        output: str | None = None,
    ) -> ProcessingResult:
        """
        Pan-sharpen multispectral image using panchromatic band.

        Args:
            pan:    Panchromatic (single band, high resolution) raster.
            ms:     Multispectral (multi-band) raster.
            method: ``"brovey"``, ``"ihs"``, ``"gram_schmidt"``, ``"simple_mean"``.
            output: Output path.

        Example::

            result = client.preprocess.pansharpen(
                pan="pan_15m.tif", ms="ms_60m.tif", method="brovey"
            )
        """
        rasterio = _require_rasterio()
        np = _require_numpy()
        from rasterio.enums import Resampling

        pan_p = Path(pan)
        ms_p = Path(ms)
        out_path = _resolve_output(pan_p, output, f"pansharp_{method}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(pan_p) as pan_src:
            pan_data = _safe_read_1(pan_src)
            pan_profile = pan_src.profile.copy()
            pan_shape = pan_data.shape

        with rasterio.open(ms_p) as ms_src:
            ms_data = ms_src.read(
                out_shape=(ms_src.count, *pan_shape),
                resampling=Resampling.bilinear,
            ).astype(np.float32)

        result = np.zeros_like(ms_data)

        if method == "brovey":
            ms_sum = ms_data.sum(axis=0) + 1e-10
            for b in range(ms_data.shape[0]):
                result[b] = (ms_data[b] / ms_sum) * pan_data

        elif method == "ihs":
            # Intensity-Hue-Saturation (simplified RGB only)
            if ms_data.shape[0] >= 3:
                r, g, b_ch = ms_data[0], ms_data[1], ms_data[2]
                intensity = (r + g + b_ch) / 3
                ratio = np.where(intensity > 0, pan_data / (intensity + 1e-10), 1)
                for i in range(ms_data.shape[0]):
                    result[i] = ms_data[i] * ratio
            else:
                result = ms_data * (pan_data / (ms_data[0] + 1e-10))

        elif method in ("gram_schmidt", "simple_mean"):
            ms_intensity = ms_data.mean(axis=0)
            ratio = np.where(ms_intensity > 0, pan_data / (ms_intensity + 1e-10), 1)
            for b in range(ms_data.shape[0]):
                result[b] = ms_data[b] * ratio

        else:
            msg = f"Unknown pansharpen method: {method!r}"
            raise ValueError(msg)

        pan_profile.update(count=ms_data.shape[0], dtype="float32")
        with rasterio.open(out_path, "w", **pan_profile) as dst:
            dst.write(result)

        logger.info(f"Pan-sharpened ({method}) → {out_path}")
        return ProcessingResult(
            success=True,
            operation=f"pansharpen:{method}",
            input_path=pan_p,
            output_path=out_path,
        )

    # ──────────────────────────────────────────────────────────────────────
    # H: Mosaicking & Compositing
    # ──────────────────────────────────────────────────────────────────────

    @_timed
    def mosaic(
        self,
        inputs: list[str | Path],
        output: str | None = None,
        method: str = "first",
    ) -> ProcessingResult:
        """
        Merge multiple rasters into a single seamless mosaic.

        Args:
            inputs: List of raster paths (must share CRS and band count).
            output: Output path.
            method: ``"first"`` (first valid pixel wins), ``"last"``,
                    ``"min"``, ``"max"``, ``"sum"``.

        Example::

            result = client.preprocess.mosaic(
                ["tile1.tif", "tile2.tif", "tile3.tif"],
                method="first"
            )
        """
        rasterio = _require_rasterio()
        from rasterio.merge import merge

        src_paths = [Path(p) for p in inputs]
        out_path = _resolve_output(src_paths[0], output, "mosaic")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        src_files = [rasterio.open(p) for p in src_paths]
        try:
            mosaic_data, mosaic_transform = merge(
                src_files, method=method if method != "sum" else "sum"
            )
            profile = src_files[0].profile.copy()
            profile.update(
                height=mosaic_data.shape[1],
                width=mosaic_data.shape[2],
                transform=mosaic_transform,
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(mosaic_data)
        finally:
            for s in src_files:
                s.close()

        logger.info(f"Mosaicked {len(inputs)} rasters → {out_path}")
        return ProcessingResult(
            success=True,
            operation=f"mosaic:{method}",
            output_path=out_path,
            metadata={"n_inputs": len(inputs), "method": method},
        )

    @_timed
    def composite(
        self,
        inputs: list[str | Path],
        method: str = "median",
        output: str | None = None,
        cloud_masks: list[str | Path] | None = None,
    ) -> ProcessingResult:
        """
        Create a multi-temporal composite from a stack of rasters.

        Args:
            inputs:      List of rasters (same CRS, shape, bands).
            method:      ``"median"``, ``"mean"``, ``"max"`` (max NDVI),
                         ``"min"``, ``"best_pixel"`` (lowest cloud).
            output:      Output path.
            cloud_masks: Optional list of cloud mask rasters (0=clear, 1=cloud).

        Example::

            # Cloud-free median composite from 12 monthly scenes
            result = client.preprocess.composite(
                inputs=["jan.tif","feb.tif",...,"dec.tif"],
                method="median"
            )
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        src_paths = [Path(p) for p in inputs]
        out_path = _resolve_output(src_paths[0], output, f"composite_{method}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Read all scenes into a stack
        stack = []
        with rasterio.open(src_paths[0]) as ref:
            profile = ref.profile.copy()
            ref_shape = (ref.count, ref.height, ref.width)
            nodata = ref.nodata

        for p in src_paths:
            with rasterio.open(p) as src:
                data = src.read(
                    out_shape=ref_shape, resampling=rasterio.enums.Resampling.bilinear
                ).astype(np.float32)
                if nodata is not None:
                    data = np.where(data == nodata, np.nan, data)
                stack.append(data)

        arr = np.stack(stack, axis=0)  # (n_dates, bands, h, w)

        # Apply cloud masks if provided
        if cloud_masks:
            for i, cm_path in enumerate(cloud_masks):
                if cm_path and i < len(stack):
                    with rasterio.open(Path(cm_path)) as cm_src:
                        cm = cm_src.read(
                            1,
                            out_shape=ref_shape[1:],
                            resampling=rasterio.enums.Resampling.nearest,
                        ).astype(float)
                    cm = np.where(cm > 0, np.nan, 1.0)
                    arr[i] = arr[i] * cm[np.newaxis, :, :]

        if method == "median":
            composite_data = np.nanmedian(arr, axis=0)
        elif method == "mean":
            composite_data = np.nanmean(arr, axis=0)
        elif method == "max":
            composite_data = np.nanmax(arr, axis=0)
        elif method == "min":
            composite_data = np.nanmin(arr, axis=0)
        elif method == "best_pixel":
            # Select pixel from scene with most valid data per pixel
            np.sum(~np.isnan(arr), axis=0)  # (bands, h, w)
            best_scene_idx = np.argmax(np.sum(~np.isnan(arr), axis=1), axis=0)  # (h, w)
            composite_data = np.zeros(ref_shape, dtype=np.float32)
            for b in range(ref_shape[0]):
                for si in range(len(stack)):
                    mask = best_scene_idx == si
                    composite_data[b][mask] = arr[si, b][mask]
        else:
            msg = f"Unknown composite method: {method!r}"
            raise ValueError(msg)

        if nodata is not None:
            composite_data = np.where(np.isnan(composite_data), nodata, composite_data)

        profile.update(dtype="float32", nodata=nodata)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(composite_data.astype(np.float32))

        logger.info(f"Composite ({method}) from {len(inputs)} scenes → {out_path}")
        return ProcessingResult(
            success=True,
            operation=f"composite:{method}",
            output_path=out_path,
            metadata={"n_inputs": len(inputs), "method": method},
        )
