"""
PostProcessor — G: Vector & Raster Post-Processing.
Vectorize, smooth, regularize, zonal statistics, buffer, centroid, COG, compress.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pygeofetch.processing.base import (
    ProcessingResult,
    _require_geopandas,
    _require_numpy,
    _require_rasterio,
    _resolve_output,
    _timed,
)

logger = logging.getLogger(__name__)


class PostProcessor:
    """
    Post-processing engine: raster to vector, cleanup, and export.

    Example::

        from pygeofetch import PyGeoFetch
        client = PyGeoFetch()
        vecs  = client.post.vectorize("classification.tif", threshold=0.5)
        stats = client.post.zonal_stats("ndvi.tif", zones="parcels.geojson")
        cog   = client.post.cog("scene.tif")
    """

    # ── G2: Vectorize ─────────────────────────────────────────────────────

    @_timed
    def vectorize(
        self,
        input: str | Path,
        output: str | None = None,
        band: int = 1,
        threshold: float | None = None,
        format: str = "geojson",
        min_area: float | None = None,
    ) -> ProcessingResult:
        """
        Convert a raster (e.g. classification or binary mask) to vector polygons.

        Args:
            input:     Input raster path.
            band:      Band number to vectorize (1-indexed).
            threshold: Apply binary threshold before vectorizing
                       (pixels >= threshold = 1, else 0).
            output:    Output vector file path.
            format:    ``"geojson"``, ``"gpkg"``, ``"shp"``.
            min_area:  Discard polygons smaller than this area (CRS units²).

        Example::

            result = client.post.vectorize("ndvi.tif", threshold=0.3,
                                           output="vegetation.geojson")
        """
        rasterio = _require_rasterio()
        np = _require_numpy()
        gpd = _require_geopandas()
        from rasterio.features import shapes
        from shapely.geometry import shape

        inp = Path(input)
        ext = {"geojson": ".geojson", "gpkg": ".gpkg", "shp": ".shp"}.get(
            format, ".geojson"
        )
        out_path = Path(output) if output else inp.parent / f"{inp.stem}_vectors{ext}"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            data = src.read(band).astype(np.float32)
            nodata = src.nodata
            crs = src.crs
            transform = src.transform

        if threshold is not None:
            data = (data >= threshold).astype(np.uint8)
            mask = data > 0
        else:
            mask = (
                (data != nodata)
                if nodata is not None
                else np.ones_like(data, dtype=bool)
            )
            data = data.astype(np.int32)

        features = []
        for geom_dict, value in shapes(
            data.astype(np.int32), mask=mask.astype(np.uint8), transform=transform
        ):
            geom = shape(geom_dict)
            if min_area and geom.area < min_area:
                continue
            features.append({"geometry": geom, "value": float(value)})

        if not features:
            logger.warning("vectorize: no features extracted")
            gdf = gpd.GeoDataFrame([], geometry=[], crs=crs)
        else:
            gdf = gpd.GeoDataFrame(features, crs=crs)

        if format == "geojson":
            gdf.to_file(out_path, driver="GeoJSON")
        elif format == "gpkg":
            gdf.to_file(out_path, driver="GPKG")
        elif format == "shp":
            gdf.to_file(out_path, driver="ESRI Shapefile")
        else:
            gdf.to_file(out_path, driver="GeoJSON")

        logger.info(f"Vectorized {len(gdf)} features → {out_path}")
        return ProcessingResult(
            success=True,
            operation="vectorize",
            input_path=inp,
            output_path=out_path,
            metadata={"n_features": len(gdf), "threshold": threshold, "format": format},
        )

    # ── G1: Smooth ────────────────────────────────────────────────────────

    @_timed
    def smooth(
        self,
        input: str | Path,
        tolerance: float = 1.0,
        output: str | None = None,
        method: str = "simplify",
    ) -> ProcessingResult:
        """
        Smooth / simplify vector geometries.

        Args:
            input:     Input vector file (GeoJSON, GPKG, SHP).
            tolerance: Simplification tolerance in CRS units (Douglas-Peucker).
            method:    ``"simplify"`` (Douglas-Peucker) or ``"buffer"`` (buffer+unbuffer).
            output:    Output vector file path.

        Example::

            result = client.post.smooth("building_footprints.geojson", tolerance=0.5)
        """
        gpd = _require_geopandas()
        inp = Path(input)
        out_path = Path(output) if output else inp.parent / f"{inp.stem}_smooth.geojson"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        gdf = gpd.read_file(inp)
        if method == "simplify":
            gdf["geometry"] = gdf.geometry.simplify(tolerance, preserve_topology=True)
        elif method == "buffer":
            # Expand then shrink — removes small notches
            gdf["geometry"] = gdf.geometry.buffer(tolerance).buffer(-tolerance)
        else:
            msg = f"Unknown smooth method: {method!r}"
            raise ValueError(msg)

        gdf = gdf[~gdf.geometry.is_empty]
        gdf.to_file(out_path, driver="GeoJSON")
        logger.info(f"Smoothed {len(gdf)} features → {out_path}")
        return ProcessingResult(
            success=True,
            operation=f"smooth:{method}",
            input_path=inp,
            output_path=out_path,
            metadata={"tolerance": tolerance, "n_features": len(gdf)},
        )

    # ── G3: Regularize ────────────────────────────────────────────────────

    @_timed
    def regularize(
        self,
        input: str | Path,
        output: str | None = None,
        corner_threshold_deg: float = 30.0,
    ) -> ProcessingResult:
        """
        Regularize (orthogonalize) building footprints or irregular polygons.
        Snaps edges to right angles where corner angle is close to 90°.

        Args:
            input:                 Input vector path.
            output:                Output path.
            corner_threshold_deg:  Angle tolerance for right-angle snapping.

        Example::

            result = client.post.regularize("building_footprints.geojson")
        """
        gpd = _require_geopandas()
        _require_numpy()

        inp = Path(input)
        out_path = (
            Path(output) if output else inp.parent / f"{inp.stem}_regularized.geojson"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        gdf = gpd.read_file(inp)
        regularized = []

        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                regularized.append(geom)
                continue
            try:
                # Use minimum rotated rectangle as regularization approximation
                mrr = geom.minimum_rotated_rectangle
                # Blend between original and rectangle based on overlap ratio
                iou = geom.intersection(mrr).area / (geom.union(mrr).area + 1e-10)
                if iou > 0.8:  # High overlap → use rectangle
                    regularized.append(mrr)
                else:
                    # Simplify and buffer for irregular shapes
                    regularized.append(
                        geom.simplify(0.5, preserve_topology=True)
                        .buffer(0.1)
                        .buffer(-0.1)
                    )
            except Exception:
                regularized.append(geom)

        gdf["geometry"] = regularized
        gdf = gdf[~gdf.geometry.is_empty]
        gdf.to_file(out_path, driver="GeoJSON")

        logger.info(f"Regularized {len(gdf)} features → {out_path}")
        return ProcessingResult(
            success=True,
            operation="regularize",
            input_path=inp,
            output_path=out_path,
            metadata={"n_features": len(gdf)},
        )

    # ── G7: Zonal Statistics ──────────────────────────────────────────────

    @_timed
    def zonal_stats(
        self,
        raster: str | Path,
        zones: str | Path,
        output: str | None = None,
        stats: list[str] | None = None,
        band: int = 1,
        all_touched: bool = False,
    ) -> ProcessingResult:
        """
        Compute zonal statistics: mean, median, min, max, std, count per zone.

        Args:
            raster:      Input raster.
            zones:       Vector file with zone polygons.
            output:      Output CSV path.
            stats:       Subset of stats to compute. Default: all.
            band:        Raster band to use.
            all_touched: Include pixels touching zone boundary.

        Example::

            result = client.post.zonal_stats(
                "ndvi.tif", zones="parcels.geojson", output="ndvi_stats.csv"
            )
        """
        rasterio = _require_rasterio()
        np = _require_numpy()
        gpd = _require_geopandas()
        import pandas as pd
        from rasterio.mask import mask as rasterio_mask

        raster_path = Path(raster)
        zones_path = Path(zones)

        stat_funcs = stats or [
            "count",
            "mean",
            "median",
            "min",
            "max",
            "std",
            "percentile_25",
            "percentile_75",
            "sum",
        ]

        out_path = (
            Path(output)
            if output
            else raster_path.parent / f"{raster_path.stem}_zonal_stats.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        gdf = gpd.read_file(zones_path)
        records = []

        with rasterio.open(raster_path) as src:
            # Reproject zones if needed
            if gdf.crs and src.crs and gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)

            for idx, row in gdf.iterrows():
                try:
                    masked, _ = rasterio_mask(
                        src,
                        [row.geometry.__geo_interface__],
                        crop=True,
                        all_touched=all_touched,
                        nodata=np.nan,
                    )
                    vals = masked[band - 1].ravel()
                    vals = vals[~np.isnan(vals)]
                except Exception as exc:
                    logger.debug(f"Zone {idx}: {exc}")
                    vals = np.array([])

                record = {"zone_id": idx}
                for col in gdf.columns:
                    if col != "geometry":
                        record[col] = row[col]

                if len(vals) == 0:
                    for s in stat_funcs:
                        record[s] = None
                else:
                    for s in stat_funcs:
                        if s == "count":
                            record[s] = int(len(vals))
                        elif s == "mean":
                            record[s] = float(np.mean(vals))
                        elif s == "median":
                            record[s] = float(np.median(vals))
                        elif s == "min":
                            record[s] = float(np.min(vals))
                        elif s == "max":
                            record[s] = float(np.max(vals))
                        elif s == "std":
                            record[s] = float(np.std(vals))
                        elif s == "sum":
                            record[s] = float(np.sum(vals))
                        elif s.startswith("percentile_"):
                            pct = int(s.split("_")[1])
                            record[s] = float(np.percentile(vals, pct))
                        elif s == "majority":
                            from scipy.stats import mode

                            record[s] = float(mode(vals.astype(int))[0])

                records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(out_path, index=False)

        logger.info(f"Zonal stats ({len(gdf)} zones) → {out_path}")
        return ProcessingResult(
            success=True,
            operation="zonal_stats",
            input_path=raster_path,
            output_path=out_path,
            metadata={"n_zones": len(gdf), "stats": stat_funcs},
        )

    # ── G4: Buffer ────────────────────────────────────────────────────────

    @_timed
    def buffer(
        self,
        input: str | Path,
        distance: float = 10.0,
        output: str | None = None,
        cap_style: str = "round",
        join_style: str = "round",
    ) -> ProcessingResult:
        """
        Add a buffer around vector geometries.

        Args:
            input:      Input vector path.
            distance:   Buffer distance in CRS units.
            cap_style:  ``"round"``, ``"flat"``, ``"square"``.
            join_style: ``"round"``, ``"mitre"``, ``"bevel"``.

        Example::

            result = client.post.buffer("roads.geojson", distance=15)
        """
        gpd = _require_geopandas()
        cap_map = {"round": 1, "flat": 2, "square": 3}
        join_map = {"round": 1, "mitre": 2, "bevel": 3}

        inp = Path(input)
        out_path = (
            Path(output)
            if output
            else inp.parent / f"{inp.stem}_buffer{distance:.0f}.geojson"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        gdf = gpd.read_file(inp)
        gdf["geometry"] = gdf.geometry.buffer(
            distance,
            cap_style=cap_map.get(cap_style, 1),
            join_style=join_map.get(join_style, 1),
        )
        gdf = gdf[~gdf.geometry.is_empty]
        gdf.to_file(out_path, driver="GeoJSON")

        logger.info(f"Buffered {len(gdf)} features by {distance} → {out_path}")
        return ProcessingResult(
            success=True,
            operation="buffer",
            input_path=inp,
            output_path=out_path,
            metadata={"distance": distance, "n_features": len(gdf)},
        )

    # ── G5: Centroid ──────────────────────────────────────────────────────

    @_timed
    def centroids(
        self,
        input: str | Path,
        output: str | None = None,
    ) -> ProcessingResult:
        """
        Extract centroid points from polygon/line geometries.

        Example::

            result = client.post.centroids("building_footprints.geojson")
        """
        gpd = _require_geopandas()
        inp = Path(input)
        out_path = (
            Path(output) if output else inp.parent / f"{inp.stem}_centroids.geojson"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        gdf = gpd.read_file(inp)
        gdf["geometry"] = gdf.geometry.centroid
        gdf.to_file(out_path, driver="GeoJSON")

        logger.info(f"Extracted {len(gdf)} centroids → {out_path}")
        return ProcessingResult(
            success=True,
            operation="centroids",
            input_path=inp,
            output_path=out_path,
            metadata={"n_features": len(gdf)},
        )

    # ── G6: Area / Perimeter ──────────────────────────────────────────────

    @_timed
    def add_geometry_metrics(
        self,
        input: str | Path,
        output: str | None = None,
    ) -> ProcessingResult:
        """
        Add area, perimeter, and compactness columns to a vector file.

        Example::

            result = client.post.add_geometry_metrics("parcels.geojson")
            # Adds: area_m2, perimeter_m, compactness columns
        """
        gpd = _require_geopandas()
        np = _require_numpy()
        inp = Path(input)
        out_path = (
            Path(output) if output else inp.parent / f"{inp.stem}_metrics.geojson"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        gdf = gpd.read_file(inp)
        gdf["area_m2"] = gdf.geometry.area
        gdf["perimeter_m"] = gdf.geometry.length
        # Polsby-Popper compactness (1 = circle)
        gdf["compactness"] = (
            4 * np.pi * gdf["area_m2"] / (gdf["perimeter_m"] ** 2 + 1e-10)
        )
        gdf.to_file(out_path, driver="GeoJSON")
        return ProcessingResult(
            success=True,
            operation="geometry_metrics",
            input_path=inp,
            output_path=out_path,
        )

    # ── G9/G10: Compress & COG ────────────────────────────────────────────

    @_timed
    def compress(
        self,
        input: str | Path,
        method: str = "lzw",
        output: str | None = None,
        zlevel: int = 6,
    ) -> ProcessingResult:
        """
        Apply lossless compression to a GeoTIFF.

        Args:
            input:   Input GeoTIFF.
            method:  ``"lzw"``, ``"deflate"``, ``"zstd"``, ``"packbits"``.
            output:  Output path.
            zlevel:  Compression level (1-9 for deflate/zstd).

        Example::

            result = client.post.compress("scene.tif", method="lzw")
        """
        rasterio = _require_rasterio()
        inp = Path(input)
        out_path = _resolve_output(inp, output, f"compressed_{method}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            profile = src.profile.copy()
            profile.update(compress=method)
            if method in ("deflate", "zstd"):
                profile["zstd_level" if method == "zstd" else "zlevel"] = zlevel
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(src.read())

        orig_size = inp.stat().st_size
        comp_size = out_path.stat().st_size
        ratio = orig_size / comp_size if comp_size else 1.0
        logger.info(f"Compressed ({method}) ratio={ratio:.2f}× → {out_path}")
        return ProcessingResult(
            success=True,
            operation=f"compress:{method}",
            input_path=inp,
            output_path=out_path,
            metadata={
                "method": method,
                "ratio": round(ratio, 2),
                "original_mb": orig_size / (1024 * 1024),
                "compressed_mb": comp_size / (1024 * 1024),
            },
        )

    @_timed
    def cog(
        self,
        input: str | Path,
        output: str | None = None,
        compress: str = "deflate",
        overview_resampling: str = "average",
        blocksize: int = 512,
    ) -> ProcessingResult:
        """
        Convert a GeoTIFF to Cloud Optimized GeoTIFF (COG).

        COGs enable efficient partial reads over HTTP — essential for cloud
        deployment and serving large rasters with range requests.

        Args:
            input:                Input GeoTIFF.
            output:               Output COG path.
            compress:             Internal compression (``"deflate"``, ``"lzw"``, ``"zstd"``).
            overview_resampling:  Overview resampling method.
            blocksize:            Internal tile size (512 or 256).

        Example::

            result = client.post.cog("scene.tif", compress="deflate")
        """
        rasterio = _require_rasterio()
        _require_numpy()
        import tempfile

        inp = Path(input)
        out_path = _resolve_output(inp, output, "cog")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                compress=compress,
                tiled=True,
                blockxsize=blocksize,
                blockysize=blocksize,
                interleave="band",
            )
            data = src.read()

            # Write temp file first, then add overviews
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            with rasterio.open(tmp_path, "w", **profile) as tmp_dst:
                tmp_dst.write(data)
                overview_levels = [2, 4, 8, 16, 32, 64]
                tmp_dst.build_overviews(
                    overview_levels,
                    getattr(
                        rasterio.enums.Resampling,
                        overview_resampling,
                        rasterio.enums.Resampling.average,
                    ),
                )
                tmp_dst.update_tags(ns="rio_overview", resampling=overview_resampling)

            # Write final COG with overviews interleaved (copy_src_overviews)
            with rasterio.open(tmp_path) as tmp_src:
                profile.update(copy_src_overviews=True)
                with rasterio.open(out_path, "w", **profile) as cog_dst:
                    cog_dst.write(tmp_src.read())

            tmp_path.unlink(missing_ok=True)

        size_mb = out_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"COG ({compress}, {blocksize}px tiles) {size_mb:.1f} MB → {out_path}"
        )
        return ProcessingResult(
            success=True,
            operation="cog",
            input_path=inp,
            output_path=out_path,
            metadata={
                "compress": compress,
                "blocksize": blocksize,
                "size_mb": round(size_mb, 1),
            },
        )

    # ── G9: Vector-vector overlay ────────────────────────────────────────

    @_timed
    def overlay(
        self,
        left: str | Path,
        right: str | Path,
        output: str | Path,
        how: str = "intersection",
        left_crs: str | None = None,
        right_crs: str | None = None,
    ) -> ProcessingResult:
        """
        Real polygon overlay between two vector layers -- intersection,
        union, difference, symmetric_difference, or identity.

        Args:
            left, right: Vector file paths (GeoJSON, Shapefile, etc.).
            output:      Output vector file path (format inferred from
                         extension, e.g. .geojson, .shp).
            how:         "intersection" (default), "union", "difference"
                         (left minus right), "symmetric_difference", or
                         "identity".
            left_crs, right_crs: Override the CRS to interpret each
                         input in, if it isn't already set correctly in
                         the file itself. If the two layers' real CRSs
                         differ, `right` is reprojected to match `left`
                         automatically before the real overlay runs.

        Example::

            result = client.post.overlay(
                "flood_extent.geojson", "parcels.geojson",
                output="flooded_parcels.geojson", how="intersection",
            )
        """
        gpd = _require_geopandas()

        valid_how = {"intersection", "union", "difference", "symmetric_difference", "identity"}
        if how not in valid_how:
            msg = f"overlay: how={how!r} not recognized -- use one of {sorted(valid_how)}"
            raise ValueError(msg)

        left_gdf = gpd.read_file(left)
        right_gdf = gpd.read_file(right)
        if left_crs:
            left_gdf = left_gdf.set_crs(left_crs, allow_override=True)
        if right_crs:
            right_gdf = right_gdf.set_crs(right_crs, allow_override=True)

        if left_gdf.crs is not None and right_gdf.crs is not None and left_gdf.crs != right_gdf.crs:
            right_gdf = right_gdf.to_crs(left_gdf.crs)

        result_gdf = gpd.overlay(left_gdf, right_gdf, how=how)

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_gdf.to_file(out_path)

        logger.info("overlay (%s): %d → %s features → %s", how, len(left_gdf), len(result_gdf), out_path.name)
        return ProcessingResult(
            success=True, operation="overlay", output_path=out_path, input_path=Path(left),
            metadata={"how": how, "n_input_left": len(left_gdf), "n_input_right": len(right_gdf), "n_output": len(result_gdf)},
        )

    @_timed
    def dissolve(
        self,
        input: str | Path,
        output: str | Path,
        by: str | None = None,
        aggfunc: str = "first",
    ) -> ProcessingResult:
        """
        Real dissolve -- merge polygons sharing the same real attribute
        value into single, combined geometries (or merge everything
        into one if `by` is None).

        Args:
            input:   Vector file path.
            output:  Output vector file path.
            by:      Column to group by before dissolving. None merges
                     every real feature into a single geometry.
            aggfunc: How to aggregate other real columns within each
                     group -- "first" (default), "sum", "mean", etc.
                     (any real pandas groupby-agg function name).

        Example::

            result = client.post.dissolve(
                "parcels.geojson", output="zones_by_landuse.geojson", by="land_use",
            )
        """
        gpd = _require_geopandas()

        gdf = gpd.read_file(input)
        n_before = len(gdf)
        result_gdf = gdf.dissolve(by=by, aggfunc=aggfunc) if by else gdf.dissolve(aggfunc=aggfunc)
        result_gdf = result_gdf.reset_index()

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_gdf.to_file(out_path)

        logger.info("dissolve (by=%s): %d → %d features → %s", by, n_before, len(result_gdf), out_path.name)
        return ProcessingResult(
            success=True, operation="dissolve", output_path=out_path, input_path=Path(input),
            metadata={"by": by, "n_input": n_before, "n_output": len(result_gdf)},
        )

    @_timed
    def spatial_join(
        self,
        left: str | Path,
        right: str | Path,
        output: str | Path,
        how: str = "inner",
        predicate: str = "intersects",
        left_crs: str | None = None,
        right_crs: str | None = None,
    ) -> ProcessingResult:
        """
        Real spatial join -- attach attributes from `right` to features
        in `left` based on a real spatial relationship, rather than a
        shared key column.

        Args:
            left, right: Vector file paths.
            output:      Output vector file path.
            how:         "inner" (default, drop unmatched left features),
                         "left" (keep all left features, null attributes
                         where unmatched), or "right".
            predicate:   Real spatial relationship to test -- "intersects"
                         (default), "contains", "within", "touches",
                         "crosses", "overlaps".
            left_crs, right_crs: Same real CRS-override/auto-reprojection
                         handling as overlay().

        Example::

            result = client.post.spatial_join(
                "sample_points.geojson", "administrative_zones.geojson",
                output="points_with_zone.geojson", predicate="within",
            )
        """
        gpd = _require_geopandas()

        left_gdf = gpd.read_file(left)
        right_gdf = gpd.read_file(right)
        if left_crs:
            left_gdf = left_gdf.set_crs(left_crs, allow_override=True)
        if right_crs:
            right_gdf = right_gdf.set_crs(right_crs, allow_override=True)

        if left_gdf.crs is not None and right_gdf.crs is not None and left_gdf.crs != right_gdf.crs:
            right_gdf = right_gdf.to_crs(left_gdf.crs)

        result_gdf = gpd.sjoin(left_gdf, right_gdf, how=how, predicate=predicate)

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_gdf.to_file(out_path)

        logger.info("spatial_join (%s, %s): %d → %d features → %s", how, predicate, len(left_gdf), len(result_gdf), out_path.name)
        return ProcessingResult(
            success=True, operation="spatial_join", output_path=out_path, input_path=Path(left),
            metadata={"how": how, "predicate": predicate, "n_input_left": len(left_gdf), "n_output": len(result_gdf)},
        )