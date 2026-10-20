"""
geoparquet_io.py — real, GeoParquet 1.0-compliant export for
pygeofetch's search results, vectorized features, and point-based
InSAR time series.

Deliberately uses geopandas' own real `to_parquet()`/`read_parquet()`
(backed by pyarrow) rather than hand-rolling Parquet's `geo` metadata
key -- modern geopandas (>=0.11) already writes spec-compliant
GeoParquet 1.0 (WKB geometry encoding, PROJJSON CRS, correct bbox
metadata) correctly. Reimplementing that by hand would be solving an
already-solved problem, and more likely to introduce a real spec
compliance bug than to improve on a mature, widely-used real writer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pygeofetch.core.data_formats import MissingDependencyError

logger = logging.getLogger(__name__)


def _require_pyarrow():
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        msg = (
            "pyarrow is required for GeoParquet export.\n"
            'Install with: pip install "pygeofetch[geoparquet]"  (or: pip install pyarrow)'
        )
        raise MissingDependencyError(msg) from None


def _require_geopandas_for_parquet():
    try:
        import geopandas as gpd
    except ImportError:
        msg = (
            "geopandas is required for GeoParquet export.\n"
            'Install with: pip install "pygeofetch[geo]"'
        )
        raise MissingDependencyError(msg) from None
    return gpd


def satellite_data_to_geoparquet(
    results: list[Any],
    output: str | Path,
    partition_by: str | None = None,
) -> Path:
    """
    Real GeoParquet export of federated search results.

    Args:
        results:      A list of real SatelliteData objects (from
                      `client.search(...)`).
        output:       Output `.parquet` path, or a directory if
                      `partition_by` is set.
        partition_by: A real SatelliteData attribute name (e.g.
                      `"provider"`, `"satellite"`) to write real,
                      separate Hive-style partitioned Parquet files
                      under `output/`, rather than one flat file --
                      useful for large, multi-provider result sets
                      queried later by a single partition without
                      reading the whole real dataset.

    Returns:
        The real output path (the file, or the partition directory).

    Raises:
        MissingDependencyError: if pyarrow or geopandas aren't
        installed -- a clear, actionable message naming the real,
        specific extra to install, not a bare ImportError.
        ValueError: if `results` is empty, or a scene has no real
        `bbox`/`geometry` to build a geometry column from.
    """
    _require_pyarrow()
    gpd = _require_geopandas_for_parquet()
    from shapely.geometry import box, shape

    if not results:
        msg = (
            "satellite_data_to_geoparquet: results is empty -- nothing real to export."
        )
        raise ValueError(msg)

    rows = []
    for r in results:
        if r.geometry is not None:
            geom = shape(r.geometry)
        elif r.bbox is not None:
            geom = box(*r.bbox)
        else:
            msg = (
                f"satellite_data_to_geoparquet: scene {r.id!r} has neither a real "
                f"geometry nor a real bbox -- cannot build a real geometry column "
                f"for it."
            )
            raise ValueError(msg)

        rows.append(
            {
                "id": r.id,
                "provider": r.provider,
                "collection": r.collection,
                "satellite": r.satellite,
                "sensor": r.sensor,
                "datetime": r.datetime,
                "cloud_cover": r.cloud_cover,
                "processing_level": (
                    str(r.processing_level) if r.processing_level else None
                ),
                "geometry": geom,
            }
        )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    output = Path(output)
    if partition_by:
        if partition_by not in gdf.columns:
            msg = f"satellite_data_to_geoparquet: partition_by={partition_by!r} is not a real exported column."
            raise ValueError(msg)
        output.mkdir(parents=True, exist_ok=True)
        for value, group in gdf.groupby(partition_by):
            part_dir = output / f"{partition_by}={value}"
            part_dir.mkdir(parents=True, exist_ok=True)
            group.to_parquet(part_dir / "part-0.parquet")
        logger.info(
            "Wrote real, partitioned GeoParquet (%d partitions) to %s",
            gdf[partition_by].nunique(),
            output,
        )
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output)
    logger.info("Wrote real GeoParquet (%d features) to %s", len(gdf), output)
    return output


def vector_to_geoparquet(input_vector: str | Path, output: str | Path) -> Path:
    """
    Real, general GeoParquet export for any vector file GeoPandas can
    read (GeoJSON, Shapefile, GeoPackage) -- e.g. output from
    `client.post.vectorize()`, `overlay()`, or `dissolve()`, when a
    GeoParquet copy is wanted for large-scale downstream analytics
    rather than GeoJSON's real, comparatively heavy text encoding.
    """
    _require_pyarrow()
    gpd = _require_geopandas_for_parquet()

    gdf = gpd.read_file(input_vector)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output)
    logger.info("Wrote real GeoParquet (%d features) to %s", len(gdf), output)
    return output


def points_to_geoparquet(
    points: list[tuple[float, float]],
    attributes: dict[str, list[Any]],
    output: str | Path,
    crs: str = "EPSG:4326",
) -> Path:
    """
    Real GeoParquet export for point-based data -- built specifically
    for large Persistent Scatterer point sets or per-pixel InSAR time
    series, where millions of real points make GeoJSON genuinely
    impractical (both to write and to later read back).

    Args:
        points:     Real (x, y) or (lon, lat) coordinate pairs, one
                    per point, in `crs`'s real units.
        attributes: Real column name -> list of per-point values,
                    e.g. {"velocity_mm_yr": [...], "coherence": [...]}.
                    Every list must have the same real length as
                    `points`.
        output:     Output `.parquet` path.
        crs:        Real CRS the coordinates are in.

    Raises:
        ValueError: if any attribute list's real length doesn't match
        `len(points)` -- a real, common, easy mistake to make when
        building this from separate numpy arrays.
    """
    _require_pyarrow()
    gpd = _require_geopandas_for_parquet()
    from shapely.geometry import Point

    for name, values in attributes.items():
        if len(values) != len(points):
            msg = (
                f"points_to_geoparquet: attribute {name!r} has {len(values)} real "
                f"values but there are {len(points)} real points -- lengths must match."
            )
            raise ValueError(msg)

    geometries = [Point(x, y) for x, y in points]
    gdf = gpd.GeoDataFrame({**attributes, "geometry": geometries}, crs=crs)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output)
    logger.info("Wrote real GeoParquet (%d points) to %s", len(gdf), output)
    return output


def read_geoparquet(path: str | Path) -> Any:
    """Real, thin read wrapper -- returns a real GeoDataFrame."""
    gpd = _require_geopandas_for_parquet()
    return gpd.read_parquet(path)
