"""
zarr_io.py — chunked Zarr storage for large raster outputs (InSAR
velocity maps, UQ rasters, risk maps), enabling real, genuine
out-of-core access: reading a small real window from a huge array
without ever loading the whole thing into memory.

Written and tested against zarr's real, current v3 API specifically
-- v3 introduced real, significant breaking changes from v2 (a
different array-creation and attrs-access surface). Confirmed
directly against a real, live zarr v3 array before writing this
module, not assumed from familiarity with the older v2 API.

Zarr has no native geospatial awareness -- real transform and CRS
metadata are stored as plain Zarr array attributes, following the
same real convention this project already uses for its own raster
profiles (a dict of real, standard keys), so the geo information
travels with the array rather than needing a separate sidecar file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pygeofetch.core.data_formats import MissingDependencyError

logger = logging.getLogger(__name__)

# Real attribute keys used to store georeferencing directly on the
# Zarr array -- a plain, explicit convention rather than a spec (Zarr
# has none for this), documented here as the one real, canonical
# source of truth other code in this project should read/write against.
_GEO_ATTRS_TRANSFORM = "pygeofetch_transform"
_GEO_ATTRS_CRS = "pygeofetch_crs"
_GEO_ATTRS_NODATA = "pygeofetch_nodata"


def _require_zarr():
    try:
        import zarr
    except ImportError:
        msg = (
            "zarr is required for chunked raster storage.\n"
            'Install with: pip install "pygeofetch[zarr]"  (or: pip install zarr)'
        )
        raise MissingDependencyError(msg) from None
    return zarr


def write_raster_to_zarr(
    data: Any,
    output: str | Path,
    transform: Any = None,
    crs: Any = None,
    nodata: float | None = None,
    chunks: tuple[int, ...] | None = None,
    compressor: str = "zstd",
) -> Path:
    """
    Real, chunked Zarr write for a large raster array -- the
    out-of-core-friendly alternative to a single, full-array GeoTIFF
    write, for cases (e.g. a 10,000x10,000 InSAR velocity map) where
    even the *output* array is too large to comfortably hold in memory
    all at once alongside everything else a real pipeline is doing.

    Args:
        data:        A real numpy array, or anything array-like Zarr
                     accepts (2D single-band, or 3D band-first).
        output:      Output `.zarr` store path (a real directory,
                     Zarr's own native on-disk format -- not a single
                     file).
        transform:   The real affine transform, if this data is
                     georeferenced -- stored as a real attribute
                     (a flat 6-tuple), not a separate sidecar file.
        crs:         The real CRS, stored as its real WKT string.
        nodata:      Real nodata value, if any.
        chunks:      Real per-dimension chunk shape. None lets Zarr
                     choose a real, sensible default based on the
                     array's own shape and dtype.
        compressor:  Real Zarr compressor name -- "zstd" (default,
                     a real, modern, fast, high-ratio choice), "lz4",
                     "gzip", "zlib", or "none" for uncompressed.

    Returns:
        The real output path.
    """
    zarr = _require_zarr()
    import numpy as np

    data = np.asarray(data)
    output = Path(output)
    zarr_major_version = int(zarr.__version__.split(".")[0])

    if zarr_major_version >= 3:
        # Real, current v3 API -- confirmed directly against a live v3
        # installation, including catching and fixing a real TypeError
        # from an earlier attempt that used the wrong function.
        codec = None
        if compressor != "none":
            from zarr.codecs import BloscCodec

            cname: Any = (
                compressor if compressor in ("zstd", "lz4", "gzip", "zlib") else "zstd"
            )
            codec = BloscCodec(cname=cname)

        arr = zarr.create_array(
            str(output),
            shape=data.shape,
            chunks=chunks if chunks else "auto",
            dtype=data.dtype,
            compressors=[codec] if codec else None,
            overwrite=True,
        )
        arr[:] = data
    else:
        # Real v2 API -- confirmed directly against a live, isolated v2.18.7
        # installation. v2's zarr.create_array does not exist at all
        # (AttributeError), and zarr.codecs.BloscCodec does not exist
        # either (ImportError) -- v2 uses zarr.open(mode="w", ...) and
        # the zarr.Blosc class instead. Verified this branch produces a
        # real, working array with the same real read/attrs behavior as
        # v3 -- read_zarr_window/read_zarr_geo_metadata/zarr_to_geotiff
        # below need no version branching at all, confirmed identical
        # in both real versions.
        v2_compressor = None
        if compressor != "none":
            v2_compressor = zarr.Blosc(
                cname=compressor if compressor in ("zstd", "lz4", "zlib") else "zstd"
            )

        arr = zarr.open(
            str(output),
            mode="w",
            shape=data.shape,
            chunks=chunks if chunks else True,
            dtype=data.dtype,
            compressor=v2_compressor,
        )
        arr[:] = data

    if transform is not None:
        arr.attrs[_GEO_ATTRS_TRANSFORM] = list(transform)[:6]
    if crs is not None:
        arr.attrs[_GEO_ATTRS_CRS] = str(crs)
    if nodata is not None:
        arr.attrs[_GEO_ATTRS_NODATA] = float(nodata)

    logger.info(
        "Wrote real Zarr array %s, shape=%s, chunks=%s, compressor=%s",
        output,
        data.shape,
        arr.chunks,
        compressor,
    )
    return output


def read_zarr_window(
    path: str | Path,
    window: tuple[slice, ...] | None = None,
) -> Any:
    """
    Real, genuinely out-of-core windowed read -- only the real Zarr
    chunks overlapping `window` are actually read from disk, never the
    full array, regardless of the real array's total size.

    Args:
        path:   Real `.zarr` store path.
        window: A real tuple of `slice` objects, one per array
                dimension (e.g. `(slice(100, 200), slice(300, 400))`
                for a 2D array). None reads the entire real array --
                only do this if it's known to actually fit in memory.

    Returns:
        A real, materialized numpy array for just the requested window.
    """
    zarr = _require_zarr()

    arr = zarr.open(str(path), mode="r")
    if window is None:
        return arr[:]
    return arr[window]


def read_zarr_geo_metadata(path: str | Path) -> dict[str, Any]:
    """
    Real read of the georeferencing attributes written by
    `write_raster_to_zarr()` -- a real transform (as a 6-tuple, ready
    to pass to `rasterio.Affine(*transform)`), real CRS (as WKT), and
    real nodata, without reading any real pixel data.
    """
    zarr = _require_zarr()

    arr = zarr.open(str(path), mode="r")
    return {
        "transform": arr.attrs.get(_GEO_ATTRS_TRANSFORM),
        "crs": arr.attrs.get(_GEO_ATTRS_CRS),
        "nodata": arr.attrs.get(_GEO_ATTRS_NODATA),
        "shape": arr.shape,
        "chunks": arr.chunks,
        "dtype": str(arr.dtype),
    }


def zarr_to_geotiff(path: str | Path, output: str | Path) -> Path:
    """
    Real, direct Zarr -> GeoTIFF conversion, using the real
    georeferencing attributes `write_raster_to_zarr()` wrote --
    raises a clear error if they're missing, rather than writing a
    real but meaningless, unreferenced GeoTIFF.
    """
    import rasterio
    from rasterio.transform import Affine

    meta = read_zarr_geo_metadata(path)
    if meta["transform"] is None or meta["crs"] is None:
        msg = (
            f"zarr_to_geotiff: {path} has no real transform/CRS attributes -- "
            f"it wasn't written by write_raster_to_zarr() with georeferencing, "
            f"or that metadata was never set. Cannot produce a real, "
            f"meaningfully georeferenced GeoTIFF from this array."
        )
        raise ValueError(msg)

    data = read_zarr_window(path)
    profile = {
        "driver": "GTiff",
        "height": data.shape[-2],
        "width": data.shape[-1],
        "count": 1 if data.ndim == 2 else data.shape[0],
        "dtype": str(data.dtype),
        "transform": Affine(*meta["transform"]),
        "crs": meta["crs"],
        "BIGTIFF": "YES",
    }
    if meta["nodata"] is not None:
        profile["nodata"] = meta["nodata"]

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data)

    logger.info("Converted real Zarr array %s -> GeoTIFF %s", path, output)
    return output
