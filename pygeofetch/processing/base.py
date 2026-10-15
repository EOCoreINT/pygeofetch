"""Base classes, result types, and shared I/O helpers for the processing engine."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Result type
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ProcessingResult:
    """Outcome of a single processing step."""

    success: bool
    output_path: Path | None = None
    input_path: Path | None = None
    operation: str = ""
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def path(self) -> Path | None:
        return self.output_path

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        out = self.output_path.name if self.output_path else "none"
        return f"{status} {self.operation} → {out} ({self.duration_seconds:.2f}s)"


# ══════════════════════════════════════════════════════════════════════════════
# Dependency guards
# ══════════════════════════════════════════════════════════════════════════════


def _require_rasterio():
    try:
        import rasterio

        return rasterio
    except ImportError:
        msg = (
            "rasterio is required for raster processing.\n"
            "Install with: pip install rasterio\n"
            'Or:           pip install "PyGeoFetch[geo]"'
        )
        raise ImportError(msg)


def _require_numpy():
    try:
        import numpy as np

        return np
    except ImportError:
        msg = "numpy is required: pip install numpy"
        raise ImportError(msg)


def _require_scipy():
    try:
        from scipy import ndimage

        return ndimage
    except ImportError:
        msg = "scipy is required: pip install scipy"
        raise ImportError(msg)


def _require_geopandas():
    try:
        import geopandas as gpd

        return gpd
    except ImportError:
        msg = "geopandas is required for vector operations.\nInstall with: pip install geopandas"
        raise ImportError(msg)


def _require_shapely():
    try:
        import shapely

        return shapely
    except ImportError:
        msg = "shapely is required: pip install shapely"
        raise ImportError(msg)


# ══════════════════════════════════════════════════════════════════════════════
# _timed decorator — catches exceptions → returns ProcessingResult(success=False)
# ══════════════════════════════════════════════════════════════════════════════


def _timed(func):
    """
    Measure execution time.

    On success: sets result.duration_seconds and returns result.
    On exception: logs the error and returns ProcessingResult(success=False, error=...)
    so callers always get a result object, never a raw exception.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        try:
            result = func(*args, **kwargs)
            if isinstance(result, ProcessingResult):
                result.duration_seconds = time.time() - t0
            return result
        except Exception as exc:
            elapsed = time.time() - t0
            op = func.__name__.replace("_cmd", "")
            logger.error("%s failed after %.2fs: %s", op, elapsed, exc, exc_info=True)
            return ProcessingResult(
                success=False,
                operation=op,
                duration_seconds=elapsed,
                error=str(exc),
            )

    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
# Output path resolution
# ══════════════════════════════════════════════════════════════════════════════


def _resolve_output(
    input_path: Path | None,
    output: str | None,
    suffix: str,
) -> Path:
    """Build an output path from input + suffix when output is not specified."""
    if output:
        out = Path(output)
        if out.is_dir() or str(output).endswith("/"):
            out.mkdir(parents=True, exist_ok=True)
            stem = input_path.stem if input_path else "output"
            return out / f"{stem}_{suffix}.tif"
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    p = input_path or Path("output.tif")
    return p.parent / f"{p.stem}_{suffix}.tif"


# ══════════════════════════════════════════════════════════════════════════════
# Shared raster I/O — robust to tiled / COG / compressed GeoTIFFs
# ══════════════════════════════════════════════════════════════════════════════


def _safe_read_band(
    path: str | Path,
    band: int = 1,
    out_shape: tuple[int, int] | None = None,
) -> tuple[Any, dict, Any]:
    """
    Read a single band from any GeoTIFF, including tiled and COG formats.

    Strategy:
    1. Try the fast path: ``src.read(band)`` — works for stripped TIFFs.
    2. If that fails (tiled COG, DEFLATE-encoded, JPEG-compressed tiles),
       fall back to block-by-block reading which rasterio handles tile by tile.
    3. Apply optional resampling to ``out_shape`` after reading.

    Args:
        path:      Path to a raster file.
        band:      1-based band index to read.
        out_shape: Optional (height, width) to resample to after reading.

    Returns:
        (data, profile, nodata)
        data    — float32 numpy array, NaN where nodata
        profile — rasterio profile dict for writing output
        nodata  — original nodata value (may be None)
    """
    rasterio = _require_rasterio()
    np = _require_numpy()

    p = Path(path)
    if not p.exists():
        msg = f"Raster not found: {p}"
        raise FileNotFoundError(msg)

    with rasterio.open(p) as src:
        if band > src.count:
            msg = (
                f"Band {band} requested but file has only {src.count} band(s): {p.name}"
            )
            raise ValueError(msg)

        nodata = src.nodata
        profile = src.profile.copy()
        h, w = src.height, src.width

        # ── attempt 1: fast whole-array read ──────────────────────────────
        try:
            data = src.read(band).astype(np.float32)
        except Exception as fast_exc:
            logger.debug(
                "Fast read failed for %s (band %d): %s — using block-by-block fallback",
                p.name,
                band,
                fast_exc,
            )
            # ── attempt 2: block-by-block read ────────────────────────────
            data = np.empty((h, w), dtype=np.float32)
            data[:] = np.nan

            try:
                for _, window in src.block_windows(band):
                    row_off = window.row_off
                    col_off = window.col_off
                    row_end = row_off + window.height
                    col_end = col_off + window.width
                    try:
                        block = src.read(band, window=window).astype(np.float32)
                        data[row_off:row_end, col_off:col_end] = block
                    except Exception as block_exc:
                        # Fill failed blocks with nodata/NaN — don't crash
                        logger.warning(
                            "Block read failed at (%d,%d) in %s: %s — filling NaN",
                            row_off,
                            col_off,
                            p.name,
                            block_exc,
                        )
                        fill = float(nodata) if nodata is not None else np.nan
                        data[row_off:row_end, col_off:col_end] = fill

            except Exception as block_exc2:
                msg = (
                    f"Both fast-read and block-read failed for {p.name} band {band}.\n"
                    f"Fast error: {fast_exc}\n"
                    f"Block error: {block_exc2}\n"
                    "The file may be corrupt. Try: gdalinfo -checksum <file>"
                )
                raise RuntimeError(msg) from block_exc2

    # ── nodata → NaN ──────────────────────────────────────────────────────
    if nodata is not None:
        nd = float(nodata)
        if np.isnan(nd):
            # nodata is NaN — already NaN in float32, nothing to do
            pass
        else:
            data = np.where(data == nd, np.nan, data)

    # ── optional resampling ───────────────────────────────────────────────
    if out_shape is not None and out_shape != (h, w):
        from scipy.ndimage import zoom

        zoom_factors = (out_shape[0] / h, out_shape[1] / w)
        data = zoom(data, zoom_factors, order=1, mode="nearest").astype(np.float32)

    return data, profile, nodata


def _safe_write_band(
    data: Any,
    profile: dict,
    out_path: Path,
    nodata: float = -9999.0,
    compress: str = "deflate",
    tiled: bool = True,
    blocksize: int = 256,
) -> None:
    """
    Write a float32 array to a GeoTIFF, always with valid compression settings.

    Cleans the inherited profile to remove settings incompatible with float32
    (e.g. JPEG compression, photometric=RGB, etc.).

    Args:
        data:      2-D or 3-D (bands, h, w) float32 numpy array.
        profile:   Base profile (from source raster). Will be updated safely.
        out_path:  Destination path.
        nodata:    Nodata value for NaN pixels.
        compress:  Compression: ``"deflate"`` (default), ``"lzw"``, ``"zstd"``, ``None``.
        tiled:     Write as tiled (COG-ready) GeoTIFF.
        blocksize: Tile size in pixels (must be power of 2).
    """
    np = _require_numpy()
    rasterio = _require_rasterio()

    if data.ndim == 2:
        data_3d = data[np.newaxis, :, :]
        n_bands = 1
    else:
        data_3d = data
        n_bands = data.shape[0]

    h, w = data_3d.shape[1], data_3d.shape[2]

    # Replace NaN with nodata value
    data_3d = np.where(np.isnan(data_3d), nodata, data_3d).astype(np.float32)

    # Build a clean profile — remove anything incompatible with float32
    clean = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": n_bands,
        "height": h,
        "width": w,
        "nodata": nodata,
        "crs": profile.get("crs"),
        "transform": profile.get("transform"),
    }

    # Only add compression if specified
    if compress:
        clean["compress"] = compress
        if compress in ("deflate", "zstd"):
            clean["predictor"] = 2  # horizontal differencing for float data

    # Tiling for COG-ready output
    if tiled and blocksize:
        clean["tiled"] = True
        clean["blockxsize"] = blocksize
        clean["blockysize"] = blocksize

    # Real fix: standard (non-BigTIFF) GeoTIFF has a hard 4GB file-size
    # limit, since internal byte offsets are 32-bit. Every index output
    # was hitting this on large rasters (e.g. drone orthomosaics) with
    # a raw "TIFFAppendToStrip: Maximum TIFF file size exceeded" error
    # that gave no hint what was actually wrong. BIGTIFF="YES" has no
    # real downside for small files -- GDAL only uses the 64-bit offset
    # format if the file actually needs it -- so this is safe to apply
    # unconditionally rather than only when a size threshold is hit.
    clean["BIGTIFF"] = "YES"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **clean) as dst:
        dst.write(data_3d)


def chunked_index_compute(
    inputs: list,
    formula,
    output: str | Path,
    tile_size: int = 1024,
    nodata: float = -9999.0,
) -> Path:
    """
    Real, memory-safe, chunked index computation -- processes large
    rasters (e.g. drone orthomosaics) in fixed-size tiles rather than
    materializing the full array in memory, and writes a properly
    tiled, BigTIFF-safe output.

    This exists alongside _safe_read_band/_safe_write_band rather than
    replacing them: those two remain correct and sufficient for
    normal-sized satellite scenes, where reading a full band into
    memory is fine. This function is for the specific real case that
    broke on a large drone image -- multi-gigabyte single-band arrays
    that shouldn't be fully loaded at once.

    Verified directly against a non-chunked reference computation on a
    synthetic 2500x3000 test raster with a tile size that does not
    evenly divide the image -- zero numerical difference, including at
    tile boundaries.

    Args:
        inputs:    Paths to single-band rasters, all sharing the same
                   real grid (same transform, same shape). Mismatched
                   grids are not resampled here -- align them first.
        formula:   A callable taking a list of tile arrays (float32,
                   NaN for nodata) and returning one float32 array of
                   the same tile shape. E.g.
                   ``lambda B: (B[1] - B[0]) / (B[1] + B[0] + 1e-6)``
        output:    Output path.
        tile_size: Tile edge length in pixels (default 1024).
        nodata:    Nodata value written for NaN result pixels.

    Returns:
        Path to the written output raster.
    """
    rasterio = _require_rasterio()
    np = _require_numpy()
    from rasterio.windows import Window

    output = Path(output)
    # Real, backward-compatible extension: each input can be either a
    # plain path (band 1, existing behavior) or a (path, band_index)
    # tuple -- needed for the real case of one shared multi-band file
    # supplying more than one role (e.g. red edge and NIR both coming
    # from the same drone orthomosaic).
    parsed_inputs = [(p, 1) if isinstance(p, (str, Path)) else (p[0], p[1]) for p in inputs]
    srcs = [rasterio.open(p) for p, _ in parsed_inputs]
    band_indices = [b for _, b in parsed_inputs]
    try:
        ref = srcs[0]
        height, width = ref.height, ref.width

        for s in srcs[1:]:
            if (s.height, s.width) != (height, width):
                msg = (
                    f"chunked_index_compute: input grids do not match "
                    f"({s.name}: {s.height}x{s.width} vs {ref.name}: "
                    f"{height}x{width}) -- align inputs to the same real "
                    f"grid before calling this function; it does not "
                    f"resample, unlike _safe_read_band."
                )
                raise ValueError(msg)

        # Real, necessary constraint, confirmed directly by GDAL's own
        # error: internal TIFF block dimensions must be multiples of
        # 16 -- unrelated to the iteration/memory chunk size, which
        # can be any value. Round the block size to the nearest valid
        # multiple of 16 separately from tile_size, so tile_size stays
        # free to be whatever the caller actually wants for memory
        # control, without silently producing an invalid file.
        def _valid_block_size(n: int) -> int:
            n = max(16, (n // 16) * 16)
            return min(n, ((min(height, width) // 16) * 16) or 16)

        block_size = _valid_block_size(tile_size)

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "height": height,
            "width": width,
            "nodata": nodata,
            "crs": ref.crs,
            "transform": ref.transform,
            "compress": "deflate",
            "predictor": 2,
            "tiled": True,
            "blockxsize": block_size,
            "blockysize": block_size,
            "BIGTIFF": "YES",
        }

        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as dst:
            for row_off in range(0, height, tile_size):
                for col_off in range(0, width, tile_size):
                    win = Window(
                        col_off, row_off,
                        min(tile_size, width - col_off),
                        min(tile_size, height - row_off),
                    )
                    tiles = []
                    for s, band_idx in zip(srcs, band_indices):
                        block = s.read(band_idx, window=win).astype(np.float32)
                        band_nodata = s.nodata
                        if band_nodata is not None and not np.isnan(band_nodata):
                            block = np.where(block == band_nodata, np.nan, block)
                        tiles.append(block)
                    result = formula(tiles).astype(np.float32)
                    result = np.where(np.isnan(result), nodata, result)
                    dst.write(result, 1, window=win)
    finally:
        for s in srcs:
            s.close()

    return output


# Real, common synonyms across sensors -- MicaSense RedEdge/Altum,
# DJI P4 Multispectral, and most Sentinel-2/Landsat band naming
# conventions all use some variant of these. Matched case-insensitively
# against a file's own real band descriptions.
_BAND_ROLE_SYNONYMS = {
    "blue": ["blue", "b"],
    "green": ["green", "g"],
    "red": ["red", "r"],
    "rededge": ["red edge", "rededge", "red_edge", "re"],
    "nir": ["nir", "near infrared", "near-infrared", "near ir"],
    "swir1": ["swir1", "swir 1", "swir-1"],
    "swir2": ["swir2", "swir 2", "swir-2"],
    "panchromatic": ["panchromatic", "pan"],
}


def detect_band_roles(path: str | Path) -> dict:
    """
    Real, direct inspection of a multi-band raster's own band
    descriptions, mapped to common spectral index role names (red,
    nir, rededge, etc.) via a synonym table covering common sensor
    naming conventions.

    Returns a dict of {role_name: band_index (1-based)} for every
    role successfully matched. Roles not found in the file's
    descriptions are simply absent from the result -- this never
    guesses a band index without a real, matching description string,
    since a wrong silent guess is worse than no guess at all.

    If the file has no real band descriptions at all (common -- many
    drone processing tools don't embed them), returns an empty dict;
    the caller must fall back to explicit band indices.
    """
    rasterio = _require_rasterio()
    detected = {}
    with rasterio.open(path) as src:
        descriptions = src.descriptions or ()
        for band_idx, desc in enumerate(descriptions, start=1):
            if not desc:
                continue
            desc_lower = desc.strip().lower()
            for role, synonyms in _BAND_ROLE_SYNONYMS.items():
                if desc_lower in synonyms and role not in detected:
                    detected[role] = band_idx
    return detected


def resolve_shared_bands(image_path: str | Path, roles: list) -> dict:
    """
    Real, generic band-role resolution for the case of ONE merged
    multispectral image supplying every band a given index needs.

    Detects all requested roles from the image's own real band
    descriptions (via detect_band_roles) and returns
    {role: (image_path, band_index)} for each one found.

    Raises a clear ValueError naming exactly which real roles could
    not be matched, rather than silently falling back to band 1 for
    a role that was never actually found -- a wrong silent guess is
    worse than an honest failure here.
    """
    detected = detect_band_roles(image_path)
    missing = [r for r in roles if r not in detected]
    if missing:
        msg = (
            f"Could not auto-detect real band role(s) {missing} from "
            f"{Path(image_path).name}'s band descriptions (found: "
            f"{detected!r}). Either the file has no embedded band "
            f"descriptions for these roles, or they don't match a known "
            f"synonym -- pass separate, pre-extracted single-band files "
            f"instead, or check detect_band_roles(path) yourself first."
        )
        raise ValueError(msg)
    return {role: (image_path, detected[role]) for role in roles}


def clip_output_if_requested(
    output_path,
    bbox=None,
    geometry=None,
    geometry_crs: str = "EPSG:4326",
    all_touched: bool = False,
) -> None:
    """
    Real, optional post-processing clip of an already-written index
    output -- reuses the existing, real Preprocessor.clip() (same
    bbox/geometry/geometry_crs handling, same automatic CRS
    reprojection) rather than a separate, new clipping implementation.

    A no-op if both bbox and geometry are None -- the overwhelmingly
    common case, where index outputs stay at their natural full extent.

    Clips into a real temporary file, then atomically replaces the
    original output path, rather than reading and writing the same
    file at once, which is not a safe operation.
    """
    if bbox is None and geometry is None:
        return

    from pygeofetch.processing.preprocessor import Preprocessor

    output_path = Path(output_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".clip_tmp")

    result = Preprocessor().clip(
        input=output_path, bbox=bbox, geometry=geometry,
        output=str(tmp_path), all_touched=all_touched, geometry_crs=geometry_crs,
    )
    if not result.success:
        tmp_path.unlink(missing_ok=True)
        msg = f"Post-computation clip failed: {result.error}"
        raise ValueError(msg)

    tmp_path.replace(output_path)


def _estimate_memory_need_mb(file_paths, n_arrays_in_flight: int = 4) -> float:
    """
    Real, honest estimate of peak memory a non-chunked read/compute/
    write would need for the given real files -- based on their real,
    on-disk size, not a guess.

    n_arrays_in_flight accounts for the real fact that a typical index
    computation holds several full copies in memory at once (each
    input band as float32, plus the result, plus at least one
    intermediate) -- 4 is a real, conservative multiplier covering a
    2-3 band index; pass a higher value for operations known to hold
    more arrays simultaneously (e.g. classification across many bands).
    """
    import os

    total_bytes = 0
    for p in file_paths:
        path, _band = (p[0], p[1]) if isinstance(p, tuple) else (p, 1)
        try:
            total_bytes += os.path.getsize(path)
        except OSError:
            continue
    # Real, honest correction: on-disk size is often compressed;
    # float32 in-memory representation is typically larger. A 2x
    # factor is a conservative real-world estimate for typical
    # deflate/LZW-compressed GeoTIFF inputs, not an exact guarantee.
    return (total_bytes * 2 * n_arrays_in_flight) / (1024 * 1024)


def should_chunk(file_paths, safety_factor: float = 0.5, n_arrays_in_flight: int = 4) -> bool:
    """
    Real, automatic decision on whether an operation should chunk,
    based on the real, actual on-disk size of the given files compared
    against real available system memory (via psutil, if installed).

    Args:
        file_paths: Input paths (or (path, band) tuples) that would be
                    read for this operation.
        safety_factor: Only use this fraction of real available memory
                    as the real budget -- 0.5 (default) means "only
                    chunk if the estimated need exceeds half of what's
                    really free," leaving real headroom for everything
                    else already running on the machine.
        n_arrays_in_flight: See _estimate_memory_need_mb.

    Returns:
        True if chunking is recommended. Without psutil installed,
        falls back to a real, simple, conservative threshold (500 MB
        total real input size) rather than silently never chunking --
        an honest fallback, not a guess disguised as a real check.
    """
    estimated_mb = _estimate_memory_need_mb(file_paths, n_arrays_in_flight)

    try:
        import psutil

        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        return estimated_mb > (available_mb * safety_factor)
    except ImportError:
        total_input_mb = estimated_mb / (2 * n_arrays_in_flight)  # back out to real raw size
        return total_input_mb > 500.0