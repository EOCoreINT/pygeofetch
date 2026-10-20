"""
AOI-windowed raster reads: pull only the pixels inside an area of interest
from a (possibly remote) Cloud-Optimized GeoTIFF instead of downloading and
loading the whole scene.

Why this exists
---------------
`client.download()` fetches every selected asset as a complete file. For a
Sentinel-2 or Landsat band that is typically 100-300 MB per band per scene,
even when the area of interest is a few kilometres wide, and
`rasterio.open(...).read()` then decodes the whole raster into RAM. A COG is
internally tiled and carries overviews, and its host serves HTTP range
requests, so GDAL can fetch just the tiles that intersect a window. This
module does that: it converts the AOI into a pixel window, reads only that
window, and can write it straight to a small AOI-sized GeoTIFF block by
block so peak memory is one block rather than one scene.

What it does NOT do
-------------------
* It cannot help with providers that only serve archives (Copernicus CDSE
  OData `.SAFE.zip`, USGS M2M `.tar` bundles). A zip/tar is not randomly
  accessible raster data; those providers still need the full download.
* The window is the AOI's *bounding box* in the raster's CRS, rounded
  outwards to whole pixels. Polygon masking is a separate, later step.
* `decimation > 1` reads from COG overviews when present (cheap) or
  resamples on read (not cheaper in bytes). It changes pixel size.
* The byte counts reported here are decoded window sizes, not network
  traffic. Actual transfer depends on tile layout and compression.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

logger = logging.getLogger("pygeofetch.streaming.windowed")

# GDAL options that make /vsicurl/ reads of COGs efficient and quiet.
# GDAL_DISABLE_READDIR_ON_OPEN stops GDAL listing the parent "directory" on
# every open, which on object stores means extra requests that return 403/404.
# Note: CPL_VSIL_CURL_ALLOWED_EXTENSIONS is deliberately NOT set; with
# SAS-signed URLs the extension is followed by a query string and an
# allow-list can wrongly reject the signed URL.
CLOUD_GDAL_OPTIONS: dict[str, str] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": str(32 * 1024 * 1024),
}

_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".gz", ".safe")


class AOIOutsideRasterError(ValueError):
    """The AOI does not overlap the raster's extent."""


class StreamingUnsupportedError(ValueError):
    """The source is an archive or otherwise not randomly accessible raster."""


@dataclass
class WindowedRaster:
    """An AOI window read from a source raster.

    Attributes:
        data: Array shaped ``(bands, height, width)``.
        transform: Affine transform of ``data`` (pixel -> CRS coordinates).
        crs: Source raster CRS (rasterio ``CRS``).
        nodata: Source nodata value, if any.
        window: The window in *source* pixel coordinates that was read.
        source_shape: ``(height, width)`` of the full source raster.
        decimation: Integer factor the window was decimated by (1 = native).
        source: The path/URL that was read (without any signing query).
        aoi_coverage: Share of the requested AOI that lies inside the raster
            (1.0 = fully covered; lower = the AOI runs past the raster edge).
    """

    data: np.ndarray
    transform: Any
    crs: Any
    nodata: float | None
    window: Any
    source_shape: tuple[int, int]
    decimation: int
    source: str
    aoi_coverage: float = 1.0

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.data.shape)  # type: ignore[return-value]

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        from rasterio.transform import array_bounds

        return array_bounds(self.data.shape[1], self.data.shape[2], self.transform)

    @property
    def area_fraction(self) -> float:
        """Share of the source raster's pixels covered by the window read."""
        total = self.source_shape[0] * self.source_shape[1]
        return float(self.window.height * self.window.width) / float(total)


def to_vsi_path(href: str) -> str:
    """Map an href to a GDAL virtual path (http(s)/s3 -> /vsicurl/ or /vsis3/)."""
    h = str(href)
    low = h.lower()
    if low.startswith(("http://", "https://")):
        return f"/vsicurl/{h}"
    if low.startswith("s3://"):
        return f"/vsis3/{h[5:]}"
    if low.startswith(("/vsi", "file://")):
        return h[7:] if low.startswith("file://") else h
    return h


def _strip_query(href: str) -> str:
    return str(href).split("?", 1)[0]


def assert_streamable(href: str) -> None:
    """Raise StreamingUnsupportedError for archive hrefs (zip/tar/SAFE)."""
    base = _strip_query(href).lower().rstrip("/")
    if base.endswith(_ARCHIVE_SUFFIXES):
        raise StreamingUnsupportedError(
            f"{_strip_query(href)!r} is an archive, which cannot be windowed. "
            "Providers that only serve .zip/.tar bundles (Copernicus OData, "
            "USGS M2M) need a full download; use a COG-serving STAC provider "
            "(planetary_computer, element84) for AOI-only streaming."
        )


def _bbox_bounds(bbox: Any) -> tuple[float, float, float, float]:
    """Accept (minx, miny, maxx, maxy), a shapely geometry, or a GeoDataFrame."""
    if hasattr(bbox, "total_bounds"):
        b = bbox.total_bounds
    elif hasattr(bbox, "bounds") and not isinstance(bbox, (tuple, list)):
        b = bbox.bounds
    else:
        b = bbox
    b = tuple(float(v) for v in b)
    if len(b) != 4 or not (b[0] < b[2] and b[1] < b[3]):
        raise ValueError(
            f"bbox must be (minx, miny, maxx, maxy) with min < max, got {b!r}"
        )
    return b  # type: ignore[return-value]


def compute_window_and_coverage(
    src: Any, bbox: Any, bbox_crs: str = "EPSG:4326"
) -> tuple[Any, float]:
    """Pixel window of ``src`` covering ``bbox`` plus the AOI coverage.

    The window is rounded outwards to whole pixels and clipped to the raster.
    ``coverage`` is clipped_area / requested_area in [0, 1]: below 1 means the
    AOI runs past the raster's edge (e.g. it straddles a Sentinel-2 tile
    boundary) and the returned data is only part of the AOI. Raises
    AOIOutsideRasterError if there is no overlap at all.
    """
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    if src.crs is None:
        raise ValueError(f"{src.name!r} has no CRS; cannot place an AOI on it.")
    t = src.transform
    if t.b != 0 or t.d != 0:
        raise NotImplementedError(
            "Rotated/sheared rasters are not supported for windowed reads."
        )

    minx, miny, maxx, maxy = _bbox_bounds(bbox)
    if str(bbox_crs).upper() != str(src.crs).upper():
        minx, miny, maxx, maxy = transform_bounds(
            bbox_crs, src.crs, minx, miny, maxx, maxy, densify_pts=21
        )

    win = from_bounds(minx, miny, maxx, maxy, transform=t)
    # Round outwards so the AOI is fully covered.
    c0 = math.floor(win.col_off + 1e-9)
    r0 = math.floor(win.row_off + 1e-9)
    c1 = math.ceil(win.col_off + win.width - 1e-9)
    r1 = math.ceil(win.row_off + win.height - 1e-9)
    want_area = max(1, (c1 - c0)) * max(1, (r1 - r0))
    cc0, rr0 = max(c0, 0), max(r0, 0)
    cc1, rr1 = min(c1, src.width), min(r1, src.height)
    if cc1 <= cc0 or rr1 <= rr0:
        raise AOIOutsideRasterError(
            f"AOI {(minx, miny, maxx, maxy)} does not overlap raster bounds {tuple(src.bounds)}"
        )
    coverage = ((cc1 - cc0) * (rr1 - rr0)) / float(want_area)
    return Window(cc0, rr0, cc1 - cc0, rr1 - rr0), min(1.0, coverage)


def compute_window(src: Any, bbox: Any, bbox_crs: str = "EPSG:4326") -> Any:
    """Pixel window of ``src`` covering ``bbox`` (see compute_window_and_coverage)."""
    return compute_window_and_coverage(src, bbox, bbox_crs)[0]


def _resolve_decimation(
    width: int, height: int, decimation: int, max_pixels: int | None, bands: int
) -> int:
    d = max(1, int(decimation))
    if max_pixels is not None and (width // d) * (height // d) * bands > max_pixels:
        need = math.ceil(math.sqrt((width * height * bands) / float(max_pixels)))
        logger.warning(
            "AOI window %dx%d (x%d bands) exceeds max_pixels=%d; decimating by %d "
            "(coarser pixels). Raise max_pixels or use stream_window_to_geotiff() "
            "for a full-resolution, low-memory copy.",
            width,
            height,
            bands,
            max_pixels,
            need,
        )
        d = max(d, need)
    return d


def read_window(
    href: str,
    bbox: Any,
    *,
    bbox_crs: str = "EPSG:4326",
    indexes: int | Sequence[int] | None = 1,
    decimation: int = 1,
    max_pixels: int | None = None,
    resampling: str = "nearest",
    gdal_options: dict[str, str] | None = None,
) -> WindowedRaster:
    """Read only the AOI window of a raster into memory.

    Args:
        href: Local path, http(s) URL (read via /vsicurl/) or s3:// URL.
        bbox: (minx, miny, maxx, maxy) in ``bbox_crs``, or a shapely
            geometry / GeoDataFrame.
        indexes: Band index or list of 1-based band indexes (None = all).
        decimation: Integer downsample factor (reads COG overviews when
            present).
        max_pixels: Memory guard. If the window would exceed this many
            values, decimation is raised automatically (with a warning)
            rather than allocating more.
        resampling: rasterio resampling name used when decimating.

    Returns:
        WindowedRaster with ``(bands, H, W)`` data.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine

    assert_streamable(href)
    opts = dict(CLOUD_GDAL_OPTIONS)
    opts.update(gdal_options or {})
    with rasterio.Env(**opts):
        with rasterio.open(to_vsi_path(href)) as src:
            win, cov = compute_window_and_coverage(src, bbox, bbox_crs)
            if indexes is None:
                idx = list(range(1, src.count + 1))
            elif isinstance(indexes, int):
                idx = [indexes]
            else:
                idx = list(indexes)
            d = _resolve_decimation(
                int(win.width), int(win.height), decimation, max_pixels, len(idx)
            )
            out_h = max(1, math.ceil(win.height / d))
            out_w = max(1, math.ceil(win.width / d))
            data = src.read(
                idx,
                window=win,
                out_shape=(len(idx), out_h, out_w),
                resampling=getattr(Resampling, resampling),
            )
            wt = src.window_transform(win)
            transform = Affine(
                wt.a * (win.width / out_w),
                wt.b,
                wt.c,
                wt.d,
                wt.e * (win.height / out_h),
                wt.f,
            )
            return WindowedRaster(
                data=data,
                transform=transform,
                crs=src.crs,
                nodata=src.nodata,
                window=win,
                source_shape=(src.height, src.width),
                decimation=d,
                source=_strip_query(href),
                aoi_coverage=cov,
            )


def iter_blocks(
    width: int, height: int, block: int
) -> Iterator[tuple[int, int, int, int]]:
    """Yield (col_off, row_off, w, h) tiles covering a width x height area."""
    for r in range(0, height, block):
        for c in range(0, width, block):
            yield c, r, min(block, width - c), min(block, height - r)


def stream_window_to_geotiff(
    href: str,
    bbox: Any,
    out_path: str | Path,
    *,
    bbox_crs: str = "EPSG:4326",
    indexes: int | Sequence[int] | None = 1,
    block_size: int = 1024,
    gdal_options: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Copy the AOI window of a (remote) raster into a small local GeoTIFF,
    one block at a time, at native resolution.

    Peak memory is one ``block_size`` x ``block_size`` block per band, no
    matter how large the AOI or the source scene is.

    Returns a dict with ``path``, ``window``, ``shape``, ``transform``,
    ``crs``, ``nodata``, ``dtype``, ``bytes_written`` (size of the
    output file on disk) and ``aoi_coverage`` (see WindowedRaster).
    """
    import rasterio

    assert_streamable(href)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp name per call: the same scene can appear twice in one batch
    # (overlapping search results), and two writers must not share a file.
    tmp_path = out_path.with_name(f"{out_path.name}.{uuid.uuid4().hex[:8]}.part")
    opts = dict(CLOUD_GDAL_OPTIONS)
    opts.update(gdal_options or {})

    with rasterio.Env(**opts):
        with rasterio.open(to_vsi_path(href)) as src:
            win, cov = compute_window_and_coverage(src, bbox, bbox_crs)
            if indexes is None:
                idx = list(range(1, src.count + 1))
            elif isinstance(indexes, int):
                idx = [indexes]
            else:
                idx = list(indexes)
            w, h = int(win.width), int(win.height)
            profile = {
                "driver": "GTiff",
                "height": h,
                "width": w,
                "count": len(idx),
                "dtype": src.dtypes[idx[0] - 1],
                "crs": src.crs,
                "transform": src.window_transform(win),
                "nodata": src.nodata,
                "compress": "deflate",
                "BIGTIFF": "IF_SAFER",
            }
            if w >= 256 and h >= 256:
                profile.update(tiled=True, blockxsize=256, blockysize=256)
            transform, crs, nodata, dtype = (
                profile["transform"],
                src.crs,
                src.nodata,
                profile["dtype"],
            )
            try:
                with rasterio.open(tmp_path, "w", **profile) as dst:
                    from rasterio.windows import Window

                    for c, r, bw, bh in iter_blocks(w, h, int(block_size)):
                        sub = Window(win.col_off + c, win.row_off + r, bw, bh)
                        block = src.read(idx, window=sub)
                        dst.write(block, window=Window(c, r, bw, bh))
                tmp_path.replace(out_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
    return {
        "path": out_path,
        "window": win,
        "shape": (len(idx), h, w),
        "transform": transform,
        "crs": crs,
        "nodata": nodata,
        "dtype": dtype,
        "bytes_written": out_path.stat().st_size,
        "aoi_coverage": cov,
    }
