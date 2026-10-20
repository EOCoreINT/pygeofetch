"""
window_calculator.py — real, precise translation between geographic
bounds (bbox, CRS) and rasterio pixel Windows, for remote COG reads.

Kept as a small, standalone module rather than folded into
cog_streamer.py: window math is real, pure geometry (no I/O), so it's
independently unit-testable without a network connection, and reused
identically by both cog_streamer.py and, in principle, any future
remote-read code that needs the same real translation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PixelWindow:
    """
    A real, concrete pixel window into a raster -- deliberately a
    plain dataclass, not rasterio's own Window, so this module has no
    hard rasterio import at class-definition time (only inside the
    functions that need it) and stays trivially testable.
    """

    col_off: int
    row_off: int
    width: int
    height: int

    def to_rasterio_window(self) -> Any:
        """Real conversion to rasterio.windows.Window, imported lazily."""
        from rasterio.windows import Window

        return Window(self.col_off, self.row_off, self.width, self.height)


def bbox_to_pixel_window(
    bbox: tuple[float, float, float, float],
    transform: Any,
    raster_width: int,
    raster_height: int,
    bbox_crs: Any = None,
    raster_crs: Any = None,
) -> PixelWindow:
    """
    Real, precise translation from a real geographic (or projected)
    bbox to a pixel window in a specific raster's own grid.

    Args:
        bbox:          (min_x, min_y, max_x, max_y) in bbox_crs's real
                       units.
        transform:     The real raster's affine transform (e.g. from
                       `rasterio.open(url).transform`, without needing
                       to have read any pixel data yet).
        raster_width, raster_height: Real full raster dimensions, used
                       to clip the window to the actual real extent --
                       a bbox that extends beyond the raster's real
                       coverage does not produce an out-of-bounds
                       window or a negative size.
        bbox_crs:      The real CRS bbox's coordinates are in. If it
                       differs from raster_crs, the bbox corners are
                       reprojected first -- this is a real, necessary
                       step frequently skipped by hand-rolled window
                       math, and skipping it silently reads the wrong
                       real ground area whenever the two CRSs differ.
        raster_crs:    The real raster's own CRS. Required if bbox_crs
                       is given and differs.

    Returns:
        A real PixelWindow, already clipped to the raster's actual
        real extent (never returns a negative or out-of-bounds window).

    Raises:
        ValueError: if the real bbox does not overlap the raster's
                    real extent at all -- an honest failure rather
                    than a silently empty, meaningless window.
    """
    import rasterio

    min_x, min_y, max_x, max_y = bbox

    if (
        bbox_crs is not None
        and raster_crs is not None
        and str(bbox_crs) != str(raster_crs)
    ):
        from rasterio.warp import transform_bounds

        min_x, min_y, max_x, max_y = transform_bounds(
            bbox_crs, raster_crs, min_x, min_y, max_x, max_y
        )

    row_start, col_start = rasterio.transform.rowcol(transform, min_x, max_y)
    row_stop, col_stop = rasterio.transform.rowcol(transform, max_x, min_y)

    col_off = max(0, min(col_start, col_stop))
    row_off = max(0, min(row_start, row_stop))
    col_end = min(raster_width, max(col_start, col_stop))
    row_end = min(raster_height, max(row_start, row_stop))

    width = col_end - col_off
    height = row_end - row_off

    if width <= 0 or height <= 0:
        msg = (
            f"bbox_to_pixel_window: real bbox {bbox} does not overlap this "
            f"raster's real extent ({raster_width}x{raster_height} pixels) at all "
            f"-- computed window would be {width}x{height}."
        )
        raise ValueError(msg)

    return PixelWindow(
        col_off=int(col_off), row_off=int(row_off), width=int(width), height=int(height)
    )


def pixel_window_to_bounds(
    window: PixelWindow, transform: Any
) -> tuple[float, float, float, float]:
    """
    Real, inverse translation -- a pixel window back to its real
    geographic bounds in the raster's own CRS. Useful for confirming
    what ground area a computed window actually covers, e.g. before
    logging or writing output georeferencing.
    """
    from rasterio.transform import array_bounds

    return array_bounds(
        window.height,
        window.width,
        transform * transform.translation(window.col_off, window.row_off),
    )
