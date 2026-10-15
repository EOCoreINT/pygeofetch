"""
chunked_processor.py — memory-efficient, overlap-aware tiled raster
processing, writing directly to a COG or Zarr store.

Generalizes the real, already-proven tiling pattern from
`pygeofetch.processing.base.chunked_index_compute` (verified there
against a non-chunked reference, zero difference including at tile
boundaries) with one genuinely new real capability: overlapping tiles.

Why overlap matters, concretely: a simple non-overlapping tile split
is correct for pixel-independent formulas (NDVI, band math -- each
output pixel depends only on the same input pixel), which is exactly
what chunked_index_compute already handles correctly. It is NOT
correct for neighborhood operations (a moving-window mean, GLCM
texture, a convolution) -- a pixel near a tile's edge needs real
neighbor pixels that live in the *adjacent* tile, and without them,
the result at every tile boundary is wrong. This was identified
directly as an unsupported case earlier in this project's own
spectral-indices work (`texture()` was explicitly left out of the
chunking retrofit for exactly this reason). This module is the real
fix: each tile is read with a real pixel margin on all sides, the
neighborhood operation runs correctly using real neighbor data, and
only the margin is trimmed before writing -- so adjacent tiles' real
results agree with no seam.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ChunkedProcessor:
    """
    Real, overlap-aware, memory-efficient tiled raster processor.

    Args:
        tile_size: Real tile edge length in pixels (default 1024).
        overlap:   Real pixel margin read on every side of each tile,
                   beyond `tile_size`, then trimmed from the output
                   before writing. 0 (default) matches
                   `chunked_index_compute`'s existing, pixel-independent
                   behavior exactly. Set > 0 for any real neighborhood
                   operation whose window radius requires it (e.g.
                   overlap >= (glcm_window_size // 2) for GLCM texture).
        output_format: `"cog"` (default, a real, properly tiled,
                   BigTIFF-safe GeoTIFF) or `"zarr"` (a real, chunked
                   Zarr store, for genuinely out-of-core downstream
                   access -- see zarr_io.py).

    Example, a real 3x3 neighborhood mean filter (needs overlap=1)::

        def mean_filter_3x3(tile):
            import numpy as np
            from scipy.ndimage import uniform_filter
            return uniform_filter(tile[0], size=3, mode="nearest")

        ChunkedProcessor(tile_size=512, overlap=1).process(
            inputs=["large_raster.tif"], formula=mean_filter_3x3,
            output="smoothed.tif",
        )
    """

    tile_size: int = 1024
    overlap: int = 0
    output_format: str = "cog"

    def __post_init__(self) -> None:
        if self.overlap < 0:
            msg = f"ChunkedProcessor: overlap must be >= 0, got {self.overlap}"
            raise ValueError(msg)
        if self.output_format not in ("cog", "zarr"):
            msg = f"ChunkedProcessor: output_format must be 'cog' or 'zarr', got {self.output_format!r}"
            raise ValueError(msg)

    def process(
        self,
        inputs: list[Any],
        formula: Callable[[list[Any]], Any],
        output: str | Path,
        nodata: float = -9999.0,
    ) -> Path:
        """
        Real, tiled processing with overlap-aware neighborhood
        correctness -- only ever holds one tile-plus-margin's worth of
        data (per input) in memory, regardless of the real total
        raster size.

        Args:
            inputs:  Paths to single-band rasters (or (path, band)
                     tuples), all sharing the same real grid -- same
                     contract as `chunked_index_compute`.
            formula: A callable taking a list of tile arrays (each
                     with the real overlap margin included) and
                     returning ONE array of the SAME shape (margin
                     included) -- the margin is trimmed automatically
                     after the call, the formula itself does not need
                     to handle trimming.
            output:  Output path -- a `.tif` for `output_format="cog"`,
                     or a real Zarr store directory for `"zarr"`.
            nodata:  Real nodata value for the output.

        Returns:
            The real output path.
        """
        from pygeofetch.processing.base import _require_numpy, _require_rasterio

        rasterio = _require_rasterio()
        np = _require_numpy()
        from rasterio.windows import Window

        output = Path(output)
        parsed_inputs = [(p, 1) if isinstance(p, (str, Path)) else (p[0], p[1]) for p in inputs]
        srcs = [rasterio.open(p) for p, _ in parsed_inputs]
        band_indices = [b for _, b in parsed_inputs]

        try:
            ref = srcs[0]
            height, width = ref.height, ref.width

            if self.output_format == "cog":
                return self._process_to_cog(srcs, band_indices, height, width, formula, output, nodata, np, rasterio, Window)
            return self._process_to_zarr(srcs, band_indices, height, width, formula, output, nodata, np, Window, ref)
        finally:
            for s in srcs:
                s.close()

    def _iter_tiles(self, height: int, width: int):
        """
        Real tile iteration -- yields (read_window_params, write_offset,
        write_shape) for every real tile, where read_window_params
        already include the real overlap margin (clipped to the
        raster's actual bounds at the edges), and write_offset/shape
        describe exactly where in the real output this tile's
        (trimmed) result belongs.
        """
        for row_off in range(0, height, self.tile_size):
            for col_off in range(0, width, self.tile_size):
                out_h = min(self.tile_size, height - row_off)
                out_w = min(self.tile_size, width - col_off)

                read_row_off = max(0, row_off - self.overlap)
                read_col_off = max(0, col_off - self.overlap)
                read_row_end = min(height, row_off + out_h + self.overlap)
                read_col_end = min(width, col_off + out_w + self.overlap)

                # Real, exact trim amounts: how much of the read tile's
                # margin, on EACH side, needs to be cut off after the
                # formula runs, to get back down to just this tile's
                # own real output region.
                trim_top = row_off - read_row_off
                trim_left = col_off - read_col_off

                yield {
                    "read": (read_row_off, read_col_off, read_row_end - read_row_off, read_col_end - read_col_off),
                    "write_offset": (row_off, col_off),
                    "write_shape": (out_h, out_w),
                    "trim": (trim_top, trim_left),
                }

    def _process_to_cog(self, srcs, band_indices, height, width, formula, output, nodata, np, rasterio, Window):
        ref = srcs[0]
        profile = {
            "driver": "GTiff", "dtype": "float32", "count": 1,
            "height": height, "width": width, "nodata": nodata,
            "crs": ref.crs, "transform": ref.transform,
            "compress": "deflate", "predictor": 2, "tiled": True,
            "blockxsize": self._valid_block_size(width), "blockysize": self._valid_block_size(height),
            "BIGTIFF": "YES",
        }
        output.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output, "w", **profile) as dst:
            for tile in self._iter_tiles(height, width):
                result = self._compute_tile(tile, srcs, band_indices, formula, Window, np)
                out_h, out_w = tile["write_shape"]
                row_off, col_off = tile["write_offset"]
                dst.write(result[:out_h, :out_w], 1, window=Window(col_off, row_off, out_w, out_h))

        logger.info("ChunkedProcessor: wrote real COG %s (%dx%d, tile_size=%d, overlap=%d)",
                    output, height, width, self.tile_size, self.overlap)
        return output

    def _process_to_zarr(self, srcs, band_indices, height, width, formula, output, nodata, np, Window, ref):
        from pygeofetch.core.data_formats.zarr_io import _require_zarr

        zarr = _require_zarr()
        zarr_major_version = int(zarr.__version__.split(".")[0])

        if zarr_major_version >= 3:
            arr = zarr.create_array(
                str(output), shape=(height, width), chunks=(self.tile_size, self.tile_size),
                dtype="float32", overwrite=True,
            )
        else:
            # Real v2 API -- confirmed directly against a live, isolated
            # v2.18.7 install; v2 has no zarr.create_array at all.
            arr = zarr.open(
                str(output), mode="w", shape=(height, width),
                chunks=(self.tile_size, self.tile_size), dtype="float32",
            )
        arr.attrs["pygeofetch_transform"] = list(ref.transform)[:6]
        arr.attrs["pygeofetch_crs"] = str(ref.crs)
        arr.attrs["pygeofetch_nodata"] = float(nodata)

        for tile in self._iter_tiles(height, width):
            result = self._compute_tile(tile, srcs, band_indices, formula, Window, np)
            out_h, out_w = tile["write_shape"]
            row_off, col_off = tile["write_offset"]
            arr[row_off:row_off + out_h, col_off:col_off + out_w] = result[:out_h, :out_w]

        logger.info("ChunkedProcessor: wrote real Zarr %s (%dx%d, tile_size=%d, overlap=%d)",
                    output, height, width, self.tile_size, self.overlap)
        return output

    def _compute_tile(self, tile, srcs, band_indices, formula, Window, np):
        r_row, r_col, r_h, r_w = tile["read"]
        win = Window(r_col, r_row, r_w, r_h)
        tiles_data = [s.read(b, window=win).astype(np.float32) for s, b in zip(srcs, band_indices)]
        result = np.asarray(formula(tiles_data)).astype(np.float32)

        trim_top, trim_left = tile["trim"]
        out_h, out_w = tile["write_shape"]
        return result[trim_top:trim_top + out_h, trim_left:trim_left + out_w]

    @staticmethod
    def _valid_block_size(n: int) -> int:
        # Same real constraint confirmed earlier in this project: GDAL
        # requires TIFF block dimensions to be multiples of 16.
        return max(16, min(1024, (n // 16) * 16) or 16)
