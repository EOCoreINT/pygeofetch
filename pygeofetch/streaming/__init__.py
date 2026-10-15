"""AOI-windowed streaming of Cloud-Optimized rasters (see ``windowed`` docs)."""

from pygeofetch.streaming.scene import (
    read_scene_bands,
    select_streamable_assets,
    stream_many,
    stream_scene,
    streamed_band_paths,
)
from pygeofetch.streaming.windowed import (
    CLOUD_GDAL_OPTIONS,
    AOIOutsideRasterError,
    StreamingUnsupportedError,
    WindowedRaster,
    compute_window,
    compute_window_and_coverage,
    read_window,
    stream_window_to_geotiff,
    to_vsi_path,
)

__all__ = [
    "CLOUD_GDAL_OPTIONS",
    "AOIOutsideRasterError",
    "StreamingUnsupportedError",
    "WindowedRaster",
    "compute_window",
    "compute_window_and_coverage",
    "read_scene_bands",
    "read_window",
    "select_streamable_assets",
    "stream_many",
    "stream_scene",
    "stream_window_to_geotiff",
    "streamed_band_paths",
    "to_vsi_path",
]
