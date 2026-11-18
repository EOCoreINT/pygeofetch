"""Memory check before a large SLC pair is read.

A Sentinel-1 sub-swath SLC is about 1 GB as complex64; a pair needs several full-size arrays at once. The
check reads the two image headers, estimates the peak and compares it with the memory the operating system
reports as available, so a run that cannot fit stops with a message instead of being killed (taking the
editor and the notebook kernel with it).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Peak working set of process_pair as a multiple of one full-size complex64 array. Measured with a synthetic
# pair at 1 x 1 looks (10.9 x with a small image, where fixed-size buffers weigh more); the values below are
# deliberately a little under that, so they do not refuse runs that would fit. Not measured on a real scene.
PEAK_FACTOR_CHUNKED = 8.0
PEAK_FACTOR_UNCHUNKED = 14.0


def available_gb() -> Optional[float]:
    """Memory the OS says can be allocated without swapping, in GB (None if it cannot be read)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except Exception:  # noqa: BLE001
        return None


def image_bytes(path: Union[str, Path]) -> Optional[int]:
    """Size of the image as complex64, from its header (None if it cannot be opened)."""
    try:
        import rasterio

        with rasterio.open(path) as src:
            return int(src.height) * int(src.width) * 8
    except Exception:  # noqa: BLE001
        return None


def estimate_peak_gb(reference: Union[str, Path], secondary: Union[str, Path], chunked: bool) -> Optional[float]:
    sizes = [s for s in (image_bytes(reference), image_bytes(secondary)) if s]
    if not sizes:
        return None
    return max(sizes) * (PEAK_FACTOR_CHUNKED if chunked else PEAK_FACTOR_UNCHUNKED) / 1e9


def check_memory(reference, secondary, chunked: bool, mode: str = "warn") -> Tuple[Optional[float], Optional[float]]:
    """Compare the estimated peak with the available memory. mode: "off" | "warn" | "raise".

    Returns (estimated_peak_gb, available_gb). Raises MemoryError in "raise" mode when the estimate exceeds what is
    available, with the numbers and what to do about it.
    """
    if mode == "off":
        return None, None
    need, have = estimate_peak_gb(reference, secondary, chunked), available_gb()
    if need is None or have is None or need <= have:
        return need, have
    how = "" if chunked else " Chunked processing (chunk_size / chunk_rows) would cut it."
    msg = (f"this pair needs about {need:.1f} GB at its peak and the system reports {have:.1f} GB available "
           f"(estimate: {PEAK_FACTOR_CHUNKED if chunked else PEAK_FACTOR_UNCHUNKED:g} x the larger SLC; not measured on "
           f"real scenes).{how} Close other programs, restart the kernel, or crop the SLCs to the AOI before processing.")
    if mode == "raise":
        raise MemoryError(msg)
    logger.warning("memory: %s", msg)
    return need, have
