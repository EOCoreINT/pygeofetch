"""
GPU/CPU array module abstraction for the InSAR pipeline.

Coherence estimation (windowed cross-correlation over the full scene)
and SBAS inversion (a single large matrix solve over every pixel at
once) are the two steps that genuinely benefit from GPU acceleration
at scale — both are already vectorized, large-array operations, not
per-pixel Python loops, which is exactly the shape of computation
CuPy accelerates well.

Usage::

    from pygeofetch.insar.gpu import get_array_module

    xp, ndi, using_gpu = get_array_module(prefer_gpu=True)
    # xp is either cupy or numpy; ndi is the matching ndimage module
    # (cupyx.scipy.ndimage or scipy.ndimage). Code written against xp/ndi
    # works unchanged on either backend, since CuPy's array API mirrors
    # numpy's directly.

Honesty note (important, not boilerplate): this module's CPU fallback
path and GPU-detection logic are both directly tested and confirmed
working. The actual GPU-accelerated numerical path (real CuPy array
operations executing on real CUDA hardware) has NOT been verified
against real GPU hardware — no GPU was available in the environment
this was built and tested in. The array operations are written to
mirror the existing, already-verified numpy/scipy logic line-for-line
specifically to minimize the risk of a behavioral difference, but
"correct by construction against a numpy-mirroring API" is not the
same claim as "confirmed correct by execution." Test on real GPU
hardware before relying on this for production results, and please
report back either way — that's a real gap in this module's
verification, not a formality.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

logger = logging.getLogger("pygeofetch.insar.gpu")

_gpu_checked = False
_gpu_available = False


def gpu_available() -> bool:
    """
    Check whether a usable CUDA GPU is actually present — not just
    whether the cupy package is importable. Confirmed necessary: CuPy
    imports successfully even with no GPU or an insufficient driver
    present; only cupy.cuda.is_available() reliably detects that case
    without raising.
    """
    global _gpu_checked, _gpu_available
    if _gpu_checked:
        return _gpu_available
    _gpu_checked = True
    try:
        import cupy as cp

        _gpu_available = bool(cp.cuda.is_available())
    except ImportError:
        _gpu_available = False
    except Exception as exc:  # pragma: no cover - defensive, real driver errors vary
        logger.debug(
            "GPU check failed with an unexpected error, assuming no GPU: %s", exc
        )
        _gpu_available = False
    return _gpu_available


def get_array_module(prefer_gpu: bool = True) -> Tuple[Any, Any, bool]:
    """
    Returns (xp, ndimage_module, using_gpu).

    Args:
        prefer_gpu: If True (default) and a usable GPU is detected, use
                   CuPy + cupyx.scipy.ndimage. Otherwise (or if no GPU
                   is available), use numpy + scipy.ndimage.

    Returns:
        xp:            cupy or numpy
        ndimage_module: cupyx.scipy.ndimage or scipy.ndimage
        using_gpu:     True if the GPU backend was actually selected
    """
    if prefer_gpu and gpu_available():
        import cupy as cp
        from cupyx.scipy import ndimage as gpu_ndimage

        logger.info("GPU acceleration active (CuPy)")
        return cp, gpu_ndimage, True

    import numpy as np
    from scipy import ndimage

    if prefer_gpu:
        logger.info("No usable GPU detected — using CPU (numpy/scipy)")
    return np, ndimage, False


def to_numpy(arr: Any) -> Any:
    """Move an array back to host memory if it's a CuPy array; a no-op
    for numpy arrays. Use this at the boundary between GPU-accelerated
    steps and the rest of the pipeline, which is not GPU-aware."""
    if hasattr(arr, "get"):  # cupy.ndarray has .get(), numpy.ndarray does not
        return arr.get()
    return arr
