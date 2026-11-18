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
_gpu_reason = ""
_gpu_name = ""


def _probe() -> None:
    """Run the GPU check once and record WHY it failed, not just that it did."""
    global _gpu_checked, _gpu_available, _gpu_reason, _gpu_name
    if _gpu_checked:
        return
    _gpu_checked = True
    try:
        import cupy as cp
    except ImportError as exc:
        _gpu_reason = (
            "the cupy package is not installed in this Python environment "
            f"({exc}). Install the build that matches your CUDA version, e.g. "
            "'pip install cupy-cuda12x' (CUDA 12) or 'pip install cupy-cuda13x' (CUDA 13); "
            "'nvidia-smi' shows your CUDA version. torch having CUDA does not give cupy a GPU."
        )
        return
    except Exception as exc:  # cupy present but broken (wrong CUDA libraries, ...)
        _gpu_reason = f"importing cupy failed: {type(exc).__name__}: {exc}"
        return
    try:
        if not bool(cp.cuda.is_available()):
            _gpu_reason = (
                "cupy is installed but reports no usable CUDA device "
                "(driver missing or too old for this cupy build, or the process cannot see the GPU)."
            )
            return
        n = int(cp.cuda.runtime.getDeviceCount())
        if n < 1:
            _gpu_reason = "cupy found 0 CUDA devices"
            return
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props.get("name", b"")
        _gpu_name = name.decode() if isinstance(name, bytes) else str(name)
        # Prove it by computing something: is_available() can be True while kernels still fail
        # (for example a cupy build for a different CUDA major version).
        float(cp.asarray([1.0, 2.0]).sum())
        _gpu_available = True
    except Exception as exc:
        _gpu_reason = f"cupy could not run on the GPU: {type(exc).__name__}: {exc}"
        _gpu_available = False


def gpu_available() -> bool:
    """True only when a CUDA device exists AND a trivial cupy computation runs on it
    (not just whether the cupy package imports, which it does with no GPU present)."""
    _probe()
    return _gpu_available


def gpu_info() -> dict:
    """Why the GPU is or is not used: {"available", "device", "reason", "cupy", "torch_cuda"}.
    Call this in a notebook cell when use_gpu=True does not do what you expect."""
    _probe()
    info = {"available": _gpu_available, "device": _gpu_name or None,
            "reason": _gpu_reason or None, "cupy": None, "torch_cuda": None}
    try:
        import cupy as cp
        info["cupy"] = cp.__version__
    except Exception:
        pass
    try:
        import torch
        info["torch_cuda"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    return info


def get_array_module(prefer_gpu: bool = True) -> Tuple[Any, Any, bool]:
    """
    Returns (xp, ndimage_module, using_gpu).

    prefer_gpu=True and a working GPU -> cupy + cupyx.scipy.ndimage; otherwise numpy + scipy.
    When the GPU was asked for but cannot be used, the reason is logged once at WARNING.
    """
    if prefer_gpu and gpu_available():
        import cupy as cp
        from cupyx.scipy import ndimage as gpu_ndimage

        return cp, gpu_ndimage, True

    import numpy as np
    from scipy import ndimage

    if prefer_gpu:
        _warn_once()
    return np, ndimage, False


_warned = False


def _warn_once() -> None:
    global _warned
    if _warned:
        return
    _warned = True
    logger.warning("use_gpu=True but the GPU is not being used: %s", _gpu_reason or "unknown reason")


def to_numpy(arr: Any) -> Any:
    """Move an array back to host memory if it's a CuPy array; a no-op
    for numpy arrays. Use this at the boundary between GPU-accelerated
    steps and the rest of the pipeline, which is not GPU-aware."""
    if hasattr(arr, "get"):  # cupy.ndarray has .get(), numpy.ndarray does not
        return arr.get()
    return arr