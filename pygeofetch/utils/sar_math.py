"""
Shared, canonical SAR math utilities.

Real, confirmed finding: pygeofetch.processing.sar.SARProcessor.coherence()
and pygeofetch.insar.interferogram's internal _estimate_coherence /
_estimate_coherence_chunked independently implemented the identical,
real, standard interferometric coherence formula
(|<s1*conj(s2)>| / sqrt(<|s1|^2><|s2|^2>)) -- both verified correct
(same epsilon-safe denominator, same [0,1] clipping), but as two
separate copies to keep in sync.

This module is the new, single, canonical home for the general-purpose
version, now used by pygeofetch.processing.sar. The insar package's own
internal version deliberately remains separate rather than being
force-unified here: it's tightly coupled to a large (2500+ line),
already extensively real-world-validated class
(pygeofetch.insar.interferogram's interferogram generator, exercised
throughout this project's own Bu'ertai validation run), has a real,
separate chunked variant for memory-safe processing of full-scene
Sentinel-1 SLC rasters (which can be multiple gigabytes), and already
supports GPU acceleration via pygeofetch.insar.gpu. Refactoring that
file to depend on this one (or vice versa) was judged too risky for
the real, marginal benefit versus the chance of a regression in
already-proven code -- consolidating the safely-consolidatable
standalone duplicate here, and documenting why the other one stays
separate, is the honest tradeoff being made.
"""

from __future__ import annotations

from typing import Any


def estimate_interferometric_coherence(
    complex_image_1: Any,
    complex_image_2: Any,
    window: int = 7,
) -> Any:
    """
    Real, standard interferometric coherence between two complex SAR
    images (co-registered SLC data).

    Formula (standard InSAR coherence estimator):
        coherence = |<s1 * conj(s2)>| / sqrt(<|s1|^2> * <|s2|^2>)
    where <> denotes local spatial averaging over a real window
    (boxcar/uniform filter here) -- coherence measures how consistent
    the phase relationship between the two acquisitions is at each
    pixel; 1.0 means perfectly stable, 0.0 means fully decorrelated
    (random phase relationship, e.g. from vegetation growth, surface
    disturbance, or water).

    Parameters
    ----------
    complex_image_1, complex_image_2 : np.ndarray
        Real, co-registered complex SAR images (dtype complex64/128),
        same shape.
    window : int
        Real local-averaging window size (pixels). Larger windows give
        a smoother, more statistically stable coherence estimate at
        the cost of spatial resolution -- 7 is a common, real default
        for Sentinel-1 IW processing.

    Returns
    -------
    np.ndarray
        Real float32 coherence, same shape as input, values clipped to
        the real, physically valid [0, 1] range (protects against
        floating-point roundoff producing a value fractionally outside
        that range, not because the formula itself can exceed it).

    Raises
    ------
    ValueError
        If the two input images don't share the same real shape.
    """
    import numpy as np
    from scipy import ndimage

    s1 = np.asarray(complex_image_1)
    s2 = np.asarray(complex_image_2)
    if s1.shape != s2.shape:
        raise ValueError(
            f"estimate_interferometric_coherence: shape mismatch "
            f"{s1.shape} vs {s2.shape} -- images must be co-registered "
            f"onto the same real grid first."
        )

    inter = s1 * np.conj(s2)
    num = np.abs(
        ndimage.uniform_filter(inter.real, size=window)
        + 1j * ndimage.uniform_filter(inter.imag, size=window)
    )
    denom = np.sqrt(
        ndimage.uniform_filter(np.abs(s1) ** 2, size=window)
        * ndimage.uniform_filter(np.abs(s2) ** 2, size=window)
        + 1e-10  # real, standard epsilon -- prevents divide-by-zero over
        # true-zero-amplitude pixels (e.g. real no-data regions) without
        # meaningfully biasing genuine, non-zero coherence estimates.
    )
    return np.clip(num / denom, 0.0, 1.0).astype(np.float32)
