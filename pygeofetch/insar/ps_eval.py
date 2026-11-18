"""
Using extra persistent-scatterer pixels in unwrapping and SBAS, and checking whether they helped.

Both functions take plain arrays and know nothing about how the pixels were selected (ADI, a model, ...).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def boost_coherence_with_ps(
    coherence: np.ndarray,
    ps_mask: np.ndarray,
    ps_transform: Any,
    ps_crs: Any,
    dst_transform: Any,
    dst_crs: Any,
    boost: float = 0.6,
    top_share: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Raise the coherence weight of the unwrap-grid cells that hold the most selected PS pixels.

    ``ps_mask`` lives on the PS stack grid (``ps_transform``/``ps_crs``); ``coherence`` lives on the
    (coarser, multilooked) unwrap grid (``dst_transform``/``dst_crs``). The mask is averaged onto the
    unwrap grid, so each cell gets "the share of its area that was selected". Among the cells that overlap
    the PS region, the ``top_share`` with the highest share (and at least one selected pixel) get their
    coherence lifted to at least ``boost``. Using a fixed *count* of cells keeps different selectors
    comparable: they all lift the same number of cells. Only the *weight* changes (what SNAPHU and the SBAS
    threshold see); the interferometric phase is never touched.

    Returns ``(new_coherence, boosted_cells)``.
    """
    from rasterio.warp import Resampling, reproject

    coherence = np.asarray(coherence, np.float32)
    frac = np.zeros(coherence.shape, np.float32)
    cover = np.zeros(coherence.shape, np.float32)
    kw = dict(src_transform=ps_transform, src_crs=ps_crs, dst_transform=dst_transform, dst_crs=dst_crs,
              resampling=Resampling.average)
    reproject(np.asarray(ps_mask, np.float32), frac, **kw)
    reproject(np.ones(np.shape(ps_mask), np.float32), cover, **kw)
    region = (cover > 0.5) & np.isfinite(coherence)
    boosted = np.zeros(coherence.shape, bool)
    k = int(round(top_share * region.sum()))
    if k > 0:
        idx = np.flatnonzero(region.ravel())
        order = idx[np.argsort(-frac.ravel()[idx], kind="stable")][:k]
        order = order[frac.ravel()[order] > 0]
        boosted.ravel()[order] = True
    out = coherence.copy()
    out[boosted] = np.maximum(out[boosted], np.float32(boost))
    return out, boosted


def _loo_neighbour_mean(v: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Mean of the finite values of ``v`` in a window, excluding the centre pixel; also the neighbour count."""
    from scipy import ndimage

    ok = np.isfinite(v)
    vz = np.where(ok, v, 0.0)
    n = ndimage.uniform_filter(ok.astype(float), window, mode="constant") * window**2
    s = ndimage.uniform_filter(vz, window, mode="constant") * window**2
    n_o, s_o = n - ok, s - vz
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_o >= 3, s_o / n_o, np.nan), n_o


def compare_velocity_maps(
    base: np.ndarray,
    new: np.ndarray,
    residual_base: Optional[np.ndarray] = None,
    residual_new: Optional[np.ndarray] = None,
    window: int = 7,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Did the extra pixels add useful velocities, or noise? No ground truth, so three honest checks.

    1. Coverage: pixels gained / lost against the baseline map.
    2. Unchanged where it should be: on pixels both maps have, the velocities should agree.
    3. Plausibility of the gained pixels: a real subsidence field is spatially smooth, so a gained pixel
       should sit close to the mean of the baseline velocities around it. Their deviation is compared with
       the same deviation measured for the baseline pixels themselves (leave-one-out), and with the
       inversion residual if given.

    Returns ``(summary, gained_mask)``.
    """
    base, new = np.asarray(base, float), np.asarray(new, float)
    fb, fn = np.isfinite(base), np.isfinite(new)
    gained, lost, both = fn & ~fb, fb & ~fn, fb & fn
    out: Dict[str, Any] = {
        "baseline_pixels": int(fb.sum()), "new_pixels": int(fn.sum()), "gained": int(gained.sum()),
        "lost": int(lost.sum()), "coverage_gain_pct": round(100 * (fn.sum() / max(fb.sum(), 1) - 1), 1),
    }
    if both.any():
        d = new[both] - base[both]
        out["common_median_diff"] = round(float(np.median(d)), 3)
        out["common_p95_abs_diff"] = round(float(np.percentile(np.abs(d), 95)), 3)
    ref, _ = _loo_neighbour_mean(base, window)
    dev = np.abs(new - ref)
    dev_base = np.abs(base - ref)
    g = gained & np.isfinite(dev)
    b = fb & np.isfinite(dev_base)
    out["gained_with_neighbours"] = int(g.sum())
    if g.sum() >= 5 and b.sum() >= 5:
        sig = 1.4826 * float(np.median(dev_base[b]))
        out["baseline_median_dev_from_neighbours"] = round(float(np.median(dev_base[b])), 3)
        out["gained_median_dev_from_neighbours"] = round(float(np.median(dev[g])), 3)
        out["gained_vs_baseline_dev_ratio"] = round(float(np.median(dev[g]) / max(np.median(dev_base[b]), 1e-9)), 2)
        out["gained_share_beyond_3sigma_pct"] = round(100 * float(np.mean(dev[g] > 3 * max(sig, 1e-9))), 1)
        out["baseline_share_beyond_3sigma_pct"] = round(100 * float(np.mean(dev_base[b] > 3 * max(sig, 1e-9))), 1)
    if residual_base is not None and residual_new is not None:
        rb, rn = np.asarray(residual_base, float), np.asarray(residual_new, float)
        if (gained & np.isfinite(rn)).any() and (fb & np.isfinite(rb)).any():
            out["baseline_median_residual"] = round(float(np.median(rb[fb & np.isfinite(rb)])), 3)
            out["gained_median_residual"] = round(float(np.median(rn[gained & np.isfinite(rn)])), 3)
    return out, gained


__all__ = ["boost_coherence_with_ps", "compare_velocity_maps"]
