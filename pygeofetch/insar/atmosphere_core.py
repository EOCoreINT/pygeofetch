"""Numerics for tropospheric (and other smooth-in-space) phase screens: estimation, validation, network consistency.

Why this exists
---------------
``atmosphere.AtmosphericCorrector`` could fit a phase-versus-height slope and remove it, or subtract an ERA5/GACOS delay, and
return the result with no check that the result was better. Reviewing it found:

* a spatial filter applied to *wrapped* phase by averaging angles (wrong near +/-pi) with NaNs replaced by zero (biases edges);
* a global fit that treats every pixel as independent, so a deforming area on a slope is absorbed into the "atmosphere";
* no ramp term, so a long-wavelength ramp (orbit error, ionosphere) is aliased into the height slope wherever height and
  position correlate (most valleys);
* a significance test that ignores spatial correlation (neighbouring pixels share the same atmosphere);
* no test that the correction reduced the phase variance. Corrections from reanalysis can add noise in low relief;
* per-interferogram fits that ignore that the atmosphere belongs to a *date*, and that a DEM error looks like a height slope that
  scales with the perpendicular baseline.

This module provides the estimation and validation pieces; ``atmosphere.AtmosphericCorrector`` uses them.

References: Hanssen (2001), Radar Interferometry (structure function of the atmospheric phase); Doin et al. (2009) and
Bekaert et al. (2015) (phase-height correlation, spatially varying fits); Yu et al. (2018), Murray et al. (2019) (judging a
correction by variance reduction and independent data); Fattahi & Amelung (2013) (DEM error appears as a baseline-proportional term).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("pygeofetch.insar.atmosphere_core")


def wrap(phase):
    return np.angle(np.exp(1j * np.asarray(phase)))


# ---------------------------------------------------------------------------------------------------
# smoothing and structure function
# ---------------------------------------------------------------------------------------------------
def smooth_complex(phase: np.ndarray, valid: np.ndarray, sigma_px: float) -> np.ndarray:
    """Low-pass a *wrapped* phase through its complex exponential, ignoring invalid pixels (normalised convolution).

    Averaging the angles is wrong near +/-pi, and filling invalid pixels with zero biases every pixel near a gap.
    Returns the smoothed phase (NaN where the local weight is negligible).
    """
    from scipy.ndimage import gaussian_filter

    v = np.asarray(valid, dtype=bool) & np.isfinite(phase)
    z = np.where(v, np.exp(1j * np.nan_to_num(phase)), 0.0)
    w = v.astype(np.float64)
    num = gaussian_filter(z.real, sigma_px) + 1j * gaussian_filter(z.imag, sigma_px)
    den = gaussian_filter(w, sigma_px)
    out = np.angle(num)
    out[den < 1e-3] = np.nan
    return out


def structure_function(
    phase: np.ndarray,
    valid: np.ndarray,
    lags_px: Sequence[int],
    wrapped: bool = True,
    n_pairs: int = 200_000,
    seed: int = 0,
) -> Dict[int, float]:
    """RMS phase difference (radians) between pixels a given distance apart: sqrt of the structure function.

    The tropospheric phase screen is spatially correlated, so this grows with lag over the first tens of kilometres; a
    correction that removes atmosphere lowers it at those lags. Differences are wrapped when the phase is wrapped, which
    makes the measure insensitive to ramps and to the arbitrary 2*pi offsets. Each lag uses random pixel pairs in a random
    direction (not only along rows and columns).
    """
    rng = np.random.default_rng(seed)
    h, w = phase.shape
    v = np.asarray(valid, dtype=bool) & np.isfinite(phase)
    out: Dict[int, float] = {}
    for lag in lags_px:
        i = rng.integers(0, h, n_pairs)
        j = rng.integers(0, w, n_pairs)
        ang = rng.uniform(0, 2 * np.pi, n_pairs)
        i2 = np.rint(i + lag * np.sin(ang)).astype(int)
        j2 = np.rint(j + lag * np.cos(ang)).astype(int)
        ok = (i2 >= 0) & (i2 < h) & (j2 >= 0) & (j2 < w)
        i, j, i2, j2 = i[ok], j[ok], i2[ok], j2[ok]
        ok = v[i, j] & v[i2, j2]
        d = phase[i2[ok], j2[ok]] - phase[i[ok], j[ok]]
        if wrapped:
            d = wrap(d)
        out[int(lag)] = float(np.sqrt(np.mean(d ** 2))) if d.size >= 100 else float("nan")
    return out


def variance_reduction(
    before: np.ndarray,
    after: np.ndarray,
    valid: np.ndarray,
    lags_px: Sequence[int],
    wrapped: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compare the structure function before and after a correction, on the same pixel pairs.

    ``reduction[lag] = 1 - D_after / D_before`` with D the structure function (squared RMS difference); positive means the
    correction removed spatially correlated phase. ``median_reduction`` over the lags is the number to gate on.
    """
    sb = structure_function(before, valid, lags_px, wrapped, seed=seed)
    sa = structure_function(after, valid, lags_px, wrapped, seed=seed)
    red = {lag: (1.0 - (sa[lag] ** 2) / (sb[lag] ** 2)) if sb[lag] and np.isfinite(sb[lag]) and np.isfinite(sa[lag]) else float("nan")
           for lag in lags_px}
    finite = [v for v in red.values() if np.isfinite(v)]
    return {
        "lags_px": [int(l) for l in lags_px],
        "rms_before": sb, "rms_after": sa, "reduction": red,
        "median_reduction": float(np.median(finite)) if finite else float("nan"),
    }


# ---------------------------------------------------------------------------------------------------
# geocoded model data and time interpolation
# ---------------------------------------------------------------------------------------------------
class GeoGrid:
    """A regular lon/lat grid (north-up) that can be sampled bilinearly at arbitrary points. NaN outside it or where it has no data."""

    def __init__(self, array: np.ndarray, transform, nodata: Optional[float] = None):
        a = np.asarray(array, dtype=np.float64).copy()
        if nodata is not None:
            a[a == nodata] = np.nan
        self.array = a
        self.transform = transform
        if abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12:
            raise ValueError("a rotated lon/lat grid is not supported")

    @classmethod
    def from_raster(cls, path, band: int = 1) -> "GeoGrid":
        import rasterio

        with rasterio.open(str(path)) as src:
            if src.crs is not None and not src.crs.is_geographic:
                raise ValueError(f"{path}: expected a geographic (lon/lat) grid, found {src.crs}")
            return cls(src.read(band), src.transform, src.nodata)

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates

        t = self.transform
        col = (np.asarray(lon, dtype=np.float64) - t.c) / t.a - 0.5
        row = (np.asarray(lat, dtype=np.float64) - t.f) / t.e - 0.5
        h, w = self.array.shape
        inside = (col >= 0) & (col <= w - 1) & (row >= 0) & (row <= h - 1)
        out = np.full(col.shape, np.nan)
        if inside.any():
            out[inside] = map_coordinates(self.array, [row[inside], col[inside]], order=1, mode="nearest")
        return out


def hour_weights(dt):
    """Linear-in-time weights of the two ERA5 hours that bracket an acquisition time: ``[(hour_datetime, weight), ...]``.

    ERA5 is hourly. Taking the hour by truncating the time (what a plain ``strftime('%H')`` does) is wrong by up to an hour;
    the delay changes by millimetres per 30 minutes in a convective atmosphere.
    """
    from datetime import timedelta

    h0 = dt.replace(minute=0, second=0, microsecond=0)
    frac = (dt - h0).total_seconds() / 3600.0
    out = [(h0, 1.0 - frac)]
    if frac > 1e-9:
        out.append((h0 + timedelta(hours=1), frac))
    return [(h, w) for h, w in out if w > 1e-9]


def lowpass(phase: np.ndarray, valid: np.ndarray, sigma_px: float, wrapped: bool) -> np.ndarray:
    """Valid-pixel-normalised Gaussian low-pass of a phase screen (complex mean if wrapped, plain mean if unwrapped)."""
    if wrapped:
        return smooth_complex(phase, valid, sigma_px)
    from scipy.ndimage import gaussian_filter

    v = np.asarray(valid, dtype=bool) & np.isfinite(phase)
    num = gaussian_filter(np.where(v, phase, 0.0), sigma_px)
    den = gaussian_filter(v.astype(np.float64), sigma_px)
    out = num / np.maximum(den, 1e-12)
    out[den < 1e-3] = np.nan
    return out


# ---------------------------------------------------------------------------------------------------
# phase-height dependence
# ---------------------------------------------------------------------------------------------------
@dataclass
class HeightFit:
    slope_rad_per_km: float
    slope_se_rad_per_km: float
    t_stat: float
    partial_r2: Optional[float]
    relief_m: float
    n_used: int
    degree: int
    ramp: bool
    wrapped: bool
    coefficients: Dict[str, float]
    h_ref_m: float
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.reason == ""

    def component(self, heights: np.ndarray) -> np.ndarray:
        """The height-dependent phase only (the ramp is estimated to keep it out of the height term, but is not returned)."""
        hk = (np.asarray(heights, dtype=np.float64) - self.h_ref_m) / 1000.0
        out = self.coefficients["h1"] * hk
        if self.degree >= 2:
            out = out + self.coefficients["h2"] * hk ** 2
        return out

    def to_dict(self) -> Dict[str, Any]:
        d = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in self.__dict__.items()
             if k not in ("coefficients",)}
        d["coefficients"] = {k: float(v) for k, v in self.coefficients.items()}
        d["usable"] = self.usable
        return d


def _huber_irls(X: np.ndarray, y: np.ndarray, n_iter: int = 12, k: float = 1.345) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    for _ in range(n_iter):
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        u = np.abs(r) / (k * s)
        w = np.where(u <= 1.0, 1.0, 1.0 / u)
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta


def _design(x, y, hk, degree, ramp):
    cols = [np.ones_like(hk)]
    names = ["const"]
    if ramp:
        cols += [x, y]
        names += ["x", "y"]
    cols.append(hk)
    names.append("h1")
    if degree >= 2:
        cols.append(hk ** 2)
        names.append("h2")
    return np.column_stack(cols), names


def fit_height_dependence(
    phase: np.ndarray,
    heights: np.ndarray,
    valid: np.ndarray,
    *,
    wrapped: bool = False,
    degree: int = 1,
    ramp: bool = True,
    exclude: Optional[np.ndarray] = None,
    min_relief_m: float = 200.0,
    min_pixels: int = 2000,
    max_pixels: int = 60_000,
    n_blocks: int = 4,
    seed: int = 0,
) -> HeightFit:
    """Phase-versus-height slope with a ramp, robust to outliers, with a spatially-aware standard error.

    Unwrapped phase: Huber-weighted least squares on ``a + b x + c y + s h`` (+ ``s2 h^2``). The ramp term stops a long-wavelength
    trend from being credited to height wherever height and position are correlated. Wrapped phase: the slope that maximises the
    resultant length of ``exp(i (phase - s h))`` (no ramp, and only degree 1 -- unwrap first when a ramp or curvature matters).

    ``exclude`` masks pixels that must not drive the fit (known deformation, water, low coherence). The standard error comes
    from a block jackknife over ``n_blocks x n_blocks`` spatial blocks, because pixels that share an atmosphere are not
    independent and the textbook error from the residual variance is far too small.

    The fit is *refused* (``usable == False`` with ``reason``) when the relief is below ``min_relief_m`` (nothing to
    correlate with), when there are too few pixels, or when it cannot be estimated.
    """
    h_, w_ = phase.shape
    ok = np.asarray(valid, dtype=bool) & np.isfinite(phase) & np.isfinite(heights)
    if exclude is not None:
        ok &= ~np.asarray(exclude, dtype=bool)
    n_ok = int(ok.sum())

    def refuse(reason, relief=float("nan")):
        return HeightFit(float("nan"), float("nan"), float("nan"), None, relief, n_ok, degree, ramp, wrapped,
                         {"h1": 0.0}, float("nan"), reason)

    if n_ok < min_pixels:
        return refuse(f"only {n_ok} usable pixels (< {min_pixels})")
    rr, cc = np.nonzero(ok)
    hv = heights[rr, cc].astype(np.float64)
    relief = float(np.percentile(hv, 95) - np.percentile(hv, 5))
    if relief < min_relief_m:
        return refuse(f"relief {relief:.0f} m (< {min_relief_m:.0f} m): no height range to correlate with", relief)
    rng = np.random.default_rng(seed)
    if n_ok > max_pixels:
        pick = rng.choice(n_ok, max_pixels, replace=False)
        rr, cc, hv = rr[pick], cc[pick], hv[pick]
    pv = phase[rr, cc].astype(np.float64)
    x = 2.0 * cc / max(w_ - 1, 1) - 1.0
    y = 2.0 * rr / max(h_ - 1, 1) - 1.0
    h_ref = float(hv.mean())
    hk = (hv - h_ref) / 1000.0
    blk = (np.minimum((rr * n_blocks) // h_, n_blocks - 1) * n_blocks + np.minimum((cc * n_blocks) // w_, n_blocks - 1))

    if not wrapped:
        X, names = _design(x, y, hk, degree, ramp)
        beta = _huber_irls(X, pv)
        i_h = names.index("h1")
        slope = float(beta[i_h])
        # partial R2: variance explained by the height terms beyond the ramp, on the robust residual scale
        X0, _ = _design(x, y, hk * 0.0, 1, ramp)
        X0 = X0[:, :-1]
        b0 = _huber_irls(X0, pv)
        rss = float(np.sum((pv - X @ beta) ** 2))
        rss0 = float(np.sum((pv - X0 @ b0) ** 2))
        pr2 = float(1.0 - rss / rss0) if rss0 > 0 else 0.0
        jack = []
        for b in np.unique(blk):
            m = blk != b
            if m.sum() < 5 * X.shape[1]:
                continue
            jack.append(_huber_irls(X[m], pv[m], n_iter=6)[i_h])
        coef = {n: float(v) for n, v in zip(names, beta)}
    else:
        if degree != 1 or ramp:
            logger.info("wrapped-phase height fit: degree 1, no ramp (unwrap for more)")
        span = float(np.ptp(hv)) / 1000.0
        max_slope = 25.0 * 2 * np.pi / max(span, 1e-3)        # at most 25 fringes across the relief

        def best_slope(pp, hh):
            def flat(c):
                return np.abs(np.mean(np.exp(1j * (pp[None, :] - c[:, None] * hh[None, :])), axis=1))

            coarse = np.linspace(-max_slope, max_slope, 400)
            b0_ = float(coarse[np.argmax(flat(coarse))])
            fine = np.linspace(b0_ - (coarse[1] - coarse[0]), b0_ + (coarse[1] - coarse[0]), 200)
            return float(fine[np.argmax(flat(fine))])

        sub = rng.choice(len(hk), min(len(hk), 20_000), replace=False)
        slope = best_slope(pv[sub], hk[sub])
        r_full = float(np.abs(np.mean(np.exp(1j * (pv - slope * hk)))))
        r_zero = float(np.abs(np.mean(np.exp(1j * pv))))
        pr2 = float(max(0.0, 1.0 - (1 - r_full) / max(1 - r_zero, 1e-9)))
        jack = []
        for b in np.unique(blk):
            m = np.flatnonzero(blk != b)
            if m.size < 200:
                continue
            m = m[rng.choice(m.size, min(m.size, 8000), replace=False)]
            jack.append(best_slope(pv[m], hk[m]))
        coef = {"h1": slope}
    if len(jack) < 4:
        return refuse("too few spatial blocks for an error estimate", relief)
    jack = np.asarray(jack)
    n_b = len(jack)
    se = float(np.sqrt((n_b - 1) / n_b * np.sum((jack - jack.mean()) ** 2)))
    t = slope / se if se > 0 else float("inf")
    return HeightFit(slope, se, float(t), pr2, relief, int(len(hk)), degree, bool(ramp and not wrapped), wrapped, coef, h_ref)


# ---------------------------------------------------------------------------------------------------
# network: the troposphere belongs to a date; DEM error scales with the baseline
# ---------------------------------------------------------------------------------------------------
@dataclass
class DateSlopes:
    dates: List[str]
    slope_rad_per_km: Dict[str, float]
    beta_dem_scale: Optional[float]            # relative DEM height error implied by the covariance of slope with baseline
    beta_se: Optional[float]
    n_dates_for_beta: int
    pair_residual_rad_per_km: Dict[Tuple[str, str], float]
    reference_date: str
    rms_residual_rad_per_km: float
    n_pairs: int

    @property
    def dem_error_suspected(self) -> bool:
        """The date slopes covary with the dates' perpendicular baselines (needs >= 6 dates)."""
        return (self.beta_dem_scale is not None and self.beta_se is not None and self.n_dates_for_beta >= 6
                and abs(self.beta_dem_scale) > 3.0 * self.beta_se)


def _solve_date_values(pairs, values: Dict[Tuple[str, str], float], weights: Dict[Tuple[str, str], float], ref_d: str):
    dates = sorted({d for p in pairs for d in p})
    free = [d for d in dates if d != ref_d]
    idx = {d: i for i, d in enumerate(free)}
    A = np.zeros((len(pairs), len(free)))
    y = np.zeros(len(pairs))
    w = np.zeros(len(pairs))
    for n, p in enumerate(pairs):
        r, s = p
        if s in idx:
            A[n, idx[s]] += 1.0
        if r in idx:
            A[n, idx[r]] -= 1.0
        y[n] = values[p]
        w[n] = weights[p]
    sol, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    return {ref_d: 0.0, **{d: float(sol[idx[d]]) for d in free}}, y - A @ sol, w


def estimate_height_slopes_per_date(
    pair_fits: Dict[Tuple[str, str], Tuple[float, float]],
    k1_rad_per_m: Optional[Dict[Tuple[str, str], float]] = None,
    reference_date: Optional[str] = None,
) -> DateSlopes:
    """Per-date height slopes from per-pair slopes, and a test for a baseline-proportional (DEM) term.

    The stratified troposphere belongs to a *date*: ``slope_rs = S_s - S_r``. Solving for the ``S_d`` (reference date fixed at
    zero) by weighted least squares gives a correction that is consistent around loops, which separate per-pair fits are
    not, and the pair residuals show which fits were outliers.

    A DEM whose heights are wrong by a factor ``beta`` leaves ``k1 * beta * h`` of phase (``k1`` the topographic phase per
    metre of height, proportional to the perpendicular baseline), i.e. a height slope that scales with the baseline, whereas
    the troposphere has no reason to. A baseline term cannot be added to the pair model above, because a pair's baseline is the
    difference of its two dates' baselines and the term would be absorbed by the free ``S_d``. So it is tested *after*: the same
    network solve gives each date's baseline ``K_d`` relative to the reference (from ``k1_rad_per_m``), and ``beta`` is the slope
    of ``S_d`` on ``1000 * K_d`` over the dates, with its standard error from the scatter. It needs about 6 dates or more to mean
    anything; ``DateSlopes.dem_error_suspected`` encodes that.

    ``pair_fits`` maps ``(ref, sec) -> (slope, standard_error)`` in rad/km; ``k1_rad_per_m`` the pairs' signed ``k1``.
    """
    pairs = [p for p, (s, e) in pair_fits.items() if np.isfinite(s) and np.isfinite(e) and e > 0]
    dates = sorted({d for p in pairs for d in p})
    if len(pairs) < 2 or len(dates) < 3:
        raise ValueError("need at least 3 dates and 2 pairs to estimate per-date slopes")
    ref_d = reference_date or dates[0]
    if ref_d not in dates:
        raise ValueError(f"reference date {ref_d} is not in the network")
    vals = {p: pair_fits[p][0] for p in pairs}
    wts = {p: 1.0 / pair_fits[p][1] for p in pairs}
    S, res, w = _solve_date_values(pairs, vals, wts, ref_d)
    beta = beta_se = None
    n_beta = 0
    if k1_rad_per_m is not None and all(p in k1_rad_per_m for p in pairs):
        K, _, _ = _solve_date_values(pairs, {p: 1000.0 * k1_rad_per_m[p] for p in pairs}, wts, ref_d)
        d_use = [d for d in dates if d != ref_d]
        kx = np.array([K[d] for d in d_use])
        sy = np.array([S[d] for d in d_use])
        n_beta = len(d_use)
        if n_beta >= 3 and np.ptp(kx) > 0:
            X = np.column_stack([np.ones_like(kx), kx])
            coef, *_ = np.linalg.lstsq(X, sy, rcond=None)
            r = sy - X @ coef
            s2 = float(np.sum(r ** 2) / max(n_beta - 2, 1))
            beta = float(coef[1])
            beta_se = float(np.sqrt(s2 / np.sum((kx - kx.mean()) ** 2)))
    return DateSlopes(
        dates=dates, slope_rad_per_km=S, beta_dem_scale=beta, beta_se=beta_se, n_dates_for_beta=n_beta,
        pair_residual_rad_per_km={p: float(r_) for p, r_ in zip(pairs, res)}, reference_date=ref_d,
        rms_residual_rad_per_km=float(np.sqrt(np.mean(res ** 2))), n_pairs=len(pairs),
    )
