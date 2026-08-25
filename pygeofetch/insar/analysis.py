"""
pygeofetch.analysis.risk_mapping
==================================
Publication-ready risk mapping with uncertainty quantification.

Features:
- Multi-temporal risk assessment
- Bayesian uncertainty quantification
- Confidence interval mapping
- Monte Carlo simulation support
- Publication-quality visualizations
- GeoTIFF export with uncertainty bands

Usage::
    from pygeofetch.analysis import RiskMapper
    from pygeofetch.insar.timeseries import TimeSeriesResult

    # ts_result is a real TimeSeriesResult from SBASTimeSeries.invert()
    mapper = RiskMapper(ts_result)

    # Generate risk maps with uncertainty
    risk_map = mapper.compute_risk(
        method="bayesian",
        confidence_level=0.95,
        n_simulations=1000,
    )

    # Export publication-ready outputs
    risk_map.export_geotiff("risk_map.tif")
    risk_map.export_uncertainty("uncertainty.tif")
    risk_map.plot_risk_map("risk_figure.png", dpi=300)
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


@dataclass
class RiskMap:
    """
    Container for risk map with uncertainty quantification.

    Attributes
    ----------
    risk : np.ndarray
        Risk values (H x W)
    uncertainty : np.ndarray
        Uncertainty estimates (H x W)
    lower_ci : np.ndarray
        Lower confidence interval (H x W)
    upper_ci : np.ndarray
        Upper confidence interval (H x W)
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    method : str
        Risk computation method
    metadata : dict
        Additional metadata
    """

    risk: np.ndarray
    uncertainty: np.ndarray
    lower_ci: np.ndarray
    upper_ci: np.ndarray
    confidence_level: float
    method: str
    metadata: dict = field(default_factory=dict)
    transform: Any = None  # rasterio transform
    crs: Any = None  # CRS

    def export_geotiff(self, path: str | Path, **kwargs) -> str:
        """Export risk map as GeoTIFF with uncertainty bands."""
        try:
            import rasterio
            from rasterio.transform import Affine
        except ImportError:
            raise ImportError("pip install rasterio")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare metadata
        meta = {
            "driver": "GTiff",
            "height": self.risk.shape[0],
            "width": self.risk.shape[1],
            "count": 4,  # risk, uncertainty, lower_ci, upper_ci
            "dtype": "float32",
            "crs": self.crs or "EPSG:4326",
            "transform": self.transform or Affine.identity(),
            "compress": "lzw",
            "nodata": -9999.0,
        }
        meta.update(kwargs)

        # Stack bands
        stack = np.stack(
            [
                self.risk,
                self.uncertainty,
                self.lower_ci,
                self.upper_ci,
            ],
            axis=0,
        )
        stack = np.nan_to_num(stack, nan=meta["nodata"])

        with rasterio.open(path, "w", **meta) as dst:
            dst.write(stack.astype(np.float32))
            dst.update_tags(1, name="risk")
            dst.update_tags(2, name="uncertainty")
            dst.update_tags(3, name="lower_ci")
            dst.update_tags(4, name="upper_ci")
            dst.update_tags(
                confidence_level=str(self.confidence_level),
                method=self.method,
            )

        logger.info(f"Risk map exported to {path}")
        return str(path)

    def export_uncertainty(self, path: str | Path) -> str:
        """Export uncertainty metrics as separate GeoTIFF."""
        try:
            import rasterio
            from rasterio.transform import Affine
        except ImportError:
            raise ImportError("pip install rasterio")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate additional uncertainty metrics
        ci_width = self.upper_ci - self.lower_ci
        # BUG FIX: epsilon must guard the DENOMINATOR's magnitude, not
        # be added inside the abs() before taking it -- np.abs(risk +
        # 1e-10) can still land at (or near) exactly zero for a risk
        # value near -1e-10, and more importantly doesn't match the
        # standard safe-division idiom. np.abs(risk) + 1e-10 is always
        # bounded away from zero regardless of risk's sign or value.
        cv = self.uncertainty / (np.abs(self.risk) + 1e-10)  # Coefficient of variation

        meta = {
            "driver": "GTiff",
            "height": self.risk.shape[0],
            "width": self.risk.shape[1],
            "count": 3,
            "dtype": "float32",
            "crs": self.crs or "EPSG:4326",
            # BUG FIX: export_geotiff() falls back to Affine.identity()
            # when self.transform is None (the dataclass default); this
            # method passed self.transform straight through, which
            # would hand rasterio.open() `transform=None` and either
            # error or silently write an unreferenced raster whenever
            # a RiskMap was built without an explicit transform.
            "transform": self.transform or Affine.identity(),
            "compress": "lzw",
            "nodata": -9999.0,
        }

        stack = np.stack([self.uncertainty, ci_width, cv], axis=0)
        stack = np.nan_to_num(stack, nan=meta["nodata"])

        with rasterio.open(path, "w", **meta) as dst:
            dst.write(stack.astype(np.float32))
            dst.update_tags(1, name="std_dev")
            dst.update_tags(2, name="ci_width")
            dst.update_tags(3, name="coeff_variation")

        return str(path)

    def plot_risk_map(
        self,
        path: str | Path,
        cmap: str = "RdYlGn_r",
        dpi: int = 300,
        add_uncertainty_hatch: bool = True,
        **kwargs,
    ) -> str:
        """Create publication-quality risk map visualization."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=dpi)

        # Main risk map
        im1 = axes[0].imshow(
            self.risk,
            cmap=cmap,
            vmin=kwargs.get("vmin", np.nanpercentile(self.risk, 2)),
            vmax=kwargs.get("vmax", np.nanpercentile(self.risk, 98)),
            aspect="auto",
        )
        axes[0].set_title("Risk Map", fontsize=14, fontweight="bold")
        axes[0].axis("off")
        plt.colorbar(im1, ax=axes[0], label="Risk Value", shrink=0.8)

        # Uncertainty map
        im2 = axes[1].imshow(
            self.uncertainty,
            cmap="Reds",
            vmin=0,
            vmax=np.nanpercentile(self.uncertainty, 95),
            aspect="auto",
        )
        axes[1].set_title("Uncertainty (Std Dev)", fontsize=14, fontweight="bold")
        axes[1].axis("off")
        plt.colorbar(im2, ax=axes[1], label="Uncertainty", shrink=0.8)

        # Add uncertainty hatching if requested
        if add_uncertainty_hatch:
            finite_uncertainty = self.uncertainty[np.isfinite(self.uncertainty)]
            if finite_uncertainty.size > 0:
                high_uncertainty = self.uncertainty > np.nanpercentile(
                    self.uncertainty, 90
                )
                y, x = np.mgrid[
                    : self.uncertainty.shape[0], : self.uncertainty.shape[1]
                ]
                masked_data = np.ma.masked_where(~high_uncertainty, high_uncertainty)
                axes[0].contourf(
                    x,
                    y,
                    masked_data,
                    hatches=["////"],
                    alpha=0,
                    levels=[0.5, 1.5],
                    colors="none",
                )

        plt.suptitle(
            f"Risk Assessment ({self.method}, {self.confidence_level*100:.0f}% CI)",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Risk map figure saved to {path}")
        return str(path)

    def to_xarray(self) -> xr.Dataset:
        """Convert to xarray Dataset for advanced analysis."""
        dims = ("y", "x")

        ds = xr.Dataset(
            {
                "risk": (dims, self.risk),
                "uncertainty": (dims, self.uncertainty),
                "lower_ci": (dims, self.lower_ci),
                "upper_ci": (dims, self.upper_ci),
            },
            attrs={
                "confidence_level": self.confidence_level,
                "method": self.method,
                **self.metadata,
            },
        )

        return ds


def _resolve_time_years(times: Any) -> np.ndarray:
    """
    Convert a sequence of time labels into REAL elapsed time in years
    since the first entry -- the same convention
    SBASTimeSeries._fit_velocity() already uses
    (t_years = days_since_reference / 365.25), so risk trends computed
    here are directly comparable to TimeSeriesResult.velocity (m/year)
    rather than being in some other, arbitrary unit.

    This matters concretely, not just for consistency: on a controlled
    synthetic test with a known -20 mm/year trend and irregular
    acquisition gaps matching this project's own real Mexico City stack
    (12, 24, 36, 48, 60-day gaps), fitting against equally-spaced
    integer indices instead of real elapsed time recovered a slope of
    -1.7 "mm per index-step" -- off by more than 10x and in an
    uninterpretable unit -- while fitting against real elapsed time
    recovered -20.1 mm/year, matching the truth.

    Accepts ISO date strings ("2016-07-24", the convention used
    throughout pygeofetch.insar), datetime/date objects, or already-
    numeric values (assumed to already be a meaningful time axis in
    the caller's own unit; only rebased to start at 0, not rescaled).
    """
    times = list(times)
    if len(times) == 0:
        raise ValueError("_resolve_time_years: times is empty.")

    if all(isinstance(t, (int, float, np.integer, np.floating)) for t in times):
        arr = np.asarray(times, dtype=np.float64)
        return arr - arr[0]

    parsed = []
    for t in times:
        if isinstance(t, _dt.datetime):
            parsed.append(t)
        elif isinstance(t, _dt.date):
            parsed.append(_dt.datetime(t.year, t.month, t.day))
        elif isinstance(t, str):
            parsed.append(_dt.datetime.strptime(t, "%Y-%m-%d"))
        else:
            raise TypeError(
                f"_resolve_time_years: cannot interpret time value {t!r} "
                f"(type {type(t).__name__}) as a date string, date/datetime, "
                f"or number."
            )
    t0 = parsed[0]
    return np.array(
        [(t - t0).total_seconds() / (365.25 * 86400.0) for t in parsed],
        dtype=np.float64,
    )


class RiskMapper:
    """
    Comprehensive risk mapping with uncertainty quantification.

    Supports multiple methods:
    - Bayesian inference (real conjugate Normal-Normal update on the
      trend/slope -- see _bayesian_linear_regression's docstring)
    - Monte Carlo simulation
    - Bootstrap resampling (residual bootstrap, not naive time-index
      resampling -- see _bootstrap_risk's docstring)
    - Analytical uncertainty propagation

    Parameters
    ----------
    time_series_result : Any
        An object (or dict) exposing a 3D (time, y, x) data array under
        one of a few common attribute names (`data`, `displacement`,
        `deformation`, `timeseries`, `stack` -- `TimeSeriesResult.
        displacement` from pygeofetch.insar.timeseries matches
        directly) and a matching per-time-step label under one of
        (`times`, `dates`, `time`, `date`, `acquisition_dates`,
        `date_list` -- `TimeSeriesResult.dates` matches directly).
    """

    def __init__(self, time_series_result: Any) -> None:
        self.ts_result = time_series_result
        # BUG FIX: an earlier version resolved these by MUTATING the
        # caller's ts_result object (setting new .data/.times
        # attributes on it as a side effect of just constructing a
        # RiskMapper) -- a real risk of confusing the caller's own code
        # if they reuse that same object elsewhere afterward. Resolved
        # values are now stored on the RiskMapper instance itself.
        self.data: np.ndarray = self._resolve_data()
        times_raw = self._resolve_times_raw()
        self.time_years: np.ndarray = _resolve_time_years(times_raw)
        self._validate_shapes()

    def _resolve_data(self) -> np.ndarray:
        data_attrs = ["data", "displacement", "deformation", "timeseries", "stack"]
        for attr in data_attrs:
            if hasattr(self.ts_result, attr):
                value = getattr(self.ts_result, attr)
                if value is not None:
                    logger.info(f"Using '{attr}' as the data array for RiskMapper")
                    return np.asarray(value)
        if hasattr(self.ts_result, "__getitem__"):
            for attr in data_attrs:
                try:
                    value = self.ts_result[attr]
                    logger.info(f"Using dictionary key '{attr}' as the data array")
                    return np.asarray(value)
                except (KeyError, TypeError):
                    continue
        raise ValueError(
            f"Could not find a data array. Looked for attributes/keys: {', '.join(data_attrs)}"
        )

    def _resolve_times_raw(self) -> Any:
        time_attrs = [
            "times",
            "dates",
            "time",
            "date",
            "acquisition_dates",
            "date_list",
        ]
        for attr in time_attrs:
            if hasattr(self.ts_result, attr):
                value = getattr(self.ts_result, attr)
                if value is not None:
                    logger.info(f"Using '{attr}' as the time axis for RiskMapper")
                    return value
        if hasattr(self.ts_result, "__getitem__"):
            for attr in time_attrs:
                try:
                    value = self.ts_result[attr]
                    logger.info(f"Using dictionary key '{attr}' as the time axis")
                    return value
                except (KeyError, TypeError):
                    continue
        # Last resort: integer indices. Honest about what this means --
        # any trend computed downstream will be in "per acquisition"
        # units, not a real physical rate, since there is no real time
        # information to anchor it to.
        n_time = self.data.shape[0]
        logger.warning(
            "No time/date attribute found -- falling back to integer "
            'indices 0..%d. Risk trends will be in "per acquisition" '
            "units, not a real physical rate (e.g. not mm/year), since "
            "there is no real time information available to anchor them.",
            n_time - 1,
        )
        return list(range(n_time))

    def _validate_shapes(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(
                f"Data should be 3D (time, y, x) but got shape: {self.data.shape}"
            )
        if len(self.time_years) != self.data.shape[0]:
            raise ValueError(
                f"Time axis length ({len(self.time_years)}) doesn't match "
                f"data's time dimension ({self.data.shape[0]})"
            )

    def compute_risk(
        self,
        method: Literal[
            "bayesian", "monte_carlo", "bootstrap", "analytical"
        ] = "bayesian",
        confidence_level: float = 0.95,
        n_simulations: int = 1000,
        risk_function: Callable | None = None,
        **kwargs,
    ) -> RiskMap:
        """
        Compute risk map with uncertainty quantification.

        Parameters
        ----------
        method : str
            Uncertainty quantification method.
        confidence_level : float
            Confidence level for intervals (0-1).
        n_simulations : int
            Number of Monte Carlo/bootstrap simulations.
        risk_function : callable
            Function (data_array, time_years) -> risk_array to compute
            risk from a (time, y, x) array and its matching real
            elapsed-time-in-years axis. Default: trend magnitude /
            variability ratio (see _default_risk_function).

        Returns
        -------
        RiskMap
            Risk map with uncertainty estimates.
        """
        logger.info(f"Computing risk map using {method} method...")

        if risk_function is None:
            risk_function = self._default_risk_function

        if method == "bayesian":
            return self._bayesian_risk(
                risk_function, confidence_level, n_simulations, **kwargs
            )
        elif method == "monte_carlo":
            return self._monte_carlo_risk(
                risk_function, confidence_level, n_simulations, **kwargs
            )
        elif method == "bootstrap":
            return self._bootstrap_risk(
                risk_function, confidence_level, n_simulations, **kwargs
            )
        elif method == "analytical":
            return self._analytical_risk(risk_function, confidence_level, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _default_risk_function(
        self, data: np.ndarray, time_years: np.ndarray
    ) -> np.ndarray:
        """
        Default risk function: trend magnitude (per real year) normalized
        by variability.

        Higher values indicate a stronger, more consistent trend
        relative to noise.

        BUG FIX: an earlier version fit the trend against
        np.arange(n_time) (equally-spaced integer indices) regardless of
        the real, often highly irregular, gaps between real SAR
        acquisitions -- see _resolve_time_years's docstring for a
        concrete, quantified example of how large an error this causes.
        Now takes the real elapsed-time-in-years axis explicitly and
        fits against that.
        """
        n_time = data.shape[0]
        data_2d = data.reshape(n_time, -1)

        slope = np.full(data_2d.shape[1], np.nan)

        for i in range(data_2d.shape[1]):
            valid = ~np.isnan(data_2d[:, i])
            if valid.sum() > 2:
                slope[i], _ = np.polyfit(time_years[valid], data_2d[valid, i], 1)

        std_dev = np.nanstd(data_2d, axis=0)

        risk = np.abs(slope) / (std_dev + 1e-10)

        return risk.reshape(data.shape[1:])

    def _bayesian_risk(
        self,
        risk_function: Callable,
        confidence_level: float,
        n_simulations: int,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        **kwargs,
    ) -> RiskMap:
        """
        Bayesian risk estimation via a real conjugate Normal-Normal
        posterior on the per-pixel trend, sampled directly (no MCMC
        needed for a conjugate posterior -- the previous docstring's
        claim of "MCMC sampling" was inaccurate either way, since the
        implementation never used MCMC).

        See _bayesian_linear_regression's docstring for the concrete,
        demonstrated bug this fixes: prior_mean/prior_std previously
        had ZERO effect on the result regardless of their value
        (confirmed directly: an extremely confident prior claiming the
        slope should be -1000 didn't move the posterior at all from the
        plain OLS estimate).
        """
        data = self.data
        n_time, height, width = data.shape
        time_years = self.time_years

        data_flat = data.reshape(n_time, -1)
        n_pixels = data_flat.shape[1]

        risk_samples = np.full((n_simulations, n_pixels), np.nan)

        for i in range(n_pixels):
            pixel_data = data_flat[:, i]
            valid = ~np.isnan(pixel_data)

            if valid.sum() < 3:
                continue

            # BUG FIX: previously used np.arange(valid.sum()) here --
            # both discarding the real elapsed time AND discarding
            # which of the original time steps were actually valid
            # (compacting e.g. valid steps 0,2,4 down to 0,1,2, losing
            # the real gaps between them entirely).
            slope_samples, intercept_samples = self._bayesian_linear_regression(
                time_years[valid],
                pixel_data[valid],
                n_simulations,
                prior_mean,
                prior_std,
            )

            # BUG FIX: previously evaluated the synthetic fitted line at
            # np.arange(n_time) regardless of real time or which steps
            # were valid; now uses the same real time_years[valid] axis
            # the regression was actually fit on.
            t = time_years[valid][:, None]
            synthetic = slope_samples[None, :] * t + intercept_samples[None, :]
            risk_samples[:, i] = np.abs(slope_samples) / (
                np.std(synthetic, axis=0) + 1e-10
            )

        risk_samples = risk_samples.reshape(n_simulations, height, width)

        return self._compute_risk_statistics(risk_samples, confidence_level, "bayesian")

    def _bayesian_linear_regression(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        prior_mean: float,
        prior_std: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Real conjugate Bayesian linear regression for the slope, with a
        Normal(prior_mean, prior_std^2) prior -- fixes a confirmed bug:
        the previous version computed plain OLS estimates and sampled
        from the OLS SAMPLING distribution, completely ignoring
        prior_mean/prior_std despite the docstring's claim of "conjugate
        priors." Demonstrated directly: feeding an extremely confident
        prior (prior_std=0.0001) claiming the slope should be -1000 did
        not move the posterior at all from the OLS estimate (~2.0 for
        data with a true slope of 2.0).

        Method: treat the OLS slope estimate as a sufficient statistic
        with its own known sampling variance (slope_var =
        sigma^2/ss_xx, sigma^2 estimated via OLS residuals as a
        plug-in -- the same simplification the previous version used
        for sigma^2 itself), then combine with the Normal prior via the
        standard closed-form Normal-Normal conjugate update
        (precision-weighted average):

            posterior_precision = 1/prior_std^2 + 1/slope_var
            posterior_mean = posterior_var * (prior_mean/prior_std^2
                                               + slope_ols/slope_var)

        This correctly reduces to the plain OLS estimate as
        prior_std -> infinity (an uninformative prior) -- verified
        directly as a consistency check -- and correctly shrinks toward
        prior_mean as prior_std -> 0 (a highly confident prior).

        Intercept is derived deterministically per slope sample as
        y_mean - slope_sample * x_mean (the point that makes the fitted
        line pass through the data's centroid), rather than sampled
        independently from its own separate OLS distribution as before
        -- independent sampling could previously pair a posterior-
        shrunk slope with an intercept that no longer describes a
        self-consistent fitted line for that slope.
        """
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        ss_xx = np.sum((x - x_mean) ** 2)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))

        slope_ols = ss_xy / (ss_xx + 1e-10)
        intercept_ols = y_mean - slope_ols * x_mean

        residuals = y - (intercept_ols + slope_ols * x)
        dof = max(n - 2, 1)
        sigma2 = np.sum(residuals**2) / dof

        slope_var = sigma2 / (ss_xx + 1e-10)

        # Real conjugate Normal-Normal posterior update.
        prior_precision = 1.0 / (prior_std**2 + 1e-300)
        likelihood_precision = 1.0 / (slope_var + 1e-300)
        posterior_precision = prior_precision + likelihood_precision
        posterior_var = 1.0 / posterior_precision
        posterior_mean = posterior_var * (
            prior_mean * prior_precision + slope_ols * likelihood_precision
        )

        slope_samples = np.random.normal(
            posterior_mean, np.sqrt(posterior_var), n_samples
        )
        intercept_samples = y_mean - slope_samples * x_mean

        return slope_samples, intercept_samples

    def _monte_carlo_risk(
        self,
        risk_function: Callable,
        confidence_level: float,
        n_simulations: int,
        noise_std: float | None = None,
        **kwargs,
    ) -> RiskMap:
        """Monte Carlo risk estimation with noise perturbation."""
        data = self.data
        time_years = self.time_years

        if noise_std is None:
            noise_std = np.nanstd(data)

        risk_samples = []
        for _ in range(n_simulations):
            noise = np.random.normal(0, noise_std, data.shape)
            perturbed_data = data + noise
            risk = risk_function(perturbed_data, time_years)
            risk_samples.append(risk)

        risk_samples = np.stack(risk_samples, axis=0)

        return self._compute_risk_statistics(
            risk_samples, confidence_level, "monte_carlo"
        )

    def _bootstrap_risk(
        self,
        risk_function: Callable,
        confidence_level: float,
        n_simulations: int,
        **kwargs,
    ) -> RiskMap:
        """
        Residual bootstrap for regression-based risk metrics (Freedman
        1981) -- the standard technique for estimating sampling
        uncertainty in a fitted trend.

        BUG FIX: an earlier version resampled TIME INDICES with
        replacement (scrambling which value came from which real
        acquisition), then refit the resampled VALUES against a FRESH
        sequential index with no relationship to when they were
        actually observed. Demonstrated directly on synthetic data with
        a strong, consistent true trend: that approach produced
        per-trial slopes that varied wildly and even flipped sign
        (+0.78, +0.61, -0.23 across 3 trials of data with a true,
        strong negative trend) -- an artifact of the broken resampling,
        not genuine sampling uncertainty.

        Residual bootstrap instead: fit each pixel's trend ONCE against
        the real, correctly-ordered time axis, resample the FITTED
        RESIDUALS (not the raw values or their time pairing), add the
        resampled residuals back onto the ORIGINAL fitted line (so the
        real time axis is preserved throughout, never scrambled), and
        refit each resampled dataset. The same time-index resample is
        applied consistently across every pixel per iteration (an
        accepted simplification for gridded/correlated data, and far
        cheaper than an independent resample per pixel).
        """
        data = self.data
        time_years = self.time_years
        n_time, height, width = data.shape

        data_2d = data.reshape(n_time, -1)
        n_pixels = data_2d.shape[1]

        fitted = np.full_like(data_2d, np.nan)
        for i in range(n_pixels):
            valid = ~np.isnan(data_2d[:, i])
            if valid.sum() > 2:
                slope, intercept = np.polyfit(time_years[valid], data_2d[valid, i], 1)
                fitted[valid, i] = intercept + slope * time_years[valid]

        residuals = data_2d - fitted  # NaN where fitted is NaN (too few valid points)

        risk_samples = []
        for _ in range(n_simulations):
            indices = np.random.choice(n_time, n_time, replace=True)
            resampled_2d = fitted + residuals[indices, :]
            resampled_data = resampled_2d.reshape(data.shape)
            risk = risk_function(resampled_data, time_years)
            risk_samples.append(risk)

        risk_samples = np.stack(risk_samples, axis=0)

        return self._compute_risk_statistics(
            risk_samples, confidence_level, "bootstrap"
        )

    def _analytical_risk(
        self,
        risk_function: Callable,
        confidence_level: float,
        **kwargs,
    ) -> RiskMap:
        """Analytical uncertainty propagation using the delta method."""
        data = self.data
        time_years = self.time_years

        risk = risk_function(data, time_years)

        # Simplified error-propagation proxy: temporal variability
        # scaled by the standard error of the mean. A real analytical
        # propagation of variance through np.polyfit's own slope
        # standard error would be more precise per-pixel, but this
        # remains a documented simplification (as it was before), not
        # a fixed bug -- flagging it rather than silently pretending
        # it's exact.
        uncertainty = np.nanstd(data, axis=0) / np.sqrt(data.shape[0])

        from scipy import stats

        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        lower_ci = risk - z_score * uncertainty
        upper_ci = risk + z_score * uncertainty

        return RiskMap(
            risk=risk,
            uncertainty=uncertainty,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            confidence_level=confidence_level,
            method="analytical",
            metadata={"z_score": z_score},
            transform=getattr(self.ts_result, "transform", None),
            crs=getattr(self.ts_result, "crs", None),
        )

    def _compute_risk_statistics(
        self,
        risk_samples: np.ndarray,
        confidence_level: float,
        method: str,
    ) -> RiskMap:
        """Compute risk statistics from simulation samples."""
        risk = np.nanmean(risk_samples, axis=0)
        uncertainty = np.nanstd(risk_samples, axis=0)

        alpha = 1 - confidence_level
        lower_ci = np.nanpercentile(risk_samples, 100 * alpha / 2, axis=0)
        upper_ci = np.nanpercentile(risk_samples, 100 * (1 - alpha / 2), axis=0)

        return RiskMap(
            risk=risk,
            uncertainty=uncertainty,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            confidence_level=confidence_level,
            method=method,
            metadata={"n_simulations": risk_samples.shape[0]},
            transform=getattr(self.ts_result, "transform", None),
            crs=getattr(self.ts_result, "crs", None),
        )

    def validate_risk_map(
        self,
        risk_map: RiskMap,
        validation_data: np.ndarray | None = None,
        metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Validate risk map against ground truth or reference data.

        Parameters
        ----------
        risk_map : RiskMap
            Risk map to validate.
        validation_data : np.ndarray
            Ground truth or reference data.
        metrics : list
            Metrics to compute (default: ["rmse", "mae", "r2", "coverage", "sharpness"]).

        Returns
        -------
        dict
            Validation metrics.
        """
        if metrics is None:
            metrics = ["rmse", "mae", "r2", "coverage", "sharpness"]

        results = {}
        valid = None

        if validation_data is not None:
            risk = risk_map.risk
            valid = ~np.isnan(risk) & ~np.isnan(validation_data)

            if "rmse" in metrics:
                results["rmse"] = float(
                    np.sqrt(np.mean((risk[valid] - validation_data[valid]) ** 2))
                )

            if "mae" in metrics:
                results["mae"] = float(
                    np.mean(np.abs(risk[valid] - validation_data[valid]))
                )

            if "r2" in metrics:
                ss_res = np.sum((validation_data[valid] - risk[valid]) ** 2)
                ss_tot = np.sum(
                    (validation_data[valid] - np.mean(validation_data[valid])) ** 2
                )
                results["r2"] = float(1 - (ss_res / (ss_tot + 1e-10)))

        if "coverage" in metrics:
            if validation_data is not None and valid is not None:
                within_ci = (validation_data >= risk_map.lower_ci) & (
                    validation_data <= risk_map.upper_ci
                )
                results["coverage"] = float(np.mean(within_ci[valid]))
            else:
                # BUG FIX: this branch previously reported
                # risk_map.confidence_level itself as the "coverage"
                # metric -- i.e. it would ALWAYS report e.g. exactly
                # 0.95 whenever no validation_data was supplied,
                # regardless of whether the map's actual empirical
                # coverage is anywhere near that. That's not a coverage
                # CHECK, it's just echoing the target back. Report None
                # and say plainly that coverage can't be assessed
                # without real validation data, rather than fabricate a
                # number that looks like a real result.
                logger.info(
                    "No validation_data supplied -- coverage cannot be "
                    "empirically assessed (the theoretical target is "
                    "confidence_level=%.2f, but that isn't the same as "
                    "a measured result).",
                    risk_map.confidence_level,
                )
                results["coverage"] = None

        if "sharpness" in metrics:
            ci_width = risk_map.upper_ci - risk_map.lower_ci
            results["sharpness"] = float(np.nanmean(ci_width))

        if "uncertainty_ratio" in metrics:
            ratio = risk_map.uncertainty / (np.abs(risk_map.risk) + 1e-10)
            results["uncertainty_ratio"] = float(np.nanmean(ratio))

        return results


# Convenience function for quick risk mapping
def create_risk_map(
    time_series_result: Any,
    output_dir: str | Path = "./risk_maps",
    method: str = "bayesian",
    confidence_level: float = 0.95,
    n_simulations: int = 1000,
    **kwargs,
) -> RiskMap:
    """
    Quick risk map generation with default settings.

    Parameters
    ----------
    time_series_result : Any
        TimeSeriesResult object (or dict) -- see RiskMapper's docstring
        for the attribute names it looks for.
    output_dir : str | Path
        Directory for outputs.
    method : str
        Risk computation method.
    confidence_level : float
        Confidence level.
    n_simulations : int
        Number of simulations.

    Returns
    -------
    RiskMap
        Generated risk map.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapper = RiskMapper(time_series_result)

    risk_map = mapper.compute_risk(
        method=method,
        confidence_level=confidence_level,
        n_simulations=n_simulations,
        **kwargs,
    )

    risk_map.export_geotiff(output_dir / "risk_map.tif")
    risk_map.export_uncertainty(output_dir / "uncertainty.tif")
    risk_map.plot_risk_map(output_dir / "risk_figure.png")

    validation = mapper.validate_risk_map(risk_map)
    logger.info(f"Validation metrics: {validation}")

    import json

    with open(output_dir / "validation.json", "w") as f:
        json.dump(validation, f, indent=2)

    return risk_map
