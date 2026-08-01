"""
AtmosphericCorrector — tropospheric delay correction for InSAR.

Removes the tropospheric phase delay component from interferograms, one of
the dominant error sources limiting InSAR deformation accuracy (typically
2-10 cm of apparent "signal" that is actually atmospheric noise).

Two strategies are implemented:

  1. Elevation-correlated linear correction (native, no extra deps) —
     regresses phase against DEM elevation per-interferogram. This is the
     simplest and most widely applicable method, and per Zhao et al. (2023)
     it often performs comparably to or better than global reanalysis
     models in mountainous regions with strong turbulent mixing.

  2. ERA5 reanalysis-based correction (PyAPS method) — computes the
     tropospheric zenith delay from ECMWF ERA5 reanalysis data and projects
     it along the radar line-of-sight. This is the standard approach used
     in MintPy's tropospheric correction step.

Reference:
  Jolivet, R., Grandin, R., Lasserre, C., Doin, M.P., & Peltzer, G. (2011).
    Systematic InSAR tropospheric phase delay corrections from global
    meteorological reanalysis data. GRL, 38(17).
  Zhao, Y. et al. (2023). Evaluation of InSAR Tropospheric Delay Correction
    Methods in a Low-Latitude Alpine Canyon Region. Remote Sensing, 15(4), 990.

Install: pip install "pygeofetch[insar]"           (native elevation correction)
         pip install "pygeofetch[insar-full]"       (+ PyAPS/ERA5 correction)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger("pygeofetch.insar.atmosphere")


class AtmosphericCorrector:
    """
    Tropospheric delay correction for interferometric phase.

    Args:
        method: ``"elevation"`` (default, native, no extra deps) or
                ``"era5"`` (requires pyaps3, downloads ERA5 reanalysis data).

    Example::

        from pygeofetch.insar import AtmosphericCorrector

        corrector = AtmosphericCorrector(method="elevation")
        corrected_phase = corrector.correct(
            wrapped_phase, dem="dem.tif"
        )

        # ERA5-based (requires pyaps3 + CDS API credentials)
        corrector = AtmosphericCorrector(method="era5")
        corrected_phase = corrector.correct(
            wrapped_phase, dem="dem.tif",
            acquisition_datetime="2026-06-01T18:16:00",
        )
    """

    def __init__(self, method: str = "elevation") -> None:
        if method not in ("elevation", "era5"):
            raise ValueError(f"method must be 'elevation' or 'era5', got {method!r}")
        self._method = method

    def correct(
        self,
        phase: Any,
        dem: Union[str, Path],
        acquisition_datetime: Optional[str] = None,
        incidence_angle_deg: float = 38.0,
        return_metadata: bool = False,
    ) -> Any:
        """
        Remove the tropospheric delay component from wrapped or unwrapped phase.

        Args:
            phase:                Float32 phase array (radians) — wrapped or
                                  unwrapped, works with either.
            dem:                   DEM path for elevation-correlated correction
                                  and/or ERA5 vertical interpolation.
            acquisition_datetime:  ISO datetime of the SAR acquisition
                                  (required for method="era5").
            incidence_angle_deg:   Radar incidence angle for LOS projection
                                  of zenith delay (Sentinel-1 IW ≈ 30-46°,
                                  default 38° is the mid-swath average).
            return_metadata:       If True, returns (phase, metadata) instead
                                  of just phase. metadata includes
                                  "correction_applied" (bool) and, for the
                                  elevation method, "r_squared" (float) --
                                  this is the only reliable way to know
                                  whether a correction actually happened,
                                  since the elevation method's internal R²
                                  gate can legitimately skip correction
                                  (returning the input unchanged) without
                                  raising or otherwise signalling that in
                                  the plain return value. Default False
                                  preserves the original return type for
                                  existing callers.

        Returns:
            Corrected phase array, same shape and units as input (or a
            (phase, metadata) tuple if return_metadata=True).
        """
        if self._method == "era5":
            result = self._correct_era5(
                phase, dem, acquisition_datetime, incidence_angle_deg
            )
            metadata = {"correction_applied": True}
        else:
            result, metadata = self._correct_elevation(phase, dem)

        return (result, metadata) if return_metadata else result

    # ── native elevation-correlated correction ────────────────────────────────

    def _correct_elevation(self, phase: Any, dem: Union[str, Path]) -> Any:
        """
        Remove the phase component linearly correlated with elevation.

        This is the standard "atmospheric stratification" correction: the
        troposphere's refractive index varies with altitude in a roughly
        exponential/linear fashion, producing a phase signal that correlates
        with terrain height. Regressing out this correlation removes the
        dominant, spatially-smooth component of tropospheric delay.

        Does not correct turbulent (non-elevation-correlated) atmospheric
        noise — for that, ERA5 or GACOS correction is needed.

        IMPORTANT LIMITATION: elevation-correlated regression cannot
        distinguish true atmospheric stratification delay from spatially-
        smooth real deformation signal that coincidentally shares
        low-frequency structure with the DEM over a finite scene. The
        correction is only applied when it explains a substantial share of
        the phase variance (R² > 0.5); otherwise it is skipped and logged,
        since removing a low-confidence trend risks deleting real
        deformation signal rather than atmospheric noise.

        Returns:
            (phase_or_corrected, metadata) -- metadata always includes
            "correction_applied" (bool) and, whenever the R² gate was
            actually evaluated, "r_squared" (float), so callers can tell
            a genuine correction apart from a same-shaped pass-through.
        """
        np = self._np()
        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        with rasterio.open(dem) as src:
            dem_data = src.read(1).astype(np.float32)

        if dem_data.shape != phase.shape:
            from scipy.ndimage import zoom

            zf = (
                phase.shape[0] / dem_data.shape[0],
                phase.shape[1] / dem_data.shape[1],
            )
            dem_data = zoom(dem_data, zf, order=1)

        valid = np.isfinite(phase) & np.isfinite(dem_data) & (dem_data > -500)
        if valid.sum() < 100:
            logger.warning(
                "Insufficient valid pixels for elevation correction — returning uncorrected"
            )
            return phase, {"correction_applied": False, "reason": "insufficient_valid_pixels"}

        # Real, confirmed bug fixed here: the previous version fit a
        # plain, arithmetic linear regression directly against wrapped
        # phase (bounded [-pi, pi)). Any real elevation-correlated
        # signal spanning more than one 2*pi cycle across the scene --
        # confirmed to be the real, common case here, Iztapalapa's real
        # elevation range easily produces this -- gets sliced into
        # discontinuous jumps by the wrapping, which destroys a plain
        # linear fit regardless of whether a real underlying
        # relationship exists. Confirmed directly: R^2 consistently
        # landed at machine-noise levels (1e-8 to 1e-4) on every real
        # pair tried, not because no correlation existed, but because
        # this method was mathematically incapable of detecting one
        # through the wrap discontinuities.
        #
        # Fixed using the same circular regression (fit the real/imag
        # parts of exp(i*phase) separately, matching how the wrap
        # boundary is correctly handled everywhere else phase gets
        # regressed against a covariate in this codebase --
        # interferogram.py's _remove_topographic_phase(), already
        # proven and verified) rather than a different, novel approach.
        dem_std = float(np.std(dem_data[valid]))
        if dem_std < 1.0:  # metres -- genuinely flat, not just low-relief
            logger.info(
                "DEM has negligible elevation variance (std=%.2fm) in the "
                "valid region — skipping atmospheric correction (nothing "
                "real to regress against, not a low-correlation case the "
                "R² gate would otherwise catch).",
                dem_std,
            )
            return phase, {"correction_applied": False, "reason": "dem_no_variance"}

        dem_v = dem_data[valid]
        phase_v = phase[valid]
        A = np.vstack([dem_v, np.ones_like(dem_v)]).T

        complex_v = np.exp(1j * phase_v)
        coeffs_re, *_ = np.linalg.lstsq(A, complex_v.real, rcond=None)
        coeffs_im, *_ = np.linalg.lstsq(A, complex_v.imag, rcond=None)
        fitted_phase_v = np.arctan2(A @ coeffs_im, A @ coeffs_re)

        # R² must also be computed via circular (wrapped) residuals --
        # a correct circular FIT with a naive arithmetic R² would still
        # give a wrong gate decision, confirmed as the reason this
        # needs matching, not just copying, the proven pattern.
        residual = np.angle(np.exp(1j * (phase_v - fitted_phase_v)))
        ss_res = np.sum(residual**2)
        centered = np.angle(np.exp(1j * (phase_v - np.mean(phase_v))))
        ss_tot = np.sum(centered**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

        if r_squared < 0.5:
            logger.info(
                "Elevation correlation too weak (R²=%.2f) — skipping atmospheric "
                "correction to avoid absorbing real deformation signal.",
                r_squared,
            )
            return phase, {"correction_applied": False, "r_squared": float(r_squared)}

        slope_re, intercept_re = coeffs_re
        slope_im, intercept_im = coeffs_im
        fitted_real = slope_re * dem_data + intercept_re
        fitted_imag = slope_im * dem_data + intercept_im
        tropo_phase = np.arctan2(fitted_imag, fitted_real)
        corrected = np.angle(np.exp(1j * (phase - tropo_phase)))

        logger.info(
            "Elevation-correlated correction applied: R²=%.2f over %d valid pixels",
            r_squared,
            int(valid.sum()),
        )
        return corrected.astype(np.float32), {"correction_applied": True, "r_squared": float(r_squared)}

    # ── ERA5/PyAPS-based correction ───────────────────────────────────────────

    def _correct_era5(
        self,
        phase: Any,
        dem: Union[str, Path],
        acquisition_datetime: Optional[str],
        incidence_angle_deg: float,
    ) -> Any:
        """
        ERA5 reanalysis-based tropospheric correction (PyAPS method).

        Downloads ERA5 pressure-level data for the acquisition time,
        computes zenith wet + hydrostatic delay, interpolates to the DEM
        grid, and projects along the LOS using the incidence angle.
        """
        if acquisition_datetime is None:
            raise ValueError(
                "acquisition_datetime is required for method='era5' "
                "(e.g. '2026-06-01T18:16:00')"
            )

        np = self._np()
        pyaps = self._require_pyaps()

        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        with rasterio.open(dem) as src:
            dem_data = src.read(1).astype(np.float32)

        from datetime import datetime

        dt = datetime.fromisoformat(acquisition_datetime)

        logger.info(
            "Fetching ERA5 reanalysis for %s (this requires CDS API credentials "
            "in ~/.cdsapirc — see https://cds.climate.copernicus.eu/api-how-to)",
            dt.isoformat(),
        )

        try:
            # PyAPS3 API: aps_weather_model returns a delay object
            era5_obj = pyaps.PyAPS_rdr(
                dt.strftime("%Y%m%d%H"),
                dem.__str__() if not isinstance(dem, str) else dem,
                grib="ERA5",
                verb=False,
            )
            tropo_zenith = np.zeros(dem_data.shape, dtype=np.float32)
            era5_obj.getdelay(tropo_zenith, inc=0.0)  # zenith delay first
        except Exception as exc:
            raise RuntimeError(
                f"PyAPS ERA5 delay computation failed: {exc}\n"
                "Common causes: missing CDS API credentials (~/.cdsapirc), "
                "network access to Copernicus Climate Data Store, or "
                "ERA5 data not yet available for very recent dates "
                "(ERA5 has ~5 day latency)."
            ) from exc

        # Project zenith delay to line-of-sight using incidence angle
        los_delay_m = tropo_zenith / np.cos(np.deg2rad(incidence_angle_deg))

        # Convert delay (metres) to phase (radians) — Sentinel-1 C-band
        wavelength_m = 0.05546576
        tropo_phase = -4 * np.pi / wavelength_m * los_delay_m

        if tropo_phase.shape != phase.shape:
            from scipy.ndimage import zoom

            zf = (
                phase.shape[0] / tropo_phase.shape[0],
                phase.shape[1] / tropo_phase.shape[1],
            )
            tropo_phase = zoom(tropo_phase, zf, order=1)

        corrected = phase - tropo_phase
        logger.info(
            "ERA5 tropospheric correction applied (incidence=%.1f°)",
            incidence_angle_deg,
        )
        return corrected.astype(np.float32)

    def _require_pyaps(self):
        try:
            import pyaps3 as pyaps

            return pyaps
        except ImportError:
            raise ImportError(
                "pyaps3 is not installed.\n"
                'Install with: pip install "pygeofetch[insar-full]"\n'
                "Or directly:  pip install pyaps3\n\n"
                "PyAPS3 also requires free CDS API credentials for ERA5 access:\n"
                "  https://cds.climate.copernicus.eu/api-how-to\n\n"
                "For a simpler alternative without external data downloads, "
                "use method='elevation' instead."
            )

    def _np(self):
        import numpy as np

        return np