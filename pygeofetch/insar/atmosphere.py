"""
AtmosphericCorrector — tropospheric delay correction for InSAR.

Removes the tropospheric phase delay component from interferograms, one of
the dominant error sources limiting InSAR deformation accuracy (typically
2-10 cm of apparent "signal" that is actually atmospheric noise).

Three strategies are implemented:

  1. Elevation-correlated correction (native, no extra deps) —
     regresses phase against DEM elevation per-interferogram. Now supports
     polynomial fitting (to capture non-linear stratification) and spatial
     low-pass filtering (to prevent localized deformation from biasing the
     atmospheric estimate).
  2. ERA5 reanalysis-based correction (PyAPS method) — computes the
     tropospheric zenith delay from ECMWF ERA5 reanalysis data. Now includes
     automatic bounding-box padding to prevent edge artifacts from ERA5's
     coarse (~31km) native resolution.
  3. GACOS (Generic Atmospheric Correction Online Service) — uses high-
     resolution ZTD maps blending GNSS and weather models. Widely considered
     the gold standard for single-interferogram correction.

References:
  Jolivet, R., et al. (2011). Systematic InSAR tropospheric phase delay
    corrections from global meteorological reanalysis data. GRL, 38(17).
  Zhao, Y. et al. (2023). Evaluation of InSAR Tropospheric Delay Correction
    Methods in a Low-Latitude Alpine Canyon Region. Remote Sensing, 15(4), 990.
  Hu, Z., et al. (2021). InSAR tropospheric delay correction using GACOS.
    Journal of Geodesy.

Install: pip install "pygeofetch[insar]"           (native elevation correction)
         pip install "pygeofetch[insar-full]"       (+ PyAPS/ERA5 correction)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger("pygeofetch.insar.atmosphere")


def _ztd_to_los_phase(zenith_delay_m, incidence_angle_deg: float, wavelength_m: float):
    """
    Real, standard conversion: Zenith Total Delay (metres, the real,
    standard output unit from GACOS/ERA5/pyaps3) to line-of-sight
    phase delay (radians).
    """
    import numpy as np

    los_delay_m = zenith_delay_m / np.cos(np.deg2rad(incidence_angle_deg))
    return (4 * np.pi / wavelength_m) * los_delay_m


class AtmosphericCorrector:
    """
    Tropospheric delay correction for interferometric phase.

    Args:
        method: ``"elevation"`` (default), ``"era5"``, or ``"gacos"``.

    Example::

        from pygeofetch.insar import AtmosphericCorrector

        # 1. Elevation-correlated (with spatial filtering to protect deformation)
        corrector = AtmosphericCorrector(method="elevation")
        corrected_phase = corrector.correct(
            wrapped_phase, dem="dem.tif",
            poly_degree=2, spatial_filter_m=2000, profile=result.profile
        )

        # 2. ERA5-based
        corrector = AtmosphericCorrector(method="era5")
        corrected_phase = corrector.correct(
            wrapped_phase, dem="dem.tif",
            reference_datetime="2026-06-01T18:16:00",
            secondary_datetime="2026-06-13T18:16:00",
        )

        # 3. GACOS-based (requires downloaded GACOS .tif files)
        corrector = AtmosphericCorrector(method="gacos")
        corrected_phase = corrector.correct(
            wrapped_phase,
            gacos_ref="gacos_20260601.tif",
            gacos_sec="gacos_20260613.tif"
        )
    """

    def __init__(self, method: str = "elevation", cds_api_key: Optional[str] = None) -> None:
        if method not in ("elevation", "era5", "gacos"):
            raise ValueError(f"method must be 'elevation', 'era5', or 'gacos', got {method!r}")
        self._method = method
        if cds_api_key is not None:
            self._write_cdsapirc(cds_api_key)

    def _write_cdsapirc(self, api_key: str) -> None:
        """Writes ~/.cdsapirc for PyAPS3/ERA5 access."""
        cdsapirc_path = Path.home() / ".cdsapirc"
        content = f"url: https://cds.climate.copernicus.eu/api\nkey: {api_key}\n"

        if cdsapirc_path.exists():
            existing = cdsapirc_path.read_text()
            if existing.strip() == content.strip():
                logger.info("~/.cdsapirc already contains this exact key — nothing to do.")
                return
            logger.warning(
                "~/.cdsapirc already exists with different content — "
                "NOT overwriting it automatically. Remove or update it "
                "manually if you want to replace it with the key just "
                "provided, at %s", cdsapirc_path,
            )
            return

        cdsapirc_path.write_text(content)
        try:
            cdsapirc_path.chmod(0o600)
        except OSError:
            pass
        logger.info("Wrote CDS API credentials to %s", cdsapirc_path)

    def correct(
        self,
        phase: Any,
        dem: Optional[Union[str, Path]] = None,
        reference_datetime: Optional[str] = None,
        secondary_datetime: Optional[str] = None,
        incidence_angle_deg: float = 38.0,
        wavelength_m: float = 0.05546576,
        return_metadata: bool = False,
        profile: Optional[dict] = None,
        # New Elevation Parameters
        poly_degree: int = 1,
        spatial_filter_m: Optional[float] = None,
        unwrapped: bool = False,
        # New GACOS Parameters
        gacos_ref: Optional[Union[str, Path]] = None,
        gacos_sec: Optional[Union[str, Path]] = None,
    ) -> Any:
        """
        Remove the tropospheric delay component from wrapped or unwrapped phase.

        Args:
            phase:                Float32 phase array (radians).
            dem:                  DEM path (required for 'elevation' and 'era5').
            reference_datetime:   ISO datetime (required for 'era5').
            secondary_datetime:   ISO datetime (required for 'era5').
            incidence_angle_deg:  Radar incidence angle (default 38° for S1 IW).
            wavelength_m:         Radar wavelength (default 0.0554m for S1 C-band).
            profile:              Rasterio profile dict (crs, transform) for the phase
                                  array. Crucial for spatial filtering and DEM reprojection.
            poly_degree:          Polynomial degree for elevation regression (1=linear, 2=quadratic).
                                  Captures non-linear stratification. Default 1.
            spatial_filter_m:     If set, applies a Gaussian low-pass filter (in meters) to the
                                  phase before regression. Prevents localized deformation from
                                  biasing the atmospheric estimate. Highly recommended.
            unwrapped:            Set to True if the input `phase` is already unwrapped.
                                  Bypasses circular regression for faster, more robust polynomial fitting.
            gacos_ref:            Path to GACOS ZTD GeoTIFF for reference date.
            gacos_sec:            Path to GACOS ZTD GeoTIFF for secondary date.

        Returns:
            Corrected phase array (or tuple with metadata if return_metadata=True).
        """
        metadata = {"correction_applied": False, "method": self._method}

        if self._method == "era5":
            if not dem or not reference_datetime or not secondary_datetime:
                raise ValueError("ERA5 requires dem, reference_datetime, and secondary_datetime.")
            result = self._correct_era5(
                phase, dem, reference_datetime, secondary_datetime,
                incidence_angle_deg, wavelength_m
            )
            metadata["correction_applied"] = True

        elif self._method == "gacos":
            if not gacos_ref or not gacos_sec:
                raise ValueError("GACOS requires gacos_ref and gacos_sec paths.")
            result = self._correct_gacos(
                phase, gacos_ref, gacos_sec, incidence_angle_deg, wavelength_m
            )
            metadata["correction_applied"] = True

        else:  # elevation
            if not dem:
                raise ValueError("Elevation method requires a DEM path.")
            result, elev_meta = self._correct_elevation(
                phase, dem, profile=profile, poly_degree=poly_degree,
                spatial_filter_m=spatial_filter_m, unwrapped=unwrapped
            )
            metadata.update(elev_meta)

        return (result, metadata) if return_metadata else result

    # ── native elevation-correlated correction ────────────────────────────────

    def _correct_elevation(
        self, phase: Any, dem: Union[str, Path], profile: Optional[dict] = None,
        poly_degree: int = 1, spatial_filter_m: Optional[float] = None, unwrapped: bool = False
    ) -> Any:
        """
        Remove the phase component correlated with elevation.
        Now supports polynomial fitting and spatial filtering to protect
        localized deformation signals from being absorbed by the correction.
        """
        np = self._np()
        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        with rasterio.open(dem) as src:
            if profile is not None and profile.get("crs") is not None and profile.get("transform") is not None:
                from rasterio.warp import Resampling, reproject
                dem_data = np.empty(phase.shape, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1), destination=dem_data,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=profile["transform"], dst_crs=profile["crs"],
                    resampling=Resampling.bilinear, src_nodata=src.nodata, dst_nodata=np.nan,
                )
            else:
                dem_data = src.read(1).astype(np.float32)
                if dem_data.shape != phase.shape:
                    logger.warning("No profile supplied. Falling back to shape-ratio DEM resample.")
                    from scipy.ndimage import zoom
                    zf = (phase.shape[0] / dem_data.shape[0], phase.shape[1] / dem_data.shape[1])
                    dem_data = zoom(dem_data, zf, order=1)

        valid = np.isfinite(phase) & np.isfinite(dem_data) & (dem_data > -500)
        if valid.sum() < 100:
            logger.warning("Insufficient valid pixels for elevation correction.")
            return phase, {"correction_applied": False, "reason": "insufficient_valid_pixels"}

        dem_v = dem_data[valid]
        phase_v = phase[valid]

        # ── Spatial Filtering (Crucial for protecting deformation) ────────────
        # If a deformation signal (e.g. subsidence) sits on a mountain, a global
        # regression will mistake it for atmospheric stratification. Low-pass
        # filtering the phase isolates the broad atmospheric trend.
        if spatial_filter_m and profile and profile.get("transform"):
            from scipy.ndimage import gaussian_filter
            pixel_size = abs(profile["transform"].a)
            sigma_pixels = spatial_filter_m / pixel_size

            # Create a full-size array for filtering, fill with NaNs where invalid
            phase_full = np.full(phase.shape, np.nan, dtype=np.float32)
            phase_full[valid] = phase_v

            # Gaussian filter ignores NaNs if we handle them, but standard gaussian_filter
            # propagates NaNs. A quick workaround is to replace NaNs with 0, filter,
            # and then we only use the filtered values at the valid pixels.
            phase_temp = np.nan_to_num(phase_full, nan=0.0)
            phase_smooth_full = gaussian_filter(phase_temp, sigma=sigma_pixels)

            # Update phase_v with the smoothed long-wavelength phase
            phase_v = phase_smooth_full[valid]
            logger.info(f"Applied spatial low-pass filter ({spatial_filter_m}m) before regression.")

        # ── Regression ────────────────────────────────────────────────────────
        if unwrapped:
            # Unwrapped phase allows standard, robust polynomial least-squares
            if poly_degree > 3:
                logger.warning("poly_degree > 3 is prone to overfitting. Capping at 3.")
                poly_degree = 3
            coeffs = np.polyfit(dem_v, phase_v, poly_degree)
            tropo_phase = np.polyval(coeffs, dem_data)

            # Calculate R²
            ss_res = np.sum((phase_v - np.polyval(coeffs, dem_v))**2)
            ss_tot = np.sum((phase_v - np.mean(phase_v))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

        else:
            # Wrapped phase requires circular regression
            if poly_degree > 1:
                logger.warning(
                    "Circular regression for wrapped phase is only stable for degree 1. "
                    "Falling back to linear. Pass unwrapped=True for polynomial fits."
                )
                poly_degree = 1

            dem_std = float(np.std(dem_v))
            if dem_std < 1.0:
                return phase, {"correction_applied": False, "reason": "dem_no_variance"}

            elev_range = float(np.ptp(dem_v))
            max_slope = (25.0 * 2 * np.pi) / elev_range if elev_range >= 1.0 else 0.5

            rng = np.random.default_rng(0)
            n_valid = len(dem_v)
            n_search = 20000
            if n_valid > n_search:
                search_idx = rng.choice(n_valid, size=n_search, replace=False)
                dem_search, phase_search = dem_v[search_idx], phase_v[search_idx]
            else:
                dem_search, phase_search = dem_v, phase_v

            def _flatness(candidate_slopes):
                phase_matrix = phase_search[None, :] - candidate_slopes[:, None] * dem_search[None, :]
                return np.abs(np.mean(np.exp(1j * phase_matrix), axis=1))

            coarse = np.linspace(-max_slope, max_slope, 400)
            best_slope = float(coarse[np.argmax(_flatness(coarse))])
            fine_half_width = coarse[1] - coarse[0]
            fine = np.linspace(best_slope - fine_half_width, best_slope + fine_half_width, 400)
            best_slope = float(fine[np.argmax(_flatness(fine))])

            residual_v = np.angle(np.exp(1j * (phase_v - best_slope * dem_v)))
            intercept = float(np.angle(np.mean(np.exp(1j * residual_v))))

            tropo_phase = np.angle(np.exp(1j * (best_slope * dem_data + intercept)))

            # R² calculation
            fitted_phase_v = np.angle(np.exp(1j * (best_slope * dem_v + intercept)))
            residual = np.angle(np.exp(1j * (phase_v - fitted_phase_v)))
            ss_res = np.sum(residual**2)
            circular_mean_phase = np.angle(np.mean(np.exp(1j * phase_v)))
            centered = np.angle(np.exp(1j * (phase_v - circular_mean_phase)))
            ss_tot = np.sum(centered**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

        if r_squared < 0.3:  # Lowered threshold slightly as spatial filtering changes variance
            logger.info(
                "Elevation correlation too weak (R²=%.2f) — skipping atmospheric correction.",
                r_squared,
            )
            return phase, {"correction_applied": False, "r_squared": float(r_squared)}

        corrected = np.angle(np.exp(1j * (phase - tropo_phase))) if not unwrapped else (phase - tropo_phase)

        logger.info(
            "Elevation-correlated correction applied: R²=%.2f (degree=%d) over %d valid pixels",
            r_squared, poly_degree, int(valid.sum()),
        )
        return corrected.astype(np.float32), {"correction_applied": True, "r_squared": float(r_squared)}

    # ── ERA5/PyAPS-based correction ───────────────────────────────────────────

    def _correct_era5(
        self, phase: Any, dem: Union[str, Path], reference_datetime: str,
        secondary_datetime: str, incidence_angle_deg: float, wavelength_m: float
    ) -> Any:
        """ERA5 reanalysis-based tropospheric correction (PyAPS method)."""
        np = self._np()
        pyaps = self._require_pyaps()

        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        with rasterio.open(dem) as src:
            dem_data = src.read(1).astype(np.float32)
            dem_transform = src.transform
            dem_height, dem_width = src.height, src.width
            dem_bounds = src.bounds

        cols, rows = np.meshgrid(np.arange(dem_width), np.arange(dem_height))

        # rasterio.transform.xy returns (xs, ys) corresponding to (lons, lats)
        xs, ys = rasterio.transform.xy(dem_transform, rows, cols)
        lon_grid = np.array(xs, dtype=np.float32).reshape(dem_height, dem_width)
        lat_grid = np.array(ys, dtype=np.float32).reshape(dem_height, dem_width)

        # FIX: Pad bounding box by 0.5 degrees (~50km). ERA5 is coarse (0.25 deg).
        # Without padding, PyAPS interpolates from far outside the scene, causing
        # severe edge artifacts and "Longitude array size mismatch" errors.
        pad = 0.5
        snwe = [
            dem_bounds.bottom - pad,
            dem_bounds.top + pad,
            dem_bounds.left - pad,
            dem_bounds.right + pad
        ]

        import tempfile
        grib_dir = Path(tempfile.gettempdir()) / "pygeofetch_era5_grib"
        grib_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        dt_ref = datetime.fromisoformat(reference_datetime)
        dt_sec = datetime.fromisoformat(secondary_datetime)

        logger.info("Fetching ERA5 reanalysis for %s and %s...", dt_ref.isoformat(), dt_sec.isoformat())

        def _los_phase_delay_for(dt):
            grib_files = pyaps.ECMWFdload(
                [dt.strftime("%Y%m%d")], dt.strftime("%H"), str(grib_dir),
                model="ERA5", snwe=snwe,
            )
            grib_path = grib_files[0] if isinstance(grib_files, (list, tuple)) else grib_files

            aps_obj = pyaps.PyAPS(
                grib_path, dem_data, lat_grid, lon_grid,
                inc=incidence_angle_deg, grib="ERA5", verb=False,
            )
            phase_out = np.zeros(dem_data.shape, dtype=np.float32)
            aps_obj.getdelay(phase_out, wvl=wavelength_m)
            return phase_out

        try:
            phase_ref = _los_phase_delay_for(dt_ref)
            phase_sec = _los_phase_delay_for(dt_sec)
        except Exception as exc:
            raise RuntimeError(f"PyAPS ERA5 delay computation failed: {exc}") from exc

        atmo_phase = phase_sec - phase_ref

        if atmo_phase.shape != phase.shape:
            from scipy.ndimage import zoom
            zf = (phase.shape[0] / atmo_phase.shape[0], phase.shape[1] / atmo_phase.shape[1])
            atmo_phase = zoom(atmo_phase, zf, order=1)

        corrected = phase - atmo_phase
        logger.info("ERA5 tropospheric correction applied (incidence=%.1f°).", incidence_angle_deg)
        return corrected.astype(np.float32)

    # ── GACOS-based correction ────────────────────────────────────────────────

    def _correct_gacos(
        self, phase: Any, gacos_ref: Union[str, Path], gacos_sec: Union[str, Path],
        incidence_angle_deg: float, wavelength_m: float
    ) -> Any:
        """
        GACOS (Generic Atmospheric Correction Online Service) correction.
        Expects GeoTIFFs containing Zenith Total Delay (ZTD) in meters.
        """
        np = self._np()
        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        logger.info("Loading GACOS ZTD maps: %s and %s", gacos_ref, gacos_sec)

        with rasterio.open(gacos_ref) as src:
            ztd_ref = src.read(1).astype(np.float32)
        with rasterio.open(gacos_sec) as src:
            ztd_sec = src.read(1).astype(np.float32)

        # Convert ZTD (meters) to LOS Phase (radians)
        phase_ref = _ztd_to_los_phase(ztd_ref, incidence_angle_deg, wavelength_m)
        phase_sec = _ztd_to_los_phase(ztd_sec, incidence_angle_deg, wavelength_m)

        atmo_phase = phase_sec - phase_ref

        if atmo_phase.shape != phase.shape:
            from scipy.ndimage import zoom
            zf = (phase.shape[0] / atmo_phase.shape[0], phase.shape[1] / atmo_phase.shape[1])
            atmo_phase = zoom(atmo_phase, zf, order=1)

        corrected = phase - atmo_phase
        logger.info("GACOS tropospheric correction applied.")
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
