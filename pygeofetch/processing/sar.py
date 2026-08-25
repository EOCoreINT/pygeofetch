"""
SARProcessor — SAR-specific processing operations.
Speckle filtering, radiometric calibration, flood mapping, coherence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from pygeofetch.processing.base import (
    ProcessingResult,
    _require_numpy,
    _require_rasterio,
    _require_scipy,
    _resolve_output,
    _safe_read_band,
    _safe_write_band,
    _timed,
)

logger = logging.getLogger(__name__)


class SARProcessor:
    """
    SAR (Synthetic Aperture Radar) processing engine.

    All read operations use the block-by-block fallback so they work on
    tiled/COG/compressed GeoTIFFs without crashing.

    Example::

        from pygeofetch import PyGeoFetch
        client = PyGeoFetch()
        result = client.sar.despeckle("sentinel1_vv.tif", filter="lee")
        result = client.sar.flood_map("sentinel1.tif", threshold=-15.0)
    """

    @staticmethod
    def _read(path: str | Path, ref_shape=None):
        return _safe_read_band(path, band=1, out_shape=ref_shape)

    # ── Despeckle ────────────────────────────────────────────────────────

    @_timed
    def despeckle(
        self,
        input: str | Path,
        filter: str = "lee",
        window: int = 5,
        output: str | None = None,
        num_looks: int = 1,
    ) -> ProcessingResult:
        """
        SAR speckle filtering.

        Args:
            input:     SAR GeoTIFF (linear amplitude or power).
            filter:    ``"lee"`` (default), ``"enhanced_lee"``, ``"frost"``,
                       ``"gamma"``, ``"boxcar"``.
            window:    Filter window size (must be odd, default 5).
            output:    Output path.
            num_looks: Number of looks (affects Lee/Gamma threshold).

        Example::

            result = client.sar.despeckle("s1c_vv.tif", filter="enhanced_lee")
        """
        ndimage = _require_scipy()
        np = _require_numpy()

        inp = Path(input)
        out_path = _resolve_output(inp, output, f"despeckle_{filter}")

        data, profile, nodata = self._read(inp)

        valid = np.isfinite(data) & (
            True if nodata is None else (data != float(nodata))
        )
        work = np.where(valid, data, 0.0).astype(np.float64)

        if filter == "boxcar":
            result = ndimage.uniform_filter(work, size=window)

        elif filter == "lee":
            mean = ndimage.uniform_filter(work, size=window)
            sq_mean = ndimage.uniform_filter(work**2, size=window)
            var = np.maximum(sq_mean - mean**2, 0)
            weight = var / (var + mean**2 / (num_looks + 1e-10) + 1e-10)
            result = mean + weight * (work - mean)

        elif filter == "enhanced_lee":
            mean = ndimage.uniform_filter(work, size=window)
            sq_mean = ndimage.uniform_filter(work**2, size=window)
            var = np.maximum(sq_mean - mean**2, 0)
            cv_local = np.sqrt(var) / (mean + 1e-10)
            cv_thresh = 1.0 / np.sqrt(num_looks + 1e-10)
            weight = np.where(
                cv_local <= cv_thresh,
                0.0,
                np.where(
                    cv_local > 3 * cv_thresh,
                    1.0,
                    (cv_local - cv_thresh) / (2 * cv_thresh + 1e-10),
                ),
            )
            result = (1 - weight) * mean + weight * work

        elif filter == "frost":
            mean = ndimage.uniform_filter(work, size=window)
            sq_mean = ndimage.uniform_filter(work**2, size=window)
            var = np.maximum(sq_mean - mean**2, 0)
            cv = np.sqrt(var) / (mean + 1e-10)
            sigma = window / (2 * (1 + cv.mean()) + 1e-10)
            smoothed = ndimage.gaussian_filter(work, sigma=max(sigma, 0.1))
            alpha = np.exp(-cv * window)
            result = alpha * work + (1 - alpha) * smoothed

        elif filter == "gamma":
            mean = ndimage.uniform_filter(work, size=window)
            sq_mean = ndimage.uniform_filter(work**2, size=window)
            var = np.maximum(sq_mean - mean**2, 0)
            b = (var - mean**2 / (num_looks + 1e-10)) / (
                var + mean**2 / (num_looks + 1e-10) + 1e-10
            )
            b = np.clip(b, 0, 1)
            result = mean + b * (work - mean)

        else:
            msg = (
                f"Unknown SAR filter: {filter!r}. "
                "Choose from: lee, enhanced_lee, frost, gamma, boxcar"
            )
            raise ValueError(msg)

        nd_fill = float(nodata) if nodata is not None else -9999.0
        result = np.where(valid, result, nd_fill).astype(np.float32)
        _safe_write_band(result, profile, out_path, nodata=nd_fill)

        logger.info("SAR despeckle (%s, window=%d) → %s", filter, window, out_path.name)
        return ProcessingResult(
            success=True,
            operation=f"despeckle:{filter}",
            input_path=inp,
            output_path=out_path,
            metadata={"filter": filter, "window": window, "num_looks": num_looks},
        )

    # ── Calibrate ────────────────────────────────────────────────────────

    @_timed
    def calibrate(
        self,
        input: str | Path,
        output_type: str = "sigma0",
        output: str | None = None,
        in_db: bool = True,
    ) -> ProcessingResult:
        """
        Radiometric calibration — convert SAR DN to backscatter coefficient.

        Args:
            input:       SAR raster in DN (digital numbers).
            output_type: ``"sigma0"`` (default), ``"gamma0"``, ``"beta0"``.
            output:      Output path.
            in_db:       Convert result to dB scale (default True).

        Example::

            result = client.sar.calibrate("sentinel1_dn.tif",
                                           output_type="sigma0", in_db=True)
        """
        np = _require_numpy()

        inp = Path(input)
        out_path = _resolve_output(inp, output, f"cal_{output_type}")

        data, profile, nodata = self._read(inp)
        data = data.astype(np.float64)

        # Sentinel-1 GRD calibration: sigma0 = DN² / A²
        # Real calibration constant A comes from the annotation XML;
        # A=1 is the identity (data already in linear power units).
        A = 1.0
        calibrated = (data**2) / (A**2)

        if output_type == "gamma0":
            # Simplified terrain-flattening via nominal incidence angle
            calibrated *= np.cos(np.deg2rad(38.0))
        elif output_type == "beta0":
            # beta0 = sigma0 / sin(incidence_angle)
            calibrated /= np.sin(np.deg2rad(38.0)) + 1e-10

        if in_db:
            with np.errstate(divide="ignore", invalid="ignore"):
                calibrated = 10 * np.log10(np.maximum(calibrated, 1e-10))

        nd_fill = float(nodata) if nodata is not None else -9999.0
        if nodata is not None:
            calibrated = np.where(data == float(nodata), nd_fill, calibrated)

        _safe_write_band(
            calibrated.astype(np.float32), profile, out_path, nodata=nd_fill
        )
        logger.info(
            "SAR calibration (%s, dB=%s) → %s", output_type, in_db, out_path.name
        )
        return ProcessingResult(
            success=True,
            operation=f"calibrate:{output_type}",
            input_path=inp,
            output_path=out_path,
            metadata={"output_type": output_type, "in_db": in_db},
        )

    # ── Flood Map ────────────────────────────────────────────────────────

    @_timed
    def flood_map(
        self,
        input: str | Path,
        threshold: float = -15.0,
        output: str | None = None,
        reference: str | Path | None = None,
        detect_direction: str = "decrease",
    ) -> ProcessingResult:
        """
        Flood mapping from SAR backscatter via simple threshold or change detection.

        Args:
            input:     SAR raster in dB (VV or VH polarisation).
            threshold: Backscatter below this = water (typical -15 to -20 dB).
                     Also sets the change-detection sensitivity (see
                     detect_direction) as abs(threshold * 0.5).
            output:    Output binary mask (1=water, 0=land, 255=nodata).
            reference: Optional pre-event reference for change-based detection.
            detect_direction: For change detection only. "decrease" (default)
                     flags pixels where backscatter DROPPED — the correct
                     signature for open water (smooth surfaces are
                     near-specular reflectors). "increase" flags pixels
                     where backscatter ROSE — the signature of flooded
                     urban/built-up areas, where water at the base of
                     buildings creates a double-bounce reflection
                     (ground-wall-sensor) stronger than dry ground alone.
                     "both" flags either direction — the most robust
                     choice when an AOI mixes open water and dense urban
                     flooding (e.g. Accra), since a one-directional
                     threshold structurally cannot detect the other
                     pattern at all, not just detect it poorly.

        Example::

            result = client.sar.flood_map("post_event.tif",
                                           reference="pre_event.tif",
                                           detect_direction="both")
        """
        np = _require_numpy()
        rasterio = _require_rasterio()

        if detect_direction not in ("decrease", "increase", "both"):
            raise ValueError(
                f"detect_direction must be 'decrease', 'increase', or "
                f"'both', got {detect_direction!r}"
            )

        inp = Path(input)
        out_path = _resolve_output(inp, output, "flood_map")

        data, profile, nodata = self._read(inp)
        valid = np.isfinite(data)
        if nodata is not None:
            valid &= data != float(nodata)

        if reference is not None:
            ref_d, _, _ = self._read(Path(reference), ref_shape=data.shape)
            change = ref_d - data  # positive = backscatter dropped (open water)
            change_threshold = abs(threshold * 0.5)
            if detect_direction == "decrease":
                water_mask = (change > change_threshold) & valid
            elif detect_direction == "increase":
                water_mask = (change < -change_threshold) & valid
            else:  # "both"
                water_mask = (np.abs(change) > change_threshold) & valid
        else:
            water_mask = (data < threshold) & valid

        result = water_mask.astype(np.uint8)
        water_pct = 100 * np.sum(water_mask) / (np.sum(valid) + 1e-10)

        # Write uint8 binary mask — use a clean profile, not float profile
        clean = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "height": profile.get("height", data.shape[0]),
            "width": profile.get("width", data.shape[1]),
            "crs": profile.get("crs"),
            "transform": profile.get("transform"),
            "nodata": 255,
            "compress": "lzw",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **clean) as dst:
            dst.write(result[None, :, :])

        logger.info("Flood map: %.1f%% water → %s", water_pct, out_path.name)
        metadata: Dict[str, Any] = {
            "threshold_db": threshold,
            "water_pct": round(float(water_pct), 2),
        }
        if reference is not None:
            metadata["detect_direction"] = detect_direction
            decrease_pct = (
                100
                * np.sum((change > change_threshold) & valid)
                / (np.sum(valid) + 1e-10)
            )
            increase_pct = (
                100
                * np.sum((change < -change_threshold) & valid)
                / (np.sum(valid) + 1e-10)
            )
            metadata["decrease_pct"] = round(float(decrease_pct), 2)
            metadata["increase_pct"] = round(float(increase_pct), 2)

        return ProcessingResult(
            success=True,
            operation="flood_map",
            input_path=inp,
            output_path=out_path,
            metadata=metadata,
        )

    # ── Coherence ────────────────────────────────────────────────────────

    @_timed
    def coherence(
        self,
        image1: str | Path,
        image2: str | Path,
        window: int = 7,
        output: str | None = None,
    ) -> ProcessingResult:
        """
        Interferometric coherence between two co-registered SLC images.
        Coherence = |<s1·s2*>| / sqrt(<|s1|²><|s2|²>)  — range 0 to 1.
        High coherence = stable surface; low = change/decorrelation.

        Args:
            image1, image2: Co-registered complex SLC rasters (float or complex64).
            window:         Estimation window size (default 7).
            output:         Output coherence raster.

        Example::

            result = client.sar.coherence("slc_20260601.tif", "slc_20260613.tif")
        """
        ndimage = _require_scipy()
        np = _require_numpy()
        rasterio = _require_rasterio()

        inp1, inp2 = Path(image1), Path(image2)
        out_path = _resolve_output(inp1, output, "coherence")

        # Read — complex GeoTIFFs report dtype complex64
        with rasterio.open(inp1) as src:
            profile = src.profile.copy()
            if src.dtypes[0] == "complex64":
                s1 = (
                    _safe_read_band(inp1, band=1)[0]
                    .view(np.complex64)
                    .reshape(src.height, src.width)
                )
            else:
                s1_r, _, _ = _safe_read_band(inp1, band=1)
                s1 = s1_r.astype(np.float32) + 0j

        with rasterio.open(inp2) as src2:
            if src2.dtypes[0] == "complex64":
                s2 = (
                    _safe_read_band(inp2, band=1)[0]
                    .view(np.complex64)
                    .reshape(src2.height, src2.width)
                )
            else:
                # Resample to s1 shape if needed
                s2_r, _, _ = _safe_read_band(
                    inp2, band=1, out_shape=s1.shape if s1.ndim == 2 else None
                )
                s2 = s2_r.astype(np.float32) + 0j

        # Interferogram
        inter = s1 * np.conj(s2)
        num = np.abs(
            ndimage.uniform_filter(inter.real, size=window)
            + 1j * ndimage.uniform_filter(inter.imag, size=window)
        )
        denom = np.sqrt(
            ndimage.uniform_filter(np.abs(s1) ** 2, size=window)
            * ndimage.uniform_filter(np.abs(s2) ** 2, size=window)
            + 1e-10
        )
        coh = np.clip(num / denom, 0.0, 1.0).astype(np.float32)

        _safe_write_band(coh, profile, out_path, nodata=-1.0)
        mean_coh = float(coh[coh >= 0].mean()) if (coh >= 0).any() else 0.0
        logger.info("Coherence mean=%.3f → %s", mean_coh, out_path.name)
        return ProcessingResult(
            success=True,
            operation="coherence",
            output_path=out_path,
            metadata={"mean_coherence": round(mean_coh, 4), "window": window},
        )
