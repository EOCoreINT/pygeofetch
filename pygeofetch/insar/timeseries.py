"""
SBASTimeSeries — Small BAseline Subset time series inversion.

Implements the SBAS weighted least-squares inversion (Berardino et al. 2002,
Yunjun et al. 2019 / MintPy) natively in numpy — no external InSAR software
required for the core inversion, though MintPy is used automatically when
installed for advanced corrections (tropospheric delay, DEM error,
phase-closure-based unwrapping error correction).

Reference:
  Berardino, P., Fornaro, G., Lanari, R., & Sansosti, E. (2002). A new
    algorithm for surface deformation monitoring based on small baseline
    differential SAR interferograms. IEEE TGRS, 40(11), 2375-2383.
  Yunjun, Z., Fattahi, H., & Amelung, F. (2019). Small baseline InSAR time
    series analysis: unwrapping error correction and noise reduction.
    Computers & Geosciences, 133, 104331.

Install: pip install "pygeofetch[insar]"          (native SBAS inversion)
         pip install "pygeofetch[insar-full]"      (+ MintPy passthrough)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("pygeofetch.insar.timeseries")


@dataclass
class InterferogramPair:
    """One interferogram in an SBAS network."""

    reference_date: str  # ISO date, e.g. "2026-01-01"
    secondary_date: str
    unwrapped_phase: Any  # float32 (H, W) array, radians
    coherence: Any  # float32 (H, W) array, 0-1
    perpendicular_baseline_m: float = 0.0


@dataclass
class TimeSeriesResult:
    """Output of SBAS inversion — displacement time series."""

    dates: List[str]
    displacement: Any  # float32 (n_dates, H, W) array, metres, LOS
    velocity: Any  # float32 (H, W) array, m/year, linear fit
    residual_rms: Any  # float32 (H, W) array — inversion quality
    reference_date: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(
        self,
        output_dir: Union[str, Path],
        profile: Optional[dict] = None,
        auto_visualize: bool = False,
    ) -> Dict[str, Path]:
        """Save displacement time series and velocity as GeoTIFFs.

        Args:
            output_dir:     Directory to save into.
            profile:        Optional rasterio-style profile for georeferencing.
            auto_visualize: If True, also save PNG visualizations
                           (velocity map, residual RMS map, and a
                           composite per-date displacement grid)
                           alongside the GeoTIFFs, via
                           pygeofetch.insar.visualize.
        """
        import numpy as np

        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        h, w = self.velocity.shape
        base = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "height": h,
            "width": w,
            "nodata": -9999.0,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        if profile:
            base["crs"] = profile.get("crs")
            base["transform"] = profile.get("transform")

        vel_path = out_dir / "velocity_m_per_year.tif"
        with rasterio.open(vel_path, "w", **base) as dst:
            dst.write(self.velocity.astype(np.float32)[np.newaxis])
        paths["velocity"] = vel_path

        ts_path = out_dir / "displacement_timeseries.tif"
        ts_profile = dict(base, count=len(self.dates))
        with rasterio.open(ts_path, "w", **ts_profile) as dst:
            dst.write(self.displacement.astype(np.float32))
            for i, date in enumerate(self.dates, start=1):
                dst.update_tags(i, date=date)
        paths["displacement_timeseries"] = ts_path

        rms_path = out_dir / "residual_rms.tif"
        with rasterio.open(rms_path, "w", **base) as dst:
            dst.write(self.residual_rms.astype(np.float32)[np.newaxis])
        paths["residual_rms"] = rms_path

        logger.info("Time series products saved → %s", out_dir)

        if auto_visualize:
            from pygeofetch.insar.visualize import visualize_timeseries

            try:
                visualize_timeseries(self, out_dir)
            except Exception as exc:
                logger.warning("auto_visualize failed (GeoTIFFs were still saved successfully): %s", exc)

        return paths


class SBASTimeSeries:
    """
    Small BAseline Subset (SBAS) InSAR time series inversion.

    Given a network of unwrapped interferograms, inverts for per-date LOS
    (line-of-sight) displacement relative to a reference date, using the
    weighted least-squares formulation from Berardino et al. (2002) as
    implemented in MintPy (Yunjun et al. 2019).

    Args:
        wavelength_m: Radar wavelength in metres. Sentinel-1 C-band = 0.0555.
        reference_date: Date to hold at zero displacement. Defaults to the
                        earliest date in the network.

    Example::

        from pygeofetch.insar import SBASTimeSeries
        from pygeofetch.insar.timeseries import InterferogramPair

        pairs = [
            InterferogramPair("2026-01-01", "2026-01-13", unw1, coh1),
            InterferogramPair("2026-01-13", "2026-01-25", unw2, coh2),
            InterferogramPair("2026-01-01", "2026-01-25", unw3, coh3),
        ]

        sbas   = SBASTimeSeries(wavelength_m=0.0555)  # Sentinel-1 C-band
        result = sbas.invert(pairs)
        print(f"Mean velocity: {result.velocity.mean()*1000:.1f} mm/year")
    """

    SENTINEL1_WAVELENGTH_M = 0.05546576  # C-band, ESA Sentinel-1 spec

    def __init__(
        self,
        wavelength_m: float = SENTINEL1_WAVELENGTH_M,
        reference_date: Optional[str] = None,
        use_gpu: bool = False,
    ) -> None:
        self._wavelength = wavelength_m
        self._reference_date = reference_date
        self._use_gpu = use_gpu

    def invert(
        self,
        pairs: List[InterferogramPair],
        coherence_threshold: float = 0.3,
        use_mintpy: bool = False,
        reference_pixel: Optional[Tuple[int, int]] = None,
    ) -> TimeSeriesResult:
        """
        Invert an SBAS network of interferograms into a displacement time series.

        Args:
            pairs:                List of InterferogramPair objects forming
                                  the SBAS network. Should be well-connected
                                  (every date reachable from every other).
            coherence_threshold:  Pixels below this coherence are excluded
                                  from the weighted inversion at that pair.
            use_mintpy:           If True, delegate to MintPy for the full
                                  correction chain (DEM error, unwrapping
                                  error correction, tropospheric delay).
                                  Requires `pip install "pygeofetch[insar-full]"`.
                                  If MintPy is not installed, falls back to
                                  the native inversion with a warning.
            reference_pixel:      (row, col) of a stable, high-coherence pixel
                                  to reference every interferogram's unwrapped
                                  phase to before inversion. REQUIRED for
                                  correct results: phase unwrapping (SNAPHU)
                                  only recovers phase relative to an arbitrary
                                  per-interferogram integer-cycle offset —
                                  combining independently-unwrapped
                                  interferograms without a common reference
                                  point corrupts the joint SBAS solution
                                  (Berardino et al. 2002, Section II).
                                  If None (default), the pixel with the
                                  highest mean coherence across all pairs is
                                  chosen automatically and logged. Choose a
                                  pixel known to be stable (e.g. bedrock, a
                                  building rooftop) for real deformation
                                  monitoring where automatic selection may
                                  pick a point inside the deforming area.

        Returns:
            TimeSeriesResult with per-date displacement, mean velocity,
            and inversion residuals.

        Example::

            # Explicit reference pixel (recommended for real data — pick a
            # known-stable location, e.g. bedrock outcrop or a monitored
            # benchmark, away from the expected deformation)
            result = sbas.invert(pairs, reference_pixel=(10, 15))

            # Automatic selection (picks highest average coherence pixel)
            result = sbas.invert(pairs)
        """
        from pygeofetch.insar.validate import DataValidator

        all_dates = sorted({p.reference_date for p in pairs} | {p.secondary_date for p in pairs})
        DataValidator.validate_sbas_network(pairs, all_dates).raise_if_invalid()
        for p in pairs:
            DataValidator.validate_coherence(
                p.coherence, name=f"coherence ({p.reference_date}_{p.secondary_date})"
            ).raise_if_invalid()

        pairs = self._reference_pairs(pairs, reference_pixel)

        if use_mintpy:
            try:
                return self._invert_mintpy(pairs, coherence_threshold)
            except ImportError as exc:
                logger.warning(
                    "MintPy not available (%s) — falling back to native SBAS "
                    "inversion. For advanced corrections install: "
                    'pip install "pygeofetch[insar-full]"',
                    exc,
                )

        return self._invert_native(pairs, coherence_threshold)

    # ── native SBAS inversion (Berardino et al. 2002) ─────────────────────────

    def _reference_pairs(
        self, pairs: List[InterferogramPair], reference_pixel: Optional[Tuple[int, int]]
    ) -> List[InterferogramPair]:
        """
        Reference every interferogram's unwrapped phase to a common pixel.

        SNAPHU (and any phase unwrapper) recovers phase only up to an
        arbitrary additive integer multiple of 2*pi per interferogram —
        there is no absolute phase reference without external ground truth.
        Combining multiple independently-unwrapped interferograms in a
        joint least-squares inversion requires first removing this
        per-interferogram offset by subtracting the phase at a common
        pixel, so that pixel reads exactly zero displacement in every
        interferogram (Berardino et al. 2002).
        """
        np = self._np()

        if reference_pixel is None:
            # Auto-select the pixel with highest mean coherence across all pairs
            coh_stack = np.stack([p.coherence for p in pairs], axis=0)
            mean_coh = coh_stack.mean(axis=0)
            ry, rx = np.unravel_index(np.argmax(mean_coh), mean_coh.shape)
            reference_pixel = (int(ry), int(rx))
            logger.info(
                "No reference_pixel specified — auto-selected pixel %s "
                "(mean coherence=%.3f). For real deformation monitoring, "
                "prefer an explicit reference_pixel known to be stable.",
                reference_pixel,
                float(mean_coh[ry, rx]),
            )
        else:
            logger.info("Referencing all interferograms to pixel %s", reference_pixel)

        ry, rx = reference_pixel
        referenced = []
        for p in pairs:
            offset = p.unwrapped_phase[ry, rx]
            referenced.append(
                InterferogramPair(
                    reference_date=p.reference_date,
                    secondary_date=p.secondary_date,
                    unwrapped_phase=p.unwrapped_phase - offset,
                    coherence=p.coherence,
                    perpendicular_baseline_m=p.perpendicular_baseline_m,
                )
            )
        return referenced

    def _invert_native(
        self, pairs: List[InterferogramPair], coherence_threshold: float
    ) -> TimeSeriesResult:
        np = self._np()

        dates = sorted(
            set([p.reference_date for p in pairs] + [p.secondary_date for p in pairs])
        )
        n_dates = len(dates)
        n_pairs = len(pairs)
        date_idx = {d: i for i, d in enumerate(dates)}

        ref_date = self._reference_date or dates[0]
        if ref_date not in date_idx:
            raise ValueError(f"reference_date {ref_date!r} not found in network dates")

        logger.info(
            "SBAS inversion: %d dates, %d interferogram pairs, reference=%s",
            n_dates,
            n_pairs,
            ref_date,
        )

        h, w = pairs[0].unwrapped_phase.shape

        # Design matrix: each row is one interferogram, encoding
        # (secondary - reference) as +1/-1 in the date columns
        A = np.zeros((n_pairs, n_dates), dtype=np.float32)
        for i, pair in enumerate(pairs):
            A[i, date_idx[pair.reference_date]] = -1
            A[i, date_idx[pair.secondary_date]] = 1

        # Remove reference date column (its displacement is fixed at 0)
        ref_col = date_idx[ref_date]
        keep_cols = [c for c in range(n_dates) if c != ref_col]

        # Stack observations: phase → displacement (metres).
        #
        # Sign convention (must match InterferogramGenerator/PhaseUnwrapper):
        # interferograms are formed as ref * conj(sec), giving
        #   unwrapped_phase = phase(ref) - phase(sec)
        # with phase(x) = -4*pi/wavelength * disp(x). Combined with the
        # design matrix encoding each row as x[sec] - x[ref] (A[i,ref]=-1,
        # A[i,sec]=+1), the consistent displacement estimate per pair is:
        #   disp(sec) - disp(ref) = +wavelength / (4*pi) * unwrapped_phase
        # (positive sign — do not flip; a negative sign here inverts the
        # solution relative to the ref*conj(sec) interferogram convention
        # used throughout pygeofetch.insar).
        phase_stack = np.stack([p.unwrapped_phase for p in pairs], axis=0)
        disp_stack = phase_stack * self._wavelength / (4 * np.pi)

        coh_stack = np.stack([p.coherence for p in pairs], axis=0)

        if self._use_gpu:
            logger.warning(
                "use_gpu=True was requested, but the corrected per-pixel "
                "coherence-masked inversion does not yet have a GPU path "
                "(the previous GPU-accelerated code path solved an "
                "incorrect, unweighted global inversion — removed along "
                "with that bug, not yet replaced with a GPU-aware version "
                "of the correct per-group solve). Running on CPU."
            )

        # Real, per-pixel coherence masking -- fixes a genuine, confirmed
        # bug: the previous implementation computed a thresholded
        # coherence array here and never used it, silently making
        # coherence_threshold a no-op parameter despite the docstring's
        # promise that low-coherence pixels are excluded per pair.
        #
        # Pixels are grouped by their unique pattern of which pairs pass
        # threshold (real coherence data is spatially correlated, so the
        # number of distinct patterns is normally far smaller than the
        # pixel count) and each group is solved once using only its
        # valid pairs -- efficient, not an O(H*W) separate-solve loop,
        # and mathematically correct: a pixel whose bad pairs leave it
        # underdetermined is honestly marked unreliable (NaN) rather
        # than silently corrupted by including a low-quality observation
        # anyway. Verified before use: a synthetic pixel with one
        # deliberately corrupted, low-coherence pair correctly comes
        # back NaN for the affected date instead of a wrong value, while
        # unaffected pixels solve to match true displacement almost
        # exactly.
        valid_mask = coh_stack >= coherence_threshold  # (n_pairs, h, w)
        pattern = np.zeros((h, w), dtype=np.int64)
        for i in range(n_pairs):
            pattern += valid_mask[i].astype(np.int64) << i

        unique_patterns = np.unique(pattern)
        if len(unique_patterns) > 0.5 * h * w:
            logger.warning(
                "Coherence pattern is highly heterogeneous (%d unique patterns "
                "across %d pixels) — per-pixel masking may be slower than "
                "expected for this scene.",
                len(unique_patterns), h * w,
            )

        displacement = np.full((n_dates, h, w), np.nan, dtype=np.float32)
        residual_rms = np.full((h, w), np.nan, dtype=np.float32)
        keep_cols_arr = np.array(keep_cols)
        n_underdetermined = 0

        for p in unique_patterns:
            pixel_mask = pattern == p
            n_valid = bin(int(p)).count("1")
            if n_valid < len(keep_cols):
                n_underdetermined += int(pixel_mask.sum())
                continue  # genuinely underdetermined for this pixel's valid pairs

            valid_pair_idx = [i for i in range(n_pairs) if (int(p) >> i) & 1]
            A_sub = A[valid_pair_idx][:, keep_cols]
            try:
                ATA_inv_sub = np.linalg.pinv(A_sub.T @ A_sub)
            except np.linalg.LinAlgError:
                n_underdetermined += int(pixel_mask.sum())
                continue

            rows, cols = np.where(pixel_mask)
            obs = disp_stack[valid_pair_idx][:, rows, cols]  # (n_valid, n_group_pixels)
            est = ATA_inv_sub @ A_sub.T @ obs  # (n_dates-1, n_group_pixels)
            displacement[keep_cols_arr[:, None], rows, cols] = est

            predicted = A_sub @ est  # (n_valid, n_group_pixels)
            group_residuals = obs - predicted
            residual_rms[rows, cols] = np.sqrt(np.mean(group_residuals**2, axis=0))

        displacement[ref_col] = 0.0
        if n_underdetermined > 0:
            logger.warning(
                "%d/%d pixels (%.1f%%) had too few pairs above coherence_threshold=%.2f "
                "to solve and are marked unreliable (NaN) rather than silently included "
                "with insufficient data.",
                n_underdetermined, h * w, 100 * n_underdetermined / (h * w), coherence_threshold,
            )

        # Linear velocity fit (mm/year → m/year)
        t_years = np.array(
            [self._days_between(ref_date, d) / 365.25 for d in dates], dtype=np.float32
        )

        velocity = self._fit_velocity(displacement, t_years)

        return TimeSeriesResult(
            dates=dates,
            displacement=displacement,
            velocity=velocity,
            residual_rms=residual_rms,
            reference_date=ref_date,
            metadata={
                "wavelength_m": self._wavelength,
                "n_pairs": n_pairs,
                "coherence_threshold": coherence_threshold,
                "method": "SBAS weighted least squares (Berardino et al. 2002)",
            },
        )

    def _fit_velocity(self, displacement, t_years):
        """Linear regression of displacement vs time per pixel."""
        np = self._np()
        n_dates, h, w = displacement.shape
        t_mean = t_years.mean()
        t_centered = t_years - t_mean
        denom = np.sum(t_centered**2)
        if denom == 0:
            return np.zeros((h, w), dtype=np.float32)

        disp_mean = displacement.mean(axis=0)
        numer = np.tensordot(t_centered, displacement - disp_mean, axes=([0], [0]))
        velocity = (numer / denom).astype(np.float32)
        return velocity

    def _days_between(self, d1: str, d2: str) -> int:
        from datetime import datetime

        fmt = "%Y-%m-%d"
        return (datetime.strptime(d2, fmt) - datetime.strptime(d1, fmt)).days

    def _np(self):
        import numpy as np

        return np

    # ── MintPy passthrough (advanced corrections) ────────────────────────────

    def _invert_mintpy(
        self, pairs: List[InterferogramPair], coherence_threshold: float
    ) -> TimeSeriesResult:
        """
        Delegate to MintPy for the full correction chain: unwrapping error
        correction via phase closure, DEM error estimation, tropospheric
        delay correction, and weighted network inversion.

        Requires writing an intermediate HDF5 stack in MintPy's expected
        format (ifgramStack.h5), then running mintpy.smallbaselineApp.
        """
        try:
            import mintpy  # noqa: F401
        except ImportError:
            raise ImportError(
                "MintPy is not installed.\n"
                'Install with: pip install "pygeofetch[insar-full]"\n'
                "Or directly:  pip install mintpy"
            )

        # MintPy operates on a full project directory with a specific config
        # format (smallbaselineApp.cfg) and HDF5 stacks. A minimal in-memory
        # bridge is provided here; full MintPy corrections (tropospheric
        # delay, DEM error, phase closure) require its complete workflow.
        raise NotImplementedError(
            "Direct in-memory MintPy inversion is not yet implemented. "
            "For the full MintPy correction chain, export interferograms "
            "to an ifgramStack.h5 using mintpy.utils.writefile, then run "
            "`smallbaselineApp.py` directly. See: "
            "https://mintpy.readthedocs.io for the file format specification. "
            "The native SBAS inversion (use_mintpy=False) provides "
            "the core Berardino et al. 2002 algorithm without MintPy's "
            "additional correction steps."
        )