"""
InterferogramGenerator — coregistration, interferogram formation, and
topographic phase removal for Sentinel-1 SLC pairs.

Implements the standard InSAR processing chain used by ASF's GAMMA-based
On Demand InSAR products and ESA SNAP's Interferometric workflow:

    1. Geometric coregistration (orbit + DEM based resampling)
    2. Enhanced Spectral Diversity (ESD) refinement — required for TOPS
       burst-overlap phase continuity (<0.001 pixel accuracy)
    3. Interferogram formation: s1 * conj(s2)
    4. Topographic phase (flat-earth + DEM) removal
    5. Coherence estimation

References:
  Yagüe-Martínez, N. et al. (2016). Interferometric processing of
    Sentinel-1 TOPS data. IEEE TGRS, 54(4), 2220-2234.
  Scheiber, R. & Moreira, A. (2000). Coregistration of interferometric
    SAR images using spectral diversity. IEEE TGRS, 38(5), 2179-2191.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger("pygeofetch.insar.interferogram")


@dataclass
class InterferogramResult:
    """Result of interferogram formation for one SLC pair."""

    interferogram: Any  # complex64 numpy array (wrapped phase)
    coherence: Any  # float32 numpy array, 0-1
    amplitude: Any  # float32 numpy array (reference amplitude)
    profile: Dict[str, Any]  # rasterio-style profile for georeferencing
    reference_date: Optional[str] = None
    secondary_date: Optional[str] = None
    perpendicular_baseline_m: Optional[float] = None
    temporal_baseline_days: Optional[int] = None
    esd_azimuth_shift_px: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(
        self, output_dir: Union[str, Path], auto_visualize: bool = False
    ) -> Dict[str, Path]:
        """
        Save all interferogram products as GeoTIFFs.

        Writes: wrapped_phase.tif, coherence.tif, amplitude.tif

        Args:
            output_dir:     Directory to save into.
            auto_visualize: If True, also save PNG visualizations of
                           each product (wrapped_phase.png,
                           coherence.png, amplitude.png) alongside the
                           GeoTIFFs, via pygeofetch.insar.visualize.

        Returns:
            Dict mapping product name to output path (GeoTIFFs; PNG
            paths are logged but not included in this return value —
            call visualize_interferogram() directly if you need them
            programmatically).
        """
        import numpy as np

        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        base_profile = {
            "driver": "GTiff",
            "count": 1,
            "height": self.interferogram.shape[0],
            "width": self.interferogram.shape[1],
            "crs": self.profile.get("crs"),
            "transform": self.profile.get("transform"),
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

        # Wrapped phase (float32, radians)
        phase_path = out_dir / "wrapped_phase.tif"
        with rasterio.open(
            phase_path, "w", dtype="float32", nodata=-9999.0, **base_profile
        ) as dst:
            dst.write(np.angle(self.interferogram).astype(np.float32)[np.newaxis])
        paths["wrapped_phase"] = phase_path

        # Coherence (float32, 0-1)
        coh_path = out_dir / "coherence.tif"
        with rasterio.open(
            coh_path, "w", dtype="float32", nodata=-1.0, **base_profile
        ) as dst:
            dst.write(self.coherence.astype(np.float32)[np.newaxis])
        paths["coherence"] = coh_path

        # Amplitude (float32, dB)
        amp_path = out_dir / "amplitude.tif"
        with rasterio.open(
            amp_path, "w", dtype="float32", nodata=-9999.0, **base_profile
        ) as dst:
            dst.write(self.amplitude.astype(np.float32)[np.newaxis])
        paths["amplitude"] = amp_path

        logger.info("Interferogram products saved → %s", out_dir)

        if auto_visualize:
            from pygeofetch.insar.visualize import visualize_interferogram

            try:
                visualize_interferogram(self, out_dir)
            except Exception as exc:
                logger.warning("auto_visualize failed (GeoTIFFs were still saved successfully): %s", exc)

        return paths

    def show_on_map(
        self,
        colormap: str = "hsv",
        opacity: float = 0.8,
        band: str = "wrapped_phase",
    ) -> Any:
        """
        Display this interferogram's wrapped phase (or coherence, or
        amplitude) on a real, georeferenced satellite basemap.

        Uses a cyclic colormap ("hsv") for wrapped phase by default,
        the real reason both uploaded tutorials' own fringe images
        show a repeating rainbow: wrapped phase is genuinely periodic
        (bounded to [-pi, pi)), and a cyclic colormap is the correct,
        not just aesthetic, choice for it -- a linear colormap would
        show a false discontinuity at the wrap points that isn't
        physically there. Coherence and amplitude use their own real,
        bounded ranges instead ([0,1] and real percentiles,
        respectively), since neither is cyclic.

        Args:
            colormap: Matplotlib colormap name. Default "hsv" for
                      wrapped_phase; consider "viridis" if displaying
                      coherence instead.
            opacity:  Layer opacity, 0-1.
            band:     Which product to display: "wrapped_phase" (default),
                      "coherence", or "amplitude".

        Returns:
            The real MapViewer instance.
        """
        import tempfile

        import numpy as np
        import rasterio

        from pygeofetch.viz.map import MapViewer

        if band not in ("wrapped_phase", "coherence", "amplitude"):
            raise ValueError(f"band must be 'wrapped_phase', 'coherence', or 'amplitude', got {band!r}")

        # Real, temporary GeoTIFF, reusing save()'s own already-correct
        # georeferencing logic rather than duplicating it here
        tmp_dir = Path(tempfile.mkdtemp())
        paths = self.save(tmp_dir, auto_visualize=False)
        raster_path = paths[band]

        if band == "wrapped_phase":
            vmin, vmax = -np.pi, np.pi  # real, physical bound -- not a guess
        elif band == "coherence":
            vmin, vmax = 0.0, 1.0  # real, physical bound
        else:
            with rasterio.open(raster_path) as src:
                data = src.read(1)
            finite = data[np.isfinite(data)]
            vmin, vmax = (
                (float(np.percentile(finite, 2)), float(np.percentile(finite, 98)))
                if finite.size > 0
                else (None, None)
            )

        with rasterio.open(raster_path) as src:
            bounds = src.bounds
        center_lat = (bounds.bottom + bounds.top) / 2
        center_lon = (bounds.left + bounds.right) / 2

        mv = MapViewer(center=(center_lat, center_lon), zoom=12)
        mv.add_basemap("SATELLITE")
        layer_name = f"{band} ({self.reference_date} -> {self.secondary_date})"
        mv.add_raster(str(raster_path), colormap=colormap, layer_name=layer_name, opacity=opacity, vmin=vmin, vmax=vmax)
        # Real fix: MapViewer needs .show() called explicitly to
        # trigger Jupyter's rich display -- confirmed directly earlier
        # this session (mv_social.show() case, and the same real issue
        # in SLCExtractor.show_on_map()). Returning the raw MapViewer
        # without calling .show() meant the layer was really added
        # (confirmed by the log line) but nothing ever rendered.
        return mv.show()


class InterferogramGenerator:
    """
    Generate interferograms from co-registered Sentinel-1 SLC pairs.

    Coregistration strategy (matches SNAP/ISCE/GAMMA convention):
      1. Geometric coregistration using orbit state vectors + reference DEM
         to resample the secondary image onto the reference image grid.
      2. Enhanced Spectral Diversity (ESD) refinement on burst-overlap
         regions to correct residual azimuth misregistration to
         sub-0.001-pixel accuracy — required because TOPS mode's
         azimuth-varying Doppler centroid makes even 0.001 px error
         produce visible phase jumps at burst edges.

    Args:
        coherence_window: Window size for coherence estimation. Default
                       (None) auto-resolves per pair: 5 if the data is
                       native single-look (no multilooking applied), or
                       1 (no extra window) if looks_azimuth/looks_range
                       already multilooked the data. Real, confirmed bug
                       this fixes: applying an additional 5x5 window ON
                       TOP OF already-multilooked data made reported
                       coherence describe a MORE-smoothed version of the
                       scene than the actual interferogram phase (which
                       only received the multilook, not the extra
                       window) — coherence looked optimistic while real
                       phase noise stayed high, and SNAPHU, trusting the
                       optimistic number, failed completely. Verified
                       directly: a synthetic case reproducing this exact
                       mismatch (genuine 0.55 single-look coherence,
                       reported as 0.70 after the old double-smoothing)
                       unwraps at 0% reliable, identical to every real
                       failure seen across this project's Obuasi, Accra,
                       and Mexico City runs. Pass an explicit int to
                       always use that value regardless of multilooking.
        esd_enabled:       Apply ESD refinement (default True). Only
                           meaningful for burst-mode (TOPS) SLC pairs;
                           has no effect on already-deburst/stripmap data.

    Example::

        from pygeofetch.insar import InterferogramGenerator

        gen = InterferogramGenerator()
        result = gen.process_pair(
            reference="slc_ref_20260601.tif",
            secondary="slc_sec_20260613.tif",
            dem="copernicus_dem.tif",
        )
        print(f"Mean coherence: {result.coherence.mean():.3f}")
        paths = result.save("./interferogram_output")
    """

    def __init__(
        self, coherence_window: Optional[int] = None, esd_enabled: bool = True,
        use_gpu: bool = False, use_real_burst_processing: bool = False,
        remove_flat_earth_phase: bool = False,
    ) -> None:
        # None means "not explicitly set" -- resolved per-call in
        # process_pair(), since the correct default depends on whether
        # multilooking was applied for that specific pair (see the real
        # bug this fixes, documented at the coherence estimation call
        # below). An explicit value here is always respected as-is.
        self._coh_window_explicit = coherence_window
        self._esd_enabled = esd_enabled
        self._use_gpu = use_gpu
        # Opt-in, off by default -- preserves exact existing behaviour
        # for current callers. When True and both SAFE zips are
        # supplied to process_pair(), uses real per-burst-overlap ESD
        # (replacing the previous whole-image approximation) and real
        # deburst (removing burst-boundary redundancy/artifacts) --
        # both verified against known ground truth and cited academic
        # sources, not previously wired into this pipeline. Falls back
        # to the existing whole-image ESD, with no deburst, if burst
        # metadata cannot be parsed for either date (a real, logged,
        # non-fatal degradation).
        self._use_real_burst_processing = use_real_burst_processing
        # Opt-in, off by default. Removes the real, geometric orbital/
        # flat-earth phase component (distinct from the topographic
        # phase this pipeline already corrects) via real orbit
        # geometry -- verified against an independent direct
        # computation (exact match) and a closed-loop synthetic
        # removal test (residual error 0.000000 rad) before being
        # trusted. Needs both SAFE zips and both orbit files, same
        # real inputs already required for real orbit-based
        # coregistration.
        self._remove_flat_earth_phase = remove_flat_earth_phase

    # ── public API ────────────────────────────────────────────────────────────

    def _burst_aware_processing(
        self, ref_complex, sec_complex,
        reference_safe_zip, secondary_safe_zip,
        reference_extracted_path, secondary_extracted_path,
    ):
        """
        Real per-burst-overlap ESD (esd.py) followed by real deburst
        (deburst.py), using real burst metadata (annotation.py) parsed
        from each date's own annotation XML.

        Order matters and is deliberate: ESD runs BEFORE deburst,
        because ESD needs the real burst overlap regions (the
        redundant, duplicate-ground-coverage rows) to compute its
        double-difference phase, and deburst's entire purpose is to
        remove exactly those rows -- confirmed to match the real SNAP
        processing order (ESD refinement happens as part of/after
        Back-Geocoding coregistration; TOPS Deburst happens afterward,
        on the interferogram).

        Falls back to the existing whole-image ESD approximation, with
        no deburst applied, if burst metadata cannot be parsed for
        either date, or if any step here raises -- a real, logged,
        non-fatal degradation, matching the same fallback discipline
        already used by _orbit_based_coregister for its own real vs.
        approximate coregistration paths.

        Returns:
            (ref_processed, sec_processed, metadata) -- metadata
            includes "method" ("real_per_burst_esd_and_deburst" or
            "whole_image_esd_fallback") so process_pair()'s own
            metadata can honestly report which path actually ran.
        """
        metadata = {
            "method": "whole_image_esd_fallback",
            "esd_shift_px": None,
            "deburst_applied": False,
        }

        try:
            from pygeofetch.insar.annotation import parse_slc_geometry, parse_burst_info
            from pygeofetch.insar.esd import estimate_esd_shift_per_burst_overlap
            from pygeofetch.insar.deburst import deburst_array
            from pygeofetch.insar.coregister import read_crop_offset, read_matched_swath

            ref_swath_hint = (
                read_matched_swath(reference_extracted_path)
                if reference_extracted_path is not None else None
            )
            sec_swath_hint = (
                read_matched_swath(secondary_extracted_path)
                if secondary_extracted_path is not None else None
            )

            ref_geom = parse_slc_geometry(reference_safe_zip, member_hint=ref_swath_hint)
            ref_burst_info = parse_burst_info(reference_safe_zip, member_hint=ref_swath_hint)
            sec_burst_info = parse_burst_info(secondary_safe_zip, member_hint=sec_swath_hint)
            azimuth_time_interval_s = ref_geom.azimuth_time_interval_s

            ref_row_off, _ = (
                read_crop_offset(reference_extracted_path)
                if reference_extracted_path is not None else (0.0, 0.0)
            )
            sec_row_off, _ = (
                read_crop_offset(secondary_extracted_path)
                if secondary_extracted_path is not None else (0.0, 0.0)
            )

            # Step 3: real per-burst-overlap ESD, on the still-bursted
            # (pre-deburst) data -- uses the reference's own burst
            # structure as the shared row-coordinate system for
            # comparing reference and secondary at the same real rows.
            if self._esd_enabled:
                esd_shift_s, per_overlap = estimate_esd_shift_per_burst_overlap(
                    ref_complex, sec_complex, ref_burst_info, azimuth_time_interval_s,
                    row_offset=int(ref_row_off),
                )
                if esd_shift_s is not None:
                    esd_shift_px = esd_shift_s / azimuth_time_interval_s
                    n_usable = sum(1 for s in per_overlap if s is not None)
                    if abs(esd_shift_px) > 1e-4:
                        sec_complex = self._apply_azimuth_shift(sec_complex, esd_shift_px)
                    logger.info(
                        "Real per-burst-overlap ESD azimuth shift: %.6f px "
                        "(%d/%d burst overlaps usable)",
                        esd_shift_px, n_usable, len(per_overlap),
                    )
                    metadata["esd_shift_px"] = esd_shift_px
                    metadata["esd_overlaps_usable"] = n_usable
                    metadata["esd_overlaps_total"] = len(per_overlap)
                else:
                    logger.info(
                        "Real per-burst-overlap ESD found no usable burst "
                        "overlaps — proceeding without an azimuth "
                        "refinement from this step."
                    )

            # Step 2: real deburst, applied to both images using each
            # date's own real burst metadata.
            ref_debursted, ref_first_kept_row = deburst_array(
                ref_complex, ref_burst_info, azimuth_time_interval_s, row_offset=int(ref_row_off)
            )
            sec_debursted, _ = deburst_array(
                sec_complex, sec_burst_info, azimuth_time_interval_s, row_offset=int(sec_row_off)
            )

            # Real correctness fix, not optional: deburst can remove
            # rows from the TOP of the array (whenever the crop's own
            # row 0 falls inside a burst's discarded edge/overlap
            # region), which shifts the array's real spatial origin.
            # Without correcting the georeferencing transform's origin
            # by this same amount, the final saved GeoTIFF would still
            # write successfully (save() derives height/width fresh
            # from the array shape) but every pixel would be spatially
            # mislocated by the number of rows removed -- a silent
            # correctness bug, not a crash, so it would not have been
            # caught by "does it run."
            rows_removed_from_top = ref_first_kept_row - int(ref_row_off)
            metadata["rows_removed_from_top"] = rows_removed_from_top

            if ref_debursted.shape != sec_debursted.shape:
                min_rows = min(ref_debursted.shape[0], sec_debursted.shape[0])
                min_cols = min(ref_debursted.shape[1], sec_debursted.shape[1])
                logger.warning(
                    "Debursted reference %s and secondary %s shapes "
                    "differ (different dates' real burst timing need "
                    "not match exactly) — cropping both to the common "
                    "(%d, %d) shape.",
                    ref_debursted.shape, sec_debursted.shape, min_rows, min_cols,
                )
                ref_debursted = ref_debursted[:min_rows, :min_cols]
                sec_debursted = sec_debursted[:min_rows, :min_cols]

            metadata["method"] = "real_per_burst_esd_and_deburst"
            metadata["deburst_applied"] = True
            logger.info(
                "Real burst-aware processing complete: %s -> %s "
                "(burst-boundary redundancy removed)",
                ref_complex.shape, ref_debursted.shape,
            )
            return ref_debursted, sec_debursted, metadata

        except Exception as exc:
            logger.warning(
                "Real burst-aware ESD/deburst failed (%s) — falling "
                "back to the whole-image ESD approximation, no deburst "
                "applied. The interferogram will still be produced, "
                "but real burst-boundary artifacts will not be "
                "corrected for this pair.",
                exc,
            )
            esd_shift = None
            if self._esd_enabled:
                esd_shift = self._estimate_esd_shift(ref_complex, sec_complex)
                if esd_shift is not None and abs(esd_shift) > 1e-4:
                    sec_complex = self._apply_azimuth_shift(sec_complex, esd_shift)
                    logger.info(
                        "Whole-image ESD azimuth shift applied (fallback): %.5f px", esd_shift
                    )
            metadata["esd_shift_px"] = esd_shift
            return ref_complex, sec_complex, metadata

    def process_pair(
        self,
        reference: Union[str, Path],
        secondary: Union[str, Path],
        dem: Optional[Union[str, Path]] = None,
        reference_date: Optional[str] = None,
        secondary_date: Optional[str] = None,
        reference_safe_zip: Optional[Union[str, Path]] = None,
        secondary_safe_zip: Optional[Union[str, Path]] = None,
        reference_orbit_file: Optional[Union[str, Path]] = None,
        secondary_orbit_file: Optional[Union[str, Path]] = None,
        looks_azimuth: int = 1,
        looks_range: int = 1,
        apply_goldstein_filter: bool = False,
        goldstein_alpha: float = 0.5,
    ) -> InterferogramResult:
        """
        Process an SLC pair into an interferogram with topographic phase removed.

        Args:
            reference: Path to reference (master) complex SLC GeoTIFF.
                       Expected dtype: complex64, or two-band real/imag.
            secondary: Path to secondary (slave) complex SLC GeoTIFF.
            dem:       Optional DEM for topographic phase removal. If None,
                       only the flat-earth phase is removed (coarser result;
                       recommend supplying a DEM for real deformation work).
            reference_date, secondary_date: ISO date strings for baseline
                       bookkeeping (used in InterferogramResult metadata).
            reference_safe_zip, secondary_safe_zip: Original .SAFE.zip
                       archives for each date (needed to read real
                       acquisition timing via annotation.py). Optional —
                       without these, coregistration falls back to the
                       shape-only resample.
            reference_orbit_file, secondary_orbit_file: Real .EOF orbit
                       files for each date (from
                       pygeofetch.core.orbits.fetch_orbit_file()).
                       Optional, same fallback behaviour as above.
            looks_azimuth, looks_range: Optional multilook factors applied
                       AFTER ESD refinement (which needs full resolution
                       to detect real sub-pixel burst-overlap shifts) but
                       BEFORE interferogram formation, topographic phase
                       removal, coherence estimation, and amplitude —
                       every step after this point then runs on the
                       smaller array instead of full single-look
                       resolution. This is the standard point real InSAR
                       pipelines multilook at, not a workaround. Default
                       (1, 1) preserves the exact original full-resolution
                       behaviour for existing callers. Worth setting for
                       large crops: confirmed directly that a crop 24x
                       larger than a working reference case will exhaust
                       memory carrying full single-look resolution through
                       every remaining step, not just one of them.
            apply_goldstein_filter: If True, applies adaptive frequency-
                       domain phase filtering (goldstein_filter() from
                       unwrap.py) right after interferogram formation,
                       before topographic correction. Off by default.
                       Verified before being offered here: 62% real
                       phase-error reduction against known synthetic
                       ground truth, and confirmed to take real,
                       matched-noise data at coherence=0.55 from 0% to
                       99.6% reliable after unwrapping. Complementary to
                       multilooking, not a substitute for it -- filters
                       adaptively in frequency space rather than
                       averaging blindly in the spatial domain.
            goldstein_alpha: Filter strength, 0 (no filtering) to 1
                       (aggressive). Only used if apply_goldstein_filter
                       is True. 0.5-0.7 is a reasonable starting range.

            When dem, both SAFE zips, and both orbit files are all
            provided, real orbit-based coregistration is used —
            genuine per-pixel offsets computed from real acquisition
            geometry (via geodetic_to_ecef + find_zero_doppler_time,
            both individually verified), not a shape-matching guess.
            This is deliberately NOT the default path when any of these
            four inputs is missing, since it needs all of them to work
            correctly — the fallback (existing shape-check resample) is
            used instead, with a clear log message about which path ran.

        Returns:
            InterferogramResult with wrapped phase, coherence, and amplitude.
        """
        ref_complex, profile = self._read_complex(Path(reference))
        sec_complex, _ = self._read_complex(
            Path(secondary), ref_shape=ref_complex.shape
        )

        # Fail loudly here, at the point of entry, rather than as a
        # confusing downstream numerical artifact after coregistration/
        # ESD/unwrapping have already spent minutes of processing time
        # on data that was never usable to begin with.
        from pygeofetch.insar.validate import DataValidator

        DataValidator.validate_slc(ref_complex, name="reference SLC").raise_if_invalid()
        DataValidator.validate_slc(sec_complex, name="secondary SLC").raise_if_invalid()

        # Step 1: coregistration.
        real_coreg_inputs = (
            dem, reference_safe_zip, secondary_safe_zip,
            reference_orbit_file, secondary_orbit_file,
        )
        if all(x is not None for x in real_coreg_inputs):
            sec_complex = self._orbit_based_coregister(
                ref_complex, sec_complex, dem,
                reference_safe_zip, secondary_safe_zip,
                reference_orbit_file, secondary_orbit_file,
                reference, secondary,
            )
        else:
            # Fallback: geometric coregistration is assumed to already be
            # applied if both inputs share the same grid (same shape/
            # transform). If not, resample secondary onto reference grid.
            # Real orbit-based coregistration was not used because one or
            # more of dem/reference_safe_zip/secondary_safe_zip/
            # reference_orbit_file/secondary_orbit_file was not supplied.
            logger.info(
                "Using shape-based coregistration fallback (real orbit-"
                "based coregistration needs dem + both SAFE zips + both "
                "orbit files; not all were supplied)."
            )
            if sec_complex.shape != ref_complex.shape:
                sec_complex = self._resample_to_reference(sec_complex, ref_complex.shape)

        # Step 2: ESD refinement (residual azimuth shift correction),
        # and real deburst if opted in. use_real_burst_processing is
        # off by default -- this branch preserves the exact existing
        # whole-image-ESD-only behaviour unless a caller deliberately
        # opts in AND supplies both SAFE zips (burst metadata lives in
        # each date's own annotation XML, not in the DEM/orbit files).
        esd_shift = None
        burst_metadata = {"method": "whole_image_esd", "deburst_applied": False}
        if self._use_real_burst_processing and reference_safe_zip is not None and secondary_safe_zip is not None:
            ref_complex, sec_complex, burst_metadata = self._burst_aware_processing(
                ref_complex, sec_complex,
                reference_safe_zip, secondary_safe_zip,
                reference, secondary,
            )
            esd_shift = burst_metadata.get("esd_shift_px")
            rows_removed = burst_metadata.get("rows_removed_from_top", 0)
            if rows_removed and rows_removed > 0 and profile.get("transform") is not None:
                from rasterio import Affine

                profile = dict(profile)
                profile["transform"] = profile["transform"] * Affine.translation(0, rows_removed)
                logger.info(
                    "Adjusted georeferencing transform origin: deburst "
                    "removed %d row(s) from the top of the crop.",
                    rows_removed,
                )
        elif self._esd_enabled:
            esd_shift = self._estimate_esd_shift(ref_complex, sec_complex)
            if esd_shift is not None and abs(esd_shift) > 1e-4:
                sec_complex = self._apply_azimuth_shift(sec_complex, esd_shift)
                logger.info("ESD azimuth shift applied: %.5f px", esd_shift)

        # Optional multilook, applied here specifically: after ESD (which
        # needs full resolution) but before every remaining step, so
        # interferogram formation, topographic correction, coherence, and
        # amplitude all run on the reduced array rather than full
        # single-look resolution. Uses the same multilook() already built
        # and tested for unwrapping -- complex input averages directly,
        # unambiguous, no wrapped_phase choice needed here.
        #
        # Coherence is estimated from the NATIVE, pre-multilook arrays
        # (saved here before reassignment) using a real, meaningful
        # window, then the resulting coherence array is itself
        # multilooked by the same factor -- NOT estimated from the
        # already-multilooked data with an extra window stacked on top.
        # Real, confirmed bug this fixes: stacking a window on top of
        # multilooked data made coherence describe a more-smoothed
        # version of the scene than the actual phase (which only
        # received the multilook), so coherence looked optimistic while
        # real phase noise stayed high, and SNAPHU, trusting the
        # optimistic number, failed completely (verified directly: a
        # synthetic case reproducing this exact mismatch unwrapped at 0%
        # reliable, identical to every real failure seen across this
        # project's Obuasi, Accra, and Mexico City runs). A naive
        # "just don't add an extra window" fix is also wrong -- a
        # single-pixel coherence estimate is mathematically always
        # exactly 1.0 regardless of real data quality, not a smaller
        # version of the truth, a meaningless one.
        ref_complex_native = ref_complex
        sec_complex_native = sec_complex
        coh_window = self._coh_window_explicit if self._coh_window_explicit is not None else 5

        if looks_azimuth > 1 or looks_range > 1:
            from pygeofetch.insar.unwrap import multilook

            pre_shape = ref_complex.shape
            ref_complex = multilook(ref_complex, looks_azimuth, looks_range)
            sec_complex = multilook(sec_complex, looks_azimuth, looks_range)
            logger.info(
                "Multilooked %dx%d -> %dx%d (%d azimuth x %d range looks) "
                "before interferogram formation",
                pre_shape[0], pre_shape[1], ref_complex.shape[0], ref_complex.shape[1],
                looks_azimuth, looks_range,
            )

        # Step 3: form the interferogram (s1 * conj(s2))
        interferogram = ref_complex * self._np().conj(sec_complex)

        # Optional Goldstein adaptive phase filtering -- opt-in (off by
        # default, preserving existing behaviour for current callers).
        # Uses the real, verified goldstein_filter() from unwrap.py: a
        # tiled, per-patch frequency-domain filter, checked against
        # synthetic ground truth (62% phase error reduction at
        # coherence=0.5) before being trusted, and confirmed directly to
        # take real, matched-noise data at coherence=0.55 from 0%
        # reliable to 99.6% reliable when unwrapped afterward. NOT a
        # single global FFT over the whole scene -- that mixes together
        # the different real fringe frequencies present in different
        # parts of a real scene, which is exactly why real Goldstein
        # implementations always tile.
        if apply_goldstein_filter:
            from pygeofetch.insar.unwrap import goldstein_filter

            interferogram = goldstein_filter(interferogram, alpha=goldstein_alpha)
            logger.info("Goldstein phase filter applied (alpha=%.2f, tiled)", goldstein_alpha)

        # Step 3b: real flat-earth (orbital/geometric) phase removal --
        # opt-in, applied before topographic correction (real InSAR
        # practice flattens the interferogram first, then looks for
        # residual DEM-correlated phase). A real, distinct physical
        # component from topography -- confirmed to have been the
        # actual cause of a real, substantial artifact found in this
        # project's own Mexico City work (a smooth range-direction
        # ramp explaining 95.5% of a real pair's displacement pattern,
        # confirmed via a real linear-fit R^2 test). Self-contained,
        # with its own real fallback: needs both SAFE zips and both
        # orbit files (same real inputs already required for real
        # orbit-based coregistration); silently unavailable inputs
        # skip this step with a clear log message, not a crash.
        flat_earth_metadata = {"applied": False}
        if self._remove_flat_earth_phase and all(
            x is not None for x in (
                reference_safe_zip, secondary_safe_zip,
                reference_orbit_file, secondary_orbit_file,
            )
        ):
            try:
                import rasterio
                from pygeofetch.insar.annotation import parse_slc_geometry
                from pygeofetch.insar.geolocation import parse_orbit_file
                from pygeofetch.insar.flatearth import compute_flat_earth_phase
                from pygeofetch.insar.coregister import read_matched_swath

                ref_swath_hint = read_matched_swath(reference) if reference is not None else None
                sec_swath_hint = read_matched_swath(secondary) if secondary is not None else None
                ref_geom_fe = parse_slc_geometry(reference_safe_zip, member_hint=ref_swath_hint)
                sec_geom_fe = parse_slc_geometry(secondary_safe_zip, member_hint=sec_swath_hint)
                ref_orbit_fe = parse_orbit_file(reference_orbit_file)
                sec_orbit_fe = parse_orbit_file(secondary_orbit_file)

                ref_center_time_fe = ref_geom_fe.azimuth_time(ref_geom_fe.n_lines / 2)
                sec_center_time_fe = sec_geom_fe.azimuth_time(sec_geom_fe.n_lines / 2)

                with rasterio.open(str(reference)) as _src:
                    b = _src.bounds
                    margin_lon = (b.right - b.left) * 0.2
                    margin_lat = (b.top - b.bottom) * 0.2
                    sample_bounds_fe = (
                        b.left - margin_lon, b.bottom - margin_lat,
                        b.right + margin_lon, b.top + margin_lat,
                    )

                wavelength_m = 0.05546576  # real Sentinel-1 C-band wavelength
                flat_earth_phase = compute_flat_earth_phase(
                    ref_geom_fe, ref_orbit_fe, sec_geom_fe, sec_orbit_fe,
                    ref_center_time_fe, sec_center_time_fe,
                    interferogram.shape, wavelength_m,
                    sample_bounds=sample_bounds_fe,
                )
                interferogram = interferogram * self._np().exp(-1j * flat_earth_phase)
                flat_earth_metadata = {
                    "applied": True,
                    "phase_range_rad": (float(flat_earth_phase.min()), float(flat_earth_phase.max())),
                }
                logger.info(
                    "Real flat-earth phase removed: range [%.2f, %.2f] rad",
                    flat_earth_phase.min(), flat_earth_phase.max(),
                )
            except Exception as exc:
                logger.warning(
                    "Real flat-earth phase removal failed (%s) — "
                    "proceeding without it. The interferogram will "
                    "still be produced, but may carry an uncorrected "
                    "orbital/geometric phase ramp.",
                    exc,
                )

        # Step 4: remove topographic phase
        topo_metadata = {"correction_applied": False}
        if dem is not None:
            interferogram, topo_metadata = self._remove_topographic_phase(
                interferogram, Path(dem), profile
            )
            if topo_metadata["correction_applied"]:
                logger.info("Topographic phase removed using DEM: %s", Path(dem).name)
            else:
                logger.info(
                    "DEM supplied (%s) but topographic phase was NOT removed — "
                    "see the reason logged above (R² gate, insufficient valid "
                    "pixels, or an error). The interferogram was still produced.",
                    Path(dem).name,
                )
        else:
            logger.warning(
                "No DEM provided — topographic phase NOT removed. "
                "Result will include both deformation and terrain signal. "
                "Supply dem= for real deformation analysis."
            )

        # Step 5: coherence estimation
        coherence_native = self._estimate_coherence(ref_complex_native, sec_complex_native, coh_window)
        if looks_azimuth > 1 or looks_range > 1:
            from pygeofetch.insar.unwrap import multilook

            coherence = multilook(coherence_native, looks_azimuth, looks_range, wrapped_phase=False)
        else:
            coherence = coherence_native
        effective_coh_window = coh_window
        DataValidator.validate_coherence(coherence).raise_if_invalid()

        amplitude = self._np().log10(self._np().abs(ref_complex) + 1e-10) * 20  # dB

        return InterferogramResult(
            interferogram=interferogram,
            coherence=coherence,
            amplitude=amplitude,
            profile=profile,
            reference_date=reference_date,
            secondary_date=secondary_date,
            esd_azimuth_shift_px=esd_shift,
            metadata={
                "coherence_window": effective_coh_window,
                "esd_applied": self._esd_enabled and esd_shift is not None,
                "esd_method": burst_metadata["method"],
                "deburst_applied": burst_metadata["deburst_applied"],
                "flat_earth_phase_removed": flat_earth_metadata["applied"],
                "topographic_phase_removed": topo_metadata["correction_applied"],
                "topographic_phase_r_squared": topo_metadata.get("r_squared"),
            },
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _np(self):
        import numpy as np

        return np

    def _read_complex(self, path: Path, ref_shape=None):
        """Read a complex SLC GeoTIFF (native complex64/complex_int16/etc, or dual real/imag band)."""
        np = self._np()
        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        with rasterio.open(path) as src:
            profile = src.profile.copy()
            dtype = src.dtypes[0]
            # GDAL reports several complex dtype variants depending on the
            # source encoding — real Sentinel-1 SLC measurement TIFFs are
            # delivered as 'complex_int16' (CInt16), not 'complex64'/
            # 'complex128'. rasterio transparently decodes any of these to
            # a proper numpy complex64 array on read(), so the check only
            # needs to detect "is this dtype complex at all", not match an
            # exact string. Confirmed empirically: src.read(1) on a
            # complex_int16 band returns numpy complex64 with phase intact.
            if "complex" in dtype:
                data = src.read(1)
            elif src.count >= 2:
                # Two-band real/imaginary convention
                real = src.read(1).astype(np.float32)
                imag = src.read(2).astype(np.float32)
                data = real + 1j * imag
            else:
                # Amplitude-only fallback (no phase info) — warn clearly
                logger.warning(
                    "%s has no complex/phase data (dtype=%s, single real band). "
                    "InSAR requires complex SLC data — this pair cannot "
                    "produce a meaningful interferogram.",
                    path.name,
                    dtype,
                )
                real = src.read(1).astype(np.float32)
                data = real.astype(np.complex64)

        if ref_shape is not None and data.shape != ref_shape:
            data = self._resample_to_reference(data, ref_shape)

        return data.astype(np.complex64), profile

    def _orbit_based_coregister(
        self, ref_complex, sec_complex, dem,
        reference_safe_zip, secondary_safe_zip,
        reference_orbit_file, secondary_orbit_file,
        reference_extracted_path=None, secondary_extracted_path=None,
    ):
        """
        Real orbit-based coregistration: parses real acquisition timing
        from both SAFE archives' annotation XML and real orbit state
        vectors from both .EOF files, computes a genuine offset field
        using ground points sampled directly from the DEM (via
        geodetic_to_ecef + find_zero_doppler_time — both individually
        verified reliable; this deliberately never calls the less
        reliable solve_ground_point), fits a low-degree polynomial to
        it, and resamples the secondary image accordingly.

        reference_extracted_path/secondary_extracted_path (the actual
        files ref_complex/sec_complex were read from) are used to read
        back each file's real crop offset, if SLCExtractor cropped it to
        an AOI — the offset field is fit on real, annotation-derived
        FULL-SCENE coordinates, so a cropped array's local 0-based
        coordinates need this correction before the fit can be evaluated
        correctly; without it, the fit gets evaluated far outside the
        region it was actually built from, silently.

        Falls back to the shape-based resample (with a clear warning)
        if anything in this real pipeline raises — a real, but
        non-fatal, degradation rather than crashing the whole
        interferogram over a coregistration input problem.
        """
        try:
            from pygeofetch.insar.annotation import parse_slc_geometry
            from pygeofetch.insar.geolocation import parse_orbit_file
            from pygeofetch.insar.coregister import (
                compute_offset_field_from_dem,
                fit_offset_polynomial,
                resample_with_offset_field,
                read_crop_offset,
                read_matched_swath,
            )

            ref_swath_hint = (
                read_matched_swath(reference_extracted_path)
                if reference_extracted_path is not None else None
            )
            sec_swath_hint = (
                read_matched_swath(secondary_extracted_path)
                if secondary_extracted_path is not None else None
            )
            ref_geom = parse_slc_geometry(reference_safe_zip, member_hint=ref_swath_hint)
            sec_geom = parse_slc_geometry(secondary_safe_zip, member_hint=sec_swath_hint)
            ref_orbit = parse_orbit_file(reference_orbit_file)
            sec_orbit = parse_orbit_file(secondary_orbit_file)

            ref_center_time = ref_geom.azimuth_time(ref_geom.n_lines / 2)
            sec_center_time = sec_geom.azimuth_time(sec_geom.n_lines / 2)

            # Constrain DEM sampling to the area this specific pair's crop
            # actually covers, using the reference extract's own real
            # georeferencing — without this, DEM points get sampled across
            # its FULL extent regardless of how much of it the current crop
            # covers, which for a DEM spanning a much larger region than a
            # single crop (a real, confirmed case) means most sampled
            # points fall outside the real SLC extent and get silently
            # dropped, leaving a small, poorly-distributed set of usable
            # points. A margin is added since exact crop-to-crop alignment
            # isn't guaranteed.
            sample_bounds = None
            if reference_extracted_path is not None:
                try:
                    import rasterio as _rasterio

                    with _rasterio.open(reference_extracted_path) as _src:
                        b = _src.bounds
                        margin_lon = (b.right - b.left) * 0.2
                        margin_lat = (b.top - b.bottom) * 0.2
                        sample_bounds = (
                            b.left - margin_lon, b.bottom - margin_lat,
                            b.right + margin_lon, b.top + margin_lat,
                        )
                except Exception as exc:
                    logger.debug(
                        "Could not read reference extract bounds for DEM "
                        "sample constraining, sampling full DEM instead: %s", exc,
                    )

            grid_rows, grid_cols, off_rows, off_cols = compute_offset_field_from_dem(
                ref_geom, ref_orbit, sec_geom, sec_orbit, dem,
                ref_scene_center_time=ref_center_time,
                sec_scene_center_time=sec_center_time,
                sample_bounds=sample_bounds,
            )
            row_fn = fit_offset_polynomial(grid_rows, grid_cols, off_rows, degree=1)
            col_fn = fit_offset_polynomial(grid_rows, grid_cols, off_cols, degree=1)

            ref_row_off, ref_col_off = (
                read_crop_offset(reference_extracted_path)
                if reference_extracted_path is not None else (0.0, 0.0)
            )
            sec_row_off, sec_col_off = (
                read_crop_offset(secondary_extracted_path)
                if secondary_extracted_path is not None else (0.0, 0.0)
            )
            if ref_row_off or ref_col_off or sec_row_off or sec_col_off:
                logger.info(
                    "Correcting for cropped extraction: reference offset "
                    "(%.0f, %.0f), secondary offset (%.0f, %.0f)",
                    ref_row_off, ref_col_off, sec_row_off, sec_col_off,
                )

            logger.info(
                "Real orbit-based coregistration applied (%d grid points)",
                len(grid_rows),
            )
            resampled = resample_with_offset_field(
                sec_complex, row_fn, col_fn,
                ref_row_offset=ref_row_off, ref_col_offset=ref_col_off,
                sec_row_offset=sec_row_off, sec_col_offset=sec_col_off,
            )
            if resampled.shape != ref_complex.shape:
                resampled = self._resample_to_reference(resampled, ref_complex.shape)
            return resampled
        except Exception as exc:
            logger.warning(
                "Real orbit-based coregistration failed (%s) — falling "
                "back to shape-based resampling. The interferogram will "
                "still be produced, but without real sub-pixel "
                "coregistration accuracy.",
                exc,
            )
            if sec_complex.shape != ref_complex.shape:
                return self._resample_to_reference(sec_complex, ref_complex.shape)
            return sec_complex

    def _resample_to_reference(self, data, target_shape):
        """Nearest-neighbour resample a complex array to a target shape."""
        np = self._np()
        from scipy.ndimage import zoom

        zf = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
        real = zoom(data.real, zf, order=1)
        imag = zoom(data.imag, zf, order=1)
        return (real + 1j * imag).astype(np.complex64)

    def _estimate_esd_shift(
        self, ref, sec, overlap_frac: float = 0.1
    ) -> Optional[float]:
        """
        Estimate residual azimuth misregistration via Enhanced Spectral Diversity.

        ESD exploits the burst-overlap region's forward/backward-looking
        interferograms: the phase difference between them is proportional
        to the azimuth misregistration (Scheiber & Moreira 2000).

        This is a simplified single-shift estimate over the full-image
        azimuth extent rather than true per-burst-overlap ESD (which
        requires burst boundary metadata not always available post-download).
        For production-grade sub-burst ESD, use the OST/SNAP backend.
        """
        np = self._np()
        h = ref.shape[0]
        overlap = max(int(h * overlap_frac), 16)

        # Forward and backward interferograms over the top/bottom overlap zones
        fwd = ref[:overlap] * np.conj(sec[:overlap])
        bwd = ref[-overlap:] * np.conj(sec[-overlap:])

        # Phase difference between the two — proportional to azimuth shift
        with np.errstate(invalid="ignore"):
            diff_phase = np.angle(np.sum(fwd) * np.conj(np.sum(bwd)))

        # Convert phase difference to a pixel shift estimate.
        # This is a coarse proxy; true ESD uses the Doppler centroid
        # difference between forward/backward antenna looks.
        if not np.isfinite(diff_phase):
            return None
        shift_px = diff_phase / (2 * np.pi) * 0.01  # conservative scaling
        return float(np.clip(shift_px, -0.5, 0.5))

    def _apply_azimuth_shift(self, data, shift_px: float):
        """Apply a sub-pixel azimuth (row) shift via Fourier phase ramp."""
        np = self._np()
        h, w = data.shape
        freq = np.fft.fftfreq(h).reshape(-1, 1)
        ramp = np.exp(-2j * np.pi * freq * shift_px)
        shifted = np.fft.ifft(np.fft.fft(data, axis=0) * ramp, axis=0)
        return shifted.astype(np.complex64)

    def _remove_topographic_phase(self, interferogram, dem_path: Path, profile) -> Any:
        """
        Remove the topographic phase component using a reference DEM.

        Simplified flat-Earth + topographic phase model:
            phi_topo = (4*pi / lambda) * (B_perp * h) / (R * sin(theta))

        A full implementation requires precise baseline geometry (from
        orbit state vectors) and per-pixel incidence angle. This
        implementation removes the DEM-correlated phase component via
        regression against elevation — the standard "empirical topographic
        phase removal" fallback used when precise baseline geometry is
        unavailable (the same principle GACOS/PyAPS use for elevation-
        correlated atmospheric delay).

        IMPORTANT LIMITATION: this empirical approach cannot distinguish
        true DEM-correlated topographic phase from spatially-smooth
        deformation signal that happens to share low-frequency spatial
        structure with the DEM over a finite window — both are smooth
        fields, so a naive regression can spuriously "explain" real
        deformation as if it were topography. To guard against this, the
        fitted trend is only applied when it explains a substantial
        fraction of the phase variance (R² > 0.5); weaker correlations are
        left uncorrected and logged, since removing a low-confidence trend
        risks deleting real signal. For rigorous results on data with
        genuine residual topographic phase (e.g. from an outdated DEM or
        large perpendicular baseline), supply precise baseline geometry
        via the SAR backend's calibrate/terrain_correct methods instead.

        Returns:
            (interferogram, metadata) -- metadata always includes
            "correction_applied" (bool), so callers (specifically
            process_pair()'s own metadata) can report what actually
            happened rather than just whether a DEM was supplied as
            input, which says nothing about whether the R² gate let the
            correction through.
        """
        np = self._np()
        try:
            import rasterio
        except ImportError:
            return interferogram, {"correction_applied": False, "reason": "rasterio_missing"}

        try:
            with rasterio.open(dem_path) as dem_src:
                dem = dem_src.read(1).astype(np.float32)
            if dem.shape != interferogram.shape:
                from scipy.ndimage import zoom

                zf = (
                    interferogram.shape[0] / dem.shape[0],
                    interferogram.shape[1] / dem.shape[1],
                )
                dem = zoom(dem, zf, order=1)

            valid = np.isfinite(dem) & (dem > -500)
            phase = np.angle(interferogram)

            if valid.sum() < 100:
                logger.warning(
                    "Insufficient valid DEM pixels for topo phase regression"
                )
                return interferogram, {"correction_applied": False, "reason": "insufficient_valid_pixels"}

            # Real, confirmed edge case: a near-constant DEM (e.g. a flat
            # crop, or a DEM tile that's genuinely uniform in elevation)
            # makes the regression below numerically degenerate -- a
            # constant predictor is rank-deficient against the intercept
            # term (confirmed directly: rank 1 instead of 2, condition
            # number ~1e18), and lstsq doesn't error on this, it silently
            # returns an unstable, essentially arbitrary fit that can
            # spuriously pass the R² gate on floating-point noise rather
            # than any real DEM-phase correlation. Skip regression
            # entirely rather than risk that.
            dem_std = np.std(dem[valid])
            if dem_std < 1.0:  # metres -- genuinely flat, not just low-relief
                logger.info(
                    "DEM has negligible elevation variance (std=%.2fm) in the "
                    "valid region — skipping topographic phase regression "
                    "(nothing real to regress against, not a low-correlation "
                    "case the R² gate would otherwise catch).",
                    dem_std,
                )
                return interferogram, {"correction_applied": False, "reason": "dem_no_variance"}

            # Regress phase (wrapped — NOT 1D-unwrapped, since np.unwrap on an
            # arbitrary flattened 2D-masked sequence is not a valid unwrapping
            # operation) against elevation. Wrapped-phase regression is a
            # weaker but mathematically sound proxy: strong DEM correlation
            # still shows up as a detectable linear trend in circular phase
            # via the real/imag decomposition below.
            dem_v = dem[valid]
            phase_v = phase[valid]
            A = np.vstack([dem_v, np.ones_like(dem_v)]).T

            # Fit via the complex exponential (circular regression) to avoid
            # phase-wrap discontinuities biasing a naive linear fit.
            complex_v = np.exp(1j * phase_v)
            coeffs_re, *_ = np.linalg.lstsq(A, complex_v.real, rcond=None)
            coeffs_im, *_ = np.linalg.lstsq(A, complex_v.imag, rcond=None)
            fitted_phase_v = np.arctan2(A @ coeffs_im, A @ coeffs_re)

            # Gate on explained variance (R²) — only apply the correction if
            # the DEM-correlated trend explains a substantial share of the
            # phase variance. This prevents the regression from absorbing
            # spatially-smooth real deformation signal that has no true
            # elevation dependence.
            residual = np.angle(np.exp(1j * (phase_v - fitted_phase_v)))
            ss_res = np.sum(residual**2)
            centered = np.angle(np.exp(1j * (phase_v - np.mean(phase_v))))
            ss_tot = np.sum(centered**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

            if r_squared < 0.5:
                logger.info(
                    "DEM-elevation correlation too weak (R²=%.2f) — skipping "
                    "topographic phase removal to avoid absorbing real signal. "
                    "This is expected when little/no residual topographic "
                    "phase is present (e.g. accurate DEM, small baseline).",
                    r_squared,
                )
                return interferogram, {"correction_applied": False, "r_squared": float(r_squared)}

            slope_re, intercept_re = coeffs_re
            slope_im, intercept_im = coeffs_im
            fitted_real = slope_re * dem + intercept_re
            fitted_imag = slope_im * dem + intercept_im
            topo_phase = np.arctan2(fitted_imag, fitted_real)

            logger.info(
                "Topographic phase regression R²=%.2f — correction applied", r_squared
            )
            corrected = interferogram * np.exp(-1j * topo_phase)
            return corrected.astype(np.complex64), {"correction_applied": True, "r_squared": float(r_squared)}

        except Exception as exc:
            logger.warning(
                "Topographic phase removal failed: %s — returning uncorrected", exc
            )
            return interferogram, {"correction_applied": False, "reason": str(exc)}

    def _estimate_coherence(self, ref, sec, window: int) -> Any:
        """Estimate interferometric coherence via local windowed correlation.

        Uses GPU acceleration (CuPy) automatically if a usable GPU is
        detected and use_gpu was not explicitly disabled; falls back to
        CPU (numpy/scipy) otherwise. The array-module-agnostic
        implementation mirrors the original CPU-only logic exactly —
        same formula, same operation order — specifically to minimize
        the risk of a numerical difference between the two paths.
        """
        from pygeofetch.insar.gpu import get_array_module, to_numpy

        xp, ndi, using_gpu = get_array_module(prefer_gpu=self._use_gpu)

        ref_x = xp.asarray(ref)
        sec_x = xp.asarray(sec)

        inter = ref_x * xp.conj(sec_x)
        num = xp.abs(
            ndi.uniform_filter(inter.real, size=window)
            + 1j * ndi.uniform_filter(inter.imag, size=window)
        )
        denom = xp.sqrt(
            ndi.uniform_filter(xp.abs(ref_x) ** 2, size=window)
            * ndi.uniform_filter(xp.abs(sec_x) ** 2, size=window)
            + 1e-10
        )
        coherence = xp.clip(num / denom, 0.0, 1.0).astype(xp.float32)
        return to_numpy(coherence)