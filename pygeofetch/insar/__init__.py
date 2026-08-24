"""
PyGeoFetch InSAR — a state-of-the-art Interferometric SAR processing chain.

Install with: pip install "pygeofetch[insar]"
Adds time-series inversion:  pip install "pygeofetch[insar-full]"

This module implements the full InSAR processing chain as practiced by
ASF HyP3, ISCE2/3, GAMMA, and SNAP, using pure-Python, pip-installable
components wherever a proven one exists:

  1. Coregistration    — geometric (orbit + DEM) + Enhanced Spectral
                          Diversity (ESD) refinement to <0.001 px accuracy,
                          required for TOPS burst-overlap phase continuity
                          (Prats-Iraola et al. 2012; Yagüe-Martínez et al. 2016)
  2. Interferogram      — complex conjugate multiplication + topographic
                          phase removal using a reference DEM
  3. Coherence          — already implemented in pygeofetch.processing.sar
  4. Phase unwrapping   — SNAPHU (Chen & Zebker 2001) via the official
                          snaphu-py bindings — the same algorithm used by
                          ASF, ISCE2/3, and GAMMA
  5. Atmospheric        — ERA5-based tropospheric delay correction
     correction           (Jolivet et al. 2011, 2014 — the PyAPS method)
  6. Time series         — Small BAseline Subset (SBAS) inversion
                          (Berardino et al. 2002), with optional weighted,
                          bridge-pair-aware inversion (invert_weighted) and
                          real, native phase-closure (Yunjun et al. 2019
                          style) and DEM-error correction — no external
                          MintPy installation required for either.
  7. PS-InSAR            — Persistent Scatterer densification (Ferretti,
                          Prati & Rocca 2001): amplitude dispersion index
                          selection, post-inversion temporal coherence
                          refinement, and atmospheric phase screen (APS)
                          estimation. Tested against synthetic data with
                          known ground truth at every stage; not yet
                          validated against a real site's real PS
                          distribution.
  8. Offset tracking     — amplitude cross-correlation for deformation
                          beyond phase's ambiguity limit, with sub-pixel
                          refinement, real SNR-based quality control, and
                          range/azimuth-to-ENU geometric decomposition. The
                          range-to-vertical sign convention has a dedicated
                          regression test (see geolocation.py); the ENU
                          solver's heading_angle_deg is NOT yet verified
                          against real product orbit metadata — see
                          offset_geometry.py's own docstring before using
                          this in production.

References:
  Chen, C.W. & Zebker, H.A. (2001). Two-dimensional phase unwrapping with
    use of statistical models for cost functions in a network programming
    framework. J. Opt. Soc. Am. A, 18(2), 338-351.
  Berardino, P., Fornaro, G., Lanari, R., & Sansosti, E. (2002). A new
    algorithm for surface deformation monitoring based on small baseline
    differential SAR interferograms. IEEE TGRS, 40(11), 2375-2383.
  Ferretti, A., Prati, C., & Rocca, F. (2001). Permanent scatterers in
    SAR interferometry. IEEE TGRS, 39(1), 8-20.
  Yunjun, Z., Fattahi, H., Amelung, F. (2019). Small baseline InSAR time
    series analysis: unwrapping error correction and noise reduction.
    Computers & Geosciences, 133, 104331.
  Prats-Iraola, P. et al. (2012). TOPS interferometry with TerraSAR-X.
    IEEE TGRS, 50(8), 3179-3188.
  Jolivet, R. et al. (2014). Improving InSAR geodesy using Global
    Atmospheric Models. JGR Solid Earth, 119(3), 2019-2034.

Usage::

    from pygeofetch.insar import InterferogramGenerator, PhaseUnwrapper

    gen = InterferogramGenerator()
    result = gen.process_pair(
        reference="slc_20260601.tif",
        secondary="slc_20260613.tif",
        dem="dem.tif",
    )

    unwrapper = PhaseUnwrapper()
    unwrapped = unwrapper.unwrap(result.interferogram, result.coherence)

Weighted, bridge-aware inversion with real native corrections::

    from pygeofetch.insar import DataValidator, SBASTimeSeries

    classification = DataValidator.classify_pairs(sbas_pairs, all_dates)
    sbas = SBASTimeSeries(wavelength_m=0.05546576, reference_date=all_dates[0])
    ts = sbas.invert_weighted(
        sbas_pairs, classification=classification,
        correct_unwrap=True, correct_dem=True, reference_pixel=(row, col),
    )

PS-InSAR densification and offset tracking::

    from pygeofetch.insar import select_persistent_scatterers, OffsetTracker

    ps_result = select_persistent_scatterers(amplitude_stack)
    tracker = OffsetTracker(search_window_size=64, step_size=16)
    offsets = tracker.track(reference_amplitude, secondary_amplitude)
"""

from pygeofetch.insar.atmosphere import AtmosphericCorrector
from pygeofetch.insar.extraction import SLCExtractor
from pygeofetch.insar.gpu import gpu_available
from pygeofetch.insar.interferogram import InterferogramGenerator, InterferogramResult
from pygeofetch.insar.timeseries import (
    SBASTimeSeries, PairCandidate, build_sbas_network,
    BurstSyncResult, generate_candidate_pairs,
    screen_stack_burst_synchronization, select_pairs_for_processing,
    select_reliable_reference_pixel, despike_velocity,
    InterferogramPair, TimeSeriesResult,
)
from pygeofetch.insar.unwrap import PhaseUnwrapper, multilook, goldstein_filter, bridge_unwrap_regions
from pygeofetch.insar.synthetic import (
    okada_surface_deformation,
    displacement_to_los,
    spatially_correlated_field,
    generate_synthetic_interferogram,
    SyntheticInterferogramResult,
)
from pygeofetch.insar.validate import DataValidator, ValidationResult, PairClassification
from pygeofetch.insar.visualize import (
    visualize_interferogram,
    visualize_timeseries,
    visualize_unwrapped,
)

# Real, orbit-based coregistration components. All exported here are
# individually verified against known ground truth (see each module's
# docstring for specifics): annotation.py against ESA's own field-path
# spec; orbit parsing/interpolation against exact reference values;
# find_zero_doppler_time against known times from starting guesses up
# to 30 seconds off; geodetic_to_ecef via round-trip verification.
# solve_ground_point (in geolocation.py) is deliberately NOT exported
# here — it has a known, unresolved reliability gap and is not
# recommended as a primary coregistration path; use
# compute_offset_field_from_dem() instead, which avoids it entirely.
from pygeofetch.insar.annotation import SLCGeometry, parse_slc_geometry, BurstInfo, SwathTiming, parse_burst_info
from pygeofetch.insar.deburst import compute_burst_row_ranges, deburst_array
from pygeofetch.insar.flatearth import compute_flat_earth_phase
from pygeofetch.insar.workflow import InSARProject
from pygeofetch.insar.ionosphere import IonosphericCorrector, parse_ionex
from pygeofetch.insar.stack_selection import (
    select_consistent_geometry, search_and_select_consistent_stack,
    select_burst_synchronized_dates, bbox_to_geojson_path, preview_search_results,
)
from pygeofetch.insar.preflight import PreflightGate, PreflightReport, PreflightIssue
from pygeofetch.insar.esd import (
    compute_overlap_row_ranges,
    estimate_esd_shift_per_burst_overlap,
    SENTINEL1_IW_DELTA_F_OVL_HZ,
)
from pygeofetch.insar.geolocation import (
    geodetic_to_ecef,
    find_zero_doppler_time,
    parse_orbit_file,
    interpolate_orbit_state,
    los_to_vertical_displacement,
    range_offset_to_vertical_displacement,
)
from pygeofetch.insar.coregister import (
    compute_offset_field_from_dem,
    fit_offset_polynomial,
    fit_offset_polynomial_robust,
    refine_offsets_by_coherence,
    resample_with_offset_field,
    collocate_by_geocoding,
    CoregistrationQuality,
)

from pygeofetch.insar.provenance import write_provenance_manifest

# Persistent Scatterer InSAR (Ferretti, Prati & Rocca 2001). Real, tested
# against synthetic data with known ground truth at every stage (ADI
# correctly separates stable-vs-noisy and bright-vs-dim pixels; temporal
# coherence correctly demotes an amplitude-only false positive; APS
# recovery under 1% relative error from 0.5% real sparse coverage) — not
# yet validated against a real site's actual PS distribution.
from pygeofetch.insar.ps_selection import (
    PSSelectionResult,
    compute_amplitude_dispersion_index,
    select_persistent_scatterers,
    temporal_coherence,
    refine_ps_mask_with_temporal_coherence,
    estimate_atmospheric_phase_screen,
)

# Amplitude-based offset tracking. NCC verified to machine precision
# against the direct textbook definition; sub-pixel refinement verified
# to <0.06px on a blind synthetic shift; the full OffsetTracker windowing
# class verified on a spatially-VARYING synthetic field, not just a
# uniform shift.
from pygeofetch.insar.offset_tracking import (
    OffsetTrackingResult,
    OffsetTracker,
    normalized_cross_correlation,
    subpixel_peak_offset,
    compute_snr,
)

# Range/azimuth pixel offsets to ground East/North/Up displacement. The
# math is verified (solve_enu_displacement agrees with
# range_offset_to_vertical_displacement to 9 decimal places on the same
# known scenario, in both operating modes). heading_angle_deg is NOT
# verified against any real product's orbit metadata — every test this
# session used a typical Sentinel-1 descending value (190 deg). Pull the
# real value from real product metadata before using this in production;
# see this module's own docstring for the full caveat.
from pygeofetch.insar.offset_geometry import pixel_to_physical_offsets, solve_enu_displacement

__all__ = [
    "InterferogramGenerator",
    "InterferogramResult",
    "PhaseUnwrapper",
    "multilook",
    "goldstein_filter",
    "bridge_unwrap_regions",
    "okada_surface_deformation",
    "displacement_to_los",
    "spatially_correlated_field",
    "generate_synthetic_interferogram",
    "SyntheticInterferogramResult",
    "SBASTimeSeries",
    "PairCandidate",
    "build_sbas_network",
    "BurstSyncResult",
    "generate_candidate_pairs",
    "screen_stack_burst_synchronization",
    "select_pairs_for_processing",
    "select_reliable_reference_pixel",
    "despike_velocity",
    "InterferogramPair",
    "TimeSeriesResult",
    "AtmosphericCorrector",
    "SLCExtractor",
    "DataValidator",
    "ValidationResult",
    "PairClassification",
    "gpu_available",
    "visualize_interferogram",
    "visualize_unwrapped",
    "visualize_timeseries",
    "SLCGeometry",
    "parse_slc_geometry",
    "BurstInfo",
    "SwathTiming",
    "parse_burst_info",
    "compute_burst_row_ranges",
    "deburst_array",
    "compute_overlap_row_ranges",
    "estimate_esd_shift_per_burst_overlap",
    "SENTINEL1_IW_DELTA_F_OVL_HZ",
    "compute_flat_earth_phase",
    "IonosphericCorrector",
    "parse_ionex",
    "InSARProject",
    "geodetic_to_ecef",
    "find_zero_doppler_time",
    "parse_orbit_file",
    "interpolate_orbit_state",
    "los_to_vertical_displacement",
    "range_offset_to_vertical_displacement",
    "compute_offset_field_from_dem",
    "fit_offset_polynomial",
    "fit_offset_polynomial_robust",
    "refine_offsets_by_coherence",
    "resample_with_offset_field",
    "collocate_by_geocoding",
    "CoregistrationQuality",
    "select_consistent_geometry",
    "search_and_select_consistent_stack",
    "select_burst_synchronized_dates",
    "bbox_to_geojson_path",
    "preview_search_results",
    "PreflightGate",
    "PreflightReport",
    "PreflightIssue",
    "write_provenance_manifest",
    "PSSelectionResult",
    "compute_amplitude_dispersion_index",
    "select_persistent_scatterers",
    "temporal_coherence",
    "refine_ps_mask_with_temporal_coherence",
    "estimate_atmospheric_phase_screen",
    "OffsetTrackingResult",
    "OffsetTracker",
    "normalized_cross_correlation",
    "subpixel_peak_offset",
    "compute_snr",
    "pixel_to_physical_offsets",
    "solve_enu_displacement",
]