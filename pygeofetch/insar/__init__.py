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
                          (Berardino et al. 2002; Yunjun et al. 2019 — MintPy)

References:
  Chen, C.W. & Zebker, H.A. (2001). Two-dimensional phase unwrapping with
    use of statistical models for cost functions in a network programming
    framework. J. Opt. Soc. Am. A, 18(2), 338-351.
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
"""

from pygeofetch.insar.atmosphere import AtmosphericCorrector
from pygeofetch.insar.extraction import SLCExtractor
from pygeofetch.insar.gpu import gpu_available
from pygeofetch.insar.interferogram import InterferogramGenerator, InterferogramResult
from pygeofetch.insar.timeseries import SBASTimeSeries
from pygeofetch.insar.unwrap import PhaseUnwrapper, multilook, goldstein_filter, bridge_unwrap_regions
from pygeofetch.insar.validate import DataValidator, ValidationResult
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
from pygeofetch.insar.annotation import SLCGeometry, parse_slc_geometry
from pygeofetch.insar.geolocation import (
    geodetic_to_ecef,
    find_zero_doppler_time,
    parse_orbit_file,
    interpolate_orbit_state,
    los_to_vertical_displacement,
)
from pygeofetch.insar.coregister import (
    compute_offset_field_from_dem,
    fit_offset_polynomial,
    resample_with_offset_field,
)

__all__ = [
    "InterferogramGenerator",
    "InterferogramResult",
    "PhaseUnwrapper",
    "multilook",
    "goldstein_filter",
    "bridge_unwrap_regions",
    "SBASTimeSeries",
    "AtmosphericCorrector",
    "SLCExtractor",
    "DataValidator",
    "ValidationResult",
    "gpu_available",
    "visualize_interferogram",
    "visualize_unwrapped",
    "visualize_timeseries",
    "SLCGeometry",
    "parse_slc_geometry",
    "geodetic_to_ecef",
    "find_zero_doppler_time",
    "parse_orbit_file",
    "interpolate_orbit_state",
    "los_to_vertical_displacement",
    "compute_offset_field_from_dem",
    "fit_offset_polynomial",
    "resample_with_offset_field",
]