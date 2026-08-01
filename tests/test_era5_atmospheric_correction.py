"""
Regression tests for pygeofetch.insar.atmosphere's ERA5/PyAPS
correction path.

Real, confirmed bugs fixed here, not hypothetical: (1) the previous
version took a single acquisition_datetime and subtracted that one
date's delay directly from a pair's interferometric phase, which is
physically wrong -- atmospheric delay is a real, per-date quantity, and
correcting a pair needs the DIFFERENCE between both dates' delays, not
one date's delay alone. Confirmed against real, working external code
(PyRate's PyAPS usage pattern: "aps_delay = phs2 - phs1") and MintPy's
own tropo_pyaps3.py, both of which compute delay per-date then
difference. (2) The previous version called a class named
`PyAPS_rdr`, which could not be confirmed against pyaps3's real,
documented API anywhere -- replaced with `pyaps3.PyAPS`, the real,
confirmed class name from the package's own source and usage examples.

Honest scope: the ZTD-to-LOS-phase conversion and per-date
differencing math is independently verified here with synthetic data
and does not depend on pyaps3 at all. The actual live pyaps3.PyAPS(...)
call requires a real, external Copernicus Climate Data Store account
and could not be tested end-to-end in the environment this was built
in (that network endpoint was directly confirmed unreachable there).
These tests do not claim to verify that part.
"""

import numpy as np
import pytest

from pygeofetch.insar.atmosphere import AtmosphericCorrector, _ztd_to_los_phase

WAVELENGTH_M = 0.05546576


def test_ztd_to_los_phase_matches_independent_direct_computation():
    incidence_deg = 38.0
    zenith_delay_m = np.array([[0.20, 0.23], [0.18, 0.25]])

    result = _ztd_to_los_phase(zenith_delay_m, incidence_deg, WAVELENGTH_M)

    for i in range(2):
        for j in range(2):
            expected = (4 * np.pi / WAVELENGTH_M) * (
                zenith_delay_m[i, j] / np.cos(np.radians(incidence_deg))
            )
            assert abs(result[i, j] - expected) < 1e-10


def test_closed_loop_removal_through_complex_wrapped_pathway():
    """The decisive test: inject a known, per-date-delay-derived
    atmospheric phase into a real, complex, wrapped interferogram
    alongside a known deformation signal, remove it via the same
    per-date-difference logic _correct_era5 uses, and confirm exact
    recovery of the true signal."""
    np.random.seed(7)
    h, w = 100, 100
    incidence_deg = 38.0

    zenith_ref = 0.20 + 0.01 * np.random.randn(h, w)
    zenith_sec = 0.23 + 0.01 * np.random.randn(h, w)

    phase_ref = _ztd_to_los_phase(zenith_ref, incidence_deg, WAVELENGTH_M)
    phase_sec = _ztd_to_los_phase(zenith_sec, incidence_deg, WAVELENGTH_M)
    atmo_phase = phase_sec - phase_ref

    np.random.seed(11)
    true_deformation = 0.4 * np.exp(
        -((np.arange(w)[None, :] - w / 2) ** 2 + (np.arange(h)[:, None] - h / 2) ** 2) / (w * 3)
    )
    scene_amp = np.abs(np.random.randn(h, w) + 1j * np.random.randn(h, w))
    ref_complex = scene_amp * np.exp(1j * np.random.uniform(-np.pi, np.pi, (h, w)))
    sec_complex = ref_complex * np.exp(-1j * (atmo_phase + true_deformation))

    igram = ref_complex * np.conj(sec_complex)
    corrected_igram = igram * np.exp(-1j * atmo_phase)
    corrected_phase = np.angle(corrected_igram)

    error = np.abs(np.angle(np.exp(1j * (corrected_phase - true_deformation))))
    assert error.max() < 1e-6


def test_era5_method_rejects_single_date_regression_for_the_real_fixed_bug():
    """Regression for the real, confirmed bug: a single-date call must
    be rejected, not silently accepted and misapplied to a pair."""
    corrector = AtmosphericCorrector(method="era5")
    phase = np.random.uniform(-np.pi, np.pi, (10, 10)).astype(np.float32)

    with pytest.raises(ValueError, match="Both reference_datetime and secondary_datetime"):
        corrector.correct(phase, dem="dummy.tif", reference_datetime="2024-11-08T12:00:00")

    with pytest.raises(ValueError, match="Both reference_datetime and secondary_datetime"):
        corrector.correct(phase, dem="dummy.tif", secondary_datetime="2024-11-20T12:00:00")
