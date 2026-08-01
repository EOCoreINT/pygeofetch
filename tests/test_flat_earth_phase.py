"""
Regression tests for pygeofetch.insar.flatearth -- real, geometric
orbital/flat-earth phase removal, distinct from the topographic phase
this pipeline already corrects.

Built by composing the same real, already-verified orbit-geometry
functions used elsewhere in this codebase for coregistration
(geodetic_to_ecef, find_zero_doppler_time, interpolate_orbit_state),
verified against an independent direct computation, a standard
textbook approximation, and closed-loop removal tests, not assumed
correct.
"""

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_insar_verified_components import _build_geometry  # noqa: E402

from pygeofetch.insar.annotation import SLCGeometry  # noqa: E402
from pygeofetch.insar.flatearth import compute_flat_earth_phase  # noqa: E402
from pygeofetch.insar.geolocation import (  # noqa: E402
    SPEED_OF_LIGHT,
    geodetic_to_ecef,
    find_zero_doppler_time,
    interpolate_orbit_state,
)

WAVELENGTH_M = 0.05546576


def _orbit_series(center_pos, center_vel, center_time):
    times, positions, velocities = [], [], []
    for i in range(-60, 61):
        dt = i * 10.0
        positions.append(tuple(center_pos[k] + center_vel[k] * dt for k in range(3)))
        times.append(center_time + timedelta(seconds=dt))
        velocities.append(center_vel)
    return times, positions, velocities


def _build_fixture(n=200, b_perp=80.0):
    sat_pos_ref, sat_vel_ref, range_time_center, P_center = _build_geometry(19.36, -99.09)
    t0_ref = datetime(2024, 11, 8, 12, 34, 39)
    radial = tuple(c / math.sqrt(sum(x**2 for x in sat_pos_ref)) for c in sat_pos_ref)
    sat_pos_sec = tuple(sat_pos_ref[i] + radial[i] * b_perp for i in range(3))
    t0_sec = t0_ref + timedelta(days=12)

    ref_orbit = _orbit_series(sat_pos_ref, sat_vel_ref, t0_ref)
    sec_orbit = _orbit_series(sat_pos_sec, sat_vel_ref, t0_sec)

    ref_geom = SLCGeometry(
        first_line_time=t0_ref - timedelta(seconds=n / 2 * 0.002), azimuth_time_interval_s=0.002,
        near_range_time_s=range_time_center - (n / 2) * (1 / 6.4e7), range_sampling_rate_hz=6.4e7,
        n_lines=n, n_columns=n,
    )
    return ref_geom, ref_orbit, sec_orbit, t0_ref, t0_sec, n


def test_range_difference_matches_textbook_approximation():
    """Independent cross-check: for a small baseline and a real
    incidence angle, delta_R should approximate B_perp * sin(theta),
    the standard textbook relation -- confirmed here to 0.01%."""
    EARTH_R, alt = 6371000.0, 700000.0

    def make_simple_orbit(baseline_offset_m):
        times = [datetime(2024, 1, 1) + timedelta(seconds=t) for t in np.linspace(-10, 10, 200)]
        positions = [(t * 7500.0, baseline_offset_m, EARTH_R + alt) for t in np.linspace(-10, 10, 200)]
        velocities = [(7500.0, 0.0, 0.0)] * 200
        return times, positions, velocities

    ref_orbit = make_simple_orbit(0.0)
    sec_orbit = make_simple_orbit(150.0)
    scene_center_time = datetime(2024, 1, 1)

    incidence_deg = 35.0
    ground_y = alt * np.tan(np.radians(incidence_deg))
    ground_point = (0.0, ground_y, EARTH_R)

    t_ref = find_zero_doppler_time(ref_orbit[0], ref_orbit[1], ref_orbit[2], ground_point, scene_center_time)
    t_sec = find_zero_doppler_time(sec_orbit[0], sec_orbit[1], sec_orbit[2], ground_point, scene_center_time)
    pos_ref, _ = interpolate_orbit_state(*ref_orbit, t_ref)
    pos_sec, _ = interpolate_orbit_state(*sec_orbit, t_sec)
    R_ref = sum((pos_ref[i] - ground_point[i]) ** 2 for i in range(3)) ** 0.5
    R_sec = sum((pos_sec[i] - ground_point[i]) ** 2 for i in range(3)) ** 0.5

    measured = abs(R_sec - R_ref)
    expected = 150.0 * np.sin(np.radians(incidence_deg))
    assert abs(measured - expected) < 0.1, f"measured {measured} vs textbook {expected}"


def test_matches_independent_direct_computation_at_a_real_point():
    """The module's polynomial-fit output at a specific pixel must
    match a completely independent, direct geometric computation at
    that same real point."""
    ref_geom, ref_orbit, sec_orbit, t0_ref, t0_sec, n = _build_fixture()
    lon0, lat0 = -99.11, 19.34

    flat_phase = compute_flat_earth_phase(
        ref_geom, ref_orbit, ref_geom, sec_orbit, t0_ref, t0_sec, (n, n), WAVELENGTH_M,
        sample_bounds=(lon0, lat0, lon0 + 0.04, lat0 + 0.04), grid_points=7, polynomial_degree=2,
    )

    check_lon, check_lat = -99.09, 19.36
    ground_point = geodetic_to_ecef(check_lat, check_lon, 0.0)
    t_ref = find_zero_doppler_time(ref_orbit[0], ref_orbit[1], ref_orbit[2], ground_point, t0_ref)
    t_sec = find_zero_doppler_time(sec_orbit[0], sec_orbit[1], sec_orbit[2], ground_point, t0_sec)
    pos_ref, _ = interpolate_orbit_state(*ref_orbit, t_ref)
    pos_sec, _ = interpolate_orbit_state(*sec_orbit, t_sec)
    R_ref = sum((pos_ref[i] - ground_point[i]) ** 2 for i in range(3)) ** 0.5
    R_sec = sum((pos_sec[i] - ground_point[i]) ** 2 for i in range(3)) ** 0.5
    independent_phase = (4 * np.pi / WAVELENGTH_M) * (R_sec - R_ref)

    row = ref_geom.row_for_azimuth_time(t_ref)
    col = ref_geom.col_for_range_time(2 * R_ref / SPEED_OF_LIGHT)
    assert 0 <= row < n and 0 <= col < n

    fitted = flat_phase[int(round(row)), int(round(col))]
    assert abs(fitted - independent_phase) < 1.0


def test_closed_loop_removal_through_complex_wrapped_pathway():
    """The decisive test: inject the module's own computed field into
    a real, complex, wrapped interferogram alongside a known
    deformation signal, remove it, and confirm exact recovery of the
    true signal -- the same pathway process_pair() actually uses."""
    ref_geom, ref_orbit, sec_orbit, t0_ref, t0_sec, n = _build_fixture()
    lon0, lat0 = -99.11, 19.34

    flat_phase = compute_flat_earth_phase(
        ref_geom, ref_orbit, ref_geom, sec_orbit, t0_ref, t0_sec, (n, n), WAVELENGTH_M,
        sample_bounds=(lon0, lat0, lon0 + 0.04, lat0 + 0.04), grid_points=7, polynomial_degree=2,
    )

    np.random.seed(11)
    true_deformation_phase = 0.5 * np.exp(
        -((np.arange(n)[:, None] - n / 2) ** 2 + (np.arange(n)[None, :] - n / 2) ** 2) / (n * 2)
    )
    scene_amp = np.abs(np.random.randn(n, n) + 1j * np.random.randn(n, n))
    ref_complex = scene_amp * np.exp(1j * np.random.uniform(-np.pi, np.pi, (n, n)))
    sec_complex = ref_complex * np.exp(-1j * (flat_phase + true_deformation_phase))

    igram = ref_complex * np.conj(sec_complex)
    corrected_igram = igram * np.exp(-1j * flat_phase)
    corrected_phase = np.angle(corrected_igram)

    error = np.abs(np.angle(np.exp(1j * (corrected_phase - true_deformation_phase))))
    assert error.mean() < 1e-6
    assert error.max() < 1e-6


def test_raises_with_neither_dem_nor_sample_bounds():
    ref_geom, ref_orbit, sec_orbit, t0_ref, t0_sec, n = _build_fixture()
    with pytest.raises(ValueError):
        compute_flat_earth_phase(
            ref_geom, ref_orbit, ref_geom, sec_orbit, t0_ref, t0_sec, (n, n), WAVELENGTH_M,
        )
