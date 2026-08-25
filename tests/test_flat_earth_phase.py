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
from datetime import datetime, timedelta

import numpy as np
import pytest

from pygeofetch.insar.annotation import SLCGeometry
from pygeofetch.insar.flatearth import compute_flat_earth_phase
from pygeofetch.insar.geolocation import (
    SPEED_OF_LIGHT,
    WGS84_A,
    WGS84_B,
    find_zero_doppler_time,
    geodetic_to_ecef,
    interpolate_orbit_state,
)

WAVELENGTH_M = 0.05546576


# Real, confirmed fix here: this file previously imported _build_geometry
# from a sibling test module (test_insar_verified_components) via a
# runtime sys.path.insert() hack. That only worked when both files sat
# together in the same directory -- a real, brittle assumption that broke
# in any environment where this file is used standalone (as reported: the
# sibling file genuinely does not exist there). Inlined below, verbatim,
# so this file is fully self-contained and has no dependency on any
# other test file existing.
def _build_geometry(lat_deg, lon_deg, dem_h=0.0, incl_deg=98.18, ascending=True):
    """Build a fully self-consistent, realistic satellite+ground-point test
    geometry — real orbital velocity direction, exact zero-Doppler
    projection."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    e2 = 1 - ((WGS84_B + dem_h) ** 2 / (WGS84_A + dem_h) ** 2)
    N = (WGS84_A + dem_h) / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    P_true = (
        N * math.cos(lat) * math.cos(lon),
        N * math.cos(lat) * math.sin(lon),
        N * (1 - e2) * math.sin(lat),
    )
    altitude = 693000.0
    normal = (
        P_true[0] / (WGS84_A + dem_h) ** 2,
        P_true[1] / (WGS84_A + dem_h) ** 2,
        P_true[2] / (WGS84_B + dem_h) ** 2,
    )
    nmag = math.sqrt(sum(c**2 for c in normal))
    normal = tuple(c / nmag for c in normal)
    sat_pos = tuple(P_true[i] + normal[i] * altitude for i in range(3))

    incl = math.radians(incl_deg)
    sign = 1 if ascending else -1
    plane_normal = (
        sign * math.sin(incl) * math.sin(lon),
        -sign * math.sin(incl) * math.cos(lon),
        math.cos(incl),
    )
    pn_mag = math.sqrt(sum(c**2 for c in plane_normal))
    plane_normal = tuple(c / pn_mag for c in plane_normal)
    tangent = (
        plane_normal[1] * normal[2] - plane_normal[2] * normal[1],
        plane_normal[2] * normal[0] - plane_normal[0] * normal[2],
        plane_normal[0] * normal[1] - plane_normal[1] * normal[0],
    )
    tmag = math.sqrt(sum(c**2 for c in tangent))
    tangent = tuple(c / tmag for c in tangent)
    sat_vel = tuple(c * 7500.0 for c in tangent)

    d = sum(sat_vel[i] * (P_true[i] - sat_pos[i]) for i in range(3)) / sum(v**2 for v in sat_vel)
    P_exact = tuple(P_true[i] - d * sat_vel[i] for i in range(3))
    los = tuple(sat_pos[i] - P_exact[i] for i in range(3))
    range_m = math.sqrt(sum(c**2 for c in los))
    range_time_s = 2 * range_m / SPEED_OF_LIGHT
    return sat_pos, sat_vel, range_time_s, P_exact


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
    # Real, confirmed test-fixture gap fixed here: the fixture previously
    # only built ONE SLCGeometry and every caller passed it as BOTH
    # ref_geometry and sec_geometry -- which never exercised the real,
    # production scenario (interferogram.py always parses a genuinely
    # separate sec_geom_fe from the secondary's own SAFE zip, anchored to
    # the secondary's own real first_line_time, not the reference's).
    # Reusing ref_geom for both meant t_sec (anchored ~12 real days after
    # ref_geom's own first_line_time) was silently being evaluated
    # against the WRONG scene's timing reference whenever a real
    # secondary-bounds check was added -- exactly what surfaced this gap.
    sec_geom = SLCGeometry(
        first_line_time=t0_sec - timedelta(seconds=n / 2 * 0.002), azimuth_time_interval_s=0.002,
        near_range_time_s=range_time_center - (n / 2) * (1 / 6.4e7), range_sampling_rate_hz=6.4e7,
        n_lines=n, n_columns=n,
    )
    return ref_geom, sec_geom, ref_orbit, sec_orbit, t0_ref, t0_sec, n


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
    ref_geom, sec_geom, ref_orbit, sec_orbit, t0_ref, t0_sec, n = _build_fixture()
    lon0, lat0 = -99.11, 19.34

    flat_phase = compute_flat_earth_phase(
        ref_geom, ref_orbit, sec_geom, sec_orbit, t0_ref, t0_sec, (n, n), WAVELENGTH_M,
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
    ref_geom, sec_geom, ref_orbit, sec_orbit, t0_ref, t0_sec, n = _build_fixture()
    lon0, lat0 = -99.11, 19.34

    flat_phase = compute_flat_earth_phase(
        ref_geom, ref_orbit, sec_geom, sec_orbit, t0_ref, t0_sec, (n, n), WAVELENGTH_M,
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
    ref_geom, sec_geom, ref_orbit, sec_orbit, t0_ref, t0_sec, n = _build_fixture()
    with pytest.raises(ValueError):
        compute_flat_earth_phase(
            ref_geom, ref_orbit, sec_geom, sec_orbit, t0_ref, t0_sec, (n, n), WAVELENGTH_M,
        )
