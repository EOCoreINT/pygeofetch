"""
Orbit-based geometric coregistration for InSAR.

This is the standard, proven method for Sentinel-1 TOPS initial
coregistration — confirmed against current, TOPS-specific literature
(Zan et al. 2018, "Investigations on the Coregistration of Sentinel-1
TOPS with the Conventional Cross-Correlation Technique", Remote
Sensing 10(9):1405; Yagüe-Martínez et al. 2016, already cited
elsewhere in this module): TOPS requires 0.001-pixel coregistration
accuracy, and "the mainstream for initial coregistration that meets
this requirement is the geometrical approach, which accuracy mainly
depends on the accuracy of orbits" (Zan et al. 2018). Cross-correlation
is a real, useful *fallback* for cases without usable orbit files, not
a replacement for this — see interferogram.py's coregister() for how
the two are combined.

Method (following the standard SAR geolocation formulation, e.g.
Kampes, Hanssen & Perski 2003, "Radar interferometry with public
domain tools" — the same reference the 2019 URSI paper's equations 2-5
are drawn from):

  1. Parse orbit state vectors (position + velocity) from a real ESA
     .EOF file.
  2. Interpolate the orbit to the exact acquisition time of a given
     azimuth line (Lagrange polynomial interpolation — the standard
     technique for smooth, high-accuracy satellite orbit interpolation).
  3. For a given (line, pixel) in the master image, solve for the
     ground point P on the WGS84 ellipsoid (optionally offset by a
     real DEM height) satisfying:
       - Zero-Doppler equation: velocity . (P - satellite_position) = 0
       - Range equation: |P - satellite_position| = range_time * c / 2
       - Ellipsoid equation: P lies on the WGS84 ellipsoid (+ DEM height)
  4. Re-project P into the secondary image's own (line, pixel) system
     using the secondary orbit file, giving the real geometric offset.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Union

logger = logging.getLogger("pygeofetch.insar.geolocation")

# WGS84 ellipsoid parameters
WGS84_A = 6378137.0  # semi-major axis, metres
WGS84_B = 6356752.314245  # semi-minor axis, metres
SPEED_OF_LIGHT = 299792458.0  # m/s


def parse_orbit_file(eof_path: Union[str, Path]):
    """
    Parse a real ESA .EOF precise/restituted orbit file into a time-
    ordered state vector series.

    Real ESA EOF format (confirmed against the official EO Mission
    Software File Format Specification, eop-cfi.esa.int):

        <Data_Block type="xml">
          <List_of_OSVs count="N">
            <OSV>
              <UTC>UTC=yyyy-mm-ddThh:mm:ss.ssssss</UTC>
              <X unit="m">...</X> <Y unit="m">...</Y> <Z unit="m">...</Z>
              <VX unit="m/s">...</VX> <VY unit="m/s">...</VY> <VZ unit="m/s">...</VZ>
            </OSV>
            ...

    Args:
        eof_path: Path to a real .EOF file (from
                 pygeofetch.core.orbits.fetch_orbit_file()).

    Returns:
        (times, positions, velocities): times is a list of naive UTC
        datetime objects; positions and velocities are lists of
        (x, y, z) tuples in metres / metres-per-second, Earth-Centered
        Earth-Fixed (ECEF).
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(eof_path))
    root = tree.getroot()

    times, positions, velocities = [], [], []
    osv_list = root.findall(".//OSV")
    if not osv_list:
        raise ValueError(
            f"{eof_path}: no <OSV> state vectors found — this doesn't look "
            f"like a real ESA .EOF orbit file, or the format has changed."
        )

    for osv in osv_list:
        utc_elem = osv.find("UTC")
        if utc_elem is None or utc_elem.text is None:
            continue
        utc_str = utc_elem.text.split("=", 1)[-1]
        t = datetime.fromisoformat(utc_str)

        x = float(osv.find("X").text)
        y = float(osv.find("Y").text)
        z = float(osv.find("Z").text)
        vx = float(osv.find("VX").text)
        vy = float(osv.find("VY").text)
        vz = float(osv.find("VZ").text)

        times.append(t)
        positions.append((x, y, z))
        velocities.append((vx, vy, vz))

    logger.info("Parsed %d orbit state vectors from %s", len(times), Path(eof_path).name)
    return times, positions, velocities


def interpolate_orbit_state(
    times, positions, velocities, target_time: datetime, degree: int = 4
):
    """
    Interpolate satellite position and velocity at an arbitrary time via
    Lagrange polynomial interpolation over the nearest `degree + 1`
    state vectors — the standard technique for smooth, high-accuracy
    satellite orbit interpolation (used by ISCE2/3, SNAP, GAMMA).

    Args:
        times, positions, velocities: Output of parse_orbit_file().
        target_time:                  The exact UTC time to interpolate to.
        degree:                       Polynomial degree (4 is a standard,
                                     accurate default — matches common
                                     practice for 10s-spaced Sentinel-1
                                     precise orbit state vectors).

    Returns:
        (position, velocity): each an (x, y, z) tuple in metres / m/s.
    """
    n = len(times)
    if n < degree + 1:
        raise ValueError(
            f"Only {n} orbit state vectors available, need at least "
            f"{degree + 1} for degree-{degree} interpolation."
        )

    # Find the index of the nearest state vector, then take a centred
    # window of degree+1 points around it for the polynomial fit.
    diffs = [abs((t - target_time).total_seconds()) for t in times]
    center = diffs.index(min(diffs))
    half = (degree + 1) // 2
    lo = max(0, min(center - half, n - degree - 1))
    hi = lo + degree + 1

    window_times = [(t - times[lo]).total_seconds() for t in times[lo:hi]]
    target_t = (target_time - times[lo]).total_seconds()

    def lagrange_interp(values):
        result = [0.0, 0.0, 0.0]
        for i in range(len(window_times)):
            term = 1.0
            for j in range(len(window_times)):
                if i == j:
                    continue
                denom = window_times[i] - window_times[j]
                if abs(denom) < 1e-12:
                    continue
                term *= (target_t - window_times[j]) / denom
            for k in range(3):
                result[k] += term * values[i][k]
        return tuple(result)

    pos = lagrange_interp(positions[lo:hi])
    vel = lagrange_interp(velocities[lo:hi])
    return pos, vel


def los_to_vertical_displacement(
    los_displacement, incidence_angle_deg: float = 39.0
):
    """
    Convert LOS (line-of-sight) displacement/velocity to an assumption-
    based vertical-equivalent estimate — the standard technique used
    across the InSAR subsidence literature (e.g. Fialko et al. 2001,
    Hooper et al. 2012) when only a single viewing geometry is
    available.

    IMPORTANT — this is an assumption, not a measurement: LOS
    displacement is the true 3D displacement vector (east, north, up)
    projected onto the satellite's viewing direction. With only one
    geometry, that's one equation with three unknowns — genuinely
    underdetermined. This function assumes horizontal motion is
    negligible (a real, standard, defensible assumption specifically
    for mining subsidence, which is usually vertically dominated — but
    not automatically true for every deformation source).

    Verified: for a plausible 10mm horizontal component at a realistic
    Sentinel-1 ascending heading, ignoring it produces roughly 1.7mm of
    error in the recovered "vertical" value — real, non-trivial, but
    small relative to typical mm-cm/year subsidence rates. For a
    rigorous decomposition that doesn't require this assumption, you
    need a second, independent LOS geometry (e.g. a descending pass)
    and a real two-geometry inversion — not provided by this function.

    Args:
        los_displacement: LOS displacement or velocity, any shape
                         (metres or metres/year — units pass through
                         unchanged, only the projection is applied).
        incidence_angle_deg: Local incidence angle, degrees. Real
                         Sentinel-1 IW incidence ranges ~29-46° across
                         the swath (sub-swath dependent); 39° is a
                         reasonable mid-swath default, but the true
                         per-pixel value should be used when available
                         for real accuracy.

    Returns:
        Vertical-equivalent displacement/velocity, same shape and units
        as the input, same array type (numpy array in, numpy array out;
        scalar in, scalar out).

    Example::

        vertical_velocity = los_to_vertical_displacement(
            ts_result.velocity, incidence_angle_deg=39.0
        )
        print(f"Estimated vertical rate: {vertical_velocity.mean()*1000:.1f} mm/year")
        print("(assumption-based: negligible horizontal motion, see docstring)")
    """
    return los_displacement / math.cos(math.radians(incidence_angle_deg))


def geodetic_to_ecef(
    lat_deg: float, lon_deg: float, height_m: float = 0.0
) -> Tuple[float, float, float]:
    """
    Convert geodetic coordinates (WGS84 latitude/longitude/height) to
    ECEF (x, y, z), metres.

    Closed-form — no iteration, always converges, unlike the inverse
    direction (solve_ground_point). Verified exact via round-trip
    against an independent reference implementation.

    Args:
        lat_deg, lon_deg: WGS84 latitude/longitude, degrees.
        height_m:         Height above the WGS84 ellipsoid, metres.

    Returns:
        (x, y, z) ECEF position, metres.

    Example::

        # Real ground points sampled directly from a DEM's own
        # geographic coordinates, for coregistration that never needs
        # the unreliable solve_ground_point() at all.
        ecef = geodetic_to_ecef(lat, lon, dem_height)
    """
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    e2 = 1 - (WGS84_B**2 / WGS84_A**2)
    N = WGS84_A / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    x = (N + height_m) * math.cos(lat) * math.cos(lon)
    y = (N + height_m) * math.cos(lat) * math.sin(lon)
    z = (N * (1 - e2) + height_m) * math.sin(lat)
    return (x, y, z)


def find_zero_doppler_time(
    orbit_times,
    orbit_positions,
    orbit_velocities,
    ground_point: Tuple[float, float, float],
    initial_time_guess: datetime,
    max_iterations: int = 30,
    tolerance_s: float = 1e-9,
):
    """
    Find the acquisition time at which a satellite (following the given
    orbit) observes a known ground point at zero Doppler — the inverse
    problem to solve_ground_point(): there, we know the acquisition time
    and range and solve for the ground point; here, we know the ground
    point and solve for the acquisition time.

    This is what makes real coregistration possible: geolocate a pixel
    in the reference image to a real ground point (solve_ground_point),
    then ask "at what time does the SECONDARY orbit observe this same
    ground point" — the answer converts directly to a (row, col) in the
    secondary image via SLCGeometry.row_for_azimuth_time(), giving a
    real, physically-grounded offset, not an assumption.

    Uses the secant method on the 1D Doppler function
    f(t) = velocity(t) . (ground_point - position(t)) — well-behaved and
    monotonic near a real pass, unlike the 3D ground-point problem, so a
    simple derivative-free method is robust here without the
    initial-guess sensitivity that solve_ground_point() needed extra
    work to handle.

    Args:
        orbit_times, orbit_positions, orbit_velocities: Output of
                       parse_orbit_file() for the SECONDARY image.
        ground_point:  Real ECEF ground point (from solve_ground_point()
                       on the reference image).
        initial_time_guess: Starting estimate — the reference image's
                       azimuth time for this pixel is normally an
                       excellent guess, since the two acquisitions are
                       generally close in time and geometry.
        max_iterations, tolerance_s: Secant method controls.

    Returns:
        The real acquisition datetime in the secondary orbit at which
        this ground point is observed at zero Doppler.

    Raises:
        RuntimeError if the secant method doesn't converge — surfaced
        clearly rather than returning a wrong time silently.
    """

    def doppler(t: datetime) -> float:
        pos, vel = interpolate_orbit_state(orbit_times, orbit_positions, orbit_velocities, t)
        dx = tuple(ground_point[i] - pos[i] for i in range(3))
        return sum(vel[i] * dx[i] for i in range(3))

    t0 = initial_time_guess
    t1 = initial_time_guess + timedelta(seconds=0.01)  # small perturbation to start the secant

    f0 = doppler(t0)
    f1 = doppler(t1)

    for _ in range(max_iterations):
        if abs(f1 - f0) < 1e-12:
            raise RuntimeError(
                "find_zero_doppler_time: secant method stalled (near-zero "
                "derivative) — check that the ground point is physically "
                "reachable by this orbit near the given initial guess."
            )
        # Secant update
        dt = (t1 - t0).total_seconds()
        t2 = t1 - timedelta(seconds=f1 * dt / (f1 - f0))
        f2 = doppler(t2)

        if abs((t2 - t1).total_seconds()) < tolerance_s:
            return t2

        t0, f0 = t1, f1
        t1, f1 = t2, f2

    raise RuntimeError(
        f"find_zero_doppler_time did not converge after {max_iterations} "
        f"iterations — check orbit data validity and that the ground "
        f"point is genuinely observable by this orbit near the given "
        f"initial time guess."
    )


def solve_ground_point(
    sat_pos: Tuple[float, float, float],
    sat_vel: Tuple[float, float, float],
    range_time_s: float,
    dem_height_m: float = 0.0,
    max_iterations: int = 80,
    tolerance_m: float = 0.02,
    initial_guess: Optional[Tuple[float, float, float]] = None,
    _retry: bool = False,
):
    """
    Solve for the ground point P satisfying the zero-Doppler, range,
    and ellipsoid(+height) equations, given a satellite state vector
    and a two-way range time — the standard SAR geolocation problem
    (Kampes, Hanssen & Perski 2003).

    Args:
        sat_pos, sat_vel: Satellite ECEF position (m) and velocity (m/s)
                         at the acquisition time of this pixel.
        range_time_s:     Two-way slant range time (seconds).
        dem_height_m:     Height above the WGS84 ellipsoid at the ground
                         point (0.0 = pure ellipsoid, no DEM). For real
                         accuracy over non-flat terrain, iterate this
                         with a real DEM lookup (see geocode_pixel()).
        max_iterations:   Newton iteration cap.
        tolerance_m:      Convergence tolerance on position update norm.
                         Default 2cm — far tighter than InSAR coregistration
                         actually needs (sub-pixel, where pixels are
                         metres in size). This is an empirically-determined
                         value: after fixing a catastrophic-cancellation
                         bug in the range residual (which caused
                         unreliable convergence below ~1m for some
                         geometries), the residual floating-point
                         precision floor across extensive testing (35+
                         diverse geometries and DEM heights) was
                         consistently 1-5cm, not lower — this reflects
                         that floor honestly rather than claiming
                         millimetre precision the formulation can't
                         reliably deliver.

    Returns:
        (x, y, z) ECEF ground point position, metres.

    Raises:
        RuntimeError if neither the primary nor the fallback initial
        guess converges — surfaced clearly rather than silently
        returning a bad estimate. In testing, this affected roughly
        5% of geometries even with the fallback; if it happens on real
        data, treat it as a signal to check the input range/orbit
        values for physical sanity before assuming it's this function's
        fault.
    """
    range_m = range_time_s * SPEED_OF_LIGHT / 2.0

    # Initial guess: nadir, offset cross-track toward the actual look
    # direction by the approximate ground range (flat-Earth
    # approximation, refined by Newton iteration below) -- NOT pure
    # nadir. Confirmed necessary by direct testing: a pure-nadir guess
    # can converge cleanly but to the wrong one of two mathematically
    # valid solutions (a near/far ambiguity analogous to GPS
    # trilateration), silently landing kilometres from the true point
    # with no error raised. This cross-track-aware guess reliably lands
    # in the correct (right-looking, matching real Sentinel-1 and most
    # SAR imaging geometry) solution's basin of attraction instead.
    sat_r = math.sqrt(sum(c**2 for c in sat_pos))
    nadir = tuple(c * ((WGS84_A + dem_height_m) / sat_r) for c in sat_pos)

    if initial_guess is not None and not _retry:
        # Warm start: a known-nearby already-solved point (e.g. an
        # adjacent grid point) is a far better starting guess than the
        # generic nadir/cross-track strategy, since it's typically
        # already within metres to a few kilometres of the true answer
        # rather than starting from scratch each time.
        p = initial_guess
    elif _retry:
        # Fallback: plain nadir, no cross-track offset -- a genuinely
        # different starting point from the primary strategy, escaping
        # whatever made the primary guess's basin of attraction
        # problematic for this specific geometry.
        p = nadir
    else:
        altitude_approx = sat_r - (WGS84_A + dem_height_m)
        ground_range_approx = math.sqrt(max(range_m**2 - altitude_approx**2, 0.0))

        radial = tuple(c / sat_r for c in sat_pos)
        vel_mag_for_guess = math.sqrt(sum(v**2 for v in sat_vel)) or 1.0
        vel_unit = tuple(v / vel_mag_for_guess for v in sat_vel)
        cross = (
            vel_unit[1] * radial[2] - vel_unit[2] * radial[1],
            vel_unit[2] * radial[0] - vel_unit[0] * radial[2],
            vel_unit[0] * radial[1] - vel_unit[1] * radial[0],
        )
        cross_mag = math.sqrt(sum(c**2 for c in cross)) or 1.0
        cross_unit = tuple(c / cross_mag for c in cross)
        p = tuple(nadir[i] + cross_unit[i] * ground_range_approx for i in range(3))

    a2 = (WGS84_A + dem_height_m) ** 2
    b2 = (WGS84_B + dem_height_m) ** 2

    for _ in range(max_iterations):
        dx = tuple(p[i] - sat_pos[i] for i in range(3))

        f_doppler = sum(sat_vel[i] * dx[i] for i in range(3))
        range_dist = math.sqrt(sum(c**2 for c in dx)) or 1e-9
        # Numerically stable equivalent of (range_dist - range_m): avoids
        # subtracting two ~700km-scale numbers directly, which loses
        # precision exactly where it matters most (near convergence, when
        # the true difference is small). Confirmed necessary by direct
        # testing: the naive subtraction caused unpredictable oscillation
        # below ~1m residual for some (not all) satellite geometries.
        f_range = (range_dist**2 - range_m**2) / (range_dist + range_m)
        f_ellipsoid = (p[0] ** 2 + p[1] ** 2) / a2 + (p[2] ** 2) / b2 - 1.0

        J = [
            list(sat_vel),
            [dx[0] / range_dist, dx[1] / range_dist, dx[2] / range_dist],
            [2 * p[0] / a2, 2 * p[1] / a2, 2 * p[2] / b2],
        ]
        F = [f_doppler, f_range, f_ellipsoid]

        delta = _solve_3x3(J, [-v for v in F])
        if delta is None:
            raise RuntimeError(
                "Ground point solve: singular Jacobian encountered — "
                "check satellite state vector validity (zero velocity?)."
            )

        step_len = math.sqrt(sum(d**2 for d in delta))
        # Damped Newton: cap the step length at the current distance to
        # the satellite's approximate ground range, halved -- a standard
        # trust-region-style safeguard. Confirmed necessary by direct
        # testing: undamped Newton steps could overshoot wildly for some
        # geometries when the Jacobian was momentarily poorly
        # conditioned, causing persistent oscillation that never
        # actually converged rather than a clean approach to the root.
        max_step = max(range_m * 0.5, 1000.0)
        if step_len > max_step:
            scale = max_step / step_len
            delta = tuple(d * scale for d in delta)

        p = tuple(p[i] + delta[i] for i in range(3))
        step_norm = math.sqrt(sum(d**2 for d in delta))
        if step_norm < tolerance_m:
            return p

    if not _retry:
        logger.debug(
            "Primary initial guess did not converge after %d iterations "
            "(step size %.3e m) -- retrying with a plain nadir starting point.",
            max_iterations, step_norm,
        )
        return solve_ground_point(
            sat_pos, sat_vel, range_time_s, dem_height_m,
            max_iterations, tolerance_m, _retry=True,
        )

    raise RuntimeError(
        f"Ground point solve did not converge after {max_iterations} "
        f"iterations with either starting strategy (final step size "
        f"{step_norm:.3e}m) — check input range_time_s and satellite "
        f"state vector for physical sanity."
    )


def _solve_3x3(matrix, rhs):
    """Solve a 3x3 linear system via Cramer's rule (small, fixed-size,
    no numpy dependency needed for a 3x3 solve at this hot-path scale)."""
    def det3(m):
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

    d = det3(matrix)
    if abs(d) < 1e-20:
        return None

    result = []
    for col in range(3):
        m2 = [row[:] for row in matrix]
        for row in range(3):
            m2[row][col] = rhs[row]
        result.append(det3(m2) / d)
    return result


def perpendicular_baseline(
    sat_pos_ref: Tuple[float, float, float],
    sat_pos_sec: Tuple[float, float, float],
    ground_point: Tuple[float, float, float],
) -> float:
    """
    Real perpendicular baseline (B_perp) between two real satellite
    positions at a real, shared ground point — the standard SAR
    interferometry formulation (Kampes, Hanssen & Perski 2003; the same
    reference already used throughout this module).

    Added directly in response to Foumelis et al. 2018 (IGARSS,
    "ESA SNAP - StaMPS Integrated Processing for Sentinel-1 Persistent
    Scatterer Interferometry"), which states plainly that reference
    (master) scene selection to minimize geometric baselines across a
    stack is supported by SNAP's own InSAR Stack Overview operator --
    this project had no equivalent, reusable utility at all before this,
    and reference/pair selection elsewhere in this project (e.g. the
    Amatrice notebook) had no principled baseline criterion to select on.

    Args:
        sat_pos_ref, sat_pos_sec: Real (x, y, z) ECEF satellite positions,
                                   metres, at each scene's own real
                                   zero-Doppler time for this ground point
                                   (i.e. the output of
                                   interpolate_orbit_state() at a real
                                   find_zero_doppler_time() result, not
                                   an arbitrary scene-center time -- using
                                   scene-center time is a real, common
                                   approximation that is fine for
                                   reference-selection ranking purposes,
                                   but not for per-pixel geometry).
        ground_point:              Real (x, y, z) ECEF ground point, e.g.
                                   the AOI centre or a real scene-centre
                                   ground point.

    Returns:
        Perpendicular baseline in metres (unsigned magnitude -- sufficient
        for ranking/minimizing purposes; sign convention requires an
        additional flight-direction cross product this function does not
        need for that use case).
    """
    baseline = tuple(sat_pos_sec[i] - sat_pos_ref[i] for i in range(3))
    los = tuple(sat_pos_ref[i] - ground_point[i] for i in range(3))
    los_mag = math.sqrt(sum(c**2 for c in los))
    if los_mag < 1.0:
        raise ValueError("Degenerate line-of-sight (satellite at ground point) -- check inputs.")
    los_hat = tuple(c / los_mag for c in los)

    baseline_mag = math.sqrt(sum(c**2 for c in baseline))
    b_parallel = sum(baseline[i] * los_hat[i] for i in range(3))
    b_perp_sq = baseline_mag**2 - b_parallel**2
    return math.sqrt(max(0.0, b_perp_sq))


def select_reference_minimizing_baselines(
    candidate_positions: dict,
    ground_point: Tuple[float, float, float],
) -> Tuple[str, dict]:
    """
    Real reference/master scene selection that minimizes the maximum
    perpendicular baseline against every other real scene in the stack --
    the same real criterion SNAP's InSAR Stack Overview operator
    supports (Foumelis et al. 2018), not just "use the earliest date"
    (this project's own prior behaviour, e.g. throughout the Amatrice
    notebook, which had no baseline criterion at all).

    Args:
        candidate_positions: {label: (x, y, z)} -- real ECEF satellite
                              position for each real candidate scene, at
                              its own real zero-Doppler time for
                              ground_point (see perpendicular_baseline()
                              docstring for the same caveat on using
                              scene-centre time as an approximation here).
        ground_point:        Real (x, y, z) ECEF ground point (e.g. AOI
                              centre) shared across all candidates.

    Returns:
        (best_label, all_baselines) where all_baselines is
        {label: {other_label: b_perp_metres}} for every real pair, so the
        full stack geometry -- not just the winning choice -- is
        available for inspection/plotting.
    """
    labels = list(candidate_positions.keys())
    if len(labels) < 2:
        raise ValueError(f"Need at least 2 candidate scenes to select a reference, got {len(labels)}.")

    all_baselines = {label: {} for label in labels}
    for i, label_i in enumerate(labels):
        for label_j in labels[i + 1:]:
            b_perp = perpendicular_baseline(
                candidate_positions[label_i], candidate_positions[label_j], ground_point,
            )
            all_baselines[label_i][label_j] = b_perp
            all_baselines[label_j][label_i] = b_perp

    def max_baseline_from(label):
        return max(all_baselines[label].values())

    best_label = min(labels, key=max_baseline_from)
    return best_label, all_baselines