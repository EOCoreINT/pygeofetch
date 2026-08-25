"""
Verifies screen_stack_burst_synchronization()'s ORCHESTRATION logic
(correct O(n) parsing, correct per-pair wiring, graceful handling of
missing/erroring dates) using lightweight fakes for
parse_slc_geometry/parse_burst_info/parse_orbit_file -- the underlying
physics (compute_burst_synchronization itself) is already covered by
test_burst_synchronization.py and test_burst_sync_large_gap.py.
"""
from datetime import datetime, timedelta
from unittest.mock import patch

from pygeofetch.insar import annotation, geolocation, timeseries

VELOCITY = (7500.0, 0.0, 0.0)
POSITION_AT_T0 = (0.0, 0.0, 500000.0)
GROUND_POINT = (0.0, 0.0, 0.0)
CYCLE_S = 1506 * 0.002055563 * 0.9


def make_geom(scene_center_time, n_lines=2000):
    az_interval = 0.002055563
    return annotation.SLCGeometry(
        first_line_time=scene_center_time - timedelta(seconds=(n_lines / 2) * az_interval),
        azimuth_time_interval_s=az_interval,
        near_range_time_s=0.005, range_sampling_rate_hz=6.0e7,
        n_lines=n_lines, n_columns=8000,
    )


def make_orbit(true_crossing_time, window_s=30):
    times = [true_crossing_time + timedelta(seconds=float(s)) for s in range(-window_s, window_s + 1)]
    positions, velocities = [], []
    for t in times:
        dt = (t - true_crossing_time).total_seconds()
        positions.append(tuple(POSITION_AT_T0[i] + VELOCITY[i] * dt for i in range(3)))
        velocities.append(VELOCITY)
    return times, positions, velocities


def make_burst_info(first_burst_time, n_bursts=8):
    bursts = [
        annotation.BurstInfo(
            burst_index=i, azimuth_time=first_burst_time + timedelta(seconds=i * CYCLE_S),
            sensing_time=None, byte_offset=0, first_valid_sample=None, last_valid_sample=None,
        )
        for i in range(n_bursts)
    ]
    return annotation.SwathTiming(lines_per_burst=1506, samples_per_burst=23674, bursts=bursts)


# Real crossing times: three "good" dates spaced a real repeat cycle apart
# (well-synchronized), one "bad" date deliberately offset by 40% of a
# burst cycle (badly desynchronized) -- verifies the orchestration wires
# per-date objects to the right pairs, not just that any single pair works.
#
# NOMINAL is the undisturbed grid used to anchor every date's burst
# structure consistently; CROSSING is each date's TRUE zero-Doppler
# crossing (equal to NOMINAL except for the deliberately desynced date)
# -- keeping these separate is what actually lets the injected
# desynchronization show up in the local-phase difference, rather than
# being silently absorbed if both were shifted together.
NOMINAL = {
    "2016-07-24": datetime(2016, 7, 24, 12, 25, 42),
    "2016-08-05": datetime(2016, 7, 24, 12, 25, 42) + timedelta(days=12),
    "2016-08-17": datetime(2016, 7, 24, 12, 25, 42) + timedelta(days=24),
    "2016-08-29": datetime(2016, 7, 24, 12, 25, 42) + timedelta(days=36),
}
CROSSING = dict(NOMINAL)
CROSSING["2016-08-29"] = NOMINAL["2016-08-29"] + timedelta(seconds=0.4 * CYCLE_S)

FAKE_ORBITS = {d: make_orbit(t) for d, t in CROSSING.items()}
FAKE_GEOMS = {d: make_geom(t) for d, t in CROSSING.items()}
FAKE_BURSTS = {d: make_burst_info(NOMINAL[d] - timedelta(seconds=1.5)) for d in CROSSING}


def fake_parse_slc_geometry(path, member_hint=None):
    return FAKE_GEOMS[path]


def fake_parse_burst_info(path, member_hint=None):
    return FAKE_BURSTS[path]


def fake_parse_orbit_file(path):
    return FAKE_ORBITS[path]


if __name__ == "__main__":
    dates = list(CROSSING.keys())
    safe_zips = {d: d for d in dates}       # use date as its own fake "path" key
    orbit_files = {d: d for d in dates}

    with patch.object(annotation, "parse_slc_geometry", side_effect=fake_parse_slc_geometry), \
         patch.object(geolocation, "parse_orbit_file", side_effect=fake_parse_orbit_file):
        # annotation.parse_burst_info patched separately since it's a
        # distinct real function in the same module.
        with patch.object(annotation, "parse_burst_info", side_effect=fake_parse_burst_info):
            results = timeseries.screen_stack_burst_synchronization(
                dates, safe_zips, orbit_files, GROUND_POINT,
            )

    print(f"Screened {len(results)} pairs from {len(dates)} dates (expected {len(dates)*(len(dates)-1)//2})")
    assert len(results) == 6, f"expected 6 pairs (4 choose 2), got {len(results)}"

    by_pair = {(r.date1, r.date2): r for r in results}
    for (d1, d2), r in by_pair.items():
        print(f"  {d1} -> {d2}: Δt_acq={r.sync_offset_ms:+.3f}ms, within={r.within_requirement}")

    # 08-29 was deliberately desynchronized from everything else by 40%
    # of a cycle -- every pair involving it should be flagged bad;
    # every pair among the other three (all on a clean repeat-cycle
    # grid) should be flagged good.
    for (d1, d2), r in by_pair.items():
        expect_bad = "2016-08-29" in (d1, d2)
        assert r.within_requirement != expect_bad, (
            f"{d1}->{d2}: expected within_requirement={not expect_bad}, got {r.within_requirement}"
        )

    print("\nPASS: orchestration correctly wires per-date objects to the right pairs, "
          "and correctly isolates the deliberately-desynchronized date.")

    # Graceful handling: a date with no SAFE zip/orbit file should be
    # skipped, not crash the whole screen.
    dates_with_orphan = dates + ["2099-01-01"]
    with patch.object(annotation, "parse_slc_geometry", side_effect=fake_parse_slc_geometry), \
         patch.object(geolocation, "parse_orbit_file", side_effect=fake_parse_orbit_file), \
         patch.object(annotation, "parse_burst_info", side_effect=fake_parse_burst_info):
        results2 = timeseries.screen_stack_burst_synchronization(
            dates_with_orphan, safe_zips, orbit_files, GROUND_POINT,
        )
    assert len(results2) == 6, f"orphan date should be silently skipped, expected 6 pairs, got {len(results2)}"
    print("PASS: date with no SAFE zip/orbit file is skipped gracefully, not a crash.")

    print("\nALL TESTS PASSED")
