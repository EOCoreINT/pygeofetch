"""
Validation for compute_burst_synchronization().

Uses a deliberately simplified (not geodetically realistic) but exactly
solvable orbit/ground-point setup so the expected answer is known by
construction, not just plausible: both satellites fly a straight-line,
constant-velocity path; the ground point sits perpendicular to the
reference's velocity at t=0, so the reference's zero-Doppler crossing
for that point is EXACTLY t=0 by construction. The secondary flies the
identical spatial trajectory, offset in absolute time by a value we
choose -- so the secondary's zero-Doppler crossing for the same point
is exactly t = orbit_time_shift_s, also by construction. This gives
exact, hand-computable expected sync_offset_ms for every test case,
rather than only a plausibility check.

find_zero_doppler_time() itself is not re-validated here (already
covered by earlier test suites) -- this isolates and validates
compute_burst_synchronization()'s own logic: the modulo-reduction to a
single burst cycle and the threshold comparison.
"""
import sys
sys.path.insert(0, "/home/claude/work")

import importlib.util
from datetime import datetime, timedelta

# Load geolocation.py, annotation.py, esd.py directly (avoids pulling in
# the full pygeofetch package and its unrelated dependencies).
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

geolocation = _load("pygeofetch.insar.geolocation", "pygeofetch/insar/geolocation.py")
annotation = _load("pygeofetch.insar.annotation", "pygeofetch/insar/annotation.py")
esd = _load("pygeofetch.insar.esd", "pygeofetch/insar/esd.py")

GROUND_POINT = (0.0, 0.0, 0.0)
VELOCITY = (7500.0, 0.0, 0.0)          # constant, straight-line motion
POSITION_AT_T0 = (0.0, 0.0, 500000.0)  # perpendicular to VELOCITY, so
                                        # doppler(t0) = V . (GP - P0) = 0 exactly

T0 = datetime(2016, 7, 24, 12, 25, 42)


def build_orbit(time_shift_s: float):
    """Straight-line orbit, identical trajectory to the reference,
    offset in absolute time by time_shift_s. By construction, this
    orbit's zero-Doppler crossing for GROUND_POINT is exactly
    T0 + time_shift_s."""
    times = [T0 + timedelta(seconds=float(s)) for s in range(-30, 31)]
    positions = []
    velocities = []
    for t in times:
        dt = (t - T0).total_seconds() - time_shift_s
        pos = tuple(POSITION_AT_T0[i] + VELOCITY[i] * dt for i in range(3))
        positions.append(pos)
        velocities.append(VELOCITY)
    return times, positions, velocities


def build_swath_timing(first_burst_time: datetime, n_bursts: int, cycle_s: float, lines_per_burst: int = 1506):
    bursts = [
        annotation.BurstInfo(
            burst_index=i,
            azimuth_time=first_burst_time + timedelta(seconds=i * cycle_s),
            sensing_time=None, byte_offset=0,
            first_valid_sample=None, last_valid_sample=None,
        )
        for i in range(n_bursts)
    ]
    return annotation.SwathTiming(lines_per_burst=lines_per_burst, samples_per_burst=23674, bursts=bursts)


CYCLE_S = 1506 * 0.002055563 * 0.9  # nominal ~10% overlap, matches real S1 IW


def run_case(label, orbit_time_shift_s, expected_sync_offset_ms, expect_within_requirement):
    print(f"=== {label} ===")
    ref_orbit = build_orbit(0.0)
    sec_orbit = build_orbit(orbit_time_shift_s)

    # Burst timing itself is irrelevant to WHICH ground-truth answer we
    # expect (that's fixed by orbit_time_shift_s alone) -- any real,
    # consistent burst structure works; use the same one for both here
    # since only the cycle period (identical for both, by construction)
    # matters to the modulo reduction.
    ref_burst_info = build_swath_timing(T0 - timedelta(seconds=15), 8, CYCLE_S)
    sec_burst_info = build_swath_timing(T0 - timedelta(seconds=15), 8, CYCLE_S)

    result = esd.compute_burst_synchronization(
        ref_orbit, sec_orbit, ref_burst_info, sec_burst_info,
        GROUND_POINT, ref_time_guess=T0, sec_time_guess=T0,
    )

    print(f"  orbit_time_shift_s={orbit_time_shift_s}, burst_cycle_s={result['burst_cycle_s']:.4f}")
    print(f"  ref_zero_doppler_time={result['ref_zero_doppler_time']}")
    print(f"  sec_zero_doppler_time={result['sec_zero_doppler_time']}")
    print(f"  sync_offset_ms={result['sync_offset_ms']:.4f} (expected ~{expected_sync_offset_ms:.4f})")
    print(f"  within_esa_requirement={result['within_esa_requirement']} (expected {expect_within_requirement})")

    # find_zero_doppler_time solves to ~1e-9s tolerance; allow generous
    # float slack for the modulo arithmetic itself.
    assert abs(result["sync_offset_ms"] - expected_sync_offset_ms) < 0.01, (
        f"sync_offset_ms off: got {result['sync_offset_ms']}, expected {expected_sync_offset_ms}"
    )
    assert result["within_esa_requirement"] == expect_within_requirement
    print("  PASS\n")


if __name__ == "__main__":
    # Case 1: near-perfect synchronization (0.2 ms raw offset, well within 5ms).
    run_case("well-synchronized (0.2 ms)", orbit_time_shift_s=0.0002,
              expected_sync_offset_ms=-0.2, expect_within_requirement=True)

    # Case 2: badly desynchronized -- 40% of a burst cycle raw offset,
    # should reduce to -0.4*cycle (well outside 5ms). This is the
    # signature of a genuine "different burst family" pair.
    bad_shift = 0.4 * CYCLE_S
    run_case("desynchronized (40% of burst cycle)", orbit_time_shift_s=bad_shift,
              expected_sync_offset_ms=-0.4 * CYCLE_S * 1000, expect_within_requirement=False)

    # Case 3: the core thing this function has to get right -- a raw
    # offset of exactly 3 whole burst cycles PLUS a small 0.3ms residual
    # must reduce to just the 0.3ms residual, not the ~8.4-second raw
    # difference. This is the "whole-burst offset vs. fine residual"
    # distinction from Yagüe-Martínez et al. Section III-A.
    whole_cycles_plus_residual = 3 * CYCLE_S + 0.0003
    run_case("3 whole burst cycles + 0.3ms residual", orbit_time_shift_s=whole_cycles_plus_residual,
              expected_sync_offset_ms=-0.3, expect_within_requirement=True)

    print("ALL TESTS PASSED")
