"""
Proves the large-N numerical amplification bug in the ORIGINAL
compute_burst_synchronization() (raw dt modulo one averaged cycle
estimate) using a REALISTIC weeks-apart scenario -- exactly the regime
that broke in the real Mexico City logs -- and confirms the FIXED
version (per-date local burst phase, difference of two small numbers)
is robust in that same regime.

Uses the same exactly-solvable straight-line orbit construction as
test_burst_synchronization.py, but with reference and secondary
zero-Doppler crossings 60 REAL DAYS apart (matching the largest gap in
the actual Mexico City stack) plus a small, KNOWN synchronization
residual -- so the correct answer is known by construction.
"""
from datetime import datetime, timedelta

from pygeofetch.insar import annotation, esd

VELOCITY = (7500.0, 0.0, 0.0)
POSITION_AT_T0 = (0.0, 0.0, 500000.0)
GROUND_POINT = (0.0, 0.0, 0.0)

CYCLE_S = 1506 * 0.002055563 * 0.9


def build_orbit(true_crossing_time: datetime, window_s=30):
    """Straight-line orbit whose zero-Doppler crossing for GROUND_POINT
    is EXACTLY true_crossing_time, by construction (same technique as
    test_burst_synchronization.py)."""
    times = [true_crossing_time + timedelta(seconds=float(s)) for s in range(-window_s, window_s + 1)]
    positions, velocities = [], []
    for t in times:
        dt = (t - true_crossing_time).total_seconds()
        pos = tuple(POSITION_AT_T0[i] + VELOCITY[i] * dt for i in range(3))
        positions.append(pos)
        velocities.append(VELOCITY)
    return times, positions, velocities


def build_swath_timing(first_burst_time: datetime, n_bursts=8, cycle_s=CYCLE_S):
    bursts = [
        annotation.BurstInfo(
            burst_index=i, azimuth_time=first_burst_time + timedelta(seconds=i * cycle_s),
            sensing_time=None, byte_offset=0, first_valid_sample=None, last_valid_sample=None,
        )
        for i in range(n_bursts)
    ]
    return annotation.SwathTiming(lines_per_burst=1506, samples_per_burst=23674, bursts=bursts)


def old_buggy_formula(t_ref, t_sec, ref_burst_info, sec_burst_info):
    """Reproduces the ORIGINAL (pre-fix) approach exactly: raw dt,
    reduced modulo one averaged cycle estimate."""
    ref_cycle = esd._mean_burst_cycle_s(ref_burst_info)
    sec_cycle = esd._mean_burst_cycle_s(sec_burst_info)
    cycle = (ref_cycle + sec_cycle) / 2.0
    dt = (t_ref - t_sec).total_seconds()
    sync_offset_s = ((dt + cycle / 2.0) % cycle) - cycle / 2.0
    return sync_offset_s * 1000.0, cycle


if __name__ == "__main__":
    T_REF = datetime(2016, 7, 24, 12, 25, 42, 252132)
    GAP_DAYS = 60  # matches the real 07-24 -> 09-22 pair
    KNOWN_RESIDUAL_MS = 3.2
    T_SEC = T_REF + timedelta(days=GAP_DAYS)  # clean gap; residual injected via burst phase below

    print(f"Reference crossing: {T_REF}")
    print(f"Secondary crossing: {T_SEC}  (exactly {GAP_DAYS} days later)")
    print(f"Real burst cycles spanned: {(T_SEC - T_REF).total_seconds() / CYCLE_S:,.0f}")
    print()

    ref_orbit = build_orbit(T_REF)
    sec_orbit = build_orbit(T_SEC)

    # Burst timing: choose each date's first-burst time so the "local
    # phase" (offset from the true crossing back to the nearest
    # preceding burst start) is a KNOWN, precomputed value -- reference
    # gets exactly 1.5s, secondary gets exactly 1.5s - 3.2ms, so their
    # difference is exactly the KNOWN_RESIDUAL_MS we're injecting.
    # sec's cycle also carries a deliberate ~0.3 microsecond
    # perturbation relative to ref's, exactly the kind of sub-
    # microsecond estimation noise real ISO-8601-parsed timestamps
    # would carry -- present in both formulas equally.
    ref_burst_info = build_swath_timing(T_REF - timedelta(seconds=1.5), cycle_s=CYCLE_S)
    sec_burst_info = build_swath_timing(
        T_SEC - timedelta(seconds=1.5 - KNOWN_RESIDUAL_MS / 1000.0), cycle_s=CYCLE_S + 3e-7
    )

    print("=== OLD (buggy) formula: raw dt modulo one averaged cycle ===")
    old_result_ms, cycle_used = old_buggy_formula(T_REF, T_SEC, ref_burst_info, sec_burst_info)
    print(f"  cycle estimate used: {cycle_used:.9f}s (true: {CYCLE_S:.9f}s, error: {cycle_used-CYCLE_S:.2e}s)")
    print(f"  result: {old_result_ms:.3f} ms  (true residual: {KNOWN_RESIDUAL_MS} ms)")
    old_error = abs(old_result_ms - KNOWN_RESIDUAL_MS)
    print(f"  error: {old_error:.1f} ms")
    print()

    print("=== NEW (fixed) formula: per-date local burst phase ===")
    result = esd.compute_burst_synchronization(
        ref_orbit, sec_orbit, ref_burst_info, sec_burst_info,
        GROUND_POINT, ref_time_guess=T_REF, sec_time_guess=T_SEC,
    )
    new_result_ms = result["sync_offset_ms"]
    print(f"  result: {new_result_ms:.3f} ms  (true residual: {KNOWN_RESIDUAL_MS} ms)")
    new_error = abs(new_result_ms - KNOWN_RESIDUAL_MS)
    print(f"  error: {new_error:.4f} ms")
    print()

    print("=== Verdict ===")
    assert old_error > 50, (
        f"expected the OLD formula to fail badly (amplification bug) in this weeks-apart "
        f"regime, but error was only {old_error:.3f}ms"
    )
    print(f"OLD formula: {old_error:.1f}ms error from a {cycle_used-CYCLE_S:.2e}s cycle estimation "
          f"error amplified by ~{(T_SEC-T_REF).total_seconds()/CYCLE_S:,.0f} cycles -- "
          f"confirms the bug is real and this severe in the actual weeks-apart regime.")

    assert new_error < 0.5, f"NEW formula should recover the true residual accurately, got error={new_error:.4f}ms"
    print(f"NEW formula: {new_error:.4f}ms error -- recovers the true {KNOWN_RESIDUAL_MS}ms residual "
          f"correctly despite the same weeks-apart gap and the same cycle-estimation imprecision, "
          f"because it never forms the huge raw difference in the first place.")
    print("\nPASS: bug reproduced and fix verified in the realistic regime.")
