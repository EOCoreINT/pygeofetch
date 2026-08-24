"""
Validates select_pairs_for_processing() / screen_stack_burst_synchronization()
against the ACTUAL real Δt_acq values from the Mexico City (Iztapalapa)
stack's real log (the one confirming 6/6 same-family within spec, 9/9
cross-family outside it) -- not synthetic data. Confirms the
pre-screening selector reduces 15 candidate pairs down to the same kind
of compact, redundant, connected set build_sbas_network() found from
real measured coherence, but derived purely from the cheap burst-sync
signal, before any full processing would have run.
"""
import sys
sys.path.insert(0, "/home/mrtenkorang/open-source-projects/")

import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


timeseries = _load("pygeofetch.insar.timeseries", "pygeofetch/insar/timeseries.py")

BurstSyncResult = timeseries.BurstSyncResult
select_pairs_for_processing = timeseries.select_pairs_for_processing

# Real Δt_acq values, read directly from the actual Mexico City log.
REAL_SYNC_MS = {
    ("2016-07-24", "2016-08-05"): 8.645,
    ("2016-07-24", "2016-08-17"): -3.928,
    ("2016-07-24", "2016-08-29"): 6.308,
    ("2016-07-24", "2016-09-10"): 0.247,
    ("2016-07-24", "2016-09-22"): 5.592,
    ("2016-08-05", "2016-08-17"): -12.573,
    ("2016-08-05", "2016-08-29"): -2.337,
    ("2016-08-05", "2016-09-10"): -8.398,
    ("2016-08-05", "2016-09-22"): -3.053,
    ("2016-08-17", "2016-08-29"): 10.237,
    ("2016-08-17", "2016-09-10"): 4.175,
    ("2016-08-17", "2016-09-22"): 9.520,
    ("2016-08-29", "2016-09-10"): -6.061,
    ("2016-08-29", "2016-09-22"): -0.716,
    ("2016-09-10", "2016-09-22"): 5.345,
}
DATES = ["2016-07-24", "2016-08-05", "2016-08-17", "2016-08-29", "2016-09-10", "2016-09-22"]
FAMILY = {
    "2016-07-24": "A", "2016-08-17": "A", "2016-09-10": "A",
    "2016-08-05": "B", "2016-08-29": "B", "2016-09-22": "B",
}
REQUIREMENT_MS = 5.0
BURST_CYCLE_S = 2.759  # approximate, matches the real logged values


def test_real_data_reproduces_family_split():
    print("=== 1. Confirm real Δt_acq values reproduce the exact same/cross-family split ===")
    for (d1, d2), ms in REAL_SYNC_MS.items():
        expected_within = FAMILY[d1] == FAMILY[d2]
        actual_within = abs(ms) < REQUIREMENT_MS
        assert actual_within == expected_within, (
            f"{d1}->{d2}: |Δt_acq|={abs(ms)}ms, within={actual_within}, "
            f"but family match={expected_within} -- real data should agree perfectly"
        )
    print("  confirmed: all 15 pairs' within-requirement status exactly matches family membership")
    print("  PASS\n")


def test_select_pairs_for_processing_on_real_data():
    print("=== 2. select_pairs_for_processing() on the real burst-sync data ===")
    sync_results = [
        BurstSyncResult(
            date1=d1, date2=d2, sync_offset_ms=ms,
            burst_cycle_s=BURST_CYCLE_S, within_requirement=abs(ms) < REQUIREMENT_MS,
        )
        for (d1, d2), ms in REAL_SYNC_MS.items()
    ]

    selected_pairs, report = select_pairs_for_processing(sync_results, DATES, redundancy=2)

    print(f"  selected for full processing: {len(selected_pairs)} of {len(REAL_SYNC_MS)} candidate pairs")
    print(f"    good: {len(report['good_pairs'])}, bridge: {len(report['bridge_pairs'])}")
    for d1, d2 in selected_pairs:
        tag = "BRIDGE" if (d1, d2) in report["bridge_pairs"] else "good"
        ms = REAL_SYNC_MS[(d1, d2)]
        print(f"    {d1} -> {d2}: Δt_acq={ms:+.3f}ms [{tag}, "
              f"{'cross' if FAMILY[d1]!=FAMILY[d2] else 'same'}-family]")
    print(f"  connected: {report['connected']}, unconnected: {report['unconnected_dates']}")
    print(f"  excluded (not worth full processing): {len(report['excluded_pairs'])} pairs")
    for d1, d2 in report["excluded_pairs"]:
        print(f"    skipped: {d1} -> {d2} (Δt_acq={REAL_SYNC_MS[(d1,d2)]:+.3f}ms)")

    assert report["connected"], "all 6 dates should end up connected"
    assert not report["unconnected_dates"]
    # All 6 same-family pairs should be selected as "good" (each family
    # is a 3-clique; redundancy=2 naturally admits all 3 edges per clique).
    assert len(report["good_pairs"]) == 6, f"expected all 6 same-family pairs selected, got {len(report['good_pairs'])}"
    for d1, d2 in report["good_pairs"]:
        assert FAMILY[d1] == FAMILY[d2], f"a 'good' pair should be same-family: {d1}->{d2}"
    # Exactly one necessary cross-family bridge, and it should be the
    # REAL best (lowest |Δt_acq|) cross-family pair: 09-10 -> 09-22 (5.345ms).
    assert len(report["bridge_pairs"]) == 1, f"expected exactly 1 necessary bridge, got {len(report['bridge_pairs'])}"
    bridge = report["bridge_pairs"][0]
    assert bridge == ("2016-09-10", "2016-09-22"), (
        f"expected the real best cross-family pair (09-10->09-22, 5.345ms) as bridge, got {bridge}"
    )
    # The 8 remaining cross-family pairs should be correctly excluded --
    # not worth full processing at all.
    assert len(report["excluded_pairs"]) == 8, f"expected 8 excluded pairs, got {len(report['excluded_pairs'])}"

    print(f"\n  Full processing would only be needed for {len(selected_pairs)}/15 pairs "
          f"instead of all 15 -- a real ~{100*(1-len(selected_pairs)/15):.0f}% reduction in "
          f"expensive (coregistration+ESD+deburst+interferogram) processing for this stack, "
          f"known BEFORE any of that processing runs.")
    print("  PASS\n")


if __name__ == "__main__":
    test_real_data_reproduces_family_split()
    test_select_pairs_for_processing_on_real_data()
    print("ALL TESTS PASSED")
