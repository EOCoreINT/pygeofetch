"""
Validation for build_sbas_network(), using the ACTUAL real numbers
from the Mexico City (Iztapalapa) run: 6 dates split into two
coregistration "families" (A: 07-24/08-17/09-10, B: 08-05/08-29/09-22),
same-family pairs at real coherence 0.4-0.56, cross-family pairs stuck
at 0.24-0.27, and real perpendicular baselines from that same run.

Confirms:
  1. The old bare-spanning-tree approach (sorted by baseline alone)
     reproduces the real, bad outcome actually observed (4/5 selected
     pairs cross-family, mean selected coherence ~0.25).
  2. build_sbas_network() instead selects mostly same-family (good)
     pairs, uses the minimum necessary cross-family bridge(s), stays
     fully connected, and achieves substantially higher mean coherence
     across its selected network.
  3. Redundancy: same-family dates end up with more than one
     connection each, so a single excluded pair can't disconnect them.
"""
from pygeofetch.insar.timeseries import PairCandidate, build_sbas_network

# Real data from the actual Mexico City run (cell 16 coherence + cell 22 baselines).
REAL_PAIRS = [
    # (date1, date2, baseline_m, coherence)
    ("2016-07-24", "2016-08-05", 70.2, 0.237),
    ("2016-07-24", "2016-08-17", 19.5, 0.560),
    ("2016-07-24", "2016-08-29", 7.7, 0.239),
    ("2016-07-24", "2016-09-10", 59.6, 0.470),
    ("2016-07-24", "2016-09-22", 51.6, 0.240),
    ("2016-08-05", "2016-08-17", 50.7, 0.264),
    ("2016-08-05", "2016-08-29", 77.9, 0.406),
    ("2016-08-05", "2016-09-10", 12.4, 0.265),
    ("2016-08-05", "2016-09-22", 18.7, 0.454),
    ("2016-08-17", "2016-08-29", 27.2, 0.240),
    ("2016-08-17", "2016-09-10", 40.3, 0.492),
    ("2016-08-17", "2016-09-22", 32.1, 0.244),
    ("2016-08-29", "2016-09-10", 67.1, 0.262),
    ("2016-08-29", "2016-09-22", 59.2, 0.461),
    ("2016-09-10", "2016-09-22", 9.0, 0.243),
]
DATES = ["2016-07-24", "2016-08-05", "2016-08-17", "2016-08-29", "2016-09-10", "2016-09-22"]
FAMILY = {
    "2016-07-24": "A", "2016-08-17": "A", "2016-09-10": "A",
    "2016-08-05": "B", "2016-08-29": "B", "2016-09-22": "B",
}


def old_baseline_only_network(pairs, dates):
    """Reproduce the notebook's ORIGINAL cell-22 algorithm exactly:
    greedy union-find over pairs sorted by baseline alone."""
    parent = {d: d for d in dates}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    selected = []
    for d1, d2, b, coh in sorted(pairs, key=lambda p: p[2]):
        r1, r2 = find(d1), find(d2)
        if r1 != r2:
            parent[r1] = r2
            selected.append((d1, d2, coh))
    return selected


def test_reproduces_real_bad_outcome():
    print("=== 1. Confirm old baseline-only algorithm reproduces the real bad outcome ===")
    old_selected = old_baseline_only_network(REAL_PAIRS, DATES)
    mean_coh = sum(c for _, _, c in old_selected) / len(old_selected)
    cross_family = sum(1 for d1, d2, _ in old_selected if FAMILY[d1] != FAMILY[d2])
    print(f"  old network: {len(old_selected)} pairs, {cross_family} cross-family, mean coherence {mean_coh:.3f}")
    for d1, d2, c in old_selected:
        print(f"    {d1} -> {d2}: coherence={c:.3f} ({'cross' if FAMILY[d1]!=FAMILY[d2] else 'same'}-family)")
    assert cross_family >= 3, "should reproduce the real bad outcome (mostly cross-family pairs)"
    assert mean_coh < 0.35, "should reproduce the real low mean coherence"
    print("  confirmed: this matches what actually happened in the notebook run\n")


def test_build_sbas_network_fixes_it():
    print("=== 2. build_sbas_network() selects a much better network ===")
    candidates = [
        PairCandidate(d1, d2, perpendicular_baseline_m=b, coherence=c)
        for d1, d2, b, c in REAL_PAIRS
    ]
    selected, report = build_sbas_network(candidates, DATES, min_coherence=0.3, redundancy=2)

    coh_by_pair = {(d1, d2): c for d1, d2, b, c in REAL_PAIRS}
    mean_coh = sum(coh_by_pair[p] for p in selected) / len(selected)
    cross_family = sum(1 for d1, d2 in selected if FAMILY[d1] != FAMILY[d2])

    print(f"  selected: {len(selected)} pairs ({len(report['good_pairs'])} good, "
          f"{len(report['bridge_pairs'])} bridge)")
    for d1, d2 in selected:
        tag = "BRIDGE" if (d1, d2) in report["bridge_pairs"] else "good"
        print(f"    {d1} -> {d2}: coherence={coh_by_pair[(d1,d2)]:.3f} [{tag}, "
              f"{'cross' if FAMILY[d1]!=FAMILY[d2] else 'same'}-family]")
    print(f"  connected: {report['connected']}, unconnected dates: {report['unconnected_dates']}")
    print(f"  mean coherence of selected network: {mean_coh:.3f}")

    assert report["connected"], "all 6 dates should end up connected"
    assert not report["unconnected_dates"]
    # Should need exactly one cross-family bridge (A and B are each
    # internally fully connectable with good pairs, but never connect
    # to each other via any good pair -- confirmed from the real data).
    assert len(report["bridge_pairs"]) == 1, (
        f"expected exactly 1 necessary cross-family bridge, got {len(report['bridge_pairs'])}"
    )
    assert cross_family == 1, "only the necessary bridge pair should be cross-family"
    assert mean_coh > 0.40, f"expected substantially better mean coherence, got {mean_coh:.3f}"

    # Redundancy: every date should have more than one connection
    # (unlike a bare spanning tree, which gives most dates exactly one).
    degree = {}
    for d1, d2 in selected:
        degree[d1] = degree.get(d1, 0) + 1
        degree[d2] = degree.get(d2, 0) + 1
    print(f"  per-date connection counts: {degree}")
    assert min(degree.values()) >= 2, "every date should have redundant connections, not just one"

    old_selected = old_baseline_only_network(REAL_PAIRS, DATES)
    old_mean = sum(c for _, _, c in old_selected) / len(old_selected)
    print(f"\n  old (baseline-only) mean coherence: {old_mean:.3f}")
    print(f"  new (build_sbas_network) mean coherence: {mean_coh:.3f}")
    assert mean_coh > old_mean + 0.10, "new network should be substantially better than the old one"
    print("  PASS\n")


def test_no_good_bridge_available():
    print("=== 3. Graceful degradation: no candidate pair connects an isolated date ===")
    candidates = [
        PairCandidate(d1, d2, perpendicular_baseline_m=b, coherence=c)
        for d1, d2, b, c in REAL_PAIRS
    ]
    dates_plus_orphan = DATES + ["2016-10-05"]  # no candidate pair involves this date at all
    selected, report = build_sbas_network(candidates, dates_plus_orphan, min_coherence=0.3, redundancy=2)
    print(f"  connected: {report['connected']}, unconnected: {report['unconnected_dates']}")
    assert report["connected"] is False
    assert report["unconnected_dates"] == ["2016-10-05"]
    print("  correctly reports the orphan date instead of silently dropping or crashing")
    print("  PASS\n")


if __name__ == "__main__":
    test_reproduces_real_bad_outcome()
    test_build_sbas_network_fixes_it()
    test_no_good_bridge_available()
    print("ALL TESTS PASSED")
