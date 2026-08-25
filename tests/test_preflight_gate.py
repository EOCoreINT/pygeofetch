"""
Validates the fixed PreflightGate -- in particular, that
_screen_burst_families no longer gives a false clean pass on a
reproduction of the exact real scenario (Mexico City's within-track
cross-family split) that the original, buggy version would have
missed.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pygeofetch.insar.preflight as preflight


class FakeScene:
    def __init__(self, dt, track, satellite, properties=None):
        self.datetime = dt
        self.geometry = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        }
        self.satellite = satellite
        self.properties = (
            properties if properties is not None else {"relativeOrbitNumber": track}
        )


class FakeBBox:
    def __init__(self):
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = 0.0, 0.0, 1.0, 1.0


def make_gate(**kwargs):
    kwargs.setdefault(
        "attempt_real_burst_check", False
    )  # keep existing tests network-independent
    return preflight.PreflightGate(
        client=None,
        aoi_bbox=FakeBBox(),
        start_date="2016-07-01",
        end_date="2016-10-01",
        **kwargs,
    )


def test_reproduces_mexico_city_same_track_split_not_missed():
    print(
        "=== 1. Real bug regression: same-track cross-family scenario is no longer silently missed ==="
    )
    # Exactly the real Mexico City situation: 6 dates, all real track
    # 143, all Sentinel-1A -- the actual cross-family split (confirmed
    # via compute_burst_synchronization) happened WITHIN this single
    # (track, satellite) group, which the old code could never see.
    dates = [
        "2016-07-24",
        "2016-08-05",
        "2016-08-17",
        "2016-08-29",
        "2016-09-10",
        "2016-09-22",
    ]
    selected = [
        FakeScene(
            datetime.strptime(d, "%Y-%m-%d"),
            track=143,
            satellite="SENTINEL-1A",
            properties={"relativeOrbitNumber": 143, "platformSerialIdentifier": "A"},
        )
        for d in dates
    ]
    geometry_report = {"track": 143, "satellites": {"S1A"}}

    gate = make_gate()
    _, issues = gate._screen_burst_families(selected, geometry_report)
    codes = [i.code for i in issues]
    print(f"  issue codes returned: {codes}")

    # The fixed version must NEVER return an empty list for this real
    # scenario -- it must always carry the honest "risk unassessed"
    # advisory, since it has no way to know the real family split
    # exists.
    assert issues, "must never silently return zero issues -- that's the exact old bug"
    assert "BURST_FAMILY_RISK_UNASSESSED" in codes
    unassessed = next(i for i in issues if i.code == "BURST_FAMILY_RISK_UNASSESSED")
    assert unassessed.severity == preflight.SEVERITY_YELLOW
    assert "select_burst_synchronized_dates" in unassessed.message
    print("  PASS -- correctly refuses to claim this stack is safe\n")


def test_old_buggy_grouping_would_have_returned_empty():
    print(
        "=== 2. Confirms the OLD grouping logic really would have returned [] for this exact case ==="
    )
    # Reproduce the old (Track, Satellite) grouping directly, using the
    # same real scene data, to concretely demonstrate the false pass.
    dates = [
        "2016-07-24",
        "2016-08-05",
        "2016-08-17",
        "2016-08-29",
        "2016-09-10",
        "2016-09-22",
    ]
    from collections import defaultdict

    families = defaultdict(list)
    for d in dates:
        track, sat = 143, "SENTINEL-1A"  # every real date shares both
        families[(track, sat)].append(d)
    print(f"  old grouping result: {dict(families)}")
    assert (
        len(families) == 1
    ), "confirms: old logic groups all 6 real dates into ONE family, missing the real split"
    print(
        "  Confirmed: old logic's `if len(families) <= 1: return []` would have fired here."
    )
    print("  PASS\n")


def test_satellite_key_extraction_bug_reproduction():
    print(
        "=== 3. Confirms the old satellite-key bug against the real scene structure ==="
    )
    real_properties = {
        "relativeOrbitNumber": 147,
        "platformShortName": "SENTINEL-1",
        "platformSerialIdentifier": "A",
        # no "platform" key, no "satellite" key -- matches the real confirmed dump
    }
    old_extraction = real_properties.get("platform") or real_properties.get("satellite")
    print(
        f"  old code's extraction result: {old_extraction!r} (should be 'A' or similar, got None)"
    )
    assert (
        old_extraction is None
    ), "confirms the old key names never matched the real object"
    print("  PASS\n")


def test_multiple_real_tracks_flagged_red():
    print(
        "=== 4. Genuinely mixed tracks in `selected` are still caught (defensive check) ==="
    )
    selected = [
        FakeScene(datetime(2016, 7, 24), track=143, satellite="SENTINEL-1A"),
        FakeScene(datetime(2016, 8, 5), track=45, satellite="SENTINEL-1A"),
    ]
    geometry_report = {"track": 143, "satellites": {"S1A"}}
    gate = make_gate()
    _, issues = gate._screen_burst_families(selected, geometry_report)
    codes = [i.code for i in issues]
    print(f"  issue codes: {codes}")
    assert "MULTIPLE_TRACKS_IN_SELECTION" in codes
    multi = next(i for i in issues if i.code == "MULTIPLE_TRACKS_IN_SELECTION")
    assert multi.severity == preflight.SEVERITY_RED
    print("  PASS\n")


def test_temporal_network_matches_generate_candidate_pairs():
    print(
        "=== 5. _check_temporal_network now reuses generate_candidate_pairs (no drift between the two) ==="
    )
    from pygeofetch.insar.timeseries import generate_candidate_pairs

    dates = [
        "2016-07-24",
        "2016-08-05",
        "2016-08-17",
        "2016-08-29",
        "2016-09-10",
        "2016-09-22",
    ]
    selected = [
        FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A")
        for d in dates
    ]

    gate = make_gate()
    gate.max_temporal_baseline_days = (
        15  # tight enough to create some real disconnection risk
    )
    issues = gate._check_temporal_network(selected)

    expected_pairs = generate_candidate_pairs(dates, max_temporal_baseline_days=15)
    print(
        f"  generate_candidate_pairs found {len(expected_pairs)} candidate pairs at 15-day baseline"
    )
    print(f"  temporal network issues: {[i.code for i in issues]}")
    # With a 12-day real repeat and a 15-day cutoff, every consecutive
    # pair should connect -- network should be fully connected.
    assert not any(i.code == "NETWORK_DISCONNECTED" for i in issues)
    print("  PASS\n")


def test_no_dead_yellow_to_green_function():
    print("=== 6. Dead YELLOW_TO_GREEN helper removed ===")
    assert not hasattr(preflight, "YELLOW_TO_GREEN")
    print("  PASS\n")


def test_full_run_end_to_end_mexico_city_scenario():
    print(
        "=== 7. Full gate.run() on the real Mexico City-shaped stack -- report is honest, not falsely clean ==="
    )
    dates = [
        "2016-07-24",
        "2016-08-05",
        "2016-08-17",
        "2016-08-29",
        "2016-09-10",
        "2016-09-22",
    ]
    selected = [
        FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A")
        for d in dates
    ]
    search_report = {
        "geometry_report": {"track": 143, "satellites": {"S1A"}, "dropped": {}},
        "hit_max_results": False,
        "raw_result_count": 6,
    }
    gate = make_gate()
    gate.min_coverage_fraction = (
        0.0  # fake geometry above isn't real AOI-shaped; don't drop on that here
    )
    report = gate.run(selected, search_report)
    print(report.summary())
    codes = [i.code for i in report.issues]
    assert (
        "BURST_FAMILY_RISK_UNASSESSED" in codes
    ), "the honest advisory must always be present"
    print("  PASS\n")


def test_summary_reminds_caller_when_scenes_were_actually_excluded():
    print(
        "=== 7b. REAL BUG regression: summary() loudly reminds the caller to use report.selected when something was dropped ==="
    )
    # Reproduces the exact real notebook bug: preflight correctly
    # excluded real scenes (confirmed directly, found in an actual
    # uploaded notebook -- a cell literally titled "Download the real,
    # filtered scenes" called client.download(selected, ...) using the
    # ORIGINAL pre-preflight variable, never reassigned from
    # report.selected, so every excluded scene got downloaded anyway).
    dates = [
        "2016-07-24",
        "2016-08-05",
        "2016-08-17",
        "2016-08-29",
        "2016-09-10",
        "2016-09-22",
    ]
    selected = [
        FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A")
        for d in dates
    ]
    search_report = {
        "geometry_report": {"track": 143, "satellites": {"S1A"}, "dropped": {}},
        "hit_max_results": False,
        "raw_result_count": 6,
    }
    gate = make_gate(attempt_real_burst_check=True)
    gate.min_coverage_fraction = 0.0
    fake_family_report = {
        "good_dates": dates[:5],
        "bridge_only_dates": [dates[5]],
        "used_majority_only": True,
        "majority_self_connected": True,
        "majority_internal_bridges": [],
        "sync_results": [],
    }
    gate._try_real_burst_family_check = MagicMock(
        return_value=(dates[:5], fake_family_report)
    )

    report = gate.run(selected, search_report)
    print(report.summary())

    assert report.original_count == 6
    assert len(report.selected) == 5
    assert "[REMINDER]" in report.summary()
    assert "report.selected" in report.summary()
    assert "1 of 6 scene(s) were excluded" in report.summary()
    print(
        "  PASS -- the exact real mistake (using the original list, not report.selected) now gets a loud, unmissable reminder\n"
    )


def test_summary_stays_quiet_when_nothing_was_excluded():
    print(
        "=== 7c. summary() does NOT show the reminder when nothing was actually dropped (no false alarms) ==="
    )
    dates = ["2016-07-24", "2016-08-05", "2016-08-17"]
    selected = [
        FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A")
        for d in dates
    ]
    search_report = {
        "geometry_report": {"track": 143, "satellites": {"S1A"}, "dropped": {}},
        "hit_max_results": False,
        "raw_result_count": 3,
    }
    gate = make_gate(
        attempt_real_burst_check=False
    )  # skip the real check -> honest-unassessed path, nothing dropped
    gate.min_coverage_fraction = 0.0

    report = gate.run(selected, search_report)
    print(report.summary())

    assert report.original_count == 3
    assert len(report.selected) == 3
    assert (
        "[REMINDER]" not in report.summary()
    ), "should not warn when nothing was actually excluded"
    print("  PASS\n")


def test_original_count_not_confused_by_truncation_widening():
    print(
        "=== 7d. A truncation-driven re-search that GROWS the pool doesn't trigger a false 'dropped' reminder ==="
    )
    dates_small = ["2016-07-24", "2016-08-05"]
    selected_small = [
        FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A")
        for d in dates_small
    ]
    dates_wide = ["2016-07-24", "2016-08-05", "2016-08-17", "2016-08-29"]
    selected_wide = [
        FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A")
        for d in dates_wide
    ]
    search_report = {
        "geometry_report": {"track": 143, "satellites": {"S1A"}, "dropped": {}},
        "hit_max_results": True,  # triggers the truncation re-search path
        "raw_result_count": 2,
    }
    gate = make_gate(attempt_real_burst_check=False)
    gate.min_coverage_fraction = 0.0
    gate._check_truncation = MagicMock(
        return_value=(
            selected_wide,
            {**search_report, "hit_max_results": False},
            None,
        )
    )

    report = gate.run(selected_small, search_report)
    print(report.summary())
    assert (
        report.original_count == 4
    ), "baseline should be captured AFTER truncation resolves (the grown pool), not before"
    assert (
        "[REMINDER]" not in report.summary()
    ), "growing the pool via re-search is not a 'drop' and must not warn"
    print("  PASS\n")


def test_real_check_success_produces_burst_family_detected():
    print(
        "=== 8. Real check success -> BURST_FAMILY_DETECTED, AND selected is actually filtered ==="
    )
    dates = [
        "2016-07-24",
        "2016-08-05",
        "2016-08-17",
        "2016-08-29",
        "2016-09-10",
        "2016-09-22",
    ]
    selected = [
        FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A")
        for d in dates
    ]
    geometry_report = {"track": 143, "satellites": {"S1A"}}

    gate = make_gate(attempt_real_burst_check=True)
    fake_family_report = {
        "good_dates": dates[:5],
        "bridge_only_dates": [dates[5]],
        "used_majority_only": True,
        "sync_results": [],
    }
    gate._try_real_burst_family_check = MagicMock(
        return_value=(dates[:5], fake_family_report)
    )

    filtered_selected, issues = gate._screen_burst_families(selected, geometry_report)
    codes = [i.code for i in issues]
    print(f"  issue codes: {codes}")
    assert "BURST_FAMILY_DETECTED" in codes
    assert (
        "BURST_FAMILY_RISK_UNASSESSED" not in codes
    ), "should NOT also carry the unassessed advisory when a real result is available"
    detected = next(i for i in issues if i.code == "BURST_FAMILY_DETECTED")
    assert detected.severity == preflight.SEVERITY_GREEN
    assert detected.auto_fixed is True
    assert "used exclusively" in detected.message
    # Real bug prevention: the copernicus_nodes version must be visible
    # directly in this message, since this is the block the user has
    # consistently shared every time -- not just at module-import time,
    # which can be missed if a shared log snippet starts partway through.
    from pygeofetch.providers.copernicus_nodes import MODULE_VERSION

    assert f"[copernicus_nodes {MODULE_VERSION}]" in detected.message, (
        "the running copernicus_nodes version must be visible in the final "
        "summary block, not just at import time"
    )

    # REAL BUG regression: auto_fixed=True must mean the exclusion was
    # actually APPLIED to the returned selection, not just described in
    # the message text.
    filtered_dates = sorted(str(s.datetime)[:10] for s in filtered_selected)
    print(f"  filtered selected dates: {filtered_dates}")
    assert filtered_dates == sorted(dates[:5]), (
        "the minority date (2016-09-22) must actually be excluded from "
        "the returned selection, not just mentioned in the message"
    )
    assert len(filtered_selected) == 5
    gate._try_real_burst_family_check.assert_called_once()
    print("  PASS\n")


def test_compromise_case_uses_yellow_not_green_and_auto_fixed_false():
    print(
        "=== 8c. Third-party review fix: compromise (minority KEPT) is YELLOW + auto_fixed=False, not GREEN ==="
    )
    # Reproduces the real run a third-party review flagged: majority
    # family too small to exclude the minority, so ALL dates get kept
    # (a real compromise, not a fix) -- this must NOT be reported the
    # same way as a genuine, clean exclusion.
    dates = [
        "2016-11-03",
        "2016-11-15",
        "2016-11-27",
        "2016-12-09",
        "2016-12-15",
        "2016-12-21",
    ]
    selected = [
        FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1B")
        for d in dates
    ]
    geometry_report = {"track": 143, "satellites": {"S1A", "S1B"}}

    gate = make_gate(attempt_real_burst_check=True)
    fake_family_report = {
        "good_dates": dates[:5],
        "bridge_only_dates": [dates[5]],
        "used_majority_only": False,  # majority too small -- real compromise
        "sync_results": [],
    }
    gate._try_real_burst_family_check = MagicMock(
        return_value=(dates, fake_family_report)
    )  # ALL 6 kept

    filtered_selected, issues = gate._screen_burst_families(selected, geometry_report)
    detected = next(i for i in issues if i.code == "BURST_FAMILY_DETECTED")
    print(f"  severity: {detected.severity}, auto_fixed: {detected.auto_fixed}")
    print(f"  message: {detected.message}")

    assert (
        detected.severity == preflight.SEVERITY_YELLOW
    ), "a real compromise (minority kept) must be YELLOW, not GREEN"
    assert (
        detected.auto_fixed is False
    ), "nothing was actually excluded -- this is not a real auto-fix"
    assert "compromise, not a fix" in detected.message
    assert (
        len(filtered_selected) == 6
    ), "no date should have been excluded in the compromise case"
    print("  PASS\n")


def test_resolved_case_still_green_and_auto_fixed_true():
    print(
        "=== 8d. Genuine exclusion (majority large enough) is still correctly GREEN + auto_fixed=True ==="
    )
    dates = [
        "2016-07-24",
        "2016-08-05",
        "2016-08-17",
        "2016-08-29",
        "2016-09-10",
        "2016-09-22",
    ]
    selected = [
        FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A")
        for d in dates
    ]
    geometry_report = {"track": 143, "satellites": {"S1A"}}

    gate = make_gate(attempt_real_burst_check=True)
    fake_family_report = {
        "good_dates": dates[:5],
        "bridge_only_dates": [dates[5]],
        "used_majority_only": True,  # genuinely resolved
        "sync_results": [],
    }
    gate._try_real_burst_family_check = MagicMock(
        return_value=(dates[:5], fake_family_report)
    )

    _, issues = gate._screen_burst_families(selected, geometry_report)
    detected = next(i for i in issues if i.code == "BURST_FAMILY_DETECTED")
    assert (
        detected.severity == preflight.SEVERITY_GREEN
    ), "a genuine exclusion should still be GREEN"
    assert detected.auto_fixed is True
    assert "genuinely resolved" in detected.message
    print("  PASS\n")


def test_majority_internal_bridge_surfaced_in_preflight_message():
    print(
        "=== 8f. Real scenario: exclusion happens, but internal majority bridge is honestly surfaced (not implied clean) ==="
    )
    # Reproduces exactly what a direct user question surfaced: excluding
    # a known-poor minority date doesn't automatically mean the
    # remaining majority is bridge-free.
    dates = [
        "2016-11-03",
        "2016-11-15",
        "2016-11-27",
        "2016-12-09",
        "2016-12-15",
        "2016-12-21",
    ]
    selected = [
        FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1B")
        for d in dates
    ]
    geometry_report = {"track": 143, "satellites": {"S1A", "S1B"}}

    gate = make_gate(attempt_real_burst_check=True)
    fake_family_report = {
        "good_dates": dates[:5],
        "bridge_only_dates": [dates[5]],
        "used_majority_only": True,
        "majority_self_connected": True,
        "majority_internal_bridges": [("2016-11-03", "2016-12-09", -7.1)],
        "sync_results": [],
    }
    gate._try_real_burst_family_check = MagicMock(
        return_value=(dates[:5], fake_family_report)
    )

    filtered, issues = gate._screen_burst_families(selected, geometry_report)
    detected = next(i for i in issues if i.code == "BURST_FAMILY_DETECTED")
    print(f"  message: {detected.message}")
    assert (
        detected.severity == preflight.SEVERITY_GREEN
    ), "exclusion still genuinely happened -- GREEN is correct"
    assert "2016-12-21" not in [
        str(s.datetime)[:10] for s in filtered
    ], "minority date should still be excluded"
    assert "internal bridge" in detected.message
    assert (
        "('2016-11-03', '2016-12-09', -7.1)" in detected.message
    ), "the specific internal bridge must be named, not hidden"
    print("  PASS\n")


def test_pair_level_fraction_reported_not_just_date_level_split():
    print(
        "=== 8e. Third-party review fix: pair-level within-requirement fraction is now reported ==="
    )
    # Reproduces the real finding: only 4/15 (27%) of real candidate
    # pairs were within the 5ms requirement, including a pair WITHIN
    # the 5-date majority family itself -- the date-level split alone
    # ('5 majority / 1 minority') doesn't surface how pervasive this is.
    from types import SimpleNamespace

    dates = [
        "2016-11-03",
        "2016-11-15",
        "2016-11-27",
        "2016-12-09",
        "2016-12-15",
        "2016-12-21",
    ]
    selected = [
        FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1B")
        for d in dates
    ]
    geometry_report = {"track": 143, "satellites": {"S1A", "S1B"}}

    within_flags = [
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
    ]
    fake_sync_results = [SimpleNamespace(within_requirement=w) for w in within_flags]
    assert len(fake_sync_results) == 15

    gate = make_gate(attempt_real_burst_check=True)
    fake_family_report = {
        "good_dates": dates[:5],
        "bridge_only_dates": [dates[5]],
        "used_majority_only": False,
        "sync_results": fake_sync_results,
    }
    gate._try_real_burst_family_check = MagicMock(
        return_value=(dates, fake_family_report)
    )

    _, issues = gate._screen_burst_families(selected, geometry_report)
    detected = next(i for i in issues if i.code == "BURST_FAMILY_DETECTED")
    print(f"  message: {detected.message}")
    assert (
        "4/15" in detected.message
    ), "the real pair-level fraction must appear in the message"
    assert "27%" in detected.message
    print("  PASS\n")


def test_burst_family_exclusion_actually_applied_end_to_end_via_run():
    print(
        "=== 8b. REAL BUG regression: report.selected reflects the burst-family exclusion via the full run() path ==="
    )
    dates = [
        "2016-07-24",
        "2016-08-05",
        "2016-08-17",
        "2016-08-29",
        "2016-09-10",
        "2016-09-22",
    ]
    selected = [
        FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A")
        for d in dates
    ]
    search_report = {
        "geometry_report": {"track": 143, "satellites": {"S1A"}, "dropped": {}},
        "hit_max_results": False,
        "raw_result_count": 6,
    }
    gate = make_gate(attempt_real_burst_check=True)
    gate.min_coverage_fraction = 0.0  # fake geometry isn't real AOI-shaped

    fake_family_report = {
        "good_dates": dates[:5],
        "bridge_only_dates": [dates[5]],
        "used_majority_only": True,
        "sync_results": [],
    }
    gate._try_real_burst_family_check = MagicMock(
        return_value=(dates[:5], fake_family_report)
    )

    report = gate.run(selected, search_report)
    result_dates = sorted(str(s.datetime)[:10] for s in report.selected)
    print(f"  report.selected dates: {result_dates}")
    assert result_dates == sorted(dates[:5]), (
        "report.selected -- what a real caller actually downloads -- must reflect the "
        "burst-family exclusion, not just the issue's message text"
    )
    assert "2016-09-22" not in result_dates
    print("  PASS\n")


def test_real_check_failure_falls_back_to_unassessed():
    print("=== 9. Real check returns None -> honest fallback still fires ===")
    dates = ["2016-07-24", "2016-08-05"]
    selected = [
        FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A")
        for d in dates
    ]
    geometry_report = {"track": 143, "satellites": {"S1A"}}

    gate = make_gate(attempt_real_burst_check=True)
    gate._try_real_burst_family_check = MagicMock(return_value=None)

    filtered_selected, issues = gate._screen_burst_families(selected, geometry_report)
    codes = [i.code for i in issues]
    print(f"  issue codes: {codes}")
    assert "BURST_FAMILY_RISK_UNASSESSED" in codes
    unassessed = next(i for i in issues if i.code == "BURST_FAMILY_RISK_UNASSESSED")
    assert (
        "attempted" in unassessed.message.lower()
        or "did not succeed" in unassessed.message.lower()
    )
    # When the real check fails, selected must pass through UNCHANGED --
    # no filtering should happen based on a check that didn't succeed.
    assert [s.datetime for s in filtered_selected] == [s.datetime for s in selected]
    print("  PASS\n")


def test_attempt_real_burst_check_false_skips_entirely():
    print("=== 10. attempt_real_burst_check=False never even calls the real check ===")
    dates = ["2016-07-24", "2016-08-05"]
    selected = [
        FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A")
        for d in dates
    ]
    geometry_report = {"track": 143, "satellites": {"S1A"}}

    gate = make_gate(attempt_real_burst_check=False)
    gate._try_real_burst_family_check = MagicMock(
        return_value=("should not be used", {})
    )

    _, issues = gate._screen_burst_families(selected, geometry_report)
    gate._try_real_burst_family_check.assert_not_called()
    codes = [i.code for i in issues]
    assert "BURST_FAMILY_RISK_UNASSESSED" in codes
    unassessed = next(i for i in issues if i.code == "BURST_FAMILY_RISK_UNASSESSED")
    assert "attempt_real_burst_check=False" in unassessed.message
    print("  PASS\n")


def test_try_real_burst_family_check_orchestration_partial_failures():
    print(
        "=== 11. _try_real_burst_family_check tolerates per-date failures, never raises ==="
    )
    dates = ["2016-07-24", "2016-08-05", "2016-08-17"]
    selected = [
        FakeScene(
            datetime.strptime(d, "%Y-%m-%d"),
            track=143,
            satellite="SENTINEL-1A",
            properties={"relativeOrbitNumber": 143, "name": f"S1A_{d}.SAFE"},
        )
        for d in dates
    ]
    gate = make_gate()

    def fake_fetch_annotation_zip(client, sat, work_dir, **kw):
        if "08-05" in sat.properties["name"]:
            raise RuntimeError("simulated real annotation-fetch failure for this date")
        return f"/fake/{sat.properties['name']}.zip"

    def fake_fetch_orbit_file(product_name=None, output_dir=None, orbit_type=None):
        return f"/fake/{product_name}.EOF"

    fake_chosen = ["2016-07-24", "2016-08-17"]
    fake_report = {
        "good_dates": fake_chosen,
        "bridge_only_dates": [],
        "used_majority_only": True,
        "sync_results": [],
    }

    with (
        patch(
            "pygeofetch.providers.copernicus_nodes.fetch_annotation_zip",
            side_effect=fake_fetch_annotation_zip,
        ),
        patch(
            "pygeofetch.core.orbits.fetch_orbit_file", side_effect=fake_fetch_orbit_file
        ),
        patch(
            "pygeofetch.insar.stack_selection.select_burst_synchronized_dates",
            return_value=(fake_chosen, fake_report),
        ) as mock_select,
    ):
        result = gate._try_real_burst_family_check(selected)

    print(f"  result: {result}")
    assert result is not None
    chosen, report = result
    assert chosen == fake_chosen
    # confirm the failed date (08-05) was excluded from what got passed onward
    call_args = mock_select.call_args
    usable_dates_passed = call_args[0][0]
    print(
        f"  usable dates passed to select_burst_synchronized_dates: {usable_dates_passed}"
    )
    assert (
        "2016-08-05" not in usable_dates_passed
    ), "the date whose annotation fetch failed should be excluded, not crash the whole check"
    assert "2016-07-24" in usable_dates_passed and "2016-08-17" in usable_dates_passed
    print("  PASS\n")


def test_try_real_burst_family_check_too_few_usable_dates_returns_none():
    print(
        "=== 12. _try_real_burst_family_check returns None (not a crash) when too few dates succeed ==="
    )
    dates = ["2016-07-24", "2016-08-05"]
    selected = [
        FakeScene(
            datetime.strptime(d, "%Y-%m-%d"),
            track=143,
            satellite="SENTINEL-1A",
            properties={"relativeOrbitNumber": 143, "name": f"S1A_{d}.SAFE"},
        )
        for d in dates
    ]
    gate = make_gate()

    with patch(
        "pygeofetch.providers.copernicus_nodes.fetch_annotation_zip",
        side_effect=RuntimeError("no network here"),
    ):
        result = gate._try_real_burst_family_check(selected)

    print(f"  result: {result}")
    assert result is None
    print("  PASS\n")


def test_try_real_burst_family_check_unexpected_exception_returns_none_not_raise():
    print(
        "=== 13. A genuinely unexpected exception inside the real check is swallowed, not propagated ==="
    )
    dates = ["2016-07-24", "2016-08-05", "2016-08-17"]
    selected = [
        FakeScene(
            datetime.strptime(d, "%Y-%m-%d"),
            track=143,
            satellite="SENTINEL-1A",
            properties={"relativeOrbitNumber": 143, "name": f"S1A_{d}.SAFE"},
        )
        for d in dates
    ]
    gate = make_gate()

    def fake_fetch_annotation_zip(client, sat, work_dir, **kw):
        return f"/fake/{sat.properties['name']}.zip"

    def fake_fetch_orbit_file(product_name=None, output_dir=None, orbit_type=None):
        return f"/fake/{product_name}.EOF"

    with (
        patch(
            "pygeofetch.providers.copernicus_nodes.fetch_annotation_zip",
            side_effect=fake_fetch_annotation_zip,
        ),
        patch(
            "pygeofetch.core.orbits.fetch_orbit_file", side_effect=fake_fetch_orbit_file
        ),
        patch(
            "pygeofetch.insar.stack_selection.select_burst_synchronized_dates",
            side_effect=RuntimeError("simulated unexpected failure"),
        ),
    ):
        result = gate._try_real_burst_family_check(selected)  # must not raise

    print(f"  result: {result}")
    assert result is None
    print("  PASS\n")


if __name__ == "__main__":
    test_reproduces_mexico_city_same_track_split_not_missed()
    test_old_buggy_grouping_would_have_returned_empty()
    test_satellite_key_extraction_bug_reproduction()
    test_multiple_real_tracks_flagged_red()
    test_temporal_network_matches_generate_candidate_pairs()
    test_no_dead_yellow_to_green_function()
    test_full_run_end_to_end_mexico_city_scenario()
    test_summary_reminds_caller_when_scenes_were_actually_excluded()
    test_summary_stays_quiet_when_nothing_was_excluded()
    test_original_count_not_confused_by_truncation_widening()
    test_real_check_success_produces_burst_family_detected()
    test_compromise_case_uses_yellow_not_green_and_auto_fixed_false()
    test_resolved_case_still_green_and_auto_fixed_true()
    test_majority_internal_bridge_surfaced_in_preflight_message()
    test_pair_level_fraction_reported_not_just_date_level_split()
    test_burst_family_exclusion_actually_applied_end_to_end_via_run()
    test_real_check_failure_falls_back_to_unassessed()
    test_attempt_real_burst_check_false_skips_entirely()
    test_try_real_burst_family_check_orchestration_partial_failures()
    test_try_real_burst_family_check_too_few_usable_dates_returns_none()
    test_try_real_burst_family_check_unexpected_exception_returns_none_not_raise()
    print("ALL TESTS PASSED")
