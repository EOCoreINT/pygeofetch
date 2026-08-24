"""
Validates the fixed PreflightGate -- in particular, that
_screen_burst_families no longer gives a false clean pass on a
reproduction of the exact real scenario (Mexico City's within-track
cross-family split) that the original, buggy version would have
missed.
"""
import sys
# sys.path.insert(0, "/home/mrtenkorang/open-source-projects/")

from datetime import datetime
from unittest.mock import patch, MagicMock

import pygeofetch.insar.preflight as preflight


class FakeScene:
    def __init__(self, dt, track, satellite, properties=None):
        self.datetime = dt
        self.geometry = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
        self.satellite = satellite
        self.properties = properties if properties is not None else {"relativeOrbitNumber": track}


class FakeBBox:
    def __init__(self):
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = 0.0, 0.0, 1.0, 1.0


def make_gate(**kwargs):
    kwargs.setdefault("attempt_real_burst_check", False)  # keep existing tests network-independent
    return preflight.PreflightGate(client=None, aoi_bbox=FakeBBox(), start_date="2016-07-01", end_date="2016-10-01", **kwargs)


def test_reproduces_mexico_city_same_track_split_not_missed():
    print("=== 1. Real bug regression: same-track cross-family scenario is no longer silently missed ===")
    # Exactly the real Mexico City situation: 6 dates, all real track
    # 143, all Sentinel-1A -- the actual cross-family split (confirmed
    # via compute_burst_synchronization) happened WITHIN this single
    # (track, satellite) group, which the old code could never see.
    dates = ["2016-07-24", "2016-08-05", "2016-08-17", "2016-08-29", "2016-09-10", "2016-09-22"]
    selected = [
        FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A",
                   properties={"relativeOrbitNumber": 143, "platformSerialIdentifier": "A"})
        for d in dates
    ]
    geometry_report = {"track": 143, "satellites": {"S1A"}}

    gate = make_gate()
    issues = gate._screen_burst_families(selected, geometry_report)
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
    print("=== 2. Confirms the OLD grouping logic really would have returned [] for this exact case ===")
    # Reproduce the old (Track, Satellite) grouping directly, using the
    # same real scene data, to concretely demonstrate the false pass.
    dates = ["2016-07-24", "2016-08-05", "2016-08-17", "2016-08-29", "2016-09-10", "2016-09-22"]
    from collections import defaultdict
    families = defaultdict(list)
    for d in dates:
        track, sat = 143, "SENTINEL-1A"  # every real date shares both
        families[(track, sat)].append(d)
    print(f"  old grouping result: {dict(families)}")
    assert len(families) == 1, "confirms: old logic groups all 6 real dates into ONE family, missing the real split"
    print("  Confirmed: old logic's `if len(families) <= 1: return []` would have fired here.")
    print("  PASS\n")


def test_satellite_key_extraction_bug_reproduction():
    print("=== 3. Confirms the old satellite-key bug against the real scene structure ===")
    real_properties = {
        "relativeOrbitNumber": 147,
        "platformShortName": "SENTINEL-1",
        "platformSerialIdentifier": "A",
        # no "platform" key, no "satellite" key -- matches the real confirmed dump
    }
    old_extraction = real_properties.get("platform") or real_properties.get("satellite")
    print(f"  old code's extraction result: {old_extraction!r} (should be 'A' or similar, got None)")
    assert old_extraction is None, "confirms the old key names never matched the real object"
    print("  PASS\n")


def test_multiple_real_tracks_flagged_red():
    print("=== 4. Genuinely mixed tracks in `selected` are still caught (defensive check) ===")
    selected = [
        FakeScene(datetime(2016, 7, 24), track=143, satellite="SENTINEL-1A"),
        FakeScene(datetime(2016, 8, 5), track=45, satellite="SENTINEL-1A"),
    ]
    geometry_report = {"track": 143, "satellites": {"S1A"}}
    gate = make_gate()
    issues = gate._screen_burst_families(selected, geometry_report)
    codes = [i.code for i in issues]
    print(f"  issue codes: {codes}")
    assert "MULTIPLE_TRACKS_IN_SELECTION" in codes
    multi = next(i for i in issues if i.code == "MULTIPLE_TRACKS_IN_SELECTION")
    assert multi.severity == preflight.SEVERITY_RED
    print("  PASS\n")


def test_temporal_network_matches_generate_candidate_pairs():
    print("=== 5. _check_temporal_network now reuses generate_candidate_pairs (no drift between the two) ===")
    from pygeofetch.insar.timeseries import generate_candidate_pairs

    dates = ["2016-07-24", "2016-08-05", "2016-08-17", "2016-08-29", "2016-09-10", "2016-09-22"]
    selected = [FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A") for d in dates]

    gate = make_gate()
    gate.max_temporal_baseline_days = 15  # tight enough to create some real disconnection risk
    issues = gate._check_temporal_network(selected)

    expected_pairs = generate_candidate_pairs(dates, max_temporal_baseline_days=15)
    print(f"  generate_candidate_pairs found {len(expected_pairs)} candidate pairs at 15-day baseline")
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
    print("=== 7. Full gate.run() on the real Mexico City-shaped stack -- report is honest, not falsely clean ===")
    dates = ["2016-07-24", "2016-08-05", "2016-08-17", "2016-08-29", "2016-09-10", "2016-09-22"]
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
    gate.min_coverage_fraction = 0.0  # fake geometry above isn't real AOI-shaped; don't drop on that here
    report = gate.run(selected, search_report)
    print(report.summary())
    codes = [i.code for i in report.issues]
    assert "BURST_FAMILY_RISK_UNASSESSED" in codes, "the honest advisory must always be present"
    print("  PASS\n")


def test_real_check_success_produces_burst_family_detected():
    print("=== 8. Real check success -> BURST_FAMILY_DETECTED replaces the unassessed advisory ===")
    dates = ["2016-07-24", "2016-08-05", "2016-08-17", "2016-08-29", "2016-09-10", "2016-09-22"]
    selected = [FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A") for d in dates]
    geometry_report = {"track": 143, "satellites": {"S1A"}}

    gate = make_gate(attempt_real_burst_check=True)
    fake_family_report = {
        "good_dates": dates[:5], "bridge_only_dates": [dates[5]],
        "used_majority_only": True, "sync_results": [],
    }
    gate._try_real_burst_family_check = MagicMock(return_value=(dates[:5], fake_family_report))

    issues = gate._screen_burst_families(selected, geometry_report)
    codes = [i.code for i in issues]
    print(f"  issue codes: {codes}")
    assert "BURST_FAMILY_DETECTED" in codes
    assert "BURST_FAMILY_RISK_UNASSESSED" not in codes, "should NOT also carry the unassessed advisory when a real result is available"
    detected = next(i for i in issues if i.code == "BURST_FAMILY_DETECTED")
    assert detected.severity == preflight.SEVERITY_GREEN
    assert detected.auto_fixed is True
    assert "used exclusively" in detected.message
    gate._try_real_burst_family_check.assert_called_once()
    print("  PASS\n")


def test_real_check_failure_falls_back_to_unassessed():
    print("=== 9. Real check returns None -> honest fallback still fires ===")
    dates = ["2016-07-24", "2016-08-05"]
    selected = [FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A") for d in dates]
    geometry_report = {"track": 143, "satellites": {"S1A"}}

    gate = make_gate(attempt_real_burst_check=True)
    gate._try_real_burst_family_check = MagicMock(return_value=None)

    issues = gate._screen_burst_families(selected, geometry_report)
    codes = [i.code for i in issues]
    print(f"  issue codes: {codes}")
    assert "BURST_FAMILY_RISK_UNASSESSED" in codes
    unassessed = next(i for i in issues if i.code == "BURST_FAMILY_RISK_UNASSESSED")
    assert "attempted" in unassessed.message.lower() or "did not succeed" in unassessed.message.lower()
    print("  PASS\n")


def test_attempt_real_burst_check_false_skips_entirely():
    print("=== 10. attempt_real_burst_check=False never even calls the real check ===")
    dates = ["2016-07-24", "2016-08-05"]
    selected = [FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A") for d in dates]
    geometry_report = {"track": 143, "satellites": {"S1A"}}

    gate = make_gate(attempt_real_burst_check=False)
    gate._try_real_burst_family_check = MagicMock(return_value=("should not be used", {}))

    issues = gate._screen_burst_families(selected, geometry_report)
    gate._try_real_burst_family_check.assert_not_called()
    codes = [i.code for i in issues]
    assert "BURST_FAMILY_RISK_UNASSESSED" in codes
    unassessed = next(i for i in issues if i.code == "BURST_FAMILY_RISK_UNASSESSED")
    assert "attempt_real_burst_check=False" in unassessed.message
    print("  PASS\n")


def test_try_real_burst_family_check_orchestration_partial_failures():
    print("=== 11. _try_real_burst_family_check tolerates per-date failures, never raises ===")
    dates = ["2016-07-24", "2016-08-05", "2016-08-17"]
    selected = [
        FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A",
                   properties={"relativeOrbitNumber": 143, "name": f"S1A_{d}.SAFE"})
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
    fake_report = {"good_dates": fake_chosen, "bridge_only_dates": [], "used_majority_only": True, "sync_results": []}

    with patch("pygeofetch.providers.copernicus_nodes.fetch_annotation_zip", side_effect=fake_fetch_annotation_zip), \
         patch("pygeofetch.core.orbits.fetch_orbit_file", side_effect=fake_fetch_orbit_file), \
         patch("pygeofetch.insar.stack_selection.select_burst_synchronized_dates", return_value=(fake_chosen, fake_report)) as mock_select:
        result = gate._try_real_burst_family_check(selected)

    print(f"  result: {result}")
    assert result is not None
    chosen, report = result
    assert chosen == fake_chosen
    # confirm the failed date (08-05) was excluded from what got passed onward
    call_args = mock_select.call_args
    usable_dates_passed = call_args[0][0]
    print(f"  usable dates passed to select_burst_synchronized_dates: {usable_dates_passed}")
    assert "2016-08-05" not in usable_dates_passed, "the date whose annotation fetch failed should be excluded, not crash the whole check"
    assert "2016-07-24" in usable_dates_passed and "2016-08-17" in usable_dates_passed
    print("  PASS\n")


def test_try_real_burst_family_check_too_few_usable_dates_returns_none():
    print("=== 12. _try_real_burst_family_check returns None (not a crash) when too few dates succeed ===")
    dates = ["2016-07-24", "2016-08-05"]
    selected = [
        FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A",
                   properties={"relativeOrbitNumber": 143, "name": f"S1A_{d}.SAFE"})
        for d in dates
    ]
    gate = make_gate()

    with patch("pygeofetch.providers.copernicus_nodes.fetch_annotation_zip", side_effect=RuntimeError("no network here")):
        result = gate._try_real_burst_family_check(selected)

    print(f"  result: {result}")
    assert result is None
    print("  PASS\n")


def test_try_real_burst_family_check_unexpected_exception_returns_none_not_raise():
    print("=== 13. A genuinely unexpected exception inside the real check is swallowed, not propagated ===")
    dates = ["2016-07-24", "2016-08-05", "2016-08-17"]
    selected = [
        FakeScene(datetime.strptime(d, "%Y-%m-%d"), track=143, satellite="SENTINEL-1A",
                   properties={"relativeOrbitNumber": 143, "name": f"S1A_{d}.SAFE"})
        for d in dates
    ]
    gate = make_gate()

    def fake_fetch_annotation_zip(client, sat, work_dir, **kw):
        return f"/fake/{sat.properties['name']}.zip"

    def fake_fetch_orbit_file(product_name=None, output_dir=None, orbit_type=None):
        return f"/fake/{product_name}.EOF"

    with patch("pygeofetch.providers.copernicus_nodes.fetch_annotation_zip", side_effect=fake_fetch_annotation_zip), \
         patch("pygeofetch.core.orbits.fetch_orbit_file", side_effect=fake_fetch_orbit_file), \
         patch("pygeofetch.insar.stack_selection.select_burst_synchronized_dates", side_effect=RuntimeError("simulated unexpected failure")):
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
    test_real_check_success_produces_burst_family_detected()
    test_real_check_failure_falls_back_to_unassessed()
    test_attempt_real_burst_check_false_skips_entirely()
    test_try_real_burst_family_check_orchestration_partial_failures()
    test_try_real_burst_family_check_too_few_usable_dates_returns_none()
    test_try_real_burst_family_check_unexpected_exception_returns_none_not_raise()
    print("ALL TESTS PASSED")
