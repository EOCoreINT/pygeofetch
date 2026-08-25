"""
Validates search_and_select_consistent_stack() and
select_burst_synchronized_dates() -- the two new consolidated library
functions, replacing hand-copied notebook boilerplate that had real,
confirmed bugs independently discovered twice (Obuasi's truncation and
zero-coverage dedup bug; Mexico City's duplicate-slice dedup bug;
Mexico City's cross-family burst-sync issue).
"""
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

from pygeofetch.insar import stack_selection
from pygeofetch.models import BoundingBox


class FakeScene:
    def __init__(self, dt, geometry, track):
        self.datetime = dt
        self.geometry = geometry
        self.properties = {"relativeOrbitNumber": track, "name": "S1A_TEST"}
        self.id = "S1A_TEST"


def poly(cx, cy, half=1.0):
    return {"type": "Polygon", "coordinates": [[
        [cx - half, cy - half], [cx + half, cy - half],
        [cx + half, cy + half], [cx - half, cy + half], [cx - half, cy - half],
    ]]}


AOI = BoundingBox(min_lon=-1.0, min_lat=-1.0, max_lon=1.0, max_lat=1.0)


def test_reproduces_mexico_city_duplicate_slice_fix():
    print("=== 1. Reproduces the real Mexico City duplicate-slice dedup fix ===")
    d = datetime(2016, 11, 9)
    scenes = [
        FakeScene(d, poly(50, 50), track=143),   # non-covering slice, arrives first
        FakeScene(d, poly(0, 0), track=143),     # covering slice, arrives second
        FakeScene(d + timedelta(days=6), poly(0, 0), track=143),
    ]

    class FakeClient:
        def search(self, query, providers=None):
            return scenes

    selected, report = stack_selection.search_and_select_consistent_stack(
        FakeClient(), AOI, "2016-11-01", "2016-11-15", max_results=100,
    )
    dates = sorted(str(s.datetime)[:10] for s in selected)
    print(f"  selected dates: {dates}")
    print(f"  multi_candidate_dates: {report['multi_candidate_dates']}")
    print(f"  picked_non_first: {report['picked_non_first']}")

    assert report["picked_non_first"] == 1, "should detect the one date that needed the fix"
    assert "2016-11-09" in dates, "the covering slice should have been kept"
    kept_2016_11_09 = next(s for s in selected if str(s.datetime)[:10] == "2016-11-09")
    assert kept_2016_11_09.geometry == poly(0, 0), "should have kept the COVERING slice, not the first-arriving one"
    print("  PASS\n")


def test_reproduces_obuasi_truncation_warning():
    print("=== 2. Reproduces the real Obuasi max_results truncation warning ===")
    d0 = datetime(2019, 1, 1)
    scenes = [FakeScene(d0 + timedelta(days=12 * i), poly(0, 0), track=45) for i in range(5)]

    class FakeClient:
        def search(self, query, providers=None):
            return scenes  # exactly matches max_results, simulating truncation

    selected, report = stack_selection.search_and_select_consistent_stack(
        FakeClient(), AOI, "2019-01-01", "2019-06-01", max_results=len(scenes),
    )
    print(f"  hit_max_results: {report['hit_max_results']}")
    assert report["hit_max_results"] is True
    print("  PASS\n")


def test_zero_result_search_raises_clearly():
    print("=== 3. Empty search raises a clear, honest error, not a downstream crash ===")

    class FakeClient:
        def search(self, query, providers=None):
            return []

    try:
        stack_selection.search_and_select_consistent_stack(FakeClient(), AOI, "2019-01-01", "2019-06-01")
        raise AssertionError("expected ValueError for zero results")
    except ValueError as exc:
        print(f"  correctly raised: {exc}")
    print("  PASS\n")


def test_final_coverage_verification_drops_genuine_gaps():
    print("=== 4. Final verification drops a genuine gap even after coverage-aware dedup ===")
    d0 = datetime(2020, 1, 1)
    scenes = [
        FakeScene(d0, poly(0, 0), track=45),               # good
        FakeScene(d0 + timedelta(days=12), poly(50, 50), track=45),  # ONLY candidate for this date, and it's bad
        FakeScene(d0 + timedelta(days=24), poly(0, 0), track=45),    # good
    ]

    class FakeClient:
        def search(self, query, providers=None):
            return scenes

    selected, report = stack_selection.search_and_select_consistent_stack(
        FakeClient(), AOI, "2020-01-01", "2020-02-01", max_results=100,
    )
    dates = sorted(str(s.datetime)[:10] for s in selected)
    print(f"  selected dates: {dates}, dropped: {report['final_low_coverage_dates']}")
    assert "2020-01-13" not in dates, "the genuinely non-covering date should be dropped"
    assert len(dates) == 2
    print("  PASS\n")


def test_burst_family_classification_majority_exclusive():
    print("=== 5. select_burst_synchronized_dates: majority family used exclusively when large enough ===")

    # Monkeypatch the underlying functions this wraps, so this test is
    # isolated from real burst-sync physics (already covered by
    # test_burst_synchronization.py etc.) and just verifies the
    # classification/decision logic itself.
    calls = {}

    def fake_screen(dates, safe_zips, orbit_files, ground_point, swath_hints=None):
        calls["screened_dates"] = dates
        return "FAKE_SYNC_RESULTS"

    def fake_select_pairs(sync_results, dates, redundancy=3):
        assert sync_results == "FAKE_SYNC_RESULTS"
        # Simulate: 8 majority-family dates well-connected, 2 minority
        # dates only reachable via bridges.
        good_pairs = [("d1", "d2"), ("d2", "d3"), ("d3", "d4"), ("d1", "d5"),
                      ("d5", "d6"), ("d6", "d7"), ("d7", "d8"), ("d1", "d8")]
        bridge_pairs = [("d4", "d9"), ("d9", "d10")]
        return None, {"good_pairs": good_pairs, "bridge_pairs": bridge_pairs}

    fake_timeseries = SimpleNamespace(
        screen_stack_burst_synchronization=fake_screen,
        select_pairs_for_processing=fake_select_pairs,
    )
    _real_timeseries = sys.modules["pygeofetch.insar.timeseries"]
    sys.modules["pygeofetch.insar.timeseries"] = fake_timeseries

    all_dates = [f"d{i}" for i in range(1, 11)]
    safe_zips = {d: f"/fake/{d}.zip" for d in all_dates}
    orbit_files = {d: f"/fake/{d}.EOF" for d in all_dates}

    try:
        chosen, report = stack_selection.select_burst_synchronized_dates(
            all_dates, safe_zips, orbit_files, (0, 0, 0), min_majority_dates=8,
        )
    finally:
        sys.modules["pygeofetch.insar.timeseries"] = _real_timeseries
    print(f"  good_dates: {report['good_dates']}")
    print(f"  bridge_only_dates: {report['bridge_only_dates']}")
    print(f"  used_majority_only: {report['used_majority_only']}")
    print(f"  chosen: {chosen}")

    assert report["used_majority_only"] is True
    assert set(report["good_dates"]) == {"d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8"}
    assert set(report["bridge_only_dates"]) == {"d9", "d10"}
    assert set(chosen) == set(report["good_dates"]), "minority dates should be excluded entirely"
    print("  PASS\n")


def test_burst_family_classification_keeps_bridges_when_majority_too_small():
    print("=== 6. select_burst_synchronized_dates: keeps bridges when majority family is too small ===")

    def fake_screen(dates, safe_zips, orbit_files, ground_point, swath_hints=None):
        return "FAKE_SYNC_RESULTS"

    def fake_select_pairs(sync_results, dates, redundancy=3):
        good_pairs = [("d1", "d2"), ("d2", "d3")]
        bridge_pairs = [("d3", "d4")]
        return None, {"good_pairs": good_pairs, "bridge_pairs": bridge_pairs}

    fake_timeseries = SimpleNamespace(
        screen_stack_burst_synchronization=fake_screen,
        select_pairs_for_processing=fake_select_pairs,
    )
    _real_timeseries = sys.modules["pygeofetch.insar.timeseries"]
    sys.modules["pygeofetch.insar.timeseries"] = fake_timeseries

    all_dates = [f"d{i}" for i in range(1, 5)]
    safe_zips = {d: f"/fake/{d}.zip" for d in all_dates}
    orbit_files = {d: f"/fake/{d}.EOF" for d in all_dates}

    try:
        chosen, report = stack_selection.select_burst_synchronized_dates(
            all_dates, safe_zips, orbit_files, (0, 0, 0), min_majority_dates=8,
        )
    finally:
        sys.modules["pygeofetch.insar.timeseries"] = _real_timeseries
    print(f"  used_majority_only: {report['used_majority_only']}, chosen: {chosen}")
    assert report["used_majority_only"] is False
    assert set(chosen) == {"d1", "d2", "d3", "d4"}, "minority date(s) should be kept as bridges"
    print("  PASS\n")


if __name__ == "__main__":
    test_reproduces_mexico_city_duplicate_slice_fix()
    test_reproduces_obuasi_truncation_warning()
    test_zero_result_search_raises_clearly()
    test_final_coverage_verification_drops_genuine_gaps()
    test_burst_family_classification_majority_exclusive()
    test_burst_family_classification_keeps_bridges_when_majority_too_small()
    print("ALL TESTS PASSED")
