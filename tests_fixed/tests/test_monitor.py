"""
Validates generate_incremental_pairs and plan_monitoring_run against
the spec's own explicit requirements: connecting new dates to their
N nearest neighbors (not the full candidate pool), and real
idempotency -- running twice with no new scenes does nothing the
second time.
"""
from pygeofetch.core import monitor as monitor_mod
from pygeofetch.core import state as state_mod

generate_incremental_pairs = monitor_mod.generate_incremental_pairs
plan_monitoring_run = monitor_mod.plan_monitoring_run
ProjectState = state_mod.ProjectState

import tempfile
from pathlib import Path


def test_new_date_connects_to_n_nearest_existing_neighbors():
    print("=== 1. A single new date connects to its N nearest chronological neighbors, not the full archive ===")
    existing = ["2024-01-01", "2024-01-13", "2024-01-25", "2024-02-06", "2024-02-18"]
    new = ["2024-03-01"]  # chronologically after all existing dates

    pairs = generate_incremental_pairs(new, existing, n_neighbors=3)
    print(f"  new pairs: {pairs}")

    # The 3 nearest to 2024-03-01 should be the 3 MOST RECENT existing dates
    expected_neighbors = {"2024-02-18", "2024-02-06", "2024-01-25"}
    actual_neighbors = {p[0] if p[1] == "2024-03-01" else p[1] for p in pairs}
    assert actual_neighbors == expected_neighbors, f"expected {expected_neighbors}, got {actual_neighbors}"
    assert len(pairs) == 3, "should form exactly 3 new pairs (n_neighbors=3), not the full archive"
    print("  PASS -- confirms this does NOT reprocess the whole archive, matching the spec's real intent\n")


def test_batch_of_new_dates_connects_to_each_other_too():
    print("=== 2. A BATCH of new dates arriving together correctly connects to each other, not just the old archive ===")
    existing = ["2024-01-01", "2024-01-13"]
    new = ["2024-02-01", "2024-02-13"]  # two new dates, close to each other, far from existing

    pairs = generate_incremental_pairs(new, existing, n_neighbors=2)
    print(f"  new pairs: {pairs}")

    # The two new dates are each other's nearest neighbor -- must be paired
    assert ("2024-02-01", "2024-02-13") in pairs, \
        "two new dates close together must connect to each other, not just to the distant old archive"
    print("  PASS -- confirms a real batch of new scenes forms a real, connected sub-network, not isolated spokes\n")


def test_no_new_dates_plans_nothing():
    print("=== 3. plan_monitoring_run correctly plans NOTHING when there are genuinely no new dates ===")
    with tempfile.TemporaryDirectory() as d:
        state = ProjectState(Path(d) / "state.db")
        state.mark_dates_processed(["2024-01-01", "2024-01-13"])

        result = plan_monitoring_run(state, available_dates=["2024-01-01", "2024-01-13"])
        print(f"  {result.message}")
        assert result.network_changed is False
        assert result.new_dates == []
        assert result.new_pairs == []
    print("  PASS\n")


def test_running_twice_with_no_new_scenes_is_idempotent():
    print("=== 4. Real idempotency, matching the spec's explicit requirement: running twice on the same day does the same nothing both times ===")
    with tempfile.TemporaryDirectory() as d:
        state = ProjectState(Path(d) / "state.db")
        state.mark_dates_processed(["2024-01-01", "2024-01-13", "2024-01-25"])

        # Real search returns the same dates both times (nothing new
        # published since yesterday) -- simulates two real cron runs
        # on the same day.
        result_1 = plan_monitoring_run(state, available_dates=["2024-01-01", "2024-01-13", "2024-01-25"])
        result_2 = plan_monitoring_run(state, available_dates=["2024-01-01", "2024-01-13", "2024-01-25"])

        assert result_1.network_changed is False
        assert result_2.network_changed is False
        assert result_1.new_dates == result_2.new_dates == []
    print("  PASS -- two real, identical runs plan the exact same (real) nothing\n")


def test_plan_correctly_detects_real_new_dates():
    print("=== 5. plan_monitoring_run correctly identifies genuinely new dates against real persisted state ===")
    with tempfile.TemporaryDirectory() as d:
        state = ProjectState(Path(d) / "state.db")
        state.mark_dates_processed(["2024-01-01", "2024-01-13"])

        # A real search now also returns one real new scene
        result = plan_monitoring_run(state, available_dates=["2024-01-01", "2024-01-13", "2024-01-25"])
        print(f"  {result.message}")
        assert result.new_dates == ["2024-01-25"]
        assert result.network_changed is True
        assert len(result.new_pairs) > 0
    print("  PASS\n")


if __name__ == "__main__":
    test_new_date_connects_to_n_nearest_existing_neighbors()
    test_batch_of_new_dates_connects_to_each_other_too()
    test_no_new_dates_plans_nothing()
    test_running_twice_with_no_new_scenes_is_idempotent()
    test_plan_correctly_detects_real_new_dates()
    print("ALL TESTS PASSED")
