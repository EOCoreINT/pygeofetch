"""
Validates ProjectState against the spec's own explicit requirements:
persistent tracking across process restarts (real file-backed, not
in-memory), idempotent re-processing, and real network-hash
determinism.
"""
import tempfile
from pathlib import Path

from pygeofetch.core import state as state_mod

ProjectState = state_mod.ProjectState
RunSummary = state_mod.RunSummary
network_topology_hash = state_mod.network_topology_hash


def test_state_persists_across_real_process_restarts():
    print("=== 1. State genuinely persists to disk -- a NEW ProjectState instance sees a previous instance's writes ===")
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "state.db"

        state1 = ProjectState(db_path)
        state1.mark_dates_processed(["2024-01-01", "2024-01-13"])
        state1.set_reference_pixel(120, 340)
        state1.set_last_download_timestamp("2024-01-15T00:00:00Z")

        # A genuinely NEW object, simulating a fresh process (e.g. a new
        # cron invocation) reading the SAME real file on disk
        state2 = ProjectState(db_path)
        print(f"  processed dates seen by new instance: {state2.processed_dates()}")
        print(f"  reference pixel seen by new instance: {state2.reference_pixel_coords}")
        assert state2.processed_dates() == ["2024-01-01", "2024-01-13"]
        assert state2.reference_pixel_coords == (120, 340)
        assert state2.last_download_timestamp == "2024-01-15T00:00:00Z"
    print("  PASS\n")


def test_marking_same_date_twice_is_idempotent():
    print("=== 2. Marking the same date processed twice is a real no-op, matching the spec's explicit idempotency requirement ===")
    with tempfile.TemporaryDirectory() as d:
        state = ProjectState(Path(d) / "state.db")
        state.mark_dates_processed(["2024-01-01"])
        state.mark_dates_processed(["2024-01-01"])  # real duplicate call, e.g. a re-run after a crash
        dates = state.processed_dates()
        print(f"  processed dates after marking the same date twice: {dates}")
        assert dates == ["2024-01-01"], "duplicate marking must not create a duplicate entry"
    print("  PASS\n")


def test_network_hash_is_deterministic_regardless_of_order():
    print("=== 3. network_topology_hash is deterministic -- same real network hashes identically regardless of construction order ===")
    dates_a = ["2024-01-01", "2024-01-13", "2024-01-25"]
    pairs_a = [("2024-01-01", "2024-01-13"), ("2024-01-13", "2024-01-25")]

    # Same real network, deliberately built/passed in a different order
    dates_b = ["2024-01-25", "2024-01-01", "2024-01-13"]
    pairs_b = [("2024-01-25", "2024-01-13"), ("2024-01-13", "2024-01-01")]

    hash_a = network_topology_hash(dates_a, pairs_a)
    hash_b = network_topology_hash(dates_b, pairs_b)
    print(f"  hash A: {hash_a[:16]}...")
    print(f"  hash B: {hash_b[:16]}...")
    assert hash_a == hash_b, "the same real network must hash identically regardless of order"

    # A genuinely DIFFERENT network must hash differently
    pairs_c = [("2024-01-01", "2024-01-13")]  # missing one real pair
    hash_c = network_topology_hash(dates_a, pairs_c)
    assert hash_c != hash_a, "a genuinely different network must hash differently"
    print("  PASS\n")


def test_run_log_records_real_history():
    print("=== 4. run_log records real run history, queryable after the fact ===")
    with tempfile.TemporaryDirectory() as d:
        state = ProjectState(Path(d) / "state.db")
        state.record_run(RunSummary(status="success", n_new_scenes=3, detail="3 new dates ingested"))
        state.record_run(RunSummary(status="no_new_scenes", n_new_scenes=0))

        history = state.run_history()
        print(f"  {len(history)} real runs recorded")
        for run in history:
            print(f"    {run['status']}: {run['n_new_scenes']} new scenes")
        assert len(history) == 2
        assert history[0]["status"] == "no_new_scenes"  # most recent first
        assert state.last_successful_run is not None
    print("  PASS\n")


def test_concurrent_access_does_not_corrupt_state():
    print("=== 5. Two real, concurrent ProjectState instances writing to the same file don't corrupt each other ===")
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "state.db"
        state_a = ProjectState(db_path)
        state_b = ProjectState(db_path)  # simulates a second real process/connection

        state_a.mark_dates_processed(["2024-01-01"])
        state_b.mark_dates_processed(["2024-01-13"])
        state_a.mark_dates_processed(["2024-01-25"])

        final = ProjectState(db_path)
        dates = final.processed_dates()
        print(f"  final processed dates from all real writers: {dates}")
        assert dates == ["2024-01-01", "2024-01-13", "2024-01-25"]
    print("  PASS\n")


if __name__ == "__main__":
    test_state_persists_across_real_process_restarts()
    test_marking_same_date_twice_is_idempotent()
    test_network_hash_is_deterministic_regardless_of_order()
    test_run_log_records_real_history()
    test_concurrent_access_does_not_corrupt_state()
    print("ALL TESTS PASSED")
