"""Download-budget thinning must shrink the stack AND keep the network connected."""

from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from pygeofetch.insar.network_thinning import network_stats, thin_dates
from pygeofetch.insar.preflight import PreflightGate


def _dates(start="2017-10-25", n=225, step=6, gaps=()):
    d0 = date.fromisoformat(start)
    out, cur = [], d0
    for i in range(n):
        out.append(cur.isoformat())
        cur += timedelta(days=step + (12 if i in gaps else 0))
    return out


@pytest.mark.parametrize("every", [12, 24, 36])
def test_thin_reduces_and_stays_connected(every):
    ds = _dates()
    r = thin_dates(ds, max_temporal_baseline_days=48, thin_to_days=every)
    assert len(r.kept) < len(ds)
    assert r.stats_after.n_components == 1
    assert r.gb_after < r.gb_before
    assert set(r.kept) <= set(ds)
    assert r.kept == sorted(r.kept)
    assert r.kept[0] == ds[0] and r.kept[-1] == ds[-1]


def test_cadence_is_even():
    r = thin_dates(_dates(), max_temporal_baseline_days=48, thin_to_days=12)
    gaps = [
        (date.fromisoformat(b) - date.fromisoformat(a)).days
        for a, b in zip(r.kept, r.kept[1:])
    ]
    assert 6 <= min(gaps) and max(gaps) <= 18


def test_budget_is_met_when_possible():
    ds = _dates()
    r = thin_dates(ds, max_temporal_baseline_days=48, max_download_gb=300)
    assert r.gb_after <= 300 and r.budget_met
    assert r.stats_after.n_components == 1


def test_network_beats_budget_when_impossible():
    ds = _dates()  # ~1344 days; baseline 24 d -> needs >= 57 scenes
    r = thin_dates(ds, max_temporal_baseline_days=24, max_download_gb=100)
    assert not r.budget_met
    assert r.stats_after.n_components == 1
    assert r.gb_after > 100
    assert any("Budget" in n for n in r.notes)


def test_too_coarse_cadence_is_lowered_for_redundancy():
    r = thin_dates(_dates(), max_temporal_baseline_days=48, thin_to_days=60)
    assert r.effective_days <= 24
    assert r.stats_after.min_redundancy >= 2
    assert any("lowered" in n for n in r.notes)


def test_archive_gap_is_not_made_worse():
    ds = _dates(n=80, gaps=(40,))  # 18-day hole, still bridgeable at 48 d
    r = thin_dates(ds, max_temporal_baseline_days=48, thin_to_days=24)
    assert r.stats_after.n_components == r.stats_before.n_components == 1
    ds2 = [
        d for d in ds if not ("2018-" in d and d > "2018-03-01" and d < "2018-09-01")
    ]
    full = network_stats(ds2, 48)
    r2 = thin_dates(ds2, max_temporal_baseline_days=48, thin_to_days=24)
    assert r2.stats_after.n_components <= full.n_components


def test_off_by_default():
    ds = _dates(n=30)
    r = thin_dates(ds, max_temporal_baseline_days=48)
    assert r.kept == ds and not r.changed


def _scene(d):
    return SimpleNamespace(
        datetime=d + "T05:00:00",
        properties={"name": d},
        geometry={
            "type": "Polygon",
            "coordinates": [[[-1, -1], [2, -1], [2, 2], [-1, 2], [-1, -1]]],
        },
    )


def test_gate_budget_step_keeps_scene_objects_and_reports():
    ds = _dates()
    scenes = [_scene(d) for d in ds]
    gate = PreflightGate(
        None,
        None,
        "2017-10-25",
        "2021-06-01",
        max_temporal_baseline_days=48,
        thin_to_days=12,
        max_download_gb=400,
    )
    sel, issues, res = gate._apply_download_budget(scenes)
    assert len(sel) * 7.5 <= 400
    assert {s.datetime[:10] for s in sel} == set(res.kept)
    assert issues[0].code == "DOWNLOAD_BUDGET_APPLIED" and issues[0].auto_fixed
    assert gate._recheck_network_after_screening(sel, res) == []
    # dropping the middle date after screening is caught
    broken = [s for s in sel if not ("2019-03" <= s.datetime[:7] <= "2019-09")]
    out = gate._recheck_network_after_screening(broken, res)
    assert out and out[0].code == "NETWORK_DISCONNECTED_AFTER_SCREEN"


def test_gate_without_budget_is_unchanged():
    gate = PreflightGate(None, None, "2017-10-25", "2021-06-01")
    scenes = [_scene(d) for d in _dates(n=20)]
    sel, issues, res = gate._apply_download_budget(scenes)
    assert sel == scenes and issues == [] and res is None


def test_full_run_with_budget_end_to_end():
    bbox = SimpleNamespace(min_lon=0, min_lat=0, max_lon=1, max_lat=1)
    scenes = [_scene(d) for d in _dates()]
    gate = PreflightGate(
        None,
        bbox,
        "2017-10-25",
        "2021-06-01",
        attempt_real_burst_check=False,
        max_temporal_baseline_days=48,
        thin_to_days=12,
        max_download_gb=300,
        print_summary=False,
    )
    rep = gate.run(scenes, {"raw_result_count": 225, "geometry_report": {}})
    assert rep.go
    assert len(rep.selected) * 7.5 <= 300
    th = rep.manifest["thinning"]
    assert th["budget_met"] and th["max_download_gb"] == 300
    assert th["gb_after"] == len(rep.selected) * 7.5
    assert "DOWNLOAD_BUDGET_APPLIED" in rep.summary()
