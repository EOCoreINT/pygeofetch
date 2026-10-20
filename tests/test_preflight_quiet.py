"""Tests for the notebook-friendly PreflightGate (no network needed)."""

from __future__ import annotations

import io
import logging
import time
import zipfile
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from pygeofetch.core import logging as pgl
from pygeofetch.core import orbits
from pygeofetch.core.progress import ProgressLine
from pygeofetch.insar import preflight as pf
from pygeofetch.models import BoundingBox

AOI = BoundingBox(min_lon=18.85, max_lon=19.15, min_lat=50.15, max_lat=50.35)
FOOT = {
    "type": "Polygon",
    "coordinates": [
        [[18.0, 49.5], [20.0, 49.5], [20.0, 51.0], [18.0, 51.0], [18.0, 49.5]]
    ],
}


def make_scenes(n: int):
    t0 = datetime(2018, 1, 1, 16, 0, tzinfo=timezone.utc)
    out = []
    for i in range(n):
        dt = t0 + timedelta(days=6 * i)
        name = f"S1A_IW_SLC__1SDV_{dt:%Y%m%dT%H%M%S}_{dt:%Y%m%dT%H%M%S}_000000_000000_ABCD.SAFE"
        out.append(
            SimpleNamespace(
                id=f"id{i}",
                datetime=dt,
                geometry=FOOT,
                relative_orbit=102,
                properties={"name": name, "platform": "S1A", "polarisation": "VV&VH"},
            )
        )
    return out


SEARCH_REPORT = {
    "geometry_report": {"track": 102, "satellites": {"S1A"}},
    "hit_max_results": False,
    "raw_result_count": 40,
}


@pytest.fixture()
def fake_network(monkeypatch, tmp_path):
    """Replace every network call with a fast fake that logs like the real code."""
    calls = {"annotation": 0, "orbit": 0, "threads": set()}
    nodes_log = logging.getLogger("pygeofetch.providers.copernicus_nodes")
    orbit_log = logging.getLogger("pygeofetch.orbits")

    def fake_annotation(client, scene, out_dir, **kw):
        import threading

        calls["annotation"] += 1
        calls["threads"].add(threading.get_ident())
        time.sleep(0.01)
        for k in range(6):  # the real code logged one INFO line per member
            nodes_log.info(
                "Fetched real annotation member %s (%d bytes)", f"m{k}.xml", 1000
            )
        z = Path(out_dir) / f"{scene.id}.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("a.xml", "x")
        return z

    def fake_orbit(product_name, output_dir, orbit_type="precise", timeout=60):
        calls["orbit"] += 1
        orbit_log.info("Downloading orbit file: %s", product_name)
        orbit_log.info("Orbit file saved: %s", product_name)
        p = Path(output_dir) / f"{product_name[:30]}.EOF"
        p.write_text("orbit")
        return str(p)

    def fake_select(dates, safe_zips, orbit_files, ground_point, **kw):
        return list(dates), {
            "good_dates": list(dates),
            "bridge_only_dates": [],
            "used_majority_only": True,
            "sync_results": [],
            "majority_internal_bridges": [],
            "majority_self_connected": True,
        }

    monkeypatch.setattr(
        "pygeofetch.providers.copernicus_nodes.fetch_annotation_zip", fake_annotation
    )
    monkeypatch.setattr("pygeofetch.core.orbits.fetch_orbit_file", fake_orbit)
    monkeypatch.setattr(
        "pygeofetch.insar.stack_selection.select_burst_synchronized_dates", fake_select
    )
    monkeypatch.setattr(
        pf.PreflightGate,
        "_compute_spatial_baselines",
        lambda self, *a, **k: {
            "high_baseline_pairs": [],
            "dates_resolved": 5,
            "dates_total": 5,
        },
    )
    monkeypatch.setenv("PYGEOFETCH_LOG_DIR", str(tmp_path / "logs"))
    return calls


def _setup_console_logging():
    root = logging.getLogger("pygeofetch")
    root.handlers.clear()
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root.addHandler(h)
    root.setLevel(logging.INFO)
    return buf


def test_quiet_run_keeps_console_short_and_writes_full_log(fake_network, tmp_path):
    buf = _setup_console_logging()
    gate = pf.PreflightGate(
        None,
        AOI,
        "2018-01-01",
        "2019-01-01",
        max_workers=4,
        show_progress=False,
        log_file=tmp_path / "pf.log",
    )
    out = io.StringIO()
    with redirect_stdout(out):
        report = gate.run(make_scenes(40), dict(SEARCH_REPORT))
    console = buf.getvalue() + out.getvalue()
    assert report.go
    assert len(report.selected) == 40
    # 40 scenes x (6 annotation + 2 orbit) INFO lines would be 320 lines; none may reach the console
    assert "Fetched real annotation member" not in console
    assert "Orbit file saved" not in console
    assert len(console.splitlines()) < 25
    log_text = (tmp_path / "pf.log").read_text()
    assert (
        log_text.count("Fetched real annotation member") == 240
    )  # nothing lost in the file
    assert report.log_file == str(tmp_path / "pf.log")
    assert "Full details of this run" in out.getvalue()


def test_verbose_prints_everything(fake_network):
    buf = _setup_console_logging()
    gate = pf.PreflightGate(
        None,
        AOI,
        "2018-01-01",
        "2019-01-01",
        verbose=True,
        show_progress=False,
        print_summary=False,
        max_workers=2,
    )
    gate.run(make_scenes(5), dict(SEARCH_REPORT))
    assert buf.getvalue().count("Fetched real annotation member") == 30


def test_fetch_runs_in_parallel_and_keeps_original_order(fake_network):
    gate = pf.PreflightGate(
        None,
        AOI,
        "2018-01-01",
        "2019-01-01",
        max_workers=6,
        show_progress=False,
        print_summary=False,
    )
    scenes = make_scenes(30)
    report = gate.run(scenes, dict(SEARCH_REPORT))
    assert len(fake_network["threads"]) > 1
    assert fake_network["annotation"] == 30 and fake_network["orbit"] == 30
    assert [s.id for s in report.selected] == [s.id for s in scenes]


def test_sequential_mode_still_works(fake_network):
    gate = pf.PreflightGate(
        None,
        AOI,
        "2018-01-01",
        "2019-01-01",
        max_workers=1,
        show_progress=False,
        print_summary=False,
    )
    report = gate.run(make_scenes(8), dict(SEARCH_REPORT))
    assert report.go and len(fake_network["threads"]) == 1


def test_time_budget_stops_instead_of_hanging(monkeypatch, fake_network):
    def slow_annotation(client, scene, out_dir, **kw):
        time.sleep(5)
        raise RuntimeError("never finishes in time")

    monkeypatch.setattr(
        "pygeofetch.providers.copernicus_nodes.fetch_annotation_zip", slow_annotation
    )
    gate = pf.PreflightGate(
        None,
        AOI,
        "2018-01-01",
        "2019-01-01",
        max_workers=2,
        max_runtime_s=1.0,
        show_progress=False,
        print_summary=False,
    )
    t0 = time.monotonic()
    report = gate.run(make_scenes(10), dict(SEARCH_REPORT))
    assert time.monotonic() - t0 < 4.0
    codes = {i.code for i in report.issues}
    assert "BURST_FAMILY_RISK_UNASSESSED" in codes  # honest advisory, not a silent pass
    assert len(report.selected) == 10  # nothing dropped on a timeout


def test_truncation_rewiden_never_narrower_than_hit_cap(monkeypatch, fake_network):
    seen = []

    def fake_search(client, bbox, start, end, **kw):
        seen.append(kw["max_results"])
        cap = kw["max_results"]
        n = min(cap, 600)
        return make_scenes(5), {
            "geometry_report": {"track": 102, "satellites": {"S1A"}},
            "hit_max_results": n >= cap,
            "raw_result_count": n,
        }

    monkeypatch.setattr(
        "pygeofetch.insar.stack_selection.search_and_select_consistent_stack",
        fake_search,
    )
    gate = pf.PreflightGate(
        None,
        AOI,
        "2018-01-01",
        "2019-01-01",
        max_results=20,
        attempt_real_burst_check=False,
        show_progress=False,
        print_summary=False,
    )
    first_report = {
        "geometry_report": {"track": 102, "satellites": {"S1A"}},
        "hit_max_results": True,
        "raw_result_count": 250,
    }
    gate.run(make_scenes(5), first_report)
    assert seen and seen[0] >= 250 * 3  # old code started at 20*3 = 60


def test_tables_are_suppressed_in_quiet_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("PYGEOFETCH_LOG_DIR", str(tmp_path))
    out = io.StringIO()
    with redirect_stdout(out):
        with pgl.quiet_logs(announce=False):
            pgl.print_search_header(["copernicus"], AOI, "2018", "2019", 100, "SLC")
            pgl.print_search_results(make_scenes(3))
        pgl.print_search_header(["copernicus"], AOI, "2018", "2019", 100, "SLC")
    assert (
        out.getvalue().count("SEARCH PARAMETERS") == 1
    )  # only the one outside the block


def test_quiet_logs_restores_levels_even_on_error(tmp_path):
    buf = _setup_console_logging()
    root = logging.getLogger("pygeofetch")
    handler = root.handlers[0]
    before = (handler.level, root.level, len(root.handlers))
    with pytest.raises(RuntimeError):
        with pgl.quiet_logs(log_file=tmp_path / "x.log", announce=False):
            logging.getLogger("pygeofetch.x").info("hidden")
            raise RuntimeError("boom")
    assert (handler.level, root.level, len(root.handlers)) == before
    assert not handler.filters
    assert "hidden" not in buf.getvalue()
    assert "hidden" in (tmp_path / "x.log").read_text()


def test_repeat_limiter_collapses_identical_messages(tmp_path):
    buf = _setup_console_logging()
    out = io.StringIO()
    with redirect_stdout(out):
        with pgl.quiet_logs(
            console_level="INFO", log_file=tmp_path / "r.log", repeat_limit=3
        ):
            for i in range(100):
                logging.getLogger("pygeofetch.orbits").info("Orbit file saved: %s", i)
            logging.getLogger("pygeofetch.orbits").error("real failure %s", 1)
    assert buf.getvalue().count("Orbit file saved") == 3
    assert "real failure" in buf.getvalue()
    assert "97 repeated log line(s) hidden" in out.getvalue()
    assert (tmp_path / "r.log").read_text().count("Orbit file saved") == 100


def test_orbit_listing_is_fetched_once_per_month(monkeypatch):
    orbits.clear_orbit_listing_cache()
    calls = []

    class R:
        text = 'href="S1A_OPER_AUX_POEORB_OPOD_20200125T120000_V20200101T225942_20200103T005942.EOF.zip"'

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        orbits.requests, "get", lambda url, timeout=60: calls.append(url) or R()
    )
    for _ in range(20):
        assert orbits._get_listing("https://example/x/", 5) == R.text
    assert len(calls) == 1
    orbits.clear_orbit_listing_cache()


def test_progress_line_plain_mode_prints_few_lines():
    buf = io.StringIO()
    with ProgressLine(total=100, desc="x", stream=buf, min_interval=0) as bar:
        for _ in range(100):
            bar.update()
    assert len(buf.getvalue().splitlines()) <= 7
    assert "100/100" in buf.getvalue()


def test_long_lists_are_shortened_in_messages():
    assert pf._fmt_list(list(range(3))) == "[0, 1, 2]"
    s = pf._fmt_list([f"2018-01-{i:02d}" for i in range(1, 30)])
    assert "+23 more" in s and len(s) < 120
