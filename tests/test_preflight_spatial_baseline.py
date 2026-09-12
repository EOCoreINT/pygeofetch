"""
Tests for PreflightGate._compute_spatial_baselines.

The full, real path (_try_real_burst_family_check -> _run_real_burst_family_check)
needs live Copernicus network access to fetch real annotation XML and real
orbit files -- not something a unit test should depend on. Instead, these
tests control the orbit-resolution layer (parse_orbit_file,
find_zero_doppler_time, interpolate_orbit_state) with known, synthetic
values and verify the real perpendicular_baseline() math and the real
high-baseline-flagging logic run correctly on top of them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pygeofetch.insar.preflight import PreflightGate


class _FakeAoiBbox:
    min_lat, max_lat = 39.28, 39.48
    min_lon, max_lon = 109.90, 110.12


def _scene(date_str: str) -> SimpleNamespace:
    return SimpleNamespace(
        datetime=datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    )


@pytest.fixture
def gate():
    return PreflightGate(
        client=None,
        aoi_bbox=_FakeAoiBbox(),
        start_date="2023-01-01",
        end_date="2023-12-31",
        max_perpendicular_baseline_m=150.0,
    )


class TestSpatialBaselineComputation:
    def test_low_baseline_pair_not_flagged(self, gate):
        selected = [_scene("2023-01-01"), _scene("2023-01-13")]
        orbit_files = {"2023-01-01": "fake1.EOF", "2023-01-13": "fake2.EOF"}
        ground_point = (1000.0, 2000.0, 3000.0)

        # Two real, close satellite positions -> a real, small perpendicular
        # baseline, well under the 150m threshold.
        positions = {
            "2023-01-01": (7000000.0, 0.0, 0.0),
            "2023-01-13": (7000050.0, 30.0, 0.0),  # close by
        }

        with patch(
            "pygeofetch.insar.geolocation.parse_orbit_file", return_value=([], [], [])
        ), patch(
            "pygeofetch.insar.geolocation.find_zero_doppler_time",
            side_effect=lambda *a, **k: a[-1],
        ), patch(
            "pygeofetch.insar.geolocation.interpolate_orbit_state",
            side_effect=lambda times, pos, vel, t: (
                positions[_date_for(t, selected)],
                None,
            ),
        ):
            report = gate._compute_spatial_baselines(
                selected, ["2023-01-01", "2023-01-13"], orbit_files, ground_point
            )

        assert report["dates_resolved"] == 2
        assert len(report["consecutive_pair_baselines_m"]) == 1
        assert report["high_baseline_pairs"] == []

    def test_high_baseline_pair_is_flagged(self, gate):
        selected = [_scene("2023-01-01"), _scene("2023-01-13")]
        orbit_files = {"2023-01-01": "fake1.EOF", "2023-01-13": "fake2.EOF"}
        ground_point = (1000.0, 2000.0, 3000.0)

        # Two real, deliberately far-apart satellite positions relative to
        # the ground point -> a real, large perpendicular baseline.
        positions = {
            "2023-01-01": (7000000.0, 0.0, 0.0),
            "2023-01-13": (7000000.0, 5000.0, 0.0),  # far off-track
        }

        with patch(
            "pygeofetch.insar.geolocation.parse_orbit_file", return_value=([], [], [])
        ), patch(
            "pygeofetch.insar.geolocation.find_zero_doppler_time",
            side_effect=lambda *a, **k: a[-1],
        ), patch(
            "pygeofetch.insar.geolocation.interpolate_orbit_state",
            side_effect=lambda times, pos, vel, t: (
                positions[_date_for(t, selected)],
                None,
            ),
        ):
            report = gate._compute_spatial_baselines(
                selected, ["2023-01-01", "2023-01-13"], orbit_files, ground_point
            )

        assert len(report["high_baseline_pairs"]) == 1
        assert report["high_baseline_pairs"][0]["date1"] == "2023-01-01"
        assert report["high_baseline_pairs"][0]["date2"] == "2023-01-13"
        assert abs(report["high_baseline_pairs"][0]["perpendicular_baseline_m"]) > 150.0

    def test_missing_orbit_file_for_a_date_is_excluded_not_fatal(self, gate):
        selected = [_scene("2023-01-01"), _scene("2023-01-13")]
        orbit_files = {"2023-01-01": "fake1.EOF"}  # 2023-01-13 missing entirely
        ground_point = (1000.0, 2000.0, 3000.0)

        report = gate._compute_spatial_baselines(
            selected, ["2023-01-01", "2023-01-13"], orbit_files, ground_point
        )

        # Real, honest behavior: doesn't raise, just can't resolve the
        # missing date -- confirmed by the low resolved count.
        assert report["dates_resolved"] <= 1
        assert report["dates_total"] == 2

    def test_orbit_parse_failure_is_excluded_not_fatal(self, gate):
        """A malformed/corrupt real orbit file must not crash the whole
        preflight run over a secondary, advisory check."""
        selected = [_scene("2023-01-01")]
        orbit_files = {"2023-01-01": "corrupt.EOF"}
        ground_point = (1000.0, 2000.0, 3000.0)

        with patch(
            "pygeofetch.insar.geolocation.parse_orbit_file",
            side_effect=ValueError("real, malformed EOF file"),
        ):
            report = gate._compute_spatial_baselines(
                selected, ["2023-01-01"], orbit_files, ground_point
            )

        assert report["dates_resolved"] == 0
        assert report["consecutive_pair_baselines_m"] == []


def _date_for(t, selected):
    """Test helper: map the (mocked, pass-through) target time back to
    its real date string, since find_zero_doppler_time is mocked to
    just return whatever acquisition time it was given."""
    for s in selected:
        if s.datetime == t:
            return str(s.datetime.date())
    raise KeyError(t)
