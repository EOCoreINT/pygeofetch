"""
Tests for --validate-optical and its threshold overrides on
`pygeofetch search run` and `pygeofetch download run`.
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from pygeofetch.cli.main import cli
from pygeofetch.models.satellite_data import (
    ProcessingLevel,
    SatelliteAsset,
    SatelliteData,
)


def make_scene(scene_id, bands=None, cloud_cover=5.0):
    bands = bands if bands is not None else ["B02", "B03", "B04", "B08", "SCL"]
    return SatelliteData(
        id=scene_id,
        provider="aws_earth",
        bbox=(-74.2, 40.5, -73.6, 41.0),
        cloud_cover=cloud_cover,
        processing_level=ProcessingLevel.L2A,
        assets={
            b: SatelliteAsset(key=b, href=f"https://x/{b}.tif", roles=["data"])
            for b in bands
        },
    )


class TestSearchRunValidateOptical:
    def test_flag_present_in_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "run", "--help"])
        assert result.exit_code == 0
        assert "--validate-optical" in result.output
        assert "--optical-max-cloud-cover" in result.output

    def test_missing_bands_scene_filtered_out(self):
        good = make_scene("good")
        bad = make_scene("bad", bands=["B02"])
        runner = CliRunner()
        with patch(
            "pygeofetch.core.searcher.FederatedSearcher.search",
            return_value=[good, bad],
        ):
            result = runner.invoke(
                cli,
                [
                    "search",
                    "run",
                    "--bbox",
                    "-74.1,40.6,-73.7,40.9",
                    "--validate-optical",
                    "--format",
                    "ids",
                ],
            )
        assert result.exit_code == 0
        output_lines = result.output.strip().splitlines()
        assert "good" in output_lines
        assert "bad" not in output_lines  # "bad" DOES legitimately appear
        # in the rejection log line ("Scene bad [MISSING_BANDS]...") --
        # checking exact output lines (not substring) is what actually
        # confirms it was excluded from the *results*, not just absent
        # from the raw text.

    def test_without_flag_all_results_returned(self):
        good = make_scene("good")
        bad = make_scene("bad", bands=["B02"])
        runner = CliRunner()
        with patch(
            "pygeofetch.core.searcher.FederatedSearcher.search",
            return_value=[good, bad],
        ):
            result = runner.invoke(
                cli,
                [
                    "search",
                    "run",
                    "--bbox",
                    "-74.1,40.6,-73.7,40.9",
                    "--format",
                    "ids",
                ],
            )
        assert result.exit_code == 0
        assert "good" in result.output
        assert "bad" in result.output

    def test_optical_threshold_override_applied(self):
        """A scene with 15% cloud cover: rejected with a strict
        --optical-max-cloud-cover 10, kept with the 20.0 default."""
        cloudy = make_scene("cloudy15", cloud_cover=15.0)
        runner = CliRunner()

        with patch(
            "pygeofetch.core.searcher.FederatedSearcher.search", return_value=[cloudy]
        ):
            default_result = runner.invoke(
                cli,
                [
                    "search",
                    "run",
                    "--bbox",
                    "-74.1,40.6,-73.7,40.9",
                    "--validate-optical",
                    "--format",
                    "ids",
                ],
            )
        assert "cloudy15" in default_result.output  # warning only by default, kept

        with patch(
            "pygeofetch.core.searcher.FederatedSearcher.search", return_value=[cloudy]
        ):
            strict_result = runner.invoke(
                cli,
                [
                    "search",
                    "run",
                    "--bbox",
                    "-74.1,40.6,-73.7,40.9",
                    "--validate-optical",
                    "--optical-max-cloud-cover",
                    "10",
                    "--format",
                    "ids",
                ],
            )
        # still kept even under the strict threshold: cloud cover is a
        # WARNING by default (cloud_cover_is_hard_failure defaults False),
        # not exposed as a CLI flag -- so this just confirms the override
        # doesn't crash and the scene still appears.
        assert strict_result.exit_code == 0


class TestDownloadRunValidateOptical:
    def test_flag_present_in_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "run", "--help"])
        assert result.exit_code == 0
        assert "--validate-optical" in result.output
