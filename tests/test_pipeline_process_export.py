"""
Regression tests for PipelineRunner._step_process / _step_export.

These two step handlers were previously stub implementations that
logged a message and returned {"status": "stub"} without doing any
real work. This file verifies the real, replacement implementations
actually process and move files -- not just that they run without
raising.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pygeofetch.core.downloader import AdaptiveDownloader
from pygeofetch.core.scheduler import PipelineRunner
from pygeofetch.models.download_task import DownloadResult, DownloadStatus


def _write_tif(path: Path, value: float = 1.0) -> Path:
    transform = from_origin(-74.0, 40.9, 0.001, 0.001)
    data = np.full((10, 10), value, dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def runner(tmp_path):
    engine = SimpleNamespace(downloader=AdaptiveDownloader(auth_manager=None))
    return PipelineRunner(engine)


@pytest.fixture
def download_context(tmp_path):
    tif = _write_tif(tmp_path / "scene1.tif")
    result = DownloadResult(
        status=DownloadStatus.COMPLETED,
        data_id="scene1",
        output_path=tif,
        output_paths=[tif],
    )
    return {"download": [result]}


class TestStepProcessIsReal:
    def test_compress_action_actually_runs(self, runner, download_context):
        """
        REAL BUG FIXED: _step_process used to just log and return
        {"status": "stub"} -- no compression, no reprojection, nothing.
        This confirms a real "compress" action now genuinely produces a
        new, real compressed GeoTIFF on disk.
        """
        original_paths = list(download_context["download"][0].output_paths)
        config = "compress:lzw"
        result = runner._step_process(config, download_context)

        assert isinstance(result, list)
        assert len(result) == 1
        processed = result[0]

        # A real new file must have been written, distinct from the
        # original -- this is the direct evidence the action really ran.
        # (Note: _run_post_process mutates the DownloadResult in place,
        # so we compare against a path captured *before* the call.)
        assert processed.output_paths != original_paths
        new_path = processed.output_paths[0]
        assert new_path.exists()
        assert "lzw" in new_path.name

        with rasterio.open(new_path) as src:
            assert src.profile["compress"].lower() == "lzw"

    def test_no_downloaded_items_returns_empty(self, runner):
        result = runner._step_process("compress:lzw", {})
        assert result == []

    def test_no_actions_configured_passes_through_unchanged(
        self, runner, download_context
    ):
        result = runner._step_process("", download_context)
        assert result == download_context["download"]

    def test_list_of_dict_actions_supported(self, runner, download_context):
        config = [{"action": "compress", "params": {"value": "deflate"}}]
        result = runner._step_process(config, download_context)
        new_path = result[0].output_paths[0]
        assert new_path.exists()
        assert "deflate" in new_path.name


class TestStepExportIsReal:
    def test_local_export_actually_copies_files(
        self, runner, download_context, tmp_path
    ):
        """
        REAL BUG FIXED: _step_export used to just log the destination
        and return {"status": "stub"} -- nothing was ever written
        anywhere. This confirms files genuinely land on disk at the
        real destination now.
        """
        dest = tmp_path / "exported"
        config = {"destination": str(dest), "format": "original"}

        result = runner._step_export(config, download_context)

        assert result["status"] == "completed"
        assert result["exported"] == 1
        exported_file = dest / "scene1.tif"
        assert exported_file.exists()
        # Real content, not an empty placeholder file.
        assert exported_file.stat().st_size > 0

    def test_uses_process_context_over_download_when_present(self, runner, tmp_path):
        raw_tif = _write_tif(tmp_path / "raw.tif")
        processed_tif = _write_tif(tmp_path / "processed.tif", value=2.0)
        context = {
            "download": [
                DownloadResult(
                    status=DownloadStatus.COMPLETED,
                    data_id="x",
                    output_path=raw_tif,
                    output_paths=[raw_tif],
                )
            ],
            "process": [
                DownloadResult(
                    status=DownloadStatus.COMPLETED,
                    data_id="x",
                    output_path=processed_tif,
                    output_paths=[processed_tif],
                )
            ],
        }
        dest = tmp_path / "out"
        result = runner._step_export({"destination": str(dest)}, context)

        assert (dest / "processed.tif").exists()
        assert not (dest / "raw.tif").exists()
        assert result["exported"] == 1

    def test_no_items_returns_empty_status(self, runner):
        result = runner._step_export({"destination": "./nowhere"}, {})
        assert result["status"] == "empty"
        assert result["exported"] == 0

    def test_s3_export_invalid_destination_fails_clearly(
        self, runner, download_context
    ):
        result = runner._step_export({"destination": "s3://"}, download_context)
        assert result["exported"] == 0
        assert result["status"] == "failed"

    def test_s3_export_calls_boto3_upload_file(self, runner, download_context):
        """
        Verifies the S3 code path genuinely calls boto3's upload_file
        with the right bucket/key -- without needing real AWS
        credentials or network access.
        """
        mock_s3 = SimpleNamespace(upload_file=lambda *a, **k: None)
        calls = []
        mock_s3.upload_file = lambda filename, bucket, key: calls.append(
            (filename, bucket, key)
        )

        with patch("boto3.client", return_value=mock_s3):
            result = runner._step_export(
                {"destination": "s3://my-bucket/results/"}, download_context
            )

        assert result["exported"] == 1
        assert len(calls) == 1
        filename, bucket, key = calls[0]
        assert bucket == "my-bucket"
        assert key == "results/scene1.tif"

    def test_webhook_notification_actually_posts(
        self, runner, download_context, tmp_path
    ):
        posted = []

        def fake_post(url, json=None, timeout=None):
            posted.append((url, json))
            return SimpleNamespace(status_code=200)

        dest = tmp_path / "out"
        with patch("httpx.post", side_effect=fake_post):
            runner._step_export(
                {
                    "destination": str(dest),
                    "notify": "webhook:https://example.com/hook",
                },
                download_context,
            )

        assert len(posted) == 1
        url, payload = posted[0]
        assert url == "https://example.com/hook"
        assert payload["exported"] == 1
        assert payload["total"] == 1


class TestFullPipelineNoLongerStubs:
    def test_process_then_export_end_to_end(self, runner, download_context, tmp_path):
        """A search->download->process->export pipeline segment, run
        for real end to end, with real files landing on disk -- this
        is the scenario that used to silently do nothing."""
        processed = runner._step_process("compress:lzw", download_context)
        context = {"download": download_context["download"], "process": processed}

        dest = tmp_path / "final"
        export_result = runner._step_export({"destination": str(dest)}, context)

        assert export_result["status"] == "completed"
        files = list(dest.glob("*.tif"))
        assert len(files) == 1
        assert "lzw" in files[0].name
