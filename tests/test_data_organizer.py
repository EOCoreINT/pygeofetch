"""Tests for pygeofetch.core.data_organizer, against a real temp filesystem."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pygeofetch.core.data_organizer import DataOrganizer, OrganizerConfig
from pygeofetch.models.download_task import DownloadResult, DownloadStatus
from pygeofetch.models.satellite_data import ProcessingLevel, SatelliteData


def _make_scene_and_download(
    tmp_path: Path, **overrides
) -> tuple[SatelliteData, DownloadResult]:
    defaults = dict(
        id="S2A_TEST_001",
        provider="aws_earth",
        satellite="Sentinel-2A",
        sensor="MSI",
        datetime=datetime(2024, 6, 15, 10, 30),
        bbox=(-74.1, 40.6, -73.7, 40.9),
        cloud_cover=5.0,
        processing_level=ProcessingLevel.L2A,
        relative_orbit=54,
    )
    defaults.update(overrides)
    scene = SatelliteData(**defaults)

    real_file = tmp_path / "raw" / f"{scene.id}.tif"
    real_file.parent.mkdir(parents=True, exist_ok=True)
    real_file.write_bytes(b"fake-geotiff-bytes")

    dl = DownloadResult(
        status=DownloadStatus.COMPLETED, data_id=scene.id, output_path=real_file
    )
    return scene, dl


class TestGrouping:
    def test_groups_by_date_default(self, tmp_path):
        scene, dl = _make_scene_and_download(tmp_path)
        organizer = DataOrganizer()  # default group_by=["date"]
        report = organizer.organize([(scene, dl)], tmp_path / "organized")

        expected = tmp_path / "organized" / "2024-06-15" / "S2A_TEST_001.tif"
        assert expected.exists()
        assert report.grouped == 1
        assert "2024-06-15" in report.groups

    def test_groups_by_satellite_then_date(self, tmp_path):
        scene, dl = _make_scene_and_download(tmp_path)
        organizer = DataOrganizer(OrganizerConfig(group_by=["satellite", "date"]))
        organizer.organize([(scene, dl)], tmp_path / "organized")

        expected = (
            tmp_path / "organized" / "Sentinel-2A" / "2024-06-15" / "S2A_TEST_001.tif"
        )
        assert expected.exists()

    def test_groups_by_orbit(self, tmp_path):
        scene, dl = _make_scene_and_download(tmp_path, relative_orbit=54)
        organizer = DataOrganizer(OrganizerConfig(group_by=["orbit"]))
        organizer.organize([(scene, dl)], tmp_path / "organized")

        expected = tmp_path / "organized" / "R054" / "S2A_TEST_001.tif"
        assert expected.exists()

    def test_missing_orbit_falls_back_honestly_not_silently(self, tmp_path):
        scene, dl = _make_scene_and_download(tmp_path, relative_orbit=None)
        organizer = DataOrganizer(OrganizerConfig(group_by=["orbit"]))
        organizer.organize([(scene, dl)], tmp_path / "organized")

        expected = tmp_path / "organized" / "unknown-orbit" / "S2A_TEST_001.tif"
        assert expected.exists()

    def test_failed_download_is_skipped_and_counted_not_silently_dropped(
        self, tmp_path
    ):
        scene, dl = _make_scene_and_download(tmp_path)
        dl.status = DownloadStatus.FAILED
        organizer = DataOrganizer()
        report = organizer.organize([(scene, dl)], tmp_path / "organized")

        assert report.grouped == 0
        assert report.skipped_failed_download == 1

    def test_copy_is_the_real_default_original_file_untouched(self, tmp_path):
        scene, dl = _make_scene_and_download(tmp_path)
        original = Path(dl.output_path)
        organizer = DataOrganizer()
        organizer.organize([(scene, dl)], tmp_path / "organized")

        assert original.exists()  # copy, not move -- confirms the real default

    def test_rerun_is_idempotent_does_not_error(self, tmp_path):
        scene, dl = _make_scene_and_download(tmp_path)
        organizer = DataOrganizer()
        organizer.organize([(scene, dl)], tmp_path / "organized")
        report2 = organizer.organize(
            [(scene, dl)], tmp_path / "organized"
        )  # real re-run
        assert report2.grouped == 1  # doesn't crash, doesn't double-count oddly


class TestManifest:
    def test_manifest_written_with_real_metadata(self, tmp_path):
        scene, dl = _make_scene_and_download(tmp_path)
        organizer = DataOrganizer()
        report = organizer.organize([(scene, dl)], tmp_path / "organized")

        assert report.manifest_path.exists()
        manifest = json.loads(report.manifest_path.read_text())
        assert manifest["scene_count"] == 1
        entry = manifest["scenes"][0]
        assert entry["id"] == "S2A_TEST_001"
        assert entry["satellite"] == "Sentinel-2A"
        assert entry["cloud_cover"] == 5.0
        assert entry["relative_orbit"] == 54
        assert entry["bbox"] == [-74.1, 40.6, -73.7, 40.9]
        assert entry["processing_level"] == "L2A"
        assert len(entry["paths"]) == 1

    def test_manifest_records_grouping_config(self, tmp_path):
        scene, dl = _make_scene_and_download(tmp_path)
        organizer = DataOrganizer(OrganizerConfig(group_by=["satellite", "date"]))
        report = organizer.organize([(scene, dl)], tmp_path / "organized")

        manifest = json.loads(report.manifest_path.read_text())
        assert manifest["group_by"] == ["satellite", "date"]


class TestInsarStackPreparation:
    def test_groups_slc_scenes_by_real_track(self, tmp_path):
        scenes = [
            _make_scene_and_download(
                tmp_path,
                id=f"S1A_{i}",
                relative_orbit=147,
                datetime=datetime(2023, 1, i + 1),
            )
            for i in range(3)
        ]
        organizer = DataOrganizer()
        report = organizer.prepare_insar_stack(scenes)

        assert 147 in report.tracks
        assert len(report.tracks[147]) == 3
        assert report.burst_sync_checked is False  # honest: no orbit_files given

    def test_orphaned_scenes_without_relative_orbit_are_reported(self, tmp_path):
        scenes = [
            _make_scene_and_download(tmp_path, id="S1A_orphan", relative_orbit=None)
        ]
        organizer = DataOrganizer()
        report = organizer.prepare_insar_stack(scenes)

        assert "S1A_orphan" in report.orphaned

    def test_without_orbit_files_burst_sync_is_honestly_unchecked(self, tmp_path):
        scenes = [_make_scene_and_download(tmp_path, id="S1A_x", relative_orbit=147)]
        organizer = DataOrganizer()
        report = organizer.prepare_insar_stack(scenes, orbit_files=None)

        assert report.burst_sync_checked is False
        assert report.burst_family_report is None
        # Real, honest behavior: still returns usable track grouping even
        # though the burst-sync check itself couldn't run.
        assert 147 in report.tracks
