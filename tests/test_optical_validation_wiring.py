"""
Tests for optical preflight validation wired into
PyGeoFetch.search() and PyGeoFetch.download().

Confirms: the toggle works both as an instance-level default
(__init__(validate_optical=...)) and as a per-call override; base
instantiation never requires shapely; the AOI is derived automatically
from the query in search(); and download()'s existing length/order
guarantee (every input item gets exactly one corresponding
DownloadResult, in the same order) holds even when items are rejected
by preflight and never actually attempted.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pygeofetch import PyGeoFetch
from pygeofetch.models.download_task import DownloadResult, DownloadStatus
from pygeofetch.models.satellite_data import (
    ProcessingLevel,
    SatelliteAsset,
    SatelliteData,
)
from pygeofetch.models.search_query import SearchQuery


def make_scene(
    scene_id: str, bands: list[str] | None = None, cloud_cover: float = 5.0
) -> SatelliteData:
    bands = bands if bands is not None else ["B02", "B03", "B04", "B08", "SCL"]
    return SatelliteData(
        id=scene_id,
        provider="aws_earth",
        bbox=(-74.2, 40.5, -73.6, 41.0),
        cloud_cover=cloud_cover,
        processing_level=ProcessingLevel.L2A,
        assets={
            b: SatelliteAsset(
                key=b, href=f"https://example.com/{b}.tif", roles=["data"]
            )
            for b in bands
        },
    )


class TestBaseInstantiationUnaffected:
    def test_default_instantiation_does_not_require_shapely(self):
        """The whole point of the lazy-import design: importing and
        using PyGeoFetch normally must never require shapely unless
        optical validation is actually turned on."""
        with patch.dict("sys.modules", {"shapely": None, "shapely.geometry": None}):
            pf = PyGeoFetch()
            assert pf.validate_optical is False

    def test_validate_optical_defaults_to_false(self):
        pf = PyGeoFetch()
        assert pf.validate_optical is False

    def test_validate_optical_can_be_set_at_construction(self):
        pf = PyGeoFetch(validate_optical=True)
        assert pf.validate_optical is True

    def test_optical_validation_config_defaults_to_none(self):
        pf = PyGeoFetch()
        assert pf.optical_validation_config is None


class TestSearchWiring:
    def test_disabled_by_default_search_unaffected(self):
        """With validate_optical off (the default), search() must
        return exactly what the searcher gave it, untouched -- even a
        scene missing every band."""
        pf = PyGeoFetch()
        bad = make_scene("bad", bands=[])
        with patch.object(pf.searcher, "search", return_value=[bad]):
            results = pf.search(SearchQuery(bbox=(-74.1, 40.6, -73.7, 40.9)))
        assert results == [bad]

    def test_instance_level_toggle_filters_results(self):
        pf = PyGeoFetch(validate_optical=True)
        good = make_scene("good")
        bad = make_scene("bad", bands=["B02"])
        with patch.object(pf.searcher, "search", return_value=[good, bad]):
            results = pf.search(SearchQuery(bbox=(-74.1, 40.6, -73.7, 40.9)))
        assert [r.id for r in results] == ["good"]

    def test_per_call_override_true(self):
        """validate_optical=False at construction, but True for one call."""
        pf = PyGeoFetch(validate_optical=False)
        bad = make_scene("bad", bands=["B02"])
        with patch.object(pf.searcher, "search", return_value=[bad]):
            results = pf.search(
                SearchQuery(bbox=(-74.1, 40.6, -73.7, 40.9)), validate_optical=True
            )
        assert results == []

    def test_per_call_override_false(self):
        """validate_optical=True at construction, but False for one call."""
        pf = PyGeoFetch(validate_optical=True)
        bad = make_scene("bad", bands=["B02"])
        with patch.object(pf.searcher, "search", return_value=[bad]):
            results = pf.search(
                SearchQuery(bbox=(-74.1, 40.6, -73.7, 40.9)), validate_optical=False
            )
        assert results == [bad]

    def test_aoi_derived_automatically_from_query_bbox(self):
        """A scene far outside the query's own bbox should be
        rejected -- the AOI comes from the query, not a separate
        argument."""
        pf = PyGeoFetch(validate_optical=True)
        far_away = make_scene("far")
        far_away.bbox = (10.0, 10.0, 10.5, 10.5)
        with patch.object(pf.searcher, "search", return_value=[far_away]):
            results = pf.search(SearchQuery(bbox=(-74.1, 40.6, -73.7, 40.9)))
        assert results == []

    def test_no_bbox_or_geometry_skips_aoi_check_without_crashing(self):
        pf = PyGeoFetch(validate_optical=True)
        good = make_scene("good")
        with patch.object(pf.searcher, "search", return_value=[good]):
            # SearchQuery with no bbox/geometry at all
            results = pf.search(SearchQuery())
        assert [r.id for r in results] == ["good"]

    def test_custom_validation_config_is_used(self):
        from pygeofetch.validation import OpticalValidationConfig

        pf = PyGeoFetch(
            validate_optical=True,
            optical_validation_config=OpticalValidationConfig(
                check_required_bands=False
            ),
        )
        no_bands_scene = make_scene("no_bands", bands=[])
        with patch.object(pf.searcher, "search", return_value=[no_bands_scene]):
            results = pf.search(SearchQuery(bbox=(-74.1, 40.6, -73.7, 40.9)))
        assert [r.id for r in results] == ["no_bands"]

    def test_empty_search_results_short_circuit(self):
        pf = PyGeoFetch(validate_optical=True)
        with patch.object(pf.searcher, "search", return_value=[]):
            results = pf.search(SearchQuery(bbox=(-74.1, 40.6, -73.7, 40.9)))
        assert results == []


class TestDownloadWiring:
    def test_disabled_by_default_download_unaffected(self):
        pf = PyGeoFetch()
        bad = make_scene("bad", bands=[])
        fake_result = DownloadResult(
            status=DownloadStatus.COMPLETED, data_id="bad", provider="aws_earth"
        )
        with patch.object(
            pf.downloader, "download_many", return_value=[fake_result]
        ) as mock_dl:
            results = pf.download(bad, Path("/tmp/out"))
        mock_dl.assert_called_once()
        assert results == [fake_result]

    def test_rejected_items_never_reach_the_real_downloader(self):
        pf = PyGeoFetch(validate_optical=True)
        good = make_scene("good")
        bad = make_scene("bad", bands=["B02"])
        fake_result = DownloadResult(
            status=DownloadStatus.COMPLETED, data_id="good", provider="aws_earth"
        )
        with patch.object(
            pf.downloader, "download_many", return_value=[fake_result]
        ) as mock_dl:
            pf.download([good, bad], Path("/tmp/out"))
        called_ids = [item.id for item in mock_dl.call_args[0][0]]
        assert called_ids == ["good"]

    def test_length_and_order_contract_preserved(self):
        """The core guarantee: len(results) == len(data), in the same
        order, even when some items were rejected and never attempted."""
        pf = PyGeoFetch(validate_optical=True)
        good1 = make_scene("good1")
        bad = make_scene("bad", bands=["B02"])
        good2 = make_scene("good2")
        fake_real = [
            DownloadResult(
                status=DownloadStatus.COMPLETED, data_id="good1", provider="aws_earth"
            ),
            DownloadResult(
                status=DownloadStatus.COMPLETED, data_id="good2", provider="aws_earth"
            ),
        ]
        with patch.object(pf.downloader, "download_many", return_value=fake_real):
            results = pf.download([good1, bad, good2], Path("/tmp/out"))

        assert len(results) == 3
        assert [r.data_id for r in results] == ["good1", "bad", "good2"]
        assert results[0].status == DownloadStatus.COMPLETED
        assert results[1].status == DownloadStatus.FAILED
        assert results[2].status == DownloadStatus.COMPLETED

    def test_rejected_item_has_actionable_error_message(self):
        pf = PyGeoFetch(validate_optical=True)
        bad = make_scene("bad", bands=["B02"])
        with patch.object(pf.downloader, "download_many", return_value=[]):
            results = pf.download([bad], Path("/tmp/out"))
        assert "MISSING_BANDS" in results[0].error

    def test_all_items_rejected_downloader_never_called(self):
        pf = PyGeoFetch(validate_optical=True)
        bad1 = make_scene("bad1", bands=["B02"])
        bad2 = make_scene("bad2", bands=["B03"])
        with patch.object(pf.downloader, "download_many") as mock_dl:
            results = pf.download([bad1, bad2], Path("/tmp/out"))
        mock_dl.assert_not_called()
        assert len(results) == 2
        assert all(r.status == DownloadStatus.FAILED for r in results)

    def test_per_call_override_at_download_time(self):
        pf = PyGeoFetch(validate_optical=False)
        bad = make_scene("bad", bands=["B02"])
        with patch.object(pf.downloader, "download_many") as mock_dl:
            results = pf.download([bad], Path("/tmp/out"), validate_optical=True)
        mock_dl.assert_not_called()
        assert results[0].status == DownloadStatus.FAILED

    def test_aoi_optional_at_download_time(self):
        """No query is available at download time -- aoi is an
        optional explicit kwarg, and AOI-dependent checks are simply
        skipped without it (not a crash)."""
        pf = PyGeoFetch(validate_optical=True)
        good = make_scene("good")
        fake_result = DownloadResult(
            status=DownloadStatus.COMPLETED, data_id="good", provider="aws_earth"
        )
        with patch.object(pf.downloader, "download_many", return_value=[fake_result]):
            results = pf.download([good], Path("/tmp/out"))  # no aoi kwarg given
        assert results[0].status == DownloadStatus.COMPLETED
