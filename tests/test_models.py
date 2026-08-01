"""Unit tests for PyGeoFetch data models."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pygeofetch.models.download_task import (
    DownloadOptions,
    DownloadResult,
    DownloadStatus,
    PostProcessAction,
)
from pygeofetch.models.satellite_data import (
    SatelliteData,
)
from pygeofetch.models.search_query import BoundingBox, SearchQuery
from pygeofetch.models.user_auth import AuthSession, Credentials


class TestBoundingBox:
    def test_from_string(self):
        bb = BoundingBox.from_string("-74.1,40.6,-73.7,40.9")
        assert bb.min_lon == pytest.approx(-74.1)
        assert bb.min_lat == pytest.approx(40.6)
        assert bb.max_lon == pytest.approx(-73.7)
        assert bb.max_lat == pytest.approx(40.9)

    def test_from_tuple(self):
        bb = BoundingBox.from_tuple((-74.1, 40.6, -73.7, 40.9))
        assert bb.to_tuple() == (-74.1, 40.6, -73.7, 40.9)

    def test_invalid_string_raises(self):
        with pytest.raises(Exception):
            BoundingBox.from_string("bad,data")

    def test_to_wkt(self):
        bb = BoundingBox(min_lon=-74.1, min_lat=40.6, max_lon=-73.7, max_lat=40.9)
        wkt = bb.to_wkt()
        assert "POLYGON" in wkt


class TestSearchQuery:
    def test_defaults(self, sample_bbox):
        q = SearchQuery(bbox=sample_bbox)
        assert q.cloud_cover_min == 0
        assert q.cloud_cover_max == 100
        assert q.max_results >= 10  # verify has a positive default

    def test_copy_for_provider(self, sample_query):
        q2 = sample_query.copy_for_provider("usgs")
        assert q2 is not sample_query

    def test_to_stac_filter_empty(self, sample_bbox):
        q = SearchQuery(bbox=sample_bbox)
        f = q.to_stac_filter()
        assert isinstance(f, dict)


class TestSatelliteData:
    def test_basic_construction(self, sample_satellite_data):
        item = sample_satellite_data
        assert item.id == "LC08_L2SP_013032_20240315"
        assert item.provider == "usgs"
        assert item.cloud_cover == pytest.approx(5.2)

    def test_to_stac_item(self, sample_satellite_data):
        feature = sample_satellite_data.to_stac_item()
        assert feature["type"] == "Feature"
        assert "properties" in feature
        assert "geometry" in feature
        assert feature["id"] == "LC08_L2SP_013032_20240315"

    def test_from_stac_item_roundtrip(self, sample_satellite_data):
        feature = sample_satellite_data.to_stac_item()
        restored = SatelliteData.from_stac_item(feature, "usgs")
        assert restored.id == sample_satellite_data.id
        assert restored.provider == "usgs"

    def test_score_default(self):
        item = SatelliteData(id="X", provider="test")
        assert item.score is not None  # has a default (0.0)


class TestDownloadOptions:
    def test_defaults(self):
        opts = DownloadOptions()
        assert opts.parallel >= 1  # just verify it has a value
        assert opts.retry_attempts >= 0

    def test_post_process_actions(self):
        opts = DownloadOptions(
            post_process=[
                PostProcessAction(action="unzip"),
                PostProcessAction(action="reproject", params={"value": "EPSG:4326"}),
            ]
        )
        assert len(opts.post_process) == 2
        assert opts.post_process[0].action == "unzip"


class TestDownloadResult:
    def test_success_property(self):
        result = DownloadResult(
            status=DownloadStatus.COMPLETED, data_id="X", provider="test",
        )
        assert result.success is True

    def test_failed_property(self):
        result = DownloadResult(
            status=DownloadStatus.FAILED, data_id="X", provider="test", error="Network error",
        )
        assert result.success is False

    def test_bytes_downloaded(self):
        result = DownloadResult(
            status=DownloadStatus.COMPLETED, data_id="X", provider="test",
            bytes_downloaded=10 * 1024 * 1024,
        )
        assert result.bytes_downloaded == 10 * 1024 * 1024


class TestCredentials:
    def test_construction(self):
        creds = Credentials(provider="usgs", username="user", password="secret")
        assert creds.provider == "usgs"
        assert creds.username == "user"

    def test_api_key_credential(self):
        creds = Credentials(provider="planet", api_key="PL_KEY_ABCDEF")
        assert creds.api_key is not None


class TestAuthSession:
    def test_is_valid(self, sample_auth_session):
        # access_token is set and not expired
        assert sample_auth_session.is_valid is True

    def test_is_expired(self):
        expired = AuthSession(
            provider="test",
            access_token="tok",
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=10),
        )
        assert expired.is_expired is True
