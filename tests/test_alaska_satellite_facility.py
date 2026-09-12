"""
Tests for the rewritten AlaskaSatelliteFacilityProvider, against ASF's
real, documented Search API shape (services/search/param, WKT
intersectsWith geometry, geojson output) rather than the fictional
REST search the previous version assumed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pygeofetch.models.search_query import SearchQuery
from pygeofetch.models.user_auth import Credentials
from pygeofetch.providers.alaska_satellite_facility import (
    AlaskaSatelliteFacilityProvider,
    _bbox_to_wkt_polygon,
)
from pygeofetch.providers.base import AuthenticationError


def _real_geojson_response(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}


def _real_asf_feature(scene_name: str, platform: str = "Sentinel-1A") -> dict:
    """Shaped like a real ASF Search API geojson feature."""
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[-107, 35], [-106, 35], [-106, 36], [-107, 36], [-107, 35]]
            ],
        },
        "properties": {
            "sceneName": scene_name,
            "platform": platform,
            "startTime": "2024-06-15T10:00:00.000000Z",
            "processingLevel": "SLC",
            "url": f"https://datapool.asf.alaska.edu/SLC/SA/{scene_name}.zip",
        },
    }


class TestWktConversion:
    def test_bbox_to_wkt_matches_real_documented_format(self):
        wkt = _bbox_to_wkt_polygon(-106.7975, 34.9141, -106.3267, 35.3502)
        assert wkt == (
            "POLYGON((-106.7975 34.9141,-106.3267 34.9141,"
            "-106.3267 35.3502,-106.7975 35.3502,-106.7975 34.9141))"
        )
        # Real WKT syntax check: space-separated coordinate pairs, not commas.
        assert " " in wkt.split("((")[1].split(",")[0]


class TestSearchIsPublic:
    def test_requires_auth_is_false_search_needs_no_credentials(self):
        provider = AlaskaSatelliteFacilityProvider()
        assert provider.REQUIRES_AUTH is False

    def test_search_hits_the_real_endpoint_with_no_auth_header(self):
        provider = AlaskaSatelliteFacilityProvider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_geojson_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": -107, "min_lat": 35, "max_lon": -106, "max_lat": 36}
            )
            provider.search(query)
        args, kwargs = mock_get.call_args
        assert args[0] == "https://api.daac.asf.alaska.edu/services/search/param"
        # Real, confirmed: no Authorization header sent for search.
        assert "headers" not in kwargs or "Authorization" not in kwargs.get(
            "headers", {}
        )


class TestRealSearchParams:
    def test_bbox_becomes_intersects_with_wkt_not_a_bbox_param(self):
        provider = AlaskaSatelliteFacilityProvider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_geojson_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": -107, "min_lat": 35, "max_lon": -106, "max_lat": 36}
            )
            provider.search(query)
        _, kwargs = mock_get.call_args
        assert "bbox" not in kwargs["params"]
        assert kwargs["params"]["intersectsWith"].startswith("POLYGON((")

    def test_satellites_become_real_platform_param(self):
        provider = AlaskaSatelliteFacilityProvider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_geojson_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": -107, "min_lat": 35, "max_lon": -106, "max_lat": 36},
                satellites=["Sentinel-1A", "Sentinel-1B"],
            )
            provider.search(query)
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["platform"] == "Sentinel-1A,Sentinel-1B"

    def test_product_type_becomes_real_processinglevel_param(self):
        provider = AlaskaSatelliteFacilityProvider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_geojson_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": -107, "min_lat": 35, "max_lon": -106, "max_lat": 36},
                product_type="SLC",
            )
            provider.search(query)
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["processingLevel"] == "SLC"

    def test_dates_use_real_start_end_params_not_startdate_enddate(self):
        provider = AlaskaSatelliteFacilityProvider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_geojson_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": -107, "min_lat": 35, "max_lon": -106, "max_lat": 36},
                start_date="2024-01-01",
                end_date="2024-06-01",
            )
            provider.search(query)
        _, kwargs = mock_get.call_args
        assert "startDate" not in kwargs["params"]
        assert "endDate" not in kwargs["params"]
        assert kwargs["params"]["start"] == "2024-01-01T00:00:00UTC"
        assert kwargs["params"]["end"] == "2024-06-01T23:59:59UTC"

    def test_no_cloud_cover_param_is_ever_sent(self):
        """Real, confirmed: ASF's Search API has no cloud-cover keyword
        at all -- SAR data isn't affected by clouds. Verified this is
        never sent, even if a caller sets cloud_cover_max (which
        SearchQuery allows generically, but ASF has nowhere to put it)."""
        provider = AlaskaSatelliteFacilityProvider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_geojson_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": -107, "min_lat": 35, "max_lon": -106, "max_lat": 36},
                cloud_cover_max=50.0,
            )
            provider.search(query)
        _, kwargs = mock_get.call_args
        assert "cloudCoverMax" not in kwargs["params"]
        assert "cloudCover" not in kwargs["params"]


class TestParsing:
    def test_real_feature_parsed_with_no_cloud_cover_field(self):
        provider = AlaskaSatelliteFacilityProvider()
        feature = _real_asf_feature("S1A_IW_SLC__1SDV_20240615T100000")
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_geojson_response([feature])
        with patch("httpx.get", return_value=mock_resp):
            query = SearchQuery(
                bbox={"min_lon": -107, "min_lat": 35, "max_lon": -106, "max_lat": 36}
            )
            results = provider.search(query)

        assert len(results) == 1
        assert results[0].id == "S1A_IW_SLC__1SDV_20240615T100000"
        assert results[0].satellite == "Sentinel-1A"
        assert results[0].cloud_cover is None  # real: not applicable to SAR


class TestAuthentication:
    def test_bearer_token_auth_is_supported(self):
        provider = AlaskaSatelliteFacilityProvider()
        session = provider.authenticate(
            Credentials(
                provider="alaska_satellite_facility", api_key="real-edl-token-xyz"
            )
        )
        assert session.session_data["auth_mode"] == "bearer_token"
        assert session.access_token == "real-edl-token-xyz"

    def test_missing_credentials_raises_clear_error(self):
        provider = AlaskaSatelliteFacilityProvider()
        with pytest.raises(AuthenticationError, match="Earthdata"):
            provider.authenticate(Credentials(provider="alaska_satellite_facility"))

    def test_basic_auth_verified_against_real_urs_endpoint(self):
        provider = AlaskaSatelliteFacilityProvider()
        mock_resp = MagicMock(status_code=200)
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            session = provider.authenticate(
                Credentials(
                    provider="alaska_satellite_facility",
                    username="realuser",
                    password="realpass",
                )
            )
        args, kwargs = mock_get.call_args
        assert args[0] == "https://urs.earthdata.nasa.gov/api/users/user"
        assert kwargs["auth"] == ("realuser", "realpass")
        assert session.session_data["auth_mode"] == "basic"

    def test_rejected_credentials_raise_not_silently_succeed(self):
        provider = AlaskaSatelliteFacilityProvider()
        mock_resp = MagicMock(status_code=401)
        with patch("httpx.get", return_value=mock_resp):
            with pytest.raises(AuthenticationError, match="401"):
                provider.authenticate(
                    Credentials(
                        provider="alaska_satellite_facility",
                        username="baduser",
                        password="badpass",
                    )
                )
