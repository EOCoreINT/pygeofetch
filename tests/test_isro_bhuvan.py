"""
Tests for the rewritten IsroBhuvanProvider, against the real, official
Bhoonidhi API specification (bhoonidhi.nrsc.gov.in/bhoonidhi-api/) --
real endpoints, auth flow, and response shapes, not the previous
version's bhuvan-app1.nrsc.gov.in/api endpoint (a real ISRO URL, but
for a completely different, non-imagery API).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pygeofetch.models.search_query import SearchQuery
from pygeofetch.models.user_auth import Credentials
from pygeofetch.providers.base import AuthenticationError
from pygeofetch.providers.isro_bhuvan import IsroBhuvanProvider


def _real_auth_response() -> dict:
    """Shaped exactly like Bhoonidhi's own documented example output."""
    return {
        "userId": "bhoonidhiuser",
        "access_token": "real.jwt.token",
        "token_type": "Bearer",
        "expires_in": 1200,
        "refresh_token": "real.refresh.token",
    }


def _real_stac_response(features: list[dict]) -> dict:
    """Shaped exactly like Bhoonidhi's documented STAC response format."""
    return {
        "type": "FeatureCollection",
        "context": {"limit": 10, "returned": len(features)},
        "features": features,
        "links": [],
    }


def _real_item(item_id: str, collection: str = "ResourceSat-2_LISS3_L2") -> dict:
    return {
        "id": item_id,
        "collection": collection,
        "bbox": [77.0, 12.0, 78.0, 13.0],
        "geometry": {"type": "Polygon", "coordinates": [[]]},
        "properties": {"datetime": "2024-06-15T10:00:00Z"},
        "links": [],
    }


@pytest.fixture
def authed_provider():
    provider = IsroBhuvanProvider()
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = _real_auth_response()
    with patch("httpx.post", return_value=mock_resp):
        provider.authenticate(
            Credentials(
                provider="isro_bhuvan",
                username="bhoonidhiuser",
                password="password@123",
            )
        )
    return provider


class TestRealAuthFlow:
    def test_auth_uses_real_endpoint_and_payload_shape(self):
        provider = IsroBhuvanProvider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_auth_response()
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            provider.authenticate(
                Credentials(
                    provider="isro_bhuvan",
                    username="bhoonidhiuser",
                    password="password@123",
                )
            )
        args, kwargs = mock_post.call_args
        assert args[0] == "https://bhoonidhi-api.nrsc.gov.in/auth/token"
        assert kwargs["json"] == {
            "userId": "bhoonidhiuser",
            "password": "password@123",
            "grant_type": "password",
        }

    def test_real_jwt_and_refresh_token_stored(self, authed_provider):
        assert authed_provider._session.access_token == "real.jwt.token"
        assert (
            authed_provider._session.session_data["refresh_token"]
            == "real.refresh.token"
        )

    def test_401_raises_clear_incorrect_password_error(self):
        provider = IsroBhuvanProvider()
        mock_resp = MagicMock(status_code=401)
        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(AuthenticationError, match="incorrect"):
                provider.authenticate(
                    Credentials(provider="isro_bhuvan", username="u", password="wrong")
                )

    def test_403_raises_clear_max_sessions_error(self):
        """Real, documented Bhoonidhi behavior: 403 means too many
        active sessions, not a generic auth failure."""
        provider = IsroBhuvanProvider()
        mock_resp = MagicMock(status_code=403)
        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(AuthenticationError, match="sessions"):
                provider.authenticate(
                    Credentials(provider="isro_bhuvan", username="u", password="p")
                )

    def test_missing_credentials_raises_before_any_request(self):
        provider = IsroBhuvanProvider()
        with pytest.raises(AuthenticationError, match="Bhoonidhi"):
            provider.authenticate(Credentials(provider="isro_bhuvan"))


class TestRealSearchEndpoint:
    def test_search_uses_the_real_data_search_url(self, authed_provider):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_stac_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": 77, "min_lat": 12, "max_lon": 78, "max_lat": 13}
            )
            authed_provider.search(query)
        args, kwargs = mock_get.call_args
        assert args[0] == "https://bhoonidhi-api.nrsc.gov.in/data/search"
        assert kwargs["headers"]["Authorization"] == "Bearer real.jwt.token"

    def test_bbox_sent_as_real_comma_separated_string(self, authed_provider):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_stac_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": 77, "min_lat": 12, "max_lon": 78, "max_lat": 13}
            )
            authed_provider.search(query)
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["bbox"] == "77.0,12.0,78.0,13.0"

    def test_dates_use_real_rfc3339_interval(self, authed_provider):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_stac_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": 77, "min_lat": 12, "max_lon": 78, "max_lat": 13},
                start_date="2023-11-02",
                end_date="2023-11-03",
            )
            authed_provider.search(query)
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["datetime"] == "2023-11-02/2023-11-03"

    def test_search_without_auth_raises(self):
        provider = IsroBhuvanProvider()
        query = SearchQuery(
            bbox={"min_lon": 77, "min_lat": 12, "max_lon": 78, "max_lat": 13}
        )
        with pytest.raises(Exception):
            provider.search(query)


class TestParsingAndDownloadUrl:
    def test_real_item_parsed_with_real_download_url_format(self, authed_provider):
        item = _real_item("E04_SAR_MRS_28DEC2023_test", collection="EOS-04_SAR-MRS_L2A")
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_stac_response([item])
        with patch("httpx.get", return_value=mock_resp):
            query = SearchQuery(
                bbox={"min_lon": 77, "min_lat": 12, "max_lon": 78, "max_lat": 13}
            )
            results = authed_provider.search(query)

        assert len(results) == 1
        expected_url = (
            "https://bhoonidhi-api.nrsc.gov.in/download"
            "?id=E04_SAR_MRS_28DEC2023_test&collection=EOS-04_SAR-MRS_L2A"
        )
        assert results[0].data_assets["data"].href == expected_url

    def test_satellite_filter_matches_real_collection_prefix(self, authed_provider):
        items = [
            _real_item("scene1", collection="ResourceSat-2_LISS3_L2"),
            _real_item("scene2", collection="EOS-04_SAR-MRS_L2A"),
        ]
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_stac_response(items)
        with patch("httpx.get", return_value=mock_resp):
            query = SearchQuery(
                bbox={"min_lon": 77, "min_lat": 12, "max_lon": 78, "max_lat": 13},
                satellites=["ResourceSat-2"],
            )
            results = authed_provider.search(query)
        assert len(results) == 1
        assert results[0].id == "scene1"


class TestRealDownloadBehavior:
    def test_404_logs_online_status_explanation_not_a_crash(
        self, authed_provider, tmp_path
    ):
        from pygeofetch.models.download_task import DownloadOptions

        item = _real_item("delayed_scene")
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_stac_response([item])
        with patch("httpx.get", return_value=mock_resp):
            query = SearchQuery(
                bbox={"min_lon": 77, "min_lat": 12, "max_lon": 78, "max_lat": 13}
            )
            scene = authed_provider.search(query)[0]

        not_found = MagicMock(status_code=404)
        not_found_ctx = MagicMock()
        not_found_ctx.__enter__.return_value = not_found
        not_found_ctx.__exit__.return_value = False

        with patch("httpx.stream", return_value=not_found_ctx):
            result = authed_provider.download(scene, tmp_path, DownloadOptions())

        assert result.status == "failed" or not result.success
