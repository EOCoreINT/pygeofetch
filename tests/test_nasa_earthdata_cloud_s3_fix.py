"""
Tests for the real fixes to NASAEarthdataCloudProvider:

1. There is no single, unified S3 credentials endpoint -- confirmed
   identically across NASA's own EMIT-Data-Resources repo, PO.DAAC's
   cookbook, NSIDC's help center, and multiple NASA Openscapes
   tutorials, all stating explicitly that each DAAC has its own
   endpoint and credentials from one DAAC don't work for another's
   buckets. The previous single "data.earthaccess.nasa.gov" URL didn't
   match any real DAAC's actual endpoint.
2. search() had the same unnecessary auth requirement found in
   nasa_earthdata.py -- CMR search (even with cloud_hosted=true) is
   genuinely public.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from pygeofetch.models.search_query import SearchQuery
from pygeofetch.providers.nasa_earthdata_cloud import NASAEarthdataCloudProvider


def _mock_cmr_response(entries: list[dict]) -> dict:
    return {"feed": {"entry": entries}}


class TestSearchIsGenuinelyPublic:
    def test_search_succeeds_with_no_prior_authentication(self):
        provider = NASAEarthdataCloudProvider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _mock_cmr_response([])
        with patch("httpx.get", return_value=mock_resp):
            query = SearchQuery(
                bbox={"min_lon": -74, "min_lat": 40, "max_lon": -73, "max_lat": 41}
            )
            results = provider.search(query)  # must not raise
        assert results == []

    def test_no_bearer_none_header_sent_when_unauthenticated(self):
        """Real, confirmed fix: the old code sent a literal
        'Authorization: Bearer None' header when unauthenticated."""
        provider = NASAEarthdataCloudProvider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _mock_cmr_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": -74, "min_lat": 40, "max_lon": -73, "max_lat": 41}
            )
            provider.search(query)
        _, kwargs = mock_get.call_args
        assert "Bearer None" not in str(kwargs.get("headers", {}))


class TestRealPerDaacCredentials:
    def test_daac_mapping_has_no_fictional_unified_endpoint(self):
        """The specific real bug: confirms the old
        'data.earthaccess.nasa.gov' URL is gone, replaced by real,
        confirmed per-DAAC endpoints."""
        provider = NASAEarthdataCloudProvider()
        all_endpoints = " ".join(provider.DAAC_S3_CREDENTIALS_ENDPOINTS.values())
        assert "data.earthaccess.nasa.gov" not in all_endpoints

    def test_real_confirmed_daac_endpoints_present(self):
        provider = NASAEarthdataCloudProvider()
        assert provider.DAAC_S3_CREDENTIALS_ENDPOINTS["PODAAC"] == (
            "https://archive.podaac.earthdata.nasa.gov/s3credentials"
        )
        assert provider.DAAC_S3_CREDENTIALS_ENDPOINTS["LPDAAC_ECS"] == (
            "https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials"
        )
        assert provider.DAAC_S3_CREDENTIALS_ENDPOINTS["NSIDC_ECS"] == (
            "https://data.nsidc.earthdatacloud.nasa.gov/s3credentials"
        )

    def test_authenticate_does_not_eagerly_fetch_any_s3_credentials(self):
        """Real, confirmed fix: authenticate() no longer guesses which
        DAAC to fetch credentials for -- it can't know yet."""
        provider = NASAEarthdataCloudProvider()
        mock_resp = MagicMock(status_code=201)
        mock_resp.json.return_value = {"access_token": "real-edl-token"}
        from pygeofetch.models.user_auth import Credentials

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            session = provider.authenticate(
                Credentials(provider="nasa_earthdata_cloud", username="u", password="p")
            )
        # Only one real network call (the EDL token request) -- no
        # second call guessing at S3 credentials.
        assert mock_post.call_count == 1
        assert session.session_data["s3_credentials_by_daac"] == {}

    def test_fetches_real_credentials_for_the_specific_granules_daac(self):
        provider = NASAEarthdataCloudProvider()
        provider._session = MagicMock()
        provider._session.access_token = "real-edl-token"
        provider._session.session_data = {"s3_credentials_by_daac": {}}

        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "accessKeyId": "AKIA...",
            "secretAccessKey": "secret",
            "sessionToken": "token",
            "expiration": "2024-01-01T00:00:00Z",
        }
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            creds = provider._get_s3_credentials_for_daac("LPDAAC_ECS")

        args, _ = mock_get.call_args
        assert args[0] == "https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials"
        assert creds["aws_access_key_id"] == "AKIA..."

    def test_unknown_daac_falls_back_gracefully_not_an_error(self):
        provider = NASAEarthdataCloudProvider()
        provider._session = MagicMock()
        provider._session.session_data = {"s3_credentials_by_daac": {}}

        creds = provider._get_s3_credentials_for_daac("SOME_UNMAPPED_DAAC")
        assert creds == {}

    def test_cached_credentials_reused_before_expiration(self):
        provider = NASAEarthdataCloudProvider()
        provider._session = MagicMock()
        provider._session.session_data = {
            "s3_credentials_by_daac": {
                "PODAAC": {
                    "aws_access_key_id": "cached-key",
                    "_expires_at": datetime.now(timezone.utc) + timedelta(minutes=30),
                }
            }
        }
        with patch("httpx.get") as mock_get:
            creds = provider._get_s3_credentials_for_daac("PODAAC")
        mock_get.assert_not_called()
        assert creds["aws_access_key_id"] == "cached-key"

    def test_expired_cached_credentials_are_refetched(self):
        provider = NASAEarthdataCloudProvider()
        provider._session = MagicMock()
        provider._session.access_token = "real-edl-token"
        provider._session.session_data = {
            "s3_credentials_by_daac": {
                "PODAAC": {
                    "aws_access_key_id": "old-expired-key",
                    "_expires_at": datetime.now(timezone.utc) - timedelta(minutes=5),
                }
            }
        }
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "accessKeyId": "new-key",
            "secretAccessKey": "s",
            "sessionToken": "t",
            "expiration": "x",
        }
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            creds = provider._get_s3_credentials_for_daac("PODAAC")
        mock_get.assert_called_once()
        assert creds["aws_access_key_id"] == "new-key"


class TestGranuleParsing:
    def test_data_center_captured_in_properties(self):
        provider = NASAEarthdataCloudProvider()
        entry = {
            "id": "G12345",
            "data_center": "LPDAAC_ECS",
            "title": "test granule",
            "links": [],
        }
        scene = provider._parse_granule(entry)
        assert scene.properties["data_center"] == "LPDAAC_ECS"
