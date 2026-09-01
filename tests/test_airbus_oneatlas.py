"""
Regression tests for the Airbus OneAtlas provider.

Previously: authenticate() never called any real endpoint (it just
wrapped the raw API key as if it were already a bearer token), search()
hit a fictional URL with made-up parameter names, and the result parser
never populated `assets` -- so download() always failed with "No assets
downloaded" regardless of credentials. This verifies the real,
replacement implementation against response shapes taken directly from
Airbus's own documented examples (not invented), with HTTP calls mocked
so no real network/credentials are needed to run these.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from pygeofetch.models.search_query import BoundingBox, SearchQuery
from pygeofetch.models.user_auth import Credentials
from pygeofetch.providers.airbus_oneatlas import AirbusOneatlasProvider
from pygeofetch.providers.base import AuthenticationError

# A real response shape for the token endpoint, per Airbus's own
# documented example.
REAL_TOKEN_RESPONSE = {
    "access_token": "eyJhbGciOi.FAKE.TOKEN",
    "expires_in": 3600,
    "refresh_expires_in": 0,
    "token_type": "bearer",
}

# A real response shape for /api/v2/opensearch, adapted directly from
# Airbus's own documented example (single-image and multi-image cases).
REAL_SEARCH_RESPONSE = {
    "type": "FeatureCollection",
    "totalResults": 2,
    "itemsPerPage": 10,
    "startIndex": 0,
    "features": [
        {
            "id": "df4c27e2-ce70-4f3d-8928-5e27dfe12094",
            "properties": {
                "acquisitionDate": "2024-02-14T10:30:00.000Z",
                "cloudCover": 4.2,
                "constellation": "PHR",
                "sensorType": "PAN",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [1.240702761700962, 43.71466198202646],
                        [1.567967301703224, 43.70248156062971],
                        [1.568833709452247, 43.470228154858],
                        [1.239288081223572, 43.48647760616458],
                        [1.240702761700962, 43.71466198202646],
                    ]
                ],
            },
            "_links": {
                "imagesGetBuffer": [
                    {
                        "href": "https://access.foundation.api.oneatlas.airbus.com/api/v1/items/df4c27e2-ce70-4f3d-8928-5e27dfe12094/images/e754f4a4-15b2-47aa-a06f-4498fe2e05a3/buffer",
                        "name": "panchromatic",
                        "type": "getBuffer",
                        "resourceId": "e754f4a4-15b2-47aa-a06f-4498fe2e05a3",
                    },
                ],
            },
        },
        {
            "id": "20abfc7e-ece9-4ba5-a61f-654d667990dd",
            "properties": {
                "acquisitionDate": "2024-03-01T09:15:00.000Z",
                "cloudCover": 12.0,
                "constellation": "SPOT",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-74.1, 40.6],
                        [-73.7, 40.6],
                        [-73.7, 40.9],
                        [-74.1, 40.9],
                        [-74.1, 40.6],
                    ]
                ],
            },
            "_links": {
                # Single-image products return a plain dict, not a list --
                # both shapes are real and must both be handled.
                "imagesGetBuffer": {
                    "href": "https://access.foundation.api.oneatlas.airbus.com/api/v1/items/20abfc7e-ece9-4ba5-a61f-654d667990dd/images/3a8ced78-dd58-44b9-9de0-89ef3bdb7f98/buffer",
                    "type": "getBuffer",
                    "resourceId": "3a8ced78-dd58-44b9-9de0-89ef3bdb7f98",
                },
            },
        },
    ],
}


@pytest.fixture
def provider():
    return AirbusOneatlasProvider()


@pytest.fixture
def authenticated_provider(provider):
    with patch("httpx.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200, json=lambda: REAL_TOKEN_RESPONSE
        )
        provider.authenticate(
            Credentials(provider="airbus_oneatlas", api_key="fake-key")
        )
    return provider


class TestRealAuthentication:
    def test_authenticate_calls_the_real_token_endpoint(self, provider):
        """REAL BUG FIXED: authenticate() used to never call any real
        endpoint at all."""
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200, json=lambda: REAL_TOKEN_RESPONSE
            )
            session = provider.authenticate(
                Credentials(provider="airbus_oneatlas", api_key="my-real-key")
            )

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == (
            "https://authenticate.foundation.api.oneatlas.airbus.com"
            "/auth/realms/IDP/protocol/openid-connect/token"
        )
        assert call_args.kwargs["data"] == {
            "apikey": "my-real-key",
            "grant_type": "api_key",
            "client_id": "IDP",
        }
        assert session.access_token == "eyJhbGciOi.FAKE.TOKEN"

    def test_no_api_key_raises_clear_error_without_a_network_call(self, provider):
        with patch("httpx.post") as mock_post:
            with pytest.raises(AuthenticationError, match="requires an API key"):
                provider.authenticate(Credentials(provider="airbus_oneatlas"))
        mock_post.assert_not_called()

    def test_rejected_key_raises_authentication_error(self, provider):
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=403, text="Access denied")
            with pytest.raises(AuthenticationError, match="rejected"):
                provider.authenticate(
                    Credentials(provider="airbus_oneatlas", api_key="bad-key")
                )


class TestRealSearch:
    def test_search_hits_the_real_opensearch_endpoint(self, authenticated_provider):
        """REAL BUG FIXED: search() used to hit a fictional endpoint on
        the wrong host with parameter names the real API doesn't
        recognise."""
        query = SearchQuery(
            bbox=BoundingBox(min_lon=-74.1, min_lat=40.6, max_lon=-73.7, max_lat=40.9),
            cloud_cover_max=20,
        )
        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200, json=lambda: REAL_SEARCH_RESPONSE
            )
            results = authenticated_provider.search(query)

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert (
            call_args[0][0]
            == "https://search.foundation.api.oneatlas.airbus.com/api/v2/opensearch"
        )
        params = call_args.kwargs["params"]
        assert params["bbox"] == "-74.1,40.6,-73.7,40.9"
        assert params["cloudCover"] == "[0,20.0]"
        assert params["processingLevel"] == "SENSOR"
        assert "Authorization" in call_args.kwargs["headers"]

        assert len(results) == 2

    def test_search_maps_constellation_to_real_satellite_name(
        self, authenticated_provider
    ):
        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200, json=lambda: REAL_SEARCH_RESPONSE
            )
            results = authenticated_provider.search(SearchQuery())
        assert results[0].satellite == "Pleiades"
        assert results[1].satellite == "SPOT"

    def test_search_computes_bbox_from_polygon_geometry(self, authenticated_provider):
        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200, json=lambda: REAL_SEARCH_RESPONSE
            )
            results = authenticated_provider.search(SearchQuery())
        # second feature's polygon exactly matches the requested bbox
        assert results[1].bbox == pytest.approx((-74.1, 40.6, -73.7, 40.9))

    def test_search_with_date_range_uses_real_acquisitionDate_param(
        self, authenticated_provider
    ):
        query = SearchQuery(start_date=date(2024, 1, 1), end_date=date(2024, 6, 1))
        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200, json=lambda: REAL_SEARCH_RESPONSE
            )
            authenticated_provider.search(query)
        params = mock_get.call_args.kwargs["params"]
        assert (
            params["acquisitionDate"]
            == "[2024-01-01T00:00:00.000Z,2024-06-01T00:00:00.000Z]"
        )


class TestRealAssetPopulation:
    """The core bug: assets must actually be populated so download() has
    something real to download."""

    def test_multi_image_result_populates_a_real_data_asset(
        self, authenticated_provider
    ):
        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200, json=lambda: REAL_SEARCH_RESPONSE
            )
            results = authenticated_provider.search(SearchQuery())

        item = results[0]
        assert (
            item.data_assets
        ), "data_assets must not be empty -- this was the core bug"
        asset = item.data_assets["panchromatic"]
        assert asset.href == (
            "https://access.foundation.api.oneatlas.airbus.com/api/v1/items/"
            "df4c27e2-ce70-4f3d-8928-5e27dfe12094/images/"
            "e754f4a4-15b2-47aa-a06f-4498fe2e05a3/buffer"
        )

    def test_single_image_dict_shape_also_populates_an_asset(
        self, authenticated_provider
    ):
        """Airbus returns imagesGetBuffer as a plain dict (not a list)
        for single-image products -- both shapes are real."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200, json=lambda: REAL_SEARCH_RESPONSE
            )
            results = authenticated_provider.search(SearchQuery())

        item = results[1]
        assert item.data_assets
        asset = next(iter(item.data_assets.values()))
        assert "3a8ced78-dd58-44b9-9de0-89ef3bdb7f98" in asset.href


class TestRealDownload:
    def test_download_actually_fetches_the_real_asset_url(
        self, authenticated_provider, tmp_path
    ):
        """End to end: search populates a real asset, download streams
        it -- the exact path that used to always fail with 'No assets
        downloaded'."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200, json=lambda: REAL_SEARCH_RESPONSE
            )
            results = authenticated_provider.search(SearchQuery())
        item = results[0]

        fake_stream_resp = MagicMock()
        fake_stream_resp.status_code = 200
        fake_stream_resp.headers = {"content-length": "8"}
        fake_stream_resp.iter_bytes = lambda chunk_size: [b"fakedata"]
        fake_stream_resp.__enter__ = lambda self: fake_stream_resp
        fake_stream_resp.__exit__ = lambda *a: None

        from pygeofetch.models.download_task import DownloadOptions

        with patch("httpx.stream", return_value=fake_stream_resp) as mock_stream:
            result = authenticated_provider.download(item, tmp_path, DownloadOptions())

        assert result.status.value == "completed"
        assert result.bytes_downloaded == 8
        assert len(result.output_paths) == 1
        assert result.output_paths[0].exists()
        assert result.output_paths[0].read_bytes() == b"fakedata"

        call_args = mock_stream.call_args
        assert call_args[0][1] == item.data_assets["panchromatic"].href

    def test_download_with_no_assets_fails_with_a_clear_actionable_error(
        self, authenticated_provider, tmp_path
    ):
        from pygeofetch.models.download_task import DownloadOptions
        from pygeofetch.models.satellite_data import SatelliteData

        empty_item = SatelliteData(id="no-assets-item", provider="airbus_oneatlas")
        result = authenticated_provider.download(
            empty_item, tmp_path, DownloadOptions()
        )

        assert result.status.value == "failed"
        assert "ALBUM" in result.error or "assets" in result.error.lower()
