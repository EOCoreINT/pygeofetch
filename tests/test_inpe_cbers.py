"""
Tests for the rewritten InpeCbersProvider, against the real, confirmed
Brazil Data Cube STAC API shape, rather than the fictional
www.dgi.inpe.br/CDSR/search endpoint the previous version assumed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pygeofetch.models.search_query import SearchQuery
from pygeofetch.providers.inpe_cbers import InpeCbersProvider


def _real_stac_response(items: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": items}


def _real_cbers_item(
    item_id: str, collection: str = "CBERS4-MUX-2M", cloud_cover: float = 5.0
) -> dict:
    """Shaped like a real BDC STAC item for a CBERS scene."""
    return {
        "type": "Feature",
        "id": item_id,
        "collection": collection,
        "bbox": [-47.0, -17.0, -46.0, -16.0],
        "geometry": {"type": "Polygon", "coordinates": [[]]},
        "properties": {
            "datetime": "2024-06-15T13:00:00Z",
            "eo:cloud_cover": cloud_cover,
            "platform": "CBERS-4",
        },
        "assets": {
            "BAND16": {
                "href": f"https://data.inpe.br/bdc/data/{item_id}/BAND16.tif",
                "type": "image/tiff",
                "roles": ["data"],
            }
        },
    }


class TestRealEndpoint:
    def test_uses_the_real_bdc_stac_base_url(self):
        provider = InpeCbersProvider()
        assert provider.BASE_URL == "https://data.inpe.br/bdc/stac/v1"

    def test_search_requires_no_authentication(self):
        provider = InpeCbersProvider()
        assert provider.REQUIRES_AUTH is False
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_stac_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": -47, "min_lat": -17, "max_lon": -46, "max_lat": -16}
            )
            provider.search(query)
        args, _ = mock_get.call_args
        assert args[0] == "https://data.inpe.br/bdc/stac/v1/search"


class TestRealQueryParams:
    def test_bbox_sent_as_real_comma_separated_string(self):
        provider = InpeCbersProvider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_stac_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": -47, "min_lat": -17, "max_lon": -46, "max_lat": -16}
            )
            provider.search(query)
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["bbox"] == "-47.0,-17.0,-46.0,-16.0"

    def test_dates_use_real_stac_datetime_interval(self):
        provider = InpeCbersProvider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_stac_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": -47, "min_lat": -17, "max_lon": -46, "max_lat": -16},
                start_date="2024-01-01",
                end_date="2024-06-01",
            )
            provider.search(query)
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["datetime"] == "2024-01-01/2024-06-01"

    def test_real_collections_are_included_by_default(self):
        provider = InpeCbersProvider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_stac_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": -47, "min_lat": -17, "max_lon": -46, "max_lat": -16}
            )
            provider.search(query)
        _, kwargs = mock_get.call_args
        assert "CBERS4-MUX-2M" in kwargs["params"]["collections"]


class TestParsingAndFiltering:
    def test_real_item_parsed_correctly(self):
        provider = InpeCbersProvider()
        item = _real_cbers_item("CBERS_4_MUX_20240615_163_128_L4", cloud_cover=10.0)
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_stac_response([item])
        with patch("httpx.get", return_value=mock_resp):
            query = SearchQuery(
                bbox={"min_lon": -47, "min_lat": -17, "max_lon": -46, "max_lat": -16}
            )
            results = provider.search(query)
        assert len(results) == 1
        assert results[0].id == "CBERS_4_MUX_20240615_163_128_L4"
        assert results[0].cloud_cover == 10.0
        assert results[0].bbox == (-47.0, -17.0, -46.0, -16.0)

    def test_cloud_cover_filter_applied_client_side(self):
        provider = InpeCbersProvider()
        items = [
            _real_cbers_item("clear_scene", cloud_cover=5.0),
            _real_cbers_item("cloudy_scene", cloud_cover=90.0),
        ]
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_stac_response(items)
        with patch("httpx.get", return_value=mock_resp):
            query = SearchQuery(
                bbox={"min_lon": -47, "min_lat": -17, "max_lon": -46, "max_lat": -16},
                cloud_cover_max=20.0,
            )
            results = provider.search(query)
        assert len(results) == 1
        assert results[0].id == "clear_scene"

    def test_empty_response_returns_empty_not_an_error(self):
        provider = InpeCbersProvider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_stac_response([])
        with patch("httpx.get", return_value=mock_resp):
            query = SearchQuery(
                bbox={"min_lon": -47, "min_lat": -17, "max_lon": -46, "max_lat": -16}
            )
            results = provider.search(query)
        assert results == []

    def test_server_error_returns_empty_not_a_crash(self):
        provider = InpeCbersProvider()
        mock_resp = MagicMock(status_code=503)
        with patch("httpx.get", return_value=mock_resp):
            query = SearchQuery(
                bbox={"min_lon": -47, "min_lat": -17, "max_lon": -46, "max_lat": -16}
            )
            results = provider.search(query)
        assert results == []


class TestNoAuthRequired:
    def test_validate_credentials_always_true(self):
        provider = InpeCbersProvider()
        from pygeofetch.models.user_auth import Credentials

        assert provider.validate_credentials(Credentials(provider="inpe_cbers")) is True

    def test_authenticate_never_raises(self):
        provider = InpeCbersProvider()
        from pygeofetch.models.user_auth import Credentials

        session = provider.authenticate(Credentials(provider="inpe_cbers"))
        assert session.access_token == "anonymous"
