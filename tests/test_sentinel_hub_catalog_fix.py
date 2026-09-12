"""
Tests for the real fix to SentinelHubProvider.search(): the previous
request body used a nested structure
({"collections": {"input": [{"type": ...}]}}, {"spatial": {"bbox": ...}},
{"timeRange": {"from": ..., "to": ...}}) that doesn't match the real
Catalog API schema at all. Confirmed directly against Sentinel Hub's
own documented Catalog API examples (docs.sentinel-hub.com/api/latest/
api/catalog/examples/) and the Copernicus Data Space mirror of the same
API: the real body is flat and STAC-compliant (bbox, a single datetime
interval string, collections as plain string IDs).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pygeofetch.models.search_query import SearchQuery
from pygeofetch.providers.sentinel_hub import SentinelHubProvider


def _authed_provider() -> SentinelHubProvider:
    provider = SentinelHubProvider()
    provider._session = MagicMock()
    provider._session.access_token = "real-token"
    return provider


def _mock_response(features=None):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"features": features or []}
    return resp


class TestRealCatalogApiRequestBody:
    def test_matches_sentinel_hubs_own_documented_example_exactly(self):
        """Reproduces the exact real example from Sentinel Hub's own
        Catalog API docs: bbox=[13,45,14,46], a specific date,
        sentinel-1-grd, limit=5."""
        provider = _authed_provider()
        with patch("httpx.post", return_value=_mock_response()) as mock_post:
            query = SearchQuery(
                bbox={"min_lon": 13, "min_lat": 45, "max_lon": 14, "max_lat": 46},
                start_date="2019-12-10",
                end_date="2019-12-10",
                satellites=["sentinel-1"],
                max_results=5,
            )
            provider.search(query)
        _, kwargs = mock_post.call_args
        body = kwargs["json"]
        assert body["bbox"] == [13.0, 45.0, 14.0, 46.0]
        assert body["collections"] == ["sentinel-1-grd"]
        assert body["limit"] == 5
        assert body["datetime"] == "2019-12-10T00:00:00Z/2019-12-10T23:59:59Z"

    def test_no_nested_spatial_or_timerange_keys_ever_sent(self):
        """Real, confirmed fix: neither of these nested wrapper keys
        exist in the real schema."""
        provider = _authed_provider()
        with patch("httpx.post", return_value=_mock_response()) as mock_post:
            query = SearchQuery(
                bbox={"min_lon": 13, "min_lat": 45, "max_lon": 14, "max_lat": 46},
                start_date="2024-01-01",
                end_date="2024-06-01",
            )
            provider.search(query)
        _, kwargs = mock_post.call_args
        body = kwargs["json"]
        assert "spatial" not in body
        assert "timeRange" not in body

    def test_collections_are_plain_string_ids_not_nested_type_objects(self):
        """Real, confirmed fix: {"collections": {"input": [{"type": ...}]}}
        doesn't match the real schema -- collections is a plain array
        of real STAC collection ID strings."""
        provider = _authed_provider()
        with patch("httpx.post", return_value=_mock_response()) as mock_post:
            query = SearchQuery(
                bbox={"min_lon": 13, "min_lat": 45, "max_lon": 14, "max_lat": 46},
                satellites=["sentinel-2"],
            )
            provider.search(query)
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["collections"] == ["sentinel-2-l2a"]

    def test_cloud_cover_uses_real_cql2_json_filter_format(self):
        """Real, confirmed fix: the old {"maxCloudCoverage": N} key is
        fictional -- the real Catalog API filter extension uses
        CQL2-JSON."""
        provider = _authed_provider()
        with patch("httpx.post", return_value=_mock_response()) as mock_post:
            query = SearchQuery(
                bbox={"min_lon": 13, "min_lat": 45, "max_lon": 14, "max_lat": 46},
                cloud_cover_max=15.0,
            )
            provider.search(query)
        _, kwargs = mock_post.call_args
        body = kwargs["json"]
        assert "maxCloudCoverage" not in body
        assert body["filter"] == {
            "op": "lte",
            "args": [{"property": "eo:cloud_cover"}, 15.0],
        }
        assert body["filter-lang"] == "cql2-json"


class TestRealCollectionIdMapping:
    """Confirms the collection IDs are the real, lowercase-hyphenated
    Catalog API identifiers, not the older Process API 'type' codes
    (S2L2A, S1GRD) the previous version used."""

    def test_sentinel1_maps_to_the_real_catalog_id(self):
        provider = SentinelHubProvider()
        assert provider.DATA_SOURCE_MAP["sentinel-1"] == "sentinel-1-grd"

    def test_sentinel2_maps_to_the_real_catalog_id(self):
        provider = SentinelHubProvider()
        assert provider.DATA_SOURCE_MAP["sentinel-2"] == "sentinel-2-l2a"

    def test_no_old_process_api_style_codes_remain(self):
        provider = SentinelHubProvider()
        values = list(provider.DATA_SOURCE_MAP.values())
        assert "S1GRD" not in values
        assert "S2L2A" not in values
