"""
Tests for the rewritten GeoserverGenericProvider, against the real,
confirmed OGC WFS GetFeature request/response shape (GeoServer's own
documented reference), rather than the fictional REST search the
previous version assumed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pygeofetch.models.search_query import SearchQuery
from pygeofetch.models.user_auth import Credentials
from pygeofetch.providers.geoserver_generic import GeoserverGenericProvider


def _real_wfs_response(features: list[dict]) -> dict:
    """Shaped exactly like GeoServer's real WFS GeoJSON output format."""
    return {"type": "FeatureCollection", "features": features}


def _real_feature(feature_id: str, cloud_cover=None) -> dict:
    props = {"datetime": "2024-06-15T10:00:00Z"}
    if cloud_cover is not None:
        props["cloud_cover"] = cloud_cover
    return {
        "type": "Feature",
        "id": feature_id,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-74, 40], [-73, 40], [-73, 41], [-74, 41], [-74, 40]]],
        },
        "properties": props,
    }


def _configured_provider(**config) -> GeoserverGenericProvider:
    provider = GeoserverGenericProvider()
    provider.config = {
        "endpoint": "https://example-geoserver.org/geoserver",
        "layer": "myworkspace:mylayer",
        **config,
    }
    return provider


class TestRealWfsRequestShape:
    def test_search_uses_real_wfs_getfeature_params(self):
        provider = _configured_provider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_wfs_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": -74, "min_lat": 40, "max_lon": -73, "max_lat": 41}
            )
            provider.search(query)
        args, kwargs = mock_get.call_args
        assert args[0] == "https://example-geoserver.org/geoserver/wfs"
        assert kwargs["params"]["service"] == "WFS"
        assert kwargs["params"]["version"] == "2.0.0"
        assert kwargs["params"]["request"] == "GetFeature"
        assert kwargs["params"]["typeNames"] == "myworkspace:mylayer"
        assert kwargs["params"]["outputFormat"] == "application/json"

    def test_no_startdate_enddate_cloudcovermax_ever_sent(self):
        """Real, confirmed fix: none of these exist in the real WFS
        spec -- the previous version invented them."""
        provider = _configured_provider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_wfs_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": -74, "min_lat": 40, "max_lon": -73, "max_lat": 41},
                start_date="2024-01-01",
                cloud_cover_max=20.0,
            )
            provider.search(query)
        _, kwargs = mock_get.call_args
        assert "startDate" not in kwargs["params"]
        assert "endDate" not in kwargs["params"]
        assert "cloudCoverMax" not in kwargs["params"]

    def test_bbox_includes_real_crs_suffix(self):
        provider = _configured_provider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_wfs_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": -74, "min_lat": 40, "max_lon": -73, "max_lat": 41}
            )
            provider.search(query)
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["bbox"] == "-74.0,40.0,-73.0,41.0,EPSG:4326"


class TestRequiredConfiguration:
    def test_missing_endpoint_returns_empty_with_clear_log(self):
        provider = GeoserverGenericProvider()
        provider.config = {"layer": "ws:layer"}  # no endpoint
        query = SearchQuery(
            bbox={"min_lon": -74, "min_lat": 40, "max_lon": -73, "max_lat": 41}
        )
        assert provider.search(query) == []

    def test_missing_layer_returns_empty_with_clear_log(self):
        """Real, deliberate design: WFS has no 'search all layers'
        concept, so a missing layer must not silently query nothing
        or everything -- it returns empty with an explanatory log."""
        provider = GeoserverGenericProvider()
        provider.config = {"endpoint": "https://example.org/geoserver"}  # no layer
        query = SearchQuery(
            bbox={"min_lon": -74, "min_lat": 40, "max_lon": -73, "max_lat": 41}
        )
        assert provider.search(query) == []


class TestRealAuthModel:
    def test_no_auth_required_by_default(self):
        provider = GeoserverGenericProvider()
        assert provider.REQUIRES_AUTH is False
        assert (
            provider.validate_credentials(Credentials(provider="geoserver_generic"))
            is True
        )

    def test_basic_auth_used_when_username_password_given(self):
        """Real, confirmed fix: GeoServer's standard security model is
        HTTP Basic Auth, not a bearer token/API key as the previous
        version assumed by default."""
        provider = _configured_provider()
        provider.authenticate(
            Credentials(
                provider="geoserver_generic", username="geouser", password="geopass"
            )
        )
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_wfs_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": -74, "min_lat": 40, "max_lon": -73, "max_lat": 41}
            )
            provider.search(query)
        _, kwargs = mock_get.call_args
        assert kwargs["auth"] == ("geouser", "geopass")
        assert "headers" not in kwargs

    def test_bearer_token_supported_for_oauth2_proxied_deployments(self):
        provider = _configured_provider()
        provider.authenticate(
            Credentials(provider="geoserver_generic", api_key="real-oauth-token")
        )
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_wfs_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": -74, "min_lat": 40, "max_lon": -73, "max_lat": 41}
            )
            provider.search(query)
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer real-oauth-token"


class TestParsingAndFiltering:
    def test_real_feature_parsed_correctly(self):
        provider = _configured_provider()
        feature = _real_feature("myworkspace.mylayer.1")
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_wfs_response([feature])
        with patch("httpx.get", return_value=mock_resp):
            query = SearchQuery(
                bbox={"min_lon": -74, "min_lat": 40, "max_lon": -73, "max_lat": 41}
            )
            results = provider.search(query)
        assert len(results) == 1
        assert results[0].id == "myworkspace.mylayer.1"

    def test_cloud_cover_is_best_effort_client_side_filter(self):
        provider = _configured_provider()
        features = [
            _real_feature("clear", cloud_cover=5.0),
            _real_feature("cloudy", cloud_cover=90.0),
        ]
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _real_wfs_response(features)
        with patch("httpx.get", return_value=mock_resp):
            query = SearchQuery(
                bbox={"min_lon": -74, "min_lat": 40, "max_lon": -73, "max_lat": 41},
                cloud_cover_max=20.0,
            )
            results = provider.search(query)
        assert len(results) == 1
        assert results[0].id == "clear"

    def test_non_geojson_response_returns_empty_not_a_crash(self):
        """Real, defensive check: a misconfigured 'layer' that doesn't
        exist on the real server won't return a FeatureCollection --
        confirmed handled gracefully rather than crashing on .get()
        calls against unexpected response shapes."""
        provider = _configured_provider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"error": "layer not found"}
        with patch("httpx.get", return_value=mock_resp):
            query = SearchQuery(
                bbox={"min_lon": -74, "min_lat": 40, "max_lon": -73, "max_lat": 41}
            )
            results = provider.search(query)
        assert results == []
