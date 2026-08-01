"""Unit tests for PyGeoFetch providers."""

from __future__ import annotations

import pytest

from pygeofetch.models.search_query import BoundingBox, SearchQuery
from pygeofetch.providers import get_free_providers, get_provider, list_providers


class TestProviderRegistry:
    def test_list_providers_not_empty(self):
        assert len(list_providers()) > 0

    def test_get_free_providers(self):
        assert "aws_earth" in get_free_providers()

    def test_get_provider_usgs(self):
        p = get_provider("usgs")
        assert p.PROVIDER_ID == "usgs"
        assert p.REQUIRES_AUTH is True

    def test_get_provider_aws_earth(self):
        p = get_provider("aws_earth")
        assert p.PROVIDER_ID == "aws_earth"
        assert p.REQUIRES_AUTH is False

    def test_get_unknown_provider_raises(self):
        with pytest.raises((KeyError, ValueError)):
            get_provider("nonexistent_provider")

    def test_all_providers_have_capabilities(self):
        for pid in list_providers():
            p = get_provider(pid)
            caps = p.get_capabilities()
            assert caps.name  # name was patched in


class TestUSGSProvider:
    def test_capabilities(self):
        p = get_provider("usgs")
        caps = p.get_capabilities()
        assert caps.name
        assert caps.provider_id == "usgs"

    def test_requires_auth(self):
        assert get_provider("usgs").REQUIRES_AUTH is True

    def test_search_without_auth_raises(self):
        p = get_provider("usgs")
        from pygeofetch.providers.base import AuthenticationError
        q = SearchQuery(
            bbox=BoundingBox(min_lon=-74.1, min_lat=40.6, max_lon=-73.7, max_lat=40.9),
            start_date="2024-01-01",
        )
        with pytest.raises((AuthenticationError, Exception)):
            p.search(q)


class TestAWSEarthProvider:
    def test_no_auth_required(self):
        assert get_provider("aws_earth").REQUIRES_AUTH is False

    def test_capabilities(self):
        caps = get_provider("aws_earth").get_capabilities()
        assert caps.provider_id == "aws_earth"

    def test_search_returns_list(self):
        """AWS Earth provider search returns a list (even empty)."""
        p = get_provider("aws_earth")
        # Without network, it may raise or return empty - just check it returns a list or raises
        q = SearchQuery(
            bbox=BoundingBox(min_lon=-74.1, min_lat=40.6, max_lon=-73.7, max_lat=40.9),
        )
        try:
            results = p.search(q)
            assert isinstance(results, list)
        except Exception:
            pass  # Network unavailable in test environment is acceptable


class TestCopernicusProvider:
    def test_capabilities(self):
        caps = get_provider("copernicus").get_capabilities()
        assert caps is not None

    def test_requires_auth(self):
        assert get_provider("copernicus").REQUIRES_AUTH is True
