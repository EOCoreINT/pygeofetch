"""
Regression tests for the esa_scihub provider.

Previously: search()/download() called self._check_integration_verified(),
a method that does not exist anywhere in this codebase -- every single
call crashed with AttributeError, regardless of credentials or query.
Separately, its BASE_URL points at the Copernicus Open Access Hub,
which ESA permanently decommissioned on 2023-11-02.

This verifies both are now handled honestly: no crash, and a clear,
immediate, actionable message pointing to the live `copernicus`
provider instead -- rather than a confusing AttributeError, or a real
network call doomed to fail against a dead host.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pygeofetch.models.search_query import SearchQuery
from pygeofetch.providers.esa_scihub import EsaScihubProvider


@pytest.fixture
def provider():
    p = EsaScihubProvider()
    p.authenticate(None)
    return p


class TestCrashFixed:
    def test_search_no_longer_raises_attributeerror(self, provider):
        """REAL BUG FIXED: this used to crash with AttributeError on
        the missing _check_integration_verified() method every time."""
        results = provider.search(SearchQuery())
        assert results == []

    def test_download_no_longer_raises_attributeerror(self, provider, tmp_path):
        from pygeofetch.models.download_task import DownloadOptions
        from pygeofetch.models.satellite_data import SatelliteData

        item = SatelliteData(id="scene1", provider="esa_scihub")
        result = provider.download(item, tmp_path, DownloadOptions())
        assert result.status.value == "failed"


class TestHonestDeadEndpointHandling:
    def test_search_never_makes_a_network_call(self, provider):
        """A confirmed-dead host shouldn't burn a real network timeout
        on every search -- fail fast instead."""
        with patch("httpx.get") as mock_get:
            provider.search(SearchQuery())
        mock_get.assert_not_called()

    def test_download_never_makes_a_network_call(self, provider, tmp_path):
        from pygeofetch.models.download_task import DownloadOptions
        from pygeofetch.models.satellite_data import SatelliteData

        item = SatelliteData(id="scene1", provider="esa_scihub")
        with patch("httpx.stream") as mock_stream:
            provider.download(item, tmp_path, DownloadOptions())
        mock_stream.assert_not_called()

    def test_download_error_points_to_the_live_replacement(self, provider, tmp_path):
        from pygeofetch.models.download_task import DownloadOptions
        from pygeofetch.models.satellite_data import SatelliteData

        item = SatelliteData(id="scene1", provider="esa_scihub")
        result = provider.download(item, tmp_path, DownloadOptions())
        assert "copernicus" in result.error
        assert "decommissioned" in result.error

    def test_search_does_not_require_credentials(self):
        """esa_scihub is REQUIRES_AUTH = False; a fresh, unauthenticated
        provider must still fail honestly rather than crash."""
        p = EsaScihubProvider()
        results = p.search(SearchQuery())
        assert results == []
