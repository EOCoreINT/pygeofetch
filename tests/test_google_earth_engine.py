"""
Regression tests for the google_earth_engine provider.

Previously: search()/download() called self._check_integration_verified(),
a method that does not exist anywhere in this codebase -- every single
call crashed with AttributeError. Separately, the rest of the
implementation pretended to be a simple REST search+download API, when
Earth Engine's real API is structurally different (service-account JWT
auth, asset/computation API, not a "/search" REST endpoint). This
verifies the crash is gone and the provider now fails honestly and
immediately instead of either crashing or pretending to work.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pygeofetch.models.search_query import SearchQuery
from pygeofetch.models.user_auth import Credentials
from pygeofetch.providers.base import AuthenticationError
from pygeofetch.providers.google_earth_engine import GoogleEarthEngineProvider


@pytest.fixture
def provider():
    return GoogleEarthEngineProvider()


class TestCrashFixed:
    def test_search_no_longer_raises_attributeerror(self, provider):
        """REAL BUG FIXED: this used to crash with AttributeError on
        the missing _check_integration_verified() method every time."""
        results = provider.search(SearchQuery())
        assert results == []

    def test_download_no_longer_raises_attributeerror(self, provider, tmp_path):
        from pygeofetch.models.download_task import DownloadOptions
        from pygeofetch.models.satellite_data import SatelliteData

        item = SatelliteData(id="scene1", provider="google_earth_engine")
        result = provider.download(item, tmp_path, DownloadOptions())
        assert result.status.value == "failed"


class TestHonestNotImplementedHandling:
    def test_authenticate_raises_a_clear_actionable_error(self, provider):
        with pytest.raises(AuthenticationError, match="not yet implemented"):
            provider.authenticate(
                Credentials(provider="google_earth_engine", api_key="whatever")
            )

    def test_validate_credentials_is_honestly_false(self, provider):
        assert (
            provider.validate_credentials(
                Credentials(provider="google_earth_engine", api_key="whatever")
            )
            is False
        )

    def test_search_never_makes_a_network_call(self, provider):
        """Fail fast and honestly rather than attempting a REST call
        against a URL that isn't Earth Engine's real API."""
        with patch("httpx.get") as mock_get:
            provider.search(SearchQuery())
        mock_get.assert_not_called()

    def test_download_error_explains_what_real_integration_is_missing(
        self, provider, tmp_path
    ):
        from pygeofetch.models.download_task import DownloadOptions
        from pygeofetch.models.satellite_data import SatelliteData

        item = SatelliteData(id="scene1", provider="google_earth_engine")
        result = provider.download(item, tmp_path, DownloadOptions())
        assert "service-account" in result.error
        assert "Earth Engine" in result.error
