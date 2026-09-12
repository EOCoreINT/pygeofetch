"""
Tests for the real fix to NASAEarthdataProvider.search(): CMR granule
search is genuinely public (confirmed across NASA's own LP DAAC bulk-
query tutorial, the official cmrfetch CLI, and the earthaccess
library -- none attach Earthdata credentials to the search request
itself). The previous code unconditionally indexed
`self._session.session_data["username"]`, defaulting to `{}` when no
session existed -- which raised a real KeyError on the very first
search() call for anyone who hadn't authenticated yet, even though CMR
never needed those credentials for search at all.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pygeofetch.models.search_query import SearchQuery
from pygeofetch.models.user_auth import Credentials
from pygeofetch.providers.nasa_earthdata import NASAEarthdataProvider


def _mock_cmr_response(entries: list[dict]) -> dict:
    return {"feed": {"entry": entries}}


class TestSearchIsGenuinelyPublic:
    def test_search_succeeds_with_no_prior_authentication(self):
        """The specific real bug: this used to raise KeyError before
        this fix, for anyone calling search() before authenticate()."""
        provider = NASAEarthdataProvider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _mock_cmr_response([])
        with patch("httpx.get", return_value=mock_resp):
            query = SearchQuery(
                bbox={"min_lon": -74, "min_lat": 40, "max_lon": -73, "max_lat": 41}
            )
            results = provider.search(query)  # must not raise
        assert results == []

    def test_no_auth_param_sent_when_unauthenticated(self):
        provider = NASAEarthdataProvider()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _mock_cmr_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": -74, "min_lat": 40, "max_lon": -73, "max_lat": 41}
            )
            provider.search(query)
        _, kwargs = mock_get.call_args
        assert kwargs["auth"] is None

    def test_real_credentials_still_attached_when_authenticated(self):
        """Confirms the fix doesn't break the case where a caller DID
        authenticate first -- credentials should still be sent."""
        provider = NASAEarthdataProvider()
        mock_auth_resp = MagicMock(status_code=200)
        with patch("httpx.get", return_value=mock_auth_resp):
            provider.authenticate(
                Credentials(
                    provider="nasa_earthdata", username="edluser", password="edlpass"
                )
            )

        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = _mock_cmr_response([])
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            query = SearchQuery(
                bbox={"min_lon": -74, "min_lat": 40, "max_lon": -73, "max_lat": 41}
            )
            provider.search(query)
        _, kwargs = mock_get.call_args
        assert kwargs["auth"] == ("edluser", "edlpass")

    def test_download_still_requires_real_authentication(self):
        """The other half of this fix: only download() should still
        gate on require_auth() -- actually fetching DAAC-hosted files
        does need a real Earthdata Login account."""
        import inspect

        source = inspect.getsource(NASAEarthdataProvider.download)
        assert "require_auth" in source
