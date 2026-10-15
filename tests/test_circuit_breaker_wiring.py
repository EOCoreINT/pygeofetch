"""
Regression tests for FederatedSearcher's circuit breaker wiring.

Previously: CircuitBreaker was instantiated per-provider in
AbstractBaseProvider.__init__ but never invoked anywhere in the
request path, AND get_provider() created a brand-new provider
instance (with a fresh, always-zeroed breaker) on every single call --
so even if something had wrapped a call in `with breaker:`, the
failure count could never have accumulated across searches anyway.

This file verifies both real fixes:
  1. FederatedSearcher._get_provider() now caches provider instances,
     so a provider's CircuitBreaker genuinely persists across calls.
  2. FederatedSearcher._search_provider() actually uses that breaker,
     so a provider that fails repeatedly gets skipped fast instead of
     hammered again.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pygeofetch.core.searcher import FederatedSearcher
from pygeofetch.models.search_query import SearchQuery
from pygeofetch.utils.retry_handler import CircuitBreakerOpenError


class TestProviderInstanceCaching:
    def test_same_provider_id_returns_same_instance(self):
        """
        REAL BUG FIXED: _get_provider() used to construct a brand-new
        provider object on every call. This confirms repeated calls for
        the same provider now return the identical cached instance.
        """
        searcher = FederatedSearcher()
        p1 = searcher._get_provider("aws_earth")
        p2 = searcher._get_provider("aws_earth")
        assert p1 is p2

    def test_different_providers_get_different_instances(self):
        searcher = FederatedSearcher()
        p1 = searcher._get_provider("aws_earth")
        p2 = searcher._get_provider("element84")
        assert p1 is not p2

    def test_circuit_breaker_identity_persists_across_calls(self):
        """The whole point: since the provider instance is now cached,
        its ._circuit_breaker is the same object across calls too, so
        failure state can genuinely accumulate."""
        searcher = FederatedSearcher()
        breaker1 = searcher._get_provider("aws_earth")._circuit_breaker
        breaker2 = searcher._get_provider("aws_earth")._circuit_breaker
        assert breaker1 is breaker2


class TestCircuitBreakerActuallyTrips:
    def test_repeated_failures_open_the_breaker(self):
        """
        REAL BUG FIXED: previously, no amount of provider failures would
        ever open the circuit breaker, because (a) it was never invoked
        in the request path, and (b) even if it had been, a fresh
        provider/breaker was created every call. This drives real
        failures through the actual _search_provider() path and
        confirms the breaker genuinely opens after threshold failures.
        """
        searcher = FederatedSearcher()
        query = SearchQuery(bbox=(-74.1, 40.6, -73.7, 40.9))

        provider = searcher._get_provider("aws_earth")
        assert provider._circuit_breaker.failure_threshold == 5

        with patch.object(
            provider, "search", side_effect=RuntimeError("simulated provider outage")
        ):
            for _ in range(5):
                with pytest.raises(RuntimeError):
                    searcher._search_provider("aws_earth", query, use_cache=False)

        assert provider._circuit_breaker.is_open

        # The 6th call must now fail FAST with CircuitBreakerOpenError,
        # from the breaker itself -- never even reaching provider.search().
        with patch.object(
            provider, "search", side_effect=RuntimeError("should not be called")
        ) as mock_search:
            with pytest.raises(CircuitBreakerOpenError):
                searcher._search_provider("aws_earth", query, use_cache=False)
            mock_search.assert_not_called()

    def test_success_resets_failure_count(self):
        searcher = FederatedSearcher()
        query = SearchQuery(bbox=(-74.1, 40.6, -73.7, 40.9))
        provider = searcher._get_provider("aws_earth")

        with patch.object(provider, "search", side_effect=RuntimeError("boom")):
            for _ in range(3):
                with pytest.raises(RuntimeError):
                    searcher._search_provider("aws_earth", query, use_cache=False)
        assert provider._circuit_breaker._failure_count == 3

        with patch.object(provider, "search", return_value=[]):
            searcher._search_provider("aws_earth", query, use_cache=False)
        assert provider._circuit_breaker._failure_count == 0
        assert not provider._circuit_breaker.is_open

    def test_open_breaker_surfaces_as_a_normal_provider_error_in_full_search(self):
        """A tripped breaker must degrade gracefully in the real,
        multi-provider search() path -- not crash the whole search."""
        searcher = FederatedSearcher()
        query = SearchQuery(bbox=(-74.1, 40.6, -73.7, 40.9), providers=["aws_earth"])
        provider = searcher._get_provider("aws_earth")
        provider._circuit_breaker._state = provider._circuit_breaker.OPEN
        provider._circuit_breaker._last_failure_time = __import__("time").monotonic()

        results = searcher.search(query, providers=["aws_earth"], use_cache=False)
        assert results == []
