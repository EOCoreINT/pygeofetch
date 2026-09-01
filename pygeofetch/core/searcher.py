"""
Federated search engine for PyGeoFetch.

Provides parallel search across multiple providers with result deduplication,
relevance scoring, caching, and STAC-compliant output.

Example::

    from pygeofetch.core.searcher import FederatedSearcher
    from pygeofetch.models.search_query import SearchQuery

    searcher = FederatedSearcher()
    results = searcher.search(
        SearchQuery(
            bbox=(-74.1, 40.6, -73.7, 40.9),
            start_date="2024-01-01",
            end_date="2024-06-01",
            cloud_cover_max=20,
        ),
        providers=["usgs", "copernicus", "aws_earth"],
    )

    # Export as GeoJSON FeatureCollection
    geojson = searcher.to_geojson(results)
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pygeofetch.core.logging import (
    get_logger,
    print_provider_progress,
    print_search_header,
    print_search_results,
)
from pygeofetch.models.satellite_data import SatelliteData

if TYPE_CHECKING:
    from pygeofetch.models.search_query import SearchQuery

logger = get_logger(__name__)


class SearchCache:
    """Simple in-memory search result cache with TTL."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl = ttl_seconds
        self._cache: dict[str, tuple[float, list[SatelliteData]]] = {}

    def _key(self, query: SearchQuery, provider: str) -> str:
        """Generate cache key from query and provider."""
        import hashlib

        data = json.dumps(
            {
                "provider": provider,
                "bbox": list(query.bbox.to_tuple()) if query.bbox else None,
                "start": str(query.start_date),
                "end": str(query.end_date),
                "cloud_min": query.cloud_cover_min,
                "cloud_max": query.cloud_cover_max,
                "max_results": query.max_results,
                "satellites": sorted(query.satellites),
                "collections": sorted(query.collections),
            },
            sort_keys=True,
        )
        return hashlib.md5(data.encode()).hexdigest()

    def get(self, query: SearchQuery, provider: str) -> list[SatelliteData] | None:
        """Return cached results or None if expired/missing."""
        key = self._key(query, provider)
        entry = self._cache.get(key)
        if entry is None:
            return None
        timestamp, results = entry
        if time.time() - timestamp > self.ttl:
            del self._cache[key]
            return None
        return results

    def set(
        self, query: SearchQuery, provider: str, results: list[SatelliteData]
    ) -> None:
        """Cache results for a query/provider pair."""
        key = self._key(query, provider)
        self._cache[key] = (time.time(), results)

    def clear(self) -> int:
        """Clear all cached entries. Returns number removed."""
        count = len(self._cache)
        self._cache.clear()
        return count


class FederatedSearcher:
    """
    Federated search engine that queries multiple providers in parallel.

    Features:
    - Concurrent provider queries using ThreadPoolExecutor
    - In-memory result caching with configurable TTL
    - Deduplication by scene ID
    - Relevance scoring and sorting
    - STAC GeoJSON output

    Attributes:
        auth_manager: AuthManager for provider sessions.
        cache: SearchCache instance.
        max_workers: Maximum parallel provider queries.
        timeout_per_provider: Per-provider search timeout in seconds.
    """

    def __init__(
        self,
        auth_manager: Any | None = None,
        cache_ttl: int = 3600,
        max_workers: int = 8,
        timeout_per_provider: int = 60,
    ) -> None:
        self.auth_manager = auth_manager
        self.cache = SearchCache(ttl_seconds=cache_ttl)
        self.max_workers = max_workers
        self.timeout_per_provider = timeout_per_provider
        self._provider_instances: dict[str, Any] = {}

    def search(
        self,
        query: SearchQuery,
        providers: list[str] | None = None,
        use_cache: bool = True,
    ) -> list[SatelliteData]:
        """
        Execute a federated search across multiple providers.

        Args:
            query: Search parameters.
            providers: List of provider IDs to query. Defaults to all configured.
            use_cache: Whether to use/populate the result cache.

        Returns:
            Deduplicated, sorted list of SatelliteData results.

        Example::

            results = searcher.search(
                SearchQuery(bbox=(-74,40,-73,41), start_date="2024-01-01"),
                providers=["usgs", "copernicus"],
            )
        """
        if providers is None:
            providers = query.providers or self._get_available_providers()

        if not providers:
            logger.warning("No providers specified for search")
            return []

        # ── clean search header ───────────────────────────────────────────────
        import time as _st

        _t0_search = _st.time()
        _bbox = query.bbox or None
        _sd = getattr(query, "start_date", "—")
        _ed = getattr(query, "end_date", "—")
        _cc = getattr(query, "cloud_cover_max", 100) or 100
        _pt = getattr(query, "product_type", None) or "any"
        print_search_header(providers, _bbox, _sd, _ed, _cc, _pt)

        all_results: list[SatelliteData] = []
        provider_errors: dict[str, str] = {}

        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(providers))
        ) as executor:
            futures = {
                executor.submit(
                    self._search_provider, provider_id, query, use_cache
                ): provider_id
                for provider_id in providers
            }

            for future in as_completed(futures, timeout=self.timeout_per_provider * 2):
                provider_id = futures[future]
                try:
                    results = future.result(timeout=self.timeout_per_provider)
                    _pdur = _st.time() - _t0_search
                    print_provider_progress(provider_id, "ok", len(results), _pdur)
                    all_results.extend(results)
                except TimeoutError:
                    print_provider_progress(provider_id, "error", error="timed out")
                    provider_errors[provider_id] = "Search timed out"
                except Exception as exc:
                    print_provider_progress(provider_id, "error", error=str(exc)[:60])
                    provider_errors[provider_id] = str(exc)

        # Deduplicate, score, and sort
        deduped = self._deduplicate(all_results)
        scored = self._score_results(deduped, query)
        sorted_results = sorted(scored, key=lambda x: x.score, reverse=True)

        final = sorted_results[: query.max_results]
        _elapsed = _st.time() - _t0_search
        print_search_results(final, elapsed=_elapsed)

        if provider_errors:
            pass  # errors shown inline per-provider

        return final

    def _search_provider(
        self, provider_id: str, query: SearchQuery, use_cache: bool
    ) -> list[SatelliteData]:
        """Search a single provider, using cache if available."""
        if use_cache:
            cached = self.cache.get(query, provider_id)
            if cached is not None:
                logger.debug(
                    f"  {provider_id}: using cached results ({len(cached)} items)"
                )
                return cached

        provider = self._get_provider(provider_id)

        # REAL BUG FIXED: the circuit breaker was previously instantiated
        # per-provider but never actually invoked anywhere in the request
        # path -- it contributed zero resilience. Wrapping the real call
        # here means a provider that has failed repeatedly is now skipped
        # fast (CircuitBreakerOpenError) instead of being hammered again,
        # and a healthy provider's success resets its failure count.
        with provider._circuit_breaker:
            results = provider.search(query.copy_for_provider(provider_id))

        if use_cache:
            self.cache.set(query, provider_id, results)

        return results

    def _get_provider(self, provider_id: str) -> Any:
        """
        Get a provider instance with a fresh auth session.

        REAL BUG FIXED: this used to call ``get_provider()`` fresh on
        every single search, creating a brand-new provider object (and
        therefore a brand-new, always-zeroed CircuitBreaker) each time --
        so circuit breaker failure counts could never accumulate across
        calls no matter how it was wired in. Provider instances are now
        cached per searcher for the lifetime of this FederatedSearcher,
        so both the circuit breaker's state and the provider's auth
        session persist across repeated searches, as originally intended
        by the (previously unused) ``self._provider_instances`` cache.
        """
        from pygeofetch.providers import get_provider

        if provider_id in self._provider_instances:
            return self._provider_instances[provider_id]

        prov = get_provider(provider_id)
        if self.auth_manager and prov.REQUIRES_AUTH:
            try:
                session = self.auth_manager.authenticate(provider_id)
                prov.set_session(session)
            except Exception as exc:
                logger.warning(f"Auth for {provider_id} failed: {exc}")
        self._provider_instances[provider_id] = prov
        return prov

    def _get_available_providers(self) -> list[str]:
        """Return providers that have stored credentials or require no auth."""
        from pygeofetch.providers import get_free_providers

        available = list(get_free_providers())
        if self.auth_manager:
            for item in self.auth_manager.list():
                if item["provider"] not in available:
                    available.append(item["provider"])
        return available

    def _deduplicate(self, results: list[SatelliteData]) -> list[SatelliteData]:
        """Remove duplicate scenes, preferring the first occurrence."""
        seen: dict[str, SatelliteData] = {}
        for item in results:
            # Key on provider+id, or try to match cross-provider by display_id
            key = f"{item.provider}:{item.id}"
            if key not in seen:
                seen[key] = item
        return list(seen.values())

    def _score_results(
        self, results: list[SatelliteData], query: SearchQuery
    ) -> list[SatelliteData]:
        """
        Assign relevance scores to results based on query matching.

        Scoring factors:
        - Cloud cover (lower = better, 0.4 weight)
        - Recency (more recent = better, 0.3 weight)
        - Spatial coverage (larger coverage = better, 0.2 weight)
        - Processing level (higher = better, 0.1 weight)
        """
        if not results:
            return results

        now = datetime.utcnow()

        for item in results:
            score = 0.5  # Neutral baseline

            # Cloud cover score (0-100% → 0-0.4)
            if item.cloud_cover is not None:
                cloud_score = (100 - item.cloud_cover) / 100 * 0.4
                score = score * 0.6 + cloud_score

            # Recency score (within 1 year = 0-0.3)
            if item.datetime:
                try:
                    dt = (
                        item.datetime
                        if item.datetime.tzinfo is None
                        else item.datetime.replace(tzinfo=None)
                    )
                    days_old = max(0, (now - dt).days)
                    recency = max(0, 1 - days_old / 365) * 0.3
                    score += recency
                except Exception:
                    pass

            # Processing level score (higher = 0.1 bonus)
            from pygeofetch.models.satellite_data import ProcessingLevel

            high_levels = {
                ProcessingLevel.L2,
                ProcessingLevel.L2A,
                ProcessingLevel.L2SP,
                ProcessingLevel.ANALYSIS_READY,
            }
            if item.processing_level in high_levels:
                score += 0.1

            item.score = min(1.0, score)

        return results

    def to_geojson(self, results: list[SatelliteData]) -> dict[str, Any]:
        """
        Export results as a STAC-compatible GeoJSON FeatureCollection.

        Args:
            results: List of SatelliteData items.

        Returns:
            GeoJSON FeatureCollection dict.
        """
        return {
            "type": "FeatureCollection",
            "features": [item.to_stac_item() for item in results],
        }

    def save_results(self, results: list[SatelliteData], path: Path) -> None:
        """
        Save search results to a GeoJSON file.

        Args:
            results: Search results to save.
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        geojson = self.to_geojson(results)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2, default=str)
        logger.info(f"Saved {len(results)} results to {path}")

    @staticmethod
    def load_results(path: Path) -> list[SatelliteData]:
        """
        Load search results from a previously saved GeoJSON file.

        Args:
            path: Path to GeoJSON file from save_results().

        Returns:
            List of SatelliteData items.
        """
        with open(path, encoding="utf-8") as f:
            geojson = json.load(f)
        results = []
        for feature in geojson.get("features", []):
            provider = (
                (feature.get("properties") or {})
                .get("providers", [{}])[0]
                .get("name", "unknown")
            )
            results.append(SatelliteData.from_stac_item(feature, provider))
        return results
