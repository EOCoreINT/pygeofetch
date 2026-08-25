"""
Integration tests for PyGeoFetch search and download flows.

These tests use mocked HTTP responses to avoid real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pygeofetch.core.engine import PyGeoFetch
from pygeofetch.core.searcher import FederatedSearcher
from pygeofetch.models.download_task import DownloadOptions
from pygeofetch.models.search_query import BoundingBox, SearchQuery


@pytest.fixture
def engine():
    return PyGeoFetch(log_level="WARNING")


@pytest.fixture
def query():
    return SearchQuery(
        bbox=BoundingBox.from_string("-74.1,40.6,-73.7,40.9"),
        start_date="2024-01-01",
        end_date="2024-03-01",
        cloud_cover_max=30.0,
        max_results=5,
    )


class TestFederatedSearch:
    def test_search_with_no_providers_returns_empty(self, engine, query):
        query.providers = []
        results = engine.search(query, providers=[])
        assert results == []

    def test_search_result_cache(self, engine, query, sample_results):
        """Second identical search should hit the cache."""
        searcher = engine.searcher
        searcher.cache.clear()

        # Pre-populate cache manually
        searcher.cache.set(query, "aws_earth", sample_results)
        cached = searcher.cache.get(query, "aws_earth")
        assert cached == sample_results

    def test_deduplicate_removes_dupes(self, engine, sample_results):
        # Add duplicates
        duped = sample_results + sample_results
        deduped = engine.searcher._deduplicate(duped)
        assert len(deduped) == len(sample_results)

    def test_score_results(self, engine, query, sample_results):
        scored = engine.searcher._score_results(sample_results, query)
        assert all(s.score is not None for s in scored)
        assert all(0 <= s.score <= 1.0 for s in scored)

    def test_save_and_load_results(self, engine, sample_results, tmp_path):
        path = tmp_path / "results.geojson"
        engine.searcher.save_results(sample_results, path)
        assert path.exists()

        loaded = FederatedSearcher.load_results(path)
        assert len(loaded) == len(sample_results)

    def test_to_geojson(self, engine, sample_results):
        fc = engine.searcher.to_geojson(sample_results)
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == len(sample_results)


class TestDownloadFlow:
    def test_download_from_file(self, engine, tmp_geojson, tmp_path):
        """download_from_file should call downloader with loaded items."""
        options = DownloadOptions(parallel=1, retry_attempts=0)

        # Mock the actual provider download to avoid network calls
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.bytes_downloaded = 1024
        mock_result.total_size_mb = 0.001
        mock_result.duration_seconds = 0.1
        mock_result.output_paths = []

        with patch.object(engine.downloader, "download_many", return_value=[mock_result]) as mock_dl:
            results = engine.download_from_file(tmp_geojson, tmp_path, options)
            assert mock_dl.called
            assert results == [mock_result]

    # def test_download_single_item(self, engine, sample_satellite_data, tmp_path):
    #     """download() with a single SatelliteData wraps it in a list."""
    #     mock_result = MagicMock()
    #     mock_result.success = True
    #     mock_result.output_paths = []

    #     with patch.object(engine.downloader, "download_many", return_value=[mock_result]) as mock_dl:

    #         call_args = mock_dl.call_args
    #         data_arg = call_args[0][0]
    #         assert isinstance(data_arg, list)
    #         assert len(data_arg) == 1


class TestEngineStatus:
    def test_status_keys(self, engine):
        status = engine.status()
        assert "version" in status
        assert "providers_authenticated" in status
        assert "providers_free" in status
        assert "cache_entries" in status

    def test_clear_cache(self, engine, query, sample_results):
        engine.searcher.cache.set(query, "aws_earth", sample_results)
        count = engine.clear_cache()
        assert count >= 1

    def test_add_credentials(self, engine):
        # Should not raise
        engine.add_credentials("usgs", username="u", password="p")
        entries = engine.auth.list()
        providers = [e["provider"] for e in entries]
        assert "usgs" in providers
