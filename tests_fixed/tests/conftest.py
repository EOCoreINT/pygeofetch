"""Shared pytest fixtures for PyGeoFetch tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pygeofetch.models.download_task import (
    DownloadOptions,
    DownloadResult,
    DownloadStatus,
)
from pygeofetch.models.satellite_data import ProcessingLevel, SatelliteData
from pygeofetch.models.search_query import BoundingBox, SearchQuery
from pygeofetch.models.user_auth import AuthSession, Credentials


@pytest.fixture
def sample_bbox() -> BoundingBox:
    return BoundingBox(min_lon=-74.1, min_lat=40.6, max_lon=-73.7, max_lat=40.9)


@pytest.fixture
def sample_query(sample_bbox) -> SearchQuery:
    return SearchQuery(
        bbox=sample_bbox,
        start_date="2024-01-01",
        end_date="2024-06-01",
        cloud_cover_max=20.0,
        max_results=10,
    )


@pytest.fixture
def sample_satellite_data() -> SatelliteData:
    return SatelliteData(
        id="LC08_L2SP_013032_20240315",
        provider="usgs",
        satellite="Landsat-8",
        sensor="OLI-TIRS",
        cloud_cover=5.2,
        processing_level=ProcessingLevel.L2SP,
        bbox=(-75.0, 40.0, -73.0, 42.0),
        properties={"datetime": "2024-03-15T10:30:00Z"},
    )


@pytest.fixture
def sample_results(sample_satellite_data) -> list:
    items = []
    for i in range(5):
        item = SatelliteData(
            id=f"SCENE_{i:04d}",
            provider="usgs",
            satellite="Landsat-8",
            cloud_cover=float(i * 10),
            properties={"datetime": f"2024-0{i+1}-01T00:00:00Z"},
            bbox=(-74.1, 40.6, -73.7, 40.9),
        )
        items.append(item)
    return items


@pytest.fixture
def sample_credentials() -> Credentials:
    return Credentials(provider="usgs", username="testuser", password="testpass")


@pytest.fixture
def sample_auth_session() -> AuthSession:
    return AuthSession(
        provider="usgs",
        access_token="test-token-abc123",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )


@pytest.fixture
def download_options() -> DownloadOptions:
    return DownloadOptions(parallel=1, retry_attempts=1, verify_checksum=False)


@pytest.fixture
def successful_download_result(tmp_path) -> DownloadResult:
    out = tmp_path / "test_file.tif"
    out.write_bytes(b"\x00" * 1024)
    return DownloadResult(
        status=DownloadStatus.COMPLETED,
        data_id="SCENE_0001",
        provider="usgs",
        output_path=out,
        output_paths=[out],
        bytes_downloaded=1024,
        duration_seconds=1.5,
    )


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.PROVIDER_ID = "mock"
    provider.REQUIRES_AUTH = False
    provider.search.return_value = []
    provider.download.return_value = DownloadResult(
        status=DownloadStatus.COMPLETED,
        data_id="MOCK_001",
        provider="mock",
        bytes_downloaded=512,
        duration_seconds=0.1,
    )
    return provider


@pytest.fixture
def tmp_geojson(tmp_path, sample_results) -> Path:
    from pygeofetch.core.searcher import FederatedSearcher
    searcher = FederatedSearcher()
    path = tmp_path / "results.geojson"
    searcher.save_results(sample_results, path)
    return path
