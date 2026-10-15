"""Data models for PyGeoFetch."""

from pygeofetch.models.download_task import (
    ChecksumAlgorithm,
    DownloadOptions,
    DownloadProgress,
    DownloadResult,
    DownloadStatus,
    DownloadTask,
    PostProcessAction,
    RetryStrategy,
)
from pygeofetch.models.satellite_data import (
    DataFormat,
    ProcessingLevel,
    ProviderCapabilities,
    QuotaInfo,
    SatelliteAsset,
    SatelliteData,
)
from pygeofetch.models.search_query import BoundingBox, SearchQuery
from pygeofetch.models.user_auth import AuthSession, AuthType, Credentials

__all__ = [
    "SatelliteData",
    "SatelliteAsset",
    "ProcessingLevel",
    "DataFormat",
    "ProviderCapabilities",
    "QuotaInfo",
    "SearchQuery",
    "BoundingBox",
    "DownloadTask",
    "DownloadOptions",
    "DownloadResult",
    "DownloadProgress",
    "DownloadStatus",
    "PostProcessAction",
    "RetryStrategy",
    "ChecksumAlgorithm",
    "AuthSession",
    "Credentials",
    "AuthType",
]
