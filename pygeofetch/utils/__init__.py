"""Utility functions for PyGeoFetch."""

from pygeofetch.utils.file_utils import (
    compute_checksum,
    ensure_directory,
    human_readable_size,
    safe_extract,
    verify_checksum,
)
from pygeofetch.utils.geo_utils import (
    bbox_area_km2,
    bbox_intersects,
    bbox_to_geojson,
    bbox_to_wkt,
    haversine_km,
    parse_bbox,
)
from pygeofetch.utils.logging_setup import get_logger, setup_logging
from pygeofetch.utils.retry_handler import CircuitBreaker, RetryConfig, retry_on_failure
from pygeofetch.utils.validators import (
    validate_bbox_string,
    validate_cloud_cover_string,
    validate_date_string,
    validate_provider_name,
    validate_url,
)

__all__ = [
    "compute_checksum",
    "verify_checksum",
    "safe_extract",
    "ensure_directory",
    "human_readable_size",
    "parse_bbox",
    "bbox_to_geojson",
    "bbox_to_wkt",
    "bbox_area_km2",
    "bbox_intersects",
    "haversine_km",
    "get_logger",
    "setup_logging",
    "retry_on_failure",
    "RetryConfig",
    "CircuitBreaker",
    "validate_bbox_string",
    "validate_cloud_cover_string",
    "validate_date_string",
    "validate_provider_name",
    "validate_url",
]
from pygeofetch.utils.gcp_recovery import (  # noqa: F401
    needs_recovery,
    recover_georeference,
    validate_georeference,
)
