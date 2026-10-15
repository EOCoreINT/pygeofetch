"""
Input validation utilities for PyGeoFetch.

Provides validation functions for coordinates, dates, provider names,
and other common input types.
"""

from __future__ import annotations

import re
from datetime import date


def validate_bbox_string(value: str) -> tuple[float, float, float, float]:
    """
    Validate and parse a bounding box string.

    Args:
        value: String in 'min_lon,min_lat,max_lon,max_lat' format.

    Returns:
        Parsed bbox tuple.

    Raises:
        ValueError: If format or values are invalid.
    """
    try:
        parts = [float(p.strip()) for p in value.split(",")]
    except ValueError:
        msg = f"Invalid bbox format; expected 4 numeric values, got: {value!r}"
        raise ValueError(msg)

    if len(parts) != 4:
        msg = f"Bbox must have exactly 4 values, got {len(parts)}: {value!r}"
        raise ValueError(msg)

    min_lon, min_lat, max_lon, max_lat = parts

    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        msg = "Longitude values must be between -180 and 180"
        raise ValueError(msg)
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        msg = "Latitude values must be between -90 and 90"
        raise ValueError(msg)
    if min_lon >= max_lon:
        msg = f"min_lon ({min_lon}) must be less than max_lon ({max_lon})"
        raise ValueError(msg)
    if min_lat >= max_lat:
        msg = f"min_lat ({min_lat}) must be less than max_lat ({max_lat})"
        raise ValueError(msg)

    return (min_lon, min_lat, max_lon, max_lat)


def validate_cloud_cover_string(value: str) -> tuple[float, float]:
    """
    Validate and parse a cloud cover range string.

    Args:
        value: String like '0-20' or '10-80'.

    Returns:
        (min, max) tuple of floats.

    Raises:
        ValueError: If format is invalid.
    """
    match = re.match(r"^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$", value.strip())
    if not match:
        msg = f"Cloud cover must be in 'min-max' format (e.g., '0-20'), got: {value!r}"
        raise ValueError(msg)
    lo, hi = float(match.group(1)), float(match.group(2))
    if not (0 <= lo <= 100 and 0 <= hi <= 100):
        msg = "Cloud cover values must be between 0 and 100"
        raise ValueError(msg)
    if lo > hi:
        msg = f"Cloud cover min ({lo}) must be ≤ max ({hi})"
        raise ValueError(msg)
    return lo, hi


def validate_date_string(value: str) -> date:
    """
    Parse an ISO date string.

    Args:
        value: Date string (YYYY-MM-DD or ISO datetime).

    Returns:
        date object.

    Raises:
        ValueError: If parsing fails.
    """
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        msg = f"Invalid date format; expected YYYY-MM-DD, got: {value!r}"
        raise ValueError(msg)


def validate_provider_name(name: str, valid_providers: list | None = None) -> str:
    """
    Validate a provider name string.

    Args:
        name: Provider name to validate.
        valid_providers: Optional list of acceptable provider names.

    Returns:
        Cleaned provider name.

    Raises:
        ValueError: If name is invalid or not in valid_providers.
    """
    cleaned = name.strip().lower().replace("-", "_")
    if not re.match(r"^[a-z][a-z0-9_]*$", cleaned):
        msg = (
            f"Invalid provider name {name!r}. Must start with a letter and contain "
            "only letters, numbers, and underscores."
        )
        raise ValueError(msg)
    if valid_providers and cleaned not in valid_providers:
        msg = f"Unknown provider {cleaned!r}. Valid providers: {', '.join(sorted(valid_providers))}"
        raise ValueError(msg)
    return cleaned


def validate_url(url: str) -> str:
    """
    Basic URL validation.

    Args:
        url: URL string to validate.

    Returns:
        Validated URL.

    Raises:
        ValueError: If URL is malformed.
    """
    url = url.strip()
    if not re.match(r"^https?://", url):
        msg = f"URL must start with http:// or https://, got: {url!r}"
        raise ValueError(msg)
    return url


def validate_email(email: str) -> str:
    """
    Basic email validation.

    Args:
        email: Email address string.

    Returns:
        Validated email.

    Raises:
        ValueError: If email is malformed.
    """
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    if not re.match(pattern, email.strip()):
        msg = f"Invalid email address: {email!r}"
        raise ValueError(msg)
    return email.strip().lower()
