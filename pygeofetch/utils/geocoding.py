"""
Geocoding — place name to coordinates, and back, via OpenStreetMap's
free Nominatim service.

Fills a real, verified gap: pygeofetch had no way to turn a place name
into a bbox/coordinates, or a coordinate into a place name -- every
other tool in this package assumes you already have real, numeric
coordinates in hand.

Uses Nominatim (nominatim.openstreetmap.org) specifically because it
is free, requires no API key, and matches pygeofetch's existing
"works without a paid account" providers. Real usage policy requires
a real, identifying User-Agent header and a max of 1 request/second --
both respected here.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org"
_USER_AGENT = "pygeofetch/1.0 (https://github.com/EOCoreINT/pygeofetch)"
_MIN_REQUEST_INTERVAL_S = 1.0  # Nominatim's real, documented usage policy limit

_last_request_time = [0.0]


def _throttle() -> None:
    """Real, honest rate-limit compliance -- Nominatim's usage policy
    caps free requests at 1/second; violating it risks a real, live
    IP ban, not just a slow response."""
    elapsed = time.monotonic() - _last_request_time[0]
    if elapsed < _MIN_REQUEST_INTERVAL_S:
        time.sleep(_MIN_REQUEST_INTERVAL_S - elapsed)
    _last_request_time[0] = time.monotonic()


def geocode(query: str, limit: int = 1, timeout: float = 10.0) -> list[dict[str, Any]]:
    """
    Real, forward geocoding: place name -> coordinates and bounding box.

    Args:
        query:   A real place name, address, or landmark -- e.g.
                 "Nairobi, Kenya", "Eiffel Tower", "1600 Pennsylvania Ave".
        limit:   Max real results to return (Nominatim may return
                 several matches for an ambiguous query).
        timeout: Real HTTP request timeout in seconds.

    Returns:
        A list of dicts, each with real keys: ``lat``, ``lon`` (floats),
        ``bbox`` (min_lon, min_lat, max_lon, max_lat -- ready to pass
        directly to pygeofetch's own BoundingBox/SearchQuery), and
        ``display_name`` (the real, full matched place name, useful
        for confirming the geocoder understood the query correctly).
        Empty list if nothing real matched -- never raises for "not
        found," only for a real, genuine request failure.

    Example::

        from pygeofetch.utils.geocoding import geocode
        results = geocode("Nairobi, Kenya")
        bbox = results[0]["bbox"]  # ready for SearchQuery(bbox=bbox, ...)
    """
    _throttle()
    resp = requests.get(
        f"{NOMINATIM_BASE_URL}/search",
        params={"q": query, "format": "jsonv2", "limit": limit},
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    raw_results = resp.json()

    results = []
    for r in raw_results:
        south, north, west, east = (float(x) for x in r["boundingbox"])
        results.append({
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "bbox": (west, south, east, north),
            "display_name": r["display_name"],
            "place_type": r.get("type"),
        })

    if not results:
        logger.warning("geocode: no real results found for query %r", query)
    return results


def reverse_geocode(lat: float, lon: float, timeout: float = 10.0) -> dict[str, Any] | None:
    """
    Real, reverse geocoding: coordinates -> the real place name at
    that location.

    Args:
        lat, lon: Real WGS84 coordinates.
        timeout:  Real HTTP request timeout in seconds.

    Returns:
        A dict with real keys ``display_name`` and ``address``
        (Nominatim's own structured address breakdown -- country,
        city, road, etc., whichever real fields it has for that
        location), or None if nothing real is found at that point
        (e.g. open ocean).

    Example::

        from pygeofetch.utils.geocoding import reverse_geocode
        place = reverse_geocode(-1.286389, 36.817223)
        print(place["display_name"])  # -> a real Nairobi address
    """
    _throttle()
    resp = requests.get(
        f"{NOMINATIM_BASE_URL}/reverse",
        params={"lat": lat, "lon": lon, "format": "jsonv2"},
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        logger.warning("reverse_geocode: no real place found at (%s, %s)", lat, lon)
        return None

    return {"display_name": data["display_name"], "address": data.get("address", {})}
