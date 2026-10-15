"""
Geospatial utility functions for PyGeoFetch.

Provides coordinate transformations, geometry operations, and
spatial calculations used across providers.

Example::

    from pygeofetch.utils.geo_utils import bbox_to_geojson, parse_bbox

    geojson = bbox_to_geojson((-74.1, 40.6, -73.7, 40.9))
    area_km2 = bbox_area_km2((-74.1, 40.6, -73.7, 40.9))
"""

from __future__ import annotations

import math
from typing import Any

BBoxTuple = tuple[float, float, float, float]


def parse_bbox(value: str | list | tuple | dict) -> BBoxTuple:
    """
    Parse a bounding box from various input formats.

    Args:
        value: One of:
            - Comma-separated string: "-74.1,40.6,-73.7,40.9"
            - List or tuple: [-74.1, 40.6, -73.7, 40.9]
            - Dict with min_lon/min_lat/max_lon/max_lat keys
            - Dict with west/south/east/north keys

    Returns:
        (min_lon, min_lat, max_lon, max_lat) tuple.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != 4:
            msg = f"Expected 4 coordinate values, got {len(value)}"
            raise ValueError(msg)
        return tuple(float(v) for v in value)  # type: ignore
    if isinstance(value, str):
        parts = [float(p.strip()) for p in value.split(",")]
        if len(parts) != 4:
            msg = f"Expected 4 comma-separated values, got {len(parts)}"
            raise ValueError(msg)
        return tuple(parts)  # type: ignore
    if isinstance(value, dict):
        if "min_lon" in value:
            return (
                value["min_lon"],
                value["min_lat"],
                value["max_lon"],
                value["max_lat"],
            )
        if "west" in value:
            return (value["west"], value["south"], value["east"], value["north"])
        msg = "Dict must have min_lon/max_lon/min_lat/max_lat or west/south/east/north keys"
        raise ValueError(msg)
    msg = f"Cannot parse bbox from {type(value)}"
    raise TypeError(msg)


def bbox_to_geojson(bbox: BBoxTuple) -> dict[str, Any]:
    """
    Convert a bounding box to a GeoJSON Polygon feature.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) tuple.

    Returns:
        GeoJSON Polygon geometry dict.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]
        ],
    }


def bbox_to_wkt(bbox: BBoxTuple) -> str:
    """
    Convert a bounding box to WKT POLYGON string.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) tuple.

    Returns:
        WKT POLYGON string.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    return (
        f"POLYGON(({min_lon} {min_lat},"
        f"{max_lon} {min_lat},"
        f"{max_lon} {max_lat},"
        f"{min_lon} {max_lat},"
        f"{min_lon} {min_lat}))"
    )


def bbox_area_km2(bbox: BBoxTuple) -> float:
    """
    Approximate area of a bounding box in square kilometres.

    Uses Haversine for width and height. Appropriate for small to medium areas.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) tuple.

    Returns:
        Area in km².
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    width_km = haversine_km(min_lat, min_lon, min_lat, max_lon)
    height_km = haversine_km(min_lat, min_lon, max_lat, min_lon)
    return width_km * height_km


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance in kilometres using the Haversine formula.

    Args:
        lat1, lon1: First point coordinates in decimal degrees.
        lat2, lon2: Second point coordinates in decimal degrees.

    Returns:
        Distance in kilometres.
    """
    R = 6371.0  # Earth radius in km
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def bbox_intersects(bbox1: BBoxTuple, bbox2: BBoxTuple) -> bool:
    """
    Check if two bounding boxes intersect.

    Args:
        bbox1, bbox2: (min_lon, min_lat, max_lon, max_lat) tuples.

    Returns:
        True if the bounding boxes overlap.
    """
    min_lon1, min_lat1, max_lon1, max_lat1 = bbox1
    min_lon2, min_lat2, max_lon2, max_lat2 = bbox2
    return not (
        max_lon1 < min_lon2
        or max_lon2 < min_lon1
        or max_lat1 < min_lat2
        or max_lat2 < min_lat1
    )


def bbox_union(bboxes: list[BBoxTuple]) -> BBoxTuple:
    """
    Compute the union bounding box of a list of bounding boxes.

    Args:
        bboxes: List of (min_lon, min_lat, max_lon, max_lat) tuples.

    Returns:
        Bounding box covering all inputs.
    """
    if not bboxes:
        msg = "Cannot compute union of empty list"
        raise ValueError(msg)
    min_lon = min(b[0] for b in bboxes)
    min_lat = min(b[1] for b in bboxes)
    max_lon = max(b[2] for b in bboxes)
    max_lat = max(b[3] for b in bboxes)
    return (min_lon, min_lat, max_lon, max_lat)


def point_in_bbox(lon: float, lat: float, bbox: BBoxTuple) -> bool:
    """
    Check if a point falls within a bounding box.

    Args:
        lon: Point longitude.
        lat: Point latitude.
        bbox: (min_lon, min_lat, max_lon, max_lat) bounding box.

    Returns:
        True if the point is inside or on the boundary of the bbox.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def geojson_to_bbox(geometry: dict[str, Any]) -> BBoxTuple | None:
    """
    Extract the bounding box of a GeoJSON geometry.

    Args:
        geometry: GeoJSON geometry dict (Point, LineString, Polygon, etc.).

    Returns:
        (min_lon, min_lat, max_lon, max_lat) or None if extraction fails.
    """

    def extract_coords(geom: dict) -> list[tuple[float, float]]:
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])
        if gtype == "Point":
            return [tuple(coords[:2])]  # type: ignore
        if gtype in ("LineString", "MultiPoint"):
            return [tuple(c[:2]) for c in coords]  # type: ignore
        if gtype in ("Polygon", "MultiLineString"):
            return [tuple(c[:2]) for ring in coords for c in ring]  # type: ignore
        if gtype == "MultiPolygon":
            return [tuple(c[:2]) for poly in coords for ring in poly for c in ring]  # type: ignore
        if gtype == "GeometryCollection":
            return [c for g in geom.get("geometries", []) for c in extract_coords(g)]
        return []

    try:
        coords = extract_coords(geometry)
        if not coords:
            return None
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return (min(lons), min(lats), max(lons), max(lats))
    except Exception:
        return None


def format_bbox_string(bbox: BBoxTuple, precision: int = 6) -> str:
    """
    Format a bounding box as a coordinate string.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) tuple.
        precision: Decimal places to round to.

    Returns:
        String like '-74.123456,40.600000,-73.700000,40.900000'.
    """
    return ",".join(f"{v:.{precision}f}" for v in bbox)


def _normalise_satellite_name(platform: str) -> str:
    """
    Normalise STAC platform field to a consistent short name.

    Examples:
        "SENTINEL-1C"   → "S1C"
        "sentinel-1d"   → "S1D"
        "SENTINEL-1A"   → "S1A"
        "SENTINEL-2B"   → "S2B"
    """
    platform = (platform or "").upper().strip().replace(" ", "-")
    mapping = {
        "SENTINEL-1A": "S1A",
        "SENTINEL-1B": "S1B",
        "SENTINEL-1C": "S1C",
        "SENTINEL-1D": "S1D",
        "SENTINEL-2A": "S2A",
        "SENTINEL-2B": "S2B",
        "SENTINEL-2C": "S2C",
        "LANDSAT-8": "L8",
        "LANDSAT-9": "L9",
    }
    return mapping.get(platform, platform)
