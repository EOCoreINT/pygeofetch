"""
Satellite data models for PyGeoFetch.

This module defines the core data structures used throughout the package
to represent satellite imagery metadata, assets, and collection information.
All models are STAC 1.0 compliant.

Example::

    from pygeofetch.models.satellite_data import SatelliteData, SatelliteAsset

    data = SatelliteData(
        id="LC08_L2SP_013032_20240115",
        provider="usgs",
        satellite="Landsat-8",
        sensor="OLI-TIRS",
        datetime=datetime(2024, 1, 15),
        bbox=(-75.0, 40.0, -74.0, 41.0),
        cloud_cover=5.2,
        processing_level="L2SP",
    )
"""

from __future__ import annotations

from datetime import datetime as DateTime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ProcessingLevel(str, Enum):
    """Standard satellite data processing levels."""

    RAW = "RAW"
    L0 = "L0"
    L1 = "L1"
    L1A = "L1A"
    L1B = "L1B"
    L1C = "L1C"
    L1T = "L1T"
    L2 = "L2"
    L2A = "L2A"
    L2SP = "L2SP"
    L3 = "L3"
    L4 = "L4"
    ANALYSIS_READY = "ARD"
    UNKNOWN = "UNKNOWN"


class DataFormat(str, Enum):
    """Supported satellite data file formats."""

    GEOTIFF = "GeoTIFF"
    COG = "COG"  # Cloud Optimized GeoTIFF
    NETCDF = "NetCDF"
    HDF4 = "HDF4"
    HDF5 = "HDF5"
    ZARR = "Zarr"
    JP2 = "JPEG2000"
    PNG = "PNG"
    SAFE = "SAFE"  # ESA Sentinel format
    ZIP = "ZIP"
    TAR = "TAR"
    UNKNOWN = "UNKNOWN"


class SatelliteAsset(BaseModel):
    """
    Represents a single downloadable asset within a satellite data record.

    Attributes:
        key: Unique identifier for this asset (e.g., 'B4', 'thumbnail', 'metadata').
        href: URL or path to the asset.
        title: Human-readable title.
        media_type: MIME type of the asset.
        roles: STAC asset roles (e.g., ['data', 'overview', 'thumbnail']).
        size_bytes: File size in bytes if known.
        checksum_md5: MD5 checksum for integrity verification.
        checksum_sha256: SHA-256 checksum for integrity verification.
        extra_fields: Additional provider-specific metadata.
    """

    key: str
    href: str
    title: str | None = None
    media_type: str | None = None
    roles: list[str] = Field(default_factory=list)
    size_bytes: int | None = None
    checksum_md5: str | None = None
    checksum_sha256: str | None = None
    extra_fields: dict[str, Any] = Field(default_factory=dict)

    @property
    def size_mb(self) -> float | None:
        """Return size in megabytes."""
        return self.size_bytes / (1024 * 1024) if self.size_bytes else None

    def is_data_asset(self) -> bool:
        """
        Return True if this asset is a downloadable raster data file.

        The previous implementation required "data" in roles, which wrongly
        excluded valid bands from providers that use different role conventions:
          - AWS Earth:          roles=[]            (no roles at all)
          - Some providers:     roles=["eo:band"]
          - Planetary Computer: roles=["data"]  (correct)
          - Element84:          roles=["data","reflectance"]

        New logic: EXCLUDE known non-data assets (thumbnails, metadata, overviews),
        and EXCLUDE non-raster media types. Accept everything else that has an href.
        """
        if not self.href:
            return False

        # Exclude by explicit non-data roles
        NON_DATA_ROLES = {
            "thumbnail",
            "overview",
            "metadata",
            "tiles",
            "tilejson",
            "visual",
            "rendered_preview",
            "index",
        }
        if self.roles and NON_DATA_ROLES.issuperset(set(self.roles)):
            # All roles are non-data — skip
            return False

        # Exclude by known non-raster media types
        NON_RASTER_TYPES = (
            "application/json",
            "application/xml",
            "application/geo+json",
            "text/",
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/gif",
        )
        if self.media_type and any(
            self.media_type.startswith(t) for t in NON_RASTER_TYPES
        ):
            return False

        # Exclude by key name patterns that are never rasters
        NON_DATA_KEY_PATTERNS = (
            "thumbnail",
            "preview",
            "tilejson",
            "rendered",
            "visual",
            ".json",
            "readme",
        )
        key_lower = (self.key or "").lower()
        if any(p in key_lower for p in NON_DATA_KEY_PATTERNS):
            return False

        # If href is a non-raster URL extension, skip
        href_lower = (self.href or "").lower().split("?")[0]
        if href_lower.endswith(
            (".jpg", ".jpeg", ".png", ".json", ".xml", ".html", ".txt")
        ):
            return False

        # Accept: has "data" role, or has raster media type, or empty/unknown roles
        RASTER_TYPES = (
            "image/tiff",
            "image/geotiff",
            "image/vnd.stac.geotiff",
            "application/x-tiff",
            "application/x-geotiff",
        )
        if self.media_type and any(self.media_type.startswith(t) for t in RASTER_TYPES):
            return True

        # Accept if has "data" or "eo:band" role
        if "data" in self.roles or "eo:band" in self.roles:
            return True

        # Accept if roles is empty (provider didn't set roles) AND href ends in raster extension
        raster_exts = (
            ".tif",
            ".tiff",
            ".img",
            ".jp2",
            ".nc",
            ".hdf",
            ".h5",
            ".zip",
            ".tar",
        )
        if href_lower.endswith(raster_exts):
            return True

        # If roles is completely empty and media type unknown — include (provider may be wrong)
        if not self.roles and not self.media_type:
            return True

        return False


def _processing_level_from_stac(
    collection: str | None, properties: dict[str, Any]
) -> "ProcessingLevel":
    """
    Determine a real ProcessingLevel from a STAC item's collection id
    and/or properties, since real Sentinel-2/Landsat STAC catalogs
    (Earth Search v1, AWS Earth, Planetary Computer) don't reliably
    populate a "processing:level" properties key -- the collection id
    itself (e.g. "sentinel-2-l2a", "sentinel-2-c1-l2a", "landsat-c2-l2")
    is the actually-reliable real-world signal.
    """
    candidates = [
        properties.get("processing:level"),
        properties.get("processingLevel"),
        properties.get("processing_level"),
        collection,
    ]
    text = " ".join(str(c) for c in candidates if c).upper()

    # Order matters: check the more specific tokens first so e.g.
    # "L2A" isn't shadowed by a looser "L2" match on the same string.
    ordered_checks = [
        (ProcessingLevel.L2A, ("L2A", "MSIL2A")),
        (ProcessingLevel.L2SP, ("L2SP",)),
        (ProcessingLevel.L1C, ("L1C", "MSIL1C")),
        (ProcessingLevel.L1T, ("L1T",)),
        (ProcessingLevel.L1B, ("L1B",)),
        (ProcessingLevel.L1A, ("L1A",)),
        (ProcessingLevel.L2, ("L2",)),
        (ProcessingLevel.L1, ("L1",)),
        (ProcessingLevel.L0, ("L0",)),
        (ProcessingLevel.RAW, ("RAW",)),
    ]
    for level, tokens in ordered_checks:
        if any(token in text for token in tokens):
            return level
    return ProcessingLevel.UNKNOWN


class SatelliteData(BaseModel):
    """
    Core model representing a satellite data product/scene.

    STAC 1.0 compliant. Represents a single acquisition or derived product
    from any supported satellite data provider.

    Attributes:
        id: Unique identifier for this scene/product.
        provider: Provider name (e.g., 'usgs', 'copernicus').
        collection: Collection or dataset name.
        satellite: Satellite platform name (e.g., 'Landsat-8', 'Sentinel-2A').
        sensor: Sensor instrument (e.g., 'OLI', 'MSI').
        datetime: Primary acquisition datetime (UTC).
        start_datetime: Start of acquisition window.
        end_datetime: End of acquisition window.
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat).
        geometry: GeoJSON geometry dict.
        cloud_cover: Cloud cover percentage (0-100).
        processing_level: Data processing level.
        data_format: Primary file format.
        assets: Dictionary of downloadable assets.
        stac_extensions: STAC extension URLs used.
        properties: Additional STAC-compliant properties.
        links: STAC links (self, root, collection, etc.).
        score: Relevance/quality score for search result ranking (0-1).

    Example::

        data = SatelliteData(
            id="S2A_MSIL2A_20240115T154911_N0510_R054_T18TWL",
            provider="copernicus",
            collection="SENTINEL-2",
            satellite="Sentinel-2A",
            sensor="MSI",
            datetime=datetime(2024, 1, 15, 15, 49, 11),
            bbox=(-74.1, 40.6, -73.7, 40.9),
            cloud_cover=3.5,
            processing_level=ProcessingLevel.L2A,
        )
    """

    id: str
    provider: str
    collection: str | None = None
    satellite: str | None = None
    sensor: str | None = None
    datetime: DateTime | None = None
    start_datetime: DateTime | None = None
    end_datetime: DateTime | None = None
    bbox: tuple[float, float, float, float] | None = None
    geometry: dict[str, Any] | None = None
    cloud_cover: float | None = Field(None, ge=0, le=100)
    processing_level: ProcessingLevel = ProcessingLevel.UNKNOWN
    data_format: DataFormat = DataFormat.UNKNOWN
    assets: dict[str, SatelliteAsset] = Field(default_factory=dict)
    stac_extensions: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    links: list[dict[str, str]] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    # SAR-specific fields
    product_type: str | None = None  # e.g. "GRD", "SLC", "RTC"
    polarisation: str | None = None  # e.g. "VV", "VH", "VV+VH", "HH"
    pass_direction: str | None = None  # "ascending" | "descending"
    relative_orbit: int | None = None  # Relative orbit number
    orbit_number: int | None = None  # Absolute orbit number
    incidence_angle: float | None = None  # Centre incidence angle (degrees)
    resolution_m: float | None = None  # Best resolution in metres
    gsd_m: float | None = None  # Ground sample distance in metres
    off_nadir_angle: float | None = None  # Off-nadir angle for VHR

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: tuple | None) -> tuple | None:
        """Validate bounding box coordinates."""
        if v is None:
            return v
        min_lon, min_lat, max_lon, max_lat = v
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            msg = f"Longitude values must be between -180 and 180, got {min_lon}, {max_lon}"
            raise ValueError(msg)
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            msg = (
                f"Latitude values must be between -90 and 90, got {min_lat}, {max_lat}"
            )
            raise ValueError(msg)
        if min_lon >= max_lon:
            msg = f"min_lon ({min_lon}) must be less than max_lon ({max_lon})"
            raise ValueError(msg)
        if min_lat >= max_lat:
            msg = f"min_lat ({min_lat}) must be less than max_lat ({max_lat})"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_datetime_range(self) -> SatelliteData:
        """Validate that start_datetime is before end_datetime."""
        if self.start_datetime and self.end_datetime:
            if self.start_datetime >= self.end_datetime:
                msg = "start_datetime must be before end_datetime"
                raise ValueError(msg)
        return self

    @property
    def total_size_bytes(self) -> int:
        """Total size of all assets in bytes."""
        return sum(a.size_bytes or 0 for a in self.assets.values())

    @property
    def total_size_mb(self) -> float:
        """Total size of all data assets in megabytes."""
        return self.total_size_bytes / (1024 * 1024)

    @property
    def data_assets(self) -> dict[str, SatelliteAsset]:
        """Return only primary data assets (not thumbnails/metadata)."""
        return {k: v for k, v in self.assets.items() if v.is_data_asset()}

    def to_stac_item(self) -> dict[str, Any]:
        """
        Export as a STAC 1.0 Item dictionary.

        Returns:
            STAC-compliant dictionary ready for serialization.
        """
        item: dict[str, Any] = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "stac_extensions": self.stac_extensions,
            "id": self.id,
            "geometry": self.geometry or self._bbox_to_geometry(),
            "bbox": list(self.bbox) if self.bbox else None,
            "properties": {
                **self.properties,
                "datetime": self.datetime.isoformat() if self.datetime else None,
                "platform": self.satellite,
                "instruments": [self.sensor] if self.sensor else [],
                "eo:cloud_cover": self.cloud_cover,
                "processing:level": self.processing_level.value,
                "providers": [{"name": self.provider, "roles": ["producer"]}],
            },
            "assets": {
                k: {
                    "href": v.href,
                    "title": v.title,
                    "type": v.media_type,
                    "roles": v.roles,
                    **({"size": v.size_bytes} if v.size_bytes else {}),
                }
                for k, v in self.assets.items()
            },
            "links": self.links,
        }
        if self.collection:
            item["collection"] = self.collection
        return item

    def _bbox_to_geometry(self) -> dict[str, Any] | None:
        """Convert bbox to GeoJSON Polygon geometry."""
        if not self.bbox:
            return None
        min_lon, min_lat, max_lon, max_lat = self.bbox
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

    @classmethod
    def from_stac_item(cls, item: dict[str, Any], provider: str) -> SatelliteData:
        """
        Create a SatelliteData instance from a STAC Item dictionary.

        Args:
            item: STAC-compliant item dictionary.
            provider: Provider name string.

        Returns:
            SatelliteData instance.
        """
        props = item.get("properties", {})
        assets_raw = item.get("assets", {})
        assets = {}
        for key, asset_data in assets_raw.items():
            assets[key] = SatelliteAsset(
                key=key,
                href=asset_data.get("href", ""),
                title=asset_data.get("title"),
                media_type=asset_data.get("type"),
                roles=[asset_data["roles"]]
                if isinstance(asset_data.get("roles"), str)
                else list(asset_data.get("roles") or []),
                size_bytes=asset_data.get("size"),
            )

        bbox_raw = item.get("bbox")
        bbox = tuple(bbox_raw) if bbox_raw and len(bbox_raw) == 4 else None

        dt_str = props.get("datetime")
        dt = None
        if dt_str and dt_str != "null":
            from dateutil.parser import parse

            dt = parse(dt_str)

        # Extract SAR / resolution / geometry fields from properties
        polarisation_raw = (
            props.get("sar:polarizations")
            or props.get("polarisation")
            or props.get("polarisationMode")
        )
        polarisation_str = None
        if isinstance(polarisation_raw, list):
            polarisation_str = "+".join(polarisation_raw)
        elif isinstance(polarisation_raw, str):
            polarisation_str = polarisation_raw

        incidence_raw = props.get("sar:center_frequency") or props.get("incidenceAngle")
        try:
            incidence = float(incidence_raw) if incidence_raw is not None else None
        except (TypeError, ValueError):
            incidence = None

        gsd_raw = (
            props.get("gsd")
            or props.get("resolution")
            or props.get("spatial_resolution")
        )
        try:
            gsd = float(gsd_raw) if gsd_raw is not None else None
        except (TypeError, ValueError):
            gsd = None

        return cls(
            id=item["id"],
            provider=provider,
            collection=item.get("collection"),
            satellite=props.get("platform"),
            sensor=(props.get("instruments") or [None])[0],
            datetime=dt,
            bbox=bbox,
            geometry=item.get("geometry"),
            cloud_cover=props.get("eo:cloud_cover"),
            # REAL BUG FIXED: processing_level was never set here at all,
            # for any STAC-based provider (aws_earth, element84,
            # planetary_computer, sentinel_hub) -- it always defaulted to
            # ProcessingLevel.UNKNOWN. Real Sentinel-2/Landsat STAC items
            # don't reliably expose a "processing:level" properties key;
            # the reliable signal is the collection id itself (e.g.
            # "sentinel-2-l2a", "sentinel-2-c1-l2a", "landsat-c2-l2").
            processing_level=_processing_level_from_stac(
                collection=item.get("collection"), properties=props
            ),
            properties=props,
            assets=assets,
            stac_extensions=item.get("stac_extensions", []),
            links=item.get("links", []),
            product_type=props.get("sar:product_type") or props.get("productType"),
            polarisation=polarisation_str,
            pass_direction=(
                props.get("sat:orbit_state")
                or props.get("orbitDirection")
                or props.get("flightDirection", "")
            ).lower()
            or None,
            relative_orbit=props.get("sat:relative_orbit")
            or props.get("relativeOrbitNumber"),
            orbit_number=props.get("sat:absolute_orbit") or props.get("orbitNumber"),
            incidence_angle=incidence,
            gsd_m=gsd,
            resolution_m=gsd,
        )




class ProviderCapabilities(BaseModel):
    """
    Describes what a provider supports.

    Attributes:
        search: Whether the provider supports search/discovery.
        download: Whether files can be downloaded.
        streaming: Whether streaming access is supported.
        stac: Whether the provider exposes a STAC API.
        max_results: Maximum search results returnable.
        supported_satellites: List of satellite platforms available.
        supported_formats: List of data formats available.
        requires_auth: Whether authentication is required.
        has_quota: Whether the provider enforces download quotas.
        supports_aoi_filter: Whether spatial filtering is supported.
        supports_cloud_filter: Whether cloud cover filtering is supported.
        supports_date_filter: Whether date range filtering is supported.
        extra_info: Additional provider-specific capability metadata.
    """

    provider_id: str = ""
    name: str = ""
    description: str = ""
    auth_type: str = "username_password"
    satellites: list[str] = Field(default_factory=list)
    # Capability flags
    search: bool = True
    download: bool = True
    streaming: bool = False
    stac: bool = False
    supports_sar: bool = False
    supports_sub_meter: bool = False  # <1m resolution imagery
    supports_tasking: bool = False  # New satellite tasking orders
    supports_direct_s3: bool = False  # Direct cloud storage access
    supports_cql2: bool = False  # CQL2 filter expressions
    # Filters
    supports_aoi_filter: bool = True
    supports_cloud_filter: bool = False
    supports_date_filter: bool = True
    supports_resolution_filter: bool = False
    supports_processing_level_filter: bool = False
    # Metadata
    max_results: int | None = None
    supported_satellites: list[str] = Field(default_factory=list)
    supported_formats: list[DataFormat] = Field(default_factory=list)
    requires_auth: bool = True
    has_quota: bool = False
    regions: list[str] = Field(default_factory=list)  # e.g. ["global", "europe"]
    resolution_min_m: float | None = None  # Best resolution in metres
    resolution_max_m: float | None = None  # Coarsest resolution in metres
    endpoint_url: str = ""
    docs_url: str = ""
    extra_info: dict[str, Any] = Field(default_factory=dict)


class QuotaInfo(BaseModel):
    """
    Represents a provider's quota/usage information.

    Attributes:
        provider: Provider name.
        total_bytes: Total allowed download quota in bytes.
        used_bytes: Bytes already consumed.
        remaining_bytes: Bytes remaining.
        reset_datetime: When the quota resets.
        requests_per_minute: API rate limit.
        requests_used_today: API calls made today.
        extra_info: Additional quota details.
    """

    provider: str
    total_bytes: int | None = None
    used_bytes: int | None = None
    remaining_bytes: int | None = None
    reset_datetime: DateTime | None = None
    requests_per_minute: int | None = None
    requests_used_today: int | None = None
    extra_info: dict[str, Any] = Field(default_factory=dict)

    @property
    def usage_percent(self) -> float | None:
        """Return percentage of quota used (0-100)."""
        if self.total_bytes and self.used_bytes:
            return (self.used_bytes / self.total_bytes) * 100
        return None


# ── Band name alias table ─────────────────────────────────────────────────────
# Maps user-facing Sentinel-2 / Landsat band names to provider asset key variants
_BAND_ALIASES: dict = {
    # ── Sentinel-2 numeric → common names ──────────────────────────────────
    "B01": {"coastal", "coastal_aerosol", "b01"},
    "B02": {"blue", "b02", "blue_band", "b2"},
    "B03": {"green", "b03", "green_band", "b3"},
    "B04": {"red", "b04", "red_band", "b4"},
    "B05": {"rededge", "rededge1", "red_edge_1", "b05", "vegetation_red_edge"},
    "B06": {"rededge2", "red_edge_2", "b06"},
    "B07": {"rededge3", "red_edge_3", "b07"},
    "B08": {"nir", "b08", "nir_band", "near_infrared", "b5"},
    "B8A": {"nir08", "nir08a", "nir_narrow", "b8a", "b08a"},
    "B09": {"nir09", "water_vapour", "b09"},
    "B11": {"swir", "swir1", "swir16", "b11", "shortwave_infrared_1", "b6"},
    "B12": {"swir2", "swir22", "b12", "shortwave_infrared_2", "b7"},
    "SCL": {"scl", "scene_classification", "scene_classification_map"},
    "TCI": {"tci", "true_color_image", "visual"},
    # ── Landsat-specific (non-conflicting) ──────────────────────────────────
    "B1": {"coastal_l8", "b1"},  # Landsat Coastal/Aerosol
    "B8": {"pan", "panchromatic", "b8"},  # Landsat Panchromatic
    "B9": {"cirrus_l8", "b9"},  # Landsat Cirrus
    "B10": {"tirs1", "thermal", "b10", "swir16", "cirrus"},  # Landsat TIRS-1
    # ── Common aliases without numeric prefix ─────────────────────────────
    # These work for both Landsat and Sentinel via the alias lookup
}

# Reverse lookup: alias -> canonical band name
_ALIAS_TO_CANONICAL: dict = {}
for _canonical, _aliases in _BAND_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias.upper()] = _canonical
    _ALIAS_TO_CANONICAL[_canonical.upper()] = _canonical


def _zero_pad_band(band: str) -> str:
    """B4 -> B04, B8 -> B08 (Sentinel-2 style zero-padding)."""
    import re

    m = re.match(r"^([A-Za-z]+)(\d)([A-Za-z]*)$", band)
    if m:
        return f"{m.group(1)}0{m.group(2)}{m.group(3)}"
    return band


def _strip_band_zero(band: str) -> str:
    """B04 -> B4, B08 -> B8 (Landsat style single digit)."""
    import re

    m = re.match(r"^([A-Za-z]+)0(\d)([A-Za-z]*)$", band)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    return band


def resolve_band_keys(requested_bands: list, available_keys: list) -> list:
    """
    Match user-requested band names against available STAC asset keys.

    Handles the common mismatch where users request "B02,B03,B04" but
    the provider stores assets as "blue", "green", "red".

    Args:
        requested_bands: List of band names from DownloadOptions (e.g. ["B02","B03","B04"])
        available_keys:  Asset keys from the SatelliteData (e.g. ["blue","green","red","nir"])

    Returns:
        List of matching available_keys. Falls back to all keys if nothing matches.
    """
    if not requested_bands:
        return available_keys

    matched = []
    # Build a lookup from available keys normalised -> original key
    avail_upper = {k.upper(): k for k in available_keys}

    for band in requested_bands:
        band_up = band.upper()
        # 1. Direct match (case-insensitive)
        if band_up in avail_upper:
            matched.append(avail_upper[band_up])
            continue
        # 2. Canonical form match (try exact, zero-padded, and stripped)
        canonical = _ALIAS_TO_CANONICAL.get(band_up)
        if canonical:
            # Try exact canonical
            if canonical.upper() in avail_upper:
                matched.append(avail_upper[canonical.upper()])
                continue
            # Try zero-padded variant: B4 -> B04, B8 -> B08
            padded = _zero_pad_band(canonical)
            if padded.upper() in avail_upper:
                matched.append(avail_upper[padded.upper()])
                continue
            # Try stripped variant: B04 -> B4
            stripped = _strip_band_zero(canonical)
            if stripped.upper() in avail_upper:
                matched.append(avail_upper[stripped.upper()])
                continue
        # 3. Look up all aliases for this band and try each
        aliases = _BAND_ALIASES.get(canonical or band_up, set())
        found = False
        for alias in aliases:
            if alias.upper() in avail_upper:
                matched.append(avail_upper[alias.upper()])
                found = True
                break
        if found:
            continue
        # 4. Try reverse: the requested band IS an alias, find its canonical, then check aliases
        for avail_up, avail_orig in avail_upper.items():
            avail_canonical = _ALIAS_TO_CANONICAL.get(avail_up)
            if avail_canonical and avail_canonical.upper() == (canonical or band_up):
                matched.append(avail_orig)
                break

    # Deduplicate, preserving order
    seen = set()
    result = []
    for k in matched:
        if k not in seen:
            seen.add(k)
            result.append(k)

    if not result:
        import logging

        logging.getLogger("pygeofetch.satellite_data").warning(
            "Requested bands %s not found in available keys %s — downloading all assets",
            requested_bands,
            available_keys,
        )
        return available_keys

    return result


# Resolve forward references for pydantic v2
SatelliteData.model_rebuild()