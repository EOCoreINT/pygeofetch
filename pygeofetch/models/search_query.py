"""
Search query models for PyGeoFetch.

Defines the unified SearchQuery model used across all providers, along with
supporting types for spatial, temporal, and spectral filtering.

Example::

    from pygeofetch.models.search_query import SearchQuery

    query = SearchQuery(
        bbox=(-74.1, 40.6, -73.7, 40.9),
        start_date="2024-01-01",
        end_date="2024-06-01",
        cloud_cover_max=20,
        satellites=["Sentinel-2A", "Sentinel-2B"],
        max_results=50,
    )
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class BoundingBox(BaseModel):
    """
    Axis-aligned bounding box for spatial queries.

    Attributes:
        min_lon: Minimum longitude (west), -180 to 180.
        min_lat: Minimum latitude (south), -90 to 90.
        max_lon: Maximum longitude (east), -180 to 180.
        max_lat: Maximum latitude (north), -90 to 90.
    """

    min_lon: float = Field(..., ge=-180, le=180)
    min_lat: float = Field(..., ge=-90, le=90)
    max_lon: float = Field(..., ge=-180, le=180)
    max_lat: float = Field(..., ge=-90, le=90)

    @model_validator(mode="after")
    def validate_ordering(self) -> BoundingBox:
        if self.min_lon >= self.max_lon:
            msg = f"min_lon ({self.min_lon}) must be less than max_lon ({self.max_lon})"
            raise ValueError(msg)
        if self.min_lat >= self.max_lat:
            msg = f"min_lat ({self.min_lat}) must be less than max_lat ({self.max_lat})"
            raise ValueError(msg)
        return self

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return as (min_lon, min_lat, max_lon, max_lat)."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    def to_wkt(self) -> str:
        """Return as WKT Polygon string."""
        return (
            f"POLYGON(({self.min_lon} {self.min_lat}, "
            f"{self.max_lon} {self.min_lat}, "
            f"{self.max_lon} {self.max_lat}, "
            f"{self.min_lon} {self.max_lat}, "
            f"{self.min_lon} {self.min_lat}))"
        )

    @classmethod
    def from_tuple(cls, t: tuple[float, float, float, float]) -> BoundingBox:
        """Create from (min_lon, min_lat, max_lon, max_lat) tuple."""
        return cls(min_lon=t[0], min_lat=t[1], max_lon=t[2], max_lat=t[3])

    @classmethod
    def from_string(cls, s: str) -> BoundingBox:
        """
        Parse from string like '-74.1,40.6,-73.7,40.9'.

        Args:
            s: Comma-separated coordinate string (min_lon,min_lat,max_lon,max_lat).
        """
        parts = [float(x.strip()) for x in s.split(",")]
        if len(parts) != 4:
            msg = f"Expected 4 comma-separated values, got {len(parts)}"
            raise ValueError(msg)
        return cls.from_tuple(tuple(parts))  # type: ignore


class SearchQuery(BaseModel):
    """
    Unified search query model for all satellite data providers.

    Supports spatial, temporal, spectral, and provider-specific filtering.
    All fields are optional; unset fields are ignored during search.

    Attributes:
        bbox: Bounding box for spatial filtering.
        geometry: GeoJSON geometry dict for polygon filtering.
        start_date: Start of date range (inclusive).
        end_date: End of date range (inclusive).
        cloud_cover_min: Minimum cloud cover percentage.
        cloud_cover_max: Maximum cloud cover percentage.
        satellites: List of satellite platform names to include.
        sensors: List of sensor/instrument names to include.
        collections: List of data collection identifiers.
        processing_levels: Accepted processing level strings.
        resolution_min_m: Minimum spatial resolution in meters.
        resolution_max_m: Maximum spatial resolution in meters.
        max_results: Maximum number of results to return.
        page: Page number for paginated results.
        page_size: Results per page.
        sort_by: Field to sort results by.
        sort_ascending: Sort direction (True=ascending).
        providers: Specific providers to restrict search to.
        provider_filters: Provider-specific additional filter parameters.
        cql2_filter: CQL2 filter expression string.
        ids: Specific item IDs to retrieve.

    Example::

        query = SearchQuery(
            bbox=(-74.1, 40.6, -73.7, 40.9),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 1),
            cloud_cover_max=20,
            satellites=["Sentinel-2A"],
            max_results=50,
        )
    """

    # Spatial filters
    bbox: BoundingBox | None = None
    geometry: dict[str, Any] | None = None

    # Temporal filters
    start_date: date | datetime | None = None
    end_date: date | datetime | None = None

    # Quality filters
    cloud_cover_min: float = Field(default=0, ge=0, le=100)
    cloud_cover_max: float = Field(default=100, ge=0, le=100)

    # Platform / sensor filters
    satellites: list[str] = Field(default_factory=list)
    sensors: list[str] = Field(default_factory=list)
    collections: list[str] = Field(default_factory=list)
    processing_levels: list[str] = Field(default_factory=list)

    # Resolution filters (meters)
    resolution_min_m: float | None = Field(None, gt=0)
    resolution_max_m: float | None = Field(None, gt=0)

    # Pagination
    product_type: str | None = None  # "GRD"|"SLC"|"GRD-COG"
    polarisation: str | None = None
    pass_direction: str | None = None
    max_results: int = Field(default=100, ge=1, le=10000)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=100, ge=1, le=1000)

    # Sorting
    sort_by: str = "datetime"
    sort_ascending: bool = False

    # Provider control
    providers: list[str] = Field(default_factory=list)
    provider_filters: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Advanced filtering
    cql2_filter: str | None = None
    ids: list[str] = Field(default_factory=list)

    # Geometry (alternative to bbox)
    geometry_geojson: dict[str, Any] | None = None  # GeoJSON geometry dict

    # Provider failure handling
    on_provider_failure: str = "skip"  # "skip", "abort", or "retry"

    # Request timeout
    timeout_seconds: int = Field(default=60, ge=1)

    @field_validator("bbox", mode="before")
    @classmethod
    def coerce_bbox(cls, v: Any) -> BoundingBox | None:
        """Accept tuple, list, string, dict, or BoundingBox."""
        if v is None:
            return None
        if isinstance(v, BoundingBox):
            return v
        if isinstance(v, (tuple, list)) and len(v) == 4:
            return BoundingBox.from_tuple(tuple(v))  # type: ignore
        if isinstance(v, str):
            return BoundingBox.from_string(v)
        if isinstance(v, dict):
            return BoundingBox(**v)
        msg = f"Cannot coerce {type(v)} to BoundingBox"
        raise ValueError(msg)

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def coerce_date(cls, v: Any) -> date | datetime | None:
        """Accept string dates in ISO format."""
        if v is None or isinstance(v, (date, datetime)):
            return v
        if isinstance(v, str):
            try:
                return date.fromisoformat(v)
            except ValueError:
                from dateutil.parser import parse

                return parse(v)
        msg = f"Cannot parse date from {type(v)}"
        raise ValueError(msg)

    @model_validator(mode="after")
    def validate_cloud_cover_range(self) -> SearchQuery:
        if self.cloud_cover_min > self.cloud_cover_max:
            msg = (
                f"cloud_cover_min ({self.cloud_cover_min}) must be ≤ "
                f"cloud_cover_max ({self.cloud_cover_max})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_date_range(self) -> SearchQuery:
        if self.start_date and self.end_date:
            sd = (
                self.start_date
                if isinstance(self.start_date, datetime)
                else datetime(
                    self.start_date.year, self.start_date.month, self.start_date.day
                )
            )
            ed = (
                self.end_date
                if isinstance(self.end_date, datetime)
                else datetime(
                    self.end_date.year, self.end_date.month, self.end_date.day
                )
            )
            if sd > ed:
                msg = "start_date must be before or equal to end_date"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_resolution_range(self) -> SearchQuery:
        if self.resolution_min_m and self.resolution_max_m:
            if self.resolution_min_m > self.resolution_max_m:
                msg = "resolution_min_m must be ≤ resolution_max_m"
                raise ValueError(msg)
        return self

    def set_product_type(self, product_type: str) -> SearchQuery:
        """
        Set the SAR product type.

        Args:
            product_type: "GRD" (default) | "SLC" | "GRD-COG" | "OCN"

        GRD  — Ground Range Detected. Intensity only. 800 MB–2 GB.
               Available from all providers. Use for flood/damage mapping.
        SLC  — Single Look Complex. Phase preserved. 4–8 GB.
               Available from Copernicus Dataspace and ASF Vertex.
               Required for InSAR (mm-precision deformation).
        """
        self.product_type = product_type.upper()
        return self

    def to_stac_filter(self) -> dict[str, Any]:
        """
        Convert to STAC API search parameters.

        Returns:
            Dictionary of STAC-compliant search parameters.
        """
        params: dict[str, Any] = {}

        if self.bbox:
            params["bbox"] = list(self.bbox.to_tuple())

        if self.geometry:
            params["intersects"] = self.geometry

        if self.start_date or self.end_date:

            def _to_rfc3339(d) -> str:
                """Format date/datetime as RFC3339 string required by STAC APIs."""
                if hasattr(d, "hour"):  # it's a datetime
                    return d.strftime("%Y-%m-%dT%H:%M:%SZ")
                return f"{d.year:04d}-{d.month:02d}-{d.day:02d}T00:00:00Z"

            start = _to_rfc3339(self.start_date) if self.start_date else ".."
            end = _to_rfc3339(self.end_date) if self.end_date else ".."
            params["datetime"] = f"{start}/{end}"

        if self.collections:
            params["collections"] = self.collections

        if self.ids:
            params["ids"] = self.ids

        params["limit"] = min(self.page_size, self.max_results)

        # Build CQL2-JSON filter (supported by all major STAC APIs including
        # Planetary Computer, Earth Search, Element84, and AWS Earth)
        cql2_args = []

        if self.cloud_cover_max < 100:
            cql2_args.append(
                {
                    "op": "<=",
                    "args": [{"property": "eo:cloud_cover"}, self.cloud_cover_max],
                }
            )
        if self.cloud_cover_min > 0:
            cql2_args.append(
                {
                    "op": ">=",
                    "args": [{"property": "eo:cloud_cover"}, self.cloud_cover_min],
                }
            )

        # Merge in any user-supplied CQL2 filter (accept dict or text string)
        if self.cql2_filter:
            if isinstance(self.cql2_filter, dict):
                cql2_args.append(self.cql2_filter)
            else:
                # User supplied CQL2-text — keep as text filter alongside JSON
                if cql2_args:
                    # Send JSON filter for cloud cover; text filter unsupported alongside JSON
                    # Convert text to a passthrough arg (best effort)
                    pass
                else:
                    params["filter"] = self.cql2_filter
                    params["filter-lang"] = "cql2-text"

        if cql2_args:
            if len(cql2_args) == 1:
                params["filter"] = cql2_args[0]
            else:
                params["filter"] = {"op": "and", "args": cql2_args}
            params["filter-lang"] = "cql2-json"

        return params

    @property
    def has_spatial_filter(self) -> bool:
        """Return True if any spatial constraint is set."""
        return self.bbox is not None or self.geometry is not None

    @property
    def has_temporal_filter(self) -> bool:
        """Return True if any date constraint is set."""
        return self.start_date is not None or self.end_date is not None

    def copy_for_provider(self, provider: str) -> SearchQuery:
        """
        Return a copy of this query with provider-specific filters merged.

        Args:
            provider: Provider name to get specific filters for.

        Returns:
            SearchQuery with provider-specific filters applied.
        """
        data = self.model_dump()
        provider_overrides = self.provider_filters.get(provider, {})
        data.update(provider_overrides)
        return SearchQuery(**data)
