"""
Planet Labs provider for PyGeoFetch.

Full integration with the Planet Data API v1. Supports search and
activation/download for PlanetScope, SkySat, RapidEye, and Landsat 8/9
scenes. Uses API Key authentication (Basic Auth with key as username).

Example::

    from pygeofetch.providers.planet import PlanetProvider
    from pygeofetch.models.user_auth import Credentials

    provider = PlanetProvider()
    provider.authenticate(Credentials(provider="planet", api_key="YOUR_KEY"))
    results = provider.search(SearchQuery(
        bbox=(-74.1, 40.6, -73.7, 40.9),
        start_date="2024-01-01",
        cloud_cover_max=10,
    ))
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pygeofetch.models.download_task import (
    DownloadOptions,
    DownloadResult,
    DownloadStatus,
)
from pygeofetch.models.satellite_data import (
    DataFormat,
    ProviderCapabilities,
    QuotaInfo,
    SatelliteAsset,
    SatelliteData,
)
from pygeofetch.models.user_auth import AuthSession, Credentials
from pygeofetch.providers.base import (
    AbstractBaseProvider,
    AuthenticationError,
    SearchError,
)

if TYPE_CHECKING:
    from pygeofetch.models.search_query import SearchQuery


def _plain(v) -> str:
    """Extract plain string from str or SecretStr."""
    if v is None:
        return ""
    if hasattr(v, "get_secret_value"):
        return v.get_secret_value()
    return str(v)


class PlanetProvider(AbstractBaseProvider):
    """
    Planet Labs Data API v1 provider.

    Searches and downloads PlanetScope, SkySat, and RapidEye imagery.
    Requires a Planet API key.

    Attributes:
        PROVIDER_ID: 'planet'
        REQUIRES_AUTH: True
    """

    PROVIDER_ID = "planet"
    DISPLAY_NAME = "Planet Labs"
    REQUIRES_AUTH = True
    DESCRIPTION = (
        "Daily sub-meter resolution imagery from PlanetScope, SkySat, "
        "and RapidEye satellites. Requires Planet subscription."
    )
    DATA_TYPES = ["PlanetScope", "SkySat", "RapidEye", "Landsat"]
    SATELLITES = ["PlanetScope", "SkySat", "RapidEye", "Landsat-8"]
    BASE_URL = "https://api.planet.com/data/v1"

    ITEM_TYPES: dict[str, str] = {
        "planetscope": "PSScene",
        "ps": "PSScene",
        "skysat": "SkySatCollect",
        "skysatscene": "SkySatScene",
        "rapideye": "REOrthoTile",
        "re": "REOrthoTile",
        "landsat8": "Landsat8L1G",
        "landsat9": "Landsat9L1G",
    }
    DEFAULT_ITEM_TYPES = ["PSScene", "SkySatCollect"]

    def authenticate(self, credentials: Credentials) -> AuthSession:
        """
        Authenticate with Planet API using an API key.

        Args:
            credentials: Must include api_key.

        Returns:
            AuthSession.

        Raises:
            AuthenticationError: If the API key is invalid.
        """
        import httpx

        api_key = credentials.api_key or credentials.password
        if not api_key:
            msg = "Planet Labs requires an API key. Get yours at: https://www.planet.com/account"
            raise AuthenticationError(msg)

        # Validate the key by hitting the /auth/v1/experimental/public endpoint
        try:
            resp = httpx.get(
                f"{self.BASE_URL}/item-types",
                auth=(api_key, ""),
                timeout=15,
            )
            if resp.status_code == 401:
                msg = "Invalid Planet API key."
                raise AuthenticationError(msg)
            if resp.status_code not in (200, 201):
                msg = f"Planet auth check failed: HTTP {resp.status_code}"
                raise AuthenticationError(msg)
        except AuthenticationError:
            raise
        except Exception as exc:
            msg = f"Planet auth error: {exc}"
            raise AuthenticationError(msg) from exc

        from datetime import datetime, timedelta, timezone

        session = AuthSession(
            provider=self.PROVIDER_ID,
            access_token=_plain(api_key),
            expires_at=datetime.now(timezone.utc) + timedelta(days=365),
            session_data={"api_key": _plain(api_key)},
        )
        self._session = session
        self._logger.info("Planet Labs: authenticated successfully")
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        return bool(credentials.api_key or credentials.password)

    def set_session(self, session: Any) -> None:
        """Store an authenticated session for use in requests."""
        self._session = session

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        """
        Search Planet Data API using the Quick Search endpoint.

        Args:
            query: Search parameters.

        Returns:
            List of SatelliteData records.
        """
        self.require_auth()
        import httpx

        item_types = self._resolve_item_types(query)
        filter_body = self._build_filter(query)

        payload = {
            "item_types": item_types,
            "filter": filter_body,
            "limit": min(query.max_results, 250),
        }

        api_key = (
            (self._session.session_data if self._session else {}).get("api_key", "")
            if (self._session.session_data if self._session else {})
            else ""
        )
        try:
            resp = httpx.post(
                f"{self.BASE_URL}/quick-search",
                json=payload,
                auth=(api_key, ""),
                timeout=self.config.get("timeout", 60),
            )
            if resp.status_code != 200:
                self._handle_http_error(resp)

            features = resp.json().get("features", [])
            results = [self._parse_feature(f) for f in features]
            self._logger.info(f"Planet Labs: {len(results)} scenes found")
            return results

        except Exception as exc:
            msg = f"Planet search failed: {exc}"
            raise SearchError(msg) from exc

    def _resolve_item_types(self, query: SearchQuery) -> list[str]:
        if not query.satellites:
            return self.DEFAULT_ITEM_TYPES
        types = []
        for sat in query.satellites:
            key = sat.lower().replace(" ", "").replace("-", "")
            types.append(self.ITEM_TYPES.get(key, sat))
        return types or self.DEFAULT_ITEM_TYPES

    def _build_filter(self, query: SearchQuery) -> dict[str, Any]:
        """Build a Planet AndFilter from query parameters."""
        filters = []

        if query.bbox:
            bb = query.bbox
            filters.append(
                {
                    "type": "GeometryFilter",
                    "field_name": "geometry",
                    "config": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [bb.min_lon, bb.min_lat],
                                [bb.max_lon, bb.min_lat],
                                [bb.max_lon, bb.max_lat],
                                [bb.min_lon, bb.max_lat],
                                [bb.min_lon, bb.min_lat],
                            ]
                        ],
                    },
                }
            )

        if query.start_date or query.end_date:
            date_config: dict[str, str] = {}
            if query.start_date:
                date_config["gte"] = f"{query.start_date}T00:00:00Z"
            if query.end_date:
                date_config["lte"] = f"{query.end_date}T23:59:59Z"
            filters.append(
                {
                    "type": "DateRangeFilter",
                    "field_name": "acquired",
                    "config": date_config,
                }
            )

        if query.cloud_cover_max is not None:
            filters.append(
                {
                    "type": "RangeFilter",
                    "field_name": "cloud_cover",
                    "config": {
                        "gte": (getattr(query, "cloud_cover_min", 0) or 0) / 100.0,
                        "lte": query.cloud_cover_max / 100.0,
                    },
                }
            )

        if len(filters) == 1:
            return filters[0]
        return (
            {"type": "AndFilter", "config": filters}
            if filters
            else {"type": "AndFilter", "config": []}
        )

    def _parse_feature(self, feature: dict[str, Any]) -> SatelliteData:
        """Parse a Planet API GeoJSON feature into SatelliteData."""
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})

        # Extract bbox from geometry
        bbox = None
        coords = geom.get("coordinates", [[]])
        if coords and coords[0]:
            flat = [pt for ring in coords for pt in ring]
            lons = [pt[0] for pt in flat]
            lats = [pt[1] for pt in flat]
            bbox = (min(lons), min(lats), max(lons), max(lats))

        # Build assets
        assets: dict[str, SatelliteAsset] = {}
        for name, link in feature.get("_links", {}).get("assets", {}).items():
            assets[name] = SatelliteAsset(key=name, href=link, roles=["data"])

        return SatelliteData(
            id=feature.get("id", ""),
            provider=self.PROVIDER_ID,
            satellite=props.get("item_type", "PlanetScope"),
            sensor=props.get("instrument"),
            cloud_cover=props.get("cloud_cover", 0) * 100,
            bbox=bbox,
            geometry=geom if geom.get("coordinates") else None,
            assets=assets,
            properties={
                "acquired": props.get("acquired"),
                "pixel_resolution": props.get("pixel_resolution"),
                "epsg_code": props.get("epsg_code"),
                "origin_x": props.get("origin_x"),
                "origin_y": props.get("origin_y"),
                "gsd": props.get("gsd"),
                "off_nadir": props.get("off_nadir"),
                "sun_azimuth": props.get("sun_azimuth"),
                "sun_elevation": props.get("sun_elevation"),
            },
        )

    def _activate_asset(
        self, item_id: str, item_type: str, asset_type: str, api_key: str
    ) -> bool:
        """Activate a Planet asset for download."""
        import httpx

        url = f"{self.BASE_URL}/item-types/{item_type}/items/{item_id}/assets/{asset_type}/activate"
        resp = httpx.post(url, auth=(api_key, ""), timeout=30)
        return resp.status_code in (202, 204)

    def _wait_for_activation(
        self,
        item_id: str,
        item_type: str,
        asset_type: str,
        api_key: str,
        max_wait: int = 300,
    ) -> str | None:
        """Poll until asset is activated; return download URL or None."""
        import httpx

        url = f"{self.BASE_URL}/item-types/{item_type}/items/{item_id}/assets/{asset_type}"
        deadline = time.time() + max_wait
        while time.time() < deadline:
            resp = httpx.get(url, auth=(api_key, ""), timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "active":
                    return data.get("location")
            time.sleep(5)
        return None

    def download(
        self,
        data: SatelliteData,
        destination: Path,
        options: DownloadOptions,
    ) -> DownloadResult:
        """
        Download a Planet scene by activating and fetching its asset.

        Args:
            data: SatelliteData to download.
            destination: Output directory.
            options: Download options.

        Returns:
            DownloadResult.
        """
        self.require_auth()
        import httpx

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        api_key = (
            (self._session.session_data if self._session else {}).get("api_key", "")
            if (self._session.session_data if self._session else {})
            else ""
        )
        item_type = data.satellite or "PSScene"
        asset_type = "ortho_analytic_4b_sr"  # default: surface reflectance

        start_time = time.time()

        # Activate
        self._activate_asset(data.id, item_type, asset_type, api_key)
        download_url = self._wait_for_activation(
            data.id, item_type, asset_type, api_key
        )

        if not download_url:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                error="Asset activation timed out",
            )

        out_file = destination / f"{data.id}_{asset_type}.tif"
        try:
            with httpx.stream(
                "GET",
                download_url,
                auth=(api_key, ""),
                timeout=options.timeout_seconds,
                follow_redirects=True,
            ) as resp:
                self._handle_http_error(resp)
                with open(out_file, "wb") as f:
                    f.writelines(
                        resp.iter_bytes(
                            chunk_size=int(options.chunk_size_mb * 1024 * 1024)
                        )
                    )
        except Exception as exc:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                error=str(exc),
            )

        duration = time.time() - start_time
        return DownloadResult(
            status=DownloadStatus.COMPLETED,
            data_id=data.id,
            provider=self.PROVIDER_ID,
            output_path=out_file,
            output_paths=[out_file],
            bytes_downloaded=out_file.stat().st_size,
            duration_seconds=duration,
        )

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_id=self.PROVIDER_ID,
            name=self.DISPLAY_NAME,
            description=self.DESCRIPTION,
            auth_type="api_key",
            satellites=["PlanetScope", "SkySat", "RapidEye"],
            search=True,
            download=True,
            streaming=False,
            stac=True,
            supports_sub_meter=True,
            supports_tasking=True,
            supports_aoi_filter=True,
            supports_cloud_filter=True,
            supports_date_filter=True,
            supports_resolution_filter=True,
            requires_auth=True,
            regions=["global"],
            resolution_min_m=0.5,
            resolution_max_m=5.0,
            endpoint_url=self.BASE_URL,
            docs_url="https://developers.planet.com/docs/apis/data/",
            supported_formats=[DataFormat.GEOTIFF, DataFormat.COG],
        )

    def get_quota_info(self) -> QuotaInfo:
        """Return Planet quota info (requires authenticated session)."""
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={"note": "Quota depends on Planet subscription tier."},
        )
