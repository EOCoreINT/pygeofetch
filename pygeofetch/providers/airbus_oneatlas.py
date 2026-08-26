"""
Airbus OneAtlas provider for PyGeoFetch.

Pleiades and SPOT 6/7 imagery via Airbus Defence and Space.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pygeofetch.models.download_task import (
    DownloadOptions,
    DownloadResult,
    DownloadStatus,
)
from pygeofetch.models.satellite_data import (
    DataFormat,
    ProviderCapabilities,
    QuotaInfo,
    SatelliteData,
)
from pygeofetch.models.search_query import SearchQuery
from pygeofetch.models.user_auth import AuthSession, Credentials
from pygeofetch.providers.base import AbstractBaseProvider, AuthenticationError


def _plain(v) -> str:
    """Extract plain string from str or SecretStr."""
    if v is None:
        return ""
    if hasattr(v, "get_secret_value"):
        return v.get_secret_value()
    return str(v)


class AirbusOneatlasProvider(AbstractBaseProvider):
    PROVIDER_ID = "airbus_oneatlas"
    DISPLAY_NAME = "Airbus OneAtlas"
    REQUIRES_AUTH = True
    DESCRIPTION = "Pleiades and SPOT 6/7 imagery via Airbus Defence and Space."
    DATA_TYPES = ["optical", "panchromatic", "multispectral"]
    BASE_URL = "https://search.foundation.api.oneatlas.airbus.com/api/v2/opensearch"

    def authenticate(self, credentials: Credentials) -> AuthSession:
        """
        Authenticate with Airbus OneAtlas API.OneAtlas uses API key authentication via X-API-Key header.
        """
        # Extract API key from credentials
        api_key = None

        # Check all possible credential fields
        if credentials.api_key:
            api_key = _plain(credentials.api_key)
        elif credentials.password:
            api_key = _plain(credentials.password)
        elif credentials.access_key:
            api_key = _plain(credentials.access_key)
        elif credentials.token:
            api_key = _plain(credentials.token)

        # Also check if API key is in session_data (from previous auth)
        if not api_key and self._session and self._session.session_data:
            api_key = self._session.session_data.get("api_key")

        if not api_key and credentials.username:
            # If no API key but username exists, try to use username as API key
            api_key = _plain(credentials.username)

        if not api_key:
            msg = (
                f"{self.DISPLAY_NAME} requires an API key. "
                "Get your API key from: https://oneatlas.airbus.com/api-docs/\n"
                "Set it with: pygeofetch auth add airbus_oneatlas --api-key YOUR_KEY"
            )
            raise AuthenticationError(msg)

        # Create session with the API key
        session = AuthSession(
            provider=self.PROVIDER_ID,
            access_token=api_key,
            expires_at=datetime.now(timezone.utc),
            session_data={
                "api_key": api_key,
                "username": credentials.username or "",
            },
        )
        self._session = session
        self._logger.info(f"{self.DISPLAY_NAME}: authenticated successfully")
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        """Validate that credentials contain an API key."""
        if not self.REQUIRES_AUTH:
            return True

        # Check for API key in various places
        api_key = None
        if credentials.api_key:
            api_key = _plain(credentials.api_key)
        elif credentials.password:
            api_key = _plain(credentials.password)
        elif credentials.access_key:
            api_key = _plain(credentials.access_key)
        elif credentials.token:
            api_key = _plain(credentials.token)
        elif credentials.username:
            api_key = _plain(credentials.username)

        # Also check if we have a session with API key
        if not api_key and self._session and self._session.session_data:
            api_key = self._session.session_data.get("api_key")

        return bool(api_key)

    def _build_search_payload(self, query: SearchQuery) -> dict[str, Any]:
        """Build search payload according to OneAtlas API spec."""
        payload: dict[str, Any] = {
            "itemsPerPage": min(query.max_results, 500),
            "startPage": 1,
            "processingLevel": "SENSOR",  # Living Library: cloud < 30%, incidence < 40°
        }

        # Geographic filter - bbox
        if query.bbox:
            bb = query.bbox
            payload["bbox"] = f"{bb.min_lon},{bb.min_lat},{bb.max_lon},{bb.max_lat}"

        # Date filter - acquisitionDate expects [start,end] format
        if query.start_date or query.end_date:
            start = query.start_date or "1900-01-01T00:00:00.000Z"
            end = query.end_date or datetime.now(timezone.utc).isoformat()
            payload["acquisitionDate"] = f"[{start},{end}]"

        # Cloud cover filter - expects [min,max] format
        if query.cloud_cover_max is not None:
            payload["cloudCover"] = f"[0,{query.cloud_cover_max}]"

        return payload

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers for OneAtlas API."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Try to get API key from session
        api_key = None
        if self._session:
            if self._session.session_data:
                api_key = self._session.session_data.get("api_key")
            if not api_key and self._session.access_token:
                api_key = _plain(self._session.access_token)

        if api_key:
            headers["X-API-Key"] = _plain(api_key)
            self._logger.debug(f"Using API key: {api_key[:4]}...{api_key[-4:]}")
        else:
            self._logger.warning("No API key available for authentication")

        return headers

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        """Search for satellite data matching the query using POST method."""
        import httpx

        # Ensure we're authenticated
        if self.REQUIRES_AUTH:
            try:
                self.require_auth()
            except AuthenticationError:
                self._logger.warning(
                    "Not authenticated, attempting to use stored credentials"
                )
                if not self._session:
                    raise AuthenticationError(
                        f"{self.DISPLAY_NAME} requires authentication. "
                        "Run: pygeofetch auth add airbus_oneatlas --api-key YOUR_KEY"
                    )

        if not self.BASE_URL:
            return []

        payload = self._build_search_payload(query)
        headers = self._get_auth_headers()

        self._logger.info(f"Searching {self.DISPLAY_NAME} with payload: {payload}")

        try:
            # Use POST method as per OneAtlas API documentation
            with self._circuit_breaker:
                response = httpx.post(
                    self.BASE_URL,
                    json=payload,  # Send as JSON body
                    headers=headers,
                    timeout=self.config.get("timeout", 60),
                )

                # Handle HTTP errors using the base class method
                self._handle_http_error(response)

                data = response.json()

                # Check for error response
                if data.get("error"):
                    self._logger.warning(f"{self.DISPLAY_NAME} API error: {data}")
                    return []

                # Parse features from GeoJSON response
                features = data.get("features", [])
                if not features:
                    self._logger.info(f"{self.DISPLAY_NAME}: No results found")
                    return []

                results = [self._parse_item(feature) for feature in features]
                self._logger.info(f"{self.DISPLAY_NAME}: Found {len(results)} results")
                return results

        except httpx.TimeoutException:
            self._logger.warning(f"{self.DISPLAY_NAME}: Request timeout")
            return []
        except AuthenticationError:
            raise
        except Exception as exc:
            self._logger.warning(f"{self.DISPLAY_NAME} search error: {exc}")
            return []

    def _parse_item(self, item: dict[str, Any]) -> SatelliteData:
        """Parse a OneAtlas GeoJSON feature into SatelliteData."""
        properties = item.get("properties", {})
        geometry = item.get("geometry")

        # Extract ID - prefer properties.id as it's more reliable
        item_id = str(properties.get("id", item.get("id", "")))

        # Parse bbox from geometry if available
        bbox = None
        if geometry and geometry.get("coordinates"):
            coords = geometry.get("coordinates", [])
            if coords and coords[0]:
                points = coords[0]
                if points:
                    lons = [p[0] for p in points]
                    lats = [p[1] for p in points]
                    bbox = (min(lons), min(lats), max(lons), max(lats))

        # Extract cloud cover
        cloud_raw = properties.get("cloudCover")
        cloud_cover = float(cloud_raw) if cloud_raw is not None else None

        # Determine satellite/platform
        satellite = properties.get("platform") or properties.get(
            "constellation", "unknown"
        )

        # Extract acquisition date
        acq_date = properties.get("acquisitionDate")

        # Build assets/download links
        assets = {}
        links = item.get("_links", {})

        # Look for download links in _links
        if "download" in links:
            download_href = links["download"].get("href")
            if download_href:
                assets["download"] = {
                    "href": download_href,
                    "type": "application/octet-stream",
                    "title": "Download",
                }

        # Add quicklook/thumbnail if available
        if "quicklook" in links:
            quicklook_href = links["quicklook"].get("href")
            if quicklook_href:
                assets["quicklook"] = {
                    "href": quicklook_href,
                    "type": "image/jpeg",
                    "title": "Quicklook",
                }

        return SatelliteData(
            id=item_id,
            provider=self.PROVIDER_ID,
            satellite=satellite,
            cloud_cover=cloud_cover,
            bbox=bbox,
            geometry=geometry,
            properties={
                "acquisition_date": acq_date,
                "constellation": properties.get("constellation"),
                "platform": properties.get("platform"),
                "incidence_angle": properties.get("incidenceAngle"),
                "resolution": properties.get("resolution"),
                "processing_level": properties.get("processingLevel"),
                "product_type": properties.get("productType"),
                "title": properties.get("title"),
                "cloud_cover": cloud_cover,
                "workspace_id": properties.get("workspaceId"),
                "workspace_name": properties.get("workspaceName"),
            },
            data_assets=assets,
        )

    def download(
        self,
        data: SatelliteData,
        destination: Path,
        options: DownloadOptions,
    ) -> DownloadResult:
        """Download a satellite data product."""
        import httpx

        if self.REQUIRES_AUTH:
            self.require_auth()

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        start = time.time()
        output_paths = []
        total_bytes = 0

        # Get authentication headers
        headers = self._get_auth_headers()

        # Determine what to download.
        # NOTE: data.data_assets and data.assets both return
        # dict[str, SatelliteAsset] (Pydantic model instances), never plain
        # dicts, so we pull .href off the model rather than treating it as
        # a dict.
        assets_to_download: list[tuple[str, dict[str, Any]]] = []

        # Check if we have data_assets (primary, non-thumbnail assets)
        if data.data_assets:
            for key, asset in data.data_assets.items():
                if asset.href:
                    assets_to_download.append((key, {"href": asset.href}))
        elif data.assets:
            # Fall back to all assets if there are no primary data assets.
            for key, asset in data.assets.items():
                if asset.href:
                    assets_to_download.append((key, {"href": asset.href}))

        # If no assets, try to get download from properties
        if not assets_to_download and data.properties:
            download_url = data.properties.get("download_url")
            if download_url:
                assets_to_download.append(("download", {"href": download_url}))

        if not assets_to_download:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                error="No download URLs found for this item",
            )

        # Download each asset
        # NOTE: named asset_info (not "asset") to avoid mypy treating this
        # as re-binding the earlier `asset: SatelliteAsset` loop variable
        # above to a new, incompatible dict[str, Any] type.
        for key, asset_info in assets_to_download:
            href = asset_info.get("href")
            if not href:
                continue

            # Generate output filename
            filename = href.split("/")[-1] or f"{data.id}_{key}.tif"
            out_file = destination / filename

            try:
                with httpx.stream(
                    "GET",
                    href,
                    headers=headers,
                    timeout=options.timeout_seconds,
                    follow_redirects=True,
                ) as response:
                    self._handle_http_error(response)

                    total_bytes_asset = int(response.headers.get("content-length", 0))
                    bytes_written = 0
                    chunk_t0 = time.time()

                    with open(out_file, "wb") as f:
                        for chunk in response.iter_bytes(
                            chunk_size=int(options.chunk_size_mb * 1024 * 1024)
                        ):
                            f.write(chunk)
                            bytes_written += len(chunk)

                            # Report progress
                            elapsed = time.time() - chunk_t0
                            if elapsed > 5:
                                speed = bytes_written / elapsed if elapsed > 0 else 0
                                self._logger.debug(
                                    f"Downloading {filename}: {bytes_written}/{total_bytes_asset} bytes "
                                    f"({speed/1024/1024:.2f} MB/s)"
                                )
                                chunk_t0 = time.time()
                                bytes_written = 0

                output_paths.append(out_file)
                total_bytes += out_file.stat().st_size
                self._logger.info(f"Downloaded: {filename}")

            except Exception as exc:
                self._logger.warning(f"Download of {key} failed: {exc}")
                # Continue with other assets

        if not output_paths:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                error="All downloads failed",
            )

        return DownloadResult(
            status=DownloadStatus.COMPLETED,
            data_id=data.id,
            provider=self.PROVIDER_ID,
            output_path=output_paths[0],
            output_paths=output_paths,
            bytes_downloaded=total_bytes,
            duration_seconds=time.time() - start,
        )

    def get_capabilities(self) -> ProviderCapabilities:
        """Return what this provider supports."""
        return ProviderCapabilities(
            provider_id=self.PROVIDER_ID,
            name=self.DISPLAY_NAME,
            description=self.DESCRIPTION,
            auth_type="api_key",
            satellites=["Pleiades-1A", "Pleiades-1B", "SPOT-6", "SPOT-7"],
            search=True,
            download=True,
            supports_sar=False,
            supports_sub_meter=True,
            supports_aoi_filter=True,
            supports_cloud_filter=True,
            supports_date_filter=True,
            requires_auth=self.REQUIRES_AUTH,
            has_quota=True,
            regions=["global"],
            resolution_min_m=0.5,
            resolution_max_m=6.0,
            endpoint_url=self.BASE_URL,
            docs_url="https://oneatlas.airbus.com/api-docs/",
            supported_formats=[DataFormat.GEOTIFF],
        )

    def get_quota_info(self) -> QuotaInfo:
        """Return current quota/rate-limit usage information."""
        if self.REQUIRES_AUTH:
            self.require_auth()

        return QuotaInfo(
            provider=self.PROVIDER_ID,
            remaining=None,
            total=None,
            reset_time=None,
            extra_info={
                "note": "Quota depends on subscription. Check your Airbus OneAtlas dashboard.",
                "limits": {
                    "concurrent_requests": "Limited by subscription",
                    "data_volume": "Varies by plan",
                },
            },
        )
