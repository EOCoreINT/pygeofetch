"""
TerraBotics provider for PyGeoFetch.

Full integration with the TerraBotics REST API for satellite archive
access and tasking. Uses API Key authentication.

Example::

    from pygeofetch.providers.terrabotics import TerraboticsProvider
    from pygeofetch.models.user_auth import Credentials

    provider = TerraboticsProvider()
    provider.authenticate(Credentials(provider="terrabotics", api_key="YOUR_KEY"))
    results = provider.search(query)
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


class TerraboticsProvider(AbstractBaseProvider):
    """
    TerraBotics satellite archive and tasking provider.

    Provides access to multi-source satellite imagery archives and
    new-collection tasking via the TerraBotics REST API.

    Attributes:
        PROVIDER_ID: 'terrabotics'
        REQUIRES_AUTH: True
    """

    PROVIDER_ID = "terrabotics"
    DISPLAY_NAME = "TerraBotics"
    REQUIRES_AUTH = True
    DESCRIPTION = (
        "Multi-source satellite imagery archive access and tasking via "
        "TerraBotics REST API. Supports <1m commercial imagery."
    )
    DATA_TYPES = ["Archive", "Tasking", "Optical", "High-Resolution"]
    SATELLITES = ["WorldView", "Pléiades", "SPOT", "SkySat"]
    BASE_URL = "https://api.terrabotics.earth/v1"

    def authenticate(self, credentials: Credentials) -> AuthSession:
        """
        Authenticate with TerraBotics using an API key.

        Args:
            credentials: Must include api_key.

        Returns:
            AuthSession.
        """
        import httpx

        api_key = credentials.api_key or credentials.password
        if not api_key:
            msg = "TerraBotics requires an API key. Contact support at: https://terrabotics.earth"
            raise AuthenticationError(msg)

        try:
            resp = httpx.get(
                f"{self.BASE_URL}/account",
                headers={"X-API-Key": api_key},
                timeout=15,
            )
            if resp.status_code == 401:
                msg = "Invalid TerraBotics API key."
                raise AuthenticationError(msg)
            if resp.status_code not in (200, 201, 404):
                # 404 means endpoint exists but no account — key is valid
                self._logger.warning(f"TerraBotics auth check: HTTP {resp.status_code}")
        except AuthenticationError:
            raise
        except Exception as exc:
            self._logger.warning(f"TerraBotics connectivity check failed: {exc}")

        from datetime import datetime, timedelta, timezone

        session = AuthSession(
            provider=self.PROVIDER_ID,
            access_token=_plain(api_key),
            expires_at=datetime.now(timezone.utc) + timedelta(days=365),
            session_data={"api_key": _plain(api_key)},
        )
        self._session = session
        self._logger.info("TerraBotics: authenticated")
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        return bool(credentials.api_key or credentials.password)

    def set_session(self, session: Any) -> None:
        """Store an authenticated session for use in requests."""
        self._session = session

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        """
        Search TerraBotics archive catalog.

        Args:
            query: Search parameters.

        Returns:
            List of SatelliteData.
        """
        self.require_auth()
        import httpx

        api_key = (
            (self._session.session_data if self._session else {}).get("api_key", "")
            if (self._session.session_data if self._session else {})
            else ""
        )
        params: dict[str, Any] = {
            "limit": min(query.max_results, 500),
        }

        if query.bbox:
            bb = query.bbox
            params["bbox"] = f"{bb.min_lon},{bb.min_lat},{bb.max_lon},{bb.max_lat}"
        if query.start_date:
            params["date_from"] = str(query.start_date)
        if query.end_date:
            params["date_to"] = str(query.end_date)
        if query.cloud_cover_max is not None:
            params["cloud_cover_max"] = query.cloud_cover_max
        if query.satellites:
            params["satellites"] = ",".join(query.satellites)
        if query.resolution_max_m:
            params["gsd_max"] = query.resolution_max_m

        try:
            resp = httpx.get(
                f"{self.BASE_URL}/catalog/search",
                params=params,
                headers={"X-API-Key": api_key},
                timeout=self.config.get("timeout", 60),
            )
            if resp.status_code == 404:
                # Endpoint may not exist yet — return empty
                return []
            self._handle_http_error(resp)
            items = resp.json().get(
                "items", resp.json() if isinstance(resp.json(), list) else []
            )
            results = [self._parse_item(item) for item in items]
            self._logger.info(f"TerraBotics: {len(results)} items found")
            return results
        except Exception as exc:
            msg = f"TerraBotics search failed: {exc}"
            raise SearchError(msg) from exc

    def _parse_item(self, item: dict[str, Any]) -> SatelliteData:
        bbox = None
        if "bbox" in item:
            b = item["bbox"]
            if isinstance(b, list) and len(b) == 4:
                bbox = tuple(b)
            elif isinstance(b, str):
                parts = b.split(",")
                if len(parts) == 4:
                    bbox = tuple(float(p) for p in parts)

        assets: dict[str, SatelliteAsset] = {}
        for link in item.get("download_links", []):
            key = f"asset_{len(assets)}"
            assets[key] = SatelliteAsset(key=key, href=link, roles=["data"])

        # Real fix: this parser already assumes a GeoJSON-like item
        # structure (bbox as a top-level field), so its real
        # "geometry" field is safe to extract the same way.
        geometry = item.get("geometry")
        if not (isinstance(geometry, dict) and geometry.get("coordinates")):
            geometry = None

        return SatelliteData(
            id=item.get("id", item.get("scene_id", "")),
            provider=self.PROVIDER_ID,
            satellite=item.get("satellite", item.get("sensor", "")),
            cloud_cover=item.get("cloud_cover"),
            bbox=bbox,
            geometry=geometry,
            assets=assets,
            properties={
                k: v
                for k, v in item.items()
                if k not in ("id", "satellite", "cloud_cover", "bbox", "download_links")
            },
        )

    def download(
        self,
        data: SatelliteData,
        destination: Path,
        options: DownloadOptions,
    ) -> DownloadResult:
        """Download a TerraBotics scene."""
        self.require_auth()
        import httpx

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        api_key = (
            (self._session.session_data if self._session else {}).get("api_key", "")
            if (self._session.session_data if self._session else {})
            else ""
        )
        start_time = time.time()
        output_paths = []
        total_bytes = 0

        # Try to get download URL from API
        try:
            resp = httpx.get(
                f"{self.BASE_URL}/catalog/items/{data.id}/download",
                headers={"X-API-Key": api_key},
                timeout=30,
            )
            if resp.status_code == 200:
                download_url = resp.json().get("url") or resp.json().get("download_url")
                if download_url:
                    out_file = destination / f"{data.id}.tif"
                    with httpx.stream(
                        "GET",
                        download_url,
                        headers={"X-API-Key": api_key},
                        timeout=options.timeout_seconds,
                    ) as dl:
                        self._handle_http_error(dl)
                        with open(out_file, "wb") as f:
                            f.writelines(
                                dl.iter_bytes(
                                    chunk_size=int(options.chunk_size_mb * 1024 * 1024)
                                )
                            )
                    output_paths.append(out_file)
                    total_bytes += out_file.stat().st_size
        except Exception as exc:
            self._logger.warning(f"TerraBotics download error: {exc}")

        # Fall back to assets
        if not output_paths:
            for key, asset in (data.data_assets or data.assets).items():
                if not asset.href or not asset.href.startswith("http"):
                    continue
                out_file = destination / (
                    asset.href.split("/")[-1] or f"{data.id}_{key}"
                )
                try:
                    with httpx.stream(
                        "GET",
                        asset.href,
                        headers={"X-API-Key": api_key},
                        timeout=options.timeout_seconds,
                    ) as resp:
                        self._handle_http_error(resp)
                        with open(out_file, "wb") as f:
                            f.writelines(
                                resp.iter_bytes(
                                    chunk_size=int(options.chunk_size_mb * 1024 * 1024)
                                )
                            )
                    output_paths.append(out_file)
                    total_bytes += out_file.stat().st_size
                except Exception as exc:
                    self._logger.warning(f"Asset {key} failed: {exc}")

        duration = time.time() - start_time
        if not output_paths:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                error="No assets downloaded",
            )
        return DownloadResult(
            status=DownloadStatus.COMPLETED,
            data_id=data.id,
            provider=self.PROVIDER_ID,
            output_path=output_paths[0],
            output_paths=output_paths,
            bytes_downloaded=total_bytes,
            duration_seconds=duration,
        )

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_id=self.PROVIDER_ID,
            name=self.DISPLAY_NAME,
            description=self.DESCRIPTION,
            auth_type="api_key",
            satellites=["WorldView", "Pleiades", "SPOT", "SkySat"],
            search=True,
            download=True,
            supports_sub_meter=True,
            supports_tasking=True,
            supports_aoi_filter=True,
            supports_cloud_filter=True,
            supports_date_filter=True,
            requires_auth=True,
            regions=["global"],
            resolution_min_m=0.3,
            resolution_max_m=10.0,
            endpoint_url=self.BASE_URL,
            docs_url="https://terrabotics.earth/docs",
            supported_formats=[DataFormat.GEOTIFF, DataFormat.COG],
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={"note": "Quota depends on TerraBotics subscription plan."},
        )