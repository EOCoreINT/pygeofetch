"""
Sentinel Hub provider for PyGeoFetch.

Sentinel Hub (sinergise.com) is a cloud API for satellite data processing
and access. Supports Sentinel-1/2/3, Landsat 5/7/8/9, MODIS, and more.
Uses OAuth2 client credentials.

Example::

    provider = SentinelHubProvider()
    provider.authenticate(Credentials(provider="sentinel_hub",
                          client_id="ID", client_secret="SECRET"))
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
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
    if v is None:
        return ""
    if hasattr(v, "get_secret_value"):
        return v.get_secret_value()
    return str(v)


class SentinelHubProvider(AbstractBaseProvider):
    PROVIDER_ID = "sentinel_hub"
    DISPLAY_NAME = "Sentinel Hub"
    REQUIRES_AUTH = True
    DESCRIPTION = "Cloud API for Sentinel-1/2/3, Landsat, MODIS. Requires subscription."
    DATA_TYPES = ["Sentinel-1", "Sentinel-2", "Sentinel-3", "Landsat", "MODIS", "DEM"]
    SATELLITES = [
        "Sentinel-1",
        "Sentinel-2A",
        "Sentinel-2B",
        "Sentinel-3",
        "Landsat-8",
        "Landsat-9",
    ]
    BASE_URL = "https://services.sentinel-hub.com"
    AUTH_URL = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"

    DATA_SOURCE_MAP = {
        "sentinel-1": "S1GRD",
        "sentinel-2": "S2L2A",
        "sentinel-3": "S3OLCI",
        "landsat8": "LOTL2",
        "landsat9": "LOTL2",
        "modis": "MODIS",
        "dem": "DEM",
    }

    def authenticate(self, credentials: Credentials) -> AuthSession:
        import httpx

        client_id = credentials.client_id or credentials.username
        client_secret = credentials.client_secret or credentials.password
        if not client_id or not client_secret:
            msg = (
                "Sentinel Hub requires client_id and client_secret. "
                "Register at: https://apps.sentinel-hub.com/"
            )
            raise AuthenticationError(msg)
        resp = httpx.post(
            self.AUTH_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=30,
        )
        if resp.status_code == 401:
            msg = "Invalid Sentinel Hub credentials."
            raise AuthenticationError(msg)
        resp.raise_for_status()
        data = resp.json()
        token = data["access_token"]
        expires_in = data.get("expires_in", 3600)
        session = AuthSession(
            provider=self.PROVIDER_ID,
            access_token=token,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires_in),
            session_data={"client_id": client_id},
        )
        self._session = session
        self._logger.info("Sentinel Hub: authenticated")
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        return bool(
            (credentials.client_id or credentials.username)
            and (credentials.client_secret or credentials.password)
        )

    def set_session(self, session: Any) -> None:
        """Store an authenticated session for use in requests."""
        self._session = session

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        self.require_auth()
        import httpx

        data_source = self._resolve_datasource(query)
        payload: dict[str, Any] = {
            "collections": {"input": [{"type": data_source}]},
            "limit": min(query.max_results, 100),
        }
        if query.bbox:
            bb = query.bbox
            payload["spatial"] = {
                "bbox": [bb.min_lon, bb.min_lat, bb.max_lon, bb.max_lat]
            }
        if query.start_date or query.end_date:
            payload["timeRange"] = {
                "from": f"{query.start_date}T00:00:00Z"
                if query.start_date
                else "2015-01-01T00:00:00Z",
                "to": f"{query.end_date}T23:59:59Z"
                if query.end_date
                else datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        if query.cloud_cover_max is not None:
            payload["filter"] = {"maxCloudCoverage": query.cloud_cover_max}
        try:
            resp = httpx.post(
                f"{self.BASE_URL}/api/v1/catalog/1.0.0/search",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._session.access_token if self._session else ''}"  # noqa: E501
                },
                timeout=self.config.get("timeout", 60),
            )
            if resp.status_code != 200:
                self._handle_http_error(resp)
            features = resp.json().get("features", [])
            return [SatelliteData.from_stac_item(f, self.PROVIDER_ID) for f in features]
        except Exception as exc:
            msg = f"Sentinel Hub search failed: {exc}"
            raise SearchError(msg) from exc

    def _resolve_datasource(self, query: SearchQuery) -> str:
        if not query.satellites:
            return "S2L2A"
        key = query.satellites[0].lower().replace(" ", "").replace("-", "")
        for k, ds in self.DATA_SOURCE_MAP.items():
            if k.replace("-", "") in key:
                return ds
        return "S2L2A"

    def download(
        self, data: SatelliteData, destination: Path, options: DownloadOptions
    ) -> DownloadResult:
        self.require_auth()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        import httpx

        start = time.time()
        output_paths, total_bytes = [], 0
        for key, asset in (data.data_assets or data.assets).items():
            if not asset.href or not asset.href.startswith("http"):
                continue
            # Real, confirmed bug fix -- see element84.py for the full
            # explanation. Scene disambiguation is now unconditional.
            raw_name = asset.href.split("/")[-1] or f"{key}.tif"
            out_file = destination / f"{data.id}_{raw_name}"
            try:
                with httpx.stream(
                    "GET",
                    asset.href,
                    headers={
                        "Authorization": f"Bearer {self._session.access_token if self._session else ''}"  # noqa: E501
                    },
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
            duration_seconds=time.time() - start,
        )

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_id=self.PROVIDER_ID,
            name=self.DISPLAY_NAME,
            description=self.DESCRIPTION,
            auth_type="oauth2_client",
            satellites=[
                "Sentinel-1",
                "Sentinel-2",
                "Sentinel-3",
                "Landsat-8",
                "Landsat-9",
                "MODIS",
            ],
            search=True,
            download=True,
            streaming=True,
            stac=True,
            supports_sar=True,
            supports_cql2=True,
            supports_aoi_filter=True,
            supports_cloud_filter=True,
            supports_date_filter=True,
            requires_auth=True,
            regions=["global"],
            resolution_min_m=3.0,
            resolution_max_m=1000.0,
            endpoint_url=self.BASE_URL,
            docs_url="https://docs.sentinel-hub.com/",
            supported_formats=[DataFormat.GEOTIFF, DataFormat.COG],
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={"note": "Quota depends on Sentinel Hub subscription."},
        )
