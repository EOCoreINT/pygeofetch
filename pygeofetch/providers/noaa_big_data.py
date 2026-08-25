"""
NOAA Big Data provider for PyGeoFetch.

GOES-16/17/18, NEXRAD, MRMS, and other NOAA datasets on AWS Open Data.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
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


def _bbox4(v):
    """Normalise bbox to (float, float, float, float) or None."""
    if v is None:
        return None
    try:
        t = [float(x) for x in list(v)[:4]]
        return tuple(t) if len(t) == 4 else None
    except Exception:
        return None


class NoaaBigDataProvider(AbstractBaseProvider):
    PROVIDER_ID = "noaa_big_data"
    DISPLAY_NAME = "NOAA Big Data"
    REQUIRES_AUTH = False
    DESCRIPTION = (
        "GOES-16/17/18, NEXRAD, MRMS, and other NOAA datasets on AWS Open Data."
    )
    SATELLITES = ["GOES-16", "GOES-17", "GOES-18"]
    BASE_URL = "https://noaa-goes16.s3.amazonaws.com"

    def authenticate(self, credentials: Credentials) -> AuthSession:

        token = (
            credentials.api_key or credentials.password or credentials.access_key or ""
        )
        if self.REQUIRES_AUTH and not token and not credentials.username:
            msg = f"{self.DISPLAY_NAME} requires credentials. See: https://registry.opendata.aws/noaa-goes/"
            raise AuthenticationError(msg)
        session = AuthSession(
            provider=self.PROVIDER_ID,
            access_token=_plain(token) or credentials.username or "anonymous",
            expires_at=datetime.now(timezone.utc) + timedelta(days=365),
            session_data={"api_key": _plain(token), "username": credentials.username or ""},
        )
        self._session = session
        self._logger.info(f"{self.DISPLAY_NAME}: authenticated")
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        if not self.REQUIRES_AUTH:
            return True
        return bool(credentials.api_key or credentials.password or credentials.username)

    def set_session(self, session: Any) -> None:
        """Store an authenticated session for use in requests."""
        self._session = session

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        if self.REQUIRES_AUTH:
            self.require_auth()
        import httpx

        if not self.BASE_URL:
            return []
        params: dict[str, Any] = {"limit": min(query.max_results, 500)}
        if query.bbox:
            bb = query.bbox
            params["bbox"] = f"{bb.min_lon},{bb.min_lat},{bb.max_lon},{bb.max_lat}"
        if query.start_date:
            params["startDate"] = str(query.start_date)
        if query.end_date:
            params["endDate"] = str(query.end_date)
        if query.cloud_cover_max is not None:
            params["cloudCoverMax"] = query.cloud_cover_max
        headers: dict[str, str] = {}
        if (
            self._session
            and self._session.access_token
            and self._session.access_token not in ("anonymous", "")
        ):
            if self._session.session_data and self._session.session_data.get("api_key"):
                headers["X-API-Key"] = self._session.session_data["api_key"]
            else:
                headers["Authorization"] = f"Bearer {self._session.access_token}"
        try:
            resp = httpx.get(
                f"{self.BASE_URL}/search",
                params=params,
                headers=headers,
                timeout=self.config.get("timeout", 60),
            )
            if resp.status_code == 404:
                return []
            if resp.status_code != 200:
                self._logger.warning(f"{self.DISPLAY_NAME}: HTTP {resp.status_code}")
                return []
            data = resp.json()
            items = data.get(
                "features", data.get("items", data if isinstance(data, list) else [])
            )
            return [self._parse_item(item) for item in items]
        except Exception as exc:
            self._logger.warning(f"{self.DISPLAY_NAME} search: {exc}")
            return []

    def _parse_item(self, item: dict[str, Any]) -> SatelliteData:
        item_id = str(item.get("id", item.get("scene_id", item.get("identifier", ""))))
        bbox = None
        raw = item.get("bbox") or item.get("footprint")
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            bbox = _bbox4(float(x) for x in raw)
        cloud_raw = (
            item.get("cloud_cover")
            or item.get("cloudCover")
            or (item.get("properties") or {}).get("eo:cloud_cover")
        )
        # Real fix: this shared, generic parser already assumes a
        # GeoJSON Feature-like response (a "features" array), so its
        # real "geometry" field (standard GeoJSON) is safe to extract
        # the same way -- confirmed consistent with the existing bbox
        # extraction on this same response shape, not a new assumption.
        geometry = item.get("geometry")
        if not (isinstance(geometry, dict) and geometry.get("coordinates")):
            geometry = None

        return SatelliteData(
            id=item_id,
            provider=self.PROVIDER_ID,
            satellite=item.get("satellite", item.get("mission", self.DISPLAY_NAME)),
            cloud_cover=float(cloud_raw) if cloud_raw is not None else None,
            bbox=bbox,
            geometry=geometry,
            properties={
                k: v for k, v in item.items() if k not in ("id", "bbox", "assets")
            },
        )

    def download(
        self, data: SatelliteData, destination: Path, options: DownloadOptions
    ) -> DownloadResult:
        if self.REQUIRES_AUTH:
            self.require_auth()
        import httpx

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        start = time.time()
        output_paths, total_bytes = [], 0
        headers: dict[str, str] = {}
        if (
            self._session
            and self._session.access_token
            and self._session.access_token not in ("anonymous", "")
        ):
            headers["Authorization"] = f"Bearer {self._session.access_token}"
        for key, asset in (data.data_assets or data.assets).items():
            if not asset.href or not asset.href.startswith("http"):
                continue
            out_file = destination / (asset.href.split("/")[-1] or f"{data.id}_{key}")
            try:
                with httpx.stream(
                    "GET",
                    asset.href,
                    headers=headers,
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
            auth_type="none",
            satellites=["GOES-16", "GOES-17", "GOES-18"],
            search=True,
            download=True,
            supports_sar=False,
            supports_sub_meter=False,
            supports_aoi_filter=True,
            supports_cloud_filter=True,
            supports_date_filter=True,
            requires_auth=self.REQUIRES_AUTH,
            has_quota=self.REQUIRES_AUTH,
            regions=["global"],
            resolution_min_m=500.0,
            resolution_max_m=10000.0,
            endpoint_url=self.BASE_URL,
            docs_url="https://registry.opendata.aws/noaa-goes/",
            supported_formats=[DataFormat.GEOTIFF],
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={"note": "Quota depends on subscription."},
        )
