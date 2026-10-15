"""
ESA Copernicus Hub Mirror provider for PyGeoFetch.

Mirror access to Copernicus Sentinel data. No login for open mirrors.
"""

from __future__ import annotations

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
from pygeofetch.providers.base import AbstractBaseProvider


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


class EsaScihubProvider(AbstractBaseProvider):
    PROVIDER_ID = "esa_scihub"
    DISPLAY_NAME = "ESA Copernicus Hub Mirror"
    REQUIRES_AUTH = False
    DESCRIPTION = (
        "Mirror access to Copernicus Sentinel data. No login for open mirrors."
    )
    SATELLITES = ["Sentinel-1", "Sentinel-2", "Sentinel-3", "Sentinel-5P"]
    BASE_URL = "https://apihub.copernicus.eu/apihub"

    # REAL BUG FIXED (crash): search()/download() previously called
    # self._check_integration_verified(), a method that does not exist
    # anywhere in this codebase -- every call raised AttributeError
    # immediately, regardless of credentials or query.
    #
    # REAL BUG FIXED (dead endpoint): BASE_URL points at the Copernicus
    # Open Access Hub, which ESA permanently decommissioned on
    # 2023-11-02 in favour of the Copernicus Data Space Ecosystem (the
    # `copernicus` provider in this codebase). Rather than let every
    # call burn a real network timeout against a dead host and surface
    # a generic connection error, both methods now fail fast with a
    # clear, actionable message.
    _DEAD_ENDPOINT_MESSAGE = (
        "esa_scihub targets the Copernicus Open Access Hub "
        f"({BASE_URL}), which ESA permanently decommissioned on "
        "2023-11-02. Use the 'copernicus' provider instead -- it "
        "targets the live Copernicus Data Space Ecosystem replacement."
    )

    def authenticate(self, credentials: Credentials) -> AuthSession:
        # No real auth is needed for a request that will never be
        # made, but a session object is still returned so callers that
        # check is_authenticated (e.g. before search()) behave
        # consistently with other no-auth providers.
        session = AuthSession(
            provider=self.PROVIDER_ID,
            access_token="anonymous",
            expires_at=datetime.now(timezone.utc) + timedelta(days=365),
            session_data={},
        )
        self._session = session
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        return True

    def set_session(self, session: Any) -> None:
        """Store an authenticated session for use in requests."""
        self._session = session

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        self._logger.error(self._DEAD_ENDPOINT_MESSAGE)
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
        return DownloadResult(
            status=DownloadStatus.FAILED,
            data_id=data.id,
            provider=self.PROVIDER_ID,
            error=self._DEAD_ENDPOINT_MESSAGE,
        )

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_id=self.PROVIDER_ID,
            name=self.DISPLAY_NAME,
            description=self.DESCRIPTION,
            auth_type="none",
            satellites=["Sentinel-1", "Sentinel-2", "Sentinel-3", "Sentinel-5P"],
            search=True,
            download=True,
            supports_sar=True,
            supports_sub_meter=False,
            supports_aoi_filter=True,
            supports_cloud_filter=True,
            supports_date_filter=True,
            requires_auth=self.REQUIRES_AUTH,
            has_quota=self.REQUIRES_AUTH,
            regions=["global"],
            resolution_min_m=5.0,
            resolution_max_m=300.0,
            endpoint_url=self.BASE_URL,
            docs_url="https://scihub.copernicus.eu/",
            supported_formats=[DataFormat.GEOTIFF],
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={"note": "Quota depends on subscription."},
        )
