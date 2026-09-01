"""
Google Earth Engine provider for PyGeoFetch.

Google Earth Engine catalog. Multi-petabyte multi-mission archive. Service account auth.
"""

from __future__ import annotations

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


def _bbox4(v):
    """Normalise bbox to (float, float, float, float) or None."""
    if v is None:
        return None
    try:
        t = [float(x) for x in list(v)[:4]]
        return tuple(t) if len(t) == 4 else None
    except Exception:
        return None


class GoogleEarthEngineProvider(AbstractBaseProvider):
    PROVIDER_ID = "google_earth_engine"
    DISPLAY_NAME = "Google Earth Engine"
    REQUIRES_AUTH = True
    DESCRIPTION = "Google Earth Engine catalog. Multi-petabyte multi-mission archive. Service account auth."
    SATELLITES = ["Sentinel-1", "Sentinel-2", "Landsat", "MODIS", "VIIRS"]
    BASE_URL = "https://earthengine.googleapis.com/v1alpha"

    # REAL BUG FIXED (crash): search()/download() previously called
    # self._check_integration_verified(), a method that does not exist
    # anywhere in this codebase -- every call raised AttributeError
    # immediately, regardless of credentials or query.
    #
    # Honest scope note: Earth Engine's real API is structurally
    # different from the simple REST search+stream-download pattern
    # this file was templated from. Real auth is a Google service-
    # account JWT exchange (google-auth), not a bearer API key; real
    # search/read access goes through Earth Engine's own asset and
    # computation API (ee.ImageCollection filtering, computePixels,
    # or an async export to Drive/GCS), not a "/search" REST endpoint
    # that returns a flat list of downloadable hrefs. Implementing
    # that correctly needs the google-auth and earthengine-api
    # packages as new dependencies and a genuinely different request
    # flow -- not something to fake with the same generic template
    # used elsewhere in this file. Until that real integration exists,
    # this fails clearly and immediately rather than either crashing
    # or silently pretending a REST call it does to a URL that isn't
    # Earth Engine's real API succeeded.
    _NOT_IMPLEMENTED_MESSAGE = (
        "google_earth_engine is not yet implemented against the real "
        "Earth Engine API. It needs a Google service-account JWT auth "
        "flow (google-auth) and Earth Engine's own asset/computation "
        "API (ee.ImageCollection / computePixels / export-to-GCS), "
        "none of which this provider currently implements. Track "
        "progress or contribute at "
        "https://github.com/EOCoreINT/pygeofetch — see CONTRIBUTING.md."
    )

    def authenticate(self, credentials: Credentials) -> AuthSession:
        raise AuthenticationError(self._NOT_IMPLEMENTED_MESSAGE)

    def validate_credentials(self, credentials: Credentials) -> bool:
        return False

    def set_session(self, session: Any) -> None:
        """Store an authenticated session for use in requests."""
        self._session = session

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        self._logger.error(self._NOT_IMPLEMENTED_MESSAGE)
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
            error=self._NOT_IMPLEMENTED_MESSAGE,
        )

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_id=self.PROVIDER_ID,
            name=self.DISPLAY_NAME,
            description=self.DESCRIPTION,
            auth_type="service_account",
            satellites=["Sentinel-1", "Sentinel-2", "Landsat", "MODIS", "VIIRS"],
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
            resolution_min_m=0.5,
            resolution_max_m=1000.0,
            endpoint_url=self.BASE_URL,
            docs_url="https://developers.google.com/earth-engine/",
            supported_formats=[DataFormat.GEOTIFF],
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={"note": "Quota depends on subscription."},
        )
