"""OpenTopography provider for PyGeoFetch."""

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
from pygeofetch.providers.base import AbstractBaseProvider, AuthenticationError

if TYPE_CHECKING:
    from pygeofetch.models.search_query import SearchQuery


def _plain(v) -> str:
    """Extract plain string from str or SecretStr."""
    if v is None:
        return ""
    if hasattr(v, "get_secret_value"):
        return v.get_secret_value()
    return str(v)


class OpentopographyProvider(AbstractBaseProvider):
    PROVIDER_ID = "opentopography"
    DISPLAY_NAME = "OpenTopography"
    REQUIRES_AUTH = True
    DESCRIPTION = "Global DEM and LiDAR data: SRTM, Copernicus DEM, ALOS World 3D, NASADEM. API key required."
    DATA_TYPES = ["DEM", "LiDAR", "SRTM", "Copernicus DEM"]
    SATELLITES = ["SRTM", "Copernicus", "ALOS", "ICESat"]
    BASE_URL = "https://portal.opentopography.org/API"

    DEM_TYPES = {
        # Verified against OpenTopography's live API documentation
        # (portal.opentopography.org/apidocs) — every value below is
        # confirmed to be a real, currently-accepted demtype. An earlier
        # "srtm30": "SRTMGL30" entry was removed here — that value does
        # not exist in the real API at all (confirmed against the live
        # documented list) and would have produced a 400 Bad Request if
        # ever downloaded; it was also redundant with srtm1arc below,
        # which already correctly represents the real 30m/1-arc-second
        # SRTM product.
        "srtm90": "SRTMGL3",
        "srtm1arc": "SRTMGL1",
        "cop30": "COP30",
        "cop90": "COP90",
        "nasadem": "NASADEM",
        "alos": "AW3D30",
        # A real value in the live API, not independently verified this
        # session — the documentation lists it only as "(DTM 30m)" with
        # no further detail on coverage or methodology, so it's included
        # here as an available option, not asserted to be a confirmed
        # global bare-earth product.
        "gedtm30": "GEDTM30",
    }

    def authenticate(self, credentials: Credentials) -> AuthSession:
        api_key = credentials.api_key or credentials.password
        if not api_key:
            msg = "OpenTopography requires an API key from portal.opentopography.org"
            raise AuthenticationError(msg)
        from datetime import datetime, timedelta, timezone

        session = AuthSession(
            provider=self.PROVIDER_ID,
            access_token=_plain(api_key),
            expires_at=datetime.now(timezone.utc) + timedelta(days=365),
            session_data={"api_key": _plain(api_key)},
        )
        self._session = session
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        return bool(credentials.api_key or credentials.password)

    def set_session(self, session: Any) -> None:
        """Store an authenticated session for use in requests."""
        self._session = session

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        self.require_auth()
        if not query.bbox:
            return []
        bb = query.bbox
        api_key = (
            (self._session.session_data if self._session else {}).get("api_key", "")
            if (self._session.session_data if self._session else {})
            else ""
        )
        results = []
        for dem_key, dem_type in self.DEM_TYPES.items():
            asset_url = (
                f"{self.BASE_URL}/globaldem?demtype={dem_type}"
                f"&south={bb.min_lat}&north={bb.max_lat}&west={bb.min_lon}&east={bb.max_lon}"
                f"&outputFormat=GTiff&API_Key={api_key}"
            )
            item = SatelliteData(
                id=f"opentopo_{dem_type}_{bb.min_lon}_{bb.min_lat}",
                provider=self.PROVIDER_ID,
                satellite="SRTM/Copernicus",
                bbox=(bb.min_lon, bb.min_lat, bb.max_lon, bb.max_lat),
                cloud_cover=None,
                assets={
                    "dem": SatelliteAsset(
                        key="dem",
                        href=asset_url,
                        roles=["data"],
                        media_type="image/tiff",
                    )
                },
                properties={"dem_type": dem_type, "product": dem_key},
            )
            results.append(item)
        return results

    def download(
        self, data: SatelliteData, destination: Path, options: DownloadOptions
    ) -> DownloadResult:
        self.require_auth()
        import httpx

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        start = time.time()
        api_key = (
            (self._session.session_data if self._session else {}).get("api_key", "")
            if (self._session.session_data if self._session else {})
            else ""
        )
        output_paths, total_bytes = [], 0
        asset_errors = []
        for key, asset in (data.data_assets or data.assets).items():
            href = asset.href
            # Always ensure the CURRENT session's key is actually present
            # with a real value -- not just checking whether the literal
            # substring "API_Key=" exists in the URL. search() bakes a key
            # into every href at search time; if that key was empty or
            # stale for any reason, a substring-only check would never
            # notice or correct it, silently sending an unauthenticated
            # request on every retry.
            import re

            existing_match = re.search(r"[?&]API_Key=([^&]*)", href)
            if existing_match and existing_match.group(1):
                pass  # a real, non-empty key is already present, keep it
            elif existing_match:
                # Present but empty -- replace with the current key
                href = href[: existing_match.start(1)] + api_key + href[existing_match.end(1) :]
            else:
                sep = "&" if "?" in href else "?"
                href = f"{href}{sep}API_Key={api_key}"
            out_file = destination / f"{data.id}_{key}.tif"
            try:
                with httpx.stream(
                    "GET", href, timeout=options.timeout_seconds, follow_redirects=True
                ) as resp:
                    self._handle_http_error(resp)
                    bytes_written = 0
                    with open(out_file, "wb") as f:
                        for chunk in resp.iter_bytes(
                            chunk_size=int(options.chunk_size_mb * 1024 * 1024)
                        ):
                            f.write(chunk)
                            bytes_written += len(chunk)
                    if bytes_written == 0:
                        out_file.unlink(missing_ok=True)
                        raise RuntimeError(
                            f"OpenTopography returned an empty response for "
                            f"{key} (status {resp.status_code}) — often means "
                            f"the API key is invalid/unapproved, or the "
                            f"requested area exceeds OpenTopography's size "
                            f"limit for this DEM type."
                        )
                output_paths.append(out_file)
                total_bytes += out_file.stat().st_size
            except Exception as exc:
                asset_errors.append(f"{key}: {exc}")
                self._logger.warning(f"OT download failed: {exc}")
        if not output_paths:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                error="; ".join(asset_errors) if asset_errors else "Download failed",
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
            auth_type="api_key",
            satellites=["SRTM", "Copernicus", "ALOS"],
            search=True,
            download=True,
            supports_aoi_filter=True,
            supports_date_filter=False,
            requires_auth=True,
            regions=["global"],
            resolution_min_m=30.0,
            resolution_max_m=1000.0,
            endpoint_url=self.BASE_URL,
            docs_url="https://opentopography.org/developers",
            supported_formats=[DataFormat.GEOTIFF],
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={
                "note": "Free tier: 10 requests/day. Register for higher limits."
            },
        )
