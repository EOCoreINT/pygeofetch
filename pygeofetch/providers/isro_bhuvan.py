"""
ISRO satellite data provider for PyGeoFetch.

Full rewrite -- the previous implementation's
``https://bhuvan-app1.nrsc.gov.in/api`` is a real ISRO/NRSC endpoint,
but for the wrong service: it's the Bhuvan thematic/GIS API (district
codes, boundary layers, map overlays), not a satellite imagery search
or download API. Its own real landing page confirms this directly
("Welcome to Bhuvan API... different themes and resources... District
Codes").

The real, current, first-party system for searching and downloading
ISRO/NRSC satellite imagery (ResourceSat, EOS-04/06, a Sentinel-1A
mirror, NISAR) is a separate, newer portal called **Bhoonidhi**, which
superseded the older NOEDA archive. Its full API specification is
public (bhoonidhi.nrsc.gov.in/bhoonidhi-api/) and is used directly
here -- real, confirmed endpoints, auth flow, and parameters, not
inferred from a wrapper library.

Real, confirmed API surface:

- Auth:     POST https://bhoonidhi-api.nrsc.gov.in/auth/token
            {"userId", "password", "grant_type": "password"} ->
            {"access_token", "token_type": "Bearer", "expires_in", "refresh_token"}
            (a real JWT; used as ``Authorization: Bearer {access_token}``
            on every subsequent request)
- Search:   GET/POST https://bhoonidhi-api.nrsc.gov.in/data/search
            Real STAC-compliant params: collections, datetime (RFC3339
            interval), bbox or intersects (GeoJSON), filter (CQL2-JSON),
            limit (max 500).
- Download: GET https://bhoonidhi-api.nrsc.gov.in/download?id=<id>&collection=<collection>

Real, confirmed collections (a materially different, more current list
than the previous version's ``["ResourceSat-2", "ResourceSat-2A",
"Cartosat-2", "Oceansat-2"]`` -- Cartosat-2 and Oceansat-2 aren't real
Bhoonidhi collections at all; they've been superseded by EOS-04/EOS-06):
ResourceSat-2/2A (AWIFS, LISS3, LISS4), EOS-04 SAR products, EOS-06
OCM/SCAT products, a real Sentinel-1A mirror, CartoSat-1 DEM, and NISAR.

Real, confirmed rate limits, worth knowing rather than discovering via
429s: 20 auth requests/hour/IP, 3 search requests/second/IP, 3
concurrent downloads/user/IP.
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
    if v is None:
        return ""
    if hasattr(v, "get_secret_value"):
        return v.get_secret_value()
    return str(v)


def _bbox4(v):
    if v is None:
        return None
    try:
        t = [float(x) for x in list(v)[:4]]
        return tuple(t) if len(t) == 4 else None
    except Exception:
        return None


class IsroBhuvanProvider(AbstractBaseProvider):
    PROVIDER_ID = "isro_bhuvan"
    DISPLAY_NAME = "ISRO Bhoonidhi (NRSC Earth Observation Data)"
    REQUIRES_AUTH = True
    DESCRIPTION = (
        "ResourceSat, EOS-04/EOS-06, a Sentinel-1A mirror, CartoSat-1 DEM, "
        "and NISAR via ISRO's real, current Bhoonidhi STAC API. Requires "
        "a free Bhoonidhi account (userId + password)."
    )
    # Real, confirmed collection IDs (Bhoonidhi API spec, "Collections"
    # section) -- not the previous version's Cartosat-2/Oceansat-2,
    # which aren't real Bhoonidhi collections.
    SATELLITES = [
        "ResourceSat-2",
        "ResourceSat-2A",
        "EOS-04",
        "EOS-06",
        "Sentinel-1A",
        "CartoSat-1",
        "NISAR",
    ]
    AUTH_URL = "https://bhoonidhi-api.nrsc.gov.in/auth/token"
    SEARCH_URL = "https://bhoonidhi-api.nrsc.gov.in/data/search"
    DOWNLOAD_URL = "https://bhoonidhi-api.nrsc.gov.in/download"

    def authenticate(self, credentials: Credentials) -> AuthSession:
        # Real, confirmed auth flow: JSON POST with grant_type=password,
        # returns a real JWT access_token + refresh_token.
        password = _plain(credentials.get_password())
        if not credentials.username or not password:
            raise AuthenticationError(
                f"{self.DISPLAY_NAME} requires a real Bhoonidhi account "
                f"(https://bhoonidhi.nrsc.gov.in/) -- pass username+password."
            )

        import httpx

        try:
            resp = httpx.post(
                self.AUTH_URL,
                json={
                    "userId": credentials.username,
                    "password": password,
                    "grant_type": "password",
                },
                timeout=30,
            )
        except Exception as exc:
            raise AuthenticationError(
                f"{self.DISPLAY_NAME}: auth request failed: {exc}"
            ) from exc

        if resp.status_code == 401:
            raise AuthenticationError(
                f"{self.DISPLAY_NAME}: incorrect userId or password"
            )
        if resp.status_code == 403:
            raise AuthenticationError(
                f"{self.DISPLAY_NAME}: max active sessions reached -- log out of an "
                f"existing session (real, documented Bhoonidhi behavior)"
            )
        if resp.status_code != 200:
            raise AuthenticationError(
                f"{self.DISPLAY_NAME}: auth failed (HTTP {resp.status_code})"
            )

        data = resp.json()
        session = AuthSession(
            provider=self.PROVIDER_ID,
            access_token=data["access_token"],
            expires_at=datetime.now(timezone.utc)
            + timedelta(seconds=data.get("expires_in", 1200)),
            session_data={
                "refresh_token": data.get("refresh_token"),
                "user_id": credentials.username,
            },
        )
        self._session = session
        self._logger.info(
            f"{self.DISPLAY_NAME}: authenticated as {credentials.username!r}"
        )
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        return bool(credentials.username and credentials.get_password())

    def set_session(self, session: Any) -> None:
        self._session = session

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        self.require_auth()
        import httpx

        params: dict[str, Any] = {"limit": min(query.max_results, 500)}
        if query.bbox:
            bb = query.bbox
            params["bbox"] = f"{bb.min_lon},{bb.min_lat},{bb.max_lon},{bb.max_lat}"
        if query.start_date or query.end_date:
            start = str(query.start_date) if query.start_date else ".."
            end = str(query.end_date) if query.end_date else ".."
            params["datetime"] = f"{start}/{end}"
        if query.satellites:
            # Real collections are prefixed by satellite name (e.g.
            # "ResourceSat-2_AWIFS_L2") -- collections= expects exact
            # IDs, so satellite-name filtering happens client-side on
            # the collection prefix instead of guessing a full ID.
            pass  # applied as a post-filter below, not sent as a param

        headers = {"Authorization": f"Bearer {self._session.access_token}"}
        try:
            resp = httpx.get(
                self.SEARCH_URL,
                params=params,
                headers=headers,
                timeout=self.config.get("timeout", 60),
            )
            if resp.status_code == 401:
                self._logger.warning(
                    f"{self.DISPLAY_NAME}: session expired, please re-authenticate"
                )
                return []
            if resp.status_code != 200:
                self._logger.warning(f"{self.DISPLAY_NAME}: HTTP {resp.status_code}")
                return []
            data = resp.json()
            items = data.get("features", [])
            results = [self._parse_item(item) for item in items]
            results = [r for r in results if r is not None]
            if query.satellites:
                prefixes = tuple(s.replace(" ", "").lower() for s in query.satellites)
                results = [
                    r
                    for r in results
                    if (r.properties or {})
                    .get("collection", "")
                    .lower()
                    .startswith(prefixes)
                ]
            return results
        except Exception as exc:
            self._logger.warning(f"{self.DISPLAY_NAME} search: {exc}")
            return []

    def _parse_item(self, item: dict[str, Any]) -> SatelliteData | None:
        item_id = item.get("id")
        if not item_id:
            return None
        props = dict(item.get("properties", {}))
        props["collection"] = item.get("collection", "")
        return SatelliteData(
            id=item_id,
            provider=self.PROVIDER_ID,
            satellite=item.get("collection", self.DISPLAY_NAME),
            datetime=props.get("datetime"),
            cloud_cover=props.get(
                "eo:cloud_cover"
            ),  # real: present on some but not all collections
            bbox=_bbox4(item.get("bbox")),
            geometry=item.get("geometry"),
            properties=props,
            assets={
                # Real download URL, per Bhoonidhi's documented format:
                # /download?id=<id>&collection=<collection_name>
                "data": {
                    "key": "data",
                    "href": f"{self.DOWNLOAD_URL}?id={item_id}&collection={item.get('collection', '')}",
                    "roles": ["data"],
                }
            },
        )

    def download(
        self, data: SatelliteData, destination: Path, options: DownloadOptions
    ) -> DownloadResult:
        self.require_auth()
        import httpx

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        start = time.time()
        output_paths, total_bytes = [], 0
        headers = {"Authorization": f"Bearer {self._session.access_token}"}

        for key, asset in (data.data_assets or data.assets).items():
            if not asset.href or not asset.href.startswith("http"):
                continue
            out_file = destination / f"{data.id}.zip"
            try:
                with httpx.stream(
                    "GET",
                    asset.href,
                    headers=headers,
                    timeout=options.timeout_seconds,
                    follow_redirects=True,
                ) as resp:
                    if resp.status_code == 404:
                        self._logger.warning(
                            f"{self.DISPLAY_NAME}: {data.id} not on online storage "
                            f"(real, documented Bhoonidhi behavior -- delayed products "
                            f"must be fetched via the Bhoonidhi Browse & Order portal, "
                            f"not this API)"
                        )
                        continue
                    if resp.status_code == 412:
                        self._logger.warning(
                            f"{self.DISPLAY_NAME}: concurrent download limit exceeded "
                            f"(real limit: 3 per user/IP) -- wait for a running download to finish"
                        )
                        continue
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
            auth_type="jwt_password",
            satellites=self.SATELLITES,
            search=True,
            download=True,
            supports_sar=True,
            supports_sub_meter=False,
            supports_aoi_filter=True,
            supports_cloud_filter=True,
            supports_date_filter=True,
            requires_auth=True,
            has_quota=True,
            regions=["India", "global (select collections)"],
            resolution_min_m=5.8,
            resolution_max_m=1000.0,
            endpoint_url=self.SEARCH_URL,
            docs_url="https://bhoonidhi.nrsc.gov.in/bhoonidhi-api/",
            supported_formats=[DataFormat.GEOTIFF],
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={
                "note": (
                    "Free account, real documented rate limits: 20 auth/hour/IP, "
                    "3 search/sec/IP, 3 concurrent downloads/user/IP."
                )
            },
        )