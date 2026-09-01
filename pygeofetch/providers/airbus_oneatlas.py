"""
Airbus OneAtlas provider for PyGeoFetch.

Pleiades, Pleiades Neo, and SPOT 6/7 imagery via Airbus Defence and
Space's OneAtlas Data "Living Library" API.

Real, verified endpoints (confirmed against api.oneatlas.airbus.com /
its geoapi-airbusds.com mirror -- the developer portal's older guide
pages have since been reorganised, but these endpoints are unchanged
and independently corroborated across multiple current tutorial pages
under api-catalog-v2/oad-living-library/):

- Auth:     POST https://authenticate.foundation.api.oneatlas.airbus.com
                 /auth/realms/IDP/protocol/openid-connect/token
            (form-urlencoded: apikey=<key>&grant_type=api_key&client_id=IDP)
- Search:   GET  https://search.foundation.api.oneatlas.airbus.com/api/v2/opensearch
- Download: GET  the item's own "_links.imagesGetBuffer[].href" --
            an already-absolute, ready-to-stream URL returned by search
            itself, not a URL this provider needs to construct.

REAL BUG FIXED: the previous implementation invented a fictional
"/search" endpoint on the wrong host (access.foundation... instead of
search.foundation...), authenticated by wrapping the raw API key as if
it were already a bearer token (no real token exchange call was ever
made), used query parameter names that don't exist in the real API
(limit/startDate/endDate/cloudCoverMax instead of itemsPerPage/
acquisitionDate/cloudCover), and its result parser never populated
`assets` at all -- meaning `download()` always found zero assets and
always failed, on every single call, regardless of credentials.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pygeofetch.core.logging import report_download_progress
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


def _bbox_from_geometry(
    geometry: dict[str, Any] | None
) -> tuple[float, float, float, float] | None:
    """Compute a (min_lon, min_lat, max_lon, max_lat) bbox from a GeoJSON
    Polygon/MultiPolygon geometry -- the real search API returns full
    footprint geometry, not a flat bbox array."""
    if not isinstance(geometry, dict):
        return None
    coords = geometry.get("coordinates")
    if not coords:
        return None

    def _flatten(c):
        if isinstance(c, (int, float)):
            return
        if len(c) == 2 and all(isinstance(v, (int, float)) for v in c):
            yield c
        else:
            for sub in c:
                yield from _flatten(sub)

    points = list(_flatten(coords))
    if not points:
        return None
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return (min(lons), min(lats), max(lons), max(lats))


_CONSTELLATION_TO_SATELLITE = {
    "PHR": "Pleiades",
    "PNEO": "Pleiades Neo",
    "SPOT": "SPOT",
}


class AirbusOneatlasProvider(AbstractBaseProvider):
    PROVIDER_ID = "airbus_oneatlas"
    DISPLAY_NAME = "Airbus OneAtlas"
    REQUIRES_AUTH = True
    DESCRIPTION = "Pleiades, Pleiades Neo, and SPOT 6/7 imagery via Airbus OneAtlas Living Library."
    SATELLITES = ["Pleiades", "Pleiades Neo", "SPOT-6", "SPOT-7"]

    AUTH_URL = (
        "https://authenticate.foundation.api.oneatlas.airbus.com"
        "/auth/realms/IDP/protocol/openid-connect/token"
    )
    SEARCH_URL = "https://search.foundation.api.oneatlas.airbus.com/api/v2/opensearch"
    BASE_URL = (
        SEARCH_URL  # kept for ProviderCapabilities.endpoint_url / generic tooling
    )

    def authenticate(self, credentials: Credentials) -> AuthSession:
        """
        Exchange a OneAtlas API key for a real, short-lived bearer access
        token via the documented Keycloak-style token endpoint.

        REAL BUG FIXED: this previously never called any real endpoint --
        it just repackaged the raw API key as an "access_token" with a
        fabricated 365-day expiry. Every subsequent request then sent an
        API key where Airbus expects a genuine bearer token, which the
        real API rejects.
        """
        api_key = _plain(
            credentials.api_key or credentials.password or credentials.token
        )
        if not api_key:
            msg = (
                f"{self.DISPLAY_NAME} requires an API key. Create one at "
                "https://account.foundation.oneatlas.airbus.com/ and add it "
                f"with: pygeofetch auth add {self.PROVIDER_ID} --api-key YOUR_KEY"
            )
            raise AuthenticationError(msg)

        import httpx

        try:
            resp = httpx.post(
                self.AUTH_URL,
                data={"apikey": api_key, "grant_type": "api_key", "client_id": "IDP"},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self.config.get("timeout", 30),
            )
        except Exception as exc:
            msg = f"{self.DISPLAY_NAME}: could not reach the authentication endpoint: {exc}"
            raise AuthenticationError(msg) from exc

        if resp.status_code == 403:
            msg = (
                f"{self.DISPLAY_NAME}: API key rejected (HTTP 403). Note: an "
                "incorrect key temporarily suspends further attempts, per "
                "Airbus's own documented rate-limiting behaviour -- wait "
                "before retrying rather than retrying immediately."
            )
            raise AuthenticationError(msg)
        if resp.status_code != 200:
            msg = f"{self.DISPLAY_NAME}: authentication failed (HTTP {resp.status_code}): {resp.text[:200]}"
            raise AuthenticationError(msg)

        payload = resp.json()
        access_token = payload.get("access_token")
        if not access_token:
            msg = f"{self.DISPLAY_NAME}: authentication response had no access_token: {payload}"
            raise AuthenticationError(msg)

        expires_in = int(payload.get("expires_in", 3600))
        session = AuthSession(
            provider=self.PROVIDER_ID,
            access_token=access_token,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires_in),
            session_data={"api_key": api_key},
        )
        self._session = session
        self._logger.info(
            f"{self.DISPLAY_NAME}: authenticated (token expires in {expires_in}s)"
        )
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        return bool(credentials.api_key or credentials.password or credentials.token)

    def set_session(self, session: Any) -> None:
        self._session = session

    def _auth_header(self) -> dict[str, str]:
        if self._session and self._session.access_token:
            return {"Authorization": f"Bearer {self._session.access_token}"}
        return {}

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        """
        Search the real OneAtlas Living Library /opensearch endpoint.

        REAL BUG FIXED: the previous version hit a fictional endpoint
        (`{access-host}/search`) with made-up parameter names
        (limit/startDate/endDate/cloudCoverMax) that the real API does
        not recognise -- it would have returned a 404 or ignored every
        filter, every time.
        """
        self.require_auth()
        import httpx

        params: dict[str, Any] = {
            "itemsPerPage": min(query.max_results or 100, 500),
            "startPage": 1,
            # SENSOR = Living Library items, streamable/downloadable
            # on-the-fly today; ALBUM = catalog items needing a
            # pay-per-order workflow this provider doesn't implement.
            "processingLevel": "SENSOR",
        }
        if query.bbox:
            bb = query.bbox
            params["bbox"] = f"{bb.min_lon},{bb.min_lat},{bb.max_lon},{bb.max_lat}"
        elif query.geometry and query.geometry.get("type") == "Polygon":
            coords = query.geometry["coordinates"][0]
            wkt_points = ", ".join(f"{lon} {lat}" for lon, lat in coords)
            params["geometry"] = f"POLYGON(({wkt_points}))"

        if query.start_date or query.end_date:
            start = (
                self._iso(query.start_date)
                if query.start_date
                else "1970-01-01T00:00:00.000Z"
            )
            end = (
                self._iso(query.end_date)
                if query.end_date
                else self._iso(datetime.now(timezone.utc))
            )
            params["acquisitionDate"] = f"[{start},{end}]"

        cc_min = query.cloud_cover_min if query.cloud_cover_min is not None else 0
        cc_max = query.cloud_cover_max if query.cloud_cover_max is not None else 100
        if cc_min != 0 or cc_max != 100:
            params["cloudCover"] = f"[{cc_min},{cc_max}]"

        try:
            resp = httpx.get(
                self.SEARCH_URL,
                params=params,
                headers={**self._auth_header(), "Cache-Control": "no-cache"},
                timeout=self.config.get("timeout", 60),
            )
            if resp.status_code != 200:
                self._handle_http_error(resp)
            data = resp.json()
            features = data.get("features", [])
            return [self._parse_item(item) for item in features]
        except Exception as exc:
            self._logger.warning(f"{self.DISPLAY_NAME} search: {exc}")
            return []

    @staticmethod
    def _iso(value: Any) -> str:
        """Format a date/datetime/str as the millisecond-precision
        ISO8601 UTC string the real acquisitionDate filter expects."""
        if isinstance(value, str):
            return value
        if isinstance(value, datetime):
            dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
            return (
                dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            )
        # date (no time component)
        return f"{value.isoformat()}T00:00:00.000Z"

    def _parse_item(self, item: dict[str, Any]) -> SatelliteData:
        """
        Parse one real OneAtlas /opensearch feature.

        REAL BUG FIXED: previously this only ever populated `properties`
        -- the real, documented download link at
        `_links.imagesGetBuffer[].href` (or a single dict, for
        single-image products) was never read, so `assets` stayed
        permanently empty and `download()` had nothing to download.
        """
        item_id = str(item.get("id", ""))
        props = item.get("properties") or {}
        geometry = item.get("geometry")
        bbox = _bbox_from_geometry(geometry)

        constellation = (
            props.get("constellation") or props.get("constellation_id") or ""
        )
        satellite = _CONSTELLATION_TO_SATELLITE.get(
            str(constellation).upper(), constellation or self.DISPLAY_NAME
        )

        cloud_cover = props.get("cloudCover")
        acquisition_dt = None
        acq_raw = props.get("acquisitionDate")
        if acq_raw:
            try:
                acquisition_dt = datetime.fromisoformat(
                    str(acq_raw).replace("Z", "+00:00")
                )
            except ValueError:
                acquisition_dt = None

        assets: dict[str, SatelliteAsset] = {}
        links = item.get("_links") or {}
        buffer_links = links.get("imagesGetBuffer")
        if isinstance(buffer_links, dict):
            buffer_links = [buffer_links]
        for entry in buffer_links or []:
            href = entry.get("href")
            if not href:
                continue
            name = entry.get("name") or entry.get("resourceId") or "image"
            assets[name] = SatelliteAsset(
                key=name,
                href=href, 
                title=name,
                media_type="image/jp2",
                roles=["data"],
            )
        # Thumbnail/quicklook, when present, are genuinely non-data
        # previews -- kept out of `is_data_asset()` via their roles.
        for role_name, link_key in (
            ("thumbnail", "thumbnail"),
            ("overview", "quicklook"),
        ):
            link = links.get(link_key)
            if isinstance(link, dict) and link.get("href"):
                assets[link_key] = SatelliteAsset(
                    key=link_key,
                    href=link["href"],
                    roles=[role_name],
                )

        return SatelliteData(
            id=item_id,
            provider=self.PROVIDER_ID,
            satellite=satellite,
            sensor=props.get("sensorType"),
            datetime=acquisition_dt,
            bbox=bbox,
            geometry=geometry if isinstance(geometry, dict) else None,
            cloud_cover=float(cloud_cover) if cloud_cover is not None else None,
            assets=assets,
            properties=props,
        )

    def download(
        self, data: SatelliteData, destination: Path, options: DownloadOptions
    ) -> DownloadResult:
        """
        Download real data assets via their `imagesGetBuffer` URLs.

        This part of the original implementation was structurally sound
        (streaming, chunked writes, progress reporting) -- it just never
        received real assets to iterate, because `_parse_item()` never
        populated them. No changes needed here beyond that upstream fix.
        """
        self.require_auth()
        import httpx

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        start = time.time()
        output_paths, total_bytes = [], 0
        headers = self._auth_header()

        assets = data.data_assets or data.assets
        if not assets:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                error=(
                    "No downloadable assets on this item. If it came from a "
                    "processingLevel=ALBUM search result rather than SENSOR "
                    "(Living Library), it requires the pay-per-order workflow "
                    "(pricing + order + delivery polling), which this "
                    "provider does not implement -- re-search without "
                    "relaxing processingLevel to pick up a streamable item."
                ),
            )

        for key, asset in assets.items():
            if not asset.href or not asset.href.startswith("http"):
                continue
            out_file = destination / (asset.href.split("/")[-1] or f"{data.id}_{key}")
            if out_file.suffix == "" or out_file.name in ("buffer", key):
                out_file = destination / f"{data.id}_{key}.jp2"
            try:
                with httpx.stream(
                    "GET",
                    asset.href,
                    headers=headers,
                    timeout=options.timeout_seconds,
                    follow_redirects=True,
                ) as resp:
                    self._handle_http_error(resp)
                    total_bytes_this_asset = int(resp.headers.get("content-length", 0))
                    bytes_written_this_asset = 0
                    chunk_t0 = time.time()
                    with open(out_file, "wb") as f:
                        for chunk in resp.iter_bytes(
                            chunk_size=int(options.chunk_size_mb * 1024 * 1024)
                        ):
                            f.write(chunk)
                            bytes_written_this_asset += len(chunk)
                            elapsed = time.time() - chunk_t0
                            speed = (
                                bytes_written_this_asset / elapsed
                                if elapsed > 0
                                else 0.0
                            )
                            report_download_progress(
                                bytes_written_this_asset, total_bytes_this_asset, speed
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
            auth_type="api_key",
            satellites=self.SATELLITES,
            search=True,
            download=True,
            supports_sar=False,
            supports_sub_meter=True,
            supports_aoi_filter=True,
            supports_cloud_filter=True,
            supports_date_filter=True,
            requires_auth=self.REQUIRES_AUTH,
            has_quota=self.REQUIRES_AUTH,
            regions=["global"],
            resolution_min_m=0.3,
            resolution_max_m=6.0,
            endpoint_url=self.SEARCH_URL,
            docs_url="https://api.oneatlas.airbus.com/api-catalog-v2/oad-living-library/overview/",
            supported_formats=[DataFormat.GEOTIFF],
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={"note": "Quota depends on subscription/contract balance."},
        )
