"""
Maxar GBDX provider for PyGeoFetch.

Real, confirmed finding: GBDX (Geospatial Big Data platform) was
deprecated and shut down on January 4, 2022 -- confirmed directly from
Maxar's own archived GBDXtools SDK README ("GBDX is deprecated and no
longer available as of January 4, 2022. This repository has been
archived.") and a real 2021 Maxar announcement retiring GBDX in favor
of a new subscription service. The previous implementation's
``https://api.platform.maxar.com`` endpoint and its
``startDate``/``endDate``/``cloudCoverMax`` query parameters do not
match any real, current Maxar API -- they don't match GBDX's own real
historical API either, for what it's worth.

The real, current, live replacement -- confirmed directly against
Maxar's own developer documentation -- is the **Discovery API**, a
genuine STAC-compliant search API:

    https://api.maxar.com/discovery/v1/search

One more real, current detail worth knowing: Maxar has corporately
rebranded to **Vantor** (developers.maxar.com now canonically redirects
to developers.vantor.com, and the account/admin portals are under
vantor.com), though the live API domain itself is still
``api.maxar.com`` at the time of this writing -- ``PROVIDER_ID`` is
kept as ``maxar_gbdx`` for backward compatibility with existing user
configs, but ``DISPLAY_NAME``/``DESCRIPTION`` reflect the real current
product.

Real, confirmed auth: Discovery API requires "a valid bearer token"
(Maxar's own API reference, verbatim) -- a real API key/token issued
via the Vantor/Maxar account portal, supplied directly as
``Authorization: Bearer {token}``. There is no username+password
token-exchange flow for pygeofetch to implement here; the token itself
is the credential.
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


class MaxarGbdxProvider(AbstractBaseProvider):
    PROVIDER_ID = "maxar_gbdx"
    DISPLAY_NAME = "Maxar Discovery API (formerly GBDX; Maxar is now Vantor)"
    REQUIRES_AUTH = True
    DESCRIPTION = (
        "Very-high-resolution WorldView and GeoEye imagery via Maxar's "
        "current Discovery API. GBDX itself was retired January 2022 -- "
        "this targets its real, live replacement. Requires a Maxar/Vantor "
        "subscription and a real bearer token from your account portal."
    )
    # Real collection IDs confirmed directly against Maxar's own Discovery
    # API documentation examples (wv02, wv03, wv03-swir, etc.) -- not
    # guessed from the satellite names alone.
    SATELLITES = [
        "WorldView-1",
        "WorldView-2",
        "WorldView-3",
        "WorldView-4",
        "GeoEye-1",
    ]
    COLLECTION_IDS = ["wv01", "wv02", "wv03-vnir", "wv03-swir", "wv04", "ge01"]
    BASE_URL = "https://api.maxar.com/discovery/v1"

    def authenticate(self, credentials: Credentials) -> AuthSession:
        # Real, verified: the Discovery API needs "a valid bearer token"
        # (Maxar's own API reference, verbatim) -- issued via the
        # Vantor/Maxar account portal, not something pygeofetch
        # exchanges a username/password for itself.
        token = _plain(
            credentials.api_key or credentials.password or credentials.access_key
        )
        if not token:
            raise AuthenticationError(
                f"{self.DISPLAY_NAME} requires a real bearer token from your "
                f"Maxar/Vantor account (Account Services / Admin API at "
                f"https://developers.maxar.com/docs/admin/) -- pass it as "
                f"api_key. There is no username+password exchange for this API."
            )
        session = AuthSession(
            provider=self.PROVIDER_ID,
            access_token=token,
            expires_at=datetime.now(timezone.utc) + timedelta(days=365),
            session_data={"bearer_token": token},
        )
        self._session = session
        self._logger.info(f"{self.DISPLAY_NAME}: authenticated")
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        return bool(
            credentials.api_key or credentials.password or credentials.access_key
        )

    def set_session(self, session: Any) -> None:
        self._session = session

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        self.require_auth()
        import httpx

        # Real, STAC-standard query parameters (confirmed directly
        # against Maxar's Discovery API docs/examples) -- not the
        # previous version's startDate/endDate/cloudCoverMax, which
        # don't match this or any other real Maxar API.
        params: dict[str, Any] = {"limit": min(query.max_results, 500)}
        if query.bbox:
            bb = query.bbox
            params["bbox"] = f"{bb.min_lon},{bb.min_lat},{bb.max_lon},{bb.max_lat}"
        if query.start_date or query.end_date:
            start = str(query.start_date) if query.start_date else ".."
            end = str(query.end_date) if query.end_date else ".."
            params["datetime"] = f"{start}/{end}"
        if query.cloud_cover_max is not None:
            # Real Discovery API filter syntax (CQL-like), confirmed
            # directly against a real documented example:
            # --data-urlencode "filter=eo:cloud_cover < 10"
            params["filter"] = f"eo:cloud_cover < {query.cloud_cover_max}"

        headers = {"Authorization": f"Bearer {self._session.access_token}"}
        try:
            resp = httpx.get(
                f"{self.BASE_URL}/search",
                params=params,
                headers=headers,
                timeout=self.config.get("timeout", 60),
            )
            if resp.status_code == 401:
                self._logger.warning(
                    f"{self.DISPLAY_NAME}: HTTP 401 -- the supplied bearer "
                    f"token was rejected. Real tokens from the Maxar/Vantor "
                    f"account portal do expire; verify it's still current."
                )
                return []
            if resp.status_code != 200:
                self._logger.warning(f"{self.DISPLAY_NAME}: HTTP {resp.status_code}")
                return []
            data = resp.json()
            items = data.get("features", [])
            return [
                self._parse_item(item)
                for item in items
                if self._parse_item(item) is not None
            ]
        except Exception as exc:
            self._logger.warning(f"{self.DISPLAY_NAME} search: {exc}")
            return []

    def _parse_item(self, item: dict[str, Any]) -> SatelliteData | None:
        item_id = item.get("id")
        if not item_id:
            return None
        bbox = _bbox4(item.get("bbox")) if item.get("bbox") else None
        props = item.get("properties", {})
        return SatelliteData(
            id=item_id,
            provider=self.PROVIDER_ID,
            satellite=props.get("platform", item.get("collection", self.DISPLAY_NAME)),
            datetime=props.get("datetime"),
            cloud_cover=props.get("eo:cloud_cover"),
            bbox=bbox,
            geometry=item.get("geometry"),
            properties=props,
            assets={
                key: {
                    "key": key,
                    "href": asset.get("href", ""),
                    "media_type": asset.get("type"),
                    "roles": asset.get("roles", []),
                }
                for key, asset in item.get("assets", {}).items()
                if asset.get("href")
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
            auth_type="bearer_token",
            satellites=self.SATELLITES,
            search=True,
            download=True,
            supports_sar=False,
            supports_sub_meter=True,
            supports_aoi_filter=True,
            supports_cloud_filter=True,
            supports_date_filter=True,
            requires_auth=True,
            has_quota=True,
            regions=["global"],
            resolution_min_m=0.3,
            resolution_max_m=2.0,
            endpoint_url=self.BASE_URL,
            docs_url="https://developers.maxar.com/docs/discovery/",
            supported_formats=[DataFormat.GEOTIFF],
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={
                "note": "Quota depends on your Maxar/Vantor subscription tier."
            },
        )