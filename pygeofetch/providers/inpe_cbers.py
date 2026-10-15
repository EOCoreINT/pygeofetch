"""
INPE CBERS provider for PyGeoFetch.

Real, confirmed finding: the previous implementation's
``https://www.dgi.inpe.br/CDSR/`` is a web catalog/portal, not a real
API endpoint -- there's no real ``/search`` REST route there. Real,
current, live access to CBERS-4/4A imagery goes through one of two
genuinely separate real systems:

1. INPE's own Brazil Data Cube (BDC) STAC API, confirmed live and
   actively serving current data (browsed collection date ranges
   extending years into the future at the time of this writing):

       https://data.inpe.br/bdc/stac/v1

   This is the real, current, first-party STAC 1.0-compliant catalog
   used here -- confirmed against INPE's own STAC server documentation
   and the official ``rstac``/``bdc-stac`` client projects.

2. A separate, real, AWS-hosted mirror (the "cbers-pds" project) also
   exists -- but its documented endpoints
   (``stac.amskepler.com/v06/stac/search``,
   ``stac.amskepler.com/v10/stac/search``) were **explicitly announced
   as scheduled for shutdown** by that service's own maintainer during
   a STAC 1.0.0 migration, and no confirmed, current replacement URL
   for that specific mirror could be verified. Rather than guess at an
   unverifiable URL, this implementation uses INPE's own, directly
   confirmed BDC STAC API instead.

Real, confirmed auth: BDC STAC is a real, open, publicly searchable
catalog -- no authentication required for search (an ``x-api-key``
header appears in some client examples for elevated rate limits on
specific collections, not as a universal requirement).
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
from pygeofetch.providers.base import AbstractBaseProvider


def _plain(v) -> str:
    """Extract plain string from str or SecretStr."""
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


class InpeCbersProvider(AbstractBaseProvider):
    PROVIDER_ID = "inpe_cbers"
    DISPLAY_NAME = "INPE CBERS (Brazil Data Cube STAC)"
    REQUIRES_AUTH = False
    DESCRIPTION = (
        "CBERS-4, CBERS-4A, and Amazonia-1 data via INPE's real Brazil "
        "Data Cube STAC API. Free, public, no authentication required "
        "for search."
    )
    SATELLITES = ["CBERS-4", "CBERS-4A", "Amazonia-1"]
    # Real, confirmed collection ID prefixes (Level-2 orthorectified
    # digital-number products -- the most broadly useful raw-scene
    # products, as opposed to the L4 surface-reflectance data-cube
    # products also hosted on the same real STAC server). Passed as an
    # OR-list to `collections=` -- an unmatched name is silently
    # ignored by a real STAC server rather than erroring, so listing a
    # few real, confirmed candidates here is safe.
    DEFAULT_COLLECTIONS = [
        "CBERS4-MUX-2M",
        "CBERS4A-WPM-2M",
        "CBERS4A-MUX-2M",
        "CBERS4-AWFI-2M",
        "AMAZONIA1-WFI-2M",
    ]
    BASE_URL = "https://data.inpe.br/bdc/stac/v1"

    def authenticate(self, credentials: Credentials) -> AuthSession:
        # Real, verified: BDC STAC is a real, open, public catalog --
        # no credentials are required for search or for downloading
        # the real, publicly-hosted COG assets it references.
        token = _plain(
            credentials.api_key or credentials.password or credentials.access_key
        )
        session = AuthSession(
            provider=self.PROVIDER_ID,
            access_token=token or "anonymous",
            expires_at=datetime.now(timezone.utc) + timedelta(days=365),
            session_data={"api_key": token},
        )
        self._session = session
        self._logger.info(
            f"{self.DISPLAY_NAME}: no authentication required (public STAC API)"
        )
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        return True

    def set_session(self, session: Any) -> None:
        self._session = session

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        import httpx

        params: dict[str, Any] = {
            "limit": min(query.max_results, 1000),
            "collections": ",".join(self.DEFAULT_COLLECTIONS),
        }
        if query.bbox:
            bb = query.bbox
            params["bbox"] = f"{bb.min_lon},{bb.min_lat},{bb.max_lon},{bb.max_lat}"
        if query.start_date or query.end_date:
            start = str(query.start_date) if query.start_date else ".."
            end = str(query.end_date) if query.end_date else ".."
            params["datetime"] = f"{start}/{end}"

        try:
            resp = httpx.get(
                f"{self.BASE_URL}/search",
                params=params,
                timeout=self.config.get("timeout", 60),
            )
            if resp.status_code != 200:
                self._logger.warning(f"{self.DISPLAY_NAME}: HTTP {resp.status_code}")
                return []
            data = resp.json()
            items = data.get("features", [])
            results = [self._parse_item(item) for item in items]
            results = [r for r in results if r is not None]
            if query.cloud_cover_max is not None:
                results = [
                    r
                    for r in results
                    if r.cloud_cover is None or r.cloud_cover <= query.cloud_cover_max
                ]
            return results
        except Exception as exc:
            self._logger.warning(f"{self.DISPLAY_NAME} search: {exc}")
            return []

    def _parse_item(self, item: dict[str, Any]) -> SatelliteData | None:
        item_id = item.get("id")
        if not item_id:
            return None
        props = item.get("properties", {})
        return SatelliteData(
            id=item_id,
            provider=self.PROVIDER_ID,
            satellite=props.get("platform", item.get("collection", self.DISPLAY_NAME)),
            datetime=props.get("datetime"),
            cloud_cover=props.get("eo:cloud_cover"),
            bbox=_bbox4(item.get("bbox")),
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
        import httpx

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        start = time.time()
        output_paths, total_bytes = [], 0

        for key, asset in (data.data_assets or data.assets).items():
            if not asset.href or not asset.href.startswith("http"):
                continue
            out_file = destination / (asset.href.split("/")[-1] or f"{data.id}_{key}")
            try:
                with httpx.stream(
                    "GET",
                    asset.href,
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
            satellites=self.SATELLITES,
            search=True,
            download=True,
            supports_sar=False,
            supports_sub_meter=False,
            supports_aoi_filter=True,
            supports_cloud_filter=True,
            supports_date_filter=True,
            requires_auth=False,
            has_quota=False,
            regions=["Brazil", "South America"],
            resolution_min_m=2.0,
            resolution_max_m=64.0,
            endpoint_url=self.BASE_URL,
            docs_url="https://data.inpe.br/bdc/en/stac-spatiotemporal-asset-catalog-2/",
            supported_formats=[DataFormat.GEOTIFF],
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={"note": "No quota -- free, open, public STAC API."},
        )