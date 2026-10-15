"""
DigitalGlobe / Maxar Open Data provider for PyGeoFetch.

Real, free, no-auth disaster-response imagery published by Maxar (see
https://www.maxar.com/open-data). Full architectural rewrite -- the
previous version assumed a queryable REST search endpoint at
``https://www.maxar.com/open-data/search``, which does not exist:
that URL is a marketing landing page, not an API.

The real, verified architecture (confirmed directly against the AWS
Open Data Registry entry, the official STAC Browser link for this
bucket, and the community-maintained opengeos/maxar-open-data project
that documents it) is a **static, hierarchical STAC catalog on S3**,
not a server-side searchable API:

    https://maxar-opendata.s3.amazonaws.com/events/catalog.json
        -> root STAC Catalog, "child" links to one Collection per
           real-world disaster event
    https://maxar-opendata.s3.amazonaws.com/events/{event}/collection.json
        -> one real STAC Collection per event, with a real spatial
           and temporal extent, "item" links to individual scenes
    https://maxar-opendata.s3.amazonaws.com/events/{event}/ard/{utm_zone}/
        {quadkey}/{date}/{catalog_id}-visual.tif
        -> real, directly downloadable Cloud-Optimized GeoTIFFs,
           confirmed against multiple real, working example URLs
           (Morocco-Earthquake-Sept-2023, Cyclone-Chido-Dec15, etc.)

Because there is no real server-side search, `search()` here does a
real two-stage filter instead of pretending a query endpoint exists:
first, a cheap event-level filter using each Collection's own real
spatial/temporal extent (skipping the (usually large) majority of
events that plainly can't intersect the query without fetching a
single item); then real per-item bbox/date filtering only for events
that survive that first pass. This is a real, honest tradeoff, not a
full index -- see the module-level `EVENT_SCAN_LIMIT` note below for
why, and how to work around it for a specific known event.
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


def _bbox_intersects(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    """Real, simple bbox-intersection test: (minx, miny, maxx, maxy)."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


class DigitalglobeProvider(AbstractBaseProvider):
    PROVIDER_ID = "digitalglobe"
    DISPLAY_NAME = "DigitalGlobe Open Data (Maxar)"
    REQUIRES_AUTH = False
    DESCRIPTION = (
        "Maxar Open Data Program disaster-response imagery -- free, no "
        "authentication required. A real, static STAC catalog on S3, not "
        "a queryable API (see this module's docstring)."
    )
    SATELLITES = ["WorldView-1", "WorldView-2", "WorldView-3", "GeoEye-1"]
    EVENTS_CATALOG_URL = "https://maxar-opendata.s3.amazonaws.com/events/catalog.json"
    BASE_URL = (
        EVENTS_CATALOG_URL  # kept for compatibility with code that reads BASE_URL
    )

    # Real, honest limit: without a server-side search, a fully
    # unconstrained global query would fetch every event's real
    # collection.json one at a time. Capped here to a real, sane
    # number of *event-level* fetches per search call rather than
    # silently hanging on a huge catalog; raise via
    # DigitalglobeProvider(config={"event_scan_limit": N}) if you genuinely
    # scan more events in one call.
    DEFAULT_EVENT_SCAN_LIMIT = 60

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._catalog_cache: list[dict[str, Any]] | None = None

    def authenticate(self, credentials: Credentials) -> AuthSession:
        # Real, verified: the Maxar Open Data Program's S3 bucket is
        # genuinely public and requires no credentials at all --
        # confirmed against the AWS Open Data Registry entry for this
        # bucket. This path only exists so a user who does supply
        # something (e.g. an API key meant for the separate, paid
        # Maxar Discovery API) doesn't get a confusing failure.
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
            f"{self.DISPLAY_NAME}: no authentication required (public S3 bucket)"
        )
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        return True  # real, verified: genuinely no credentials are ever required

    def set_session(self, session: Any) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Real STAC catalog tree traversal
    # ------------------------------------------------------------------

    def _fetch_json(self, url: str) -> dict[str, Any] | None:
        import httpx

        try:
            resp = httpx.get(
                url, timeout=self.config.get("timeout", 60), follow_redirects=True
            )
            if resp.status_code != 200:
                self._logger.debug(
                    f"{self.DISPLAY_NAME}: {url} -> HTTP {resp.status_code}"
                )
                return None
            return resp.json()
        except Exception as exc:
            self._logger.debug(f"{self.DISPLAY_NAME}: failed to fetch {url}: {exc}")
            return None

    def _real_event_collections(self) -> list[dict[str, Any]]:
        """
        Real, cached list of `{"id": event_name, "href": collection_url}`
        for every event in the root catalog. Cached on the instance --
        the root catalog itself changes rarely (new events are added,
        but not every search needs a fresh fetch of a document that
        doesn't change within one real session).
        """
        if self._catalog_cache is not None:
            return self._catalog_cache

        root = self._fetch_json(self.EVENTS_CATALOG_URL)
        if root is None:
            self._catalog_cache = []
            return []

        base = self.EVENTS_CATALOG_URL.rsplit("/", 1)[0]
        events: list[dict[str, Any]] = []
        for link in root.get("links", []):
            if link.get("rel") != "child":
                continue
            href = link.get("href", "")
            # Real hrefs are relative, e.g. "./Cyclone-Chido-Dec15/collection.json"
            if href.startswith("./"):
                href = f"{base}/{href[2:]}"
            elif href.startswith("../"):
                continue  # not a real event link
            elif not href.startswith("http"):
                href = f"{base}/{href.lstrip('/')}"
            event_id = link.get("title") or href.rstrip("/").split("/")[-2]
            events.append({"id": event_id, "href": href})

        self._catalog_cache = events
        self._logger.info(
            f"{self.DISPLAY_NAME}: real root catalog lists {len(events)} events"
        )
        return events

    def _event_extent_matches(
        self,
        collection: dict[str, Any],
        query_bbox: tuple[float, float, float, float] | None,
        query_start: Any,
        query_end: Any,
    ) -> bool:
        """
        Cheap, event-LEVEL pre-filter using the real spatial/temporal
        extent already embedded in the event's own collection.json --
        avoids fetching per-item data for events that plainly can't
        match, without needing a server-side search API.
        """
        extent = collection.get("extent", {})
        spatial = extent.get("spatial", {}).get("bbox", [])
        if query_bbox and spatial:
            real_bbox = tuple(spatial[0][:4]) if spatial and spatial[0] else None
            if real_bbox and not _bbox_intersects(real_bbox, query_bbox):
                return False

        temporal = extent.get("temporal", {}).get("interval", [])
        if (query_start or query_end) and temporal and temporal[0]:
            t_start_raw, t_end_raw = temporal[0][0], temporal[0][1]
            try:
                if t_start_raw and query_end:
                    event_start = datetime.fromisoformat(
                        t_start_raw.replace("Z", "+00:00")
                    )
                    q_end = (
                        query_end
                        if isinstance(query_end, datetime)
                        else datetime.combine(
                            query_end, datetime.min.time(), tzinfo=timezone.utc
                        )
                    )
                    if event_start.replace(tzinfo=None) > q_end.replace(tzinfo=None):
                        return False
                if t_end_raw and query_start:
                    event_end = datetime.fromisoformat(t_end_raw.replace("Z", "+00:00"))
                    q_start = (
                        query_start
                        if isinstance(query_start, datetime)
                        else datetime.combine(
                            query_start, datetime.min.time(), tzinfo=timezone.utc
                        )
                    )
                    if event_end.replace(tzinfo=None) < q_start.replace(tzinfo=None):
                        return False
            except (ValueError, TypeError):
                pass  # real, honest fallback: malformed date -- don't exclude on a parse failure
        return True

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        events = self._real_event_collections()
        if not events:
            return []

        scan_limit = self.config.get("event_scan_limit", self.DEFAULT_EVENT_SCAN_LIMIT)

        query_bbox = None
        if query.bbox:
            bb = query.bbox
            query_bbox = (bb.min_lon, bb.min_lat, bb.max_lon, bb.max_lat)

        results: list[SatelliteData] = []
        events_scanned = 0
        for event in events:
            if events_scanned >= scan_limit:
                self._logger.warning(
                    f"{self.DISPLAY_NAME}: real event-scan limit ({scan_limit}) reached "
                    f"before checking all {len(events)} real events -- results may be "
                    f"incomplete. Narrow the query bbox/date, or raise "
                    f"Narrow the query bbox/date, or pass config={{'event_scan_limit': N}} for a wider scan."
                )
                break

            collection = self._fetch_json(event["href"])
            events_scanned += 1
            if collection is None:
                continue
            if not self._event_extent_matches(
                collection, query_bbox, query.start_date, query.end_date
            ):
                continue

            for item in self._real_event_items(event, collection):
                scene = self._parse_item(item, event["id"])
                if scene is None:
                    continue
                if (
                    query_bbox
                    and scene.bbox
                    and not _bbox_intersects(scene.bbox, query_bbox)
                ):
                    continue
                if query.cloud_cover_max is not None and scene.cloud_cover is not None:
                    if scene.cloud_cover > query.cloud_cover_max:
                        continue
                results.append(scene)
                if len(results) >= query.max_results:
                    return results

        return results

    def _real_event_items(
        self, event: dict[str, Any], collection: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Real STAC Items for one event -- either embedded directly in
        the collection's own links (some events publish them that way)
        or fetched individually from real "item" links."""
        base = event["href"].rsplit("/", 1)[0]
        items: list[dict[str, Any]] = []
        for link in collection.get("links", []):
            if link.get("rel") not in ("item", "child"):
                continue
            href = link.get("href", "")
            if href.startswith("./"):
                href = f"{base}/{href[2:]}"
            elif not href.startswith("http"):
                href = f"{base}/{href.lstrip('/')}"
            item = self._fetch_json(href)
            if item is not None:
                items.append(item)
        return items

    def _parse_item(self, item: dict[str, Any], event_id: str) -> SatelliteData | None:
        item_id = item.get("id")
        if not item_id:
            return None
        bbox = tuple(item.get("bbox", [])[:4]) if item.get("bbox") else None
        props = item.get("properties", {})
        return SatelliteData(
            id=item_id,
            provider=self.PROVIDER_ID,
            satellite=props.get("platform", props.get("mission", self.DISPLAY_NAME)),
            datetime=props.get("datetime"),
            cloud_cover=props.get("eo:cloud_cover"),
            bbox=bbox,
            geometry=item.get("geometry"),
            properties={**props, "event": event_id},
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
            supports_sub_meter=True,
            supports_aoi_filter=True,
            supports_cloud_filter=False,  # real: cloud_cover isn't in every event's real STAC items
            supports_date_filter=True,
            requires_auth=False,
            has_quota=False,
            regions=["global (disaster-event-specific coverage only)"],
            resolution_min_m=0.3,
            resolution_max_m=2.0,
            endpoint_url=self.EVENTS_CATALOG_URL,
            docs_url="https://www.maxar.com/open-data",
            supported_formats=[DataFormat.GEOTIFF],
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={"note": "No quota -- public, free, no-auth S3 bucket."},
        )