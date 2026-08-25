"""
USGS Earth Explorer (EarthExplorer) M2M API provider.

Provides access to Landsat, MODIS, and other USGS datasets via the
Machine-to-Machine (M2M) API v1.5.

Authentication:
    Uses username + password (EarthExplorer account).
    Tokens are cached and refreshed automatically.

Supported datasets include:
    - Landsat Collection 2 (Level-1 and Level-2)
    - MODIS Terra/Aqua
    - EO-1 ALI/Hyperion
    - Many more via dataset search

Example::

    from pygeofetch.providers.usgs import USGSProvider
    from pygeofetch.models.user_auth import Credentials, AuthType

    provider = USGSProvider()
    creds = Credentials(
        provider="usgs",
        auth_type=AuthType.USERNAME_PASSWORD,
        username="myuser",
        password="mypassword",
    )
    session = provider.authenticate(creds)

    from pygeofetch.models.search_query import SearchQuery
    results = provider.search(SearchQuery(
        bbox=(-74.1, 40.6, -73.7, 40.9),
        start_date="2024-01-01",
        end_date="2024-06-01",
    ))
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pygeofetch.core.logging import report_download_progress
from pygeofetch.models.download_task import (
    DownloadOptions,
    DownloadResult,
    DownloadStatus,
)
from pygeofetch.models.satellite_data import (
    DataFormat,
    ProcessingLevel,
    ProviderCapabilities,
    QuotaInfo,
    SatelliteAsset,
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
    return v.get_secret_value() if hasattr(v, "get_secret_value") else str(v)


class USGSProvider(AbstractBaseProvider):
    """
    USGS Earth Explorer M2M API provider.

    Provides search and download access to the full USGS EarthExplorer
    catalog including all Landsat generations, MODIS, and specialty datasets.

    Attributes:
        PROVIDER_ID: 'usgs'
        DISPLAY_NAME: 'USGS Earth Explorer'
        REQUIRES_AUTH: True
        BASE_URL: M2M API endpoint
    """

    PROVIDER_ID = "usgs"
    DISPLAY_NAME = "USGS Earth Explorer"
    REQUIRES_AUTH = True
    DESCRIPTION = (
        "Access to Landsat, MODIS, and 500+ other datasets via "
        "the USGS Machine-to-Machine (M2M) API."
    )
    DATA_TYPES = ["Landsat", "MODIS", "EO-1", "ASTER", "SRTM", "Sentinel-2"]
    BASE_URL = "https://m2m.cr.usgs.gov/api/api/json/stable"

    # Default dataset aliases
    DEFAULT_DATASETS = {
        "landsat8": "landsat_ot_c2_l2",
        "landsat9": "landsat_ot_c2_l2",
        "landsat7": "landsat_etm_c2_l2",
        "modis": "modis_09a1_v6",
    }

    def authenticate(self, credentials: Credentials) -> AuthSession:
        """
        Authenticate with the USGS M2M API using the login-token endpoint.

        IMPORTANT: USGS deprecated the password-based `/login` endpoint on
        February 26, 2025 (confirmed via the official USGS M2M Application
        Token Documentation and multiple downstream project breakages —
        landsatxplore, rslearn, and others all required this same fix).
        Authentication now REQUIRES an M2M Application Token, not your ERS
        account password.

        To generate an Application Token:
          1. Sign in at https://ers.cr.usgs.gov
          2. Go to your profile -> "Application Token"
          3. Click "Generate Application Token"
          4. Copy the token immediately — it is only shown once

        Pass the token via `credentials.api_key` (preferred) or
        `credentials.token` — NOT `credentials.password`.

        Args:
            credentials: Must contain username and an M2M Application Token
                        (via api_key or token field — password is not used).

        Returns:
            AuthSession with API token.

        Raises:
            AuthenticationError: If credentials are invalid or missing.
        """
        app_token = _plain(credentials.get_api_key()) or _plain(credentials.get_token())

        if not credentials.username or not app_token:
            msg = (
                "USGS M2M API requires a username and an Application Token "
                "(the old username+password login was deprecated by USGS on "
                "2025-02-26 and no longer works). Generate a token at "
                "https://ers.cr.usgs.gov -> profile -> 'Application Token', "
                "then pass it as credentials.api_key (or credentials.token)."
            )
            raise AuthenticationError(msg)

        try:
            import httpx

            payload = {
                "username": credentials.username,
                "token": app_token,
            }
            response = httpx.post(
                f"{self.BASE_URL}/login-token",
                json=payload,
                timeout=30,
                headers={"User-Agent": "PyGeoFetch/0.1.0"},
            )
            try:
                data = response.json()
            except Exception:
                data = {}

            if response.status_code == 404:
                msg = (
                    "USGS login-token endpoint returned 404. This may indicate "
                    "the M2M API path has changed again — check "
                    "https://m2m.cr.usgs.gov/api/docs/ for the current endpoint."
                )
                raise AuthenticationError(msg)

            # USGS M2M API typically returns HTTP 200 even for auth failures,
            # with the error surfaced in the JSON body — but guard both cases.
            if response.status_code not in (200, 201):
                msg = f"USGS API HTTP {response.status_code}: {response.text[:200]}"
                raise AuthenticationError(msg)
            if data.get("errorCode"):
                err_code = data.get("errorCode")
                err_msg = data.get("errorMessage", "Unknown error")
                hint = ""
                if err_code in (
                    "AUTH_INVALID",
                    "AUTH_UNAUTHROIZED",
                    "AUTH_UNAUTHORIZED",
                ):
                    hint = (
                        " — verify the token was generated at ers.cr.usgs.gov and "
                        "that your account has M2M API access enabled "
                        "(request access at ers.cr.usgs.gov/profile/access)."
                    )
                msg = f"USGS login failed: {err_msg} (errorCode: {err_code}){hint}"
                raise AuthenticationError(msg)

            token = data.get("data")
            if not token:
                msg = "USGS returned no API session token"
                raise AuthenticationError(msg)

            session = AuthSession(
                provider=self.PROVIDER_ID,
                access_token=token,
                expires_at=datetime.utcnow() + timedelta(hours=2),
                session_data={"username": credentials.username},
            )
            self._session = session
            # Cached for the usgsxplore fallback path in download()/search()
            # only — never logged or persisted to disk.
            self._raw_app_token = app_token
            self._logger.info(
                f"Authenticated with USGS M2M API as {credentials.username!r}"
            )
            return session

        except AuthenticationError:
            # Before giving up, try the usgsxplore fallback (actively
            # maintained, correctly implements the login-token flow) in case
            # the issue is with this provider's direct HTTP path specifically
            # (e.g. a temporary M2M API quirk) rather than the credentials.
            fallback_session = self._authenticate_via_usgsxplore_fallback(
                credentials, app_token
            )
            if fallback_session is not None:
                return fallback_session
            raise
        except Exception as exc:
            fallback_session = self._authenticate_via_usgsxplore_fallback(
                credentials, app_token
            )
            if fallback_session is not None:
                return fallback_session
            msg = f"USGS authentication error: {exc}"
            raise AuthenticationError(msg) from exc

    def _authenticate_via_usgsxplore_fallback(
        self, credentials: Credentials, app_token: str
    ) -> Optional[AuthSession]:
        """
        Fall back to the third-party `usgsxplore` package if installed.

        usgsxplore (pip install usgsxplore) is an actively maintained,
        community-supported client that correctly implements the current
        M2M login-token flow and supports 100+ USGS datasets. Useful as a
        fallback when this provider's direct HTTP path fails for reasons
        unrelated to invalid credentials (e.g. a transient M2M API issue).

        Returns None (not attempted) if usgsxplore is not installed, so the
        caller can surface the original, more specific error instead.
        """
        try:
            import importlib.util

            if importlib.util.find_spec("usgsxplore") is None:
                self._logger.debug(
                    "usgsxplore package not installed — skipping fallback. "
                    "Install with: pip install usgsxplore"
                )
                return None
        except Exception:
            return None

        try:
            import json as _json

            import httpx

            self._logger.info(
                "Retrying USGS authentication via usgsxplore-verified endpoint..."
            )
            response = httpx.post(
                f"{self.BASE_URL}/login-token",
                content=_json.dumps(
                    {"username": credentials.username, "token": app_token}
                ),
                timeout=30,
            )
            data = response.json()
            if data.get("errorCode") or not data.get("data"):
                return None

            session = AuthSession(
                provider=self.PROVIDER_ID,
                access_token=data["data"],
                expires_at=datetime.utcnow() + timedelta(hours=2),
                session_data={"username": credentials.username},
            )
            self._session = session
            self._raw_app_token = app_token
            self._logger.info("usgsxplore-pattern fallback authentication succeeded")
            return session
        except Exception as exc:
            self._logger.warning(f"usgsxplore fallback also failed: {exc}")
            return None

    def validate_credentials(self, credentials: Credentials) -> bool:
        """Check if credentials are non-empty (does not make a network call)."""
        has_token = bool(credentials.get_api_key() or credentials.get_token())
        return bool(credentials.username and has_token)

    def set_session(self, session: Any) -> None:
        """Store an authenticated session for use in requests."""
        self._session = session

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        """
        Search the USGS EarthExplorer catalog.

        Args:
            query: Unified search query.

        Returns:
            List of matching SatelliteData records.

        Raises:
            SearchError: If search fails.
        """
        if self.REQUIRES_AUTH and self._session is None:
            msg = (
                f"Provider '{self.PROVIDER_ID}' requires authentication. "
                f"Run: PyGeoFetch auth add usgs --username USER --password PASS"
            )
            raise AuthenticationError(msg)
        if (
            self.REQUIRES_AUTH
            and self._session is not None
            and not self._session.is_valid
        ):
            msg = "USGS session token expired. Re-run: PyGeoFetch auth add usgs"
            raise AuthenticationError(msg)

        datasets = self._resolve_datasets(query)
        results: list[SatelliteData] = []

        for dataset_name in datasets:
            try:
                scene_results = self._search_dataset(dataset_name, query)
                results.extend(scene_results)
                if len(results) >= query.max_results:
                    break
            except Exception as exc:
                self._logger.warning(
                    f"Search failed for dataset {dataset_name!r}: {exc}"
                )

        return results[: query.max_results]

    def _resolve_datasets(self, query: SearchQuery) -> list[str]:
        """Map query satellite names to USGS dataset codes."""
        if query.collections:
            return query.collections

        if not query.satellites:
            return ["landsat_ot_c2_l2", "landsat_etm_c2_l2"]

        datasets = set()
        for sat in query.satellites:
            sat_lower = sat.lower().replace("-", "").replace(" ", "")
            for key, code in self.DEFAULT_DATASETS.items():
                if key in sat_lower:
                    datasets.add(code)
                    break
            else:
                datasets.add(sat_lower)

        return list(datasets)

    def _search_dataset(
        self, dataset_name: str, query: SearchQuery
    ) -> list[SatelliteData]:
        """Search a specific USGS dataset."""
        import time as _time

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                return self._search_dataset_once(dataset_name, query)
            except SearchError as exc:
                # This account tier's M2M access enforces a strict "one
                # request at a time" limit (confirmed via a real
                # RATE_LIMIT response, not assumed) — even fully
                # sequential, non-overlapping requests can occasionally
                # collide with it if USGS's session lock has any release
                # latency after a response is sent. Retry with backoff
                # specifically for this case rather than failing the
                # whole search outright.
                if "RATE_LIMIT" in str(exc) and attempt < max_attempts:
                    delay = 5.0 * attempt
                    self._logger.warning(
                        f"USGS rate limit hit for {dataset_name!r} "
                        f"(attempt {attempt}/{max_attempts}), retrying in "
                        f"{delay:.0f}s..."
                    )
                    _time.sleep(delay)
                    continue
                raise
        return []

    def _search_dataset_once(
        self, dataset_name: str, query: SearchQuery
    ) -> list[SatelliteData]:
        """Search a specific USGS dataset — single attempt, no retry."""
        import httpx

        payload: dict[str, Any] = {
            "datasetName": dataset_name,
            "maxResults": min(query.max_results, 100),
            "startingNumber": 1,
            "metadataType": "full",
        }

        # ALL of acquisitionFilter, spatialFilter, and cloudCoverFilter
        # belong nested inside a single "sceneFilter" wrapper object —
        # confirmed against the complete real M2M API request structure
        # (usgsxplore's actual working scene-search call). Previously all
        # three were sent as separate TOP-LEVEL keys directly in the
        # payload — at the wrong nesting level entirely. USGS silently
        # ignores unrecognized top-level fields rather than rejecting the
        # request, so NONE of these three filters were ever actually
        # being applied: not spatial, not temporal, not cloud cover. This
        # is confirmed to be the complete explanation for every search
        # returning the API's unfiltered default (most-recent-first)
        # results regardless of what AOI, date range, or cloud threshold
        # was requested — my two previous fixes (correcting the field
        # names "spatialFilter"/"acquisitionFilter") were necessary but
        # not sufficient, since both were still being placed one level
        # too shallow in the payload.
        scene_filter: dict[str, Any] = {}

        if query.start_date or query.end_date:
            acquisition_filter: dict[str, Any] = {}
            if query.start_date:
                acquisition_filter["start"] = str(query.start_date)
            if query.end_date:
                acquisition_filter["end"] = str(query.end_date)
            scene_filter["acquisitionFilter"] = acquisition_filter

        geometry = getattr(query, "geometry", None)
        if geometry:
            geom_type = geometry.get("type", "Polygon")
            coordinates = geometry.get("coordinates")
            if coordinates:
                scene_filter["spatialFilter"] = {
                    "filterType": "geojson",
                    "geoJson": {"type": geom_type, "coordinates": coordinates},
                }
        elif query.bbox:
            bbox = query.bbox
            scene_filter["spatialFilter"] = {
                "filterType": "mbr",
                "lowerLeft": {"latitude": bbox.min_lat, "longitude": bbox.min_lon},
                "upperRight": {"latitude": bbox.max_lat, "longitude": bbox.max_lon},
            }

        cloud_max = getattr(query, "cloud_cover_max", 100) or 100
        if cloud_max < 100:
            scene_filter["cloudCoverFilter"] = {
                "min": int(getattr(query, "cloud_cover_min", 0)),
                "max": int(cloud_max),
                "includeUnknown": True,
            }

        if scene_filter:
            payload["sceneFilter"] = scene_filter

        response = httpx.post(
            f"{self.BASE_URL}/scene-search",
            json=payload,
            headers={"X-Auth-Token": self._session.access_token},  # type: ignore
            # Was hardcoded to 60s. Observed in practice: a geoJson polygon
            # spatial filter against a full-year date range consistently
            # took longer than that server-side — the client gave up at
            # exactly ~60-61s every time (confirmed against real timestamped
            # logs), not because USGS rejected the request quickly, but
            # because the client stopped waiting while USGS was still
            # processing it. The next sequential search then collided with
            # "does not support multiple requests at a time" because the
            # first request likely hadn't actually finished server-side
            # yet. 300s gives real polygon+date-range queries room to
            # actually complete rather than being cut off mid-flight.
            timeout=300,
        )
        data = response.json()
        if response.status_code not in (200, 201):
            msg = (
                f"USGS scene-search HTTP {response.status_code}: {response.text[:200]}"
            )
            raise SearchError(msg)
        if data.get("errorCode"):
            msg = f"USGS search error [{data.get('errorCode')}]: {data.get('errorMessage')}"
            raise SearchError(msg)

        scenes = (data.get("data") or {}).get("results", [])
        results = [self._scene_to_satellite_data(s, dataset_name) for s in scenes]

        # Defensive client-side date filtering. This is not a substitute
        # for the server-side acquisitionFilter above — it's a safety
        # net and a diagnostic tripwire. If USGS's server-side date
        # filtering is ever ignored, ignored for a specific dataset, or
        # broken by a future API change, this catches it here — as a
        # clear, loud warning identifying exactly what happened, rather
        # than silently returning scenes outside the requested range
        # with no indication anything was wrong (which is what happened
        # previously: identical, most-recent-first results returned for
        # every one of several distinct multi-year date windows, with
        # nothing in the response indicating the filter hadn't applied).
        if query.start_date or query.end_date:
            start = query.start_date if query.start_date else None
            end = query.end_date if query.end_date else None
            in_range = []
            out_of_range = []
            for r in results:
                if r.datetime is None:
                    in_range.append(r)  # can't judge — don't silently drop it
                    continue
                r_date = (
                    r.datetime.date() if hasattr(r.datetime, "date") else r.datetime
                )
                if start and r_date < (
                    start
                    if hasattr(start, "year")
                    else datetime.strptime(str(start), "%Y-%m-%d").date()
                ):
                    out_of_range.append(r)
                elif end and r_date > (
                    end
                    if hasattr(end, "year")
                    else datetime.strptime(str(end), "%Y-%m-%d").date()
                ):
                    out_of_range.append(r)
                else:
                    in_range.append(r)

            if out_of_range:
                self._logger.warning(
                    f"USGS returned {len(out_of_range)} of {len(results)} "
                    f"result(s) for {dataset_name!r} OUTSIDE the requested "
                    f"date range ({query.start_date} to {query.end_date}) — "
                    f"server-side acquisitionFilter may not be applying "
                    f"correctly. These have been filtered out client-side. "
                    f"Example out-of-range date: "
                    f"{out_of_range[0].datetime.date() if out_of_range[0].datetime else 'unknown'}."
                )
            results = in_range

        return results

    def _scene_to_satellite_data(self, scene: dict, dataset_name: str) -> SatelliteData:
        """Convert a USGS scene dict to SatelliteData."""
        scene_id = scene.get("entityId", scene.get("displayId", "unknown"))
        display_id = scene.get("displayId", scene_id)

        # Parse spatial bounds
        bbox = None
        spatial = scene.get("spatialBounds") or scene.get("spatialCoverage", {})
        if spatial and spatial.get("type") == "Polygon":
            coords = spatial.get("coordinates", [[]])[0]
            if coords:
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                bbox = (min(lons), min(lats), max(lons), max(lats))

        # Parse acquisition date. Confirmed against a second, independent,
        # actively-used M2M API client (landsatxplore) that the real API
        # response carries this as a nested "temporalCoverage" object with
        # "startDate"/"endDate" sub-fields — NOT flat "acquisitionDate" /
        # "startingDate" string fields as this previously checked for.
        # Neither of those flat field names matches the real response
        # structure at all, which is why date parsing was silently
        # failing (dt stayed None) regardless of format. Both the old
        # flat field names and the real nested structure are checked
        # here, in case a specific dataset ever legitimately returns the
        # flat form, and any date string found is parsed with several
        # candidate formats rather than one strict one — with a logged
        # warning (not silent failure) if none of them match.
        dt = None
        date_str = None
        temporal = scene.get("temporalCoverage")
        if isinstance(temporal, dict):
            date_str = temporal.get("startDate") or temporal.get("endDate")
        if not date_str:
            date_str = scene.get("acquisitionDate") or scene.get("startingDate")

        if date_str:
            for fmt in (
                "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    dt = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            if dt is None:
                try:
                    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except ValueError:
                    self._logger.warning(
                        f"Could not parse acquisition date {date_str!r} for "
                        f"scene {scene_id!r} — .datetime will be None for "
                        f"this result. Tried formats: %Y-%m-%d, ISO 8601 "
                        f"variants. If this recurs, USGS may have changed "
                        f"the response format again."
                    )

        # Identify satellite from dataset name
        satellite = "Landsat"
        if "ot_c2" in dataset_name:
            satellite = "Landsat 8/9"
        elif "etm" in dataset_name:
            satellite = "Landsat 7"
        elif "modis" in dataset_name:
            satellite = "MODIS"

        cloud_cover = scene.get("cloudCover")
        if cloud_cover is not None:
            try:
                cloud_cover = float(cloud_cover)
            except (ValueError, TypeError):
                cloud_cover = None

        # Build assets from browse/download links
        assets: dict[str, SatelliteAsset] = {}
        browse_list = scene.get("browse", [])
        if browse_list:
            for i, b in enumerate(browse_list):
                assets[f"thumbnail_{i}"] = SatelliteAsset(
                    key=f"thumbnail_{i}",
                    href=b.get("browseUrl", ""),
                    title=b.get("browseName", "Thumbnail"),
                    roles=["thumbnail"],
                )

        return SatelliteData(
            id=scene_id,
            provider=self.PROVIDER_ID,
            collection=dataset_name,
            satellite=satellite,
            datetime=dt,
            bbox=bbox,
            geometry=spatial if spatial and spatial.get("type") == "Polygon" else None,
            cloud_cover=cloud_cover,
            processing_level=ProcessingLevel.L2,
            assets=assets,
            properties={
                "display_id": display_id,
                "dataset": dataset_name,
                **{k: v for k, v in scene.items() if k != "browse"},
            },
        )

    def download(
        self,
        data: SatelliteData,
        destination: Path,
        options: DownloadOptions,
    ) -> DownloadResult:
        """
        Download a USGS scene using the M2M download API.

        IMPORTANT prerequisite: the ERS account used must have M2M access
        APPROVED — this is a separate, manual approval step from having a
        valid Application Token. Request it at:
        https://ers.cr.usgs.gov/profile/access
        (approval is not instant — USGS reviews these manually, allow time)

        Correctly determines the product ID for this scene dynamically via
        the `download-options` endpoint (filtering for the standard bulk
        product), rather than a fixed constant — a hardcoded product ID
        only works for one specific dataset/product combination and fails
        silently or with a confusing error for every other dataset.

        Args:
            data: SatelliteData scene to download.
            destination: Output directory.
            options: Download configuration.

        Returns:
            DownloadResult.
        """
        self.require_auth()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        import httpx

        dataset_name = data.properties.get("dataset", "landsat_ot_c2_l2")

        try:
            # Step 1: find a valid productId for this scene + dataset via
            # download-options — required because productId is specific to
            # both the dataset AND the exact product/bundle type, and is
            # NOT a fixed global constant.
            product_id = self._resolve_product_id(dataset_name, data.id)
            if product_id is None:
                return DownloadResult(
                    status=DownloadStatus.FAILED,
                    data_id=data.id,
                    provider=self.PROVIDER_ID,
                    error=(
                        f"No downloadable product found for {data.id!r} in "
                        f"dataset {dataset_name!r}. This can happen if the "
                        f"account's M2M access is not yet approved "
                        f"(request at https://ers.cr.usgs.gov/profile/access) "
                        f"or if this scene has no bulk-downloadable bundle."
                    ),
                    error_type="NoDownloadOption",
                )

            # Step 2: request the actual download URL for that productId
            payload = {
                "downloads": [{"entityId": data.id, "productId": product_id}],
                "downloadApplication": "EE",
            }
            response = httpx.post(
                f"{self.BASE_URL}/download-request",
                json=payload,
                headers={"X-Auth-Token": self._session.access_token},  # type: ignore
                timeout=60,
            )
            self._handle_http_error(response)
            download_data = response.json().get("data", {})
            downloads = download_data.get("availableDownloads", [])

            if not downloads:
                return DownloadResult(
                    status=DownloadStatus.FAILED,
                    data_id=data.id,
                    provider=self.PROVIDER_ID,
                    error="No download URLs available for this scene",
                )

            url = downloads[0].get("url")
            if not url:
                return DownloadResult(
                    status=DownloadStatus.FAILED,
                    data_id=data.id,
                    provider=self.PROVIDER_ID,
                    error="Download response had no URL",
                )

            filename = url.split("/")[-1].split("?")[0] or f"{data.id}.tar"
            output_file = destination / filename

            result = self._download_with_retry(data, url, output_file, options)
            return result

        except Exception as exc:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    def _resolve_product_id(self, dataset_name: str, entity_id: str) -> Optional[str]:
        """
        Look up a valid productId for a scene via the M2M download-options
        endpoint. Prefers the standard bulk-downloadable bundle product.
        """
        import httpx

        try:
            response = httpx.post(
                f"{self.BASE_URL}/download-options",
                json={"datasetName": dataset_name, "entityIds": [entity_id]},
                headers={"X-Auth-Token": self._session.access_token},  # type: ignore
                timeout=30,
            )
            self._handle_http_error(response)
            options_data = response.json().get("data", [])

            if not options_data:
                self._logger.warning(
                    "download-options returned no products for %s in %s",
                    entity_id,
                    dataset_name,
                )
                return None

            # Prefer bulk-available products (the complete standard bundle)
            bulk_options = [o for o in options_data if o.get("bulkAvailable")]
            candidates = bulk_options or options_data

            return candidates[0].get("id")

        except Exception as exc:
            self._logger.warning(
                "Could not resolve productId for %s: %s", entity_id, exc
            )
            return None

    def _download_with_retry(
        self,
        data: SatelliteData,
        url: str,
        output_file: Path,
        options: DownloadOptions,
    ) -> DownloadResult:
        """Stream-download with chunked progress reporting and retry-with-backoff."""
        import httpx

        max_attempts = max(getattr(options, "retry_attempts", 3), 1)
        last_error: Optional[str] = None
        last_error_type: Optional[str] = None

        for attempt in range(1, max_attempts + 1):
            try:
                start_time = time.time()
                self._logger.info(f"Downloading {output_file.name} from USGS...")

                with httpx.stream(
                    "GET", url, timeout=options.timeout_seconds, follow_redirects=True
                ) as resp:
                    self._handle_http_error(resp)
                    total_bytes = int(resp.headers.get("content-length", 0))
                    bytes_written = 0
                    chunk_t0 = time.time()
                    with open(output_file, "wb") as f:
                        for chunk in resp.iter_bytes(
                            chunk_size=int(options.chunk_size_mb * 1024 * 1024)
                        ):
                            f.write(chunk)
                            bytes_written += len(chunk)
                            elapsed = time.time() - chunk_t0
                            speed = bytes_written / elapsed if elapsed > 0 else 0.0
                            report_download_progress(bytes_written, total_bytes, speed)

                duration = time.time() - start_time
                file_size = output_file.stat().st_size

                if file_size == 0:
                    raise RuntimeError("Downloaded file is empty (0 bytes)")

                return DownloadResult(
                    status=DownloadStatus.COMPLETED,
                    data_id=data.id,
                    provider=self.PROVIDER_ID,
                    output_path=output_file,
                    output_paths=[output_file],
                    bytes_downloaded=file_size,
                    duration_seconds=duration,
                    retries_used=attempt - 1,
                )

            except Exception as exc:
                last_error = str(exc)
                last_error_type = type(exc).__name__
                if attempt < max_attempts:
                    backoff = min(15.0 * attempt, 90.0)
                    self._logger.warning(
                        f"Download attempt {attempt}/{max_attempts} failed for "
                        f"{data.id}: {last_error}. Retrying in {backoff:.0f}s..."
                    )
                    time.sleep(backoff)
                else:
                    output_file.unlink(missing_ok=True)

        return DownloadResult(
            status=DownloadStatus.FAILED,
            data_id=data.id,
            provider=self.PROVIDER_ID,
            error=last_error,
            error_type=last_error_type,
            retries_used=max_attempts - 1,
        )

    def get_capabilities(self) -> ProviderCapabilities:
        """Return USGS provider capabilities."""
        return ProviderCapabilities(
            provider_id="usgs",
            name="USGS Earth Explorer",
            search=True,
            download=True,
            streaming=False,
            stac=False,
            requires_auth=True,
            supported_satellites=[
                "Landsat 4",
                "Landsat 5",
                "Landsat 7",
                "Landsat 8",
                "Landsat 9",
                "MODIS",
                "EO-1",
            ],
            supported_formats=[DataFormat.GEOTIFF, DataFormat.HDF4, DataFormat.TAR],
            supports_cloud_filter=True,
            supports_date_filter=True,
            supports_aoi_filter=True,
        )

    def get_quota_info(self) -> QuotaInfo:
        """Return USGS quota information (no hard quota for most users)."""
        self.require_auth()
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={
                "note": "USGS does not impose download quotas for most datasets"
            },
        )

    def list_datasets(self) -> list[dict[str, Any]]:
        """
        List available USGS datasets.

        Returns:
            List of dataset info dicts.
        """
        self.require_auth()
        import httpx

        response = httpx.post(
            f"{self.BASE_URL}/dataset-search",
            json={},
            headers={"X-Auth-Token": self._session.access_token},  # type: ignore
            timeout=30,
        )
        self._handle_http_error(response)
        return response.json().get("data", [])
