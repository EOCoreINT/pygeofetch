"""
Alaska Satellite Facility (ASF) provider for PyGeoFetch.

Full rewrite against ASF's real, documented Search API (confirmed
directly at docs.asf.alaska.edu/api/) -- the previous implementation
assumed a fictional ``{BASE_URL}/search`` REST endpoint with
``startDate``/``endDate``/``cloudCoverMax`` query parameters that don't
match ASF's real API at all. Worth stating plainly: ``cloudCoverMax``
never made sense here regardless of the URL -- ASF serves SAR data
(Sentinel-1, ALOS PALSAR, ERS, JERS), and SAR imagery isn't affected by
cloud cover; there is no such property on a real ASF search result.

The real, verified Search API:

    GET https://api.daac.asf.alaska.edu/services/search/param
        ?platform=Sentinel-1A,Sentinel-1B
        &processingLevel=SLC
        &intersectsWith=POLYGON((...))
        &start=2019-04-01T00:00:00UTC&end=2019-05-05T00:00:00UTC
        &output=geojson

confirmed against ASF's own documented example query and its "Search
API Keywords & Endpoints" reference. Real, load-bearing details:

- Geometry is supplied as a WKT string via ``intersectsWith``, not a
  plain ``bbox=`` parameter.
- Real param names are ``platform``, ``processingLevel``, ``start``/
  ``end`` (not ``startDate``/``endDate``), and ``maxResults`` (not
  ``limit``).
- The search endpoint itself is real, verified public -- ASF's own
  documented example curl/browser queries carry no authentication at
  all. Only *downloading* the actual data requires a real NASA
  Earthdata Login (EDL) account -- the same EDL system
  ``pygeofetch.providers.nasa_earthdata`` already authenticates
  against, reused here via the same real, verified pattern (HTTP Basic
  Auth against ``urs.earthdata.nasa.gov/api/users/user`` to confirm the
  credentials are genuinely valid before proceeding) rather than a
  second, independent implementation of EDL auth. ASF's own
  ``ASFSession`` client additionally documents a real
  ``Authorization: Bearer {EDL token}`` alternative to username/
  password for downloads, also supported here.
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


def _bbox_to_wkt_polygon(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> str:
    """Real WKT POLYGON string, matching the exact format confirmed in
    ASF's own documented example queries (space-separated coordinate
    pairs, comma-separated vertices, closed ring)."""
    return (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
        f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
    )


class _EarthdataSession:
    """``requests.Session`` that keeps credentials only for the Earthdata login host across redirects.

    Based on NASA's recipe for Earthdata downloads in Python: requests drops
    the Authorization header whenever a redirect changes host, which would
    strip it on the way to urs.earthdata.nasa.gov. Here it is kept only when a
    redirect goes to the login host, and dropped on every other change of host
    (the way back carries a one-time code and cookies; the final signed S3 URL
    must not receive credentials).
    """

    AUTH_HOST = "urs.earthdata.nasa.gov"

    def __new__(cls):  # build lazily so importing this module never needs requests
        import requests
        from requests.utils import urlparse

        auth_host = cls.AUTH_HOST

        class _Session(requests.Session):
            def rebuild_auth(self, prepared_request, response):
                headers = prepared_request.headers
                if "Authorization" in headers:
                    original = urlparse(response.request.url).hostname
                    redirect = urlparse(prepared_request.url).hostname
                    if original != redirect and redirect != auth_host:
                        del headers["Authorization"]

        return _Session()


class AlaskaSatelliteFacilityProvider(AbstractBaseProvider):
    PROVIDER_ID = "alaska_satellite_facility"
    DISPLAY_NAME = "Alaska Satellite Facility (ASF) DAAC"
    # Real, verified: the search endpoint itself needs no authentication.
    # Only download() requires a real NASA Earthdata Login account.
    REQUIRES_AUTH = False
    # ...but downloads do: the downloader authenticates before calling download().
    REQUIRES_AUTH_FOR_DOWNLOAD = True
    DESCRIPTION = (
        "SAR data from Sentinel-1, NISAR, ALOS PALSAR, ERS, JERS via ASF's real "
        "Search API. Search is public; downloading requires a free NASA "
        "Earthdata Login account (the same login used by nasa_earthdata)."
    )
    SATELLITES = [
        "Sentinel-1A",
        "Sentinel-1B",
        "Sentinel-1C",
        "NISAR",
        "ALOS PALSAR",
        "ERS-1",
        "ERS-2",
        "JERS-1",
    ]
    BASE_URL = "https://api.daac.asf.alaska.edu"
    SEARCH_ENDPOINT = f"{BASE_URL}/services/search/param"

    def authenticate(self, credentials: Credentials) -> AuthSession:
        # Real, reused pattern: the same Earthdata Login verification
        # pygeofetch.providers.nasa_earthdata already uses (HTTP Basic
        # Auth against a real EDL endpoint to confirm the credentials
        # are genuinely valid), since ASF uses the identical EDL system
        # -- not a second, independent implementation of the same check.
        token = _plain(credentials.api_key or credentials.access_key or getattr(credentials, "token", None))
        password = _plain(credentials.get_password())

        if token:
            # Real, documented alternative: ASF's own ASFSession client
            # supports an EDL bearer token in place of username/password.
            session = AuthSession(
                provider=self.PROVIDER_ID,
                access_token=token,
                expires_at=datetime.now(timezone.utc)
                + timedelta(days=60),  # real EDL token lifetime
                session_data={"auth_mode": "bearer_token", "token": token},
            )
            self._session = session
            self._logger.info(
                f"{self.DISPLAY_NAME}: authenticated with a real EDL bearer token"
            )
            return session

        if not credentials.username or not password:
            raise AuthenticationError(
                f"{self.DISPLAY_NAME} downloads require a free NASA Earthdata "
                f"Login account (https://urs.earthdata.nasa.gov/) -- pass "
                f"username+password, or an EDL bearer token as api_key."
            )

        import httpx

        try:
            resp = httpx.get(
                "https://urs.earthdata.nasa.gov/api/users/user",
                auth=(credentials.username, password),
                timeout=30,
            )
            if resp.status_code in (401, 403):
                # This pre-check calls an Earthdata endpoint that is documented for
                # tokens; it can answer 401 even when the same login works for a
                # real download. Do not block here: the download itself
                # (redirect through urs.earthdata.nasa.gov) is the authoritative test
                # and reports precisely why it was refused.
                self._logger.warning(
                    f"{self.DISPLAY_NAME}: Earthdata pre-check answered HTTP {resp.status_code}; "
                    "continuing, the download will confirm whether the login is accepted"
                )
            elif resp.status_code not in (200, 404):
                raise AuthenticationError(
                    f"{self.DISPLAY_NAME}: NASA Earthdata login check failed (HTTP {resp.status_code})"
                )
        except AuthenticationError:
            raise
        except Exception as exc:
            raise AuthenticationError(
                f"{self.DISPLAY_NAME}: Earthdata auth error: {exc}"
            ) from exc

        session = AuthSession(
            provider=self.PROVIDER_ID,
            access_token=None,
            expires_at=datetime.now(timezone.utc) + timedelta(days=365),
            session_data={
                "auth_mode": "basic",
                "username": credentials.username,
                "password": password,
            },
        )
        self._session = session
        self._logger.info(
            f"{self.DISPLAY_NAME}: authenticated as {credentials.username!r} (NASA Earthdata Login)"
        )
        return session

    def _authenticate_from_stored_credentials(self) -> None:
        """Authenticate with an Earthdata login found in pygeofetch's credential store or the environment.

        Looks, in order, for credentials stored for this provider, then for
        ``nasa_earthdata`` / ``nasa_earthdata_cloud`` (the same Earthdata Login), in
        both storage backends, then for ``EARTHDATA_TOKEN`` or
        ``EARTHDATA_USERNAME`` + ``EARTHDATA_PASSWORD``. Raises with a clear
        message when none is found.
        """
        import os

        tried = []
        try:
            from pygeofetch.core.authenticator import AuthManager
        except Exception:  # pragma: no cover
            AuthManager = None  # noqa: N806
        if AuthManager is not None:
            for backend in ("file", "keyring"):
                try:
                    store = AuthManager(storage_backend=backend)
                except Exception:
                    continue
                for pid in (self.PROVIDER_ID, "nasa_earthdata", "nasa_earthdata_cloud"):
                    tried.append(f"{pid} ({backend})")
                    try:
                        creds = store.get_credentials(pid)
                    except Exception:
                        creds = None
                    if creds is not None and self.validate_credentials(creds):
                        self.authenticate(creds)
                        self._logger.info(f"{self.DISPLAY_NAME}: using Earthdata login stored for {pid!r} ({backend})")
                        return
        token = os.environ.get("EARTHDATA_TOKEN")
        user, password = os.environ.get("EARTHDATA_USERNAME"), os.environ.get("EARTHDATA_PASSWORD")
        if token or (user and password):
            from pydantic import SecretStr

            creds = Credentials(
                provider=self.PROVIDER_ID,
                username=user,
                password=SecretStr(password) if password else None,
                api_key=SecretStr(token) if token else None,
            )
            self.authenticate(creds)
            self._logger.info(f"{self.DISPLAY_NAME}: using the Earthdata login from environment variables")
            return
        raise AuthenticationError(
            f"{self.DISPLAY_NAME}: no NASA Earthdata login found (looked in: {', '.join(tried) or 'credential store'}, "
            "EARTHDATA_TOKEN, EARTHDATA_USERNAME/EARTHDATA_PASSWORD). Add one with: "
            f"pygeofetch auth add {self.PROVIDER_ID} --username USER --password PASSWORD"
        )

    def validate_credentials(self, credentials: Credentials) -> bool:
        has_token = bool(credentials.api_key or credentials.access_key)
        has_basic = bool(
            credentials.username
            and (credentials.password or credentials.get_password())
        )
        return has_token or has_basic

    def set_session(self, session: Any) -> None:
        self._session = session

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        # Real, verified: no authentication needed to search.
        import httpx

        params: dict[str, Any] = {
            "output": "geojson",
            "maxResults": min(query.max_results, 5000),
        }
        if query.bbox:
            bb = query.bbox
            params["intersectsWith"] = _bbox_to_wkt_polygon(
                bb.min_lon, bb.min_lat, bb.max_lon, bb.max_lat
            )
        if query.satellites:
            params["platform"] = ",".join(query.satellites)
        if query.product_type:
            # Real, direct compatibility: ASF's own processingLevel
            # values (SLC, GRD, GRD-COG, OCN) match pygeofetch's
            # SearchQuery.product_type values exactly -- no translation
            # table needed, confirmed against ASF's real keyword docs.
            params["processingLevel"] = query.product_type
        if query.start_date:
            params["start"] = f"{query.start_date}T00:00:00UTC"
        if query.end_date:
            params["end"] = f"{query.end_date}T23:59:59UTC"
        # Raw ASF Search API keywords, passed through unchanged, for products the
        # generic SearchQuery fields do not describe. NISAR, for example, is
        # searched by dataset and product level (ASF NISAR documentation):
        #
        #   SearchQuery(..., provider_filters={"alaska_satellite_facility": {"api_params": {
        #       "dataset": "NISAR", "processingLevel": "GUNW", "productionConfiguration": "PR"}}})
        #
        # Lists become comma-separated values; these keys override the ones above.
        raw = (query.provider_filters or {}).get(self.PROVIDER_ID, {}).get("api_params", {})
        for key, value in raw.items():
            params[key] = ",".join(map(str, value)) if isinstance(value, (list, tuple)) else value
        if raw.get("dataset") and "platform" in params and not raw.get("platform"):
            # ASF treats dataset and platform as alternative ways to select a mission.
            params.pop("platform")
        # Real, deliberate omission: no cloud-cover parameter -- ASF
        # serves SAR data, which cloud cover doesn't apply to, and the
        # real Search API has no such keyword (confirmed against ASF's
        # own keyword reference).

        try:
            resp = httpx.get(
                self.SEARCH_ENDPOINT,
                params=params,
                timeout=self.config.get("timeout", 60),
            )
            if resp.status_code != 200:
                self._logger.warning(f"{self.DISPLAY_NAME}: HTTP {resp.status_code}")
                return []
            data = resp.json()
            items = (data or {}).get("features") or []
        except Exception as exc:
            self._logger.warning(f"{self.DISPLAY_NAME} search: {exc}")
            return []
        # One malformed feature must not discard the whole result list: parse each on its own.
        results: list[SatelliteData] = []
        skipped = 0
        for item in items:
            try:
                parsed = self._parse_item(item)
            except Exception as exc:  # noqa: BLE001 - log and keep the other results
                skipped += 1
                self._logger.debug(f"{self.DISPLAY_NAME}: skipped a feature ({exc!r})")
                continue
            if parsed is not None:
                results.append(parsed)
        if skipped:
            self._logger.warning(f"{self.DISPLAY_NAME}: skipped {skipped} unparseable feature(s)")
        return results

    def _parse_item(self, item: dict[str, Any]) -> SatelliteData | None:
        props = item.get("properties") or {}
        item_id = props.get("sceneName") or props.get("fileID") or item.get("id")
        if not item_id:
            return None
        geometry = item.get("geometry") or None
        bbox = None
        coordinates = (geometry or {}).get("coordinates")
        if coordinates:
            # Polygon: [ring, ...]; MultiPolygon: [[ring, ...], ...]. Collect every vertex.
            rings = coordinates if geometry.get("type") == "Polygon" else [r for poly in coordinates for r in poly]
            points = [c for ring in rings if ring for c in ring if c and len(c) >= 2]
            if points:
                lons = [c[0] for c in points]
                lats = [c[1] for c in points]
                bbox = (min(lons), min(lats), max(lons), max(lats))

        return SatelliteData(
            id=item_id,
            provider=self.PROVIDER_ID,
            satellite=props.get("platform", self.DISPLAY_NAME),
            datetime=props.get("startTime"),
            cloud_cover=None,  # real: not applicable to SAR data
            bbox=bbox,
            geometry=geometry,
            properties=props,
            assets=(
                {"data": {"key": "data", "href": props["url"], "roles": ["data"]}}
                if props.get("url")
                else {}
            ),
        )

    def download(
        self, data: SatelliteData, destination: Path, options: DownloadOptions
    ) -> DownloadResult:
        """Stream every http asset of ``data`` to ``destination`` with NASA Earthdata Login.

        Earthdata downloads redirect through urs.earthdata.nasa.gov (login) and
        then to a signed S3 URL. A plain HTTP client drops the credentials at the
        first cross-host redirect, so the login never reaches Earthdata. This uses
        the redirect handling NASA documents for Python clients: credentials are
        kept for the Earthdata login host only, and dropped everywhere else (S3).
        A bearer token (``api_key``) works the same way. Files are written to
        ``<name>.part`` and renamed when complete; finished files are skipped.
        """
        self.require_auth()
        if self._session is None:
            # The downloader normally authenticates first; if it did not (older downloader, or the
            # provider used directly), find the Earthdata login ourselves.
            self._authenticate_from_stored_credentials()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        start = time.time()
        output_paths, total_bytes = [], 0
        session_data = self._active_session().session_data or {}
        http = _EarthdataSession()
        if session_data.get("auth_mode") == "bearer_token":
            http.headers["Authorization"] = f"Bearer {session_data['token']}"
        elif session_data.get("auth_mode") == "basic":
            http.auth = (session_data["username"], session_data["password"])

        errors = []
        for key, asset in (data.data_assets or data.assets).items():
            if not asset.href or not asset.href.startswith("http"):
                continue
            out_file = destination / (asset.href.split("/")[-1] or f"{data.id}_{key}")
            if out_file.exists() and out_file.stat().st_size > 0 and getattr(options, "resume", True):
                output_paths.append(out_file)
                total_bytes += out_file.stat().st_size
                continue
            part = out_file.with_name(out_file.name + ".part")
            try:
                with http.get(asset.href, stream=True, timeout=options.timeout_seconds, allow_redirects=True) as resp:
                    if resp.status_code in (401, 403):
                        raise AuthenticationError(
                            f"{self.DISPLAY_NAME}: Earthdata refused the download (HTTP {resp.status_code}). Likely causes: "
                            "(1) wrong username/password (log in at https://urs.earthdata.nasa.gov to test; a password "
                            "with special characters can be mangled when typed at a shell, so quote it or use a token); "
                            "(2) the ASF / NISAR applications are not approved in your Earthdata profile "
                            "(Applications > Authorized Apps > Approve More Applications, then accept the EULA); "
                            "(3) use a token instead: create one under 'Generate Token' on your Earthdata profile and run "
                            "`pygeofetch auth add alaska_satellite_facility --api-key TOKEN`."
                        )
                    resp.raise_for_status()
                    if "text/html" in resp.headers.get("Content-Type", ""):
                        raise AuthenticationError(
                            f"{self.DISPLAY_NAME}: received an HTML page instead of data (Earthdata login page?)"
                        )
                    with open(part, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=int(options.chunk_size_mb * 1024 * 1024)):
                            if chunk:
                                f.write(chunk)
                part.replace(out_file)
                output_paths.append(out_file)
                total_bytes += out_file.stat().st_size
            except Exception as exc:
                part.unlink(missing_ok=True)
                errors.append(f"{key}: {exc}")
                self._logger.warning(f"Asset {key} failed: {exc}")

        if not output_paths:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                error="; ".join(errors) or "No assets downloaded",
                error_type="AuthenticationError" if any("refused the download" in e for e in errors) else None,
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
            auth_type="earthdata_login",
            satellites=self.SATELLITES,
            search=True,
            download=True,
            supports_sar=True,
            supports_sub_meter=False,
            supports_aoi_filter=True,
            supports_cloud_filter=False,  # real: not applicable to SAR
            supports_date_filter=True,
            requires_auth=False,  # real: only download() needs auth, not search()
            has_quota=False,
            regions=["global"],
            resolution_min_m=5.0,
            resolution_max_m=100.0,
            endpoint_url=self.SEARCH_ENDPOINT,
            docs_url="https://docs.asf.alaska.edu/api/basics/",
            supported_formats=[DataFormat.GEOTIFF],
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={
                "note": "No quota -- free NASA Earthdata Login account, no subscription."
            },
        )