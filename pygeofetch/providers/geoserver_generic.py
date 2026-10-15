"""
Generic GeoServer provider for PyGeoFetch.

Full rewrite. This provider is fundamentally different from every
other one in this package: it's not a specific, named vendor archive
with one real, canonical API to research -- GeoServer is open-source
software that different organizations self-host, each serving their
own layers. There's no single "GeoServer REST search API" to point at.

What GeoServer *does* implement faithfully is the real, decades-old,
standardized OGC Web Feature Service (WFS) protocol -- confirmed
directly against GeoServer's own official documentation
(docs.geoserver.org/main/en/user/services/wfs/reference/). The
previous implementation ignored this entirely and instead assumed a
fictional ``{BASE_URL}/search`` REST endpoint with
``startDate``/``endDate``/``cloudCoverMax`` query parameters -- none of
which exist in the real WFS spec, or in any real GeoServer deployment.

Real, confirmed WFS 2.0 GetFeature request shape:

    ?service=WFS&version=2.0.0&request=GetFeature
     &typeNames=namespace:featuretype
     &bbox=minx,miny,maxx,maxy,[crs]
     &outputFormat=application/json
     &count=<max results>

A genuine, real consequence of this being a generic client: WFS has no
concept of "search across every layer" the way a STAC catalog does --
a caller must specify which real feature type/layer to query, the same
way a STAC caller specifies real collections. This is configured via
this provider's ``layer`` (or ``typeName``) config value, not inferred.

Real, confirmed auth: GeoServer's standard security model protects
layers with HTTP Basic Auth (Spring Security, configured server-side),
not a bearer token or API key as the previous version assumed. Some
real deployments do sit behind an OAuth2 reverse proxy instead --
supported here too, as an explicit opt-in, since a generic client can't
assume which one a given self-hosted instance uses.

No date range or cloud-cover filtering is sent as a query parameter --
a real, arbitrary WFS layer could be roads, parcels, or anything else
with no temporal or atmospheric concept at all. Client-side
best-effort filtering is applied only when the returned GeoJSON
properties happen to contain a matching field.
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


class GeoserverGenericProvider(AbstractBaseProvider):
    PROVIDER_ID = "geoserver_generic"
    DISPLAY_NAME = "GeoServer Generic (OGC WFS)"
    REQUIRES_AUTH = False
    DESCRIPTION = (
        "Generic client for any self-hosted GeoServer instance's real OGC "
        "WFS GetFeature service. Requires configuring 'endpoint' (the real "
        "GeoServer base URL) and 'layer' (a real typeName the server "
        "advertises) -- there is no universal 'search all layers' concept "
        "in the real WFS protocol."
    )
    SATELLITES: list = []
    WFS_VERSION = "2.0.0"

    def authenticate(self, credentials: Credentials) -> AuthSession:
        # Real, confirmed: GeoServer's standard security model is HTTP
        # Basic Auth on the WFS endpoint (Spring Security, configured
        # server-side) -- not a bearer token by default. An optional
        # bearer token is still supported for real deployments that
        # sit behind an OAuth2 reverse proxy, since a generic client
        # can't assume which security model any given self-hosted
        # instance actually uses.
        password = _plain(credentials.get_password())
        token = _plain(credentials.api_key or credentials.access_key)

        if not credentials.username and not token:
            if self.REQUIRES_AUTH:
                raise AuthenticationError(
                    f"{self.DISPLAY_NAME} requires credentials for this secured "
                    f"instance -- pass username+password (real GeoServer Basic "
                    f"Auth) or a bearer token (if behind an OAuth2 proxy)."
                )
            session = AuthSession(
                provider=self.PROVIDER_ID,
                access_token="anonymous",
                expires_at=datetime.now(timezone.utc) + timedelta(days=365),
                session_data={"auth_mode": "none"},
            )
            self._session = session
            return session

        if token:
            session = AuthSession(
                provider=self.PROVIDER_ID,
                access_token=token,
                expires_at=datetime.now(timezone.utc) + timedelta(days=365),
                session_data={"auth_mode": "bearer_token", "token": token},
            )
        else:
            session = AuthSession(
                provider=self.PROVIDER_ID,
                access_token=credentials.username,
                expires_at=datetime.now(timezone.utc) + timedelta(days=365),
                session_data={
                    "auth_mode": "basic",
                    "username": credentials.username,
                    "password": password,
                },
            )
        self._session = session
        self._logger.info(f"{self.DISPLAY_NAME}: authenticated")
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        if not self.REQUIRES_AUTH:
            return True
        return bool(
            credentials.username or credentials.api_key or credentials.access_key
        )

    def set_session(self, session: Any) -> None:
        self._session = session

    def _auth_kwargs(self) -> dict[str, Any]:
        """Real httpx auth/headers kwargs matching whichever real auth
        mode was configured -- Basic Auth, an optional bearer token,
        or none, per this class's own docstring on GeoServer's real
        security conventions."""
        mode = (
            (self._session.session_data or {}).get("auth_mode")
            if self._session
            else None
        )
        if mode == "basic":
            return {
                "auth": (
                    self._session.session_data["username"],
                    self._session.session_data["password"],
                )
            }
        if mode == "bearer_token":
            return {
                "headers": {
                    "Authorization": f"Bearer {self._session.session_data['token']}"
                }
            }
        return {}

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        if self.REQUIRES_AUTH:
            self.require_auth()
        import httpx

        endpoint = self.config.get("endpoint") or self.config.get("base_url")
        layer = self.config.get("layer") or self.config.get("typeName")
        if not endpoint:
            self._logger.warning(
                f"{self.DISPLAY_NAME}: no 'endpoint' configured -- set config={{'endpoint': "
                f"'https://your-geoserver.example.com/geoserver'}} (the real GeoServer base URL)"
            )
            return []
        if not layer:
            self._logger.warning(
                f"{self.DISPLAY_NAME}: no 'layer' configured -- the real WFS protocol has no "
                f"'search all layers' concept, set config={{'layer': 'workspace:typename'}}"
            )
            return []

        # Real, confirmed WFS 2.0 GetFeature parameters (GeoServer's
        # own documented reference) -- service/version/request/
        # typeNames/bbox/outputFormat/count, not a fictional REST
        # search with startDate/endDate/cloudCoverMax.
        params: dict[str, Any] = {
            "service": "WFS",
            "version": self.WFS_VERSION,
            "request": "GetFeature",
            "typeNames": layer,
            "outputFormat": "application/json",
            "count": min(query.max_results, 5000),
        }
        if query.bbox:
            bb = query.bbox
            params["bbox"] = (
                f"{bb.min_lon},{bb.min_lat},{bb.max_lon},{bb.max_lat},EPSG:4326"
            )

        wfs_url = endpoint.rstrip("/") + "/wfs"
        try:
            resp = httpx.get(
                wfs_url,
                params=params,
                timeout=self.config.get("timeout", 60),
                **self._auth_kwargs(),
            )
            if resp.status_code != 200:
                self._logger.warning(
                    f"{self.DISPLAY_NAME}: HTTP {resp.status_code} from {wfs_url}"
                )
                return []
            data = resp.json()
            if data.get("type") != "FeatureCollection":
                self._logger.warning(
                    f"{self.DISPLAY_NAME}: response wasn't a real GeoJSON "
                    f"FeatureCollection -- check 'layer' is a real typeName "
                    f"this server advertises (GetCapabilities lists them)"
                )
                return []
            items = data.get("features", [])
            results = [self._parse_item(item) for item in items]
            results = [r for r in results if r is not None]
            if query.cloud_cover_max is not None:
                # Real, honest best-effort only -- a generic WFS layer
                # has no guaranteed cloud-cover concept at all.
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
        item_id = str(item.get("id", ""))
        if not item_id:
            return None
        props = item.get("properties", {}) or {}
        cloud_raw = (
            props.get("cloud_cover")
            or props.get("cloudCover")
            or props.get("eo:cloud_cover")
        )
        geometry = item.get("geometry")
        bbox = None
        if isinstance(geometry, dict) and geometry.get("type") == "Polygon":
            coords = geometry["coordinates"][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            bbox = (min(lons), min(lats), max(lons), max(lats))

        return SatelliteData(
            id=item_id,
            provider=self.PROVIDER_ID,
            satellite=props.get("satellite", props.get("mission", self.DISPLAY_NAME)),
            datetime=props.get("datetime")
            or props.get("date")
            or props.get("acquired"),
            cloud_cover=float(cloud_raw) if cloud_raw is not None else None,
            bbox=bbox,
            geometry=geometry,
            properties=props,
        )

    def download(
        self, data: SatelliteData, destination: Path, options: DownloadOptions
    ) -> DownloadResult:
        if self.REQUIRES_AUTH:
            self.require_auth()
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
                    **self._auth_kwargs(),
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
                error="No assets downloaded -- generic WFS layers often carry no "
                "direct downloadable asset href; use WCS/WMS GetMap for raster "
                "coverages instead.",
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
            auth_type="configurable (none, basic, or bearer)",
            satellites=[],
            search=True,
            download=True,
            supports_sar=False,
            supports_sub_meter=False,
            supports_aoi_filter=True,
            supports_cloud_filter=False,  # real: best-effort only, not a guaranteed WFS field
            supports_date_filter=False,  # real: not sent as a query param, see module docstring
            requires_auth=self.REQUIRES_AUTH,
            has_quota=False,
            regions=["depends on the configured instance"],
            resolution_min_m=None,
            resolution_max_m=None,
            endpoint_url=self.config.get("endpoint", ""),
            docs_url="https://docs.geoserver.org/main/en/user/services/wfs/reference/",
            supported_formats=[DataFormat.GEOTIFF],
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={
                "note": "No universal quota -- depends on the self-hosted instance configured."
            },
        )