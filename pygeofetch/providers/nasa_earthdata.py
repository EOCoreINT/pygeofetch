"""
NASA Earthdata (CMR) provider.

Provides access to NASA's Common Metadata Repository (CMR) which indexes
data from all NASA DAACs (Distributed Active Archive Centers) including
NSIDC, ORNL, LP DAAC, PO.DAAC, ASF DAAC, and more.

Authentication: NASA Earthdata Login (urs.earthdata.nasa.gov)
"""

from __future__ import annotations

import time
from datetime import datetime
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
from pygeofetch.providers.base import (
    AbstractBaseProvider,
    AuthenticationError,
    SearchError,
)

if TYPE_CHECKING:
    from pygeofetch.models.search_query import SearchQuery


def _plain(v) -> str:
    """Extract plain string from str or SecretStr."""
    if v is None:
        return ""
    if hasattr(v, "get_secret_value"):
        return v.get_secret_value()
    return str(v)


class NASAEarthdataProvider(AbstractBaseProvider):
    PROVIDER_ID = "nasa_earthdata"
    DISPLAY_NAME = "NASA Earthdata (CMR)"
    REQUIRES_AUTH = True
    DESCRIPTION = "Access to NASA Earth science data from all DAACs via the Common Metadata Repository."
    DATA_TYPES = ["MODIS", "VIIRS", "ICESat-2", "GEDI", "Landsat", "SRTM", "ASTER"]
    BASE_URL = "https://cmr.earthdata.nasa.gov/search"

    def authenticate(self, credentials: Credentials) -> AuthSession:
        if not credentials.username or not credentials.get_password():
            msg = "NASA Earthdata requires username and password"
            raise AuthenticationError(msg)
        import httpx

        try:
            resp = httpx.get(
                "https://urs.earthdata.nasa.gov/api/users/user",
                auth=(credentials.username, _plain(credentials.get_password())),
                timeout=30,
            )
            if resp.status_code not in (200, 404):
                msg = f"NASA Earthdata login failed: HTTP {resp.status_code}"
                raise AuthenticationError(msg)
            session = AuthSession(
                provider=self.PROVIDER_ID,
                access_token=None,
                session_data={
                    "username": credentials.username,
                    "password": _plain(credentials.get_password()),
                },
            )
            self._session = session
            self._logger.info(
                f"Authenticated with NASA Earthdata as {credentials.username!r}"
            )
            return session
        except AuthenticationError:
            raise
        except Exception as exc:
            msg = f"NASA Earthdata auth error: {exc}"
            raise AuthenticationError(msg) from exc

    def validate_credentials(self, credentials: Credentials) -> bool:
        return bool(credentials.username and credentials.get_password())

    def set_session(self, session: Any) -> None:
        """Store an authenticated session for use in requests."""
        self._session = session

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        self.require_auth()
        import httpx

        params: dict[str, Any] = {
            "page_size": min(query.max_results, 2000),
            "sort_key[]": "-start_date",
        }
        if query.bbox:
            b = query.bbox
            params["bounding_box[]"] = (
                f"{b.min_lon},{b.min_lat},{b.max_lon},{b.max_lat}"
            )
        if query.start_date:
            params["temporal[]"] = f"{query.start_date}T00:00:00Z,"
        if query.end_date:
            existing = params.get("temporal[]", ",")
            params["temporal[]"] = (
                f"{existing.split(',')[0]},{query.end_date}T23:59:59Z"
            )
        if (query.cloud_cover_max or 100) < 100:
            params["cloud_cover[]"] = f"0,{int(query.cloud_cover_max)}"
        if query.collections:
            params["short_name[]"] = query.collections
        try:
            resp = httpx.get(
                f"{self.BASE_URL}/granules.json",
                params=params,
                auth=(
                    (self._session.session_data if self._session else {})["username"],
                    (self._session.session_data if self._session else {})["password"],
                ),
                timeout=60,
            )
            self._handle_http_error(resp)
            data = resp.json()
            granules = data.get("feed", {}).get("entry", [])
            return [self._granule_to_satellite_data(g) for g in granules]
        except Exception as exc:
            msg = f"NASA Earthdata search failed: {exc}"
            raise SearchError(msg) from exc

    def _parse_cmr_polygons(self, polygons_raw):
        """
        Real, confirmed CMR granules.json "polygons" field parser.

        Structure confirmed directly against two independent, real
        examples (NASA's own Common-Metadata-Repository GitHub repo,
        and NSIDC's own SnowEx Hackweek tutorial), not assumed:
        polygons is a list of polygons (multi-polygon support), each
        polygon is a list of ring strings (holes), each ring string is
        a space-separated "lat lon lat lon..." sequence -- same
        lat-first order this file's own "boxes" parsing already
        handles for the bbox field.
        """
        coordinates = []
        for ring_group in polygons_raw or []:
            rings = []
            for ring_str in ring_group:
                try:
                    parts = [float(x) for x in ring_str.split()]
                except (ValueError, AttributeError):
                    continue
                if len(parts) % 2 != 0 or len(parts) < 6:
                    continue
                points = [[parts[i + 1], parts[i]] for i in range(0, len(parts), 2)]
                rings.append(points)
            if rings:
                coordinates.append(rings)
        if not coordinates:
            return None
        if len(coordinates) == 1:
            return {"type": "Polygon", "coordinates": coordinates[0]}
        return {"type": "MultiPolygon", "coordinates": coordinates}

    def _granule_to_satellite_data(self, g: dict) -> SatelliteData:
        bbox = None
        boxes = g.get("boxes", [])
        if boxes:
            parts = [float(x) for x in boxes[0].split()]
            if len(parts) == 4:
                bbox = (parts[1], parts[0], parts[3], parts[2])
        geometry = self._parse_cmr_polygons(g.get("polygons"))
        links = g.get("links", [])
        assets = {}
        for link in links:
            if link.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#":
                href = link.get("href", "")
                key = href.split("/")[-1] or "data"
                assets[key] = SatelliteAsset(key=key, href=href, roles=["data"])
        dt = None
        try:
            dt = datetime.fromisoformat(g.get("time_start", "").replace("Z", "+00:00"))
        except Exception:
            pass
        return SatelliteData(
            id=g.get("id", g.get("title", "unknown")),
            provider=self.PROVIDER_ID,
            collection=g.get("short_name", g.get("collection_concept_id")),
            satellite=(
                g.get("platforms", [{}])[0].get("short_name")
                if g.get("platforms")
                else None
            ),
            datetime=dt,
            bbox=bbox,
            geometry=geometry,
            cloud_cover=g.get("cloud_cover"),
            assets=assets,
            properties=g,
        )

    def download(
        self, data: SatelliteData, destination: Path, options: DownloadOptions
    ) -> DownloadResult:
        self.require_auth()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        import httpx

        output_paths = []
        start = time.time()
        for key, asset in list(data.assets.items())[:3]:
            filename = asset.href.split("/")[-1] or f"{data.id}.nc"
            out = destination / filename
            if out.exists() and not options.overwrite:
                output_paths.append(out)
                continue
            try:
                with httpx.stream(
                    "GET",
                    asset.href,
                    auth=(
                        (self._session.session_data if self._session else {})[
                            "username"
                        ],
                        (self._session.session_data if self._session else {})[
                            "password"
                        ],
                    ),
                    timeout=options.timeout_seconds,
                    follow_redirects=True,
                ) as resp:
                    self._handle_http_error(resp)
                    with open(out, "wb") as f:
                        f.writelines(
                            resp.iter_bytes(
                                chunk_size=int(options.chunk_size_mb * 1024 * 1024)
                            )
                        )
                output_paths.append(out)
            except Exception as exc:
                self._logger.warning(f"Failed to download {key}: {exc}")
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
            bytes_downloaded=sum(p.stat().st_size for p in output_paths),
            duration_seconds=time.time() - start,
        )

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_id="nasa_earthdata",
            name="NASA Earthdata",
            search=True,
            download=True,
            stac=False,
            requires_auth=True,
            supported_satellites=["MODIS", "VIIRS", "ICESat-2", "GEDI", "Landsat"],
            supported_formats=[
                DataFormat.HDF4,
                DataFormat.HDF5,
                DataFormat.NETCDF,
                DataFormat.GEOTIFF,
            ],
            supports_cloud_filter=True,
            supports_date_filter=True,
            supports_aoi_filter=True,
        )

    def get_quota_info(self) -> QuotaInfo:
        self.require_auth()
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={
                "note": "NASA Earthdata provides free access with Earthdata Login"
            },
        )
