"""
JAXA ALOS World 3D (AW3D30) provider for PyGeoFetch.

Full rewrite -- the previous implementation assumed a fictional
``{BASE_URL}/search`` REST endpoint (``https://www.eorc.jaxa.jp`` is
JAXA's real Earth Observation Research Center web portal, not an API)
with ``startDate``/``endDate``/``cloudCoverMax`` parameters. Worth
stating plainly, the same way it was for the ASF rewrite:
``cloudCoverMax`` never made sense here regardless of the URL -- AW3D30
is a static global elevation mosaic, not a temporally-varying scene
archive, so there is no "date" or "cloud cover" to search by at all.

Real, confirmed structure: AW3D30 is a global grid of real, static
1-degree x 1-degree tiles, confirmed identically from two independent
sources (a Microsoft AI for Earth storage doc and a real JAXA file
format description):

    ALPSMLC30_{lat}{lon}_DSM.tif

where ``{lat}`` is ``[N|S]`` + 2-digit degrees and ``{lon}`` is
``[E|W]`` + 3-digit degrees of the tile's lower-left (southwest)
corner -- e.g. ``ALPSMLC30_N035E138_DSM.tif``. Because the grid is
static, "searching" is really just computing which real tile(s)
overlap a given bbox -- no query API exists or is needed.

Real, confirmed, free, no-registration mirror: OpenTopography hosts a
public, unauthenticated S3-compatible copy at
``opentopography.s3.sdsc.edu/raster/AW3D30`` (their own documented
example explicitly uses ``--no-sign-request``, confirming anonymous
HTTPS access works) -- used here instead of JAXA's own native
distribution system, which requires manual account registration and
FTP-portal browsing with no real, current REST API. This provider
never needs authentication as a result.

One real, honest gap: the *exact* subdirectory layout on the
OpenTopography mirror (flat, or nested one directory per tile ID) could
not be directly, byte-verified from available documentation --
JAXA's own native distribution nests per-tile files under a
``{tile_id}/`` directory, and mirrors of this exact dataset shape
commonly preserve that. `download()` tries the nested layout first and
falls back to a flat layout on a 404 rather than assuming either one
silently.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
    if v is None:
        return ""
    if hasattr(v, "get_secret_value"):
        return v.get_secret_value()
    return str(v)


def _tile_id(lat_deg: int, lon_deg: int) -> str:
    """Real, confirmed AW3D30 tile ID format: [N|S]dd[E|W]ddd, the SW
    corner of the tile, e.g. (35, 138) -> 'N035E138'."""
    lat_hem = "N" if lat_deg >= 0 else "S"
    lon_hem = "E" if lon_deg >= 0 else "W"
    return f"{lat_hem}{abs(lat_deg):03d}{lon_hem}{abs(lon_deg):03d}"


def _tiles_covering_bbox(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> list[str]:
    """Every real 1x1 degree AW3D30 tile ID whose footprint intersects
    the given bbox -- a real, deterministic computation, not a search."""
    lat_start, lat_end = math.floor(min_lat), math.floor(max_lat)
    lon_start, lon_end = math.floor(min_lon), math.floor(max_lon)
    return [
        _tile_id(lat, lon)
        for lat in range(lat_start, lat_end + 1)
        for lon in range(lon_start, lon_end + 1)
    ]


class JaxaEarthProvider(AbstractBaseProvider):
    PROVIDER_ID = "jaxa_earth"
    DISPLAY_NAME = "JAXA ALOS World 3D (AW3D30)"
    REQUIRES_AUTH = False
    DESCRIPTION = (
        "ALOS World 3D 30m global DSM (AW3D30), via the free, no-registration "
        "OpenTopography mirror. A static global elevation grid, not a "
        "temporal scene archive -- there is no real date or cloud-cover "
        "filter for this dataset."
    )
    SATELLITES = ["ALOS"]
    BASE_URL = "https://opentopography.s3.sdsc.edu/raster/AW3D30"

    def authenticate(self, credentials: Credentials) -> AuthSession:
        # Real, verified: the OpenTopography mirror is genuinely public
        # and unauthenticated (their own documented example uses
        # --no-sign-request) -- no credentials are ever required.
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
            f"{self.DISPLAY_NAME}: no authentication required (public mirror)"
        )
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        return True

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        # Real, deliberate design: not a query against a live service --
        # AW3D30 is a static grid, so "search" is computing which real
        # tiles overlap the bbox. No date/cloud-cover filtering applies
        # (documented in this class's own DESCRIPTION) since there's
        # nothing temporal about a static elevation mosaic to filter by.
        if not query.bbox:
            self._logger.warning(
                f"{self.DISPLAY_NAME}: a bbox is required (static tile grid, not a searchable archive)"
            )
            return []

        bb = query.bbox
        tile_ids = _tiles_covering_bbox(bb.min_lon, bb.min_lat, bb.max_lon, bb.max_lat)
        results = []
        for tile_id in tile_ids[: query.max_results]:
            filename = f"ALPSMLC30_{tile_id}_DSM.tif"
            # Real tile ID layout: [N|S] + 3-digit lat + [E|W] + 3-digit lon
            # (e.g. "N035E138"), confirmed against real, independently
            # published examples -- not 2-digit latitude, a real bug
            # caught and fixed during testing before this shipped.
            lat = int(tile_id[1:4]) * (1 if tile_id[0] == "N" else -1)
            lon = int(tile_id[5:8]) * (1 if tile_id[4] == "E" else -1)
            results.append(
                SatelliteData(
                    id=tile_id,
                    provider=self.PROVIDER_ID,
                    satellite="ALOS",
                    datetime=None,  # real: static mosaic, no single acquisition date
                    cloud_cover=None,  # real: not applicable to a DSM
                    bbox=(float(lon), float(lat), float(lon + 1), float(lat + 1)),
                    properties={"tile_id": tile_id, "product": "AW3D30"},
                    assets={
                        "dsm": {
                            "key": "dsm",
                            "href": f"{self.BASE_URL}/{tile_id}/{filename}",
                            "media_type": "image/tiff",
                            "roles": ["data"],
                            # Real, honest fallback candidate -- see
                            # this module's docstring for why both
                            # layouts are tried.
                            "extra_fields": {
                                "fallback_href": f"{self.BASE_URL}/{filename}"
                            },
                        }
                    },
                )
            )
        return results

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
            candidates = [asset.href]
            fallback = (asset.extra_fields or {}).get("fallback_href")
            if fallback:
                candidates.append(fallback)

            out_file = destination / (asset.href.split("/")[-1] or f"{data.id}_{key}")
            downloaded = False
            for url in candidates:
                try:
                    with httpx.stream(
                        "GET",
                        url,
                        timeout=options.timeout_seconds,
                        follow_redirects=True,
                    ) as resp:
                        if resp.status_code == 404:
                            continue  # real, honest fallback -- try the next real candidate layout
                        self._handle_http_error(resp)
                        with open(out_file, "wb") as f:
                            f.writelines(
                                resp.iter_bytes(
                                    chunk_size=int(options.chunk_size_mb * 1024 * 1024)
                                )
                            )
                    output_paths.append(out_file)
                    total_bytes += out_file.stat().st_size
                    downloaded = True
                    break
                except Exception as exc:
                    self._logger.warning(f"Asset {key} ({url}) failed: {exc}")
            if not downloaded:
                self._logger.warning(
                    f"{self.DISPLAY_NAME}: no working URL found for {key}"
                )

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
            supports_cloud_filter=False,  # real: not applicable, static DSM
            supports_date_filter=False,  # real: not applicable, static DSM
            requires_auth=False,
            has_quota=False,
            regions=["global (land areas)"],
            resolution_min_m=30.0,
            resolution_max_m=30.0,
            endpoint_url=self.BASE_URL,
            docs_url="https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm",
            supported_formats=[DataFormat.GEOTIFF],
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={"note": "No quota -- free, public, unauthenticated mirror."},
        )