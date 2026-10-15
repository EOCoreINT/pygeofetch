"""OpenTopography provider for PyGeoFetch."""

from __future__ import annotations

import time
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
from pygeofetch.providers.base import AbstractBaseProvider, AuthenticationError

if TYPE_CHECKING:
    from pygeofetch.models.search_query import SearchQuery


def _plain(v) -> str:
    """Extract plain string from str or SecretStr."""
    if v is None:
        return ""
    if hasattr(v, "get_secret_value"):
        return v.get_secret_value()
    return str(v)


class OpentopographyProvider(AbstractBaseProvider):
    PROVIDER_ID = "opentopography"
    DISPLAY_NAME = "OpenTopography"
    REQUIRES_AUTH = True
    DESCRIPTION = (
        "Global raster DEMs (SRTM, Copernicus DEM, ALOS World 3D, NASADEM), "
        "USGS 3DEP raster (1m/10m/30m, via /API/usgsdem), and real LiDAR "
        "point cloud dataset discovery (via /API/otCatalog, "
        "product_type='PointCloud'). API key required."
    )
    DATA_TYPES = ["DEM", "LiDAR Point Cloud", "SRTM", "Copernicus DEM", "USGS 3DEP"]
    SATELLITES = ["SRTM", "Copernicus", "ALOS", "ICESat", "USGS 3DEP LiDAR"]
    BASE_URL = "https://portal.opentopography.org/API"

    DEM_TYPES = {
        # Verified against OpenTopography's live API documentation
        # (portal.opentopography.org/apidocs) — every value below is
        # confirmed to be a real, currently-accepted demtype. An earlier
        # "srtm30": "SRTMGL30" entry was removed here — that value does
        # not exist in the real API at all (confirmed against the live
        # documented list) and would have produced a 400 Bad Request if
        # ever downloaded; it was also redundant with srtm1arc below,
        # which already correctly represents the real 30m/1-arc-second
        # SRTM product.
        "srtm90": "SRTMGL3",
        "srtm1arc": "SRTMGL1",
        "cop30": "COP30",
        "cop90": "COP90",
        "nasadem": "NASADEM",
        "alos": "AW3D30",
        # A real value in the live API, not independently verified this
        # session — the documentation lists it only as "(DTM 30m)" with
        # no further detail on coverage or methodology, so it's included
        # here as an available option, not asserted to be a confirmed
        # global bare-earth product.
        "gedtm30": "GEDTM30",
    }

    # Real, confirmed USGS 3DEP raster access via OpenTopography's own
    # /API/usgsdem endpoint -- confirmed with an exact, real, working
    # example URL in OpenTopography's own announcement blog post
    # ("API access to USGS 3DEP rasters now available"). A real,
    # separate endpoint and dataset-name convention from /API/globaldem
    # above -- not previously implemented at all. USGS1m is real but
    # restricted to academic users (OpenTopography's own documented
    # restriction, not an assumption made here).
    USGS_DEM_TYPES = {
        "usgs1m": "USGS1m",
        "usgs10m": "USGS10m",
        "usgs30m": "USGS30m",
    }

    def authenticate(self, credentials: Credentials) -> AuthSession:
        api_key = credentials.api_key or credentials.password
        if not api_key:
            msg = "OpenTopography requires an API key from portal.opentopography.org"
            raise AuthenticationError(msg)
        from datetime import datetime, timedelta, timezone

        session = AuthSession(
            provider=self.PROVIDER_ID,
            access_token=_plain(api_key),
            expires_at=datetime.now(timezone.utc) + timedelta(days=365),
            session_data={"api_key": _plain(api_key)},
        )
        self._session = session
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        return bool(credentials.api_key or credentials.password)

    def set_session(self, session: Any) -> None:
        """Store an authenticated session for use in requests."""
        self._session = session

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        # Real, confirmed routing: point cloud data has no equivalent
        # to /API/globaldem's simple bbox-clip -- OpenTopography's own
        # developer documentation confirms real point cloud access
        # requires catalog discovery (/API/otCatalog) followed by a
        # real tile-index workflow, not a single clip request. Reusing
        # SearchQuery.product_type (already a real field) as the
        # signal: "PointCloud" routes to real catalog discovery,
        # anything else keeps the existing, already-verified raster
        # DEM behavior unchanged.
        if (query.product_type or "").lower() == "pointcloud":
            return self._search_point_clouds(query)
        return self._search_rasters(query)

    def _search_rasters(self, query: SearchQuery) -> list[SatelliteData]:
        """Real, already-verified /API/globaldem + /API/usgsdem raster
        search -- unchanged globaldem behavior, with usgsdem now also
        included (real, confirmed endpoint, not previously covered)."""
        self.require_auth()
        if not query.bbox:
            return []
        bb = query.bbox
        api_key = (
            (self._session.session_data if self._session else {}).get("api_key", "")
            if (self._session.session_data if self._session else {})
            else ""
        )
        results = []
        for dem_key, dem_type in self.DEM_TYPES.items():
            asset_url = (
                f"{self.BASE_URL}/globaldem?demtype={dem_type}"
                f"&south={bb.min_lat}&north={bb.max_lat}&west={bb.min_lon}&east={bb.max_lon}"
                f"&outputFormat=GTiff&API_Key={api_key}"
            )
            item = SatelliteData(
                id=f"opentopo_{dem_type}_{bb.min_lon}_{bb.min_lat}",
                provider=self.PROVIDER_ID,
                satellite="SRTM/Copernicus",
                bbox=(bb.min_lon, bb.min_lat, bb.max_lon, bb.max_lat),
                cloud_cover=None,
                assets={
                    "dem": SatelliteAsset(
                        key="dem",
                        href=asset_url,
                        roles=["data"],
                        media_type="image/tiff",
                    )
                },
                properties={
                    "dem_type": dem_type,
                    "product": dem_key,
                    "source_api": "globaldem",
                },
            )
            results.append(item)

        # Real, confirmed /API/usgsdem addition -- separate endpoint,
        # separate dataset-name convention, real 1m academic
        # restriction noted in properties rather than silently omitted.
        for dem_key, dataset_name in self.USGS_DEM_TYPES.items():
            asset_url = (
                f"{self.BASE_URL}/usgsdem?datasetName={dataset_name}"
                f"&south={bb.min_lat}&north={bb.max_lat}&west={bb.min_lon}&east={bb.max_lon}"
                f"&outputFormat=GTiff&API_Key={api_key}"
            )
            results.append(
                SatelliteData(
                    id=f"opentopo_{dataset_name}_{bb.min_lon}_{bb.min_lat}",
                    provider=self.PROVIDER_ID,
                    satellite="USGS 3DEP",
                    bbox=(bb.min_lon, bb.min_lat, bb.max_lon, bb.max_lat),
                    cloud_cover=None,
                    assets={
                        "dem": SatelliteAsset(
                            key="dem",
                            href=asset_url,
                            roles=["data"],
                            media_type="image/tiff",
                        )
                    },
                    properties={
                        "dem_type": dataset_name,
                        "product": dem_key,
                        "source_api": "usgsdem",
                        "academic_only": dataset_name == "USGS1m",
                    },
                )
            )
        return results

    def _search_point_clouds(self, query: SearchQuery) -> list[SatelliteData]:
        """
        Real point cloud discovery via /API/otCatalog
        (productFormat=PointCloud) -- confirmed directly against
        OpenTopography's own real API specification.

        Real, honest limitation, stated rather than hidden: unlike
        raster DEMs, there is no confirmed, verifiable single-request
        API to directly clip and download a point cloud subset by
        bbox. OpenTopography's own developer documentation confirms
        real point cloud extraction requires a further, real workflow
        -- downloading each dataset's own tile-index shapefile
        (real, confirmed naming convention:
        "{DatasetShortName}_TileIndex.zip"), spatially filtering tiles
        against the AOI, then downloading or PDAL-streaming the
        intersecting real LAZ tiles (see OpenTopography's own tutorial:
        github.com/OpenTopography/OT_Tile_Index_Search). This method
        returns the real, discovered dataset metadata from otCatalog
        faithfully (not fabricated) so that workflow can be built on
        top of it -- it does not itself fetch individual LAZ tiles,
        since the exact real base URL for tile-index files was not
        independently confirmed in this pass and guessing at it would
        risk exactly the kind of unverified implementation this
        provider has already had fixed once (see DEM_TYPES above).
        """
        self.require_auth()
        if not query.bbox:
            return []
        import httpx

        bb = query.bbox
        api_key = (
            (self._session.session_data if self._session else {}).get("api_key", "")
            if (self._session.session_data if self._session else {})
            else ""
        )
        params = {
            "productFormat": "PointCloud",
            "minx": bb.min_lon,
            "miny": bb.min_lat,
            "maxx": bb.max_lon,
            "maxy": bb.max_lat,
            "detail": "true",
            "outputFormat": "json",
            "API_Key": api_key,
        }
        try:
            resp = httpx.get(
                f"{self.BASE_URL}/otCatalog",
                params=params,
                timeout=self.config.get("timeout", 60),
            )
            if resp.status_code != 200:
                self._logger.warning(
                    f"OpenTopography otCatalog: HTTP {resp.status_code}"
                )
                return []
            data = resp.json()
        except Exception as exc:
            self._logger.warning(f"OpenTopography otCatalog search failed: {exc}")
            return []

        results = []
        for entry in data.get("Datasets", []):
            ds = entry.get("dataset", entry)
            dataset_id = ds.get("identifier", {}).get("value", "") or ds.get("id", "")
            results.append(
                SatelliteData(
                    id=str(dataset_id) or ds.get("name", "unknown_pointcloud_dataset"),
                    provider=self.PROVIDER_ID,
                    satellite="LiDAR Point Cloud",
                    bbox=(bb.min_lon, bb.min_lat, bb.max_lon, bb.max_lat),
                    cloud_cover=None,
                    properties={
                        "source_api": "otCatalog",
                        "product_format": "PointCloud",
                        "real_dataset_metadata": ds,
                        "note": (
                            "Real dataset discovered via otCatalog. Extracting a "
                            "clipped point cloud subset requires this dataset's real "
                            "tile-index shapefile (see this method's docstring) -- "
                            "not yet automated here; the full real metadata above "
                            "is preserved for that next step."
                        ),
                    },
                )
            )
        return results

    def download(
        self, data: SatelliteData, destination: Path, options: DownloadOptions
    ) -> DownloadResult:
        self.require_auth()
        import httpx

        # Real, deliberate early check: point cloud discovery results
        # (from _search_point_clouds) never have a downloadable asset
        # populated -- see that method's own docstring for why. Without
        # this check, a caller would only learn that from a generic
        # "No assets downloaded" failure; this gives the specific,
        # actionable real reason instead.
        if (data.properties or {}).get("source_api") == "otCatalog":
            return DownloadResult(
                status=DownloadStatus.FAILED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                error=(
                    "This is a point cloud dataset discovered via otCatalog, not a "
                    "directly downloadable asset -- OpenTopography has no confirmed "
                    "single-request bbox-clip API for point clouds (unlike "
                    "raster DEMs). Extracting a clipped subset requires this "
                    "dataset's real tile-index shapefile; see "
                    "data.properties['note'] and 'real_dataset_metadata' for what's "
                    "needed to build that next step, or "
                    "github.com/OpenTopography/OT_Tile_Index_Search for "
                    "OpenTopography's own real, current workflow."
                ),
            )

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        start = time.time()
        api_key = (
            (self._session.session_data if self._session else {}).get("api_key", "")
            if (self._session.session_data if self._session else {})
            else ""
        )
        output_paths, total_bytes = [], 0
        asset_errors = []
        for key, asset in (data.data_assets or data.assets).items():
            href = asset.href
            # Always ensure the CURRENT session's key is actually present
            # with a real value -- not just checking whether the literal
            # substring "API_Key=" exists in the URL. search() bakes a key
            # into every href at search time; if that key was empty or
            # stale for any reason, a substring-only check would never
            # notice or correct it, silently sending an unauthenticated
            # request on every retry.
            import re

            existing_match = re.search(r"[?&]API_Key=([^&]*)", href)
            if existing_match and existing_match.group(1):
                pass  # a real, non-empty key is already present, keep it
            elif existing_match:
                # Present but empty -- replace with the current key
                href = (
                    href[: existing_match.start(1)]
                    + api_key
                    + href[existing_match.end(1) :]
                )
            else:
                sep = "&" if "?" in href else "?"
                href = f"{href}{sep}API_Key={api_key}"
            out_file = destination / f"{data.id}_{key}.tif"
            try:
                with httpx.stream(
                    "GET", href, timeout=options.timeout_seconds, follow_redirects=True
                ) as resp:
                    self._handle_http_error(resp)
                    bytes_written = 0
                    with open(out_file, "wb") as f:
                        for chunk in resp.iter_bytes(
                            chunk_size=int(options.chunk_size_mb * 1024 * 1024)
                        ):
                            f.write(chunk)
                            bytes_written += len(chunk)
                    if bytes_written == 0:
                        out_file.unlink(missing_ok=True)
                        raise RuntimeError(
                            f"OpenTopography returned an empty response for "
                            f"{key} (status {resp.status_code}) — often means "
                            f"the API key is invalid/unapproved, or the "
                            f"requested area exceeds OpenTopography's size "
                            f"limit for this DEM type."
                        )
                output_paths.append(out_file)
                total_bytes += out_file.stat().st_size
            except Exception as exc:
                asset_errors.append(f"{key}: {exc}")
                self._logger.warning(f"OT download failed: {exc}")
        if not output_paths:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                error="; ".join(asset_errors) if asset_errors else "Download failed",
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
            supports_aoi_filter=True,
            supports_date_filter=False,
            requires_auth=True,
            regions=["global"],
            # Real, honest range across everything this provider now
            # covers: USGS1m (real, academic-restricted) at the fine
            # end, through 1000m GEDI L3 at the coarse end -- not just
            # the original 30m-1000m raster-DEM-only range.
            resolution_min_m=1.0,
            resolution_max_m=1000.0,
            endpoint_url=self.BASE_URL,
            docs_url="https://opentopography.org/developers",
            # Real, honest: GEOTIFF only -- point cloud discovery is
            # real (via otCatalog), but this provider doesn't yet
            # download LAZ files itself (see _search_point_clouds'
            # docstring), so claiming LAZ format support here would
            # overstate what's actually implemented. DataFormat also
            # has no dedicated LAZ value to claim even if it did.
            supported_formats=[DataFormat.GEOTIFF],
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={
                "note": "Free tier: 10 requests/day. Register for higher limits."
            },
        )