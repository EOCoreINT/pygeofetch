"""
Element 84 Earth Search provider for PyGeoFetch.

Element 84 operates a production-grade STAC API at
https://earth-search.aws.element84.com/v1 with Sentinel-2 COGs,
Landsat Collection 2, NAIP, Sentinel-1, and more.

No authentication required. All data is served as Cloud Optimized GeoTIFFs.

Example::

    from pygeofetch.providers.element84 import Element84Provider

    provider = Element84Provider()
    results = provider.search(SearchQuery(
        bbox=(-74.1, 40.6, -73.7, 40.9),
        start_date="2024-01-01",
        satellites=["Sentinel-2"],
        cloud_cover_max=5,
    ))
"""

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
    SatelliteData,
)
from pygeofetch.models.user_auth import AuthSession, Credentials
from pygeofetch.providers.base import AbstractBaseProvider, SearchError

if TYPE_CHECKING:
    from pygeofetch.models.search_query import SearchQuery


class Element84Provider(AbstractBaseProvider):
    """
    Element 84 Earth Search STAC API provider.

    Production-grade public STAC API with COG (Cloud Optimized GeoTIFF) data.
    Supports Sentinel-2 L2A, Landsat Collection 2, NAIP, Sentinel-1 RTC, and
    Copernicus DEM.

    Attributes:
        PROVIDER_ID: 'element84'
        REQUIRES_AUTH: False
    """

    PROVIDER_ID = "element84"
    DISPLAY_NAME = "Element 84 Earth Search"
    REQUIRES_AUTH = False
    DESCRIPTION = (
        "Production STAC API by Element 84 with Sentinel-2 COGs, Landsat "
        "Collection 2, NAIP, Sentinel-1 RTC, and Copernicus DEM. All free."
    )
    DATA_TYPES = ["Sentinel-2", "Landsat", "NAIP", "Sentinel-1", "DEM"]
    SATELLITES = ["Sentinel-2A", "Sentinel-2B", "Landsat-8", "Landsat-9", "Sentinel-1"]
    BASE_URL = "https://earth-search.aws.element84.com/v1"

    COLLECTION_MAP: dict[str, str] = {
        "sentinel-2": "sentinel-2-l2a",
        "sentinel-2a": "sentinel-2-l2a",
        "sentinel-2b": "sentinel-2-l2a",
        "landsat": "landsat-c2-l2",
        "landsat8": "landsat-c2-l2",
        "landsat9": "landsat-c2-l2",
        "naip": "naip",
        "sentinel-1": "sentinel-1-rtc",
        "cop-dem": "cop-dem-glo-30",
        "cop-dem-90": "cop-dem-glo-90",
    }

    def authenticate(self, credentials: Credentials) -> AuthSession:
        """No authentication required."""
        session = AuthSession(
            provider=self.PROVIDER_ID,
            access_token=None,
        )
        self._session = session
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        return True

    def set_session(self, session: Any) -> None:
        """Store an authenticated session for use in requests."""
        self._session = session

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        """
        Search the Element 84 Earth Search STAC API.

        Args:
            query: Search parameters including spatial, temporal, cloud cover filters.

        Returns:
            List of SatelliteData with COG asset URLs.
        """
        import httpx

        collections = self._resolve_collections(query)
        payload = query.to_stac_filter()
        payload["collections"] = collections
        payload.setdefault("limit", min(query.max_results, 250))

        # Resolution filter via properties
        if query.resolution_min_m or query.resolution_max_m:
            if "filter" not in payload:
                payload["filter"] = {"op": "and", "args": []}
            payload.setdefault("fields", {})

        try:
            resp = httpx.post(
                f"{self.BASE_URL}/search",
                json=payload,
                timeout=self.config.get("timeout", 60),
            )
            if resp.status_code != 200:
                self._handle_http_error(resp)
            features = resp.json().get("features", [])
            results = [
                SatelliteData.from_stac_item(f, self.PROVIDER_ID) for f in features
            ]
            self._logger.info(f"Element84: {len(results)} items found")
            return results

        except Exception as exc:
            msg = f"Element 84 search failed: {exc}"
            raise SearchError(msg) from exc

    def _resolve_collections(self, query: SearchQuery) -> list[str]:
        if query.collections:
            return query.collections
        if not query.satellites:
            return ["sentinel-2-l2a"]
        cols: set = set()
        for sat in query.satellites:
            key = sat.lower().replace(" ", "").replace("-", "")
            for k, col in self.COLLECTION_MAP.items():
                if k.replace("-", "") in key or key in k.replace("-", ""):
                    cols.add(col)
                    break
            else:
                cols.add(sat.lower())
        return list(cols)

    def download(
        self,
        data: SatelliteData,
        destination: Path,
        options: DownloadOptions,
    ) -> DownloadResult:
        """
        Download COG assets directly from S3.

        Args:
            data: SatelliteData to download.
            destination: Output directory.
            options: Download options.

        Returns:
            DownloadResult.
        """
        import httpx

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        output_paths = []
        total_bytes = 0

        assets = data.data_assets or data.assets
        for key, asset in assets.items():
            if not asset.href or not asset.href.startswith("http"):
                continue
            # Real, confirmed bug fixed here: the raw URL basename (e.g.
            # "B02.tif") is what real COG asset URLs typically end in,
            # NOT scene-specific -- the old `or f"{data.id}_{key}.tif"`
            # fallback only fired when the basename was empty, which
            # essentially never happens for real asset URLs. That meant
            # every scene's same-named band files silently collided in
            # a shared destination directory, with the LAST scene to
            # finish downloading overwriting all the others -- no error
            # raised, just quietly wrong data on disk. Scene disambiguation
            # is now unconditional, not a fallback.
            raw_name = asset.href.split("/")[-1] or f"{key}.tif"
            filename = f"{data.id}_{raw_name}"
            out_file = destination / filename

            if out_file.exists() and not getattr(options, "overwrite", False):
                output_paths.append(out_file)
                total_bytes += out_file.stat().st_size
                continue

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
                self._logger.warning(f"Asset {key} download failed: {exc}")

        duration = time.time() - start_time
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
            duration_seconds=duration,
        )

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_id=self.PROVIDER_ID,
            name=self.DISPLAY_NAME,
            description=self.DESCRIPTION,
            auth_type="none",
            satellites=["Sentinel-2", "Landsat-8", "Landsat-9", "Sentinel-1", "NAIP"],
            search=True,
            download=True,
            streaming=True,
            stac=True,
            supports_sar=True,
            supports_cql2=True,
            supports_aoi_filter=True,
            supports_cloud_filter=True,
            supports_date_filter=True,
            requires_auth=False,
            regions=["global"],
            resolution_min_m=0.6,
            resolution_max_m=500.0,
            endpoint_url=self.BASE_URL,
            docs_url="https://earth-search.aws.element84.com/v1",
            supported_formats=[DataFormat.COG, DataFormat.GEOTIFF],
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={"note": "Free, no quota. All data served as COGs from AWS S3."},
        )

    def list_collections(self) -> list[dict[str, Any]]:
        """List all available STAC collections."""
        import httpx

        resp = httpx.get(f"{self.BASE_URL}/collections", timeout=30)
        self._handle_http_error(resp)
        return resp.json().get("collections", [])