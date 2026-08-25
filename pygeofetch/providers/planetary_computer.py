"""
Microsoft Planetary Computer provider for PyGeoFetch.

Provides free access to a large STAC catalog of satellite data via
Microsoft's Planetary Computer API. Supports Sentinel-1/2, Landsat 8/9
Collection 2, MODIS, NAIP, HREA, JRC-GSW, ALOS DEM, and more.

No authentication required for catalog access. SAS tokens are
auto-generated for download URLs.

Example::

    from pygeofetch.providers.planetary_computer import PlanetaryComputerProvider
    from pygeofetch.models.search_query import SearchQuery

    provider = PlanetaryComputerProvider()
    results = provider.search(SearchQuery(
        bbox=(-74.1, 40.6, -73.7, 40.9),
        start_date="2024-01-01",
        satellites=["Sentinel-2"],
        cloud_cover_max=10,
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
    resolve_band_keys,
)
from pygeofetch.models.user_auth import AuthSession, Credentials
from pygeofetch.providers.base import AbstractBaseProvider, SearchError

if TYPE_CHECKING:
    from pygeofetch.models.search_query import SearchQuery


class PlanetaryComputerProvider(AbstractBaseProvider):
    """
    Microsoft Planetary Computer STAC API provider.

    Free, open access to petabyte-scale Earth observation data.
    SAS tokens are automatically generated per-request for download.

    Attributes:
        PROVIDER_ID: 'planetary_computer'
        REQUIRES_AUTH: False
    """

    PROVIDER_ID = "planetary_computer"
    DISPLAY_NAME = "Microsoft Planetary Computer"
    REQUIRES_AUTH = False
    DESCRIPTION = (
        "Free STAC catalog from Microsoft with Sentinel-1/2, Landsat 8/9, "
        "MODIS, NAIP, ALOS DEM, and more. SAS-token authenticated downloads."
    )
    DATA_TYPES = [
        "Sentinel-1",
        "Sentinel-2",
        "Landsat",
        "MODIS",
        "NAIP",
        "DEM",
        "Weather",
    ]
    SATELLITES = ["Sentinel-1", "Sentinel-2", "Landsat-8", "Landsat-9"]
    BASE_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    SAS_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token"

    # STAC collection IDs in Planetary Computer
    COLLECTION_MAP: dict[str, str] = {
        "sentinel-1": "sentinel-1-rtc",
        "sentinel-2": "sentinel-2-l2a",
        "landsat": "landsat-c2-l2",
        "landsat8": "landsat-c2-l2",
        "landsat9": "landsat-c2-l2",
        "modis": "modis-09A1-061",
        "naip": "naip",
        "alos": "alos-dem",
        "cop-dem": "cop-dem-glo-30",
        "hrea": "hrea",
        "jrc-gsw": "jrc-gsw",
    }

    def authenticate(self, credentials: Credentials) -> AuthSession:
        """No login required — return empty session."""
        session = AuthSession(
            provider=self.PROVIDER_ID,
            access_token=None,
        )
        self._session = session
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        return True  # No credentials needed

    def set_session(self, session: Any) -> None:
        """Store an authenticated session for use in requests."""
        self._session = session

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        """
        Search the Planetary Computer STAC catalog.

        Supports full CQL2 filter expressions, spatial AOI, temporal range,
        cloud cover, and collection filtering.

        Args:
            query: Search parameters.

        Returns:
            List of SatelliteData records.
        """
        import httpx

        collections = self._resolve_collections(query)
        payload = query.to_stac_filter()
        payload["collections"] = collections

        # Limit results
        payload.setdefault("limit", min(query.max_results, 250))

        try:
            resp = httpx.post(
                f"{self.BASE_URL}/search",
                json=payload,
                timeout=self.config.get("timeout", 60),
                headers={"Accept": "application/geo+json"},
            )
            if resp.status_code != 200:
                self._handle_http_error(resp)

            features = resp.json().get("features", [])
            results = [
                SatelliteData.from_stac_item(f, self.PROVIDER_ID) for f in features
            ]
            self._logger.info(f"Planetary Computer: {len(results)} results")
            return results

        except Exception as exc:
            msg = f"Planetary Computer search failed: {exc}"
            raise SearchError(msg) from exc

    def _resolve_collections(self, query: SearchQuery) -> list[str]:
        """Map satellite names to Planetary Computer collection IDs."""
        if query.collections:
            return query.collections
        if not query.satellites:
            return ["sentinel-2-l2a", "landsat-c2-l2"]
        collections: set = set()
        for sat in query.satellites:
            key = sat.lower().replace(" ", "").replace("-", "")
            for k, col in self.COLLECTION_MAP.items():
                if k.replace("-", "") in key or key in k.replace("-", ""):
                    collections.add(col)
                    break
            else:
                collections.add(sat.lower())
        return list(collections)

    def _get_sas_token(self, collection: str) -> str | None:
        """
        Request a SAS token for downloading assets from a collection.

        Args:
            collection: STAC collection ID.

        Returns:
            SAS token string, or None on failure.
        """
        import httpx

        try:
            resp = httpx.get(
                f"{self.SAS_URL}/{collection}",
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json().get("token")
        except Exception as exc:
            self._logger.warning(f"Failed to get SAS token for {collection}: {exc}")
        return None

    def _sign_url(self, href: str, token: str | None) -> str:
        """Append SAS token to a URL if present."""
        if not token:
            return href
        sep = "&" if "?" in href else "?"
        return f"{href}{sep}{token}"

    def download(
        self,
        data: SatelliteData,
        destination: Path,
        options: DownloadOptions,
    ) -> DownloadResult:
        """
        Download Planetary Computer assets using SAS-signed URLs.

        Args:
            data: SatelliteData to download.
            destination: Output directory.
            options: Download configuration.

        Returns:
            DownloadResult.
        """
        import httpx

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        # Determine collection for SAS token
        collection = data.collection or "sentinel-2-l2a"
        token = self._get_sas_token(collection)

        start_time = time.time()
        output_paths = []
        total_bytes = 0

        all_assets = data.data_assets or data.assets
        # Filter to requested bands using alias-aware resolver
        selected_bands = list(getattr(options, "bands", None) or [])
        matched_keys = resolve_band_keys(selected_bands, list(all_assets.keys()))
        assets = {k: v for k, v in all_assets.items() if k in matched_keys}
        if not assets:
            assets = all_assets  # final safety fallback

        self._logger.info(
            f"Downloading {len(assets)} asset(s) for {data.id}: {list(assets.keys())}"
        )

        for key, asset in assets.items():
            if not asset.href or not asset.href.startswith("http"):
                continue

            signed_url = self._sign_url(asset.href, token)
            # Real, confirmed bug fix -- see element84.py for the full
            # explanation. Scene disambiguation is now unconditional.
            raw_name = asset.href.split("/")[-1] or f"{key}.tif"
            filename = f"{data.id}_{raw_name}"
            out_file = destination / filename

            if out_file.exists() and not getattr(options, "overwrite", False):
                self._logger.debug(f"Skipping existing file: {out_file.name}")
                output_paths.append(out_file)
                total_bytes += out_file.stat().st_size
                continue

            try:
                self._logger.info(f"  Fetching asset {key!r} → {out_file.name}")
                with httpx.stream(
                    "GET",
                    signed_url,
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
                self._logger.info(
                    f"  ✓ {out_file.name} ({out_file.stat().st_size / 1024 / 1024:.1f} MB)"
                )
            except Exception as exc:
                self._logger.warning(f"Failed to download asset {key!r}: {exc}")

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
            satellites=[
                "Sentinel-1",
                "Sentinel-2",
                "Landsat-8",
                "Landsat-9",
                "MODIS",
                "NAIP",
            ],
            search=True,
            download=True,
            streaming=True,
            stac=True,
            supports_sar=True,
            supports_cql2=True,
            supports_aoi_filter=True,
            supports_cloud_filter=True,
            supports_date_filter=True,
            supports_processing_level_filter=True,
            requires_auth=False,
            regions=["global"],
            resolution_min_m=0.6,
            resolution_max_m=1000.0,
            endpoint_url=self.BASE_URL,
            docs_url="https://planetarycomputer.microsoft.com/docs",
            supported_formats=[DataFormat.COG, DataFormat.GEOTIFF],
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={
                "note": "Free, no quota limits. SAS tokens auto-generated per request."
            },
        )

    def list_collections(self) -> list[dict[str, Any]]:
        """List all available STAC collections."""
        import httpx

        resp = httpx.get(f"{self.BASE_URL}/collections", timeout=30)
        self._handle_http_error(resp)
        return resp.json().get("collections", [])
