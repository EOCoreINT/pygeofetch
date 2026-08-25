"""
AWS Earth Open Data provider.

Provides access to publicly available satellite datasets hosted on AWS S3:
  - Sentinel-2 (Level-2A, Cloud Optimized GeoTIFF)
  - Landsat Collection 2 (Level-2)
  - NAIP Aerial Imagery
  - NOAA Weather Data
  - and more via the AWS Open Data Registry

No authentication required. Uses the Earth Search STAC API by Element84.

Example::

    from pygeofetch.providers.aws_earth import AWSEarthProvider
    from pygeofetch.models.search_query import SearchQuery

    provider = AWSEarthProvider()
    results = provider.search(SearchQuery(
        bbox=(-74.1, 40.6, -73.7, 40.9),
        start_date="2024-01-01",
        end_date="2024-06-01",
        satellites=["Sentinel-2"],
        cloud_cover_max=10,
    ))
    # Download is free; no credentials needed
    provider.download(results[0], destination=Path("./data/"))
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
from pygeofetch.providers.base import AbstractBaseProvider, ProviderError, SearchError

if TYPE_CHECKING:
    from pygeofetch.models.search_query import SearchQuery


class AWSEarthProvider(AbstractBaseProvider):
    """
    AWS Earth Open Data provider using the Earth Search STAC API.

    Accesses Sentinel-2, Landsat, and other datasets hosted freely
    on AWS S3 via Element84's Earth Search STAC endpoint.

    No authentication required.

    Attributes:
        PROVIDER_ID: 'aws_earth'
        DISPLAY_NAME: 'AWS Earth Open Data'
        REQUIRES_AUTH: False
    """

    PROVIDER_ID = "aws_earth"
    DISPLAY_NAME = "AWS Earth Open Data"
    REQUIRES_AUTH = False
    DESCRIPTION = (
        "Free access to Sentinel-2, Landsat, NAIP, and other datasets "
        "hosted on AWS S3 via the Earth Search STAC API."
    )
    DATA_TYPES = ["Sentinel-2", "Landsat", "NAIP", "MODIS", "DEM"]
    BASE_URL = "https://earth-search.aws.element84.com/v1"

    # STAC collection IDs
    COLLECTION_MAP = {
        "sentinel-2": "sentinel-2-l2a",
        "sentinel-2a": "sentinel-2-l2a",
        "sentinel-2b": "sentinel-2-l2a",
        "landsat": "landsat-c2-l2",
        "landsat8": "landsat-c2-l2",
        "landsat9": "landsat-c2-l2",
        "naip": "naip",
        "cop-dem": "cop-dem-glo-30",
    }

    def authenticate(self, credentials: Credentials) -> AuthSession:
        """No authentication needed; return empty session."""
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
        Search the Earth Search STAC API.

        Args:
            query: Search parameters.

        Returns:
            List of matching SatelliteData records (COG assets).
        """
        import httpx

        collections = self._resolve_collections(query)
        stac_params = query.to_stac_filter()
        stac_params["collections"] = collections

        try:
            response = httpx.post(
                f"{self.BASE_URL}/search",
                json=stac_params,
                timeout=60,
            )
            if response.status_code != 200:
                self._handle_http_error(response)

            data = response.json()
            features = data.get("features", [])
            return [SatelliteData.from_stac_item(f, self.PROVIDER_ID) for f in features]

        except Exception as exc:
            msg = f"AWS Earth search failed: {exc}"
            raise SearchError(msg) from exc

    def _resolve_collections(self, query: SearchQuery) -> list[str]:
        """Map satellite names to Earth Search collection IDs."""
        if query.collections:
            return query.collections

        if not query.satellites:
            return ["sentinel-2-l2a"]

        collections = set()
        for sat in query.satellites:
            sat_lower = sat.lower().replace(" ", "").replace("-", "")
            for key, col in self.COLLECTION_MAP.items():
                if key.replace("-", "") in sat_lower or sat_lower in key.replace(
                    "-", ""
                ):
                    collections.add(col)
                    break
            else:
                collections.add(sat.lower())

        return list(collections)

    def download(
        self,
        data: SatelliteData,
        destination: Path,
        options: DownloadOptions,
    ) -> DownloadResult:
        """
        Download individual COG assets from S3 (no auth needed).

        Downloads data assets (bands) directly from their public S3 URLs.

        Args:
            data: SatelliteData to download.
            destination: Output directory.
            options: Download options.

        Returns:
            DownloadResult with paths to all downloaded band files.
        """
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        import httpx

        start_time = time.time()
        output_paths = []
        total_bytes = 0
        asset_errors: list[str] = []

        # Download data assets (bands), skip thumbnails/metadata
        data_assets = data.data_assets
        if not data_assets:
            data_assets = data.assets  # Fallback: download everything

        # Filter to requested bands using alias-aware resolver
        selected_bands = list(getattr(options, "bands", None) or [])
        matched_keys = resolve_band_keys(selected_bands, list(data_assets.keys()))
        data_assets = {k: v for k, v in data_assets.items() if k in matched_keys}

        for key, asset in data_assets.items():
            if not asset.href or not asset.href.startswith("http"):
                continue

            # Real, confirmed bug fix -- see element84.py for the full
            # explanation. Scene disambiguation is now unconditional.
            raw_name = asset.href.split("/")[-1] or f"{key}.tif"
            filename = f"{data.id}_{raw_name}"
            output_file = destination / filename

            if output_file.exists() and not options.overwrite:
                output_paths.append(output_file)
                total_bytes += output_file.stat().st_size
                continue

            try:
                with httpx.stream(
                    "GET",
                    asset.href,
                    timeout=options.timeout_seconds,
                    follow_redirects=True,
                ) as resp:
                    self._handle_http_error(resp)
                    content_length = resp.headers.get("content-length")
                    bytes_written = 0
                    with open(output_file, "wb") as f:
                        for chunk in resp.iter_bytes(
                            chunk_size=int(options.chunk_size_mb * 1024 * 1024)
                        ):
                            f.write(chunk)
                            bytes_written += len(chunk)

                if bytes_written == 0:
                    # A 200 OK with a genuinely empty body — _handle_http_error()
                    # only inspects the status code, so this passes through
                    # as an apparent "success" with nothing written. Without
                    # this check, an empty file gets kept on disk and this
                    # asset silently ends up in output_paths, only to be
                    # caught later by the separate, generic file-integrity
                    # validation in core/downloader.py ("File is empty (0
                    # bytes)") — which works, but gives no indication this
                    # happened at the HTTP layer specifically, and wastes a
                    # full write-to-disk + later re-open-to-validate cycle
                    # on a file that was never going to be usable.
                    output_file.unlink(missing_ok=True)
                    raise ProviderError(
                        f"Asset {key!r} returned an empty response body "
                        f"(HTTP {resp.status_code}, Content-Length: "
                        f"{content_length or 'not sent'}). The asset may not "
                        f"exist for this scene, or this may be a transient "
                        f"upstream issue — will retry per configured "
                        f"retry_attempts."
                    )

                output_paths.append(output_file)
                total_bytes += bytes_written
            except Exception as exc:
                self._logger.warning(f"Failed to download asset {key!r}: {exc}")
                asset_errors.append(f"{key}: {exc}")

        duration = time.time() - start_time

        if not output_paths:
            detail = "; ".join(asset_errors) if asset_errors else "unknown reason"
            return DownloadResult(
                status=DownloadStatus.FAILED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                error=f"No assets were successfully downloaded: {detail}",
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
            provider_id="aws_earth",
            name="AWS Earth Search",
            search=True,
            download=True,
            streaming=True,
            stac=True,
            requires_auth=False,
            supported_satellites=["Sentinel-2", "Landsat 8", "Landsat 9", "NAIP"],
            supported_formats=[DataFormat.COG, DataFormat.GEOTIFF],
            supports_cloud_filter=True,
            supports_date_filter=True,
            supports_aoi_filter=True,
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={
                "note": "AWS Earth Open Data is freely available with no download quotas",
                "docs": "https://registry.opendata.aws/",
            },
        )

    def list_collections(self) -> list[dict[str, Any]]:
        """
        List available STAC collections from Earth Search.

        Returns:
            List of collection info dicts.
        """
        import httpx

        response = httpx.get(f"{self.BASE_URL}/collections", timeout=30)
        self._handle_http_error(response)
        return response.json().get("collections", [])
