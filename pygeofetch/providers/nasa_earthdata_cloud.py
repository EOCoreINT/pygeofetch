"""
NASA Earthdata Cloud provider for PyGeoFetch.

Provides direct S3 access to cloud-hosted NASA datasets. Uses Earthdata
Login (OAuth2) to obtain temporary AWS credentials, then accesses data
directly from S3 without downloading through NASA servers.

Supports cloud-hosted MODIS, VIIRS, ICESat-2, GEDI, ECCO, and many more.

Example::

    from pygeofetch.providers.nasa_earthdata_cloud import NASAEarthdataCloudProvider
    from pygeofetch.models.user_auth import Credentials

    provider = NASAEarthdataCloudProvider()
    creds = Credentials(provider="nasa_earthdata_cloud",
                        username="myuser", password="mypass")
    session = provider.authenticate(creds)
    results = provider.search(query)
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
    ProviderCapabilities,
    QuotaInfo,
    SatelliteData,
)
from pygeofetch.models.user_auth import AuthSession, Credentials
from pygeofetch.providers.base import (
    AbstractBaseProvider,
    AuthenticationError,
    DownloadError,
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


class NASAEarthdataCloudProvider(AbstractBaseProvider):
    """
    NASA Earthdata Cloud provider — direct S3 access to NASA datasets.

    Uses the EDL (Earthdata Login) OAuth2 flow to obtain temporary AWS
    credentials. Data is then read directly from S3, avoiding bandwidth
    charges and latency vs. downloading through NASA servers.

    Attributes:
        PROVIDER_ID: 'nasa_earthdata_cloud'
        REQUIRES_AUTH: True
    """

    PROVIDER_ID = "nasa_earthdata_cloud"
    DISPLAY_NAME = "NASA Earthdata Cloud"
    REQUIRES_AUTH = True
    DESCRIPTION = (
        "Direct cloud access to petabyte-scale NASA Earth science data on AWS S3 "
        "via Earthdata Login. Supports MODIS, VIIRS, ICESat-2, GEDI, and more."
    )
    DATA_TYPES = ["MODIS", "VIIRS", "ICESat-2", "GEDI", "ECCO", "GPM", "MERRA-2"]
    SATELLITES = ["Terra", "Aqua", "CALIPSO", "ICESat-2", "GEDI"]
    BASE_URL = "https://cmr.earthdata.nasa.gov/search"
    EDL_URL = "https://urs.earthdata.nasa.gov"

    # Real, confirmed finding: there is no single, unified S3 credentials
    # endpoint. Each DAAC has its own, separate endpoint, and credentials
    # from one DAAC do NOT work for another DAAC's buckets -- confirmed
    # identically across NASA's own EMIT-Data-Resources GitHub repo,
    # PO.DAAC's own cookbook, NSIDC's own help center, and multiple NASA
    # Openscapes cloud-workshop tutorials, all stating this explicitly
    # ("each DAAC has their own endpoint and credentials are not usable
    # for cloud data from other DAACs"). The previous single
    # "https://data.earthaccess.nasa.gov/s3credentials" URL doesn't match
    # any real DAAC's actual endpoint -- it appears to conflate the real
    # `earthaccess` Python library's name with a fictional hostname.
    # This list covers the DAACs independently confirmed here; a DAAC
    # not in this mapping raises a clear error rather than silently
    # using the wrong (or a fictional) endpoint.
    DAAC_S3_CREDENTIALS_ENDPOINTS = {
        "PODAAC": "https://archive.podaac.earthdata.nasa.gov/s3credentials",
        "GES_DISC": "https://data.gesdisc.earthdata.nasa.gov/s3credentials",
        "LPDAAC_ECS": "https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials",
        "ORNL_DAAC": "https://data.ornldaac.earthdata.nasa.gov/s3credentials",
        "GHRC_DAAC": "https://data.ghrc.earthdata.nasa.gov/s3credentials",
        "NSIDC_ECS": "https://data.nsidc.earthdatacloud.nasa.gov/s3credentials",
    }

    def authenticate(self, credentials: Credentials) -> AuthSession:
        """
        Authenticate with NASA Earthdata Login and obtain S3 credentials.

        Args:
            credentials: Username/password for Earthdata Login.

        Returns:
            AuthSession containing S3 credentials in session_data.

        Raises:
            AuthenticationError: If Earthdata login fails.
        """
        import httpx

        if not credentials.username or not credentials.password:
            msg = (
                "NASA Earthdata Cloud requires username and password. "
                "Register at: https://urs.earthdata.nasa.gov"
            )
            raise AuthenticationError(msg)

        # Step 1: EDL login to get bearer token
        try:
            resp = httpx.post(
                f"{self.EDL_URL}/api/users/tokens",
                auth=(credentials.username, credentials.password.get_secret_value()),
                timeout=30,
            )
            if resp.status_code == 401:
                msg = "Invalid Earthdata Login credentials."
                raise AuthenticationError(msg)
            if resp.status_code not in (200, 201):
                msg = f"EDL auth failed: HTTP {resp.status_code}"
                raise AuthenticationError(msg)

            token_data = resp.json()
            edl_token = (
                token_data[0].get("access_token")
                if isinstance(token_data, list)
                else token_data.get("access_token")
            )

        except AuthenticationError:
            raise
        except Exception as exc:
            msg = f"EDL authentication error: {exc}"
            raise AuthenticationError(msg) from exc

        # Real, confirmed fix: S3 credentials are no longer fetched
        # eagerly here at all -- doing so against one generic endpoint
        # made no sense once "each DAAC has its own endpoint" was
        # confirmed (see DAAC_S3_CREDENTIALS_ENDPOINTS above). Credentials
        # are now fetched lazily, per real DAAC, in download() itself
        # once the specific granule's real data_center is known --
        # cached there per-DAAC since each set expires after about an
        # hour and different DAACs' credentials are never interchangeable.
        from datetime import datetime, timedelta, timezone

        session = AuthSession(
            provider=self.PROVIDER_ID,
            access_token=edl_token,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            session_data={
                "username": credentials.username,
                "s3_credentials_by_daac": {},
            },
        )
        self._session = session
        self._logger.info(
            f"NASA Earthdata Cloud: authenticated as {credentials.username}"
        )
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        return bool(credentials.username and credentials.password)

    def set_session(self, session: Any) -> None:
        """Store an authenticated session for use in requests."""
        self._session = session

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        """
        Search NASA CMR for cloud-hosted granules.

        Args:
            query: Search parameters.

        Returns:
            List of SatelliteData with S3 URLs.
        """
        # Real, confirmed fix (same finding as pygeofetch.providers.
        # nasa_earthdata): CMR search is genuinely public, including
        # with cloud_hosted=true -- no Earthdata credentials needed for
        # search itself, only for the S3/HTTPS download step below.
        import httpx

        params: dict[str, Any] = {
            "cloud_hosted": "true",
            "page_size": min(query.max_results, 2000),
        }

        # Spatial filter
        if query.bbox:
            bb = query.bbox
            params["bounding_box"] = (
                f"{bb.min_lon},{bb.min_lat},{bb.max_lon},{bb.max_lat}"
            )

        # Temporal filter
        if query.start_date or query.end_date:
            start = str(query.start_date) + "T00:00:00Z" if query.start_date else ""
            end = str(query.end_date) + "T23:59:59Z" if query.end_date else ""
            params["temporal"] = f"{start},{end}"

        # Collection shortnames
        if query.satellites:
            params["short_name[]"] = query.satellites

        try:
            headers = {}
            if self._session and self._session.access_token:
                headers["Authorization"] = f"Bearer {self._session.access_token}"
            resp = httpx.get(
                f"{self.BASE_URL}/granules.json",
                params=params,
                headers=headers,
                timeout=self.config.get("timeout", 60),
            )
            self._handle_http_error(resp)
            data = resp.json()
            entries = data.get("feed", {}).get("entry", [])
            results = [self._parse_granule(e) for e in entries]
            self._logger.info(f"NASA Earthdata Cloud: {len(results)} granules found")
            return results

        except Exception as exc:
            msg = f"NASA Earthdata Cloud search failed: {exc}"
            raise SearchError(msg) from exc

    def _parse_granule(self, entry: dict[str, Any]) -> SatelliteData:
        """Convert a CMR granule entry to SatelliteData."""
        from pygeofetch.models.satellite_data import SatelliteAsset

        granule_id = entry.get("id", "")
        bbox_raw = entry.get("boxes", [""])[0].split() if entry.get("boxes") else []
        bbox = None
        if len(bbox_raw) == 4:
            try:
                bbox = (
                    float(bbox_raw[1]),
                    float(bbox_raw[0]),
                    float(bbox_raw[3]),
                    float(bbox_raw[2]),
                )
            except ValueError:
                pass

        assets: dict[str, SatelliteAsset] = {}
        for link in entry.get("links", []):
            href = link.get("href", "")
            link.get("rel", "")
            if "s3://" in href or href.endswith(
                (".nc", ".h5", ".hdf", ".tif", ".zarr")
            ):
                key = f"data_{len(assets)}"
                assets[key] = SatelliteAsset(
                    key=key,
                    href=href,
                    roles=["data"],
                    media_type=link.get("type"),
                )

        return SatelliteData(
            id=granule_id,
            provider=self.PROVIDER_ID,
            collection=entry.get("collection_concept_id"),
            satellite=entry.get("data_center", "NASA"),
            bbox=bbox,
            cloud_cover=None,
            assets=assets,
            properties={
                "title": entry.get("title", ""),
                "producer_granule_id": entry.get("producer_granule_id", ""),
                "cloud_hosted": True,
                "data_center": entry.get("data_center", ""),
            },
        )

    def _get_s3_credentials_for_daac(self, data_center: str) -> dict[str, str]:
        """
        Real, lazy, per-DAAC S3 credential fetch -- cached on the
        session and refreshed once expired (these are genuinely
        short-lived, confirmed ~1 hour by NSIDC's own documentation).
        Returns {} (not an error) when the DAAC isn't in the real,
        confirmed mapping -- callers fall back to HTTPS, which always
        works regardless of DAAC.
        """
        from datetime import datetime, timedelta, timezone

        import httpx

        endpoint = self.DAAC_S3_CREDENTIALS_ENDPOINTS.get(data_center)
        if not endpoint:
            self._logger.info(
                f"NASA Earthdata Cloud: no confirmed S3 credentials endpoint for "
                f"data_center={data_center!r} -- falling back to HTTPS download."
            )
            return {}

        cache = self._session.session_data.setdefault("s3_credentials_by_daac", {})
        cached = cache.get(data_center)
        if cached:
            expiration = cached.get("_expires_at")
            if expiration and datetime.now(timezone.utc) < expiration:
                return cached

        try:
            resp = httpx.get(
                endpoint,
                headers={"Authorization": f"Bearer {self._session.access_token}"},
                timeout=30,
            )
            if resp.status_code != 200:
                self._logger.warning(
                    f"NASA Earthdata Cloud: S3 credentials request for "
                    f"{data_center} failed (HTTP {resp.status_code}) -- "
                    f"falling back to HTTPS download."
                )
                return {}
            creds_data = resp.json()
            creds = {
                "aws_access_key_id": creds_data.get("accessKeyId", ""),
                "aws_secret_access_key": creds_data.get("secretAccessKey", ""),
                "aws_session_token": creds_data.get("sessionToken", ""),
                "expiration": creds_data.get("expiration", ""),
                "_expires_at": datetime.now(timezone.utc) + timedelta(minutes=55),
            }
            cache[data_center] = creds
            self._logger.info(
                f"NASA Earthdata Cloud: obtained real S3 credentials for {data_center}"
            )
            return creds
        except Exception as exc:
            self._logger.warning(
                f"NASA Earthdata Cloud: could not obtain S3 credentials for "
                f"{data_center} ({exc}) -- falling back to HTTPS download."
            )
            return {}

    def download(
        self,
        data: SatelliteData,
        destination: Path,
        options: DownloadOptions,
    ) -> DownloadResult:
        """
        Download or directly access NASA cloud data.

        Prefers direct S3 access if AWS credentials are available,
        falls back to HTTPS download.

        Args:
            data: SatelliteData to download.
            destination: Output directory.
            options: Download configuration.

        Returns:
            DownloadResult.
        """
        self.require_auth()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        # Real, confirmed fix: fetch S3 credentials lazily, for THIS
        # granule's real DAAC specifically -- not once, eagerly, from a
        # single generic (and fictional) endpoint during authenticate().
        # See DAAC_S3_CREDENTIALS_ENDPOINTS above for why this has to be
        # per-DAAC. Falls back to HTTPS download (still real and
        # correct) when the DAAC isn't in the confirmed mapping, rather
        # than failing the whole download.
        data_center = (data.properties or {}).get("data_center", "")
        s3_creds = self._get_s3_credentials_for_daac(data_center)
        start_time = time.time()
        output_paths = []
        total_bytes = 0

        for key, asset in (data.data_assets or data.assets).items():
            href = asset.href
            if not href:
                continue

            filename = href.split("/")[-1] or f"{data.id}_{key}"
            out_file = destination / filename

            if out_file.exists() and not getattr(options, "overwrite", False):
                output_paths.append(out_file)
                continue

            try:
                if href.startswith("s3://") and s3_creds.get("aws_access_key_id"):
                    self._download_s3(href, out_file, s3_creds)
                else:
                    self._download_https(href, out_file, options)
                output_paths.append(out_file)
                total_bytes += out_file.stat().st_size
            except Exception as exc:
                self._logger.warning(f"Failed to download {key}: {exc}")

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

    def _download_s3(self, s3_uri: str, out_file: Path, creds: dict[str, str]) -> None:
        """Download from S3 using temporary credentials."""
        try:
            import boto3

            s3 = boto3.client(
                "s3",
                aws_access_key_id=creds["aws_access_key_id"],
                aws_secret_access_key=creds["aws_secret_access_key"],
                aws_session_token=creds.get("aws_session_token"),
            )
            # Parse s3://bucket/key
            parts = s3_uri.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1] if len(parts) > 1 else ""
            s3.download_file(bucket, key, str(out_file))
            self._logger.debug(f"S3 download: {s3_uri} → {out_file}")
        except ImportError:
            msg = "boto3 not installed. Install with: pip install boto3"
            raise DownloadError(msg)

    def _download_https(
        self, url: str, out_file: Path, options: DownloadOptions
    ) -> None:
        """HTTPS fallback download with Bearer token auth."""
        import httpx

        headers = {}
        if self._session and (self._session.access_token if self._session else None):
            headers["Authorization"] = (
                f"Bearer {(self._session.access_token if self._session else None)}"
            )
        with httpx.stream(
            "GET",
            url,
            headers=headers,
            timeout=options.timeout_seconds,
            follow_redirects=True,
        ) as resp:
            self._handle_http_error(resp)
            with open(out_file, "wb") as f:
                f.writelines(
                    resp.iter_bytes(chunk_size=int(options.chunk_size_mb * 1024 * 1024))
                )

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_id=self.PROVIDER_ID,
            name=self.DISPLAY_NAME,
            description=self.DESCRIPTION,
            auth_type="oauth2",
            satellites=["Terra", "Aqua", "CALIPSO", "ICESat-2", "GEDI"],
            search=True,
            download=True,
            streaming=True,
            stac=True,
            supports_direct_s3=True,
            supports_aoi_filter=True,
            supports_date_filter=True,
            requires_auth=True,
            regions=["global"],
            endpoint_url=self.BASE_URL,
            docs_url="https://www.earthdata.nasa.gov/esds/cloud-evolution",
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={
                "note": "No download quota; data accessed directly from AWS S3."
            },
        )