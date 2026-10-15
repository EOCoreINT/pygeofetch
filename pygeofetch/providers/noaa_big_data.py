"""
NOAA Big Data provider for PyGeoFetch.

GOES-16/17/18 imagery on AWS Open Data -- real, public, unsigned S3
buckets, no account or API key needed.

Real, verified structure (confirmed against AWS's own
awslabs/open-data-docs and the registry.opendata.aws listing for
noaa-goes16/17/18):

- Buckets: noaa-goes16, noaa-goes17, noaa-goes18 (us-east-1), fully
  public, listable and downloadable with UNSIGNED (anonymous) requests.
- Key layout: <product>/<year>/<julian_day>/<hour>/
  OR_<product>-M<mode>C<channel>_G<sat>_s<start>_e<end>_c<created>.nc
  e.g. ABI-L2-CMIPF/2022/229/09/OR_ABI-L2-CMIPF-M6C06_G16_s20222290900207_e20222290909521_c20222290909577.nc
- Files are NetCDF4 (.nc), directly GET-able over plain HTTPS at
  https://<bucket>.s3.amazonaws.com/<key> -- no signing needed for
  reads either.

REAL BUG FIXED: the previous implementation invented a fictional
"/search" REST endpoint on the bucket's own hostname (which is a plain
S3 bucket, not a search API) and never populated any downloadable
asset -- so download() always failed with "No assets downloaded". GOES
data isn't discoverable via a search API at all; it's discoverable by
listing real S3 keys under a real date/hour prefix, which is what this
rewrite actually does.
"""

from __future__ import annotations

import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pygeofetch.core.logging import report_download_progress
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
from pygeofetch.models.search_query import SearchQuery
from pygeofetch.models.user_auth import AuthSession, Credentials
from pygeofetch.providers.base import AbstractBaseProvider

_KEY_RE = re.compile(
    r"OR_(?P<product>[A-Za-z0-9-]+)-M(?P<mode>\d)C(?P<channel>\d{2})_"
    r"G(?P<sat>\d+)_s(?P<start>\d{14})\d?_e(?P<end>\d{14})\d?_c\d+\.nc$"
)


def _julian_start_to_datetime(value: str) -> datetime | None:
    """Parse a GOES filename's sYYYYDDDHHMMSS timestamp into a real
    datetime."""
    try:
        year = int(value[0:4])
        day_of_year = int(value[4:7])
        hour = int(value[7:9])
        minute = int(value[9:11])
        second = int(value[11:13])
        return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
            days=day_of_year - 1, hours=hour, minutes=minute, seconds=second
        )
    except (ValueError, IndexError):
        return None


class NoaaBigDataProvider(AbstractBaseProvider):
    PROVIDER_ID = "noaa_big_data"
    DISPLAY_NAME = "NOAA Big Data"
    REQUIRES_AUTH = False
    DESCRIPTION = "GOES-16/17/18 imagery on AWS Open Data. Public, unsigned S3 buckets."
    SATELLITES = ["GOES-16", "GOES-17", "GOES-18"]
    BASE_URL = "https://noaa-goes16.s3.amazonaws.com"

    DEFAULT_PRODUCT = "ABI-L2-CMIPF"

    def authenticate(self, credentials: Credentials) -> AuthSession:
        # No real credentials are needed or used -- these buckets are
        # genuinely public. A session is still returned for interface
        # consistency with authenticated providers.
        session = AuthSession(
            provider=self.PROVIDER_ID,
            access_token="anonymous",
            expires_at=datetime.now(timezone.utc) + timedelta(days=365),
            session_data={},
        )
        self._session = session
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        return True

    def set_session(self, session: Any) -> None:
        self._session = session

    def _resolve_bucket(self, query: SearchQuery) -> str:
        for sat in query.satellites or []:
            m = re.search(r"(16|17|18)", sat)
            if m:
                return f"noaa-goes{m.group(1)}"
        return "noaa-goes16"

    def _resolve_product(self, query: SearchQuery) -> str:
        return query.collections[0] if query.collections else self.DEFAULT_PRODUCT

    @staticmethod
    def _iter_hour_prefixes(product: str, start: datetime, end: datetime):
        """Yield one real S3 prefix per hour between start and end --
        GOES keys are partitioned by year/julian-day/hour, so listing
        must walk that same partitioning rather than guess a single
        broad prefix."""
        cur = start.replace(minute=0, second=0, microsecond=0)
        end = end.replace(minute=59, second=59)
        seen = 0
        while cur <= end and seen < 24 * 14:  # hard cap: 2 weeks of hours
            julian_day = cur.timetuple().tm_yday
            yield f"{product}/{cur.year}/{julian_day:03d}/{cur.hour:02d}/"
            cur += timedelta(hours=1)
            seen += 1

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        """
        List real objects from the real, public GOES S3 bucket for the
        requested date range -- there is no search API to call.
        """
        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config
        except ImportError:
            self._logger.warning(
                f"{self.DISPLAY_NAME}: boto3 is required for search/download "
                "(pip install boto3)."
            )
            return []

        bucket = self._resolve_bucket(query)
        product = self._resolve_product(query)
        max_results = min(query.max_results or 100, 500)

        now = datetime.now(timezone.utc)
        start = (
            self._to_datetime(query.start_date)
            if query.start_date
            else now - timedelta(hours=6)
        )
        end = self._to_datetime(query.end_date) if query.end_date else now

        s3 = boto3.client(
            "s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1"
        )

        results: list[SatelliteData] = []
        for prefix in self._iter_hour_prefixes(product, start, end):
            try:
                resp = s3.list_objects_v2(
                    Bucket=bucket, Prefix=prefix, MaxKeys=max_results
                )
            except Exception as exc:
                self._logger.warning(
                    f"{self.DISPLAY_NAME}: listing s3://{bucket}/{prefix} failed: {exc}"
                )
                continue
            for obj in resp.get("Contents", []):
                item = self._parse_key(bucket, obj["Key"], obj.get("Size", 0))
                if item:
                    results.append(item)
            if len(results) >= max_results:
                break

        return results[:max_results]

    @staticmethod
    def _to_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, date):
            return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
        return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc)

    def _parse_key(
        self, bucket: str, key: str, size_bytes: int
    ) -> SatelliteData | None:
        """
        Parse a real S3 object key into SatelliteData, with a real,
        directly-downloadable HTTPS asset -- the core of the fix, since
        the previous version never populated any asset at all.
        """
        filename = key.rsplit("/", 1)[-1]
        m = _KEY_RE.match(filename)
        start_dt = _julian_start_to_datetime(m.group("start")) if m else None
        sat_number = m.group("sat") if m else None
        product = m.group("product") if m else key.split("/", 1)[0]
        channel = m.group("channel") if m else None

        href = f"https://{bucket}.s3.amazonaws.com/{key}"
        satellite = f"GOES-{sat_number}" if sat_number else self.DISPLAY_NAME

        return SatelliteData(
            id=filename,
            provider=self.PROVIDER_ID,
            satellite=satellite,
            collection=product,
            datetime=start_dt,
            data_format=DataFormat.NETCDF,
            assets={
                "data": SatelliteAsset(
                    key="data",
                    href=href,
                    title=filename,
                    media_type="application/x-netcdf",
                    roles=["data"],
                    size_bytes=size_bytes or None,
                )
            },
            properties={"s3_key": key, "bucket": bucket, "channel": channel},
        )

    def download(
        self, data: SatelliteData, destination: Path, options: DownloadOptions
    ) -> DownloadResult:
        """
        Download the real object over plain HTTPS -- these public
        buckets need no request signing for GET either, so this is a
        normal streaming download once given a real href (which
        search() now actually provides).
        """
        import httpx

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        start = time.time()
        output_paths, total_bytes = [], 0

        assets = data.data_assets or data.assets
        if not assets:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                error="No downloadable assets on this item.",
            )

        for key, asset in assets.items():
            if not asset.href or not asset.href.startswith("http"):
                continue
            out_file = destination / (asset.href.split("/")[-1] or f"{data.id}_{key}")
            try:
                with httpx.stream(
                    "GET",
                    asset.href,
                    timeout=options.timeout_seconds,
                    follow_redirects=True,
                ) as resp:
                    self._handle_http_error(resp)
                    total_bytes_this_asset = int(resp.headers.get("content-length", 0))
                    bytes_written_this_asset = 0
                    chunk_t0 = time.time()
                    with open(out_file, "wb") as f:
                        for chunk in resp.iter_bytes(
                            chunk_size=int(options.chunk_size_mb * 1024 * 1024)
                        ):
                            f.write(chunk)
                            bytes_written_this_asset += len(chunk)
                            elapsed = time.time() - chunk_t0
                            speed = (
                                bytes_written_this_asset / elapsed
                                if elapsed > 0
                                else 0.0
                            )
                            report_download_progress(
                                bytes_written_this_asset, total_bytes_this_asset, speed
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
            supports_aoi_filter=False,
            supports_cloud_filter=False,
            supports_date_filter=True,
            requires_auth=self.REQUIRES_AUTH,
            has_quota=False,
            regions=["global"],
            resolution_min_m=500.0,
            resolution_max_m=10000.0,
            endpoint_url=self.BASE_URL,
            docs_url="https://registry.opendata.aws/noaa-goes/",
            supported_formats=[DataFormat.GEOTIFF],
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={
                "note": "Public AWS Open Data bucket -- no quota or account needed."
            },
        )
