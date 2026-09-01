"""
Regression tests for the NOAA Big Data provider.

Previously: search() hit a fictional "/search" REST endpoint on the S3
bucket's own hostname (a plain S3 bucket is not a search API), and the
parser never populated any downloadable asset -- so download() always
failed regardless of query. This verifies the real replacement, which
lists actual S3 object keys (mocked here, using the real key format
documented by AWS's own open-data-docs) and populates a real,
directly-downloadable HTTPS href for each.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from pygeofetch.models.search_query import SearchQuery
from pygeofetch.providers.noaa_big_data import _KEY_RE, NoaaBigDataProvider

# A real key format, taken directly from AWS's own documented example
# for the noaa-goes16 bucket.
REAL_KEY = "ABI-L2-CMIPF/2022/229/09/OR_ABI-L2-CMIPF-M6C06_G16_s20222290900207_e20222290909521_c20222290909577.nc"


@pytest.fixture
def provider():
    return NoaaBigDataProvider()


class TestRealFilenameParsing:
    def test_regex_matches_the_real_documented_filename(self):
        filename = REAL_KEY.rsplit("/", 1)[-1]
        m = _KEY_RE.match(filename)
        assert m is not None
        assert m.group("product") == "ABI-L2-CMIPF"
        assert m.group("sat") == "16"
        assert m.group("channel") == "06"

    def test_parse_key_populates_a_real_data_asset(self, provider):
        """REAL BUG FIXED: the old parser never populated any asset at
        all."""
        item = provider._parse_key("noaa-goes16", REAL_KEY, 7_500_000)
        assert item is not None
        assert (
            item.data_assets
        ), "data_assets must not be empty -- this was the core bug"
        asset = item.data_assets["data"]
        assert asset.href == f"https://noaa-goes16.s3.amazonaws.com/{REAL_KEY}"
        assert item.satellite == "GOES-16"
        assert item.datetime is not None
        assert item.datetime.year == 2022


class TestRealSearch:
    def test_search_lists_real_s3_objects_not_a_fictional_rest_endpoint(self, provider):
        """REAL BUG FIXED: search() used to call a fictional /search
        REST path on the bucket hostname."""
        query = SearchQuery(
            start_date=date(2022, 8, 17),
            end_date=date(2022, 8, 17),
            max_results=5,
        )
        fake_s3 = MagicMock()
        fake_s3.list_objects_v2.return_value = {
            "Contents": [{"Key": REAL_KEY, "Size": 7_500_000}]
        }
        with patch("boto3.client", return_value=fake_s3):
            results = provider.search(query)

        assert fake_s3.list_objects_v2.called
        call_kwargs = fake_s3.list_objects_v2.call_args.kwargs
        assert call_kwargs["Bucket"] == "noaa-goes16"
        assert call_kwargs["Prefix"].startswith("ABI-L2-CMIPF/2022/229/")

        assert len(results) >= 1
        assert results[0].data_assets

    def test_search_uses_unsigned_anonymous_requests(self, provider):
        """These buckets are genuinely public -- must never require
        real AWS credentials."""
        from botocore import UNSIGNED

        query = SearchQuery(start_date=date(2022, 8, 17), end_date=date(2022, 8, 17))
        with patch("boto3.client") as mock_client:
            mock_client.return_value.list_objects_v2.return_value = {"Contents": []}
            provider.search(query)

        call_kwargs = mock_client.call_args.kwargs
        assert call_kwargs["config"].signature_version is UNSIGNED

    def test_search_resolves_bucket_from_satellite_name(self, provider):
        query = SearchQuery(
            satellites=["GOES-18"],
            start_date=date(2022, 8, 17),
            end_date=date(2022, 8, 17),
        )
        with patch("boto3.client") as mock_client:
            mock_client.return_value.list_objects_v2.return_value = {"Contents": []}
            provider.search(query)
        assert (
            mock_client.return_value.list_objects_v2.call_args.kwargs["Bucket"]
            == "noaa-goes18"
        )

    def test_search_without_boto3_fails_gracefully(self, provider):
        with patch.dict("sys.modules", {"boto3": None}):
            results = provider.search(SearchQuery())
        assert results == []


class TestRealDownload:
    def test_download_streams_the_real_href(self, provider, tmp_path):
        item = provider._parse_key("noaa-goes16", REAL_KEY, 8)

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.headers = {"content-length": "8"}
        fake_resp.iter_bytes = lambda chunk_size: [b"fakedata"]
        fake_resp.__enter__ = lambda self: fake_resp
        fake_resp.__exit__ = lambda *a: None

        from pygeofetch.models.download_task import DownloadOptions

        with patch("httpx.stream", return_value=fake_resp) as mock_stream:
            result = provider.download(item, tmp_path, DownloadOptions())

        assert result.status.value == "completed"
        assert result.output_paths[0].read_bytes() == b"fakedata"
        assert (
            mock_stream.call_args[0][1]
            == f"https://noaa-goes16.s3.amazonaws.com/{REAL_KEY}"
        )

    def test_download_with_no_assets_fails_clearly(self, provider, tmp_path):
        from pygeofetch.models.download_task import DownloadOptions
        from pygeofetch.models.satellite_data import SatelliteData

        empty_item = SatelliteData(id="x", provider="noaa_big_data")
        result = provider.download(empty_item, tmp_path, DownloadOptions())
        assert result.status.value == "failed"
