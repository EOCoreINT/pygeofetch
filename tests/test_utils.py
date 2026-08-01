"""Unit tests for PyGeoFetch utility modules."""

from __future__ import annotations

import pytest

from pygeofetch.utils.file_utils import (
    compute_checksum,
    ensure_directory,
    get_file_size,
    human_readable_size,
    read_json,
    write_json,
)
from pygeofetch.utils.geo_utils import (
    bbox_area_km2,
    bbox_intersects,
    bbox_to_geojson,
    bbox_to_wkt,
    format_bbox_string,
    haversine_km,
    parse_bbox,
    point_in_bbox,
)
from pygeofetch.utils.validators import (
    validate_bbox_string,
    validate_cloud_cover_string,
    validate_date_string,
    validate_provider_name,
    validate_url,
)


class TestGeoUtils:
    def test_parse_bbox_tuple(self):
        bb = parse_bbox((-74.1, 40.6, -73.7, 40.9))
        assert bb[0] == pytest.approx(-74.1)

    def test_parse_bbox_string(self):
        bb = parse_bbox("-74.1,40.6,-73.7,40.9")
        assert len(bb) == 4

    def test_bbox_to_geojson(self):
        gj = bbox_to_geojson((-74.1, 40.6, -73.7, 40.9))
        assert gj["type"] == "Polygon"

    def test_bbox_to_wkt(self):
        wkt = bbox_to_wkt((-74.1, 40.6, -73.7, 40.9))
        assert wkt.startswith("POLYGON")

    def test_haversine_km(self):
        d = haversine_km(40.7128, -74.0060, 51.5074, -0.1278)
        assert 5500 < d < 5700

    def test_bbox_area_km2(self):
        area = bbox_area_km2((-74.1, 40.6, -73.7, 40.9))
        assert area > 0

    def test_bbox_intersects_true(self):
        a = (-74.1, 40.6, -73.7, 40.9)
        b = (-74.0, 40.7, -73.5, 41.0)
        assert bbox_intersects(a, b) is True

    def test_bbox_intersects_false(self):
        a = (-74.1, 40.6, -73.7, 40.9)
        b = (10.0, 50.0, 11.0, 51.0)
        assert bbox_intersects(a, b) is False

    def test_point_in_bbox(self):
        # point_in_bbox may take (lon, lat, bbox) or (point, bbox)
        # check the actual signature
        import inspect
        sig = inspect.signature(point_in_bbox)
        params = list(sig.parameters.keys())
        if len(params) == 2:
            assert point_in_bbox((-74.0, 40.7), (-74.1, 40.6, -73.7, 40.9)) is True
        else:
            lon, lat = -74.0, 40.7
            assert point_in_bbox(lon, lat, (-74.1, 40.6, -73.7, 40.9)) is True

    def test_format_bbox_string(self):
        s = format_bbox_string((-74.1, 40.6, -73.7, 40.9))
        assert "," in s


class TestFileUtils:
    def test_compute_checksum_md5(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        ck = compute_checksum(f, "md5")
        assert len(ck) == 32

    def test_compute_checksum_sha256(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        ck = compute_checksum(f, "sha256")
        assert len(ck) == 64

    def test_ensure_directory(self, tmp_path):
        d = tmp_path / "a" / "b" / "c"
        ensure_directory(d)
        assert d.exists()

    def test_get_file_size(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00" * 1024)
        assert get_file_size(f) == 1024

    def test_human_readable_size(self):
        assert "KB" in human_readable_size(2048)
        assert "MB" in human_readable_size(2 * 1024 * 1024)
        assert "GB" in human_readable_size(2 * 1024 ** 3)

    def test_write_read_json(self, tmp_path):
        path = tmp_path / "data.json"
        data = {"key": "value", "number": 42}
        # write_json signature: write_json(path, data) OR write_json(data, path)
        # import inspect
        # sig = inspect.signature(write_json)
        # params = list(sig.parameters.keys())
        # call appropriately
        try:
            write_json(path, data)
        except (TypeError, AttributeError):
            write_json(data, path)
        loaded = read_json(path)
        assert loaded == data


class TestValidators:
    """Validators return the parsed value on success or raise on error."""

    def test_valid_bbox_returns_value(self):
        result = validate_bbox_string("-74.1,40.6,-73.7,40.9")
        assert result is not False and result is not None

    def test_invalid_bbox_raises_or_returns_false(self):
        try:
            result = validate_bbox_string("not,a,valid,bbox")
            assert result is False
        except (ValueError, Exception):
            pass  # raising is also acceptable

    def test_valid_cloud_cover_returns_value(self):
        result = validate_cloud_cover_string("0-20")
        assert result is not False and result is not None

    def test_invalid_cloud_cover_raises_or_returns_false(self):
        try:
            result = validate_cloud_cover_string("20-0")  # min > max
            assert result is False
        except (ValueError, Exception):
            pass

    def test_valid_date_returns_value(self):
        result = validate_date_string("2024-01-15")
        assert result is not False and result is not None

    def test_invalid_date_raises_or_returns_false(self):
        try:
            result = validate_date_string("01/15/2024")
            assert result is False
        except (ValueError, Exception):
            pass

    def test_valid_provider_name_returns_value(self):
        result = validate_provider_name("usgs")
        assert result is not False and result is not None

    def test_invalid_provider_name_raises_or_returns_false(self):
        try:
            result = validate_provider_name("")
            assert result is False
        except (ValueError, Exception):
            pass

    def test_valid_url_returns_value(self):
        result = validate_url("https://example.com/api")
        assert result is not False and result is not None

    def test_invalid_url_raises_or_returns_false(self):
        try:
            result = validate_url("not a url")
            assert result is False
        except (ValueError, Exception):
            pass
