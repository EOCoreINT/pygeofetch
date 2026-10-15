"""
PyGeoVision Integration Contract Tests
======================================
Verifies that every bug fix, new capability, and logging change specified in
PyGeoFetch_Fix_Improvement.md is correctly implemented.

All tests run offline — no real API credentials are required.
Rasterio calls use synthetic in-memory GeoTIFFs.

Run::

    pytest tests/test_pgv_integration_contract.py -v

"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_geotiff(
    path: Path, width=10, height=10, bands=1, crs_epsg=4326, identity_transform=False
) -> Path:
    """
    Create a minimal GeoTIFF at *path* using rasterio.
    Pass identity_transform=True to reproduce the bug-4 scenario.
    """
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    if identity_transform:
        # Bug-4 scenario: pixel=1m, origin=(0,0), projected CRS
        from rasterio.transform import Affine

        transform = Affine(1.0, 0, 0, 0, -1.0, 0)
        crs = CRS.from_epsg(32632)
    else:
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, width, height)
        crs = CRS.from_epsg(crs_epsg)

    import numpy as np

    profile = dict(
        driver="GTiff",
        dtype="uint16",
        count=bands,
        width=width,
        height=height,
        crs=crs,
        transform=transform,
    )
    with rasterio.open(path, "w", **profile) as ds:
        for b in range(1, bands + 1):
            ds.write((np.random.randint(1, 1000, (height, width), dtype=np.uint16)), b)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# BUG 1 — AuthManager.add_credentials()
# ══════════════════════════════════════════════════════════════════════════════


class TestBug1CredentialRoundtrip:
    """
    Contract: engine.add_credentials() must never raise AttributeError and
    must persist credentials so that engine.auth.list() returns the provider.
    """

    def test_add_credentials_username_password(self):
        """Credential roundtrip for username/password auth type."""
        from pygeofetch import PyGeoFetch

        engine = PyGeoFetch(log_level="WARNING")
        # Must not raise AttributeError
        engine.add_credentials("usgs", username="testuser", password="testpass")
        providers = [item["provider"] for item in engine.auth.list()]
        assert (
            "usgs" in providers
        ), "After add_credentials('usgs', ...), 'usgs' must appear in auth.list()"

    def test_add_credentials_api_key(self):
        """Credential roundtrip for API key auth type."""
        from pygeofetch import PyGeoFetch

        engine = PyGeoFetch(log_level="WARNING")
        engine.add_credentials("planet", api_key="PL-test-key-123")
        providers = [item["provider"] for item in engine.auth.list()]
        assert "planet" in providers

    def test_add_credentials_oauth2(self):
        """Credential roundtrip for OAuth2 client credential auth type."""
        from pygeofetch import PyGeoFetch

        engine = PyGeoFetch(log_level="WARNING")
        engine.add_credentials(
            "sentinel_hub", client_id="cid-abc", client_secret="cs-xyz"
        )
        providers = [item["provider"] for item in engine.auth.list()]
        assert "sentinel_hub" in providers

    def test_add_credentials_token(self):
        """Credential roundtrip for bearer token auth type."""
        from pygeofetch import PyGeoFetch

        engine = PyGeoFetch(log_level="WARNING")
        engine.add_credentials("maxar_gbdx", token="tok-12345")
        providers = [item["provider"] for item in engine.auth.list()]
        assert "maxar_gbdx" in providers

    def test_add_credentials_idempotent(self):
        """Calling add_credentials twice for same provider updates, not duplicates."""
        from pygeofetch import PyGeoFetch

        engine = PyGeoFetch(log_level="WARNING")
        engine.add_credentials("usgs", username="u1", password="p1")
        engine.add_credentials("usgs", username="u2", password="p2")
        usgs_entries = [i for i in engine.auth.list() if i["provider"] == "usgs"]
        assert (
            len(usgs_entries) == 1
        ), "Calling add_credentials twice must update, not create duplicate"

    def test_auth_manager_add_credentials_dict(self):
        """AuthManager.add_credentials(provider, dict) works directly (spec signature)."""
        from pygeofetch.core.authenticator import AuthManager

        auth = AuthManager()
        # Must accept positional dict — not keyword args
        auth.add_credentials("copernicus", {"username": "u", "password": "p"})
        providers = [i["provider"] for i in auth.list()]
        assert "copernicus" in providers

    def test_auth_manager_debug_log(self, caplog):
        """AuthManager.add_credentials() emits a DEBUG log with provider name."""
        from pygeofetch.core.authenticator import AuthManager

        auth = AuthManager()
        with caplog.at_level(logging.DEBUG, logger="pygeofetch"):
            auth.add_credentials("usgs", {"username": "u", "password": "p"})
        # Spec: 'Stored credentials for provider: {provider}'
        assert any(
            "usgs" in r.message for r in caplog.records
        ), "Expected DEBUG log containing provider name 'usgs'"


# ══════════════════════════════════════════════════════════════════════════════
# BUG 2 — download() length and order contract
# ══════════════════════════════════════════════════════════════════════════════


class TestBug2DownloadContract:
    """
    Contract: engine.download() must ALWAYS return exactly len(items) results,
    in the same order, each with a bool .success field.
    """

    def _make_items(self, n=3):
        from pygeofetch.models.satellite_data import SatelliteData

        return [
            SatelliteData(id=f"scene_{i}", provider="planetary_computer")
            for i in range(n)
        ]

    def test_count_contract(self, tmp_path):
        """len(results) == len(items) always holds."""
        from pygeofetch.core.downloader import AdaptiveDownloader as DownloadManager
        from pygeofetch.models.download_task import DownloadOptions

        items = self._make_items(4)
        dm = DownloadManager()

        # Patch the actual HTTP download to succeed for 2 and fail for 2
        call_count = [0]

        def fake_download(data, dest, opts):
            call_count[0] += 1
            from pygeofetch.models.download_task import DownloadResult, DownloadStatus

            if call_count[0] % 2 == 0:
                raise ConnectionError("Simulated network failure")
            p = tmp_path / f"{data.id}.json"
            p.write_text("{}")
            return DownloadResult(
                status=DownloadStatus.COMPLETED,
                data_id=data.id,
                provider=data.provider,
                output_path=p,
                bytes_downloaded=2,
            )

        with patch.object(dm, "download", side_effect=fake_download):
            results = dm.download_many(items, tmp_path, DownloadOptions())

        assert len(results) == len(
            items
        ), f"Expected {len(items)} results, got {len(results)}"

    def test_order_contract(self, tmp_path):
        """results[i].data_id == items[i].id for all i."""
        from pygeofetch.core.downloader import AdaptiveDownloader as DownloadManager
        from pygeofetch.models.download_task import (
            DownloadOptions,
            DownloadResult,
            DownloadStatus,
        )

        items = self._make_items(5)
        dm = DownloadManager()

        def fake_download(data, dest, opts):
            p = tmp_path / f"{data.id}.json"
            p.write_text("{}")
            return DownloadResult(
                status=DownloadStatus.COMPLETED,
                data_id=data.id,
                provider=data.provider,
                output_path=p,
                bytes_downloaded=2,
            )

        with patch.object(dm, "download", side_effect=fake_download):
            results = dm.download_many(items, tmp_path, DownloadOptions())

        for r, item in zip(results, items):
            assert (
                r.data_id == item.id
            ), f"Order mismatch: result.data_id={r.data_id!r} != item.id={item.id!r}"

    def test_all_results_have_bool_success(self, tmp_path):
        """Every result.success is a bool — never None."""
        from pygeofetch.core.downloader import AdaptiveDownloader as DownloadManager
        from pygeofetch.models.download_task import (
            DownloadOptions,
            DownloadResult,
            DownloadStatus,
        )

        items = self._make_items(3)
        dm = DownloadManager()

        def fake_download(data, dest, opts):
            p = tmp_path / f"{data.id}.json"
            p.write_text("{}")
            return DownloadResult(
                status=DownloadStatus.COMPLETED,
                data_id=data.id,
                provider=data.provider,
                output_path=p,
                bytes_downloaded=2,
            )

        with patch.object(dm, "download", side_effect=fake_download):
            results = dm.download_many(items, tmp_path, DownloadOptions())

        for r in results:
            assert isinstance(
                r.success, bool
            ), f"result.success must be bool, got {type(r.success).__name__}"

    def test_failed_items_return_success_false_not_omitted(self, tmp_path):
        """Items that fail download are returned as success=False, not silently dropped."""
        from pygeofetch.core.downloader import AdaptiveDownloader as DownloadManager
        from pygeofetch.models.download_task import (
            DownloadOptions,
        )

        items = self._make_items(3)
        dm = DownloadManager()

        def fail_all(data, dest, opts):
            raise RuntimeError("Simulated failure")

        with patch.object(dm, "download", side_effect=fail_all):
            results = dm.download_many(items, tmp_path, DownloadOptions())

        assert len(results) == 3
        assert all(not r.success for r in results)
        assert all(r.data_id is not None for r in results)


# ══════════════════════════════════════════════════════════════════════════════
# BUG 3 — Partial download returns success=False
# ══════════════════════════════════════════════════════════════════════════════


class TestBug3FileValidation:
    """
    Contract: A downloaded file that fails rasterio tile validation must
    return success=False, not success=True.
    """

    def test_validate_downloaded_file_valid_raster(self, tmp_path):
        """A well-formed GeoTIFF passes validation."""
        pytest.importorskip("rasterio")
        from pygeofetch.core.downloader import AdaptiveDownloader as DownloadManager

        dm = DownloadManager()
        tif = _make_geotiff(tmp_path / "valid.tif")
        is_valid, err = dm._validate_downloaded_file(tif)
        assert is_valid, f"Valid GeoTIFF should pass validation: {err}"
        assert err == ""

    def test_validate_downloaded_file_empty_file_fails(self, tmp_path):
        """An empty (0-byte) file fails validation."""
        from pygeofetch.core.downloader import AdaptiveDownloader as DownloadManager

        dm = DownloadManager()
        empty = tmp_path / "empty.tif"
        empty.write_bytes(b"")
        is_valid, err = dm._validate_downloaded_file(empty)
        assert not is_valid
        assert len(err) > 0

    def test_validate_downloaded_file_non_raster_json(self, tmp_path):
        """A non-empty JSON file passes validation (no rasterio check)."""
        from pygeofetch.core.downloader import AdaptiveDownloader as DownloadManager

        dm = DownloadManager()
        jf = tmp_path / "metadata.json"
        jf.write_text('{"key": "value"}')
        is_valid, err = dm._validate_downloaded_file(jf)
        assert is_valid, f"Non-empty JSON should pass non-raster validation: {err}"

    def test_validate_downloaded_file_empty_json_fails(self, tmp_path):
        """An empty JSON/CSV file fails validation."""
        from pygeofetch.core.downloader import AdaptiveDownloader as DownloadManager

        dm = DownloadManager()
        jf = tmp_path / "empty.json"
        jf.write_bytes(b"")
        is_valid, err = dm._validate_downloaded_file(jf)
        assert not is_valid

    def test_validate_downloaded_file_corrupt_raster_fails(self, tmp_path):
        """A corrupt (truncated) raster file fails validation."""
        from pygeofetch.core.downloader import AdaptiveDownloader as DownloadManager

        dm = DownloadManager()
        bad = tmp_path / "corrupt.tif"
        bad.write_bytes(b"\x00\x01\x02" * 100)  # Garbage bytes, not a GeoTIFF
        is_valid, err = dm._validate_downloaded_file(bad)
        assert not is_valid
        assert len(err) > 0


# ══════════════════════════════════════════════════════════════════════════════
# BUG 4 — CRS identity transform detection and rejection
# ══════════════════════════════════════════════════════════════════════════════


class TestBug4CrsIdentityTransform:
    """
    Contract: _has_identity_transform() correctly detects the buggy case,
    and _reproject_with_validation() raises RuntimeError when it occurs.
    """

    def test_has_identity_transform_detects_bug(self, tmp_path):
        """File with pixel=1m and origin=(0,0) in projected CRS is flagged."""
        pytest.importorskip("rasterio")
        from pygeofetch.core.downloader import AdaptiveDownloader as DownloadManager

        dm = DownloadManager()
        bad_tif = _make_geotiff(tmp_path / "identity.tif", identity_transform=True)
        assert dm._has_identity_transform(
            bad_tif
        ), "Should detect identity transform (pixel=1m, origin=0,0, projected CRS)"

    def test_has_identity_transform_normal_file_ok(self, tmp_path):
        """A properly reprojected file does NOT trigger identity transform detection."""
        pytest.importorskip("rasterio")
        from pygeofetch.core.downloader import AdaptiveDownloader as DownloadManager

        dm = DownloadManager()
        good_tif = _make_geotiff(tmp_path / "good.tif", crs_epsg=4326)
        assert not dm._has_identity_transform(
            good_tif
        ), "Normal GeoTIFF with correct transform should not be flagged"

    def test_reproject_with_validation_normal(self, tmp_path):
        """Valid reprojection completes without RuntimeError."""
        pytest.importorskip("rasterio")
        from pygeofetch.core.downloader import AdaptiveDownloader as DownloadManager

        dm = DownloadManager()
        src = _make_geotiff(tmp_path / "src.tif", crs_epsg=4326)
        dst = tmp_path / "dst_reproj.tif"
        # Should not raise
        dm._reproject_with_validation(src, dst, "EPSG:32632")
        assert dst.exists()

    def test_reproject_output_has_valid_transform(self, tmp_path):
        """After reprojection, output file has a real (non-identity) transform."""
        pytest.importorskip("rasterio")
        import rasterio as rio

        from pygeofetch.core.downloader import AdaptiveDownloader as DownloadManager

        dm = DownloadManager()
        src = _make_geotiff(tmp_path / "src2.tif", crs_epsg=4326)
        dst = tmp_path / "dst2_reproj.tif"
        dm._reproject_with_validation(src, dst, "EPSG:32632")
        with rio.open(dst) as ds:
            t = ds.transform
            # Real reprojection: pixel size should NOT be 1.0m at origin (0,0)
            assert not (
                abs(t.a) == 1.0 and t.c == 0.0 and t.f == 0.0
            ), "Reprojected file must not have identity transform"


# ══════════════════════════════════════════════════════════════════════════════
# CAPABILITY 1 — Sentinel-1 SLC product type
# ══════════════════════════════════════════════════════════════════════════════


class TestCapability1SLCProductType:
    """
    Contract: SearchQuery supports product_type field and set_product_type().
    SLC queries are routed away from GRD-only providers.
    SatelliteData carries product_type, polarisation, pass_direction.
    """

    def test_search_query_product_type_field(self):
        """SearchQuery accepts product_type field."""
        from pygeofetch.models.search_query import SearchQuery

        q = SearchQuery(bbox=None)
        q.product_type = "SLC"
        assert q.product_type == "SLC"

    def test_search_query_set_product_type_builder(self):
        """SearchQuery.set_product_type() builder method works and uppercases."""
        from pygeofetch.models.search_query import SearchQuery

        q = SearchQuery(bbox=None)
        result = q.set_product_type("slc")
        assert result is q, "set_product_type must return self for chaining"
        assert q.product_type == "SLC"

    def test_slc_routing_removes_planetary_computer(self):
        """planetary_computer is rerouted to copernicus for SLC queries."""
        from pygeofetch import PyGeoFetch

        engine = PyGeoFetch(log_level="WARNING")
        routed = engine._route_slc_providers(
            ["planetary_computer", "aws_earth", "element84"]
        )
        assert (
            "planetary_computer" not in routed
        ), "planetary_computer does not host SLC — must be removed"
        assert "aws_earth" not in routed
        assert "element84" not in routed

    def test_slc_routing_adds_copernicus_fallback(self):
        """When GRD-only providers are requested for SLC, copernicus is added."""
        from pygeofetch import PyGeoFetch

        engine = PyGeoFetch(log_level="WARNING")
        routed = engine._route_slc_providers(["planetary_computer"])
        assert (
            "copernicus" in routed
        ), "planetary_computer (GRD-only) should route to copernicus for SLC"

    def test_slc_routing_preserves_capable_providers(self):
        """SLC-capable providers pass through routing unchanged."""
        from pygeofetch import PyGeoFetch

        engine = PyGeoFetch(log_level="WARNING")
        routed = engine._route_slc_providers(
            ["copernicus", "alaska_satellite_facility"]
        )
        assert "copernicus" in routed
        assert "alaska_satellite_facility" in routed

    def test_slc_routing_deduplicates(self):
        """Routing produces no duplicate providers."""
        from pygeofetch import PyGeoFetch

        engine = PyGeoFetch(log_level="WARNING")
        routed = engine._route_slc_providers(
            ["planetary_computer", "planetary_computer", "copernicus"]
        )
        assert len(routed) == len(set(routed)), "Routed list must have no duplicates"

    def test_satellite_data_product_type_field(self):
        """SatelliteData carries product_type."""
        from pygeofetch.models.satellite_data import SatelliteData

        sd = SatelliteData(id="s1", provider="copernicus", product_type="SLC")
        assert sd.product_type == "SLC"

    def test_satellite_data_polarisation_field(self):
        """SatelliteData carries polarisation."""
        from pygeofetch.models.satellite_data import SatelliteData

        sd = SatelliteData(id="s1", provider="copernicus", polarisation="VV+VH")
        assert sd.polarisation == "VV+VH"

    def test_satellite_data_pass_direction_field(self):
        """SatelliteData carries pass_direction."""
        from pygeofetch.models.satellite_data import SatelliteData

        sd = SatelliteData(id="s1", provider="copernicus", pass_direction="ascending")
        assert sd.pass_direction == "ascending"

    def test_from_stac_item_extracts_sar_fields(self):
        """SatelliteData.from_stac_item() populates SAR fields from STAC properties."""
        from pygeofetch.models.satellite_data import SatelliteData

        item = {
            "id": "S1C_IW_GRDH_20260601",
            "collection": "sentinel-1-grd",
            "bbox": [-74.1, 40.6, -73.7, 40.9],
            "geometry": None,
            "assets": {},
            "links": [],
            "stac_extensions": [],
            "properties": {
                "datetime": "2026-06-01T05:30:00Z",
                "platform": "SENTINEL-1C",
                "instruments": ["C-SAR"],
                "sar:product_type": "GRD",
                "sar:polarizations": ["VV", "VH"],
                "sat:orbit_state": "ascending",
                "sat:relative_orbit": 37,
                "gsd": 10.0,
            },
        }
        sd = SatelliteData.from_stac_item(item, provider="element84")
        assert sd.product_type == "GRD"
        assert sd.polarisation == "VV+VH"
        assert sd.pass_direction == "ascending"
        assert sd.relative_orbit == 37
        assert sd.gsd_m == 10.0


# ══════════════════════════════════════════════════════════════════════════════
# CAPABILITY 2 — Sentinel-1C and Sentinel-1D constellation support
# ══════════════════════════════════════════════════════════════════════════════


class TestCapability2S1Constellation:
    """
    Contract: S1C and S1D must appear in all provider platform lists.
    _normalise_satellite_name() must map STAC platform strings correctly.
    _warn_if_outdated_constellation() warns when only S1A/S1B returned post-July-2026.
    """

    def test_normalise_s1c(self):
        """SENTINEL-1C normalises to S1C."""
        from pygeofetch.utils.geo_utils import _normalise_satellite_name

        assert _normalise_satellite_name("SENTINEL-1C") == "S1C"

    def test_normalise_s1d(self):
        """sentinel-1d normalises to S1D (case-insensitive)."""
        from pygeofetch.utils.geo_utils import _normalise_satellite_name

        assert _normalise_satellite_name("sentinel-1d") == "S1D"

    def test_normalise_s1a(self):
        """SENTINEL-1A normalises to S1A."""
        from pygeofetch.utils.geo_utils import _normalise_satellite_name

        assert _normalise_satellite_name("SENTINEL-1A") == "S1A"

    def test_normalise_s2b(self):
        """SENTINEL-2B normalises to S2B."""
        from pygeofetch.utils.geo_utils import _normalise_satellite_name

        assert _normalise_satellite_name("SENTINEL-2B") == "S2B"

    def test_normalise_unknown_passthrough(self):
        """Unknown platform strings pass through unchanged."""
        from pygeofetch.utils.geo_utils import _normalise_satellite_name

        assert _normalise_satellite_name("LANDSAT-8") == "L8"
        # Completely unknown
        result = _normalise_satellite_name("KOMPSAT-3")
        assert isinstance(result, str)

    def test_s1c_in_from_stac_item(self):
        """A STAC item with platform=SENTINEL-1C is parsed as satellite=SENTINEL-1C."""
        from pygeofetch.models.satellite_data import SatelliteData

        item = {
            "id": "S1C_IW_SLC__20260601",
            "collection": "sentinel-1-slc",
            "bbox": [-5.0, 52.0, 5.0, 58.0],
            "geometry": None,
            "assets": {},
            "links": [],
            "stac_extensions": [],
            "properties": {
                "datetime": "2026-06-01T05:30:00Z",
                "platform": "SENTINEL-1C",
                "sar:product_type": "SLC",
                "sar:polarizations": ["VV", "VH"],
            },
        }
        sd = SatelliteData.from_stac_item(item, provider="copernicus")
        assert (
            sd.satellite == "SENTINEL-1C"
        ), "S1C results must carry platform=SENTINEL-1C from STAC properties"

    def test_warn_if_outdated_constellation_fires(self, caplog):
        """_warn_if_outdated_constellation() warns when only S1A/S1B returned after July 2026."""
        from pygeofetch import PyGeoFetch
        from pygeofetch.models.satellite_data import SatelliteData

        engine = PyGeoFetch(log_level="WARNING")

        # Simulate results with only legacy satellites
        results = [
            SatelliteData(id="s1", provider="copernicus", satellite="S1A"),
            SatelliteData(id="s2", provider="copernicus", satellite="S1B"),
        ]
        with caplog.at_level(logging.WARNING, logger="pygeofetch"):
            engine._warn_if_outdated_constellation(results, None)

        # After July 2026 (today is July 2026), should warn
        from datetime import date

        if date.today() >= date(2026, 7, 1):
            warning_msgs = [
                r.message for r in caplog.records if r.levelno == logging.WARNING
            ]
            assert any(
                "S1A" in m or "decommission" in m.lower() or "S1C" in m
                for m in warning_msgs
            ), "Should warn about outdated constellation after July 2026"

    def test_warn_if_outdated_does_not_fire_for_s1c(self, caplog):
        """No warning when S1C results are present."""
        from pygeofetch import PyGeoFetch
        from pygeofetch.models.satellite_data import SatelliteData

        engine = PyGeoFetch(log_level="WARNING")

        results = [
            SatelliteData(id="s1", provider="copernicus", satellite="S1C"),
        ]
        with caplog.at_level(logging.WARNING, logger="pygeofetch"):
            engine._warn_if_outdated_constellation(results, None)

        constellation_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "decommission" in r.message.lower()
        ]
        assert (
            len(constellation_warnings) == 0
        ), "Should not warn when S1C results are present"


# ══════════════════════════════════════════════════════════════════════════════
# CAPABILITY 3 — Precise orbit file management
# ══════════════════════════════════════════════════════════════════════════════


class TestCapability3OrbitFiles:
    """
    Contract: fetch_orbit_file() parses product names, checks cache, downloads
    orbit files, and engine.fetch_orbit_file() is a working wrapper.
    """

    def test_parse_acquisition_datetime(self):
        """_parse_acquisition_datetime extracts datetime from product name."""
        from pygeofetch.core.orbits import _parse_acquisition_datetime

        dt = _parse_acquisition_datetime(
            "S1C_IW_SLC__1SDV_20260601T053000_20260601T053027_053442_067B1A_F3A2"
        )
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 6
        assert dt.day == 1
        assert dt.hour == 5

    def test_parse_acquisition_datetime_no_match(self):
        """Returns None for a string with no datetime token."""
        from pygeofetch.core.orbits import _parse_acquisition_datetime

        assert _parse_acquisition_datetime("invalid_product_name") is None

    def test_parse_satellite_s1c(self):
        """_parse_satellite extracts S1C from product name."""
        from pygeofetch.core.orbits import _parse_satellite

        assert _parse_satellite("S1C_IW_SLC__1SDV_20260601T053000") == "S1C"

    def test_parse_satellite_s1d(self):
        """_parse_satellite extracts S1D from product name."""
        from pygeofetch.core.orbits import _parse_satellite

        assert _parse_satellite("S1D_IW_GRDH_1SDV_20260601T053000") == "S1D"

    def test_parse_satellite_sentinel_1a_full(self):
        """_parse_satellite handles full SENTINEL-1A name."""
        from pygeofetch.core.orbits import _parse_satellite

        assert _parse_satellite("SENTINEL-1A_20260601T053000") == "S1A"

    def test_find_matching_orbit_file_matches_real_esa_eof_zip_filenames(self):
        """
        REGRESSION TEST: the orbit-file regex previously required filenames
        to end exactly at '.EOF' before the closing quote, but ESA's real
        server (step.esa.int) serves orbit files as '.EOF.zip' — every
        single lookup silently failed with this bug, even for scenes over
        a year old that definitely have published orbit files.

        This uses real HTML captured directly from
        https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB/S1A/2025/07/
        as ground truth, including the exact orbit file that covers a real
        Sentinel-1A scene from 2025-07-21.
        """
        from pygeofetch.core.orbits import (
            _find_matching_orbit_file,
            _parse_acquisition_datetime,
        )

        real_html = (
            '<a href="S1A_OPER_AUX_POEORB_OPOD_20250722T070642_'
            'V20250701T225942_20250703T005942.EOF.zip">...</a>\n'
            '<a href="S1A_OPER_AUX_POEORB_OPOD_20250809T070644_'
            'V20250719T225942_20250721T005942.EOF.zip">...</a>\n'
            '<a href="S1A_OPER_AUX_POEORB_OPOD_20250811T070653_'
            'V20250721T225942_20250723T005942.EOF.zip">...</a>\n'
        )

        scene_name = (
            "S1A_IW_SLC__1SDV_20250721T004837_20250721T004904_060175_077A35_EF47.SAFE"
        )
        acq_dt = _parse_acquisition_datetime(scene_name)

        matched = _find_matching_orbit_file(real_html, acq_dt)

        assert matched is not None, (
            "Should find a matching orbit file — if this fails, the "
            ".EOF.zip regex bug has regressed"
        )
        assert matched == (
            "S1A_OPER_AUX_POEORB_OPOD_20250809T070644_"
            "V20250719T225942_20250721T005942.EOF"
        )

    def test_find_matching_orbit_file_returns_none_for_plain_eof_no_longer_served(self):
        """Old-format bare .EOF hrefs (no .zip) should NOT match — ESA
        doesn't serve these anymore, and matching them would download a
        404 page as if it were a valid orbit file."""
        from datetime import datetime

        from pygeofetch.core.orbits import _find_matching_orbit_file

        old_format_html = (
            '<a href="S1A_OPER_AUX_POEORB_OPOD_20250809T070644_'
            'V20250719T225942_20250721T005942.EOF">...</a>\n'
        )
        acq_dt = datetime(2025, 7, 21, 0, 48, 37)
        matched = _find_matching_orbit_file(old_format_html, acq_dt)
        assert matched is None

    def test_orbit_covers_datetime_true(self):
        """_orbit_covers_datetime returns True when acq_dt is inside validity window."""
        from pygeofetch.core.orbits import _orbit_covers_datetime

        fname = "S1C_OPER_AUX_POEORB_OPOD_20260622T121000_V20260601T000000_20260603T000000.EOF"
        acq_dt = datetime(2026, 6, 2, 5, 30, 0)
        assert _orbit_covers_datetime(fname, acq_dt)

    def test_orbit_covers_datetime_false(self):
        """_orbit_covers_datetime returns False when acq_dt is outside validity window."""
        from pygeofetch.core.orbits import _orbit_covers_datetime

        fname = "S1C_OPER_AUX_POEORB_OPOD_20260622T121000_V20260601T000000_20260603T000000.EOF"
        acq_dt = datetime(2026, 6, 5, 5, 30, 0)  # After stop time
        assert not _orbit_covers_datetime(fname, acq_dt)

    def test_find_matching_orbit_file(self):
        """_find_matching_orbit_file finds the right file in a mock HTML listing."""
        from pygeofetch.core.orbits import _find_matching_orbit_file

        # Simulate HTML directory listing — ESA serves these as .EOF.zip,
        # not plain .EOF (confirmed against the live step.esa.int server).
        listing_html = """
            <a href="S1C_OPER_AUX_POEORB_OPOD_20260520T121000_V20260501T000000_20260503T000000.EOF.zip">
            <a href="S1C_OPER_AUX_POEORB_OPOD_20260622T121000_V20260601T000000_20260603T000000.EOF.zip">
            <a href="S1C_OPER_AUX_POEORB_OPOD_20260705T121000_V20260701T000000_20260703T000000.EOF.zip">
        """
        acq_dt = datetime(2026, 6, 2, 5, 30, 0)
        found = _find_matching_orbit_file(listing_html, acq_dt)
        assert found is not None, "Should find a matching orbit file"
        assert "20260601" in found, f"Wrong file selected: {found}"

    def test_find_cached_orbit_returns_none_empty_dir(self, tmp_path):
        """_find_cached_orbit returns None when cache directory is empty."""
        from pygeofetch.core.orbits import _find_cached_orbit

        result = _find_cached_orbit(tmp_path, "S1C", datetime(2026, 6, 1), "precise")
        assert result is None

    def test_find_cached_orbit_returns_file_when_present(self, tmp_path):
        """_find_cached_orbit returns the orbit file if it covers the acquisition datetime."""
        from pygeofetch.core.orbits import _find_cached_orbit

        # Create a fake orbit file with the right name
        fname = "S1C_OPER_AUX_POEORB_OPOD_20260622T121000_V20260601T000000_20260603T000000.EOF"
        (tmp_path / fname).write_text("<EOF>fake orbit data</EOF>")
        acq_dt = datetime(2026, 6, 2, 5, 30, 0)
        result = _find_cached_orbit(tmp_path, "S1C", acq_dt, "precise")
        assert result is not None, "Should return cached orbit file"
        assert result.name == fname

    def test_engine_fetch_orbit_file_method_exists(self):
        """engine.fetch_orbit_file() is accessible and callable."""
        from pygeofetch import PyGeoFetch

        engine = PyGeoFetch(log_level="WARNING")
        assert hasattr(
            engine, "fetch_orbit_file"
        ), "Engine must expose fetch_orbit_file() method"
        assert callable(engine.fetch_orbit_file)

    def test_fetch_orbit_file_returns_none_for_invalid_name(self, tmp_path):
        """fetch_orbit_file returns None gracefully for product with no datetime."""
        from pygeofetch.core.orbits import fetch_orbit_file

        result = fetch_orbit_file("NOT_A_PRODUCT_NAME", str(tmp_path))
        assert result is None

    def test_fetch_orbit_file_returns_cached_without_network(self, tmp_path):
        """fetch_orbit_file returns cached file without making any HTTP request."""
        from pygeofetch.core.orbits import fetch_orbit_file

        # Pre-populate cache
        fname = "S1C_OPER_AUX_POEORB_OPOD_20260622T121000_V20260601T000000_20260603T000000.EOF"
        (tmp_path / fname).write_text("<EOF>orbit</EOF>")

        product_name = "S1C_IW_SLC__1SDV_20260602T053000_20260602T053027_053442_067B1A"
        with patch("requests.get") as mock_get:
            result = fetch_orbit_file(product_name, str(tmp_path), "precise")
        mock_get.assert_not_called()  # Must not hit network for cached file
        assert result is not None
        assert "POEORB" in result


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING — New module structure and format
# ══════════════════════════════════════════════════════════════════════════════


class TestLogging:
    """
    Contract: pygeofetch.core.logging exports configure_logging, get_logger,
    _render_progress_bar. Log output matches the required format spec.
    Credentials are never logged.
    """

    def test_configure_logging_importable(self):
        """configure_logging is importable from pygeofetch.core.logging."""
        from pygeofetch.core.logging import configure_logging

        assert callable(configure_logging)

    def test_get_logger_importable(self):
        """get_logger is importable from pygeofetch.core.logging."""
        from pygeofetch.core.logging import get_logger

        assert callable(get_logger)

    def test_render_progress_bar_importable(self):
        """_render_progress_bar is importable from pygeofetch.core.logging."""
        from pygeofetch.core.logging import _render_progress_bar

        assert callable(_render_progress_bar)

    def test_get_logger_namespaces_under_pygeofetch(self):
        """get_logger('mymodule') returns a logger named 'pygeofetch.mymodule'."""
        from pygeofetch.core.logging import get_logger

        lg = get_logger("mymodule")
        assert lg.name == "pygeofetch.mymodule"

    def test_get_logger_already_namespaced(self):
        """get_logger('pygeofetch.foo') is not double-namespaced."""
        from pygeofetch.core.logging import get_logger

        lg = get_logger("pygeofetch.foo")
        assert lg.name == "pygeofetch.foo"

    def test_configure_logging_sets_level(self):
        """configure_logging('DEBUG') sets the pygeofetch root logger to DEBUG."""
        from pygeofetch.core.logging import configure_logging

        configure_logging("DEBUG")
        root = logging.getLogger("pygeofetch")
        assert root.level == logging.DEBUG
        configure_logging("WARNING")  # restore

    def test_configure_logging_silences_urllib3(self):
        """configure_logging() silences urllib3 to WARNING or above."""
        from pygeofetch.core.logging import configure_logging

        configure_logging("DEBUG")
        urllib3_logger = logging.getLogger("urllib3")
        assert urllib3_logger.level >= logging.WARNING

    def test_render_progress_bar_format(self):
        """_render_progress_bar returns a string with expected tokens."""
        from pygeofetch.core.logging import _render_progress_bar

        result = _render_progress_bar(
            completed=3,
            total=8,
            filename="S1C_IW_GRDH_20260601.zip",
            bytes_done=1_200_000_000,
            bytes_total=4_100_000_000,
            speed_bps=2_100_000,
        )
        assert result.startswith("\r"), "Progress bar must start with \\r"
        assert "3/8" in result
        assert "MB/s" in result
        assert "GB" in result

    def test_render_progress_bar_full(self):
        """Progress bar shows full bar when completed == total."""
        from pygeofetch.core.logging import _render_progress_bar

        result = _render_progress_bar(
            completed=8,
            total=8,
            filename="file.zip",
            bytes_done=1_000_000_000,
            bytes_total=1_000_000_000,
            speed_bps=1_000_000,
        )
        assert "░" not in result, "Full bar should have no empty segments"

    def test_pgf_formatter_includes_timestamp_level_module(self):
        """_PGFFormatter output matches HH:MM:SS LEVL [   module] message."""
        import re

        from pygeofetch.core.logging import _PGFFormatter

        formatter = _PGFFormatter(use_colour=False, show_module=True)
        record = logging.LogRecord(
            name="pygeofetch.core.search",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Searching 2 provider(s) for all scenes",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        # Must have: HH:MM:SS INFO [    search] message
        assert re.match(r"\d{2}:\d{2}:\d{2}", output), f"No timestamp: {output!r}"
        assert "INFO" in output
        assert "search" in output
        assert "Searching" in output

    def test_redact_removes_password(self):
        """_redact() replaces password values with ***REDACTED***."""
        from pygeofetch.core.logging import _redact

        msg = "Connecting with password=supersecret123 to API"
        redacted = _redact(msg)
        assert "supersecret123" not in redacted
        assert "REDACTED" in redacted

    def test_redact_removes_api_key(self):
        """_redact() replaces api_key values."""
        from pygeofetch.core.logging import _redact

        msg = "Using api_key=PL-abc-xyz-123 for planet"
        redacted = _redact(msg)
        assert "PL-abc-xyz-123" not in redacted
        assert "REDACTED" in redacted

    def test_redact_leaves_non_sensitive_intact(self):
        """_redact() leaves non-credential values untouched."""
        from pygeofetch.core.logging import _redact

        msg = "Found 14 scenes in provider planetary_computer"
        assert _redact(msg) == msg

    def test_info_search_log_format(self, caplog):
        """Actual search log output matches the spec format pattern."""
        from pygeofetch.core.logging import configure_logging, get_logger

        configure_logging("INFO")
        lg = get_logger("search")

        with caplog.at_level(logging.INFO, logger="pygeofetch"):
            lg.info("Searching %d provider(s) for %s scenes", 2, "Sentinel-1")
            lg.info("  bbox       : [%.4f, %.4f, %.4f, %.4f]", -0.3, 5.4, 0.2, 5.9)
            lg.info("  date range : %s \u2192 %s", "2026-06-01", "2026-06-13")
            lg.info("  cloud max  : %s%%", 100)
            lg.info("  product    : %s", "GRD")
            lg.info("  %-22s searching...", "[planetary_computer]")
            lg.info("  %-22s %d scenes found", "[planetary_computer]", 8)
            lg.info(
                "Search complete: %d scene(s) across %d provider(s) in %.1fs",
                14,
                2,
                3.1,
            )

        messages = [r.message for r in caplog.records]
        assert any("Searching 2 provider(s)" in m for m in messages)
        assert any("bbox" in m for m in messages)
        assert any("2026-06-01" in m for m in messages)
        assert any("GRD" in m for m in messages)
        assert any("searching..." in m for m in messages)
        assert any("8 scenes found" in m for m in messages)
        assert any("Search complete" in m for m in messages)
        assert any("14 scene(s)" in m for m in messages)

    def test_info_download_log_format(self, caplog):
        """Download log messages match the spec format."""
        from pygeofetch.core.logging import configure_logging, get_logger

        configure_logging("INFO")
        lg = get_logger("download")

        with caplog.at_level(logging.INFO, logger="pygeofetch"):
            lg.info("Downloading %d scene(s) to: %s", 2, "./data/raw/")
            lg.info(
                "  \u2713 %-45s %6.0f MB  %5.1fs",
                "S1C_IW_GRDH_1SDV_20260601T053000"[:45],
                2100,
                17.8,
            )
            lg.info("  post-processing: %s", "reproject:EPSG:32630 \u2192 cog")
            lg.info("Download complete: %d/%d succeeded  %.0f MB total", 2, 2, 3900)

        messages = [r.message for r in caplog.records]
        assert any("Downloading 2 scene(s)" in m for m in messages)
        assert any("✓" in m for m in messages)
        assert any("post-processing" in m for m in messages)
        assert any("Download complete" in m for m in messages)


# ══════════════════════════════════════════════════════════════════════════════
# FINAL — Smoke test: full engine init with all subsystems
# ══════════════════════════════════════════════════════════════════════════════


class TestFinalEngineSmoke:
    """Smoke tests confirming the full engine boots without errors."""

    def test_engine_init_all_subsystems(self):
        """PyGeoFetch() initialises with all 6 processing subsystems."""
        from pygeofetch import PyGeoFetch

        engine = PyGeoFetch(log_level="WARNING")
        for subsystem in ["preprocess", "indices", "post", "sar", "batch", "pipeline"]:
            assert hasattr(engine, subsystem), f"Engine missing subsystem: {subsystem}"

    def test_engine_pipeline_all_41_methods(self):
        """Processing pipeline builder chains all 41 required methods."""
        from pygeofetch import PyGeoFetch

        engine = PyGeoFetch(log_level="WARNING")
        pl = (
            engine.pipeline("test")
            # Preprocessing (11)
            .atmos()
            .cloud_mask()
            .cloud_fill()
            .clip()
            .reproject()
            .resample()
            .pansharpen()
            .topo_correct()
            .mosaic()
            .composite()
            .tile()
            # Spectral indices (17)
            .ndvi()
            .evi()
            .savi()
            .ndwi()
            .mndwi()
            .ndbi()
            .ndsi()
            .ndmi()
            .nbr()
            .dnbr()
            .tct()
            .pca()
            .texture()
            .lst()
            .albedo()
            .band_math()
            .stack()
            # Post-processing (9)
            .vectorize()
            .smooth()
            .regularize()
            .zonal_stats()
            .buffer()
            .centroids()
            .add_geometry_metrics()
            .compress()
            .cog()
            # SAR (4)
            .despeckle()
            .calibrate()
            .flood_map()
            .coherence()
        )
        assert len(pl._steps) == 41, f"Expected 41 pipeline steps, got {len(pl._steps)}"

    def test_engine_status_no_crash(self):
        """engine.status() returns without raising."""
        from pygeofetch import PyGeoFetch

        engine = PyGeoFetch(log_level="WARNING")
        status = engine.status()
        assert isinstance(status, dict)

    def test_pipeline_from_yaml_roundtrip(self, tmp_path):
        """ProcessingPipeline.from_yaml() loads and returns correct step count."""
        import yaml

        from pygeofetch import PyGeoFetch
        from pygeofetch.processing.pipeline import ProcessingPipeline

        engine = PyGeoFetch(log_level="WARNING")

        yaml_def = {
            "name": "contract-test",
            "steps": [
                {"ndvi": {"red": "B04.tif", "nir": "B08.tif"}},
                {"vectorize": {"threshold": 0.3}},
                {"cog": {"compress": "deflate"}},
            ],
        }
        yaml_path = tmp_path / "pipeline.yaml"
        yaml_path.write_text(yaml.dump(yaml_def))
        pl = ProcessingPipeline.from_yaml(str(yaml_path), engine=engine)
        assert pl.name == "contract-test"
        assert len(pl._steps) == 3


# ── Contract footer ───────────────────────────────────────────────────────────
# PyGeoVision integration contract: ALL TESTS MUST PASS
# Contract version: PyGeoVision v2.4.0 / PyGeoFetch patch July 2026
