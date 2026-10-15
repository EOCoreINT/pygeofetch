"""
Regression tests for pygeofetch.validation.optical_validator against
real Element84 / AWS Earth STAC response shapes.

These 4 scenarios were reported as "critical production bugs" causing
a 100% rejection rate against real STAC catalogs. Empirically
re-verified against the current source before writing these tests:

- Bugs 1 (band naming) and 2 (processing baseline vs. level) are
  genuinely fixed -- confirmed here with real STAC item shapes.
- Bug 4 (cloud-cover hard-failure default) does not reproduce -- the
  described failure mode (WARN config still rejecting the scene) is
  not present in the current source; tested here to lock that in.
- Bug 3 (AOI coverage for tiled satellites) is not a bug: the check
  correctly reports low coverage when a scene genuinely only covers a
  fraction of a wider AOI -- this is real geometry, already
  configurable via ``min_coverage_ratio``, not a defect. Tested here
  as "behaves correctly," not "fixed."
"""

from __future__ import annotations

import pytest
from shapely.geometry import box

from pygeofetch.models.satellite_data import SatelliteData
from pygeofetch.validation import OpticalPreflightValidator, OpticalValidationConfig


def _real_element84_item(
    scene_id="S2A_18TXK_20240615_0_L2A",
    collection="sentinel-2-l2a",
    bbox=(-74.2, 40.5, -73.6, 41.0),
    cloud_cover=12.0,
    bands=("red", "green", "blue", "nir", "scl"),
    processing_baseline="05.10",
):
    """A STAC item shaped exactly like a real Element84/AWS Earth
    response: semantic asset keys, only s2:processing_baseline (no
    processing:level), no top-level bbox-vs-flat-band mismatch."""
    minx, miny, maxx, maxy = bbox
    return {
        "id": scene_id,
        "collection": collection,
        "bbox": list(bbox),
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]],
        },
        "properties": {
            "datetime": "2024-06-15T15:39:41Z",
            "eo:cloud_cover": cloud_cover,
            "platform": "sentinel-2a",
            "s2:processing_baseline": processing_baseline,
        },
        "assets": {b: {"href": f"https://example.com/{b}.tif"} for b in bands},
    }


def _real_eo_bands_item(
    scene_id="S2A_TEST_EOBANDS",
    bbox=(-74.2, 40.5, -73.6, 41.0),
):
    """A STAC item using the eo:bands extension nested inside each
    asset -- a real, valid alternative shape some catalogs use instead
    of (or alongside) a plain semantic asset key."""
    minx, miny, maxx, maxy = bbox
    return {
        "id": scene_id,
        "collection": "sentinel-2-l2a",
        "bbox": list(bbox),
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]],
        },
        "properties": {
            "datetime": "2024-06-15T15:39:41Z",
            "eo:cloud_cover": 5.0,
            "s2:product_type": "S2MSI2A",
        },
        "assets": {
            "B02": {"href": "https://x/B02.tif", "eo:bands": [{"common_name": "blue"}]},
            "B03": {"href": "https://x/B03.tif", "eo:bands": [{"common_name": "green"}]},
            "B04": {"href": "https://x/B04.tif", "eo:bands": [{"common_name": "red"}]},
            "B08": {"href": "https://x/B08.tif", "eo:bands": [{"common_name": "nir"}]},
            "SCL": {"href": "https://x/SCL.tif"},
        },
    }


AOI_SINGLE_TILE = box(-74.0, 40.6, -73.8, 40.8)  # comfortably within one tile's bbox


@pytest.fixture
def validator():
    return OpticalPreflightValidator(OpticalValidationConfig())


class TestBug1BandNamingAgainstRealSTAC:
    def test_element84_semantic_band_names_pass(self, validator):
        scene = SatelliteData.from_stac_item(_real_element84_item(), "aws_earth")
        report = validator.validate_scene(scene, AOI_SINGLE_TILE)
        assert report.passed, f"unexpected issues: {report.issues}"

    def test_eo_bands_extension_shape_passes(self, validator):
        scene = SatelliteData.from_stac_item(_real_eo_bands_item(), "element84")
        report = validator.validate_scene(scene, AOI_SINGLE_TILE)
        assert report.passed, f"unexpected issues: {report.issues}"

    def test_genuinely_missing_band_still_rejected(self, validator):
        scene = SatelliteData.from_stac_item(
            _real_element84_item(bands=("red", "green", "blue")), "aws_earth"  # no nir, no scl
        )
        report = validator.validate_scene(scene, AOI_SINGLE_TILE)
        assert report.passed is False
        assert report.errors[0].code == "MISSING_BANDS"


class TestBug2ProcessingBaselineVsLevel:
    def test_real_l2a_scene_with_baseline_property_passes(self, validator):
        """The specific real failure mode: a scene with
        s2:processing_baseline='05.10' and no processing:level field
        must not have '05.10' mistaken for the processing level."""
        scene = SatelliteData.from_stac_item(_real_element84_item(processing_baseline="05.10"), "aws_earth")
        report = validator.validate_scene(scene, AOI_SINGLE_TILE)
        assert report.passed, f"unexpected issues: {report.issues}"

    def test_processing_baseline_value_never_appears_as_the_level(self, validator):
        scene = SatelliteData.from_stac_item(_real_element84_item(processing_baseline="05.11"), "aws_earth")
        report = validator.validate_scene(scene, AOI_SINGLE_TILE)
        for issue in report.issues:
            assert "05.11" not in issue.message

    def test_s2msi2a_product_type_recognised_as_l2a(self, validator):
        scene = SatelliteData.from_stac_item(_real_eo_bands_item(), "element84")
        assert validator.validate_processing_level(scene) is True

    def test_genuinely_wrong_level_still_rejected(self, validator):
        scene = SatelliteData.from_stac_item(
            _real_element84_item(collection="sentinel-2-l1c"), "aws_earth"
        )
        report = validator.validate_scene(scene, AOI_SINGLE_TILE)
        assert report.passed is False
        assert report.errors[0].code == "PROCESSING_LEVEL_MISMATCH"


class TestBug3AoiCoverageIsRealGeometryNotABug:
    """Not "fixed" -- verified as already-correct behavior. A scene
    that genuinely only covers a fraction of a wider AOI SHOULD report
    low coverage; that's the check doing its job, not a defect."""

    def test_full_single_tile_coverage_passes_default_threshold(self, validator):
        scene = SatelliteData.from_stac_item(_real_element84_item(), "aws_earth")
        report = validator.validate_scene(scene, AOI_SINGLE_TILE)
        assert report.metrics["aoi_coverage"] == pytest.approx(1.0)
        assert report.passed is True

    def test_partial_coverage_from_a_real_multi_tile_aoi_correctly_flagged(self, validator):
        # A scene footprint covering only the western quarter of a
        # wider AOI -- the real situation when an AOI spans more than
        # one MGRS tile.
        narrow_scene_bbox = (-74.2, 40.5, -73.9, 40.75)
        scene = SatelliteData.from_stac_item(
            _real_element84_item(bbox=narrow_scene_bbox), "aws_earth"
        )
        wide_aoi = box(-74.1, 40.6, -73.7, 40.9)
        report = validator.validate_scene(scene, wide_aoi)
        assert report.metrics["aoi_coverage"] < 0.8
        assert report.passed is False
        assert report.errors[0].code == "LOW_AOI_COVERAGE"

    def test_configurable_for_real_multi_tile_mosaic_workflows(self):
        """The real, correct way to handle multi-tile AOIs: lower
        min_coverage_ratio explicitly, not silently change the
        default. Already fully supported -- verified here."""
        cfg = OpticalValidationConfig(min_coverage_ratio=0.05)
        validator = OpticalPreflightValidator(cfg)
        narrow_scene_bbox = (-74.2, 40.5, -73.9, 40.75)
        scene = SatelliteData.from_stac_item(
            _real_element84_item(bbox=narrow_scene_bbox), "aws_earth"
        )
        wide_aoi = box(-74.1, 40.6, -73.7, 40.9)
        report = validator.validate_scene(scene, wide_aoi)
        assert report.passed is True  # same low coverage, now within the configured threshold


class TestBug4CloudCoverHardFailureDefault:
    def test_high_cloud_cover_warns_but_keeps_scene_by_default(self, validator):
        scene = SatelliteData.from_stac_item(_real_element84_item(cloud_cover=45.0), "aws_earth")
        report = validator.validate_scene(scene, AOI_SINGLE_TILE)
        assert report.passed is True
        assert report.errors == []
        assert report.warnings[0].code == "CLOUD_COVER_EXCEEDED"
        assert report.warnings[0].severity == "WARNING"

    def test_run_preflight_keeps_high_cloud_cover_scene_in_results(self, validator):
        scene = SatelliteData.from_stac_item(_real_element84_item(cloud_cover=99.0), "aws_earth")
        safe = validator.run_preflight([scene], AOI_SINGLE_TILE)
        assert len(safe) == 1

    def test_explicit_hard_failure_opt_in_still_rejects(self):
        cfg = OpticalValidationConfig(cloud_cover_is_hard_failure=True)
        validator = OpticalPreflightValidator(cfg)
        scene = SatelliteData.from_stac_item(_real_element84_item(cloud_cover=45.0), "aws_earth")
        report = validator.validate_scene(scene, AOI_SINGLE_TILE)
        assert report.passed is False
        assert report.errors[0].code == "CLOUD_COVER_EXCEEDED"


class TestFullRealCatalogSceneEndToEnd:
    """The exact end-to-end scenario originally reported as a 100%
    rejection rate: multiple real-shaped scenes from a real provider,
    validated with default config."""

    def test_realistic_batch_from_element84_mostly_passes(self, validator):
        scenes = [
            SatelliteData.from_stac_item(_real_element84_item(scene_id=f"scene_{i}", cloud_cover=c), "aws_earth")
            for i, c in enumerate([0.0, 5.0, 12.0, 30.0, 60.0])  # varied, realistic cloud cover
        ]
        safe = validator.run_preflight(scenes, AOI_SINGLE_TILE)
        # Cloud cover alone never rejects by default (warning, not hard
        # failure) -- every one of these should pass.
        assert len(safe) == 5
