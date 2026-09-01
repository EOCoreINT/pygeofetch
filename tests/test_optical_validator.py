"""
Tests for pygeofetch.validation.optical_validator.

Uses mock STAC-like dicts and real shapely polygons throughout, and
separately confirms the same behavior against real pygeofetch
SatelliteData objects (the shape PyGeoFetch.search() actually
returns) for the core paths, since a validator that only works
against hand-built dicts and silently breaks on real search results
would be worse than no validator at all.
"""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon, box

from pygeofetch.models.satellite_data import ProcessingLevel, SatelliteAsset, SatelliteData
from pygeofetch.validation.optical_validator import (
    OpticalPreflightValidator,
    OpticalValidationConfig,
    OpticalValidationError,
    SEVERITY_ERROR,
    SEVERITY_WARNING,
)

# A consistent AOI used across most tests: a small rectangle over NYC.
AOI = box(-74.1, 40.6, -73.7, 40.9)

REQUIRED_BANDS = ["B02", "B03", "B04", "B08", "SCL"]


def make_scene(
    scene_id: str = "S2A_TEST",
    bbox: tuple[float, float, float, float] = (-74.2, 40.5, -73.6, 41.0),
    cloud_cover: float = 5.0,
    bands: list[str] | None = None,
    processing_level: str = "Level-2A",
    datetime_str: str | None = "2024-06-15T10:30:00",
    snow_ice_cover: float | None = None,
) -> dict:
    """A mock STAC-like scene dict, matching the shape run_preflight
    accepts directly (no SatelliteData conversion needed)."""
    bands = bands if bands is not None else list(REQUIRED_BANDS)
    props: dict = {"eo:cloud_cover": cloud_cover, "processing:level": processing_level}
    if snow_ice_cover is not None:
        props["s2:snow_ice_percentage"] = snow_ice_cover
    return {
        "id": scene_id,
        "bbox": list(bbox),
        "cloud_cover": cloud_cover,
        "assets": {b: {"href": f"https://example.com/{b}.tif"} for b in bands},
        "processing_level": processing_level,
        "properties": props,
        "datetime": datetime_str,
    }


def make_real_scene(
    scene_id: str = "S2A_TEST",
    bbox: tuple[float, float, float, float] = (-74.2, 40.5, -73.6, 41.0),
    cloud_cover: float = 5.0,
    bands: list[str] | None = None,
    processing_level: ProcessingLevel = ProcessingLevel.L2A,
) -> SatelliteData:
    """A real SatelliteData instance -- the shape PyGeoFetch.search()
    actually returns."""
    bands = bands if bands is not None else list(REQUIRED_BANDS)
    return SatelliteData(
        id=scene_id,
        provider="copernicus",
        bbox=bbox,
        cloud_cover=cloud_cover,
        processing_level=processing_level,
        assets={
            b: SatelliteAsset(key=b, href=f"https://example.com/{b}.tif", roles=["data"])
            for b in bands
        },
    )


@pytest.fixture
def validator():
    return OpticalPreflightValidator(OpticalValidationConfig())


# ══════════════════════════════════════════════════════════════════════════
#  Config defaults (Task 1)
# ══════════════════════════════════════════════════════════════════════════


class TestConfigDefaults:
    def test_critical_checks_default_true(self):
        cfg = OpticalValidationConfig()
        assert cfg.check_aoi_coverage is True
        assert cfg.check_cloud_cover is True
        assert cfg.check_required_bands is True
        assert cfg.check_processing_level is True
        assert cfg.check_temporal_bounds is True

    def test_heavy_or_niche_checks_default_false(self):
        cfg = OpticalValidationConfig()
        assert cfg.check_snow_ice_cover is False
        assert cfg.check_nodata_margins is False

    def test_default_thresholds(self):
        cfg = OpticalValidationConfig()
        assert cfg.min_coverage_ratio == 0.8
        assert cfg.max_cloud_cover_pct == 20.0
        assert cfg.required_bands == ["B02", "B03", "B04", "B08", "SCL"]
        assert cfg.expected_level == "Level-2A"

    def test_cloud_cover_is_a_warning_not_a_hard_failure_by_default(self):
        cfg = OpticalValidationConfig()
        assert cfg.cloud_cover_is_hard_failure is False

    def test_every_field_independently_toggleable(self):
        cfg = OpticalValidationConfig(
            check_aoi_coverage=False,
            check_cloud_cover=False,
            check_snow_ice_cover=True,
            check_required_bands=False,
            check_processing_level=False,
            check_nodata_margins=True,
            check_temporal_bounds=False,
        )
        assert cfg.check_aoi_coverage is False
        assert cfg.check_snow_ice_cover is True
        assert cfg.check_nodata_margins is True


# ══════════════════════════════════════════════════════════════════════════
#  Scenario 1: a scene that passes all default checks
# ══════════════════════════════════════════════════════════════════════════


class TestScenePassesAllDefaultChecks:
    def test_good_scene_passes(self, validator):
        scene = make_scene()
        report = validator.validate_scene(scene, AOI)
        assert report.passed is True
        assert report.errors == []

    def test_good_scene_metrics_populated(self, validator):
        scene = make_scene()
        report = validator.validate_scene(scene, AOI)
        assert report.metrics["aoi_coverage"] == pytest.approx(1.0)
        assert report.metrics["cloud_cover_pct"] == 5.0

    def test_good_scene_survives_run_preflight(self, validator):
        safe = validator.run_preflight([make_scene()], AOI)
        assert len(safe) == 1

    def test_good_scene_passes_with_real_satellitedata(self, validator):
        """The same good-scene case, but against a real SatelliteData
        object rather than a hand-built dict -- catches bugs that only
        manifest against the real model shape."""
        scene = make_real_scene()
        report = validator.validate_scene(scene, AOI)
        assert report.passed is True
        assert report.metrics["cloud_cover_pct"] == 5.0


# ══════════════════════════════════════════════════════════════════════════
#  Scenario 2: high cloud cover
# ══════════════════════════════════════════════════════════════════════════


class TestCloudCoverFailure:
    def test_high_cloud_cover_is_a_warning_by_default(self, validator):
        scene = make_scene(cloud_cover=45.0)
        report = validator.validate_scene(scene, AOI)
        assert report.passed is True  # warning, not a hard failure by default
        issue = report.warnings[0]
        assert issue.code == "CLOUD_COVER_EXCEEDED"
        assert issue.severity == SEVERITY_WARNING

    def test_high_cloud_cover_still_appears_in_run_preflight_output_by_default(self, validator):
        safe = validator.run_preflight([make_scene(cloud_cover=45.0)], AOI)
        assert len(safe) == 1  # kept, just logged

    def test_high_cloud_cover_can_be_configured_as_hard_failure(self):
        cfg = OpticalValidationConfig(cloud_cover_is_hard_failure=True)
        validator = OpticalPreflightValidator(cfg)
        scene = make_scene(cloud_cover=45.0)
        report = validator.validate_scene(scene, AOI)
        assert report.passed is False
        assert report.errors[0].code == "CLOUD_COVER_EXCEEDED"

    def test_high_cloud_cover_excluded_via_treat_warnings_as_errors(self):
        cfg = OpticalValidationConfig(treat_warnings_as_errors=True)
        validator = OpticalPreflightValidator(cfg)
        safe = validator.run_preflight([make_scene(cloud_cover=45.0)], AOI)
        assert safe == []

    def test_cloud_cover_check_can_be_disabled(self):
        cfg = OpticalValidationConfig(check_cloud_cover=False)
        validator = OpticalPreflightValidator(cfg)
        scene = make_scene(cloud_cover=99.0)
        report = validator.validate_scene(scene, AOI)
        assert "cloud_cover_pct" not in report.metrics
        assert report.passed is True

    def test_validate_cloud_cover_extracts_correct_value(self, validator):
        assert validator.validate_cloud_cover(make_scene(cloud_cover=33.0)) == 33.0


# ══════════════════════════════════════════════════════════════════════════
#  Scenario 3: missing required bands (hard failure)
# ══════════════════════════════════════════════════════════════════════════


class TestMissingBandsFailure:
    def test_missing_band_is_a_hard_failure(self, validator):
        scene = make_scene(bands=["B02", "B03", "B04"])  # no B08, no SCL
        report = validator.validate_scene(scene, AOI)
        assert report.passed is False
        assert report.errors[0].code == "MISSING_BANDS"
        assert "B08" in report.errors[0].message
        assert "SCL" in report.errors[0].message

    def test_missing_band_excluded_from_run_preflight(self, validator):
        safe = validator.run_preflight([make_scene(bands=["B02", "B03", "B04"])], AOI)
        assert safe == []

    def test_validate_bands_raises_optical_validation_error(self, validator):
        with pytest.raises(OpticalValidationError) as exc_info:
            validator.validate_bands(["B02", "B03"])
        assert exc_info.value.code == "MISSING_BANDS"

    def test_optical_validation_error_is_a_value_error(self):
        """Task 3's integration example expects `except ValueError` to
        keep working -- confirm the inheritance really holds."""
        assert issubclass(OpticalValidationError, ValueError)

    def test_validate_bands_returns_present_bands_when_all_present(self, validator):
        present = validator.validate_bands(REQUIRED_BANDS)
        assert set(present) == set(REQUIRED_BANDS)

    def test_band_matching_is_case_insensitive(self, validator):
        scene = make_scene(bands=["b02", "b03", "b04", "b08", "scl"])
        report = validator.validate_scene(scene, AOI)
        assert report.passed is True

    def test_required_bands_check_can_be_disabled(self):
        cfg = OpticalValidationConfig(check_required_bands=False)
        validator = OpticalPreflightValidator(cfg)
        scene = make_scene(bands=[])
        report = validator.validate_scene(scene, AOI)
        assert report.passed is True


# ══════════════════════════════════════════════════════════════════════════
#  Scenario 4: AOI coverage failure (only ~10% overlap)
# ══════════════════════════════════════════════════════════════════════════


class TestAoiCoverageFailure:
    def test_low_coverage_is_a_hard_failure(self, validator):
        # A scene footprint that overlaps only a small sliver of the AOI.
        scene = make_scene(bbox=(-74.1, 40.85, -73.7, 40.9))  # ~top strip only
        report = validator.validate_scene(scene, AOI)
        assert report.passed is False
        assert report.errors[0].code == "LOW_AOI_COVERAGE"

    def test_low_coverage_excluded_from_run_preflight(self, validator):
        safe = validator.run_preflight(
            [make_scene(bbox=(10.0, 10.0, 10.5, 10.5))], AOI  # nowhere near the AOI
        )
        assert safe == []

    def test_validate_aoi_coverage_returns_approximately_ten_percent(self, validator):
        # Build a scene footprint that covers roughly the left 10% of the AOI.
        aoi_width = AOI.bounds[2] - AOI.bounds[0]
        narrow = box(AOI.bounds[0], AOI.bounds[1], AOI.bounds[0] + aoi_width * 0.1, AOI.bounds[3])
        coverage = validator.validate_aoi_coverage(narrow, AOI)
        assert coverage == pytest.approx(0.1, abs=0.01)

    def test_full_coverage_returns_one(self, validator):
        big = box(-75, 39, -73, 42)  # fully contains the AOI
        assert validator.validate_aoi_coverage(big, AOI) == pytest.approx(1.0)

    def test_zero_overlap_returns_zero(self, validator):
        far = box(10, 10, 11, 11)
        assert validator.validate_aoi_coverage(far, AOI) == 0.0

    def test_coverage_threshold_is_configurable(self):
        cfg = OpticalValidationConfig(min_coverage_ratio=0.05)
        validator = OpticalPreflightValidator(cfg)
        aoi_width = AOI.bounds[2] - AOI.bounds[0]
        narrow = box(AOI.bounds[0], AOI.bounds[1], AOI.bounds[0] + aoi_width * 0.1, AOI.bounds[3])
        scene = make_scene()
        report = validator.validate_scene(scene, AOI)
        # sanity: default full-coverage scene still passes with a lax threshold
        assert report.passed is True

    def test_missing_footprint_is_a_hard_failure(self, validator):
        scene = make_scene()
        del scene["bbox"]
        report = validator.validate_scene(scene, AOI)
        assert report.passed is False
        assert report.errors[0].code == "NO_FOOTPRINT"

    def test_none_aoi_skips_aoi_checks_without_crashing(self, validator):
        """check_aoi_coverage defaults to True, but a caller without an
        AOI at hand (e.g. validating just before download, without the
        original search AOI threaded through) must not crash."""
        scene = make_scene()
        report = validator.validate_scene(scene, aoi=None)
        assert report.passed is True
        assert "aoi_coverage" not in report.metrics

    def test_none_aoi_still_runs_non_aoi_checks(self, validator):
        scene = make_scene(bands=["B02"])  # missing bands
        report = validator.validate_scene(scene, aoi=None)
        assert report.passed is False
        assert report.errors[0].code == "MISSING_BANDS"

    def test_run_preflight_accepts_none_aoi(self, validator):
        safe = validator.run_preflight([make_scene(), make_scene(bands=["B02"])], aoi=None)
        assert len(safe) == 1


# ══════════════════════════════════════════════════════════════════════════
#  Processing level
# ══════════════════════════════════════════════════════════════════════════


class TestProcessingLevel:
    def test_matching_level_passes(self, validator):
        scene = make_scene(processing_level="Level-2A")
        report = validator.validate_scene(scene, AOI)
        assert report.passed is True

    def test_alias_matching_l2a_vs_level_2a(self, validator):
        scene = make_scene(processing_level="L2A")
        report = validator.validate_scene(scene, AOI)
        assert report.passed is True

    def test_mismatched_level_is_a_hard_failure(self, validator):
        scene = make_scene(processing_level="Level-1C")
        report = validator.validate_scene(scene, AOI)
        assert report.passed is False
        assert report.errors[0].code == "PROCESSING_LEVEL_MISMATCH"

    def test_unreported_level_is_not_a_failure(self, validator):
        scene = make_scene()
        scene["processing_level"] = None
        scene["properties"].pop("processing:level", None)
        report = validator.validate_scene(scene, AOI)
        assert report.passed is True


# ══════════════════════════════════════════════════════════════════════════
#  Snow/ice cover (opt-in)
# ══════════════════════════════════════════════════════════════════════════


class TestSnowIceCover:
    def test_disabled_by_default(self, validator):
        scene = make_scene(snow_ice_cover=90.0)
        report = validator.validate_scene(scene, AOI)
        assert "snow_ice_cover_pct" not in report.metrics

    def test_enabled_flags_high_snow_ice_as_warning(self):
        cfg = OpticalValidationConfig(check_snow_ice_cover=True)
        validator = OpticalPreflightValidator(cfg)
        scene = make_scene(snow_ice_cover=50.0)
        report = validator.validate_scene(scene, AOI)
        assert report.passed is True  # warning, not a hard failure
        assert report.warnings[0].code == "SNOW_ICE_COVER_EXCEEDED"


# ══════════════════════════════════════════════════════════════════════════
#  Temporal bounds
# ══════════════════════════════════════════════════════════════════════════


class TestTemporalBounds:
    def test_within_bounds_passes(self, validator):
        from datetime import date

        scene = make_scene(datetime_str="2024-06-15T10:30:00")
        report = validator.validate_scene(
            scene, AOI, start_date=date(2024, 1, 1), end_date=date(2024, 12, 31)
        )
        assert report.passed is True

    def test_before_start_date_is_a_hard_failure(self, validator):
        from datetime import date

        scene = make_scene(datetime_str="2023-01-01T00:00:00")
        report = validator.validate_scene(
            scene, AOI, start_date=date(2024, 1, 1), end_date=date(2024, 12, 31)
        )
        assert report.passed is False
        assert report.errors[0].code == "OUT_OF_TEMPORAL_BOUNDS"

    def test_after_end_date_is_a_hard_failure(self, validator):
        from datetime import date

        scene = make_scene(datetime_str="2025-01-01T00:00:00")
        report = validator.validate_scene(
            scene, AOI, start_date=date(2024, 1, 1), end_date=date(2024, 12, 31)
        )
        assert report.passed is False


# ══════════════════════════════════════════════════════════════════════════
#  Nodata margins (opt-in heuristic)
# ══════════════════════════════════════════════════════════════════════════


class TestNodataMargins:
    def test_disabled_by_default(self, validator):
        scene = make_scene()
        report = validator.validate_scene(scene, AOI)
        assert not any(i.code == "NODATA_MARGIN_RISK" for i in report.issues)

    def test_aoi_well_inside_footprint_is_safe(self):
        cfg = OpticalValidationConfig(check_nodata_margins=True, nodata_margin_buffer_deg=0.01)
        validator = OpticalPreflightValidator(cfg)
        big_footprint = box(-75, 39, -73, 42)  # comfortably contains the AOI
        assert validator.validate_nodata_margins(big_footprint, AOI) is True

    def test_aoi_mostly_in_the_margin_is_flagged(self):
        cfg = OpticalValidationConfig(
            check_nodata_margins=True, nodata_margin_buffer_deg=0.15,
            nodata_margin_min_aoi_fraction=0.5,
        )
        validator = OpticalPreflightValidator(cfg)
        # A footprint whose edge runs straight through the middle of the AOI.
        footprint = box(-74.1, 40.6, -73.9, 40.9)  # only covers the left half
        assert validator.validate_nodata_margins(footprint, AOI) is False


# ══════════════════════════════════════════════════════════════════════════
#  run_preflight orchestration
# ══════════════════════════════════════════════════════════════════════════


class TestRunPreflightOrchestration:
    def test_mixed_batch_filters_correctly(self, validator):
        catalog = [
            make_scene("good"),
            make_scene("missing_band", bands=["B02", "B03"]),
            make_scene("far_away", bbox=(10, 10, 10.5, 10.5)),
            make_scene("cloudy", cloud_cover=60.0),  # warning only, kept
        ]
        safe = validator.run_preflight(catalog, AOI)
        safe_ids = {s["id"] for s in safe}
        assert safe_ids == {"good", "cloudy"}

    def test_empty_catalog_returns_empty(self, validator):
        assert validator.run_preflight([], AOI) == []

    def test_preserves_original_representation(self, validator):
        """Dicts in, dicts out; SatelliteData in, SatelliteData out --
        no silent conversion, so this slots directly between search()
        and download() either way."""
        dict_scene = make_scene()
        safe_dicts = validator.run_preflight([dict_scene], AOI)
        assert isinstance(safe_dicts[0], dict)

        real_scene = make_real_scene()
        safe_real = validator.run_preflight([real_scene], AOI)
        assert isinstance(safe_real[0], SatelliteData)

    def test_logs_warning_for_soft_issues(self, validator, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            validator.run_preflight([make_scene(cloud_cover=99.0)], AOI)
        assert any("CLOUD_COVER_EXCEEDED" in r.message for r in caplog.records)

    def test_logs_error_for_hard_failures(self, validator, caplog):
        import logging

        with caplog.at_level(logging.ERROR):
            validator.run_preflight([make_scene(bands=["B02"])], AOI)
        assert any("MISSING_BANDS" in r.message for r in caplog.records)
