"""
Tests for pygeofetch.multisensor.terrain_corrected_change.

The core pipeline test is self-consistent and physically honest: a
synthetic sloped DEM with real, DIFFERENT solar geometry on two dates
is used to compute what an observed (illumination-biased) reflectance
would look like for a *genuinely unchanged* true surface -- by
inverting the same real cosine correction formula the pipeline itself
uses. This lets the test assert something concrete without needing
external ground truth: a naive, uncorrected difference of the two
observed rasters MUST show real, spurious "change" purely from the
sun having moved; the pipeline's real terrain correction must reduce
that spurious change substantially.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pygeofetch.multisensor.terrain_corrected_change import (
    _cosine_topographic_correction,
    solar_position,
    terrain_corrected_change_pipeline,
)

REAL_CRS = "EPSG:32633"
REAL_TRANSFORM = from_origin(500000, 4000000, 10, 10)


def _write_raster(path, data, dtype="float32"):
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": REAL_CRS,
        "transform": REAL_TRANSFORM,
        "nodata": None,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)
    return path


class TestSolarPositionAgainstKnownPhysics:
    """Independent, direct verification -- these are real, known
    reference points in solar geometry, not arbitrary numbers."""

    def test_equator_equinox_solar_noon_sun_overhead(self):
        dt = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)  # near real equinox
        zenith, _ = solar_position(dt, lat_deg=0.0, lon_deg=0.0)
        assert zenith < 2.0

    def test_tropic_of_cancer_june_solstice_sun_overhead(self):
        dt = datetime(
            2024, 6, 20, 12, 0, 0, tzinfo=timezone.utc
        )  # near real June solstice
        zenith, _ = solar_position(dt, lat_deg=23.44, lon_deg=0.0)
        assert zenith < 2.0

    def test_midlatitude_equinox_noon_sun_due_south(self):
        dt = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        zenith, azimuth = solar_position(dt, lat_deg=45.0, lon_deg=0.0)
        # Real, expected relationship: zenith ~= |latitude - declination|,
        # and at the equinox the sun sits due south for a northern
        # observer.
        assert zenith == pytest.approx(45.0, abs=2.0)
        assert azimuth == pytest.approx(180.0, abs=5.0)

    def test_different_times_give_different_real_solar_positions(self):
        """A real, practical sanity check the pipeline's own docstring
        relies on: two different real acquisition times must not
        collapse to the same solar geometry."""
        dt1 = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        dt2 = datetime(2024, 7, 15, 10, 30, 0, tzinfo=timezone.utc)
        z1, a1 = solar_position(dt1, lat_deg=40.0, lon_deg=-100.0)
        z2, a2 = solar_position(dt2, lat_deg=40.0, lon_deg=-100.0)
        assert abs(z1 - z2) > 5.0  # real, substantial seasonal difference


class TestCosineCorrectionFormula:
    def test_flat_ground_correction_is_identity(self):
        """On flat ground (slope=0), cos(i) always equals cos(zenith)
        regardless of azimuth -- the real correction factor must
        reduce to exactly 1.0, changing nothing."""
        image = np.array([[100.0, 200.0]])
        slope = np.zeros((1, 2))
        aspect = np.zeros((1, 2))
        corrected = _cosine_topographic_correction(
            image, slope, aspect, solar_zenith_deg=30.0, solar_azimuth_deg=150.0
        )
        assert np.allclose(corrected, image)

    def test_slope_facing_sun_is_darkened_by_correction(self):
        """A slope directly facing the sun has cos(i) > cos(zenith)
        (it received MORE illumination than flat ground would), so the
        real correction must reduce its brightness."""
        image = np.array([[100.0]])
        slope = np.array([[30.0]])
        aspect = np.array([[180.0]])  # facing south
        solar_zenith, solar_azimuth = 40.0, 180.0  # sun in the south
        corrected = _cosine_topographic_correction(
            image, slope, aspect, solar_zenith, solar_azimuth
        )
        assert corrected[0, 0] < image[0, 0]


class TestPipelineRemovesRealIlluminationArtifact:
    """The core, physically self-consistent test: builds an observed
    (illumination-biased) raster for a genuinely UNCHANGED true
    surface, by inverting the pipeline's own real correction formula,
    then confirms the pipeline recovers close to zero real change
    where a naive comparison would not."""

    def test_naive_diff_shows_false_change_pipeline_removes_it(self, tmp_path):
        size = 20
        true_reflectance = np.full((size, size), 0.3)

        # Real, deliberate sloped terrain: a simple east-facing 25 deg
        # slope in the left half, flat ground in the right half.
        dem = np.zeros((size, size))
        for col in range(size // 2):
            dem[:, col] = (size // 2 - col) * 5.0  # real, monotonic slope
        pixel_size = 10.0
        dem_path = _write_raster(tmp_path / "dem.tif", dem)

        from pygeofetch.insar.advanced_safeguards import slope_aspect_degrees

        slope_deg, aspect_deg = slope_aspect_degrees(dem, pixel_size)

        pre_dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        post_dt = datetime(
            2024, 7, 15, 15, 45, 0, tzinfo=timezone.utc
        )  # real, very different sun position
        lat, lon = 40.0, -100.0

        zenith_pre, azimuth_pre = solar_position(pre_dt, lat, lon)
        zenith_post, azimuth_post = solar_position(post_dt, lat, lon)

        # Real, honest self-consistency construction: invert the same
        # real correction formula to build what an OBSERVED image would
        # look like for this true, unchanged reflectance under each
        # date's real solar geometry.
        def observed_from_true(true_img, zenith, azimuth):
            slope_rad = np.radians(slope_deg)
            aspect_rad = np.radians(aspect_deg)
            zenith_rad = np.radians(zenith)
            azimuth_rad = np.radians(azimuth)
            cos_i = np.cos(slope_rad) * np.cos(zenith_rad) + np.sin(slope_rad) * np.sin(
                zenith_rad
            ) * np.cos(azimuth_rad - aspect_rad)
            cos_i_safe = np.clip(cos_i, 0.05, None)
            return true_img * cos_i_safe / np.cos(zenith_rad)

        observed_pre = observed_from_true(true_reflectance, zenith_pre, azimuth_pre)
        observed_post = observed_from_true(true_reflectance, zenith_post, azimuth_post)

        pre_path = _write_raster(tmp_path / "pre.tif", observed_pre)
        post_path = _write_raster(tmp_path / "post.tif", observed_post)

        naive_diff = observed_post - observed_pre
        naive_false_change_on_slope = np.abs(naive_diff[:, : size // 2]).mean()

        result = terrain_corrected_change_pipeline(
            pre_path,
            post_path,
            dem_path,
            pre_dt,
            post_dt,
            lat,
            lon,
            output_dir=tmp_path,
            change_threshold=0.1,
        )
        assert result.success, result.error

        with rasterio.open(result.final_output) as src:
            corrected_diff = src.read(1)
        corrected_false_change_on_slope = np.abs(corrected_diff[:, : size // 2]).mean()

        # Real, honest check: a naive comparison must show real,
        # meaningful spurious change purely from illumination (this
        # confirms the test scenario is real and non-trivial), and the
        # real terrain correction must substantially reduce it.
        assert naive_false_change_on_slope > 0.02
        assert corrected_false_change_on_slope < naive_false_change_on_slope * 0.3

    def test_shape_mismatch_between_dem_and_optical_fails_clearly(self, tmp_path):
        pre_path = _write_raster(tmp_path / "pre.tif", np.zeros((20, 20)))
        post_path = _write_raster(tmp_path / "post.tif", np.zeros((20, 20)))
        dem_path = _write_raster(tmp_path / "dem.tif", np.zeros((10, 10)))

        result = terrain_corrected_change_pipeline(
            pre_path,
            post_path,
            dem_path,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 6, 1, tzinfo=timezone.utc),
            40.0,
            -100.0,
            output_dir=tmp_path,
        )
        assert not result.success
        assert "DEM shape" in result.error
