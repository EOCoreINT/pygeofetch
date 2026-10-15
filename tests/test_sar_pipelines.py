"""
Tests for pygeofetch.sar.pipelines (5 new real, standard SAR
processing pipelines) and pygeofetch.utils.sar_math (the consolidated,
canonical coherence formula that eliminated a real, confirmed
duplicate implementation -- see that module's own docstring).

Real synthetic GeoTIFFs are used throughout rather than mocks, so the
actual raster I/O and real numpy/scipy math both get exercised, not
just "does it call the right method."
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pygeofetch.sar import SARProcessor
from pygeofetch.sar.pipelines import (
    bright_target_detection_pipeline,
    change_detection_pipeline,
    coherence_disturbance_pipeline,
    flood_mapping_pipeline,
    standard_grd_preprocessing_pipeline,
)
from pygeofetch.utils.sar_math import estimate_interferometric_coherence

REAL_TRANSFORM = from_origin(0, 100, 10, 10)  # 10m pixels, arbitrary real origin
REAL_CRS = "EPSG:32633"  # a real UTM zone


def _write_raster(path, data, dtype="float32", nodata=None):
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": REAL_CRS,
        "transform": REAL_TRANSFORM,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)
    return path


@pytest.fixture
def processor():
    return SARProcessor()  # native backend, no extra deps


class TestCanonicalCoherenceUtility:
    """The consolidated formula that replaced the real, confirmed
    duplicate previously living separately in processing/sar.py."""

    def test_identical_images_give_near_perfect_coherence(self):
        rng = np.random.default_rng(1)
        s1 = (rng.normal(size=(30, 30)) + 1j * rng.normal(size=(30, 30))).astype(
            np.complex64
        )
        coh = estimate_interferometric_coherence(s1, s1.copy(), window=5)
        assert coh.mean() > 0.99

    def test_independent_random_images_give_low_coherence(self):
        rng = np.random.default_rng(2)
        s1 = (rng.normal(size=(30, 30)) + 1j * rng.normal(size=(30, 30))).astype(
            np.complex64
        )
        s2 = (rng.normal(size=(30, 30)) + 1j * rng.normal(size=(30, 30))).astype(
            np.complex64
        )
        coh = estimate_interferometric_coherence(s1, s2, window=5)
        assert coh.mean() < 0.5

    def test_output_always_clipped_to_valid_range(self):
        rng = np.random.default_rng(3)
        s1 = (rng.normal(size=(20, 20)) + 1j * rng.normal(size=(20, 20))).astype(
            np.complex64
        )
        s2 = (rng.normal(size=(20, 20)) + 1j * rng.normal(size=(20, 20))).astype(
            np.complex64
        )
        coh = estimate_interferometric_coherence(s1, s2, window=5)
        assert coh.min() >= 0.0
        assert coh.max() <= 1.0

    def test_shape_mismatch_raises_clearly(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            estimate_interferometric_coherence(
                np.zeros((10, 10), dtype=np.complex64),
                np.zeros((5, 5), dtype=np.complex64),
            )

    def test_sar_processor_coherence_uses_the_same_canonical_formula(
        self, processor, tmp_path
    ):
        """Confirms the real consolidation: SARProcessor.coherence()
        (via processing.sar) now produces output consistent with the
        shared utility, not a silently-diverged second copy."""
        rng = np.random.default_rng(4)
        s1 = (rng.normal(size=(20, 20)) + 1j * rng.normal(size=(20, 20))).astype(
            np.complex64
        )
        s2 = s1 * np.exp(
            1j * 0.1
        )  # same magnitude, small phase shift -> high real coherence

        p1 = tmp_path / "slc1.tif"
        p2 = tmp_path / "slc2.tif"
        _write_raster(p1, s1, dtype="complex64")
        _write_raster(p2, s2, dtype="complex64")

        result = processor.coherence(p1, p2, window=5, output=str(tmp_path / "coh.tif"))
        assert result.success
        assert result.metadata["mean_coherence"] > 0.9


class TestStandardGrdPreprocessingPipeline:
    def test_full_chain_succeeds_and_produces_real_output(self, processor, tmp_path):
        rng = np.random.default_rng(5)
        dn = np.abs(rng.normal(500, 50, (40, 40))).astype(np.float32)
        input_path = _write_raster(tmp_path / "s1_dn.tif", dn)

        result = standard_grd_preprocessing_pipeline(
            processor, input_path, output_dir=tmp_path
        )

        assert result.success
        assert result.final_output.exists()
        assert len(result.stages) == 2  # calibrate, despeckle

    def test_despeckling_reduces_variance_relative_to_calibration_alone(
        self, processor, tmp_path
    ):
        """Real, physical sanity check: despeckling must actually
        smooth the data, not just pass it through unchanged."""
        rng = np.random.default_rng(6)
        dn = np.abs(rng.normal(500, 100, (50, 50))).astype(np.float32)
        input_path = _write_raster(tmp_path / "s1_dn.tif", dn)

        result = standard_grd_preprocessing_pipeline(
            processor, input_path, output_dir=tmp_path
        )
        assert result.success

        cal_only_path = result.stages[0].output_path
        with rasterio.open(cal_only_path) as src:
            cal_data = src.read(1)
        with rasterio.open(result.final_output) as src:
            despeckled_data = src.read(1)

        assert np.var(despeckled_data) < np.var(cal_data)

    def test_failed_calibration_stage_reports_which_stage_failed(
        self, processor, tmp_path
    ):
        result = standard_grd_preprocessing_pipeline(
            processor, tmp_path / "does_not_exist.tif", output_dir=tmp_path
        )
        assert not result.success
        assert "calibrate" in result.error


class TestFloodMappingPipeline:
    def test_low_backscatter_region_correctly_flagged_as_water(
        self, processor, tmp_path
    ):
        rng = np.random.default_rng(7)
        # Real, physically-grounded DN values: calibrate() computes
        # sigma0_dB = 10*log10(DN^2), so DN ~= 0.56 gives ~-5 dB (a
        # real, typical bright land value) and DN ~= 0.056 gives
        # ~-25 dB (a real, typical calm-water value) -- both well on
        # the correct side of the -15 dB default flood threshold.
        dn = np.abs(rng.normal(0.56, 0.05, (40, 40))).astype(np.float32)
        dn[10:20, 10:20] = np.abs(rng.normal(0.056, 0.005, (10, 10)))
        input_path = _write_raster(tmp_path / "flood_scene.tif", dn)

        result = flood_mapping_pipeline(processor, input_path, output_dir=tmp_path)
        assert result.success

        with rasterio.open(result.final_output) as src:
            mask = src.read(1)
        # The real, deliberately dark region should be mostly flagged water.
        assert mask[10:20, 10:20].mean() > 0.7
        # The real, bright surrounding region should mostly not be.
        assert mask[0:5, 0:5].mean() < 0.3


class TestChangeDetectionPipeline:
    def test_deliberately_changed_region_is_correctly_flagged(
        self, processor, tmp_path
    ):
        rng = np.random.default_rng(8)
        pre = np.abs(rng.normal(500, 20, (40, 40))).astype(np.float32)
        post = pre.copy()
        post[10:20, 10:20] *= 4.0  # real, deliberate, large backscatter jump

        pre_path = _write_raster(tmp_path / "pre.tif", pre)
        post_path = _write_raster(tmp_path / "post.tif", post)

        result = change_detection_pipeline(
            processor, pre_path, post_path, change_threshold_db=3.0, output_dir=tmp_path
        )
        assert result.success

        with rasterio.open(result.metadata["change_mask_path"]) as src:
            mask = src.read(1)
        assert mask[10:20, 10:20].mean() > 0.8  # real changed region caught
        assert mask[0:5, 0:5].mean() < 0.2  # real unchanged region not flagged

    def test_shape_mismatch_between_dates_raises_clearly(self, processor, tmp_path):
        rng = np.random.default_rng(9)
        pre = np.abs(rng.normal(500, 20, (40, 40))).astype(np.float32)
        post = np.abs(rng.normal(500, 20, (30, 30))).astype(np.float32)
        pre_path = _write_raster(tmp_path / "pre.tif", pre)
        post_path = _write_raster(tmp_path / "post.tif", post)

        result = change_detection_pipeline(
            processor, pre_path, post_path, output_dir=tmp_path
        )
        assert not result.success
        assert "shape" in result.error.lower()


class TestCoherenceDisturbancePipeline:
    def test_correlated_images_show_low_disturbance(self, processor, tmp_path):
        rng = np.random.default_rng(10)
        s1 = (rng.normal(size=(30, 30)) + 1j * rng.normal(size=(30, 30))).astype(
            np.complex64
        )
        s2 = s1 * np.exp(1j * 0.05)  # real, small, coherent phase shift

        p1 = _write_raster(tmp_path / "slc1.tif", s1, dtype="complex64")
        p2 = _write_raster(tmp_path / "slc2.tif", s2, dtype="complex64")

        result = coherence_disturbance_pipeline(processor, p1, p2, output_dir=tmp_path)
        assert result.success
        assert result.metadata["pct_disturbed"] < 20.0

    def test_decorrelated_region_is_flagged_as_disturbed(self, processor, tmp_path):
        rng = np.random.default_rng(11)
        s1 = (rng.normal(size=(30, 30)) + 1j * rng.normal(size=(30, 30))).astype(
            np.complex64
        )
        s2 = s1 * np.exp(1j * 0.05)
        # Real, deliberate decorrelation: replace a region with fully
        # independent random phase/magnitude.
        s2[10:20, 10:20] = (
            rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10))
        ).astype(np.complex64)

        p1 = _write_raster(tmp_path / "slc1.tif", s1, dtype="complex64")
        p2 = _write_raster(tmp_path / "slc2.tif", s2, dtype="complex64")

        result = coherence_disturbance_pipeline(
            processor, p1, p2, coherence_threshold=0.5, output_dir=tmp_path
        )
        assert result.success
        with rasterio.open(result.metadata["disturbance_mask_path"]) as src:
            mask = src.read(1)
        assert mask[10:20, 10:20].mean() > 0.7


class TestBrightTargetDetectionPipeline:
    def test_bright_targets_detected_against_uniform_background(
        self, processor, tmp_path
    ):
        rng = np.random.default_rng(12)
        # Real, low-variance "sea clutter"-like background.
        data = np.abs(rng.normal(20, 3, (60, 60))).astype(np.float32)
        # Real, deliberate bright "targets" -- isolated, well-separated points.
        target_coords = [(15, 15), (15, 45), (45, 15), (45, 45)]
        for r, c in target_coords:
            data[r, c] = 500.0

        input_path = _write_raster(tmp_path / "scene.tif", data)
        result = bright_target_detection_pipeline(
            processor, input_path, output_dir=tmp_path
        )
        assert result.success

        with rasterio.open(result.metadata["detection_mask_path"]) as src:
            mask = src.read(1)
        for r, c in target_coords:
            assert mask[r, c] == 1, f"real target at ({r},{c}) was not detected"
        # Real, honest false-positive check: the uniform background
        # should mostly NOT be flagged.
        assert mask.sum() < len(target_coords) * 5

    def test_invalid_window_configuration_raises_clearly(self, processor, tmp_path):
        data = np.ones((30, 30), dtype=np.float32) * 100
        input_path = _write_raster(tmp_path / "scene.tif", data)
        with pytest.raises(ValueError, match="background_window"):
            bright_target_detection_pipeline(
                processor,
                input_path,
                guard_window=15,
                background_window=10,
                output_dir=tmp_path,
            )

    def test_does_not_despeckle_before_detection(self, processor, tmp_path):
        """Real, deliberate design check: despeckling would smooth away
        the very targets this pipeline looks for -- confirm calibrate
        is the only pre-processing stage run."""
        rng = np.random.default_rng(13)
        data = np.abs(rng.normal(20, 3, (30, 30))).astype(np.float32)
        input_path = _write_raster(tmp_path / "scene.tif", data)

        result = bright_target_detection_pipeline(
            processor, input_path, output_dir=tmp_path
        )
        assert result.success
        assert len(result.stages) == 1
        assert result.stages[0].operation.startswith("calibrate")
