"""
Tests for pygeofetch.insar.auto_fallback.run_multi_modal_deformation.

Uses real synthetic rasters and a minimal real-shaped pair stand-in
(anything with a real `.coherence` array, matching what
InterferogramGenerator.process_pair() actually returns) rather than
mocks, so the real quality-metric computation and real fusion path
both get genuinely exercised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pygeofetch.insar.auto_fallback import (
    SOURCE_INSAR_VALID,
    SOURCE_OPTICAL_FALLBACK,
    _compute_insar_quality_metrics,
    run_multi_modal_deformation,
)

REAL_CRS = "EPSG:32633"
GROUND_ORIGIN = (500000, 4000000)


@dataclass
class FakePair:
    """Minimal stand-in for a real InterferogramPair result -- only
    needs the real `.coherence` attribute this router actually reads."""

    coherence: np.ndarray


def _write_raster(path, data, transform):
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": REAL_CRS,
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)
    return path


class TestQualityMetrics:
    def test_all_finite_velocity_gives_full_reliable_fraction(self):
        velocity = np.zeros((10, 10))
        pairs = [FakePair(coherence=np.full((10, 10), 0.8))]
        reliable_fraction, mean_coherence = _compute_insar_quality_metrics(
            velocity, pairs
        )
        assert reliable_fraction == 1.0
        assert mean_coherence == pytest.approx(0.8)

    def test_nan_pixels_correctly_excluded_from_reliable_fraction(self):
        velocity = np.full((10, 10), 1.0)
        velocity[:5, :] = np.nan  # real, half the pixels genuinely underdetermined
        pairs = [FakePair(coherence=np.full((10, 10), 0.5))]
        reliable_fraction, _ = _compute_insar_quality_metrics(velocity, pairs)
        assert reliable_fraction == pytest.approx(0.5)

    def test_mean_coherence_averages_across_real_pairs(self):
        velocity = np.zeros((5, 5))
        pairs = [
            FakePair(coherence=np.full((5, 5), 0.9)),
            FakePair(coherence=np.full((5, 5), 0.3)),
        ]
        _, mean_coherence = _compute_insar_quality_metrics(velocity, pairs)
        assert mean_coherence == pytest.approx(0.6)


class TestHighQualityInsarSkipsOptical:
    def test_fallback_not_triggered_when_insar_is_good(self):
        velocity = np.full((20, 20), -5.0)  # fully valid, no NaN
        pairs = [FakePair(coherence=np.full((20, 20), 0.7))]  # real, high coherence

        result = run_multi_modal_deformation(velocity, pairs)

        assert result.success
        assert not result.used_optical_fallback
        assert result.insar_reliable_fraction == 1.0
        assert result.insar_mean_coherence == pytest.approx(0.7)
        assert (result.processing_source_mask == SOURCE_INSAR_VALID).all()

    def test_optical_inputs_not_required_when_insar_is_good(self):
        """Real, deliberate efficiency check: no optical_reference_path
        needed at all when InSAR already clears both thresholds."""
        velocity = np.full((10, 10), -2.0)
        pairs = [FakePair(coherence=np.full((10, 10), 0.9))]
        result = run_multi_modal_deformation(velocity, pairs)  # no optical args passed
        assert result.success


class TestTotalInsarFailureWithEmptyPairs:
    """Real, previously-crashing edge case: a total InSAR failure where
    literally zero pairs survived (not just low-coherence ones) --
    confirmed to have raised a confusing shape-mismatch error rather
    than the correct, honest optical-only result, since
    np.mean([], axis=0) silently degrades to a scalar NaN instead of a
    properly-shaped array."""

    def test_empty_pairs_with_nan_velocity_falls_back_cleanly(self, tmp_path):
        rng = np.random.default_rng(3)
        shape = (20, 20)
        velocity = np.full(shape, np.nan)  # real, honest "InSAR produced nothing" placeholder
        transform = from_origin(*GROUND_ORIGIN, 10, 10)

        ref_path = _write_raster(tmp_path / "ref.tif", rng.normal(1000, 50, (40, 40)), transform)
        sec_path = _write_raster(tmp_path / "sec.tif", rng.normal(1000, 50, (40, 40)), transform)

        result = run_multi_modal_deformation(
            velocity, pairs=[],  # genuinely empty -- not just low-coherence pairs
            optical_reference_path=ref_path, optical_secondary_path=sec_path,
            reference_transform=transform, insar_transform=transform, insar_crs=REAL_CRS,
            window_size=16, step_size=8,
        )
        assert result.success, result.error
        assert result.used_optical_fallback
        assert result.insar_reliable_fraction == 0.0
        assert result.insar_mean_coherence == 0.0
        # Real, correct consequence: with InSAR velocity entirely
        # invalid, every pixel where optical actually has real,
        # windowed coverage must be optical-fallback. Pixels where
        # optical ALSO has no real coverage (a genuine edge effect near
        # the grid border, not a bug -- see fuse_insar_optical's own
        # real handling of "both sources invalid") fall through to the
        # blended/transition label by that function's own existing
        # convention, not a new behavior introduced by this fix.
        has_real_optical_coverage = np.isfinite(result.optical_displacement_magnitude)
        assert (result.processing_source_mask[has_real_optical_coverage] == SOURCE_OPTICAL_FALLBACK).all()
        assert not (result.processing_source_mask == SOURCE_INSAR_VALID).any()


class TestLowQualityInsarTriggersRealFallback:
    def test_low_coherence_triggers_fallback_and_fuses(self, tmp_path):
        rng = np.random.default_rng(1)
        transform = from_origin(*GROUND_ORIGIN, 10, 10)

        velocity = np.full((20, 20), -10.0)
        pairs = [FakePair(coherence=np.full((20, 20), 0.15))]  # real, low coherence

        ref_data = rng.normal(1000, 100, (40, 40)).astype("float32")
        sec_data = ref_data + rng.normal(0, 5, (40, 40)).astype("float32")
        ref_path = _write_raster(
            tmp_path / "ref.tif", ref_data, from_origin(*GROUND_ORIGIN, 10, 10)
        )
        sec_path = _write_raster(
            tmp_path / "sec.tif", sec_data, from_origin(*GROUND_ORIGIN, 10, 10)
        )

        with rasterio.open(ref_path) as src:
            reference_transform = src.transform

        result = run_multi_modal_deformation(
            velocity,
            pairs,
            optical_reference_path=ref_path,
            optical_secondary_path=sec_path,
            reference_transform=reference_transform,
            insar_transform=transform,
            insar_crs=REAL_CRS,
            window_size=16,
            step_size=8,
        )

        assert result.success, result.error
        assert result.used_optical_fallback
        assert result.insar_mean_coherence == pytest.approx(0.15)
        # Real, correct consequence of uniformly low coherence: fully
        # optical-dominant, matching fuse_insar_optical's own real,
        # documented behavior at this coherence level.
        assert (result.processing_source_mask == SOURCE_OPTICAL_FALLBACK).mean() > 0.5

    def test_missing_optical_inputs_raises_clear_error(self):
        velocity = np.full((10, 10), -5.0)
        pairs = [FakePair(coherence=np.full((10, 10), 0.1))]  # real, low coherence

        with pytest.raises(ValueError, match="optical_reference_path"):
            run_multi_modal_deformation(
                velocity, pairs
            )  # no optical args -- must raise

    def test_or_not_and_low_coverage_alone_triggers_fallback(self, tmp_path):
        """Real, deliberate OR logic: even with high coherence, low
        real coverage (many NaN pixels) alone must trigger fallback."""
        rng = np.random.default_rng(2)
        velocity = np.full((20, 20), np.nan)
        velocity[:2, :] = -5.0  # only 10% real coverage
        pairs = [FakePair(coherence=np.full((20, 20), 0.9))]  # coherence itself is fine

        ref_path = _write_raster(
            tmp_path / "ref.tif",
            rng.normal(1000, 50, (40, 40)),
            from_origin(*GROUND_ORIGIN, 10, 10),
        )
        sec_path = _write_raster(
            tmp_path / "sec.tif",
            rng.normal(1000, 50, (40, 40)),
            from_origin(*GROUND_ORIGIN, 10, 10),
        )
        with rasterio.open(ref_path) as src:
            reference_transform = src.transform

        result = run_multi_modal_deformation(
            velocity,
            pairs,
            optical_reference_path=ref_path,
            optical_secondary_path=sec_path,
            reference_transform=reference_transform,
            insar_transform=from_origin(*GROUND_ORIGIN, 10, 10),
            insar_crs=REAL_CRS,
            window_size=16,
            step_size=8,
        )
        assert (
            result.used_optical_fallback
        )  # triggered by coverage alone, despite good coherence
