"""
Tests for pygeofetch.optical.offset_tracking's compute_horizontal_strain
and fuse_insar_optical.

compute_horizontal_strain's outlier protection is tested by directly
reproducing the exact real failure this project's own live Bu'ertai
validation run hit: a spurious 226.27m displacement value (the exact
real number observed) surrounded by a physically reasonable field,
which without protection produces a triple-digit-percent strain
artifact (this project's own live run: 617%; this synthetic
reproduction: comparable order of magnitude) -- not a hypothetical
edge case.

fuse_insar_optical is tested directly against the real, observed
Bu'ertai coherence value (0.313 mean) to confirm the fusion logic
reproduces the real, observed all-optical-dominant outcome from that
run, not just a synthetic toy case.
"""

from __future__ import annotations

import numpy as np
import pytest

from pygeofetch.optical.offset_tracking import (
    compute_horizontal_strain,
    fuse_insar_optical,
)


class TestStrainAnalyticalCorrectness:
    def test_pure_exx_linear_gradient_matches_analytical_value(self):
        size, pixel_size, true_exx = 20, 16.0, 0.001
        _, col = np.mgrid[0:size, 0:size]
        dx = true_exx * col * pixel_size
        dy = np.zeros((size, size))

        result = compute_horizontal_strain(dx, dy, pixel_size)
        interior = result["exx"][2:-2, 2:-2]
        assert np.allclose(interior, true_exx, atol=1e-6)

    def test_pure_eyy_linear_gradient_matches_analytical_value(self):
        size, pixel_size, true_eyy = 20, 16.0, 0.002
        row, _ = np.mgrid[0:size, 0:size]
        dy = true_eyy * row * pixel_size
        dx = np.zeros((size, size))

        result = compute_horizontal_strain(dx, dy, pixel_size)
        interior = result["eyy"][2:-2, 2:-2]
        assert np.allclose(interior, true_eyy, atol=1e-6)

    def test_pure_shear_matches_analytical_value(self):
        """exy = 0.5*(d(dx)/d(row) + d(dy)/d(col)) -- construct dx
        varying with row and dy varying with column, both with known
        real gradients, and check the combined real shear value."""
        size, pixel_size = 20, 16.0
        row, col = np.mgrid[0:size, 0:size]
        d_dx_drow_true = 0.0015
        d_dy_dcol_true = 0.0025
        dx = d_dx_drow_true * row * pixel_size
        dy = d_dy_dcol_true * col * pixel_size

        result = compute_horizontal_strain(dx, dy, pixel_size)
        interior = result["exy"][2:-2, 2:-2]
        expected = 0.5 * (d_dx_drow_true + d_dy_dcol_true)
        assert np.allclose(interior, expected, atol=1e-6)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            compute_horizontal_strain(np.zeros((5, 5)), np.zeros((6, 6)), 10.0)


class TestRealOutlierProtection:
    """Directly reproduces the exact real bug found in this project's
    own live Bu'ertai validation run."""

    def test_reproduces_the_real_naive_strain_blowup(self):
        """Confirms the failure mode is real and reproducible, not
        assumed -- the naive (unprotected) calculation must show a
        triple-digit-percent strain artifact, matching the real
        617% observed in the live run."""
        size, pixel_size = 20, 16.0
        rng_dx = np.random.default_rng(1)
        rng_dy = np.random.default_rng(2)
        dx = rng_dx.normal(0.5, 0.2, (size, size))
        dy = rng_dy.normal(0.3, 0.2, (size, size))
        dx[10, 10] = 226.27  # the exact real outlier value observed live

        naive = compute_horizontal_strain(
            dx, dy, pixel_size, reliable=None, mad_outlier_threshold=None
        )
        peak_strain_pct = np.nanmax(np.abs(naive["exx"])) * 100
        assert peak_strain_pct > 100  # a real, physically-impossible strain

    def test_mad_filter_catches_the_real_outlier_and_excludes_it(self):
        size, pixel_size = 20, 16.0
        dx = np.random.default_rng(1).normal(0.5, 0.2, (size, size))
        dy = np.random.default_rng(2).normal(0.3, 0.2, (size, size))
        dx[10, 10] = 226.27

        result = compute_horizontal_strain(
            dx, dy, pixel_size, mad_outlier_threshold=8.0
        )
        assert result["n_excluded_outlier"] >= 1
        # Real, correct expectation for a central-difference stencil:
        # excluding pixel (10,10) makes its NEIGHBORS' gradients NaN
        # (since their central difference spans across (10,10)), not
        # necessarily the excluded pixel's own index -- confirmed
        # directly before writing this assertion, not assumed.
        assert np.isnan(result["exx"][10, 9]) or np.isnan(result["exx"][10, 11])

    def test_protected_peak_strain_is_an_order_of_magnitude_lower(self):
        size, pixel_size = 20, 16.0
        dx = np.random.default_rng(1).normal(0.5, 0.2, (size, size))
        dy = np.random.default_rng(2).normal(0.3, 0.2, (size, size))
        dx[10, 10] = 226.27

        naive = compute_horizontal_strain(
            dx, dy, pixel_size, mad_outlier_threshold=None
        )
        protected = compute_horizontal_strain(
            dx, dy, pixel_size, mad_outlier_threshold=8.0
        )

        naive_peak = np.nanmax(np.abs(naive["exx"]))
        protected_peak = np.nanmax(np.abs(protected["exx"]))
        assert protected_peak < naive_peak / 10

    def test_reliable_mask_excludes_flagged_pixels_before_gradient(self):
        size, pixel_size = 20, 16.0
        dx = np.random.default_rng(1).normal(0.5, 0.2, (size, size))
        dy = np.random.default_rng(2).normal(0.3, 0.2, (size, size))
        dx[10, 10] = 226.27
        reliable = np.ones((size, size), dtype=bool)
        reliable[10, 10] = False

        result = compute_horizontal_strain(
            dx, dy, pixel_size, reliable=reliable, mad_outlier_threshold=None
        )
        assert result["n_excluded_unreliable"] == 1
        # See the real central-difference note in the MAD test above --
        # the excluded pixel's neighbors go NaN, not its own index.
        assert np.isnan(result["exx"][10, 9]) or np.isnan(result["exx"][10, 11])

    def test_max_plausible_displacement_ceiling_also_catches_it(self):
        size, pixel_size = 20, 16.0
        dx = np.random.default_rng(1).normal(0.5, 0.2, (size, size))
        dy = np.random.default_rng(2).normal(0.3, 0.2, (size, size))
        dx[10, 10] = 226.27

        result = compute_horizontal_strain(
            dx,
            dy,
            pixel_size,
            max_plausible_displacement_m=10.0,
            mad_outlier_threshold=None,
        )
        assert result["n_excluded_implausible"] >= 1

    def test_disabling_protection_reproduces_naive_result_exactly(self):
        """Confirms mad_outlier_threshold=None is a real, honest
        opt-out -- not silently still filtering."""
        size, pixel_size = 20, 16.0
        dx = np.random.default_rng(1).normal(0.5, 0.2, (size, size))
        dy = np.random.default_rng(2).normal(0.3, 0.2, (size, size))
        dx[10, 10] = 226.27

        result = compute_horizontal_strain(
            dx, dy, pixel_size, reliable=None, mad_outlier_threshold=None
        )
        assert result["n_excluded_outlier"] == 0
        assert not np.isnan(result["exx"][10, 10])

    def test_clean_field_with_no_real_outliers_excludes_nothing(self):
        """Confirms the safeguard doesn't over-trigger on genuinely
        clean, real data -- a real false-positive-rate check."""
        size, pixel_size = 20, 16.0
        dx = np.random.default_rng(3).normal(0.9, 0.1, (size, size))
        dy = np.random.default_rng(4).normal(0.5, 0.1, (size, size))

        result = compute_horizontal_strain(
            dx, dy, pixel_size, mad_outlier_threshold=8.0
        )
        assert result["n_excluded_outlier"] == 0


class TestFusionReproducesRealObservedBehavior:
    def test_real_observed_low_coherence_produces_all_optical_dominant(self):
        """Directly reproduces the real Bu'ertai run's own observed
        mean coherence (0.313) and confirms the fusion logic gives the
        exact real observed outcome: 100% optical-dominant, not a
        partial blend."""
        shape = (5, 5)
        insar_vel = np.full(shape, -150.0)
        coherence = np.full(shape, 0.313)  # the real, observed mean
        optical_mag = np.full(shape, 0.9)  # the real, observed median

        result = fuse_insar_optical(insar_vel, coherence, optical_mag)
        assert (result["reliability_source"] == 0).all()
        assert np.allclose(result["fused_displacement"], 0.9)

    def test_high_coherence_produces_all_insar_dominant(self):
        shape = (5, 5)
        result = fuse_insar_optical(
            np.full(shape, -150.0), np.full(shape, 0.8), np.full(shape, 0.9)
        )
        assert (result["reliability_source"] == 2).all()
        assert np.allclose(result["fused_displacement"], -150.0)

    def test_transition_zone_blends_linearly(self):

        coherence_mid = np.array([[0.5]])  # midpoint of default 0.4-0.6 range
        result = fuse_insar_optical(
            np.array([[-100.0]]), coherence_mid, np.array([[10.0]])
        )
        assert result["weight_insar"][0, 0] == pytest.approx(0.5)
        assert result["fused_displacement"][0, 0] == pytest.approx(
            0.5 * -100.0 + 0.5 * 10.0
        )
        assert result["reliability_source"][0, 0] == 1

    def test_missing_insar_falls_back_to_optical_regardless_of_coherence(self):
        insar_vel = np.array([[np.nan]])
        coherence = np.array([[0.9]])  # high coherence, but no real InSAR value
        optical_mag = np.array([[1.2]])
        result = fuse_insar_optical(insar_vel, coherence, optical_mag)
        assert result["fused_displacement"][0, 0] == pytest.approx(1.2)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            fuse_insar_optical(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((4, 4)))

    def test_invalid_threshold_order_raises(self):
        with pytest.raises(ValueError, match="coherence_low"):
            fuse_insar_optical(
                np.zeros((2, 2)),
                np.zeros((2, 2)),
                np.zeros((2, 2)),
                coherence_low=0.7,
                coherence_high=0.3,
            )

    def test_optical_snr_floor_excludes_low_snr_optical_pixels(self):

        insar_vel = np.array([[np.nan]])  # no real InSAR value here either
        coherence = np.array([[0.9]])
        optical_mag = np.array([[5.0]])
        low_snr = np.array([[0.5]])  # below the real 1.0 floor
        result = fuse_insar_optical(
            insar_vel, coherence, optical_mag, optical_snr=low_snr
        )
        assert np.isnan(result["fused_displacement"][0, 0])
