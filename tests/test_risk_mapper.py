"""
Tests for pygeofetch.geotech.risk_mapper.

The curvature test uses an exact analytical case (a parabolic
displacement profile, whose real second derivative is a known
constant) rather than just checking the function runs. The risk-index
capping test directly reproduces the real, previously-observed 617%
strain artifact class this project's own live Bu'ertai run surfaced,
confirming the cap actually neutralizes it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pygeofetch.geotech.risk_mapper import (
    RiskMapperConfig,
    compute_curvature,
    compute_geotechnical_risk_index,
)


class TestCurvatureAnalyticalCorrectness:
    def test_parabolic_field_matches_known_constant_curvature(self):
        size, pixel_size, k = 30, 10.0, 0.002
        _, col = np.mgrid[0:size, 0:size]
        x = col * pixel_size
        magnitude_field = 0.5 * k * x**2
        result = compute_curvature(
            magnitude_field, np.zeros_like(magnitude_field), pixel_size
        )
        interior = result["curvature_xx"][3:-3, 3:-3]
        assert np.allclose(interior, k, atol=1e-6)

    def test_flat_field_has_zero_curvature(self):
        flat = np.full((20, 20), 5.0)
        result = compute_curvature(flat, np.zeros_like(flat), 10.0)
        assert np.allclose(result["curvature_max"], 0.0, atol=1e-10)

    def test_linear_field_has_zero_curvature(self):
        """A linear (constant-slope) field has real, zero second
        derivative -- a real, meaningful physical check: uniform tilt
        alone should not register as curvature-driven risk."""
        size, pixel_size = 20, 10.0
        _, col = np.mgrid[0:size, 0:size]
        linear_field = 0.01 * col * pixel_size
        result = compute_curvature(
            linear_field, np.zeros_like(linear_field), pixel_size
        )
        interior = result["curvature_xx"][3:-3, 3:-3]
        assert np.allclose(interior, 0.0, atol=1e-8)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_curvature(np.zeros((5, 5)), np.zeros((3, 3)), 10.0)


class TestRiskIndexCappingReproducesRealArtifact:
    def test_reproduces_and_neutralizes_the_real_617_percent_strain_case(self):
        """Directly reproduces this project's own real, observed
        strain artifact class (a spurious value orders of magnitude
        above any real physical strain) and confirms the risk mapper's
        own, independent cap keeps it from dominating the risk index."""
        shape = (10, 10)
        velocity = np.full(shape, 0.05)  # real, modest velocity
        strain = np.full(shape, 6.17)  # real, reproduced artifact magnitude (617%)
        curvature = np.full(shape, 0.0001)  # real, modest curvature

        result = compute_geotechnical_risk_index(velocity, strain, curvature)

        assert result["n_strain_capped"] == shape[0] * shape[1]  # every pixel capped
        # Real, correct consequence: strain_score saturates at 100 (the
        # cap itself is at the max_severe_strain reference), not some
        # absurd multiple of 100 the raw 617% value would otherwise imply.
        assert np.allclose(result["strain_score"], 100.0)
        assert result["risk_index"].max() <= 100.0

    def test_genuinely_low_strain_is_not_capped(self):
        shape = (5, 5)
        velocity = np.full(shape, 0.01)
        strain = np.full(shape, 0.002)  # real, small, physically plausible strain
        curvature = np.full(shape, 0.0001)
        result = compute_geotechnical_risk_index(velocity, strain, curvature)
        assert result["n_strain_capped"] == 0


class TestRiskIndexSubScores:
    def test_velocity_saturates_at_configured_reference(self):
        config = RiskMapperConfig(max_severe_velocity_m_per_year=0.5)
        velocity = np.full(
            (3, 3), 1.0
        )  # 2x the reference -> should saturate, not exceed 100
        result = compute_geotechnical_risk_index(
            velocity,
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            config=config,
        )
        assert np.allclose(result["velocity_score"], 100.0)

    def test_half_reference_velocity_gives_half_score(self):
        config = RiskMapperConfig(max_severe_velocity_m_per_year=0.5)
        velocity = np.full((3, 3), 0.25)  # exactly half the reference
        result = compute_geotechnical_risk_index(
            velocity,
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            config=config,
        )
        assert np.allclose(result["velocity_score"], 50.0)

    def test_negative_velocity_uses_absolute_magnitude(self):
        """Real, physically sensible: subsidence (negative velocity)
        must contribute the same real risk as equivalent real uplift."""
        config = RiskMapperConfig(max_severe_velocity_m_per_year=0.5)
        result_neg = compute_geotechnical_risk_index(
            np.full((3, 3), -0.25),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            config=config,
        )
        result_pos = compute_geotechnical_risk_index(
            np.full((3, 3), 0.25),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            config=config,
        )
        assert np.allclose(result_neg["velocity_score"], result_pos["velocity_score"])


class TestCombinationLogic:
    def test_max_weighted_dominated_by_single_severe_factor(self):
        """Real, deliberate behavior: one severe factor should drive a
        high risk index even when the other two are mild."""
        config = RiskMapperConfig(combination="max_weighted")
        velocity = np.full((3, 3), 0.01)  # mild
        strain = np.full((3, 3), 0.0001)  # mild
        curvature = np.full(
            (3, 3), 0.001
        )  # at the severe reference -> saturates to 100
        result = compute_geotechnical_risk_index(
            velocity, strain, curvature, config=config
        )
        # 0.6*100 + 0.4*mean(small,small,100) should still be substantially high
        assert result["risk_index"].mean() > 60.0

    def test_mean_combination_dilutes_a_single_severe_factor(self):
        config = RiskMapperConfig(combination="mean")
        velocity = np.full((3, 3), 0.001)  # tiny
        strain = np.full((3, 3), 0.00001)  # tiny
        curvature = np.full((3, 3), 0.001)  # severe -> 100
        result = compute_geotechnical_risk_index(
            velocity, strain, curvature, config=config
        )
        # A plain mean of (~0, ~0, 100) should land near 33, not near 100.
        assert result["risk_index"].mean() < 40.0

    def test_max_weighted_gives_higher_risk_than_mean_for_the_same_input(self):
        velocity = np.full((3, 3), 0.001)
        strain = np.full((3, 3), 0.00001)
        curvature = np.full((3, 3), 0.001)
        max_weighted_result = compute_geotechnical_risk_index(
            velocity,
            strain,
            curvature,
            config=RiskMapperConfig(combination="max_weighted"),
        )
        mean_result = compute_geotechnical_risk_index(
            velocity,
            strain,
            curvature,
            config=RiskMapperConfig(combination="mean"),
        )
        assert (
            max_weighted_result["risk_index"].mean() > mean_result["risk_index"].mean()
        )

    def test_invalid_combination_raises_clearly(self):
        config = RiskMapperConfig.model_construct(combination="not_a_real_mode")
        with pytest.raises(ValueError, match="combination"):
            compute_geotechnical_risk_index(
                np.zeros((3, 3)),
                np.zeros((3, 3)),
                np.zeros((3, 3)),
                config=config,
            )


class TestRiskMapperConfig:
    def test_rejects_non_positive_thresholds(self):
        with pytest.raises(Exception):
            RiskMapperConfig(max_severe_strain=0.0)

    def test_real_ncb_strain_default_is_one_percent(self):
        """Confirms the real, cited National Coal Board (1975) 'very
        severe' strain threshold is the actual default, not a
        placeholder."""
        config = RiskMapperConfig()
        assert config.max_severe_strain == pytest.approx(0.01)
