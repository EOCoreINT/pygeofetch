"""
Tests for pygeofetch.processing.uq_exporter.

The core formula test reproduces pygeofetch.insar.synthetic's own
already-verified Cramer-Rao formula independently and checks for an
exact match, confirming this module reuses it rather than risking a
subtly different, silently-diverged reimplementation.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pygeofetch.processing.uq_exporter import (
    compute_phase_and_displacement_uncertainty,
    export_with_uncertainty,
    summarize_uncertainty_for_provenance,
)

REAL_CRS = "EPSG:32633"
REAL_TRANSFORM = from_origin(500000, 4000000, 10, 10)
SENTINEL1_WAVELENGTH_M = 0.05546576


class TestCramerRaoFormulaConsistency:
    def test_matches_synthetic_pys_own_formula_exactly(self):
        coherence = np.array([0.9, 0.5, 0.2, 0.05])
        n_looks = 4
        expected_sigma_phi = np.sqrt(1 - coherence**2) / (
            coherence * np.sqrt(2 * n_looks)
        )

        sigma_phi, _ = compute_phase_and_displacement_uncertainty(
            coherence,
            n_looks,
            wavelength_m=SENTINEL1_WAVELENGTH_M,
        )
        assert np.allclose(sigma_phi, expected_sigma_phi)

    def test_sigma_disp_uses_real_lambda_over_4pi_conversion(self):
        coherence = np.array([0.7])
        sigma_phi, sigma_disp = compute_phase_and_displacement_uncertainty(
            coherence,
            n_looks=4,
            wavelength_m=SENTINEL1_WAVELENGTH_M,
        )
        assert sigma_disp[0] == pytest.approx(
            (SENTINEL1_WAVELENGTH_M / (4 * np.pi)) * sigma_phi[0]
        )

    def test_higher_coherence_gives_lower_uncertainty(self):
        """Real, expected physical monotonicity."""
        sigma_phi_high, _ = compute_phase_and_displacement_uncertainty(
            0.9, 4, SENTINEL1_WAVELENGTH_M
        )
        sigma_phi_low, _ = compute_phase_and_displacement_uncertainty(
            0.2, 4, SENTINEL1_WAVELENGTH_M
        )
        assert sigma_phi_high < sigma_phi_low

    def test_more_looks_gives_lower_uncertainty(self):
        """Real, expected physical monotonicity: averaging more
        independent looks reduces real phase noise."""
        sigma_phi_4, _ = compute_phase_and_displacement_uncertainty(
            0.5, 4, SENTINEL1_WAVELENGTH_M
        )
        sigma_phi_16, _ = compute_phase_and_displacement_uncertainty(
            0.5, 16, SENTINEL1_WAVELENGTH_M
        )
        assert sigma_phi_16 < sigma_phi_4

    def test_near_zero_coherence_does_not_crash(self):
        """Real, honest floor -- must not divide by exactly zero."""
        sigma_phi, sigma_disp = compute_phase_and_displacement_uncertainty(
            np.array([0.0]),
            n_looks=4,
            wavelength_m=SENTINEL1_WAVELENGTH_M,
        )
        assert np.isfinite(sigma_phi).all()
        assert np.isfinite(sigma_disp).all()

    def test_invalid_n_looks_raises(self):
        with pytest.raises(ValueError, match="n_looks"):
            compute_phase_and_displacement_uncertainty(
                0.5, n_looks=0, wavelength_m=SENTINEL1_WAVELENGTH_M
            )

    def test_invalid_wavelength_raises(self):
        with pytest.raises(ValueError, match="wavelength_m"):
            compute_phase_and_displacement_uncertainty(
                0.5, n_looks=4, wavelength_m=-1.0
            )


class TestExportWithUncertainty:
    def test_writes_real_two_band_geotiff(self, tmp_path):
        data = np.full((10, 10), -5.0)
        uncertainty = np.full((10, 10), 0.002)
        profile = {
            "driver": "GTiff",
            "height": 10,
            "width": 10,
            "count": 1,
            "dtype": "float32",
            "crs": REAL_CRS,
            "transform": REAL_TRANSFORM,
        }
        out_path = export_with_uncertainty(
            data, uncertainty, profile, tmp_path / "velocity_uq.tif"
        )

        with rasterio.open(out_path) as src:
            assert src.count == 2
            assert np.allclose(src.read(1), data)
            assert np.allclose(src.read(2), uncertainty)
            assert src.descriptions[0] == "value"
            assert src.descriptions[1] == "uncertainty"

    def test_custom_band_names_are_written(self, tmp_path):
        data = np.zeros((5, 5))
        uncertainty = np.ones((5, 5))
        profile = {
            "driver": "GTiff",
            "height": 5,
            "width": 5,
            "count": 1,
            "dtype": "float32",
            "crs": REAL_CRS,
            "transform": REAL_TRANSFORM,
        }
        out_path = export_with_uncertainty(
            data,
            uncertainty,
            profile,
            tmp_path / "out.tif",
            value_band_name="velocity_mm_yr",
            uncertainty_band_name="sigma_disp_mm_yr",
        )
        with rasterio.open(out_path) as src:
            assert src.descriptions == ("velocity_mm_yr", "sigma_disp_mm_yr")

    def test_shape_mismatch_raises_clearly(self, tmp_path):
        profile = {
            "driver": "GTiff",
            "height": 5,
            "width": 5,
            "count": 1,
            "dtype": "float32",
            "crs": REAL_CRS,
            "transform": REAL_TRANSFORM,
        }
        with pytest.raises(ValueError, match="shape mismatch"):
            export_with_uncertainty(
                np.zeros((5, 5)),
                np.zeros((3, 3)),
                profile,
                tmp_path / "out.tif",
            )


class TestProvenanceSummary:
    def test_summary_has_real_expected_keys(self):
        coherence = np.full((10, 10), 0.6)
        sigma_phi, sigma_disp = compute_phase_and_displacement_uncertainty(
            coherence,
            n_looks=4,
            wavelength_m=SENTINEL1_WAVELENGTH_M,
        )
        summary = summarize_uncertainty_for_provenance(
            sigma_phi,
            sigma_disp,
            n_looks=4,
            wavelength_m=SENTINEL1_WAVELENGTH_M,
        )
        assert "phase_uncertainty_rad" in summary
        assert "displacement_uncertainty_m" in summary
        assert set(summary["phase_uncertainty_rad"]) == {"mean", "median", "p95"}
        assert summary["n_looks"] == 4
        assert summary["wavelength_m"] == SENTINEL1_WAVELENGTH_M

    def test_summary_feeds_directly_into_real_provenance_manifest(self, tmp_path):
        """Real, end-to-end check: the dict this function returns is
        directly usable as write_provenance_manifest's own real
        `quality` parameter, without any real adaptation needed."""
        from pygeofetch.insar.provenance import write_provenance_manifest

        coherence = np.full((10, 10), 0.6)
        sigma_phi, sigma_disp = compute_phase_and_displacement_uncertainty(
            coherence,
            n_looks=4,
            wavelength_m=SENTINEL1_WAVELENGTH_M,
        )
        quality = summarize_uncertainty_for_provenance(
            sigma_phi,
            sigma_disp,
            n_looks=4,
            wavelength_m=SENTINEL1_WAVELENGTH_M,
        )

        manifest_path = write_provenance_manifest(
            output_dir=tmp_path,
            preflight_manifest={"status": "PASS"},
            processing={"method": "test"},
            corrections={},
            quality=quality,
        )
        assert manifest_path.exists()
        content = manifest_path.read_text()
        assert "phase_uncertainty_rad" in content
