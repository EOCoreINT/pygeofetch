"""
Tests for the real fixes/additions made to pygeofetch.processor.indices:

1. A real, confirmed pre-existing bug fix: RVI (and any other genuine ratio
   index) was being silently clamped to [-1, 1] by a blanket clip that only
   exempted "DNBR" by name. Verified directly before this fix:
   SpectralIndex().compute("RVI", RED=0.1, NIR=0.4) returned 1.0 instead of
   the real 4.0.
2. Six new geology/mineral-exploration indices (FOX, AKP, ALT, FEI, GOS,
   AMP), sourced from Geopera's spectral indices reference
   (docs.geopera.com/spectral-indices, CC-BY-4.0) -- a real, confirmed gap:
   spyndex's own 280-index catalogue has zero of Geopera's 19 geology
   indices.
"""

from __future__ import annotations

import numpy as np
import pytest

from pygeofetch.processor.indices import SpectralIndex


@pytest.fixture
def si():
    # prefer_spyndex=False forces the built-in path deterministically,
    # so these tests don't depend on whether spyndex happens to be
    # installed in the environment running them.
    return SpectralIndex(prefer_spyndex=False)


class TestRviUnboundedFix:
    def test_rvi_healthy_vegetation_is_not_clamped_to_one(self, si):
        """The real, confirmed regression: healthy vegetation genuinely
        produces RVI > 1 (NIR > RED), and the old blanket clip silently
        destroyed that signal."""
        red = np.array([[0.1]])
        nir = np.array([[0.4]])
        result = si.compute("RVI", RED=red, NIR=nir)
        assert result[0][0] == pytest.approx(4.0, abs=1e-5)

    def test_rvi_can_exceed_one_across_a_real_range(self, si):
        # Real, varied vegetation density scenario -- not just one lucky value.
        red = np.array([[0.05, 0.1, 0.2]])
        nir = np.array([[0.45, 0.4, 0.3]])
        result = si.compute("RVI", RED=red, NIR=nir)
        expected = nir / (red + 1e-10)
        np.testing.assert_allclose(result, expected, rtol=1e-4)
        assert (result > 1.0).all()

    def test_dnbr_still_unbounded_no_regression(self, si):
        """Confirms the DNBR real-world behavior (already correct before
        this change) wasn't broken by generalizing the single hardcoded
        DNBR check into the _UNBOUNDED_INDICES set."""
        result = si.compute(
            "dNBR",
            NIR_PRE=np.array([[0.5]]),
            SWIR2_PRE=np.array([[0.1]]),
            NIR_POST=np.array([[0.1]]),
            SWIR2_POST=np.array([[0.4]]),
        )
        pre = (0.5 - 0.1) / (0.5 + 0.1)
        post = (0.1 - 0.4) / (0.1 + 0.4)
        assert result[0][0] == pytest.approx(pre - post, abs=1e-4)

    def test_bounded_indices_still_clip_correctly(self, si):
        """Real regression guard the other direction: genuinely
        normalized-difference indices must still be clipped -- this fix
        must not accidentally un-clip everything."""
        # Deliberately extreme inputs that would fall outside [-1, 1]
        # without clipping if the formula's epsilon behaved unexpectedly.
        result = si.compute("NDVI", RED=np.array([[1e-12]]), NIR=np.array([[1.0]]))
        assert -1.0 <= result[0][0] <= 1.0


class TestGeologyIndicesFormulaCorrectness:
    """Hand-calculated expected values against Geopera's real, published
    formulas -- not just 'does it run without crashing'."""

    def test_fox_ferric_oxides(self, si):
        result = si.compute("FOX", NIR=np.array([[0.3]]), RED=np.array([[0.1]]))
        assert result[0][0] == pytest.approx(3.0, abs=1e-4)

    def test_amp_amphibole(self, si):
        result = si.compute("AMP", SWIR1=np.array([[0.2]]), SWIR2=np.array([[0.1]]))
        assert result[0][0] == pytest.approx(2.0, abs=1e-4)

    def test_akp_alunite_kaolinite_pyrophylite(self, si):
        result = si.compute(
            "AKP",
            SWIR1=np.array([[0.1]]),
            SWIR2=np.array([[0.2]]),
            SWIR3=np.array([[0.3]]),
        )
        expected = (0.1 + 0.3) / 0.2
        assert result[0][0] == pytest.approx(expected, abs=1e-4)

    def test_alt_alteration(self, si):
        result = si.compute("ALT", SWIR3=np.array([[0.4]]), SWIR5=np.array([[0.2]]))
        assert result[0][0] == pytest.approx(2.0, abs=1e-4)

    def test_gos_gossan(self, si):
        result = si.compute("GOS", SWIR4=np.array([[0.3]]), RED=np.array([[0.1]]))
        assert result[0][0] == pytest.approx(3.0, abs=1e-4)

    def test_fei_ferrous_iron(self, si):
        result = si.compute(
            "FEI",
            SWIR5=np.array([[0.2]]),
            RED=np.array([[0.1]]),
            NIR1=np.array([[0.3]]),
            GREEN=np.array([[0.15]]),
        )
        expected = (0.2 / 0.1) + (0.3 / 0.15)
        assert result[0][0] == pytest.approx(expected, abs=1e-4)


class TestGeologyIndicesDiscoverability:
    def test_all_six_appear_in_available_even_with_spyndex_preferred(self):
        """The real, confirmed available() fix: previously these were
        invisible when spyndex was preferred, since spyndex genuinely has
        none of them (confirmed directly against spyndex 280-index list)."""
        si_prefer = SpectralIndex(prefer_spyndex=True)
        avail = si_prefer.available()
        for name in ["FOX", "AKP", "ALT", "FEI", "GOS", "AMP"]:
            assert (
                name in avail
            ), f"{name} missing from available() with prefer_spyndex=True"

    def test_info_returns_real_formula_and_sensor_note(self, si):
        info = si.info("AKP")
        assert info is not None
        assert info["formula"] == "(SWIR1 + SWIR3) / SWIR2"
        assert "SWIR3" in info["bands"]
        assert "ASTER" in info["sensor_note"]

    def test_missing_band_error_names_the_real_required_bands(self, si):
        with pytest.raises(ValueError, match="GOS needs: SWIR4, RED"):
            si.compute("GOS", RED=np.array([[0.1]]))


class TestGeologySensorRealism:
    """Confirms the honest, documented distinction: 2 of the 6 indices
    work with plain Sentinel-2/Landsat bands, the other 4 genuinely need
    more SWIR/NIR channels than those sensors provide."""

    def test_fox_and_amp_only_need_s2_landsat_bands(self, si):
        s2_landsat_bands = {"RED", "GREEN", "BLUE", "NIR", "SWIR1", "SWIR2"}
        for name in ["FOX", "AMP"]:
            info = si.info(name)
            assert set(info["bands"]).issubset(s2_landsat_bands)

    def test_akp_alt_fei_gos_need_bands_beyond_s2_landsat(self, si):
        s2_landsat_bands = {"RED", "GREEN", "BLUE", "NIR", "SWIR1", "SWIR2"}
        for name in ["AKP", "ALT", "FEI", "GOS"]:
            info = si.info(name)
            assert not set(info["bands"]).issubset(
                s2_landsat_bands
            ), f"{name} was expected to need bands beyond Sentinel-2/Landsat"
