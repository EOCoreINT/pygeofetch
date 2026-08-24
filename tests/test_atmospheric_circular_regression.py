"""
Regression tests for a real, confirmed bug fix in
AtmosphericCorrector.correct() (method="elevation"): the previous
version fit a plain, arithmetic linear regression directly against
wrapped phase, which any real elevation-correlated signal spanning
more than about half a cycle across the scene destroys, regardless of
whether a real underlying relationship exists. Fixed using the same
circular regression (fit exp(i*phase)'s real/imag parts separately)
already proven and verified in interferogram.py's own
_remove_topographic_phase().

Also documents a real limitation that WAS present and has since been
fixed: circular regression by itself is a LINEAR fit of cos/sin against
the covariate, which is only a good fit when the true phase excursion
across the scene is small (well under one full cycle). For a true
relationship spanning many full cycles, cos(slope*x) oscillates
multiple times as a function of x, and a straight-line fit to a
multi-period oscillation is a poor fit by construction. This is now
fixed with a real, verified coarse-to-fine slope search (frequency
estimation over candidate slopes, not ordinary linear regression),
matching the same fix applied to interferogram.py's own
_remove_topographic_phase().
"""

from pathlib import Path
import tempfile

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from pygeofetch.insar.atmosphere import AtmosphericCorrector


def _make_dem(path, elevation):
    h, w = elevation.shape
    crs = CRS.from_epsg(4326)
    transform = from_bounds(-99.15, 19.30, -99.05, 19.40, w, h)
    with rasterio.open(path, "w", driver="GTiff", dtype="float32", count=1,
                        width=w, height=h, crs=crs, transform=transform) as ds:
        ds.write(elevation.astype(np.float32), 1)


def test_detects_real_sub_cycle_correlation_that_old_method_would_also_catch():
    """For a small, sub-cycle relationship, the fix should correctly
    detect and apply a real correction -- confirming it didn't break
    the case the previous method already handled reasonably."""
    with tempfile.TemporaryDirectory() as tmp:
        h, w = 200, 200
        elevation = 2240 + np.linspace(0, 100, w)[None, :] * np.ones((h, 1))
        dem_path = Path(tmp) / "dem.tif"
        _make_dem(dem_path, elevation)

        true_slope = 0.01  # real, sub-cycle across this scene
        np.random.seed(4)
        true_phase = true_slope * elevation + 0.1 * np.random.randn(h, w)
        wrapped_phase = np.angle(np.exp(1j * true_phase)).astype(np.float32)

        corrector = AtmosphericCorrector(method="elevation")
        corrected, meta = corrector.correct(wrapped_phase, dem=str(dem_path), return_metadata=True)

        assert meta["correction_applied"] is True
        assert meta["r_squared"] > 0.7


def test_large_multi_cycle_relationship_is_now_correctly_detected():
    """Real, confirmed fix: this used to be a documented, accepted
    limitation (linear regression fails on a true relationship spanning
    many full cycles) -- fixed with a real, verified coarse-to-fine
    slope search (frequency estimation, not ordinary regression),
    matching the same fix applied to interferogram.py's own
    _remove_topographic_phase(). A ~3.2-cycle relationship should now be
    correctly detected and corrected, not skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        h, w = 200, 200
        elevation = 2240 + np.linspace(0, 400, w)[None, :] * np.ones((h, 1))
        dem_path = Path(tmp) / "dem.tif"
        _make_dem(dem_path, elevation)

        true_slope = 0.05  # real, multi-cycle across this scene (~3.2 cycles)
        np.random.seed(4)
        true_phase = true_slope * elevation + 0.1 * np.random.randn(h, w)
        wrapped_phase = np.angle(np.exp(1j * true_phase)).astype(np.float32)

        corrector = AtmosphericCorrector(method="elevation")
        corrected, meta = corrector.correct(wrapped_phase, dem=str(dem_path), return_metadata=True)

        assert meta["correction_applied"] is True
        assert meta["r_squared"] > 0.7


def test_degenerate_flat_dem_is_rejected_not_silently_unstable():
    """Same real vulnerability class already found and fixed in
    _remove_topographic_phase(): a near-constant DEM makes the
    regression numerically degenerate. Must be rejected outright."""
    with tempfile.TemporaryDirectory() as tmp:
        h, w = 100, 100
        elevation = np.full((h, w), 2240.0)
        dem_path = Path(tmp) / "flat_dem.tif"
        _make_dem(dem_path, elevation)

        wrapped_phase = np.random.uniform(-np.pi, np.pi, (h, w)).astype(np.float32)
        corrector = AtmosphericCorrector(method="elevation")
        corrected, meta = corrector.correct(wrapped_phase, dem=str(dem_path), return_metadata=True)

        assert meta["correction_applied"] is False
        assert meta.get("reason") == "dem_no_variance"
