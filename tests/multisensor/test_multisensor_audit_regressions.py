"""
Regression tests for defects found while auditing pygeofetch.multisensor.

Each test encodes physics or arithmetic that is computed INDEPENDENTLY of the
code under test (surface-normal . sun-vector for illumination, WGS84 geodesic
areas for volumes), because the original tests for these pipelines were
self-consistent: they built their inputs by inverting the pipeline's own
formula, so a convention error in the pipeline cancelled out and went unseen.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import rasterio
from pyproj import Geod
from rasterio.transform import from_origin

from pygeofetch.multisensor.dem_differencing import dem_differencing_pipeline
from pygeofetch.multisensor.insar_optical_fusion import insar_optical_displacement_pipeline
from pygeofetch.multisensor.terrain_corrected_change import (
    solar_position,
    terrain_corrected_change_pipeline,
)
from pygeofetch.multisensor.vegetation_disturbance import (
    CLASS_NO_DISTURBANCE,
    CLASS_SUBCANOPY_DISTURBANCE,
    CLASS_VISIBLE_DISTURBANCE,
    vegetation_disturbance_pipeline,
)


def _write(path, data, *, crs="EPSG:32630", transform=None, nodata=None, dtype="float32"):
    transform = transform or from_origin(600000, 700000, 10, 10)
    prof = dict(driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                dtype=dtype, crs=crs, transform=transform, nodata=nodata)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data.astype(dtype), 1)
    return path


# ============================================================================
# Terrain-corrected change: the aspect convention
# ============================================================================
def _lambertian_scene(n=60, slope_per_m=0.3, rho=0.3, dts=None, px_x=10.0, px_y=10.0):
    """A tent-shaped ridge (west half faces west, east half faces east) with a
    uniform Lambertian surface. Observed = rho * cos(i) / cos(zenith), with
    cos(i) = unit_normal . unit_sun, computed from vectors -- not from the
    pipeline's aspect formula."""
    x_m = (np.arange(n) + 0.5) * px_x
    xc = n * px_x / 2
    z = 500.0 - slope_per_m * np.abs(x_m - xc)               # peak in the middle
    dem = np.tile(z, (n, 1))
    gx = np.where(x_m < xc, +slope_per_m, -slope_per_m)       # dz/dx (east)
    normal = np.stack([-gx, np.zeros_like(gx), np.ones_like(gx)])   # (east, north, up)
    normal = normal / np.linalg.norm(normal, axis=0)
    obs = []
    for dt in dts:
        zen, az = solar_position(dt, 6.2, -1.69)
        zr, ar = np.radians(zen), np.radians(az)
        sun = np.array([np.sin(zr) * np.sin(ar), np.sin(zr) * np.cos(ar), np.cos(zr)])
        cos_i = np.clip(normal.T @ sun, 0.0, None)            # (n,)
        obs.append(np.tile(rho * cos_i / np.cos(zr), (n, 1)))
    return dem, obs, x_m


AM, PM = datetime(2024, 1, 15, 8, 0), datetime(2024, 1, 15, 16, 0)   # sun in the east, then west


class TestTerrainCorrectionConvention:
    def test_uncorrected_pair_shows_large_spurious_change(self):
        """Precondition: the scene is a meaningful test (the sun moving alone
        produces big apparent change on the ridge flanks)."""
        _, (pre, post), _ = _lambertian_scene(dts=[AM, PM])
        assert np.abs(post - pre).max() > 0.08

    def test_correction_recovers_true_reflectance_on_both_flanks(self, tmp_path):
        dem, (pre, post), x_m = _lambertian_scene(dts=[AM, PM], rho=0.3)
        paths = [_write(tmp_path / f"{k}.tif", d) for k, d in
                 {"pre": pre, "post": post, "dem": dem}.items()]
        res = terrain_corrected_change_pipeline(
            paths[0], paths[1], paths[2], AM, PM, 6.2, -1.69, tmp_path / "out",
            change_threshold=0.02,
        )
        assert res.success, res.error
        with rasterio.open(res.final_output) as src:
            change = src.read(1)
        interior = np.zeros(change.shape, bool)
        interior[5:-5, 5:-5] = True
        interior[:, 27:33] = False                                     # smeared ridge line
        assert np.nanmax(np.abs(change[interior])) < 0.02, (
            "A surface of constant reflectance must show ~no change after correction. "
            "With the aspect inverted, east- and west-facing flanks are 'corrected' the "
            "wrong way and apparent change is amplified instead of removed."
        )
        assert res.metadata["n_changed_pixels"] == 0 or res.metadata["pct_changed"] < 5

    def test_real_change_is_still_detected(self, tmp_path):
        dem, (pre, post), _ = _lambertian_scene(dts=[AM, PM], rho=0.3)
        post = post.copy()
        post[:, 10:20] *= 1.6                                          # genuine brightening on the west flank
        paths = [_write(tmp_path / f"{k}.tif", d) for k, d in
                 {"pre": pre, "post": post, "dem": dem}.items()]
        res = terrain_corrected_change_pipeline(
            paths[0], paths[1], paths[2], AM, PM, 6.2, -1.69, tmp_path / "out",
            change_threshold=0.1,
        )
        with rasterio.open(res.metadata["change_mask_path"]) as src:
            m = src.read(1)
        assert m[10:50, 12:18].mean() > 0.9            # true change found
        assert m[10:50, 40:55].mean() < 0.05           # untouched east flank stays quiet

    def test_geographic_raster_uses_metres_not_degrees(self, tmp_path):
        """A degree-based grid must be converted to metres; treating 1e-4 deg
        as 1e-4 m gives ~90 degree slopes everywhere and garbage correction."""
        lat = 6.2
        px_deg = 0.0001
        px_x = px_deg * 111320.0 * np.cos(np.radians(lat))
        px_y = px_deg * 110574.0
        dem, (pre, post), _ = _lambertian_scene(dts=[AM, PM], px_x=px_x, px_y=px_y)
        tf = from_origin(-1.69, lat + 0.003, px_deg, px_deg)
        paths = [_write(tmp_path / f"{k}.tif", d, crs="EPSG:4326", transform=tf)
                 for k, d in {"pre": pre, "post": post, "dem": dem}.items()]
        res = terrain_corrected_change_pipeline(
            paths[0], paths[1], paths[2], AM, PM, lat, -1.69, tmp_path / "out",
            change_threshold=0.03,
        )
        assert res.success, res.error
        with rasterio.open(res.final_output) as src:
            change = src.read(1)
        interior = np.zeros(change.shape, bool)
        interior[5:-5, 5:-5] = True
        interior[:, 27:33] = False
        assert np.nanmax(np.abs(change[interior])) < 0.03

    def test_nodata_in_dem_and_optical_is_unclassified_not_changed(self, tmp_path):
        dem, (pre, post), _ = _lambertian_scene(dts=[AM, PM])
        dem[5:10, 5:10] = -9999.0
        post = post.copy(); post[40:45, 40:45] = -9999.0
        paths = [
            _write(tmp_path / "pre.tif", pre, nodata=-9999.0),
            _write(tmp_path / "post.tif", post, nodata=-9999.0),
            _write(tmp_path / "dem.tif", dem, nodata=-9999.0),
        ]
        res = terrain_corrected_change_pipeline(
            paths[0], paths[1], paths[2], AM, PM, 6.2, -1.69, tmp_path / "out")
        assert res.success, res.error
        with rasterio.open(res.metadata["change_mask_path"]) as src:
            m = src.read(1)
        assert (m[5:10, 5:10] == 255).all() and (m[40:45, 40:45] == 255).all()
        assert res.metadata["n_invalid_pixels"] > 50


# ============================================================================
# DEM differencing: nodata sentinel and geographic grids
# ============================================================================
class TestDemDifferencing:
    def test_same_grid_new_dem_nodata_is_not_treated_as_elevation(self, tmp_path):
        old = np.full((40, 40), 100.0)
        new = old.copy(); new[10:20, 10:20] = -9999.0                   # voids in the new DEM
        a = _write(tmp_path / "old.tif", old, nodata=-9999.0)
        b = _write(tmp_path / "new.tif", new, nodata=-9999.0)
        res = dem_differencing_pipeline(a, b, tmp_path / "o")
        assert res.success
        assert res.metadata["erosion_volume_m3"] == 0.0, "voids must not become ~10 km of erosion"
        assert res.metadata["n_valid_pixels"] == 40 * 40 - 100
        assert res.metadata["n_significant_pixels"] == 0

    def test_geographic_dem_volume_matches_geodesic_area(self, tmp_path):
        px = 0.0003                                                     # ~33 m
        lat0, lon0 = 6.25, -1.70
        tf = from_origin(lon0, lat0, px, px)
        old = np.zeros((60, 60)); new = old.copy(); new[20:40, 20:40] = 5.0   # +5 m on 20x20 px
        a = _write(tmp_path / "old.tif", old, crs="EPSG:4326", transform=tf)
        b = _write(tmp_path / "new.tif", new, crs="EPSG:4326", transform=tf)
        res = dem_differencing_pipeline(a, b, tmp_path / "o")
        w, e = lon0 + 20 * px, lon0 + 40 * px
        n, s = lat0 - 20 * px, lat0 - 40 * px
        area, _ = Geod(ellps="WGS84").polygon_area_perimeter([w, e, e, w], [s, s, n, n])
        want = abs(area) * 5.0
        assert res.metadata["deposition_volume_m3"] == pytest.approx(want, rel=0.01)
        assert res.metadata["pixel_area_m2"] > 100.0, "pixel area must be in m^2, not degrees^2"

    def test_projected_volume_unchanged_for_metre_grids(self, tmp_path):
        tf = from_origin(500000, 4000000, 10, 10)
        old = np.zeros((30, 30)); new = old.copy(); new[5:15, 5:15] = 3.0
        res = dem_differencing_pipeline(_write(tmp_path / "o.tif", old, crs="EPSG:32633", transform=tf),
                                        _write(tmp_path / "n.tif", new, crs="EPSG:32633", transform=tf),
                                        tmp_path / "out")
        assert res.metadata["deposition_volume_m3"] == pytest.approx(100 * 3.0 * 100, rel=1e-9)


# ============================================================================
# Vegetation disturbance: no evidence != "no NDVI drop"
# ============================================================================
def _veg_inputs(tmp_path, size=30, seed=0, *, red=0.05, nir=0.45):
    rng = np.random.default_rng(seed)
    slc_pre = (rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))).astype("complex64")
    slc_post = (slc_pre * np.exp(1j * 0.05)).astype("complex64")
    return {
        "rp": np.full((size, size), red, "float32"), "np_": np.full((size, size), nir, "float32"),
        "rq": np.full((size, size), red, "float32"), "nq": np.full((size, size), nir, "float32"),
        "sp": slc_pre, "sq": slc_post,
    }


def _run_veg(tmp_path, d, **kw):
    w = lambda k, v, dt="float32", nd=None: _write(tmp_path / f"{k}.tif", v, dtype=dt, nodata=nd)
    return vegetation_disturbance_pipeline(
        w("rp", d["rp"]), w("np", d["np_"]), w("rq", d["rq"]), w("nq", d["nq"]),
        w("sp", d["sp"], "complex64"), w("sq", d["sq"], "complex64"), tmp_path / "o", **kw)


class TestVegetationDisturbance:
    def test_missing_optical_with_low_coherence_is_not_labelled_subcanopy(self, tmp_path):
        d = _veg_inputs(tmp_path)
        rng = np.random.default_rng(5)
        blk = (slice(10, 20), slice(10, 20))
        d["sq"][blk] = (rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10))).astype("complex64")  # decorrelated
        for k in ("rp", "np_", "rq", "nq"):
            d[k][blk] = 0.0                                             # optical has no data here
        res = _run_veg(tmp_path, d)
        assert res.success, res.error
        with rasterio.open(res.final_output) as src:
            cls = src.read(1)
        assert (cls[blk] == 255).all(), "no optical data => unclassified, not 'sub-canopy disturbance'"
        assert (cls[0:8, 0:8] == CLASS_NO_DISTURBANCE).all()
        assert res.metadata["n_unclassified_pixels"] >= 100

    def test_real_subcanopy_disturbance_is_still_found(self, tmp_path):
        d = _veg_inputs(tmp_path, seed=3)
        rng = np.random.default_rng(6)
        blk = (slice(10, 20), slice(10, 20))
        d["sq"][blk] = (rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10))).astype("complex64")
        res = _run_veg(tmp_path, d)
        with rasterio.open(res.final_output) as src:
            cls = src.read(1)
        assert (cls[12:18, 12:18] == CLASS_SUBCANOPY_DISTURBANCE).mean() > 0.9

    def test_zero_sar_footprint_is_unclassified_not_undisturbed(self, tmp_path):
        d = _veg_inputs(tmp_path)
        blk = (slice(0, 10), slice(20, 30))
        d["sp"][blk] = 0; d["sq"][blk] = 0                              # outside the SAR footprint
        res = _run_veg(tmp_path, d)
        with rasterio.open(res.final_output) as src:
            cls = src.read(1)
        assert (cls[blk] == 255).all()

    def test_dn_offset_changes_visible_disturbance_decision(self, tmp_path):
        """S2 L2A baseline>=04: DN = refl*1e4 + 1000. Reflectance NDVI drop is
        0.292 (> 0.25 threshold) but raw-DN NDVI drop is only 0.198."""
        def build(offset):
            d = _veg_inputs(tmp_path)
            f = lambda r: r * 1e4 + offset
            d["rp"][:] = f(0.06); d["np_"][:] = f(0.30); d["rq"][:] = f(0.10); d["nq"][:] = f(0.22)
            return d
        raw = _run_veg(tmp_path, build(1000.0), ndvi_drop_threshold=0.25)
        with rasterio.open(raw.final_output) as src:
            assert (src.read(1) == CLASS_VISIBLE_DISTURBANCE).sum() == 0
        fixed = _run_veg(tmp_path, build(1000.0), ndvi_drop_threshold=0.25,
                         optical_offset=-1000.0, optical_scale=1e-4)
        with rasterio.open(fixed.final_output) as src:
            assert (src.read(1) == CLASS_VISIBLE_DISTURBANCE).all()

    def test_mismatched_optical_shapes_fail_cleanly(self, tmp_path):
        d = _veg_inputs(tmp_path)
        d["np_"] = d["np_"][:20, :20]
        res = _run_veg(tmp_path, d)
        assert not res.success and "nir pre" in res.error


# ============================================================================
# InSAR + optical: geographic optical input is rejected (existing behaviour,
# provided by prepare_optical_pair; pinned here so it cannot regress)
# ============================================================================
def test_insar_optical_rejects_geographic_optical(tmp_path):
    tf = from_origin(-1.7, 6.3, 0.0001, 0.0001)
    rng = np.random.default_rng(0)
    img = rng.random((200, 200)).astype("float32")
    v = _write(tmp_path / "v.tif", np.zeros((20, 20)), crs="EPSG:4326", transform=tf)
    c = _write(tmp_path / "c.tif", np.ones((20, 20)), crs="EPSG:4326", transform=tf)
    a = _write(tmp_path / "a.tif", img, crs="EPSG:4326", transform=tf)
    b = _write(tmp_path / "b.tif", img, crs="EPSG:4326", transform=tf)
    res = insar_optical_displacement_pipeline(v, c, a, b, tmp_path / "o")
    assert not res.success
    assert "projected" in res.error
