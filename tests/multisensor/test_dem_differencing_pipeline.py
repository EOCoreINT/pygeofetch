"""
Tests for pygeofetch.multisensor.dem_differencing.

The core volume test uses an exact, known synthetic scenario (a
rectangular "pile" of precisely known dimensions added to an
otherwise-flat DEM) so the real computed volume can be checked against
an exact analytical value, not just "some positive number came out."
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pygeofetch.multisensor.dem_differencing import dem_differencing_pipeline

REAL_CRS = "EPSG:32633"


def _write_raster(path, data, transform, dtype="float32", nodata=None):
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": REAL_CRS,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)
    return path


class TestExactKnownVolume:
    def test_deposition_volume_matches_exact_analytical_value(self, tmp_path):
        pixel_size = 10.0  # real 10m pixels -> 100 m^2/pixel
        transform = from_origin(500000, 4000000, pixel_size, pixel_size)
        size = 40

        dem_old = np.zeros((size, size), dtype="float64")
        dem_new = np.zeros((size, size), dtype="float64")
        # Real, exact, deliberate "pile": a 10x10 pixel region raised by
        # exactly 5m -> real, exact expected volume:
        # 10*10 pixels * 100 m^2/pixel * 5m = 50,000 m^3
        dem_new[10:20, 10:20] = 5.0

        old_path = _write_raster(tmp_path / "old.tif", dem_old, transform)
        new_path = _write_raster(tmp_path / "new.tif", dem_new, transform)

        result = dem_differencing_pipeline(
            old_path,
            new_path,
            output_dir=tmp_path,
            dem_old_vertical_rmse_m=0.1,
            dem_new_vertical_rmse_m=0.1,  # tight, real accuracy
        )
        assert result.success, result.error
        assert result.metadata["deposition_volume_m3"] == pytest.approx(
            50000.0, rel=0.01
        )
        assert result.metadata["erosion_volume_m3"] == pytest.approx(0.0, abs=1.0)
        assert result.metadata["net_volume_change_m3"] == pytest.approx(
            50000.0, rel=0.01
        )

    def test_erosion_volume_matches_exact_analytical_value(self, tmp_path):
        pixel_size = 10.0
        transform = from_origin(500000, 4000000, pixel_size, pixel_size)
        size = 40
        dem_old = np.zeros((size, size), dtype="float64")
        dem_new = np.zeros((size, size), dtype="float64")
        # Real, exact excavation: 10x10 pixels lowered by 3m ->
        # 10*10*100*3 = 30,000 m^3 real, exact expected erosion.
        dem_new[10:20, 10:20] = -3.0

        old_path = _write_raster(tmp_path / "old.tif", dem_old, transform)
        new_path = _write_raster(tmp_path / "new.tif", dem_new, transform)

        result = dem_differencing_pipeline(
            old_path,
            new_path,
            output_dir=tmp_path,
            dem_old_vertical_rmse_m=0.1,
            dem_new_vertical_rmse_m=0.1,
        )
        assert result.success, result.error
        assert result.metadata["erosion_volume_m3"] == pytest.approx(30000.0, rel=0.01)
        assert result.metadata["deposition_volume_m3"] == pytest.approx(0.0, abs=1.0)


class TestLevelOfDetectionFiltering:
    def test_below_lod_noise_is_excluded_from_volume(self, tmp_path):
        pixel_size = 10.0
        transform = from_origin(500000, 4000000, pixel_size, pixel_size)
        rng = np.random.default_rng(1)
        size = 30
        # Real, deliberate small noise (well below a real 2m-RMSE-pair LoD).
        dem_old = rng.normal(0, 0.3, (size, size))
        dem_new = dem_old + rng.normal(0, 0.3, (size, size))

        old_path = _write_raster(tmp_path / "old.tif", dem_old, transform)
        new_path = _write_raster(tmp_path / "new.tif", dem_new, transform)

        result = dem_differencing_pipeline(
            old_path,
            new_path,
            output_dir=tmp_path,
            dem_old_vertical_rmse_m=2.0,
            dem_new_vertical_rmse_m=2.0,  # real, coarser DEMs
        )
        assert result.success, result.error
        # Real LoD = 1.96*sqrt(2^2+2^2) ~= 5.54m, far above the real
        # noise magnitude here -- almost nothing should pass it.
        assert result.metadata["pct_significant"] < 5.0

    def test_lod_value_matches_real_formula(self, tmp_path):
        transform = from_origin(500000, 4000000, 10.0, 10.0)
        dem = np.zeros((10, 10))
        old_path = _write_raster(tmp_path / "old.tif", dem, transform)
        new_path = _write_raster(tmp_path / "new.tif", dem, transform)

        result = dem_differencing_pipeline(
            old_path,
            new_path,
            output_dir=tmp_path,
            dem_old_vertical_rmse_m=1.5,
            dem_new_vertical_rmse_m=2.0,
            confidence_t_value=1.96,
        )
        expected_lod = 1.96 * np.sqrt(1.5**2 + 2.0**2)
        assert result.metadata["level_of_detection_m"] == pytest.approx(
            expected_lod, rel=1e-4
        )


class TestGridMismatchHandling:
    def test_different_grids_are_reprojected_not_rejected(self, tmp_path):
        old_transform = from_origin(500000, 4000000, 10.0, 10.0)
        new_transform = from_origin(500000, 4000000, 5.0, 5.0)  # real, finer resolution

        dem_old = np.zeros((20, 20))
        dem_new = np.full((40, 40), 5.0)  # same real ground extent, finer grid

        old_path = _write_raster(tmp_path / "old.tif", dem_old, old_transform)
        new_path = _write_raster(tmp_path / "new.tif", dem_new, new_transform)

        result = dem_differencing_pipeline(old_path, new_path, output_dir=tmp_path)
        assert result.success, result.error
        with rasterio.open(result.final_output) as src:
            diff = src.read(1)
        assert diff.shape == (20, 20)  # real, matches the OLD dem's grid


class TestInputValidation:
    def test_negative_rmse_raises_clearly(self, tmp_path):
        transform = from_origin(500000, 4000000, 10.0, 10.0)
        path = _write_raster(tmp_path / "dem.tif", np.zeros((10, 10)), transform)
        with pytest.raises(ValueError, match="RMSE"):
            dem_differencing_pipeline(
                path,
                path,
                output_dir=tmp_path,
                dem_old_vertical_rmse_m=-1.0,
            )
