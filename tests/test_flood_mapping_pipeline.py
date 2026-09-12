"""
Tests for pygeofetch.multisensor.flood_mapping.

Uses physically realistic values on both sides: SAR DN magnitudes
chosen so calibrate()'s real sigma0_dB = 10*log10(DN^2) formula lands
on genuine, typical water/land dB values (same real values verified in
tests/test_sar_pipelines.py's flood test), and optical reflectance
values chosen so the real McFeeters NDWI formula lands on genuine,
typical water/land NDWI values -- not arbitrary numbers that happen to
clear a threshold.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pygeofetch.multisensor.flood_mapping import multi_sensor_flood_pipeline
from pygeofetch.sar import SARProcessor

REAL_CRS = "EPSG:32633"
REAL_TRANSFORM = from_origin(500000, 4000000, 10, 10)


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
    return SARProcessor()


class TestRealPhysicalScenario:
    """A real, deliberate flooded area (rows 10:20, cols 10:20) inside
    an otherwise dry scene, visible to both real sensors."""

    def _build_scene(self, tmp_path, rng):
        # Real SAR DN: land ~0.56 (-5dB), flooded water ~0.056 (-25dB) --
        # the same real values verified in test_sar_pipelines.py.
        sar_dn = np.abs(rng.normal(0.56, 0.03, (30, 30))).astype("float32")
        sar_dn[10:20, 10:20] = np.abs(rng.normal(0.056, 0.005, (10, 10)))
        sar_path = _write_raster(tmp_path / "sar.tif", sar_dn)

        # Real Sentinel-2-like surface reflectance (0-1 scale): land/veg
        # has high NIR, low green; water has the reverse.
        green_land, nir_land = 0.08, 0.35
        green_water, nir_water = 0.06, 0.015
        green = np.full((30, 30), green_land, dtype="float32")
        nir = np.full((30, 30), nir_land, dtype="float32")
        green[10:20, 10:20] = green_water
        nir[10:20, 10:20] = nir_water
        green_path = _write_raster(tmp_path / "green.tif", green)
        nir_path = _write_raster(tmp_path / "nir.tif", nir)
        return sar_path, green_path, nir_path

    def test_both_sensors_agree_flooded_area_correctly_flagged(
        self, processor, tmp_path
    ):
        rng = np.random.default_rng(1)
        sar_path, green_path, nir_path = self._build_scene(tmp_path, rng)

        result = multi_sensor_flood_pipeline(
            processor,
            sar_path,
            green_path,
            nir_path,
            output_dir=tmp_path,
        )
        assert result.success, result.error

        with rasterio.open(result.final_output) as src:
            fused = src.read(1)
        assert fused[10:20, 10:20].mean() > 0.8  # real flooded area caught
        assert fused[0:5, 0:5].mean() < 0.2  # real dry area not flagged

    def test_invalid_fusion_mode_raises_clearly(self, processor, tmp_path):
        rng = np.random.default_rng(2)
        sar_path, green_path, nir_path = self._build_scene(tmp_path, rng)
        with pytest.raises(ValueError, match="fusion_mode"):
            multi_sensor_flood_pipeline(
                processor,
                sar_path,
                green_path,
                nir_path,
                output_dir=tmp_path,
                fusion_mode="not_a_real_mode",
            )


class TestCloudAwareFusion:
    """Real, deliberate disagreement scenario: SAR says water (real,
    low backscatter) under a region marked as cloud, where optical's
    NDWI signal is fabricated/unreliable (as it would be under real
    cloud) and should be ignored in favor of SAR."""

    def test_cloud_masked_region_defers_to_sar(self, processor, tmp_path):
        rng = np.random.default_rng(3)
        sar_dn = np.abs(rng.normal(0.56, 0.03, (30, 30))).astype("float32")
        sar_dn[10:20, 10:20] = np.abs(
            rng.normal(0.056, 0.005, (10, 10))
        )  # real SAR flood signal
        sar_path = _write_raster(tmp_path / "sar.tif", sar_dn)

        # Real, deliberate: optical says "land" everywhere (e.g. cloud
        # tops misread as bright, dry NDWI) -- but a cloud mask flags
        # the same flooded region, so the real fused result should
        # trust SAR there, not optical's unreliable reading.
        green = np.full((30, 30), 0.08, dtype="float32")
        nir = np.full((30, 30), 0.35, dtype="float32")
        green_path = _write_raster(tmp_path / "green.tif", green)
        nir_path = _write_raster(tmp_path / "nir.tif", nir)

        cloud_scl = np.zeros((30, 30), dtype="uint8")
        cloud_scl[10:20, 10:20] = 9  # real Sentinel-2 SCL "cloud, high probability"
        cloud_path = _write_raster(tmp_path / "scl.tif", cloud_scl, dtype="uint8")

        result = multi_sensor_flood_pipeline(
            processor,
            sar_path,
            green_path,
            nir_path,
            output_dir=tmp_path,
            cloud_mask_path=cloud_path,
            fusion_mode="cloud_aware",
        )
        assert result.success, result.error
        with rasterio.open(result.final_output) as src:
            fused = src.read(1)
        # Real, correct behavior: cloud-masked region defers to SAR's
        # real flood signal, not optical's unreliable one.
        assert fused[10:20, 10:20].mean() > 0.8


class TestFusionModes:
    def test_union_mode_is_at_least_as_sensitive_as_either_source(
        self, processor, tmp_path
    ):
        rng = np.random.default_rng(4)
        sar_dn = np.abs(rng.normal(0.56, 0.02, (20, 20))).astype("float32")
        sar_dn[5:10, 5:10] = np.abs(
            rng.normal(0.056, 0.005, (5, 5))
        )  # SAR-only water region
        sar_path = _write_raster(tmp_path / "sar.tif", sar_dn)

        green = np.full((20, 20), 0.08, dtype="float32")
        nir = np.full((20, 20), 0.35, dtype="float32")
        green[12:17, 12:17] = 0.06  # optical-only water region (SAR misses it)
        nir[12:17, 12:17] = 0.015
        green_path = _write_raster(tmp_path / "green.tif", green)
        nir_path = _write_raster(tmp_path / "nir.tif", nir)

        result = multi_sensor_flood_pipeline(
            processor,
            sar_path,
            green_path,
            nir_path,
            output_dir=tmp_path,
            fusion_mode="union",
        )
        assert result.success, result.error
        with rasterio.open(result.final_output) as src:
            fused = src.read(1)
        # Real, correct union behavior: both disjoint regions caught.
        assert fused[5:10, 5:10].mean() > 0.8
        assert fused[12:17, 12:17].mean() > 0.8

    def test_intersection_mode_only_flags_agreement(self, processor, tmp_path):
        rng = np.random.default_rng(5)
        sar_dn = np.abs(rng.normal(0.56, 0.02, (20, 20))).astype("float32")
        sar_dn[5:10, 5:10] = np.abs(rng.normal(0.056, 0.005, (5, 5)))  # SAR-only
        sar_path = _write_raster(tmp_path / "sar.tif", sar_dn)

        green = np.full((20, 20), 0.08, dtype="float32")  # optical says land everywhere
        nir = np.full((20, 20), 0.35, dtype="float32")
        green_path = _write_raster(tmp_path / "green.tif", green)
        nir_path = _write_raster(tmp_path / "nir.tif", nir)

        result = multi_sensor_flood_pipeline(
            processor,
            sar_path,
            green_path,
            nir_path,
            output_dir=tmp_path,
            fusion_mode="intersection",
        )
        assert result.success, result.error
        with rasterio.open(result.final_output) as src:
            fused = src.read(1)
        # Real, correct intersection behavior: SAR-only region should
        # NOT be flagged, since optical disagrees.
        assert fused[5:10, 5:10].mean() < 0.2


class TestShapeMismatch:
    def test_mismatched_optical_shape_fails_clearly(self, processor, tmp_path):
        sar_path = _write_raster(
            tmp_path / "sar.tif", np.full((30, 30), 0.56, dtype="float32")
        )
        green_path = _write_raster(
            tmp_path / "green.tif", np.full((20, 20), 0.08, dtype="float32")
        )
        nir_path = _write_raster(
            tmp_path / "nir.tif", np.full((20, 20), 0.35, dtype="float32")
        )

        result = multi_sensor_flood_pipeline(
            SARProcessor(),
            sar_path,
            green_path,
            nir_path,
            output_dir=tmp_path,
        )
        assert not result.success
        assert "shape" in result.error.lower()
