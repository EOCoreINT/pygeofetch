"""
Regression tests for a real, confirmed bug: reprojecting a GCP-only
georeferenced source (rasterio reports src.crs=None, src.transform=
identity for these -- common, real delivery format for Sentinel-1 GRD
and other raw SAR products) silently produced a "corrupted" output --
CRS tag correctly updated to the real target CRS, but the affine
transform left at pixel-scale (~1.0 per pixel) instead of real
metre-scale UTM values. calculate_default_transform() does not error on
src_crs=None, it silently treats pixel-space bounds as if they were
real coordinates.

Also regresses a real gap found in the original safety-net check
(_has_identity_transform): it required the garbage transform's origin
to be exactly (0, 0), but a real reproduction of the underlying bug
produced origin (0, 200) instead, from the Y-axis flip convention in
rasterio's warp math shifting it by the image height. A fixed-origin
check missed this real, observed variant entirely.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.transform import Affine

from pygeofetch.core.downloader import AdaptiveDownloader


def _make_gcp_only_source(path: Path, h: int = 200, w: int = 300) -> None:
    """A real, synthetic file matching how raw Sentinel-1 GRD is
    actually delivered: georeferenced via GCPs only, no direct
    CRS/transform."""
    data = (np.random.rand(h, w) * 1000).astype(np.float32)
    gcps = [
        GroundControlPoint(row=0, col=0, x=-0.30, y=5.70, z=0),
        GroundControlPoint(row=0, col=w - 1, x=-0.10, y=5.70, z=0),
        GroundControlPoint(row=h - 1, col=0, x=-0.30, y=5.50, z=0),
        GroundControlPoint(row=h - 1, col=w - 1, x=-0.10, y=5.50, z=0),
    ]
    with rasterio.open(
        path, "w", driver="GTiff", dtype="float32", count=1, width=w, height=h
    ) as dst:
        dst.write(data, 1)
        dst.gcps = (gcps, CRS.from_epsg(4326))


@pytest.fixture
def downloader():
    return AdaptiveDownloader.__new__(AdaptiveDownloader)


def test_gcp_only_source_reprojects_to_real_metre_scale_transform(downloader):
    """The real root-cause fix: GCP-only sources must produce a genuine,
    physically plausible metre-scale transform, not pixel-space garbage."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src_path = tmp / "s1_grd_gcp_source.tif"
        _make_gcp_only_source(src_path)
        target_path = tmp / "reprojected.tif"

        downloader._reproject_with_validation(src_path, target_path, "EPSG:32630")

        with rasterio.open(target_path) as check:
            assert check.crs is not None and check.crs.to_string() == "EPSG:32630"
            pixel_size = max(abs(check.transform.a), abs(check.transform.e))
            assert pixel_size > 5, (
                f"pixel size {pixel_size} is pixel-space garbage, not real "
                f"metre-scale UTM georeferencing"
            )


def test_no_georeferencing_at_all_raises_clearly_not_silently_corrupted(downloader):
    """A source with neither CRS, transform, nor GCPs has genuinely no
    georeferencing to reproject from -- must raise a clear error, not
    silently produce a corrupted file."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        h, w = 50, 50
        src_path = tmp / "no_georef.tif"
        with rasterio.open(
            src_path, "w", driver="GTiff", dtype="float32", count=1, width=w, height=h
        ) as dst:
            dst.write((np.random.rand(h, w) * 1000).astype(np.float32), 1)

        target_path = tmp / "should_not_exist.tif"
        with pytest.raises(RuntimeError):
            downloader._reproject_with_validation(src_path, target_path, "EPSG:32630")

        assert not target_path.exists(), "no corrupted/partial file should be left on disk"


def test_widened_safety_net_catches_the_real_confirmed_garbage_variant(downloader):
    """Real, confirmed bug in the ORIGINAL check: it required origin
    exactly (0, 0), but the actual garbage transform reproduced from
    this bug has origin (0, 200) due to the Y-flip convention. This
    must still be caught."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        garbage_path = tmp / "garbage.tif"
        h, w = 200, 300
        garbage_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 200.0)  # real, non-zero-origin case
        with rasterio.open(
            garbage_path, "w", driver="GTiff", dtype="float32", count=1,
            width=w, height=h, crs="EPSG:32630", transform=garbage_transform,
        ) as dst:
            dst.write((np.random.rand(h, w) * 1000).astype(np.float32), 1)

        assert downloader._has_identity_transform(garbage_path) is True


def test_widened_safety_net_has_no_false_positive_on_real_output(downloader):
    """The widened check must not flag genuinely correct, real-scale
    UTM output as corrupted."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        real_path = tmp / "real.tif"
        h, w = 200, 300
        real_transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0)  # real 10m UTM
        with rasterio.open(
            real_path, "w", driver="GTiff", dtype="float32", count=1,
            width=w, height=h, crs="EPSG:32630", transform=real_transform,
        ) as dst:
            dst.write((np.random.rand(h, w) * 1000).astype(np.float32), 1)

        assert downloader._has_identity_transform(real_path) is False
