"""Pins: terrain_derivatives aspect/hillshade conventions, and that the
multisensor package is actually importable (it shipped with `_init.py`
instead of `__init__.py`, so none of its pipelines could be imported)."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pygeofetch.processing.preprocessor import Preprocessor


def _dem(path, z, px=10.0):
    prof = dict(
        driver="GTiff",
        height=z.shape[0],
        width=z.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32630",
        transform=from_origin(600000, 700000, px, px),
        nodata=None,
    )
    with rasterio.open(path, "w", **prof) as d:
        d.write(z.astype("float32"), 1)
    return path


def _derive(tmp_path, z, **kw):
    res = Preprocessor().terrain_derivatives(
        _dem(tmp_path / "dem.tif", z), output_dir=str(tmp_path / "o"), **kw
    )
    out = {}
    for k in ("slope", "aspect", "hillshade"):
        with rasterio.open(res.metadata[f"{k}_path"]) as s:
            out[k] = s.read(1)
    return out


N = 31
X = np.arange(N)[None, :] * np.ones((N, 1))  # east
ROW = np.arange(N)[:, None] * np.ones((1, N))  # rows increase SOUTH


@pytest.mark.parametrize("bearing", [0, 45, 90, 135, 180, 225, 270, 315])
def test_aspect_is_the_downhill_compass_bearing(tmp_path, bearing):
    east, north = np.sin(np.radians(bearing)), np.cos(np.radians(bearing))
    z = 500 - 3.0 * (east * X + north * (-ROW))  # falls toward `bearing`
    a = _derive(tmp_path, z)["aspect"][15, 15]
    assert abs(((a - bearing + 180) % 360) - 180) < 0.5


def test_hillshade_distinguishes_slopes_facing_toward_and_away_from_the_sun(tmp_path):
    """Sun in the east at 45 deg: an east-facing slope is lit (cos i = 0.832 for a
    11.3 deg slope), a west-facing one is shaded (0.555). Computed from
    cos(i) = cos(z)cos(s) + sin(z)sin(s)cos(az - aspect)."""
    s = np.arctan(0.2)
    cz = np.cos(np.radians(45))
    sz = np.sin(np.radians(45))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    lit = _derive(tmp_path / "a", 100 - 2 * X, azimuth=90.0, altitude=45.0)[
        "hillshade"
    ][15, 15]
    dark = _derive(tmp_path / "b", 100 + 2 * X, azimuth=90.0, altitude=45.0)[
        "hillshade"
    ][15, 15]
    assert lit == pytest.approx(cz * np.cos(s) + sz * np.sin(s), abs=1e-3)  # 0.832
    assert dark == pytest.approx(cz * np.cos(s) - sz * np.sin(s), abs=1e-3)  # 0.555
    assert lit > dark + 0.2


def test_multisensor_package_imports_and_cli_registers():
    import pygeofetch.multisensor as m

    assert (
        m.__file__ is not None
    ), "multisensor must be a regular package with __init__.py"
    for name in (
        "insar_optical_displacement_pipeline",
        "multi_sensor_flood_pipeline",
        "terrain_corrected_change_pipeline",
        "dem_differencing_pipeline",
        "vegetation_disturbance_pipeline",
        "solar_position",
    ):
        assert callable(getattr(m, name)), name
    from click.testing import CliRunner

    from pygeofetch.cli.main import cli

    r = CliRunner().invoke(cli, ["multisensor", "--help"])
    assert r.exit_code == 0 and "flood-map" in r.output
