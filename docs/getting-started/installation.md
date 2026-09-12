# Installation

## Basic install

```bash
pip install pygeofetch
```

This gives you search, authenticated download, caching, and the CLI.
Heavier processing features are gated behind extras so a basic install
stays light.

## Extras

Checked directly against the real, current `pyproject.toml` — not
assumed from an earlier pass of this page, which listed a `dev` extra
containing packages (`hypothesis`, `responses`) that aren't actually
in it, and a test command (`pytest tests/unit/`) that doesn't match
the real, flat `tests/` directory this project actually uses.

| Extra | Installs | Needed for |
|---|---|---|
| `pygeofetch[geo]` | rasterio, geopandas, pyarrow, shapely, pyproj | Post-processing (reproject, COG, clip, compress), optical preflight validation |
| `pygeofetch[processor]` | rasterio, rioxarray, xarray, eoreader, spyndex, scipy, scikit-image, geopandas | Spectral indices (including the 6 real geology/mineral-exploration indices — see [Spectral Indices](../processing/spectral-indices.md)), Landsat extraction, time series |
| `pygeofetch[sar]` | sarxarray | The lightweight `sarxarray` backend for `pygeofetch.sar.SARProcessor` — not required for `client.sar` or the native backend, both of which need nothing extra |
| `pygeofetch[insar]` | scipy, shapely, xarray | Core InSAR pipeline (SLC extraction → SBAS) and the [advanced safeguards](../processing/insar.md#advanced-safeguards-custom-dems-layovershadow-and-topographic-residuals) (layover/shadow masking, custom DEM alignment, SBAS topographic residual estimation) |
| `pygeofetch[insar-full]` | + snaphu | Real SNAPHU-based phase unwrapping (falls back to a CLI executable if the Python package isn't found) |
| `pygeofetch[ost]` | opensartoolkit | The `"ost"` backend for `pygeofetch.sar.SARProcessor` — production-grade Range-Doppler terrain correction, requires a working SNAP install separately |
| `pygeofetch[cloud]` | boto3, pystac, planetary-computer | S3/GCS export destinations, and required for running the real test suite (see below) |
| `pygeofetch[viz]` | matplotlib, leafmap, folium, plotly, and others | `Plotter`, `MapViewer` |
| `pygeofetch[viz-3d]` | keplergl | 3D Kepler.gl maps — kept separate from `viz` since `keplergl`'s own packaging doesn't build cleanly everywhere (no wheel, a hardcoded pyarrow build dependency); install explicitly only if your platform can build it |
| `pygeofetch[notebook]` | ipython, ipywidgets, jupyter | Notebook-based workflows |
| `pygeofetch[schedule]` | croniter | Cron-based pipeline scheduling |
| `pygeofetch[dev]` | pytest, pytest-cov, pytest-mock, ruff, black, mypy, build, twine | Linting, formatting, type checking, and building the package — not, by itself, everything needed to *run* the test suite (see `test` below) |
| `pygeofetch[test]` | `dev` + `geo` + `insar` + `cloud` | Everything required to actually run `pytest tests/` and collect every module without import errors. This is a real, complete list — confirmed by actually running the full suite with `cloud` left out and getting real `ModuleNotFoundError`s from tests that hard-import `boto3`/`botocore` directly, not a hypothetical gap. |
| `pygeofetch[full]` (alias: `[all]`) | `geo` + `processor` + `insar` + `viz` + `cloud` + `notebook` + `schedule` | Everything that reliably installs across platforms. `viz-3d` and `sar` (the `sarxarray` backend) are deliberately excluded — `sarxarray` currently has no release supporting Python 3.9 — add either explicitly if you need it. |

Combine as needed:

```bash
pip install "pygeofetch[insar,viz]"
pip install "pygeofetch[full]"
```

## Development install

```bash
git clone https://github.com/EOCoreINT/pygeofetch.git
cd pygeofetch
pip install -e ".[test]"
pytest tests/ -v
```

`pytest tests/` — not `tests/unit/`, which doesn't exist in this
project's real layout. `pip install -e ".[test]"` is also the real,
correct extra for this: `[dev,all]` doesn't include `boto3`, so several
real tests (anything touching S3 export or the `noaa_big_data`
provider) would fail with `ModuleNotFoundError` rather than run.

If you're working on linting/formatting/type-checking only, without
running the full suite, `pip install -e ".[dev]"` alone is enough and
lighter.

## Verifying the install

```bash
pygeofetch --version
pygeofetch doctor
```

`doctor` checks the Python version, which optional dependency groups
are actually importable, keyring availability, and live connectivity
to a couple of open (no-auth) providers — the fastest way to confirm
a fresh install is actually working end to end, not just that the
package imported.
