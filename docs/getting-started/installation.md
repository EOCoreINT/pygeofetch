# Installation

pygeofetch requires **Python 3.9 or newer**. The package metadata lists Python 3.9 to 3.12 as
tested; newer versions may work but are not declared.

| Python | Notes |
|---|---|
| 3.9 | Supported. `eval_type_backport` is installed automatically so Pydantic can evaluate `str | None` annotations. `sarxarray` has no release for 3.9, so `pygeofetch[sar]` is unavailable. |
| 3.10 to 3.12 | Supported and declared. |
| 3.13 and later | Not declared. Some optional dependencies (for example `keplergl`) do not build there. |

## Basic install

```bash
pip install pygeofetch
```

The core install adds `httpx`, `click`, `pydantic`, `pydantic-settings`, `PyYAML`,
`cryptography`, `keyring`, `tenacity`, `python-dateutil`, `requests`, `anyio`, `numpy` and
`tqdm`. It provides search, authenticated download, caching and the command-line interface.
Processing features are optional extras, so a basic install stays light. A virtual environment
is recommended:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install pygeofetch
```

:::{note}
The command-line interface imports `rich`, and optical validation imports `shapely`. Neither is
declared in the `pyproject.toml` core dependencies (the conda recipe in `meta.yaml` does list
`rich`, so a conda install is unaffected). If `pygeofetch --help` fails with
`ModuleNotFoundError: No module named 'rich'`, or an import of `shapely` fails, run
`pip install rich shapely`, or use the `geo` extra, which includes `shapely`. See [Known issues](../reference/known-issues.md#undeclared-dependencies).
:::

## Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `pygeofetch[geo]` | rasterio, geopandas, pyarrow, shapely, pyproj | Raster post-processing (reproject, COG, clip, compress), optical validation, and [streaming](../data-access/streaming.md) |
| `pygeofetch[processor]` | rasterio, rioxarray, xarray, eoreader, spyndex, scipy, scikit-image, geopandas | [Spectral indices](../processing/optical/spectral-indices.md) (including the 6 geology indices), Landsat extraction, time series |
| `pygeofetch[sar]` | sarxarray | Intended for the `sarxarray` backend of `SARProcessor`. That backend is not currently reachable; see [Known issues](../reference/known-issues.md#sar-backends-are-not-wired-in). |
| `pygeofetch[insar]` | scipy, shapely, xarray | The [InSAR](../processing/sar-insar/insar.md) pipeline (SLC extraction to SBAS) and its advanced safeguards |
| `pygeofetch[insar-full]` | snaphu, scipy (add `insar` as well) | SNAPHU phase unwrapping (falls back to a `snaphu` executable on your `PATH`) |
| `pygeofetch[ost]` | opensartoolkit | Intended for the `"ost"` backend (Range-Doppler terrain correction, needs SNAP). Not currently reachable; see [Known issues](../reference/known-issues.md#sar-backends-are-not-wired-in). |
| `pygeofetch[cloud]` | boto3, pystac, planetary-computer | S3 and GCS export, and the test suite |
| `pygeofetch[viz]` | matplotlib, leafmap, folium, plotly and others | `Plotter` and `MapViewer` |
| `pygeofetch[viz-3d]` | keplergl | 3D Kepler.gl maps. Separate because `keplergl` does not build on every platform. |
| `pygeofetch[notebook]` | ipython, ipywidgets, jupyter | Notebook workflows |
| `pygeofetch[schedule]` | croniter | Cron scheduling of pipelines |
| `pygeofetch[dev]` | pytest, pytest-cov, pytest-mock, ruff, black, mypy, build, twine | Linting, formatting, type checking, building |
| `pygeofetch[test]` | `dev` + `geo` + `insar` + `cloud` | Everything needed to run `pytest tests/` |
| `pygeofetch[full]` (alias `[all]`) | `geo` + `processor` + `insar` + `viz` + `cloud` + `notebook` + `schedule` | Everything that installs reliably across platforms. `viz-3d` and `sar` are excluded: `sarxarray` has no release that supports Python 3.9. |

Combine extras as needed:

```bash
pip install "pygeofetch[geo,insar]"
pip install "pygeofetch[full]"
```

### Which extra do I need?

| I want to... | Install |
|---|---|
| Search and download only | *(none)* |
| Clip, reproject or convert downloads | `geo` |
| Stream an area of interest | `geo` |
| Use the full spectral index catalogue | `processor` |
| Process Sentinel-1 SLC with InSAR | `insar,insar-full` (the second adds unwrapping) |
| Schedule pipelines with cron | `schedule` |
| Use the notebook examples | `notebook`, `viz` |
| Contribute or run the tests | `test` |
| Draw maps and plots | `viz` |

## Other install routes

**conda.** The repository ships a conda recipe (`meta.yaml`) for the core package. Install
geospatial libraries such as `rasterio` and `gdal` from `conda-forge` before the extras: they
are much easier to obtain as binaries there than to build with pip.

```bash
conda create -n pygeofetch -c conda-forge python=3.12 rasterio geopandas
conda activate pygeofetch
pip install "pygeofetch[processor,insar]"
```

**Docker.** See [Docker](../reference/docker.md).

**Upgrade.**

```bash
pip install --upgrade "pygeofetch[full]"
```

## Development install

```bash
git clone https://github.com/EOCoreINT/pygeofetch.git
cd pygeofetch
pip install -e ".[test]"
pytest tests/ -v
```

The test suite lives in a flat `tests/` directory. Use `[test]`, not `[dev,all]`: `[dev,all]`
does not include `boto3`, so tests touching S3 export or the `noaa_big_data` provider fail with
`ModuleNotFoundError`. For linting and type checking only, `pip install -e ".[dev]"` is lighter.

## Install problems

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: rich` on `pygeofetch --help` | `pip install rich`. See the note above. |
| `ModuleNotFoundError: boto3` in tests | Install `[test]`, which includes `cloud`. |
| `rasterio` fails to build | Use a wheel (`pip install --upgrade pip` first) or install from `conda-forge`. |
| `pip install "pygeofetch[viz]"` is slow or fails on one package | `viz` pulls about two dozen packages. Install only `matplotlib` for basic plots. |
| `keplergl` fails to build | It is isolated in `viz-3d`; skip it unless you need Kepler.gl maps. |
| Keyring errors over SSH or in Docker | There is no keyring daemon. See [Error handling](../reference/error-handling.md#common-problems). |
| SNAPHU not found when unwrapping | Install `insar-full`, or put a `snaphu` executable on `PATH`. |

## Verify the install

```bash
pygeofetch --version
pygeofetch doctor
```

`doctor` checks the Python version, the required packages, which optional dependency groups
are importable, the keyring backend, and connectivity to three open providers. See
[Diagnostics](diagnostics.md).
