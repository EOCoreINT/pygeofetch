# Installation

## Basic install

```bash
pip install pygeofetch
```

This gives you search, authenticated download, caching, and the CLI.
Heavier processing features are gated behind extras so a basic install
stays light.

## Extras

| Extra | Installs | Needed for |
|---|---|---|
| `pygeofetch[geo]` | rasterio, GDAL bindings | Post-processing (reproject, COG, clip, compress), optical preflight validation |
| `pygeofetch[cloud]` | boto3, cloud SDKs | S3/GCS export destinations |
| `pygeofetch[insar]` | numpy, scipy, rasterio, snaphu-py | Core InSAR pipeline (SLC extraction → SBAS) |
| `pygeofetch[insar-full]` | + PyAPS | ERA5-based atmospheric correction |
| `pygeofetch[sar]` | native SAR backend deps | Despeckling, calibration, flood mapping |
| `pygeofetch[processor]` | numpy, rasterio | Spectral indices, Landsat extraction, time series |
| `pygeofetch[viz]` | matplotlib, leafmap, rioxarray, localtileserver | `Plotter`, `MapViewer` |
| `pygeofetch[dev]` | pytest, hypothesis, responses | Running the test suite |
| `pygeofetch[all]` | everything above | Full feature set |

Combine as needed:

```bash
pip install "pygeofetch[insar,viz]"
pip install "pygeofetch[all]"
```

## Development install

```bash
git clone https://github.com/EOCoreINT/pygeofetch.git
cd pygeofetch
pip install -e ".[dev,all]"
pytest tests/unit/ -v
```

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
