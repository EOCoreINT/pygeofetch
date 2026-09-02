# Quick Start (5 Minutes)

The full path from nothing installed to a real file on disk, using a
free, no-account provider — no login, no API key, no waiting on an
approval email.

## 1. Install

```bash
pip install pygeofetch
```

## 2. Search

```bash
pygeofetch search run \
  --bbox "-74.1,40.6,-73.7,40.9" \
  --start-date 2024-01-01 \
  --end-date 2024-03-01 \
  --cloud-cover 0-10 \
  --providers aws_earth \
  --output results.geojson
```

This searches AWS Earth's open Sentinel-2 archive and writes a real
GeoJSON results file rather than just printing to the terminal.

## 3. Download

```bash
pygeofetch download run \
  --from-search results.geojson \
  --output ./data/ \
  --max-items 1
```

Reads that same file and downloads the top-ranked scene.

## 4. Verify

```bash
ls -la ./data/
```

A real Sentinel-2 scene, on disk, ready to open in QGIS or load with
rasterio.

## The same thing in Python

If you'd rather script this than use the CLI, here's the exact same
five minutes as Python:

```python
from pygeofetch import PyGeoFetch
from pygeofetch.models import SearchQuery

client = PyGeoFetch()

results = client.search(
    SearchQuery(
        bbox=(-74.1, 40.6, -73.7, 40.9),
        start_date="2024-01-01",
        end_date="2024-03-01",
        cloud_cover_max=10,
    ),
    providers=["aws_earth"],
)

print(f"Found {len(results)} scenes")

downloaded = client.download(results[:1], destination="./data")
print(downloaded[0].output_path)
```

No credentials, no `auth add` step needed — `aws_earth` is a real,
free, no-account provider, same as the CLI example above.

## What's next

- **See it on a map first** — before committing to a download, view
  real footprints and hover for scene details. See
  {doc}`/visualization/mapviewer`.
- **Compute something** — NDVI, water indices, burn severity, and more,
  one line each. See {doc}`/processing/spectral-indices`.
- **Skip cloudy/off-target scenes automatically** — a configurable
  pre-download quality gate for optical imagery. See
  {doc}`/core-features/optical-validation`.
- **Working with SAR/Sentinel-1?** Despeckling, calibration, and the
  full InSAR chain are genuinely different from optical workflows. See
  {doc}`/processing/sar` or {doc}`/processing/insar`.
- **Doing this on a schedule?** Turn a one-off search into a repeatable
  pipeline. See {doc}`/reference/pipelines`.
- **Something went wrong?** Auth errors, timeouts, zero results,
  keyring issues on Docker/SSH. See {doc}`/reference/error-handling`.
