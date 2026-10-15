# Quick Start (5 Minutes)

The full path from nothing installed to a file on disk, using a
free, no-account provider — no login, no API key, no waiting on an
approval email.

## What you will do

This walk-through takes you from an empty environment to a scene on disk in four short steps, using a provider that needs no account so that nothing stands between you and a first result. You install the package, run a search that writes its findings to a GeoJSON file, download the best-ranked scene from that file, and check the result. Afterwards the same workflow is repeated in Python so you can see how the command line and the API map onto each other: every command in the command-line interface has a counterpart on the `PyGeoFetch` client, and they share the same query and download options.

Two ideas are worth holding on to as you read. First, search and download are separate steps. A search only returns descriptions of scenes (identifier, date, footprint, cloud cover and the list of files), which is why it is fast and why you can run it as often as you like before deciding what to download. Second, the GeoJSON results file is the hand-off between the two steps, so you can search today, review the file, and download later or on another machine. Once this works, move on to [Searching](../data-access/search.md) to learn the filters, and to [Authentication](../data-access/authentication.md) when you want providers that require a login.

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

A Sentinel-2 scene, on disk, ready to open in QGIS or load with
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
 footprints and hover for scene details. See
 [MapViewer](../visualization/mapviewer.md).
- **Compute something** — NDVI, water indices, burn severity, and more,
 one line each. See [Spectral Indices](../processing/optical/spectral-indices.md).
- **Skip cloudy/off-target scenes automatically** — a configurable
 pre-download quality gate for optical imagery. See
 [Optical Data Validation & Preflight](../data-access/optical-validation.md).
- **Working with SAR/Sentinel-1?** Despeckling, calibration, and the
 full InSAR chain are different from optical workflows. See
 [SAR Processing](../processing/sar-insar/sar.md) or [InSAR Processing](../processing/sar-insar/insar.md).
- **Doing this on a schedule?** Turn a one-off search into a repeatable
 pipeline. See [Pipelines & Batch Processing](../workflows/pipelines.md).
- **Something went wrong?** Auth errors, timeouts, zero results,
 keyring issues on Docker/SSH. See [Error Handling & Resilience](../reference/error-handling.md).
