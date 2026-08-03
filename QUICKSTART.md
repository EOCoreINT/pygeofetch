# PyGeoFetch — Local Setup & Testing Guide

## 1. Prerequisites

- Python 3.9 or later (`python3 --version`)
- pip (`pip --version`)
- git (for cloning)

---

## 2. Install

### Option A — From the downloaded archive

```bash
# Extract the archive
tar -xzf pygeofetch-1.9.4.tar.gz
cd pygeofetch

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install in editable (development) mode
pip install -e .

# Verify installation
pygeofetch --version
pygeofetch doctor
```

### Option B — With optional extras

```bash
# Geo processing (rasterio, geopandas, pyarrow)
pip install -e ".[geo]"

# Cloud access (boto3 for NASA S3, pystac)
pip install -e ".[cloud]"

# Cron scheduling
pip install -e ".[schedule]"

# Everything including dev tools
pip install -e ".[all]"
```

---

## 3. Verify Installation

```bash
# Check everything is wired up
pygeofetch doctor

# Show system status
pygeofetch status

# List all 22 providers
pygeofetch providers list

# Provider details
pygeofetch providers info aws_earth
pygeofetch providers info planetary_computer
```

---

## 4. Try Free Providers (No Login Required)

These providers work immediately with no credentials:

```bash
# Search AWS Earth Open Data (Sentinel-2)
pygeofetch search run \
    --bbox "-74.1,40.6,-73.7,40.9" \
    --start-date 2024-01-01 \
    --end-date 2024-03-01 \
    --cloud-cover 0-20 \
    --providers aws_earth \
    --format table

# Search Microsoft Planetary Computer (free, no login)
pygeofetch search run \
    --bbox "-74.1,40.6,-73.7,40.9" \
    --start-date 2024-01-01 \
    --providers planetary_computer \
    --satellites Sentinel-2 \
    --cloud-cover 0-10 \
    --output results.geojson

# Search Element 84 (free, COG data)
pygeofetch search run \
    --bbox "-74.1,40.6,-73.7,40.9" \
    --providers element84 \
    --cloud-cover 0-15 \
    --format json

# Search all free providers at once
pygeofetch search run \
    --bbox "-74.1,40.6,-73.7,40.9" \
    --providers aws_earth,planetary_computer,element84 \
    --cloud-cover 0-20 \
    --max-results 50 \
    --output results.geojson \
    --format table
```

---

## 5. Add Provider Credentials

### USGS Earth Explorer (free account)
Register at: https://ers.cr.usgs.gov/register

```bash
pygeofetch auth add usgs --username YOUR_USERNAME --password YOUR_PASSWORD
pygeofetch auth test usgs
```

### Copernicus CDSE (free account)
Register at: https://dataspace.copernicus.eu/

```bash
pygeofetch auth add copernicus \
    --username YOUR_EMAIL \
    --password YOUR_PASSWORD
pygeofetch auth test copernicus
```

### Planet Labs (API key)
```bash
pygeofetch auth add planet --api-key YOUR_PLANET_API_KEY
```

### NASA Earthdata (free account)
Register at: https://urs.earthdata.nasa.gov/

```bash
pygeofetch auth add nasa_earthdata \
    --username YOUR_USERNAME \
    --password YOUR_PASSWORD
pygeofetch auth add nasa_earthdata_cloud \
    --username YOUR_USERNAME \
    --password YOUR_PASSWORD
```

### Sentinel Hub (client credentials)
Register at: https://apps.sentinel-hub.com/

```bash
pygeofetch auth add sentinel_hub \
    --client-id YOUR_CLIENT_ID \
    --client-secret YOUR_CLIENT_SECRET
```

### OpenTopography (API key)
Register at: https://portal.opentopography.org/

```bash
pygeofetch auth add opentopography --api-key YOUR_KEY
```

List all configured credentials:
```bash
pygeofetch auth list
```

---

## 6. Search Examples

```bash
# Multi-provider search with filters
pygeofetch search run \
    --bbox "-74.1,40.6,-73.7,40.9" \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --cloud-cover 0-10 \
    --providers usgs,copernicus,aws_earth \
    --satellites Landsat-8,Sentinel-2 \
    --max-results 100 \
    --sort-by cloud_cover \
    --sort-order asc \
    --output results.geojson

# Search from a GeoJSON AOI file
pygeofetch search run \
    --geometry-file my_area.geojson \
    --cloud-cover 0-5 \
    --providers planetary_computer \
    --format stac \
    --output results_stac.json

# CQL2 advanced filter
pygeofetch search run \
    --bbox "-74.1,40.6,-73.7,40.9" \
    --providers planetary_computer \
    --cql2 "eo:cloud_cover < 5 AND platform = 'sentinel-2b'" \
    --format table

# Output as CSV
pygeofetch search run \
    --bbox "-10,35,10,55" \
    --providers aws_earth \
    --format csv \
    --output results.csv

# Output scene IDs only
pygeofetch search run \
    --bbox "-74.1,40.6,-73.7,40.9" \
    --providers aws_earth \
    --format ids
```

---

## 7. Download Examples

```bash
# Basic download from search results
pygeofetch download run \
    --from-search results.geojson \
    --output ./data/

# Download with all options
pygeofetch download run \
    --from-search results.geojson \
    --output ./data/ \
    --parallel 4 \
    --retry 5 \
    --verify-checksum \
    --resume \
    --max-items 10 \
    --on-failure skip

# With post-processing chain
pygeofetch download run \
    --from-search results.geojson \
    --output ./data/ \
    --parallel 2 \
    --post-process "unzip,reproject:EPSG:4326,compress:lzw"

# With Slack webhook notification
pygeofetch download run \
    --from-search results.geojson \
    --output ./data/ \
    --notify webhook:https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK

# Show downloaded files
pygeofetch download status ./data/
```

---

## 8. Pipeline Examples

Create `pipeline.yaml`:
```yaml
name: weekly-sentinel2
schedule: "0 6 * * 1"
description: Weekly Sentinel-2 for my AOI

steps:
  - search:
      providers: [aws_earth, planetary_computer]
      date_range: last_7_days
      cloud_cover: 0-10
      bbox: "-74.1,40.6,-73.7,40.9"
      max_results: 10

  - download:
      parallel: 2
      output: ./weekly_data/
```

```bash
# Validate
pygeofetch pipeline validate pipeline.yaml

# Run once
pygeofetch pipeline run pipeline.yaml

# Schedule for weekly recurrence
pygeofetch pipeline schedule pipeline.yaml

# List scheduled
pygeofetch pipeline list-scheduled
```

---

## 9. Cache Management

```bash
pygeofetch cache stats
pygeofetch cache location
pygeofetch cache ttl show
pygeofetch cache ttl set 7200      # 2 hours
pygeofetch cache clear --dry-run
pygeofetch cache clear --older-than 7d
pygeofetch cache prune --max-size 500MB
```

---

## 10. Configuration

```bash
# View full merged config
pygeofetch config show

# Get a value
pygeofetch config get download.parallel

# Set a value
pygeofetch config set download.parallel 8
pygeofetch config set cache.ttl_seconds 7200
pygeofetch config set search.on_provider_failure skip

# Show config file paths
pygeofetch config path
```

Or edit `~/.pygeofetch/config.yaml` directly:
```yaml
download:
  parallel: 4
  verify_checksum: false
  resume: true
cache:
  ttl_seconds: 3600
search:
  on_provider_failure: skip
```

---

## 11. Python API

```python
from pathlib import Path
from pygeofetch import PyGeoFetch
from pygeofetch.models.search_query import SearchQuery, BoundingBox
from pygeofetch.models.download_task import DownloadOptions

# Initialize
sb = PyGeoFetch(log_level="INFO")

# Add credentials
sb.add_credentials("usgs", username="user", password="pass")
sb.add_credentials("planet", api_key="PL_KEY")

# Search
results = sb.search(
    SearchQuery(
        bbox=BoundingBox.from_string("-74.1,40.6,-73.7,40.9"),
        start_date="2024-01-01",
        end_date="2024-06-01",
        cloud_cover_max=20,
        max_results=50,
    ),
    providers=["usgs", "aws_earth", "planetary_computer"],
)

print(f"Found {len(results)} scenes")
for r in results[:3]:
    print(f"  {r.id} | {r.provider} | cloud={r.cloud_cover}%")

# Save results
sb.searcher.save_results(results, Path("results.geojson"))

# Download
dl_results = sb.download(
    results[:5],
    destination=Path("./data/"),
    options=DownloadOptions(
        parallel=2,
        retry_attempts=3,
        verify_checksum=False,
        resume=True,
        on_failure="skip",
    ),
)

for dr in dl_results:
    status = "✓" if dr.success else "✗"
    size = f"{dr.bytes_downloaded // 1024 // 1024:.1f} MB" if dr.bytes_downloaded else "0 MB"
    print(f"  {status} {dr.data_id} — {size}")
```

---

## 12. Run Tests

```bash
# All unit tests (fast, no network needed)
pytest tests/unit/ -v

# With coverage report
pytest tests/unit/ -v --cov=pygeofetch --cov-report=html
open htmlcov/index.html      # macOS
xdg-open htmlcov/index.html  # Linux

# Specific test file
pytest tests/unit/test_models.py -v
pytest tests/unit/test_providers.py -v
pytest tests/unit/test_utils.py -v

# Integration tests (requires real credentials + network)
pytest tests/integration/ -v -m integration

# Single test by name
pytest tests/unit/test_models.py::TestBoundingBox::test_from_string -v
```

---

## 13. Common Issues

### `ModuleNotFoundError: No module named 'pygeofetch'`
You're not in the venv or forgot `pip install -e .`:
```bash
source .venv/bin/activate
pip install -e .
```

### `pygeofetch: command not found`
The venv isn't activated, or pip installed to a location not on PATH:
```bash
source .venv/bin/activate
# or run directly:
python3 -m pygeofetch.cli.main --version
```

### `keyring.errors.NoKeyringError`
No system keyring available (common on headless Linux). Set env var instead:
```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
pygeofetch auth add usgs --username user --password pass
```
Or use environment variables for credentials:
```bash
export PYGEOFETCH_USGS_USERNAME=myuser
export PYGEOFETCH_USGS_PASSWORD=mypass
```

### Search returns 0 results
- Check your bbox format: `"minlon,minlat,maxlon,maxlat"` (longitude first)
- Widen the date range or raise cloud-cover limit
- Run `pygeofetch doctor` to verify network connectivity

### Provider auth fails
```bash
pygeofetch auth test usgs   # test stored credentials
pygeofetch auth remove usgs  # remove and re-add
pygeofetch auth add usgs --username USER --password PASS
```
