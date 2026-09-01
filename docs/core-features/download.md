# Downloading Satellite Data

Parallel, resumable downloads with real-time progress, retry logic,
band selection, and a post-processing chain.

## CLI examples

```bash
# Basic — download 3 scenes from search results
pygeofetch download run \
  --from-search results.geojson \
  --output ./data/ \
  --parallel 2 \
  --max-items 3

# RGB bands only — ~150 MB vs 600 MB for a full Sentinel-2 scene
pygeofetch download run \
  --from-search results.geojson \
  --output ./data/ \
  --bands "B02,B03,B04" \
  --max-items 5

# Full options — checksum, resume, bandwidth throttle, Slack notify
pygeofetch download run \
  --from-search results.geojson \
  --output ./data/ \
  --parallel 4 \
  --retry 5 \
  --verify-checksum \
  --resume \
  --bandwidth-limit 10MB \
  --priority high \
  --on-failure skip \
  --notify webhook:https://hooks.slack.com/services/YOUR/WEBHOOK

# Post-processing chain
pygeofetch download run \
  --from-search results.geojson \
  --output ./processed/ \
  --post-process "unzip,reproject:EPSG:4326,compress:lzw,cog"

# NDVI workflow — download Red + NIR, compute NDVI, export COG
pygeofetch download run \
  --from-search results.geojson \
  --bands "B04,B08" \
  --post-process "reproject:EPSG:4326,ndvi,cog" \
  --output ./ndvi/
```

## Download flags

| Flag | Type | Description |
|---|---|---|
| `--from-search` | path | GeoJSON results file from `search run --output`. Required. |
| `--output` | path | Output directory, created automatically. Default `./pygeofetch_data`. |
| `--parallel` | int | Concurrent download workers. Default 2. |
| `--retry` | int | Max retry attempts with exponential backoff. Default 3. |
| `--retry-delay` | float | Base retry delay in seconds, doubles each attempt. Default 5.0. |
| `--verify-checksum` | flag | SHA256 verification after each download; auto-retries on mismatch. |
| `--resume` | flag | Resume interrupted downloads from the last byte received. On by default. |
| `--bandwidth-limit` | string | e.g. `10MB`, `500KB`. `0` = unlimited. |
| `--priority` | choice | `high`, `normal`, `low`. |
| `--bands` | list | Comma-separated band names, e.g. `B02,B03,B04`. Default: all assets. |
| `--post-process` | string | Comma-separated chain, e.g. `unzip,reproject:EPSG:4326,cog`. |
| `--on-failure` | choice | `skip`, `abort`, `retry`. |
| `--max-items` | int | Limit to first N items in the results file. |
| `--overwrite` | flag | Overwrite existing files. Default: skip existing. |
| `--notify` | string | `webhook:URL` or `email:ADDRESS`. Repeatable. |
| `--json` | flag | Output results summary as JSON. |
| `--validate-optical` | flag | Run optical preflight validation as a final gate before downloading. Rejected items are never attempted -- a `FAILED` result explaining why is returned in their place, at their original position, so output length always matches input. See {doc}`/core-features/optical-validation`. |
| `--optical-max-cloud-cover` | float | Override the max cloud cover threshold used by `--validate-optical` (default 20.0). |
| `--optical-min-coverage` | float | Override the min AOI coverage ratio used by `--validate-optical`, 0-1 (default 0.8). |
| `--optical-required-bands` | string | Comma-separated required bands for `--validate-optical` (default `B02,B03,B04,B08,SCL`). |

## Band selection for Sentinel-2

| Bands | Purpose | Resolution | Approx size/scene |
|---|---|---|---|
| `B02,B03,B04` | RGB | 10m | ~150 MB |
| `visual` | True colour composite (pre-rendered TCI) | 10m | ~200 MB |
| `B04,B08` | NDVI (Red + NIR) | 10m | ~100 MB |
| `B02,B03,B04,B08` | RGB + NIR | 10m | ~200 MB |
| `B11,B12` | SWIR (fire, burn scar, soil moisture) | 20m | ~50 MB |
| `SCL` | Scene Classification Layer (cloud mask) | 20m | ~20 MB |
| *(omit `--bands`)* | All data bands | 10/20/60m | ~600 MB |

## Post-processing actions

| Action | Syntax | Description | Requires |
|---|---|---|---|
| unzip | `unzip` | Extract ZIP/TAR archives | — |
| reproject | `reproject:EPSG:4326` | Reproject to target CRS | rasterio |
| compress | `compress:lzw` | GeoTIFF compression (lzw, deflate, zstd) | rasterio |
| cog | `cog` | Convert to Cloud Optimized GeoTIFF | rasterio |
| clip | `clip:file.geojson` | Clip raster to polygon boundary | rasterio |
| resample | `resample:30` | Resample to target resolution (metres) | rasterio |
| ndvi | `ndvi` | Compute NDVI from Red/NIR bands | rasterio |
| ndwi | `ndwi` | Compute NDWI water index | rasterio |
| atmospheric | `atmospheric:sen2cor` | Atmospheric correction | sen2cor |
| pan-sharpen | `pan-sharpen` | Pan-sharpen multispectral with panchromatic | rasterio |
| merge | `merge` | Mosaic overlapping scenes | rasterio |

## Python API

```python
from pathlib import Path
from pygeofetch import PyGeoFetch
from pygeofetch.models import DownloadOptions
from pygeofetch.models.download_task import PostProcessAction

pf = PyGeoFetch()
results = pf.search(...)

results_download = pf.download(
    results[:5],
    destination=Path("./data/"),
    options=DownloadOptions(
        parallel=4,
        retry_attempts=5,          # not "retry" — see corrected field name below
        verify_checksum=True,
        resume=True,
        bands=["B02", "B03", "B04"],
        # post_process items must be PostProcessAction objects, not plain
        # strings — DownloadOptions(post_process=["reproject:EPSG:4326"])
        # raises a pydantic ValidationError.
        post_process=[
            PostProcessAction(action="reproject", params={"value": "EPSG:4326"}),
            PostProcessAction(action="cog"),
        ],
        bandwidth_limit_mbps=10.0,  # float Mbps, not a "10MB" string
    ),
)

for r in results_download:
    if r.status == "completed":
        print(r.data_id, r.bytes_downloaded)
    else:
        print(r.data_id, "failed:", r.error)
```

```{note}
The CLI's `--retry`, `--bandwidth-limit "10MB"`, and `--post-process
"reproject:EPSG:4326"` string syntax is parsed and converted into the
correct typed `DownloadOptions` fields for you — it's only when
constructing `DownloadOptions` directly in Python that the field names
above (`retry_attempts`, `bandwidth_limit_mbps`, a list of
`PostProcessAction` objects) matter.
```

Full field reference: {doc}`/reference/python-api`.
