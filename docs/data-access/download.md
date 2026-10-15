# Downloading satellite data

Parallel, resumable downloads with retry, checksum verification, band selection and an
optional post-processing chain. To avoid downloading whole scenes at all, see
[Streaming](streaming.md).

## Quick examples

```bash
# Download the first three scenes from a saved search
pygeofetch download run --from-search results.geojson --output ./data/ --max-items 3

# Only the bands you need (about 150 MB instead of 600 MB for a Sentinel-2 scene)
pygeofetch download run --from-search results.geojson --output ./data/ \
  --bands "B02,B03,B04" --max-items 5

# Checksums, retries, bandwidth limit and a failure notification
pygeofetch download run --from-search results.geojson --output ./data/ \
  --parallel 4 --retry 5 --verify-checksum --bandwidth-limit 10MB \
  --on-failure skip --notify webhook:https://hooks.slack.com/services/YOUR/WEBHOOK

# Extract archives and convert to Cloud-Optimized GeoTIFF as part of the download
pygeofetch download run --from-search results.geojson --output ./processed/ \
  --post-process "unzip,reproject:EPSG:4326,compress:lzw,cog"
```

The full list of options is in the [command reference](../reference/cli-commands.md#download-run).

## Where files are written

`download()` writes each scene into a **sub-folder named after its provider**, below the
destination you give it:

```text
./data/                       <- destination you passed
├── copernicus/
│   └── S2B_MSIL2A_20240115T....SAFE.zip
└── planetary_computer/
    ├── S2A_..._B04.tif
    └── S2A_..._B08.tif
```

So `client.download(results, destination="data/raw/sentinel2")` produces
`data/raw/sentinel2/<provider>/...`, never files directly in `sentinel2/`. Code that looks
for downloads should search recursively:

```python
from pathlib import Path

zips = sorted(Path("data/raw/sentinel2").rglob("*.zip"))
tifs = sorted(Path("data/raw/sentinel2").rglob("*.tif"))
```

The result objects carry the exact paths: `DownloadResult.output_path` and
`DownloadResult.output_paths`. Use those when you have them.

## Archives

Providers package data differently, and pygeofetch downloads what the provider serves:

| Provider | Delivered as |
|---|---|
| `copernicus` | One `.SAFE.zip` per Sentinel scene |
| `usgs` | One `.tar` per Landsat scene |
| `planetary_computer`, `element84` | One file per band |
| `opentopography` | One GeoTIFF clipped to your bounding box |

Archives are **not extracted automatically**. Add the `unzip` post-processing action, which
extracts `.zip` and `.tar` archives next to the archive file and replaces the entries in
`output_paths` with the extracted files:

```bash
pygeofetch download run --from-search results.geojson --output ./data/ --post-process unzip
```

```python
from pygeofetch.models import DownloadOptions
from pygeofetch.models.download_task import PostProcessAction

options = DownloadOptions(post_process=[PostProcessAction(action="unzip")])
```

:::{note}
`DownloadOptions.extract_archives` exists as a field but is not read by any code, so
setting it has no effect. Use the `unzip` action.
:::

## Resuming

`--resume` is on by default. It means **skip scenes that are already complete**, not
"continue a partial file":

- Before downloading an item, pygeofetch checks whether a file already exists at the
  destination, and validates it.
- A valid file is skipped and reported as complete.
- A missing, truncated or corrupt file is downloaded again **from the first byte**. HTTP
  range requests are not used.

This makes it safe to re-run the same command after an interruption without re-fetching
finished scenes. It does not save progress on a single large file that was cut off.

## Failure handling

Every item in the batch is always attempted. `--on-failure` controls what happens after:

| Value | Behaviour |
|---|---|
| `skip` (default) | Continue and report failures in the summary. |
| `abort` | After all items are attempted, exit with status 1 if any failed. Use it to fail a script or CI job. |
| `retry` | Same as `skip`. Per-item retries with exponential backoff already happen through `--retry N`. |

## Band selection

Choosing bands before download is the largest saving for providers that serve one file
per band. Sentinel-2 examples:

| Bands | Purpose | Resolution | Approx. size per scene |
|---|---|---|---|
| `B02,B03,B04` | RGB | 10 m | 150 MB |
| `visual` | True-colour composite | 10 m | 200 MB |
| `B04,B08` | NDVI inputs | 10 m | 100 MB |
| `B02,B03,B04,B08` | RGB + NIR | 10 m | 200 MB |
| `B11,B12` | SWIR (burn scars, soil moisture, mineral indices) | 20 m | 50 MB |
| `SCL` | Scene classification (cloud mask) | 20 m | 20 MB |
| *(no `--bands`)* | All data bands | 10/20/60 m | 600 MB |

Copernicus delivers a zip, so band selection does not reduce what is transferred there.
If a band name matches nothing, `download()` logs a warning and downloads **all** assets.
[Streaming](streaming.md) treats the same situation as an error.

## Post-processing actions

Actions run in order after each file is downloaded. Syntax is `name` or `name:argument`.

| Action | Syntax | Description |
|---|---|---|
| `unzip` | `unzip` | Extract ZIP and TAR archives. |
| `reproject` | `reproject:EPSG:4326` | Reproject to a target CRS. |
| `compress` | `compress:lzw` | GeoTIFF compression (`lzw`, `deflate`, `zstd`). |
| `cog` | `cog` | Convert to Cloud-Optimized GeoTIFF. |
| `clip` | `clip:file.geojson` | Clip to a polygon. |
| `resample` | `resample:30` | Resample to a resolution in metres. |
| `merge` | `merge` | Mosaic overlapping scenes. |
| `ndvi`, `ndwi`, ... | `ndvi` | Compute a single-date index from the downloaded bands. |
| `cloud_mask` | `cloud_mask` | Mask clouds using the downloaded `SCL` band. Include `SCL` in `--bands`. |

Most actions need the `geo` extra (`rasterio`). An unknown action name raises an error
listing the supported ones. `dNBR` needs a before and an after scene, so it cannot run as a
single-scene action.

## Python

```python
from pathlib import Path
from pygeofetch import PyGeoFetch
from pygeofetch.models import DownloadOptions
from pygeofetch.models.download_task import PostProcessAction

client = PyGeoFetch()
results = client.search(...)

downloads = client.download(
    results[:5],
    destination=Path("./data/"),
    options=DownloadOptions(
        parallel=4,
        retry_attempts=5,
        verify_checksum=True,
        resume=True,
        bands=["B02", "B03", "B04"],
        bandwidth_limit_mbps=10.0,
        post_process=[
            PostProcessAction(action="reproject", params={"value": "EPSG:4326"}),
            PostProcessAction(action="cog"),
        ],
    ),
)

for r in downloads:
    if r.status == "completed":
        print(r.data_id, r.output_paths)
    else:
        print(r.data_id, "failed:", r.error)
```

The CLI converts its string syntax (`--retry`, `--bandwidth-limit 10MB`,
`--post-process "reproject:EPSG:4326"`) into these typed fields. Constructing
`DownloadOptions` directly requires the typed forms: `retry_attempts`,
`bandwidth_limit_mbps` as a float in Mbps, and `PostProcessAction` objects.

The returned list always has the same length and order as the input, including items
rejected by optical validation, which come back as failed results. See
[`DownloadResult`](../reference/python-api.md#downloadresult) for every field.

## Optical validation as a gate

`--validate-optical` runs the preflight checks (cloud cover, AOI coverage, required bands,
processing level) before any bytes are transferred. Rejected items are not attempted. See
[Optical validation](optical-validation.md).
