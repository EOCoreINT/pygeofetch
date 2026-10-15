# Streaming: read only your area

`client.download()` fetches each selected asset as a complete file. For a Sentinel-2 band
that is 100-300 MB, and reading it with `rasterio` decodes the whole raster into memory.
If your study area is a few kilometres across inside a 110 km tile, almost all of that
transfer and memory is wasted.

**Streaming** reads only the pixels under your area of interest (AOI). pygeofetch opens
each band as a [Cloud-Optimized GeoTIFF](https://cogeo.org/) (COG) over HTTP, converts
your bounding box to a pixel window, and requests just the byte ranges for that window.

:::{admonition} Verification status
:class: warning

The streaming module is tested against a local HTTP server that implements range requests
and counts the bytes it serves, and against mocked provider responses. It has **not** been
run against the live Planetary Computer or Element84 services. Run one real scene and
check the output before relying on it in production.
:::

## When streaming works

Streaming needs each band to be a separately addressable, range-readable raster. That
depends on how a provider packages its data.

| Provider | Packaging | Streaming |
|---|---|---|
| `planetary_computer` | One COG per band | Supported (SAS tokens are signed automatically) |
| `element84` | One COG per band | Supported |
| `aws_earth` | COGs on S3 | Expected to work; not tested |
| `copernicus` | One `.SAFE.zip` per scene | **Not possible**: a zip is not randomly accessible raster |
| `usgs` | One `.tar` per scene | **Not possible** |
| `opentopography` | DEM clipped to your bounding box server-side | Not needed: the download is already AOI-only |

When an asset is an archive, streaming returns a failed result with
`error_type="streaming_unsupported"` instead of silently downloading the whole file.

## Stream a scene

```python
from pathlib import Path
from pygeofetch import PyGeoFetch
from pygeofetch.models import SearchQuery

client = PyGeoFetch()
bbox = (-1.80, 6.10, -1.70, 6.20)  # min_lon, min_lat, max_lon, max_lat

scenes = client.search(
    SearchQuery(
        bbox=bbox,
        start_date="2024-01-01",
        end_date="2024-03-01",
        cloud_cover_max=20,
        satellites=["sentinel-2"],
    ),
    providers=["planetary_computer"],
)

# Rank by cloud cover yourself: STAC providers ignore sort_by.
scenes = sorted(scenes, key=lambda s: s.cloud_cover if s.cloud_cover is not None else 100)

results = client.stream(
    scenes[:3],
    bbox=bbox,
    destination=Path("./aoi_data"),
    bands=["B02", "B03", "B04", "B08"],
    parallel=4,
)

for r in results:
    coverage = f"{r.aoi_coverage:.0%}" if r.aoi_coverage is not None else "n/a"
    print(r.data_id, r.status, f"AOI coverage: {coverage}")
```

Each band is written to `<destination>/<provider>/<scene-id>_<band>.tif`, the same
provider sub-folder convention `download()` uses (see
[Download layout](download.md#where-files-are-written)). Output files are small: only the
AOI window, copied block by block, so peak memory is one block (1024 x 1024 pixels by
default) regardless of scene size. Files are written atomically, and existing files are
skipped unless `overwrite=True`.

### Check AOI coverage

A Sentinel-2 tile boundary can cross your AOI. Reading the window then succeeds but
returns a **partial** raster that is mostly empty. Every streamed result carries
`aoi_coverage`, the fraction (0-1) of your AOI the streamed raster covers. Reject scenes
below your threshold:

```python
good = [r for r in results if r.status == "completed" and (r.aoi_coverage or 0) >= 0.99]
```

### Parameters of `client.stream()`

| Parameter | Default | Description |
|---|---|---|
| `data` | | A `SatelliteData` or a list of them. |
| `bbox` | | AOI as `(minx, miny, maxx, maxy)`, a `BoundingBox`, or a shapely geometry. |
| `destination` | | Output directory. |
| `bands` | `None` | Band names or asset keys. `None` streams every raster asset. A name that matches no asset is an error, not a fallback. |
| `parallel` | `4` | Scenes streamed concurrently. |
| `overwrite` | `False` | Re-stream files that already exist. |
| `bbox_crs` | `"EPSG:4326"` | CRS of `bbox`. |
| `block_size` | `1024` | Block edge in pixels for the copy loop. |

`DownloadResult.bytes_downloaded` is the **size of the files written to disk**, not
network traffic.

## Read into memory without writing files

For interactive analysis, `read_scene_bands()` returns the AOI window of each band as a
NumPy array:

```python
from pygeofetch.streaming import read_scene_bands

bands = read_scene_bands(
    scenes[0], bbox, bands=["B04", "B08"],
    decimation=1,          # 2 reads every second pixel, using overviews if present
    max_pixels=50_000_000, # raises decimation automatically instead of allocating more
)

red, nir = bands["B04"], bands["B08"]
print(red.data.shape, red.transform, red.aoi_coverage)
ndvi = (nir.data[0] - red.data[0]) / (nir.data[0] + red.data[0] + 1e-6)
```

`read_scene_bands()` is the one to use when you only need the numbers. `client.stream()`
is the one to use when later steps need files on disk.

## Lower-level functions

| Function | Use |
|---|---|
| `read_window(href, bbox, ...)` | Read the AOI window of one raster URL or path into memory. |
| `stream_window_to_geotiff(href, bbox, out_path, ...)` | Copy one AOI window to a local GeoTIFF, block by block. |
| `stream_scene(data, bbox, destination, ...)` | Stream one `SatelliteData` record. |
| `stream_many(items, bbox, destination, ...)` | Stream several in parallel; result order matches input order. |
| `select_streamable_assets(data, bands)` | See which assets would be streamed, without reading anything. |

These live in `pygeofetch.streaming`.

## How much does it save?

Measured against a local range-request server with a 36.6 MB single-band COG and a
2.5 km AOI:

| Operation | Bytes served | Share of file |
|---|---|---|
| `read_window`, full resolution | 455 KB, 3 range requests | 1.24% |
| `read_window`, `decimation=4` | 262 KB | 0.7% |
| `stream_window_to_geotiff` | 422 KB served, 103 KB written | 1.15% |

Pixel values were asserted identical to a local read of the same window. Real savings
depend on AOI size, tile layout and the provider's COG block structure. Small AOIs in
large tiles gain the most.

## Behaviour to be aware of

- **Rectangular window.** The window is your AOI's bounding box rounded outward to whole pixels. It is not masked to a polygon.
- **Rotated or sheared rasters** raise `NotImplementedError`.
- **JP2 assets** are accepted but untested. Windowed JPEG 2000 reads are slower and less predictable than COG.
- **Collections for Planetary Computer.** `SearchQuery(collections=["SENTINEL-2"])` matches nothing because the provider passes collection names through verbatim. Use `satellites=["sentinel-2"]`, which maps to `sentinel-2-l2a`.
- **Strict band matching.** The regular `download()` falls back to *all* assets when a band name matches nothing. Streaming treats an unmatched band as an error, because the fallback would be expensive.
- **Sentinel-2 L2A offset.** Scenes processed with baseline 04.00 or later store `DN = reflectance x 10000 + 1000`. Indices such as NDVI are not offset-invariant, so subtract the offset before computing them. The [multi-sensor pipelines](../processing/multisensor.md) accept `optical_offset` and `optical_scale` for this.

## Other streaming tools

`pygeofetch.stac_compute` and the `pygeofetch stream` commands predate this module and
work on a **single URL** rather than on search results:

- `pygeofetch stream cog-window URL --bbox ... --output out.tif` reads one window from one COG.
- `pygeofetch stream to-zarr`, `to-geoparquet` and `zarr-to-tif` convert local data to cloud-native formats.
- `pygeofetch.stac_compute.burst_streaming` reads Sentinel-1 SLC bursts remotely for InSAR.

Use `client.stream()` for scenes from a search, with provider authentication, coverage
reporting and resume. Use the `stream` commands for a URL you already have.
