# Python API Reference

Use pygeofetch as a library in your own scripts, notebooks, or
applications.

The main entry point is the `PyGeoFetch` class. Data models live in `pygeofetch.models`.

```{contents}
:local:
:depth: 2
```

## Overview

The Python API is organised around one object, the `PyGeoFetch` client, and a small set of models. You describe a request with `SearchQuery`, call `client.search()` to obtain a list of `SatelliteData` scenes, and pass those scenes to `client.download()` or `client.stream()`. Download behaviour is controlled with `DownloadOptions`, and each download returns a `DownloadResult` that records the outcome. Processing capabilities are exposed as namespaces on the same client, for example `client.indices`, `client.preprocess`, `client.sar` and `client.post`, so that one import is enough for most workflows.

Because the models are Pydantic classes, they validate their inputs and can be converted to and from dictionaries and JSON, which makes it easy to store a query, log a result or build requests from configuration. The sections below list the fields of each model, the methods of the client and the namespaces it carries.

## Full workflow example

```python
from pathlib import Path
from pygeofetch import PyGeoFetch
from pygeofetch.models import SearchQuery, DownloadOptions

client = PyGeoFetch()

client.add_credentials("usgs", username="user", api_key="your_m2m_application_token")
client.add_credentials("planet", api_key="PL_KEY")

results = client.search(
    SearchQuery(
        bbox=(-74.1, 40.6, -73.7, 40.9),
        start_date="2024-01-01",
        end_date="2024-06-01",
        cloud_cover_max=20,
        max_results=50,
        sort_by="cloud_cover",
    ),
    providers=["usgs", "copernicus", "planetary_computer", "aws_earth"],
)

print(f"Found {len(results)} scenes")
for r in results[:3]:
    print(f"  {r.id} | {r.datetime} | {r.cloud_cover:.1f}% cloud")

download_results = client.download(
    results[:5],
    destination=Path("./data/"),
    options=DownloadOptions(
        parallel=4,
        verify_checksum=True,
        resume=True,
        bands=["B02", "B03", "B04"],
    ),
)

for dr in download_results:
    if dr.status == "completed":
        print(f"  ✓ {dr.data_id} ({dr.bytes_downloaded // 1024 // 1024:.1f} MB)")
    else:
        print(f"  ✗ {dr.data_id}: {dr.error}")
```

## `SearchQuery`

Describes what to search for. Every field is optional. Without an area (`bbox`, `geometry` or `geometry_geojson`), a search is not spatially restricted, which most providers reject or answer with very large result sets.

| Field | Type | Description |
|---|---|---|
| `bbox` | `tuple[float,4] \| BoundingBox \| str \| dict \| None` | Accepts a `(minlon, minlat, maxlon, maxlat)` tuple, a `BoundingBox`, a `"minlon,minlat,maxlon,maxlat"` string, or a dict — all coerced to `BoundingBox`. |
| `geometry` | `dict \| None` | GeoJSON geometry dict. Alternative to `bbox`. |
| `start_date` / `end_date` | `date \| datetime \| None` | Temporal filter. |
| `cloud_cover_min` / `cloud_cover_max` | `float` | 0–100. Default `0` / `100`. |
| `satellites` | `list[str]` | e.g. `["Sentinel-2", "Landsat-8"]`. |
| `sensors` | `list[str]` | Sensor/instrument names. |
| `collections` | `list[str]` | Data collection identifiers. |
| `processing_levels` | `list[str]` | e.g. `["L2A"]`. |
| `resolution_min_m` / `resolution_max_m` | `float \| None` | Spatial resolution in metres. **Not** `resolution_min`/`resolution_max`. |
| `product_type` | `str \| None` | `"GRD"`, `"SLC"`, `"GRD-COG"`. |
| `polarisation` | `str \| None` | e.g. `"VV"`. |
| `max_results` | `int` | 1–10000. Default 100. |
| `sort_by` | `str` | `"datetime"`, `"cloud_cover"`, `"score"`, `"satellite"`. |
| `sort_ascending` | `bool` | Default `False`. **Not** a `sort_order: "asc"/"desc"` string. |
| `providers` | `list[str]` | Restrict search to specific providers. |
| `cql2_filter` | `str \| None` | CQL2 expression, STAC-capable providers only. |
| `on_provider_failure` | `str` | `"skip"`, `"abort"`, `"retry"`. |
| `timeout_seconds` | `int` | Per-provider timeout in seconds. Default 60. |
| `ids` | `list[str]` | Retrieve specific item IDs. |
| `geometry_geojson` | `dict \| None` | GeoJSON geometry dict (alternative to `bbox`). |
| `pass_direction` | `str \| None` | SAR orbit pass, `"ASCENDING"` or `"DESCENDING"`. |
| `page` / `page_size` | `int` | Pagination, default `1` / `100` (`page_size` 1-1000). |
| `provider_filters` | `dict[str, dict]` | Extra filters per provider, keyed by provider ID. |

## `DownloadOptions`

| Field | Type | Description |
|---|---|---|
| `parallel` | `int` | 1–32. Default 4. |
| `retry_attempts` | `int` | 0–10. Default 3. **Not** `retry`. |
| `retry_delay_seconds` | `float` | Default 1.0, doubles each attempt. |
| `timeout_seconds` | `float` | Per-download timeout. Default 300. |
| `resume` | `bool` | Default `True`. |
| `verify_checksum` | `bool` | Default `False`. |
| `checksum_algorithm` | `ChecksumAlgorithm` | Default `SHA256`. |
| `on_failure` | `str` | `"skip"`, `"abort"`, `"retry"`. |
| `bandwidth_limit_mbps` | `float \| None` | Speed cap in **Mbps** (a float), not a `"10MB"` string. `None` = unlimited. |
| `priority` | `int` | Higher = higher priority. |
| `bands` | `list[str] \| None` | Default: all data assets. |
| `post_process` | `list[PostProcessAction]` | **Must** be `PostProcessAction` objects — a plain string list raises a `ValidationError`. See below. |
| `overwrite` | `bool` | Default `False` (skip existing). |
| `extract_archives` | `bool` | Not used by any code. Extract archives with the `unzip` [post-processing action](../data-access/download.md#archives). |
| `retry_strategy` | `str` | Default `"exponential_jitter"`. |
| `chunk_size_mb` | `float` | Default `16.0`. |
| `max_file_size_gb` | `float \| None` | Optional size cap. |
| `flatten_directory` | `bool` | Default `False`. |
| `output_format` | `str` | Default `"original"`. |
| `notify_webhook` / `notify_email` | `str \| None` | Notification targets. |

### Constructing `post_process` correctly

```python
from pygeofetch.models.download_task import PostProcessAction

DownloadOptions(
    post_process=[
        PostProcessAction(action="reproject", params={"value": "EPSG:4326"}),
        PostProcessAction(action="cog"),
    ]
)
```

The CLI's `--post-process "reproject:EPSG:4326,cog"` string form is
parsed into exactly this shape internally — it's only when
constructing `DownloadOptions` directly in Python that raw strings
don't work.

## `DownloadResult`

Returned by `download()`, `download_from_file()` and `stream()`, one per input item and in input order.

| Field | Type | Description |
|---|---|---|
| `status` | `DownloadStatus` | `pending`, `downloading`, `completed`, `failed`, `skipped` or `cached`. |
| `data_id` | `str \| None` | The `SatelliteData.id` this result is for. |
| `provider` | `str \| None` | Provider ID. |
| `output_path` | `Path \| None` | First output file. |
| `output_paths` | `list[Path]` | Every output file. After the `unzip` action these are the extracted files. |
| `bytes_downloaded` | `int` | Size of the files written. For streamed results this is the size on disk, not network traffic. |
| `duration_seconds` | `float` | Wall-clock time. |
| `error` / `error_type` | `str \| None` | Failure message and the exception class name or provider-specific code. |
| `retries_used` / `attempt_number` | `int` | Retry bookkeeping. |
| `checksum_verified` | `bool` | See [Error handling](error-handling.md#download-resilience) for what this does and does not guarantee. |
| `post_process_completed` | `list[str]` | Post-processing steps that ran. |
| `from_cache` | `bool` | The file already existed and was reused. |
| `aoi_coverage` | `float \| None` | Streamed results only: fraction (0-1) of the AOI covered by the streamed raster. |

## `SatelliteData`

One scene returned by a search.

| Field | Type | Description |
|---|---|---|
| `id`, `provider` | `str` | Scene ID and the provider that returned it. |
| `satellite`, `sensor`, `collection` | `str \| None` | Platform, instrument and collection. |
| `datetime`, `start_datetime`, `end_datetime` | `datetime \| None` | Acquisition time or interval. |
| `bbox`, `geometry` | `tuple`, `dict` | Footprint as a bounding box and as GeoJSON. |
| `cloud_cover` | `float \| None` | Percent. |
| `processing_level`, `data_format` | enums | `UNKNOWN` when the provider does not say. |
| `assets` | `dict[str, SatelliteAsset]` | Downloadable files, keyed by asset name. Each asset has `href`, `title`, `media_type`, `roles`, `size_bytes`, `checksum_md5` and `checksum_sha256`. |
| `product_type`, `polarisation`, `pass_direction`, `relative_orbit`, `orbit_number`, `incidence_angle` | various | SAR metadata. |
| `resolution_m`, `gsd_m`, `off_nadir_angle` | `float \| None` | Resolution and viewing geometry. |
| `score` | `float` | Relevance score assigned by the searcher. See [Searching](../data-access/search.md). |
| `properties`, `links`, `stac_extensions` | | Raw provider metadata. |

`SatelliteData.from_stac_item()` and `to_stac_item()` convert to and from STAC items.

## `PyGeoFetch` methods

**`__init__(config_path=None, log_level="INFO", log_json=False, cache_ttl=3600, max_search_workers=8, progress_callback=None, auth_backend="file", validate_optical=False, optical_validation_config=None)`**
`validate_optical` sets the instance-level default for optical
preflight validation (see [Optical Data Validation & Preflight](../data-access/optical-validation.md)),
applied automatically by both `search()` and `download()` unless
overridden per call. Defaults to `False` — fully opt-in, existing
behavior unchanged unless explicitly enabled. `optical_validation_config`
is the `OpticalValidationConfig` to use when enabled; defaults to
`OpticalValidationConfig()`'s own defaults if not given. Neither
parameter requires `shapely` to be installed unless validation is
actually turned on — the import is lazy.

**`search(query: SearchQuery, providers: list[str] | None = None, use_cache: bool = True, validate_optical: bool | None = None) -> list[SatelliteData]`**
Federated search. Returns deduplicated, scored results.
`validate_optical=None` (default) falls back to the instance-level
setting from `__init__`; pass `True`/`False` to override just this
call. When active, results are filtered through
`OpticalPreflightValidator.run_preflight()` before being returned —
the AOI is derived automatically from `query.bbox`/`query.geometry`.

**`search_and_save(query, output: Path, providers=None) -> list[SatelliteData]`**
Search and write results to a GeoJSON file in one call.

**`download(data, destination: Path, options: DownloadOptions | None = None, item_done_callback=None, validate_optical: bool | None = None, aoi: Polygon | None = None) -> list[DownloadResult]`**
Accepts a single `SatelliteData` or a list. `validate_optical` works
the same way as on `search()` — a independent gate, useful
for items that never went through `search()` at all (e.g. loaded via
`download_from_file()`). Rejected items are **never attempted** — a
synthesized `DownloadResult(status=FAILED, error=...)` stands in at
that item's original position instead, so the returned list's length
and order always match `data` regardless of how many items were
rejected. `aoi` is optional here (there's no query to derive one
from); without it, AOI-dependent checks are skipped while every other
check still runs.

**`download_from_file(search_results_path: Path, destination: Path, options=None) -> list[DownloadResult]`**
Load a GeoJSON results file (from `search_and_save`) and download it directly.

**`add_credentials(provider: str, *, username=None, password=None, api_key=None, client_id=None, client_secret=None, token=None, access_key=None, secret_key=None, **kwargs) -> None`**
Stored in the system keyring (or Fernet-encrypted file fallback — see
[Security Model](security.md)). Overrides matching env vars for the process.

**`status() -> dict`**
Registered providers, auth status, cache stats.

**`clear_cache() -> int`**
Clears the in-memory search result cache; returns entries cleared.

**`stream(data, bbox, destination, bands=None, parallel=4, overwrite=False, bbox_crs="EPSG:4326", block_size=1024) -> list[DownloadResult]`**
Reads only the AOI window of each band from Cloud-Optimized GeoTIFFs and writes small local
GeoTIFFs, instead of downloading whole scenes. Supported for providers that serve per-band COGs
(`planetary_computer`, `element84`). See [Streaming](../data-access/streaming.md).

**`fetch_orbit_file(product_name: str, output_dir: str = "./orbits/", orbit_type: str = "precise") -> str | None`**
Downloads a Sentinel-1 precise/restituted orbit file, extracts the `.EOF` automatically.

**`pipeline(name: str) -> ProcessingPipeline`**
Creates a chainable processing pipeline over one file — see
[Pipelines & Batch Processing](../workflows/pipelines.md)'s "Python Processing Pipeline" section for
the full guide. Not related to the YAML acquisition-pipeline
mechanism (`pygeofetch pipeline run`) despite the shared name.

**`batch_process(inputs, chain, output_dir=".", parallel=2) -> list[ProcessingResult]`**
Runs the same processing chain over many files in parallel —
shorthand for `client.batch.process(...)`. See
[Pipelines & Batch Processing](../workflows/pipelines.md)'s "Batch Processing" section.

## Client namespaces

`PyGeoFetch` instances expose processing engines as attributes:

| Attribute | Class | Documented in |
|---|---|---|
| `client.auth` | `AuthManager` | [Authentication](../data-access/authentication.md) |
| `client.preprocess` | `Preprocessor` | [Preprocessing](../processing/optical/preprocessing.md), [Terrain](../processing/optical/terrain.md) |
| `client.indices` | `SpectralIndices` | [Spectral indices](../processing/optical/spectral-indices.md) |
| `client.post` | `PostProcessor` | [Post-processing](../processing/optical/postprocessing.md) |
| `client.sar` | `SARProcessor` | [SAR](../processing/sar-insar/sar.md) |
| `client.classify` | `Classifier` | [Classification](../processing/optical/classification.md) |
| `client.batch` | `BatchProcessor` | [Pipelines](../workflows/pipelines.md) |

Landsat extraction (`pygeofetch.processor.LandsatExtractor`), time series
(`TimeSeriesAnalyzer`), the standalone `SpectralIndex`, the multi-sensor pipelines and the InSAR
modules are standalone imports, not attributes of the client.

:::{note}

`PyGeoFetch` has **no** `.providers()` method. List providers via
`pygeofetch.providers.list_providers()` / `list_provider_info()` — see
[Providers](../data-access/providers.md).
:::
## Optical validation quick reference

See [Optical Data Validation & Preflight](../data-access/optical-validation.md) for the full guide.

```python
from pygeofetch import PyGeoFetch
from pygeofetch.validation import OpticalValidationConfig

pf = PyGeoFetch(
    validate_optical=True,
    optical_validation_config=OpticalValidationConfig(max_cloud_cover_pct=10.0),
)
results = pf.search(query)          # already filtered
downloads = pf.download(results, "./data")  # validated again as a final gate if you pass validate_optical=True here too
```
