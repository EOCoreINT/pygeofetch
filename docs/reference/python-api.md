# Python API Reference

Use pygeofetch as a library in your own scripts, notebooks, or
applications.

```{note}
The class is `PyGeoFetch` (capitalized) — earlier docs for this
project used a lowercase `pygeofetch()` constructor in examples, which
does not match the installed package.
```

## Full workflow example

```python
from pathlib import Path
from pygeofetch import PyGeoFetch
from pygeofetch.models import SearchQuery, DownloadOptions

client = PyGeoFetch()

client.add_credentials("usgs", username="user", password="pass")
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
| `timeout_seconds` | `int` | Default 60. |

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
| `extract_archives` | `bool` | Default `True`. |

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

## `PyGeoFetch` methods

**`__init__(config_path=None, log_level="INFO", log_json=False, cache_ttl=3600, max_search_workers=8, progress_callback=None, auth_backend="file", validate_optical=False, optical_validation_config=None)`**
`validate_optical` sets the instance-level default for optical
preflight validation (see {doc}`/core-features/optical-validation`),
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
the same way as on `search()` — a genuinely independent gate, useful
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
{doc}`/reference/security`). Overrides matching env vars for the process.

**`status() -> dict`**
Registered providers, auth status, cache stats.

**`clear_cache() -> int`**
Clears the in-memory search result cache; returns entries cleared.

**`fetch_orbit_file(product_name: str, output_dir: str = "./orbits/", orbit_type: str = "precise") -> str | None`**
Downloads a Sentinel-1 precise/restituted orbit file, extracts the `.EOF` automatically.

**`pipeline(name: str) -> ProcessingPipeline`**
Creates a chainable processing pipeline over one file — see
{doc}`/reference/pipelines`'s "Python Processing Pipeline" section for
the full guide. Not related to the YAML acquisition-pipeline
mechanism (`pygeofetch pipeline run`) despite the shared name.

**`batch_process(inputs, chain, output_dir=".", parallel=2) -> list[ProcessingResult]`**
Runs the same processing chain over many files in parallel —
shorthand for `client.batch.process(...)`. See
{doc}`/reference/pipelines`'s "Batch Processing" section.

```{note}
`PyGeoFetch` has **no** `.providers()` method. List providers via
`pygeofetch.providers.list_providers()` / `list_provider_info()` — see
{doc}`/core-features/providers`.
```

## Optical validation quick reference

See {doc}`/core-features/optical-validation` for the full guide.

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
