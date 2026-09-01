# Pipelines & Batch Processing

```{note}
**Two genuinely different things share the word "pipeline" in this
codebase** — this page covers both, clearly separated:

1. **YAML pipeline orchestration** (below) — `search` → `filter` →
   `download` → `process` → `export`, run ad-hoc or on a cron
   schedule via the CLI. Built for *acquisition* workflows.
2. **The Python fluent processing pipeline** (`client.pipeline(...)`,
   further down this page) — a chainable builder over
   preprocessing/index/postprocessing/SAR operations
   (`.clip().reproject().ndvi().cog().run(...)`). Built for
   *processing* workflows on files you already have.

There's also a third, simpler option for "run this same processing
chain over many files in parallel" that isn't either of the above —
see "Batch Processing" at the bottom of this page.
```

Define recurring satellite data workflows in a single YAML file.
Schedule on cron, run ad-hoc, validate before committing, watch live
logs.

```{note}
**Previously stub, now real.** The `process` and `export` steps used
to log a message and return `{"status": "stub"}` without doing
anything. Both now delegate to real, tested implementations: `process`
reuses the same action executor `DownloadOptions.post_process` and the
CLI's `--post-process` flag use; `export` genuinely copies files to
local disk, uploads to S3 (`s3://...`) or GCS (`gs://...`), and can
POST a webhook notification on completion. See
{doc}`/reference/error-handling` for the equivalent fix to the circuit
breaker, and {doc}`/reference/security` for the credential-encryption
fix — all three were found and fixed together during this
documentation pass.
```

## Pipeline steps

| Step | Config keys | Status |
|---|---|---|
| `search` | `providers`, `bbox`, `date_range`, `cloud_cover`, `max_results` | Functional |
| `filter` | `expression`, `max_items` | Functional — `expression` is evaluated against each result's `data` object |
| `download` | `parallel`, `output`, `retry`, `verify_checksum`, `bands` | Functional |
| `process` | a comma-separated action string, or a list of action strings/dicts | Functional — real reproject/COG/NDVI/clip/etc. |
| `export` | `destination` (local path, `s3://`, or `gs://`), `format`, `notify` | Functional — real file transfer + optional webhook |

## Example pipeline YAML

```yaml
# weekly-sentinel2.yaml
name: weekly-sentinel2-ndvi
schedule: "0 6 * * 1"     # Every Monday at 06:00 UTC
description: Weekly Sentinel-2 acquisition, processed and exported to S3

steps:
  - search:
      providers: [copernicus, aws_earth, planetary_computer]
      date_range: last_7_days
      cloud_cover: 0-10
      bbox: "-74.1,40.6,-73.7,40.9"
      max_results: 20

  - filter:
      expression: "data.cloud_cover < 5"
      max_items: 5

  - download:
      parallel: 4
      output: ./raw/
      verify_checksum: true
      bands: "B04,B08"          # NDVI bands only

  - process: "reproject:EPSG:4326,ndvi,cog"

  - export:
      destination: s3://my-bucket/ndvi/
      format: cloud_optimized_geotiff
      notify: "webhook:https://hooks.slack.com/services/YOUR/WEBHOOK"
```

`process` accepts the same syntax as `DownloadOptions.post_process` /
the CLI's `--post-process` flag: a comma-separated string, a list of
strings, or a list of `{action: ..., params: {...}}` dicts. It reads
from the `download` step's output and writes its result forward for
`export` to pick up.

`export`'s `destination` decides the transfer method automatically —
`s3://bucket/prefix/` uploads via `boto3`, `gs://bucket/prefix/`
uploads via `google-cloud-storage` (not a default dependency — install
it separately if you need GCS export), and anything else is treated
as a local directory. `notify` accepts a `webhook:URL` string (or a
list of them) and POSTs a small JSON summary
(`{"exported": N, "total": M, "destination": ...}`) once the transfer
completes.

## Pipeline CLI commands

```bash
# Run immediately (one-shot)
pygeofetch pipeline run weekly-sentinel2.yaml

# Validate YAML without executing
pygeofetch pipeline validate weekly-sentinel2.yaml

# Schedule for recurring execution
pygeofetch pipeline schedule weekly-sentinel2.yaml --name ndvi-monitor

# List all scheduled pipelines
pygeofetch pipeline list-scheduled

# Watch logs live
pygeofetch pipeline logs ndvi-monitor --follow

# View run history
pygeofetch pipeline history --limit 20

# Retry a failed run
pygeofetch pipeline retry RUN_ID_HERE

# Stop scheduling
pygeofetch pipeline unschedule ndvi-monitor

# Run a specific step only
pygeofetch pipeline run weekly-sentinel2.yaml --step download
```

```{note}
`pipeline schedule` uses the system cron daemon on Linux/macOS, and
Windows Task Scheduler on Windows. Run `pygeofetch pipeline
list-scheduled` to confirm registration.
```

---

## Python Processing Pipeline — `client.pipeline(...)`

A genuinely different capability from the YAML orchestration above: a
**chainable builder** over the processing operations documented
throughout {doc}`/processing/preprocessing`,
{doc}`/processing/spectral-indices`, {doc}`/processing/postprocessing`,
and {doc}`/processing/sar` — for processing files you already have,
not for search/download orchestration.

```python
from pygeofetch import PyGeoFetch

client = PyGeoFetch()

result = (
    client.pipeline("my-ndvi")
    .clip(bbox=(-74.1, 40.6, -73.7, 40.9))
    .reproject(crs="EPSG:4326")
    .ndvi(red="B04", nir="B08")
    .cog()
    .run(input="scene.tif")
)

print(result.outputs)   # list of output paths, one per step
```

Each step's output feeds directly into the next step as input — a
real chain, not independent calls that happen to be written
sequentially. `.run(input=...)` executes every queued step in order
and returns a `PipelineRunResult` with per-step status, output paths,
and timing.

### Every real, chainable step

Nearly every operation documented elsewhere on this site is available
as a pipeline step, grouped by where it's documented in full:

| Category | Steps |
|---|---|
| Preprocessing ({doc}`/processing/preprocessing`) | `clip`, `reproject`, `resample`, `cloud_mask`, `cloud_fill`, `atmos`, `composite`, `mosaic`, `topo_correct`, `pansharpen`, `tile` |
| Spectral indices ({doc}`/processing/spectral-indices`) | `ndvi`, `evi`, `ndwi`, `ndbi`, `ndsi`, `ndmi`, `nbr`, `dnbr`, `savi`, `mndwi`, `tct`, `pca`, `lst`, `albedo`, `band_math`, `stack`, `texture` |
| Postprocessing ({doc}`/processing/postprocessing`) | `vectorize`, `smooth`, `regularize`, `zonal_stats`, `buffer`, `centroids`, `add_geometry_metrics`, `compress`, `cog` |
| SAR ({doc}`/processing/sar`) | `despeckle`, `calibrate`, `flood_map`, `coherence` |

Each step method accepts the exact same keyword arguments as its
corresponding `client.preprocess`/`client.indices`/`client.post`/
`client.sar` method documented on those pages — `.clip(bbox=...)`,
`.ndvi(red=..., nir=...)`, etc. — since the pipeline builder is a thin
queuing layer over those same real implementations, not a separate
reimplementation.

### From YAML

```python
from pygeofetch.processing.pipeline import ProcessingPipeline

pl = ProcessingPipeline.from_yaml("ndvi_workflow.yaml", engine=client)
result = pl.run(input="scene.tif")
```

```{danger}
**Real bug found in the source's own docstring, verified by testing
directly**: `ProcessingPipeline`'s class docstring shows
`client.pipeline.from_yaml("ndvi_workflow.yaml").run()` as the usage
example. `from_yaml` is a real `@classmethod` on `ProcessingPipeline`
itself — `client.pipeline` is a bound *method* (it returns a new
`ProcessingPipeline` instance when called), and Python methods don't
have a `.from_yaml` attribute. Calling it exactly as the docstring
shows raises `AttributeError: 'function' object has no attribute
'from_yaml'` — confirmed by running it. The working form is
`ProcessingPipeline.from_yaml(path, engine=client)`, shown above.
```

```{warning}
This YAML format (a chain of processing steps for one file) is **not
the same YAML format** as the acquisition-pipeline YAML at the top of
this page (`search`/`filter`/`download`/`process`/`export` for a
recurring cron job). Don't mix the two — a
`weekly-sentinel2.yaml`-style file passed to
`ProcessingPipeline.from_yaml()` won't produce the steps you expect,
and vice versa.
```

---

## Batch Processing — `client.batch`

For "run the same processing chain over many files, in parallel" —
simpler than either pipeline concept above when there's no need for
scheduling or acquisition steps, just parallel fan-out over a file
list.

```python
results = client.batch_process(
    inputs=["scene1.tif", "scene2.tif", "scene3.tif"],
    chain=[
        ("clip", {"bbox": (-74.1, 40.6, -73.7, 40.9)}),
        ("ndvi", {"red": "B04", "nir": "B08"}),
        ("cog", {}),
    ],
    output_dir="./processed/",
    parallel=4,
)
```

`chain` is an ordered list of `(step_type, kwargs)` tuples — the same
step names as the fluent pipeline builder above, applied to every
input file independently (not chained *between* files, chained
*within* each file). `on_error` (via `client.batch.process(...,
on_error="skip")`, not exposed on the `client.batch_process(...)`
shortcut) controls whether one failing file aborts the whole batch or
is skipped so the rest still complete — default `"skip"`.

### Arbitrary custom functions

```python
def my_proc(inp, out_dir, threshold=0.3):
    # your own processing logic
    return result

results = client.batch.apply(my_proc, inputs, threshold=0.5, parallel=4)
```

`apply()` runs any function of your own — `(input_path, output_dir,
**kwargs) -> result` — across a file list in parallel, for processing
steps that don't already exist as a named pipeline step.
