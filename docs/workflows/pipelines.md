# Pipelines & Batch Processing

:::{note}

**Four different things share the word "pipeline" in this
codebase** — this page covers the first two, clearly separated; the
other two each have their own dedicated page.

1. **YAML pipeline orchestration** (below) — `search` → `filter` →
 `download` → `process` → `export`, run ad-hoc or on a cron
 schedule via the CLI. Built for *acquisition* workflows.
2. **The Python fluent processing pipeline** (`client.pipeline(...)`,
 further down this page) — a chainable builder over
 preprocessing/index/postprocessing/SAR operations
 (`.clip().reproject().ndvi().cog().run(...)`). Built for
 *processing* workflows on files you already have.
3. **The five SAR processing pipelines**
 (`pygeofetch.sar.pipelines`) — standard end-to-end SAR
 workflows (`standard_grd_preprocessing_pipeline`,
 `flood_mapping_pipeline`, `change_detection_pipeline`,
 `coherence_disturbance_pipeline`, `bright_target_detection_pipeline`),
 each orchestrating `SARProcessor`'s atomic operations into one
 named analysis chain across multiple *dates of the same
 sensor*. See [SAR Processing](../processing/sar-insar/sar.md#five-standard-sar-processing-pipelines).
4. **The five multi-sensor pipelines**
 (`pygeofetch.multisensor`) — combining two *different
 sensor types* (InSAR+optical, SAR+optical, DEM+optical, DEM+DEM,
 optical+SAR) to do something neither sensor alone reliably does.
 See [Multi-Sensor Pipelines](../processing/multisensor.md).

Neither 3 nor 4 is documented on this page, since both are specific to
their own sensor-combination physics rather than general-purpose
acquisition or file-processing chains.

There's also a fifth, simpler option for "run this same processing
chain over many files in parallel" that isn't any of the above —
see "Batch Processing" at the bottom of this page.
:::
Define recurring satellite data workflows in a single YAML file.
Schedule on cron, run ad-hoc, validate before committing, watch live
logs.

:::{note}

**`process` and `export` steps.** `process` reuses the action executor behind
`DownloadOptions.post_process` and the CLI `--post-process` flag. `export` copies files to local
disk, uploads to S3 (`s3://...`) or GCS (`gs://...`), and can POST a webhook notification on
completion.
:::
## Overview

When the same sequence of steps is run repeatedly, such as every week or for every new scene, it is better written down than typed. A YAML pipeline describes the whole acquisition: the area and dates to search, the filters to apply, what to download, the processing steps to run and where to export the results. You can run it once, run it on a cron schedule, or hand it to a colleague, and it behaves the same way each time.

A pipeline file has a few named sections. The search step says what to look for, the filter step trims the results, the download step fetches them, the process steps apply post-processing actions or indices, and the export step sends the finished files to local disk, S3 or GCS with an optional webhook notification. Start from one of the built-in templates, validate the file before running it, and run it with the CLI. For work on files you already have, the Python processing pipeline lets you chain operations in code instead, and batch processing applies a function across many files in parallel.

## Pipeline steps

| Step | Config keys | Status |
|---|---|---|
| `search` | `providers`, `bbox`, `date_range`, `cloud_cover`, `max_results` | Functional |
| `filter` | `expression`, `max_items` | Functional — `expression` is evaluated against each result's `data` object |
| `download` | `parallel`, `output`, `retry`, `verify_checksum`, `bands` | Functional |
| `process` | a comma-separated action string, or a list of action strings/dicts | Functional — reproject/COG/NDVI/clip/etc. |
| `export` | `destination` (local path, `s3://`, or `gs://`), `format`, `notify` | Functional — file transfer + optional webhook |

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

:::{note}

`pipeline schedule` uses the system cron daemon on Linux/macOS, and
Windows Task Scheduler on Windows. Run `pygeofetch pipeline
list-scheduled` to confirm registration.
:::
---

## Python Processing Pipeline — `client.pipeline(...)`

A different capability from the YAML orchestration above: a
**chainable builder** over the processing operations documented
throughout [Preprocessing Engine](../processing/optical/preprocessing.md),
[Spectral Indices](../processing/optical/spectral-indices.md), [Postprocessing](../processing/optical/postprocessing.md),
and [SAR Processing](../processing/sar-insar/sar.md) — for processing files you already have,
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
chain, not independent calls that happen to be written
sequentially. `.run(input=...)` executes every queued step in order
and returns a `PipelineRunResult` with per-step status, output paths,
and timing.

### Every chainable step

Nearly every operation documented elsewhere on this site is available
as a pipeline step, grouped by where it's documented in full:

| Category | Steps |
|---|---|
| Preprocessing ([Preprocessing Engine](../processing/optical/preprocessing.md)) | `clip`, `reproject`, `resample`, `cloud_mask`, `cloud_fill`, `atmos`, `composite`, `mosaic`, `topo_correct`, `pansharpen`, `tile` |
| Spectral indices ([Spectral Indices](../processing/optical/spectral-indices.md)) | `ndvi`, `evi`, `ndwi`, `ndbi`, `ndsi`, `ndmi`, `nbr`, `dnbr`, `savi`, `mndwi`, `tct`, `pca`, `lst`, `albedo`, `band_math`, `stack`, `texture` |
| Postprocessing ([Postprocessing](../processing/optical/postprocessing.md)) | `vectorize`, `smooth`, `regularize`, `zonal_stats`, `buffer`, `centroids`, `add_geometry_metrics`, `compress`, `cog` |
| SAR ([SAR Processing](../processing/sar-insar/sar.md)) | `despeckle`, `calibrate`, `flood_map`, `coherence` |

Each step method accepts the exact same keyword arguments as its
corresponding `client.preprocess`/`client.indices`/`client.post`/
`client.sar` method documented on those pages — `.clip(bbox=...)`,
`.ndvi(red=..., nir=...)`, etc. — since the pipeline builder is a thin
queuing layer over those same implementations, not a separate
reimplementation.

### From YAML

```python
from pygeofetch.processing.pipeline import ProcessingPipeline

pl = ProcessingPipeline.from_yaml("ndvi_workflow.yaml", engine=client)
result = pl.run(input="scene.tif")
```

:::{danger}

**The class docstring's usage example does not work.** `ProcessingPipeline`'s docstring shows
`client.pipeline.from_yaml("ndvi_workflow.yaml").run()`. `from_yaml` is a classmethod on
`ProcessingPipeline`, while `client.pipeline` is a method, so that call raises
`AttributeError: 'function' object has no attribute 'from_yaml'`. Use
`ProcessingPipeline.from_yaml(path, engine=client)`, as shown above.
:::
:::{warning}

This YAML format (a chain of processing steps for one file) is **not
the same YAML format** as the acquisition-pipeline YAML at the top of
this page (`search`/`filter`/`download`/`process`/`export` for a
recurring cron job). Don't mix the two — a
`weekly-sentinel2.yaml`-style file passed to
`ProcessingPipeline.from_yaml()` won't produce the steps you expect,
and vice versa.
:::
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
