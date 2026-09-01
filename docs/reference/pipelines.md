# YAML Pipeline Orchestration

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
