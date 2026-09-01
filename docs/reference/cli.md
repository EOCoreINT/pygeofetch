# Full CLI Reference

```{danger}
**A real, confirmed dead command group exists in the codebase**:
`pygeofetch/cli/monitor_commands.py` defines a real `monitor` group
(`monitor run`, `monitor history`) with real implementation code, but
it is **never registered** in `pygeofetch/cli/main.py` — confirmed by
running `pygeofetch --help` directly and checking the real output: no
`monitor` entry appears, and `pygeofetch monitor run` fails with "No
such command". If you need this functionality, it exists in source
but needs `cli.add_command(monitor)` added to `main.py` to actually
be reachable.
```

## Global options

```
pygeofetch [OPTIONS] COMMAND [ARGS]

Options:
  --log-level TEXT   DEBUG, INFO, WARNING, ERROR  [default: INFO]
  --log-file TEXT    Write logs to file path
  --log-format TEXT  console or json  [default: console]
  --config FILE      Path to config file
  --version          Show version and exit
  --help, -h         Show help message and exit
```

## All command groups

| Group | Subcommands | Description |
|---|---|---|
| `auth` | add, login, list, test, remove, export | Manage provider credentials — {doc}`/core-features/authentication` |
| `providers` | list, info, search | Browse and inspect providers — {doc}`/core-features/providers` |
| `search` | run | Search for satellite scenes — {doc}`/core-features/search` |
| `download` | run, status, history | Download scenes to disk — {doc}`/core-features/download` |
| `cache` | stats, clear, ttl, location, prune | Manage search result cache |
| `pipeline` | run, validate, schedule, list-scheduled, unschedule, logs, history, retry | YAML *acquisition* orchestration — {doc}`/reference/pipelines` |
| `proc-pipeline` | run, validate, template | YAML *processing-chain* pipeline — a different, real capability, see below |
| `preprocess` | clip, reproject, resample, cloud-mask, cloud-fill, atmos, topo-correct, pansharpen, mosaic, composite, tile | Direct CLI access to {doc}`/processing/preprocessing` |
| `index` | ndvi, evi, savi, ndwi, mndwi, ndbi, ndsi, ndmi, nbr, dnbr, tct, pca, texture, lst, albedo, band-math, stack | Direct CLI access to {doc}`/processing/spectral-indices` |
| `post` | vectorize, smooth, regularize, zonal-stats, buffer, centroids, geometry-metrics, compress, cog | Direct CLI access to {doc}`/processing/postprocessing` |
| `sar` | despeckle, calibrate, flood-map, coherence | Direct CLI access to {doc}`/processing/sar` |
| `config` | show, get, set, path, reset | Read and modify configuration |
| `status` | — | System status dashboard |
| `doctor` | — | Diagnose installation and connectivity |
| `version` | — | Show version info |

```{note}
`preprocess`, `index`, `post`, and `sar` are thin, direct CLI wrappers
around the exact same `client.preprocess`/`client.indices`/
`client.post`/`client.sar` methods documented in full on their
respective processing pages — flag names match the Python keyword
argument names throughout (e.g. `--red`/`--nir` on `index ndvi` maps
directly to `red=`/`nir=`). This page lists every real command with a
one-line description and a few concrete examples; see the linked
processing pages for full algorithm detail, real formulas, and
verification basis.
```

## Extra download and cache subcommands

Beyond `download run` (see {doc}`/core-features/download` for its
full flag reference):

```bash
# List downloaded files in a directory with sizes
pygeofetch download status ./data/

# Show past download run history
pygeofetch download history

# Show or set the cache TTL
pygeofetch cache ttl show
pygeofetch cache ttl set 7200

# Show the real cache directory path
pygeofetch cache location

# Remove old cache entries above a size limit
pygeofetch cache prune --max-size 1GB
```

## `preprocess` — direct CLI access

```bash
pygeofetch preprocess clip scene.tif --bbox "-74.1,40.6,-73.7,40.9" --output clipped.tif
pygeofetch preprocess reproject scene.tif --crs EPSG:4326 --output reprojected.tif
pygeofetch preprocess cloud-mask scene.tif --method scl --scl-band SCL.tif
pygeofetch preprocess tile scene.tif --tile-size 256 --overlap 32
pygeofetch preprocess pansharpen multispectral.tif --pan pan.tif --method brovey
```

Full command list: `clip`, `reproject`, `resample`, `cloud-mask`,
`cloud-fill`, `atmos`, `topo-correct`, `pansharpen`, `mosaic`,
`composite`, `tile`. See {doc}`/processing/preprocessing` and
{doc}`/processing/terrain` for what each does and its full parameter
set.

## `index` — direct CLI access to spectral indices

```bash
pygeofetch index ndvi --red B04.tif --nir B08.tif --output ndvi.tif
pygeofetch index evi --blue B02.tif --red B04.tif --nir B08.tif
pygeofetch index tct --blue B02.tif --green B03.tif --red B04.tif --nir B08.tif --swir1 B11.tif --swir2 B12.tif --sensor sentinel2
pygeofetch index band-math --inputs B04.tif,B08.tif --expression "(B[1]-B[0])/(B[1]+B[0]+1e-6)"
```

Full command list: `ndvi`, `evi`, `savi`, `ndwi`, `mndwi`, `ndbi`,
`ndsi`, `ndmi`, `nbr`, `dnbr`, `tct`, `pca`, `texture`, `lst`,
`albedo`, `band-math`, `stack`. See {doc}`/processing/spectral-indices`
for real formulas, published coefficients, and the full parameter set
for each.

## `post` — direct CLI access to postprocessing

```bash
pygeofetch post vectorize classification.tif --threshold 0.5 --min-area 100 --output classes.geojson
pygeofetch post zonal-stats ndvi.tif parcels.geojson --stats mean,median,std --output stats.csv
pygeofetch post cog scene.tif --compress deflate
```

Full command list: `vectorize`, `smooth`, `regularize`,
`zonal-stats`, `buffer`, `centroids`, `geometry-metrics`, `compress`,
`cog`. See {doc}`/processing/postprocessing` for the full parameter
set for each.

## `sar` — direct CLI access to SAR processing

```bash
pygeofetch sar despeckle sentinel1_vv.tif --filter enhanced_lee --window 5
pygeofetch sar calibrate sentinel1_dn.tif --output-type sigma0 --in-db
pygeofetch sar flood-map post_event.tif --reference pre_event.tif --detect-direction both
pygeofetch sar coherence slc_20260601.tif slc_20260613.tif --window 7
```

Full command list: `despeckle`, `calibrate`, `flood-map`,
`coherence`. See {doc}`/processing/sar` — including the honest
calibration-accuracy limitation and the real duplicate-class
architecture note — for full detail.

## `proc-pipeline` — the YAML *processing-chain* pipeline

```{warning}
This is genuinely different from `pygeofetch pipeline` (search →
filter → download → process → export, for a recurring acquisition
job). `proc-pipeline` runs a **chain of processing steps on one
file** — the CLI-accessible form of `ProcessingPipeline` from
{doc}`/reference/pipelines`'s "Python Processing Pipeline" section.
Don't mix the two YAML formats.
```

```bash
pygeofetch proc-pipeline run ndvi_workflow.yaml --input scene.tif --output-dir ./processed/
pygeofetch proc-pipeline validate ndvi_workflow.yaml
pygeofetch proc-pipeline template   # print a starter YAML template
```

`ndvi_workflow.yaml` uses the same step names as the Python builder
(`clip`, `reproject`, `ndvi`, `cog`, etc. — the full list is in
{doc}`/reference/pipelines`):

```yaml
name: ndvi-workflow
steps:
  - clip:
      bbox: [-74.1, 40.6, -73.7, 40.9]
  - reproject:
      crs: EPSG:4326
  - ndvi:
      red: B04
      nir: B08
  - cog: {}
```

## Config commands

```bash
pygeofetch config show
pygeofetch config get download.parallel
pygeofetch config set download.parallel 8
pygeofetch config path
pygeofetch config reset
```

## Shell completion

```bash
# Bash
pygeofetch --install-completion bash

# Zsh / Fish also supported
pygeofetch --install-completion zsh
pygeofetch --install-completion fish
```
