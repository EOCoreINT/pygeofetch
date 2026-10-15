# Command-line guide

The `pygeofetch` command exposes search, download, processing and pipeline
orchestration. This page covers conventions and worked examples. The exact
options of every command are in the [Command reference](cli-commands.md), which is
generated from the installed package.

## Conventions

```text
pygeofetch [GLOBAL OPTIONS] GROUP COMMAND [ARGUMENTS] [OPTIONS]
```

Most commands are grouped (`pygeofetch search run`, `pygeofetch index ndvi`).
A few are single commands (`pygeofetch doctor`, `pygeofetch status`).
`--help` works at every level:

```bash
pygeofetch --help
pygeofetch index --help
pygeofetch index ndvi --help
```

### Global options

| Option | Default | Description |
|---|---|---|
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. |
| `--log-file` | none | Also write logs to this file. |
| `--log-format` | `console` | `console` or `json` (for log shippers). |
| `--config` | none | Path to a configuration YAML that overrides user defaults. |
| `--version` | | Print the version and exit. |

Global options go **before** the group name:

```bash
pygeofetch --log-level DEBUG --log-format json search run --bbox "-74.1,40.6,-73.7,40.9"
```

### Argument conventions

- **Bounding boxes** are `"min_lon,min_lat,max_lon,max_lat"` in EPSG:4326. Quote the value so a leading minus sign is not read as a flag.
- **Dates** are `YYYY-MM-DD`.
- **Output paths** are given with `--output` / `-o`. Commands that process a raster write a new file and never modify the input.
- **Raster inputs** are positional arguments. Band inputs for indices are named options (`--red`, `--nir`, ...) that match the Python keyword arguments.

## Where each group is documented

| Group | Purpose | Guide |
|---|---|---|
| `auth` | Store and test provider credentials | [Authentication](../data-access/authentication.md) |
| `providers` | List and inspect providers | [Providers](../data-access/providers.md) |
| `search` | Federated search | [Searching](../data-access/search.md) |
| `download` | Download, status and history | [Downloading](../data-access/download.md) |
| `stream` | Windowed COG reads, Zarr and GeoParquet conversion | [Streaming](../data-access/streaming.md) |
| `cache` | Search-cache management | [Configuration](configuration.md) |
| `pipeline` | YAML acquisition pipelines and scheduling | [Pipelines](../workflows/pipelines.md) |
| `proc-pipeline` | YAML processing chains | [Pipelines](../workflows/pipelines.md) |
| `preprocess` | Clip, reproject, mask, mosaic | [Preprocessing](../processing/optical/preprocessing.md) |
| `index` | Spectral indices and band math | [Spectral indices](../processing/optical/spectral-indices.md) |
| `classify` | K-means, Random Forest and SVM | [Classification](../processing/optical/classification.md) |
| `post` | Vectorize, zonal statistics, COG | [Post-processing](../processing/optical/postprocessing.md) |
| `sar` | Despeckle, calibrate, flood map, coherence | [SAR](../processing/sar-insar/sar.md) |
| `optical` | Pixel-offset tracking | [Optical offset tracking](../processing/sar-insar/optical-offset-tracking.md) |
| `multisensor` | Five multi-sensor pipelines | [Multi-sensor](../processing/multisensor.md) |
| `config` | Inspect and edit configuration | [Configuration](configuration.md) |
| `status`, `doctor`, `version` | Diagnostics | [Diagnostics](../getting-started/diagnostics.md) |
| `geocode`, `reverse-geocode` | Place name to coordinates and back | [Geocoding](../data-access/geocoding.md) |
| `assess-feasibility` | Check whether an InSAR project is viable | [InSAR](../processing/sar-insar/insar.md) |

## Search and download

```bash
# Search two providers and save the result as GeoJSON
pygeofetch search run \
  --bbox "-74.1,40.6,-73.7,40.9" \
  --start-date 2024-01-01 --end-date 2024-03-01 \
  --cloud-cover 0-10 \
  --providers aws_earth,planetary_computer \
  --output results.geojson

# Download the first three scenes from that file
pygeofetch download run --from-search results.geojson --output ./data/ --max-items 3

# Inspect what is on disk, and past runs
pygeofetch download status ./data/
pygeofetch download history
```

## Preprocessing

```bash
pygeofetch preprocess clip scene.tif --bbox "-74.1,40.6,-73.7,40.9" --output clipped.tif
pygeofetch preprocess reproject scene.tif --crs EPSG:4326 --output reprojected.tif
pygeofetch preprocess cloud-mask scene.tif --method scl --scl-band SCL.tif
pygeofetch preprocess tile scene.tif --tile-size 256 --overlap 32
```

`pansharpen` takes the **panchromatic image first, then the multispectral image**:

```bash
pygeofetch preprocess pansharpen pan.tif multispectral.tif --method brovey --output sharpened.tif
```

## Spectral indices

```bash
pygeofetch index ndvi --red B04.tif --nir B08.tif --output ndvi.tif
pygeofetch index evi --blue B02.tif --red B04.tif --nir B08.tif --output evi.tif
```

`band-math` takes the input rasters as positional arguments, and addresses them in
the expression as `B[0]`, `B[1]`, ... in the order given:

```bash
pygeofetch index band-math B04.tif B08.tif \
  --expr "(B[1]-B[0])/(B[1]+B[0]+1e-6)" \
  --output ndvi_custom.tif
```

Add `--chunked` to process large rasters (for example a drone orthomosaic) in tiles
instead of loading them whole.

## Post-processing

```bash
pygeofetch post vectorize classification.tif --threshold 0.5 --min-area 100 --output classes.geojson
pygeofetch post zonal-stats ndvi.tif parcels.geojson --stats mean,median,std --output stats.csv
pygeofetch post cog scene.tif --compress deflate
```

## SAR

```bash
pygeofetch sar despeckle sentinel1_vv.tif --filter enhanced_lee --window 5
pygeofetch sar calibrate sentinel1_dn.tif --output-type sigma0 --db
pygeofetch sar flood-map post_event.tif --reference pre_event.tif --threshold -15
pygeofetch sar coherence slc_20260601.tif slc_20260613.tif --window 7
```

`sar calibrate` writes decibels by default; pass `--linear` for linear power.
`sar coherence` currently fails on complex SLC input (see [Known issues](known-issues.md#sar-coherence-fails-on-complex-input)).
`sar flood-map` uses a fixed backscatter threshold (dB) on its own, and change
detection against the pre-event image when `--reference` is given.

## Optical offset tracking

```bash
pygeofetch optical offset-track \
  --ref sentinel2_2023-01-01_B08.tif \
  --sec sentinel2_2023-06-01_B08.tif \
  --output offsets.tif \
  --window-size 64 --step-size 16 --snr-threshold 3.0 \
  --cloud-mask sentinel2_2023-06-01_SCL.tif
```

The output is a three-band GeoTIFF: `dx`, `dy` (ground metres) and `snr`.

## Multi-sensor pipelines

```bash
pygeofetch multisensor flood-map \
  --sar sentinel1_flood.tif --optical-green B03.tif --optical-nir B08.tif \
  --output-dir ./flood --fusion-mode cloud_aware

pygeofetch multisensor dem-diff \
  --dem-old srtm_2015.tif --dem-new uav_dem_2024.tif \
  --output-dir ./dem_diff --old-rmse 8.0 --new-rmse 0.15
```

Each command prints the pipeline's metadata as JSON on success, or a specific error
naming the failed check. See [Multi-sensor pipelines](../processing/multisensor.md) for
the full set of commands and what each option controls.

## Windowed reads and format conversion

```bash
# Read only a bounding box from a remote Cloud-Optimized GeoTIFF
pygeofetch stream cog-window https://example.com/scene_B04.tif \
  --bbox "-1.80,6.10,-1.70,6.20" --output aoi_B04.tif
```

`pygeofetch stream` is for a single COG URL you already know. To stream whole scenes
returned by a search, use `PyGeoFetch.stream()` — see [Streaming](../data-access/streaming.md).

## Processing-chain pipelines

`pygeofetch proc-pipeline` runs a YAML chain of processing steps on one file. It is
distinct from `pygeofetch pipeline`, which orchestrates search, download and export.

```bash
pygeofetch proc-pipeline template            # print a starter YAML
pygeofetch proc-pipeline validate ndvi_workflow.yaml
pygeofetch proc-pipeline run ndvi_workflow.yaml --input scene.tif --output-dir ./processed/
```

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

## Configuration and cache

```bash
pygeofetch config show
pygeofetch config get download.parallel
pygeofetch config set download.parallel 8
pygeofetch config path
pygeofetch config reset

pygeofetch cache stats
pygeofetch cache ttl show
pygeofetch cache ttl set 7200
pygeofetch cache location
pygeofetch cache prune --max-size 1GB
pygeofetch cache clear
```

## Shell completion

```bash
pygeofetch --install-completion bash   # also: zsh, fish
```

This prints the line to add to your shell start-up file (for example `~/.bashrc`);
it does not edit the file for you.

## Known gaps

- The `monitor` command group (`monitor run`, `monitor history`) is implemented in
  `cli/monitor_commands.py` but is not registered, so `pygeofetch monitor` is not
  available. See [Known issues](known-issues.md).
- The five SAR pipelines in `pygeofetch.sar.pipelines` (for example `flood_mapping_pipeline`)
  are available from Python only; they have no CLI command.
