# Core concepts

A short tour of the ideas the rest of the documentation assumes.

## The big picture

```text
          SearchQuery                        SatelliteData (scene records)
 you ───────────────────►  PyGeoFetch  ───────────────────────────────────►  you
                          │   client   │
                          │            │  search()   one query, many providers, in parallel
                          │            │  download() whole files, with retry and post-processing
                          │            │  stream()   only your AOI from Cloud-Optimized GeoTIFFs
                          └─────┬──────┘
                                │ provider adapters (22 registered)
          ┌──────────┬──────────┼───────────┬────────────┐
       Copernicus  USGS   Planetary Computer  ASF  …  OpenTopography
```

pygeofetch has three layers:

1. **Data access.** `search()` queries providers; `download()` and `stream()` retrieve the data.
2. **Processing.** Optical, SAR, InSAR and multi-sensor processing that works on the files you retrieved.
3. **Orchestration.** YAML pipelines that chain search, download, process and export, optionally on a schedule.

## Providers

A **provider** is an adapter for one data source. Each translates a `SearchQuery` into that
source's API and returns results as `SatelliteData`. Providers differ in what they serve and how:

| Property | Examples |
|---|---|
| **Needs credentials** | `copernicus`, `usgs`, `planet` need them; `planetary_computer`, `element84` do not. |
| **STAC or not** | STAC providers (`planetary_computer`, `element84`, `sentinel_hub`) share a data model. Others use their own APIs. |
| **How data is packaged** | One `.SAFE.zip` per scene (`copernicus`), one `.tar` (`usgs`), or one Cloud-Optimized GeoTIFF per band (`planetary_computer`). Packaging decides whether you can [stream](../data-access/streaming.md). |

See [Providers](../data-access/providers.md) for the full list.

## The data model

| Class | Role |
|---|---|
| `SearchQuery` | What you want: area, dates, cloud cover, satellites, sorting. |
| `SatelliteData` | One scene: ID, provider, time, footprint, cloud cover, and its `assets`. |
| `SatelliteAsset` | One file of a scene (a band, a metadata file, an archive), with its `href`. |
| `DownloadOptions` | How to download: parallelism, retries, bands, post-processing. |
| `DownloadResult` | The outcome for one scene: status, output paths, errors. |

All are Pydantic models in `pygeofetch.models`. See the [Python API reference](../reference/python-api.md).

## Conventions

- **Bounding boxes** are `(min_lon, min_lat, max_lon, max_lat)` in EPSG:4326 — longitude first. The CLI takes the same order as `"min_lon,min_lat,max_lon,max_lat"`.
- **Dates** are `YYYY-MM-DD` on the CLI and `date`, `datetime` or ISO strings in Python.
- **Cloud cover** is a percentage from 0 to 100.
- **Downloads land in provider sub-folders:** `<destination>/<provider>/...`. See [Downloading](../data-access/download.md#where-files-are-written).
- **Archives are not extracted automatically.** Use the `unzip` post-processing action.

## Whole scene, band subset, or AOI window

Optical scenes are large, and you usually need a small part of them. pygeofetch offers three
levels of data reduction:

| Approach | Transfers | Use when |
|---|---|---|
| `download()` | The complete asset(s) | You need the original product, or the provider serves archives. |
| `download(bands=[...])` | Only the chosen band files | The provider serves one file per band. |
| `stream(bbox=...)` | Only the pixels under your AOI | The provider serves Cloud-Optimized GeoTIFFs and you only need a small area. |

## Data formats you will meet

- **GeoTIFF / COG.** Raster data. A *Cloud-Optimized* GeoTIFF is internally tiled so a client can fetch just part of it over HTTP.
- **SAFE.** The Sentinel product folder, usually delivered zipped.
- **SLC and GRD.** Sentinel-1 *Single Look Complex* (phase preserved; needed for InSAR) and *Ground Range Detected* (amplitude only).
- **STAC.** A standard JSON catalogue format for spatiotemporal assets, used by many providers.
- **GeoParquet and Zarr.** Cloud-native vector and array formats, available through the `pygeofetch stream` conversion commands.

## Four things called "pipeline"

The word is used for four different features:

| Feature | What it chains | Where |
|---|---|---|
| **YAML acquisition pipeline** (`pygeofetch pipeline run`) | search, filter, download, process, export; can be scheduled | [Pipelines](../workflows/pipelines.md) |
| **Fluent processing pipeline** (`client.pipeline(...)`) | clip, reproject, index, COG on files you already have | [Pipelines](../workflows/pipelines.md) |
| **SAR pipelines** (`pygeofetch.sar.pipelines`) | calibrate, despeckle, flood or change detection on one sensor over several dates | [SAR](../processing/sar-insar/sar.md) |
| **Multi-sensor pipelines** (`pygeofetch.multisensor`) | two different sensors combined | [Multi-sensor pipelines](../processing/multisensor.md) |

## Where to go next

- New to the tool: [Quick start](quickstart.md).
- Want only your study area: [Streaming](../data-access/streaming.md).
- Doing deformation monitoring: [InSAR](../processing/sar-insar/insar.md).
