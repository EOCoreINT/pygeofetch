# Changelog

A thematic record of notable changes, compiled from the project's audit history. It is not a
release-by-release log: entries are grouped by area, and the package's version strings are
inconsistent (see [Known issues](../reference/known-issues.md)), so no version numbers are
attached. For commit-level history use the GitHub repository.

## Data access

**Added**

- `PyGeoFetch.stream()` and `pygeofetch.streaming`: read only the pixels under an AOI from Cloud-Optimized GeoTIFFs using HTTP range requests. Archives (Copernicus `.zip`, USGS `.tar`) are refused. See [Streaming](../data-access/streaming.md).
- Optical data validation and preflight (`pygeofetch.validation`): a pre-download quality gate covering AOI coverage, cloud cover, required bands, processing level and temporal bounds, wired into `search()` and `download()` through `validate_optical`. See [Optical validation](../data-access/optical-validation.md).
- `DataOrganizer`: date, orbit, satellite and processing-level hierarchies, a manifest, and `prepare_insar_stack()`. See [Data organizer](../data-access/data-organizer.md).

**Providers rewritten against their real APIs**

- `airbus_oneatlas` (previously non-functional: fake authentication, fictional endpoint, no downloadable assets).
- `noaa_big_data` (now lists public S3 objects instead of calling a fictional REST endpoint).
- `digitalglobe`, `maxar_gbdx` (GBDX was shut down in 2022), `alaska_satellite_facility`, `inpe_cbers`, `jaxa_earth`, `isro_bhuvan`, `earth_explorer_additional`, `geoserver_generic`.
- Targeted fixes in `nasa_earthdata`, `nasa_earthdata_cloud`, `planet` and `sentinel_hub`.
- `esa_scihub` and `google_earth_engine` now fail with an explanatory message instead of crashing on a missing method.

## Reliability and security

- The circuit breaker is now invoked, and provider instances persist so failure counts accumulate. See [Error handling](../reference/error-handling.md#circuit-breaker).
- Default credential storage uses Fernet encryption instead of base64, with automatic migration. See [Security](../reference/security.md).
- Pipeline `process` and `export` steps, previously stubs, now run post-processing actions and transfer files to local disk, S3 or GCS with optional webhook notification. See [Pipelines](../workflows/pipelines.md).

## Processing

**Optical**

- Six geology and mineral-exploration spectral indices, and a fix for `RVI`, which was clipped to a constant by a normalized-difference range. See [Spectral indices](../processing/optical/spectral-indices.md).
- Optical pixel offset tracking, horizontal strain with outlier protection, and coherence-weighted InSAR fusion. See [Optical offset tracking](../processing/sar-insar/optical-offset-tracking.md).

**SAR and InSAR**

- Five standard SAR pipelines in `pygeofetch.sar.pipelines`, and a fix to `pygeofetch.sar.SARProcessor.coherence()` for complex SLC input. The same defect remains in `client.sar`; see [Known issues](../reference/known-issues.md#sar-coherence-fails-on-complex-input).
- InSAR advanced safeguards: geometric layover and shadow masking, custom DEM alignment, and topographic-residual co-estimation in SBAS. See [InSAR](../processing/sar-insar/insar.md).

**Multi-sensor**

- Five pipelines combining two sensors: InSAR with optical displacement, SAR with optical flood mapping, terrain-corrected optical change detection, DEM-of-Difference volume, and NDVI with coherence classification. Also available as `pygeofetch multisensor` commands. See [Multi-sensor pipelines](../processing/multisensor.md).
- Audit fixes: optical and DEM nodata handling, geodesic area for volume on geographic grids, the Sentinel-2 baseline 04.00 offset parameters, class 255 as nodata, and solar position checked against an independent reference (0.19 degrees zenith, 0.49 degrees azimuth).

## Packaging and quality

- Corrected five dependency pins in `pyproject.toml` that referenced versions never published (`whitebox`, `whiteboxgui`, `sidecar`, `contextily`, `opensartoolkit`).
- Resolved seven `mypy` errors in `scheduler.py`, `authenticator.py` and `viz/plot.py`, including a removed `matplotlib.cm.get_cmap` call.

## Documentation

- Restructured into Getting started, Data access, Processing, Workflows, Visualization, Reference and Project, with redirects from the old URLs.
- Switched the theme to Furo and added grid-card navigation.
- The CLI command reference is now generated from the code (`tools/gen_cli_reference.py`).
- Corrected claims that did not match the code: credentials are not read from environment variables; `--resume` does not continue partial files; the Settings layer is not applied at runtime; `extract_archives` is unused; downloads are nested under `<destination>/<provider>/`.
