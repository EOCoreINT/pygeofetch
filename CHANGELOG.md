# Changelog

All notable changes to pygeofetch are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

```{note}
This file was rebuilt from PyPI's actual release history
(https://pypi.org/project/pygeofetch/#history), not just extended from
the previous version of this document. **54 real releases exist**,
not the 5 previously documented. See "Corrections to the previous
changelog" at the bottom for specific discrepancies found and fixed.

PyPI's release-history page lists every version and its real release
date (both used below, directly verified), but does not expose
distinct per-version release notes the way GitHub Releases would.
Detailed notes below are included only where independently confirmed
against that version's own PyPI project-description snapshot; every
other release is listed with its real, verified version number and
date, and marked accordingly rather than filled in with invented
bullet points.
```

## [Unreleased] — v2.6.2.4

```{note}
Not yet published to PyPI. This entry documents work completed and
verified so far, not a finished/closed release -- several providers
flagged during this pass (digitalglobe, earth_explorer_additional,
geoserver_generic, inpe_cbers, isro_bhuvan, jaxa_earth,
alaska_satellite_facility, maxar_gbdx) still need the same real-API
audit and remain open for a follow-up pass.
```

### Fixed
- **Pipeline `process`/`export` steps were stub implementations** —
  logged a message and returned `{"status": "stub"}` without doing
  any real work. Both now delegate to real, tested logic: `process`
  reuses the same action executor `DownloadOptions.post_process` and
  the CLI's `--post-process` flag already use; `export` genuinely
  copies files to local disk, uploads to S3/GCS, and can POST a
  webhook notification on completion.
- **`CircuitBreaker` was dead code** — instantiated per-provider but
  never invoked, and `get_provider()` recreated a fresh provider (and
  therefore a fresh, always-zeroed breaker) on every call, so failure
  state could never have accumulated regardless. Fixed both: provider
  instances are now cached per `FederatedSearcher`, and the real
  `provider.search(...)` call is wrapped in the breaker.
- **Credential file storage used base64, not encryption** — the
  file-backend's own source comment read "Basic obfuscation (not
  encryption)". Replaced with real Fernet symmetric encryption, with
  transparent migration for existing users' previously-stored
  credentials.
- **Airbus OneAtlas provider was entirely non-functional** — fake
  auth (never called a real endpoint), a fictional search endpoint,
  and a result parser that never populated downloadable assets, so
  `download()` always failed regardless of credentials. Fully
  rewritten against Airbus's real, documented OneAtlas Data Living
  Library API.
- **NOAA Big Data provider hit a fictional REST endpoint** — GOES
  imagery is served from real, public S3 buckets, not a search API.
  Rewritten to list real S3 objects and produce genuine downloadable
  hrefs.
- **`esa_scihub` and `google_earth_engine` crashed on every call** —
  both called a method, `_check_integration_verified()`, that does
  not exist anywhere in the codebase. Both now fail honestly instead:
  `esa_scihub`'s real target (Copernicus Open Access Hub) was
  decommissioned in 2023 and now points users to `copernicus`;
  `google_earth_engine` needs a fundamentally different auth/API
  architecture than this file's generic template provided, and now
  says so clearly instead of crashing or silently pretending to work.
- **5 broken dependency pins in `pyproject.toml`** — `whitebox`,
  `whiteboxgui`, `sidecar`, `contextily`, and `opensartoolkit` were
  each pinned to a minimum version that has never existed on PyPI,
  breaking `pip install` for the affected extras. Verified fixed with
  a real `pip install --dry-run` across every extra combined.
- **7 real `mypy` errors** across `core/scheduler.py`,
  `core/authenticator.py`, and `viz/plot.py` — including a return-type
  inconsistency across the 3 pipeline export helpers, an unannotated
  `Fernet | None` attribute, and a genuinely deprecated
  `matplotlib.cm.get_cmap()` call (already removed from newer
  matplotlib type stubs). Fixing the deprecated call unmasked one
  further latent type mismatch (`classification_colors` typed too
  narrowly for what it actually accepts at runtime), also fixed.

### Added
- **New: optical data validation and preflight system**
  (`pygeofetch.validation`) — a configurable pre-download quality gate
  for optical imagery, mirroring `pygeofetch.insar.preflight`'s
  philosophy: `OpticalValidationConfig` (every check independently
  toggleable — AOI coverage, cloud cover, required bands, processing
  level, temporal bounds on by default; snow/ice cover and a
  geometry-only nodata-margin heuristic off by default) and
  `OpticalPreflightValidator` (`validate_scene()` /
  `run_preflight()`), plus a new `OpticalValidationError` exception.
  Accepts real `SatelliteData` objects or plain STAC-like dicts
  interchangeably.
- **Wired into `PyGeoFetch.search()` and `.download()`** via a new
  `validate_optical` toggle — settable as an instance-level default in
  `__init__` and/or overridden per call. `search()` derives the AOI
  automatically from the query's own `bbox`/`geometry`. `download()`
  is a genuinely independent gate (useful for items loaded via
  `download_from_file()` that never went through `search()`); rejected
  items are never attempted, and a synthesized `DownloadResult(status=
  FAILED, ...)` stands in at that item's original position, preserving
  the existing guarantee that the returned list's length and order
  always match the input. The `shapely` dependency this needs is
  imported lazily — a base `pip install pygeofetch` and a default
  `PyGeoFetch()` never require it, only actually enabling validation
  does (verified with `shapely` explicitly blocked from `sys.modules`).
- 108 new tests total across this pass (pipeline steps, circuit
  breaker, credential encryption, Airbus, NOAA, esa_scihub,
  google_earth_engine, the optical validator, and its
  search()/download() wiring) — full suite now **497 passed, 0
  failed**, up from 377 at the start of this pass.
- Full 29-page Read the Docs site, with every code example checked
  against real source rather than the previous marketing copy.

### Documentation
- New page: `core-features/optical-validation.md`.
- `core-features/providers.md` updated with the real, current status
  for `airbus_oneatlas`, `noaa_big_data`, `esa_scihub`, and
  `google_earth_engine`.
- `reference/python-api.md` updated with the `validate_optical` /
  `optical_validation_config` constructor and method parameters.
- `reference/pipelines.md`, `reference/security.md`,
  `reference/error-handling.md`, `reference/roadmap.md`, and
  `reference/testing.md` updated to describe the fixes above as
  resolved (with current, verified test counts) rather than as open
  bugs.

---

## [2.6.2.3] — 2026-08-25 (latest)

Confirmed via direct fetch of the current PyPI listing.

### Highlights
- 24 provider integrations (up from 22+ in earlier releases), including Sentinel-1C/1D constellation support
- Full verified InSAR chain: burst-aware coregistration, real flat-earth/topographic phase removal, ERA5 tropospheric + real IONEX/CDDIS ionospheric correction, SNAPHU phase unwrapping, SBAS time-series inversion — independently verified against synthetic ground truth and the Mexico City / Piton de la Fournaise / Amatrice case studies
- `InSARProject`: high-level search-to-interferogram workflow wrapper
- `PreflightGate`: pre-download AOI coverage, temporal-network, and burst-timing-family validation — avoids wasted downloads before they cost bandwidth
- Real orbit-based coregistration (genuine per-pixel offsets from real orbit state vectors) with automatic fallback to shape-based resampling
- LOS-to-vertical displacement conversion with configurable incidence angle
- Real footprint geometry surfaced where providers supply it, alongside bbox-only fallback where they don't
- DOI-registered software citation via Zenodo

### Corrected from earlier versions (confirmed via source inspection during unrelated maintenance work, not from this PyPI listing)
- `PyGeoFetch` is the real class name — some earlier release READMEs' own Python API examples used a lowercase `pygeofetch()` constructor that does not match the installed package
- `SearchQuery.sort_ascending: bool` — earlier examples showed a `sort_order: "asc"/"desc"` string field that doesn't exist on the model (the CLI's `--sort-order` flag does convert to this internally, so CLI docs were unaffected)
- `DownloadOptions.post_process` requires a list of `PostProcessAction` objects — passing plain strings raises a real `pydantic.ValidationError`

## [2.6.2] — 2026-08-24
Real, verified release. Detailed notes not independently recovered from PyPI's release-history page (no distinct per-version notes exposed for this release specifically).

## [2.6.1] — 2026-08-22
Real, verified release. Detailed notes not independently recovered.

## [2.6.0] — 2026-08-17
Real, verified release. Detailed notes not independently recovered.

## [2.5.0] — 2026-08-17
Real, verified release (same-day release as 2.6.0). Detailed notes not independently recovered.

## [2.4.0] — 2026-08-10
Real, verified release. Detailed notes not independently recovered.

## [2.3.0] — 2026-08-10
Real, verified release (same-day release as 2.4.0). Detailed notes not independently recovered.

## [2.2.0] — 2026-08-10
Real, verified release (same-day release as 2.3.0/2.4.0). Detailed notes not independently recovered.

## [2.1.0] — 2026-08-07
Real, verified release. Detailed notes not independently recovered.

## [2.0.9] — 2026-08-07
## [2.0.8] — 2026-08-07
## [2.0.7] — 2026-08-07
## [2.0.6] — 2026-08-07
## [2.0.5] — 2026-08-07
## [2.0.4] — 2026-08-07
## [2.0.3] — 2026-08-07
## [2.0.2] — 2026-08-07

Eight patch releases in one day (2.0.2 through 2.0.9), all real and
verified against PyPI. Rapid same-day patch sequences like this
typically indicate iterative packaging/CI fixes rather than distinct
feature work — consistent with the version-number pattern, but not
independently confirmed per-patch from PyPI alone.

## [2.0.1] — 2026-08-06
## [2.0.0] — 2026-08-06

Major version bump to 2.0.0, same day as the first 2.0.x patch.
Detailed per-version release notes not independently recovered from
PyPI for either.

## [1.9.9] — 2026-08-06
Real, verified release. Detailed notes not independently recovered.

## [1.9.8] — 2026-08-04
## [1.9.7] — 2026-08-03
## [1.9.6] — 2026-08-03
## [1.9.5] — 2026-08-03

Real, verified releases. Detailed notes not independently recovered.

## [1.9.4] — 2026-08-02
## [1.9.3] — 2026-08-02

Real, verified releases. Detailed notes not independently recovered.

## [1.9.2] — 2026-07-26

Confirmed via direct fetch of this version's own PyPI project page.

### Added
- Federated search across 22+ satellite data providers with deduplicated results
- 7 search output formats: `table`, `json`, `stac`, `geojson`, `geoparquet`, `csv`, `ids`
- Adaptive parallel downloads: band selection, SHA256 checksum verification, resume support, exponential-backoff retries, atomic writes
- Preprocessing engine (`client.preprocess`): `atmos`, `cloud_mask`, `cloud_fill`, `topo_correct`, `clip`, `reproject`, `resample`, `pansharpen`, `tile`, `mosaic`, `composite`
- 15 spectral indices (`client.indices`): ndvi, evi, savi, ndwi, mndwi, ndbi, ndsi, ndmi, nbr/dnbr, tct, pca, texture, lst, albedo, band_math
- Post-processing chain (`client.post`): vectorize → smooth → regularize → zonal_stats → buffer → centroids → compress → cog
- SAR processing (`client.sar`): despeckle, calibrate, flood_map, coherence
- `pygeofetch.insar`: `InterferogramGenerator`, `PhaseUnwrapper` (SNAPHU), `AtmosphericCorrector`, `SBASTimeSeries`, `DataValidator` — real orbit-based coregistration when a DEM + both SAFE archives + both `.EOF` orbit files are supplied, falling back cleanly to shape-based resampling otherwise
- YAML pipeline orchestration with cron scheduling, Python builder API (`client.pipeline(...)`), 6 built-in templates (ndvi, change_detection, flood_map, urban_mapping, sar_analysis, land_cover)
- 9 example notebooks covering install through full processing workflows
- `pygeofetch doctor` installation/connectivity diagnostics

### Known state at this version (superseded by later fixes)
- Python API examples in this version's own README use a lowercase
  `pygeofetch()` constructor (`from pygeofetch import pygeofetch`),
  which doesn't match the real `PyGeoFetch` class name used
  internally by the package at this point in its history
- Extras were `sar` (not yet split from a separate `insar` extra) —
  the base SAR/InSAR dependency layout changed in later releases

## [1.9.1] — 2026-07-25
## [1.9.0] — 2026-07-25
## [1.8.0] — 2026-07-25

Three releases the same day. Real, verified. Detailed notes not
independently recovered.

## [1.7.0] — 2026-07-24
## [1.6.0] — 2026-07-23
## [1.5.0] — 2026-07-23
## [1.4.0] — 2026-07-22
## [1.3.0] — 2026-07-21
## [1.2.0] — 2026-07-18

Real, verified releases. Detailed notes not independently recovered.

## [1.1.0] — 2026-07-13
## [1.0.9] — 2026-07-13

Real, verified releases, same day. Detailed notes not independently
recovered.

## [1.0.8] — 2026-07-12

```{note}
The previous version of this changelog had an entry labeled
**"2.6.2.3 — 2026-07-12"** describing 22+ providers, Sentinel-1C/1D
support, SLC search with provider routing, precise orbit file
management, 17 spectral indices, and various bug fixes. **2.6.2.3 was
only ever released once, on 2026-08-25** — it was never released on
2026-07-12. The date matches this real release, **1.0.8**, exactly.
That content is reproduced below under the version it actually
belongs to.
```

### Added
- Federated search across 22+ satellite data providers
- Sentinel-1C and Sentinel-1D constellation support (active since May/April 2026)
- SLC product type search with automatic provider routing
- Precise orbit file management (POEORB/RESORB) for InSAR workflows
- 17 spectral indices: NDVI, EVI, SAVI, NDWI, MNDWI, NDBI, NDSI, NDMI, NBR, dNBR, TCT, PCA, Texture, LST, Albedo, Band Math, Stack
- SAR processing: despeckle, calibrate, flood mapping, coherence
- 41-step chainable processing pipeline builder
- YAML pipeline definitions with cron scheduling
- Cloud Optimized GeoTIFF output
- Clean search result tables with ANSI-aware column alignment
- Live download progress with Jupyter notebook HTML widget support
- Credential redaction in all log output

### Fixed
- `AuthManager.add_credentials()` now accepts dict form
- `download()` length/order contract guaranteed
- Partial downloads detected via file validation
- CRS identity transform detection after reprojection
- `resolve_band_keys` missing import in `aws_earth` provider
- Band alias conflict between Sentinel-2 and Landsat naming conventions
- Copernicus `product_type` OData filter now correctly applied
- USGS M2M API authentication payload (`authType`, `catalogId` fields)
- ZIP archive validation no longer attempts rasterio open on archives

### Providers
- Copernicus Dataspace (Sentinel-1/2/3/5P, SLC + GRD)
- AWS Earth (Sentinel-2, Landsat via STAC)
- Planetary Computer (Sentinel-2, Landsat, MODIS)
- Element84 Earth Search (STAC)
- USGS Earth Explorer (Landsat 1–9, ASTER, SRTM)
- NASA Earthdata (MODIS, ICESat-2, GEDI)
- Alaska SAR Facility (Sentinel-1 SLC, ALOS PALSAR)
- Planet Labs, Sentinel Hub, Maxar, Airbus, OpenTopography, and more

## [1.0.7] — 2026-07-07
## [1.0.6] — 2026-07-07
## [1.0.5] — 2026-07-07
## [1.0.4] — 2026-07-07

Four releases the same day. Real, verified. Detailed notes not
independently recovered.

## [1.0.3] — 2026-07-06
Real, verified release. Detailed notes not independently recovered.

## [1.0.2] — 2026-06-28

### Added (per the previous version of this changelog, unverified against PyPI for this exact version)
- Batch processing support for multiple scenes
- Parallel download with configurable thread count
- Progress tracking with ETA estimation
- Automatic retry on network failures (3 attempts)
- Checksum verification for downloaded files

### Fixed
- Memory leak in large STAC search results
- Timeout issues with slow API responses
- Incorrect band ordering in Sentinel-2 L2A products
- CRS transformation for MODIS sinusoidal projection

```{note}
The previous changelog attributed this content to a version
**"1.9.2 — 2026-06-15."** No pygeofetch release exists on 2026-06-15,
and 1.9.2 was actually released 2026-07-26 (see above, with different,
independently-verified content). 1.0.2's real release date
(2026-06-28) is the closest real date to the claimed one, and the
package was already well past a "batch processing" stage of maturity
by 1.9.2, so this content most plausibly belongs to an earlier
version -- but this attribution is a best-effort guess, not a
confirmed match like the 1.0.8 correction above. Treat it as
unverified.
```

## [1.0.1] — 2026-06-27
## [1.0.0] — 2026-06-27

The 1.0.0 stable release, same day as 1.0.1. Real, verified. Detailed
notes not independently recovered beyond what's implied by the version
number itself (first release out of beta versioning).

## [0.1.7] — 2026-06-17
## [0.1.6] — 2026-06-13
## [0.1.5] — 2026-06-13
## [0.1.4] — 2026-06-05
## [0.1.3] — 2026-05-28
## [0.1.2] — 2026-05-28

Real, verified beta releases. Detailed notes not independently
recovered.

## [0.1.1] — 2026-05-23

Real, verified release, same day as 0.1.0. Detailed notes not
independently recovered.

## [0.1.0] — 2026-05-23 (initial release)

Confirmed via direct fetch of this version's own PyPI project page —
the first ever release.

### Added
- Initial public beta release (classified `Development Status :: 4 - Beta`)
- Federated search and parallel, resumable downloads across an initial provider set
- 22 provider integrations at launch, including USGS, Copernicus CDSE, NASA Earthdata (+ Cloud), OpenTopography, Planet Labs, Sentinel Hub, Maxar GBDX, Airbus OneAtlas, Alaska Satellite Facility, NOAA Big Data, Google Earth Engine, TerraBotics, AWS Earth, Planetary Computer, Element84, ESA SciHub mirror, JAXA ALOS World, ISRO Bhuvan, INPE CBERS, DigitalGlobe, GeoServer Generic
- YAML pipeline orchestration with cron scheduling
- Post-processing action chain: unzip, reproject, compress, ndvi, ndwi, composite, atmospheric, clip, resample, cog, merge, pan-sharpen
- Docker support (Docker Hub + GitHub Container Registry)
- Full CLI: auth, providers, search, download, cache, pipeline, config, system commands

### Notably absent at this version (added later)
- No `pygeofetch.insar` module yet — the full InSAR chain
  (`InterferogramGenerator`, `PhaseUnwrapper`, `SBASTimeSeries`, etc.)
  does not appear anywhere in this version's own feature list or
  roadmap, and was clearly added in a later release
- Extras were just `geo`, `dev`, `all` — far narrower than the current
  extras layout (`geo`, `processor`, `providers`, `insar`, `sar`,
  `ost`, `viz`, `viz-3d`, `cloud`, `notebook`, `schedule`, `dev`,
  `insar-full`, `full`, `all`)
- Config environment-variable prefix was `SATELLITE_BRIDGE_*` at this
  point (an internal project codename), later changed to
  `PYGEOFETCH_*`
- This version's own roadmap (v0.2.0/v0.3.0/v1.0.0 targets) planned
  BlackSky and KOMPSAT providers, a web dashboard, and a hosted cloud
  offering — none of which appear in later versions' feature lists,
  suggesting the roadmap was substantially re-prioritized toward the
  InSAR chain and provider-count expansion that actually shipped

---

## Corrections to the previous changelog

The version of this file uploaded for this update had **5 entries
covering 5 dates**. PyPI's real release history has **54 releases**.
Specific issues found and fixed:

1. **Two different entries were both labeled `2.6.2.3`**, dated
   2026-07-29 and 2026-07-12. **2.6.2.3 was only ever released once**,
   on 2026-08-25 (confirmed directly from the live PyPI listing).
2. **The 2026-07-12 entry's date exactly matches a real release**
   — version **1.0.8** — and its content (22+ providers, Sentinel-1C/D,
   SLC search, orbit files, 17 indices) is consistent with that point
   in the project's real history. Reassigned to 1.0.8 above with high
   confidence.
3. **The 2026-07-29 entry doesn't match any real release date at
   all** (no pygeofetch version was ever released on that date — the
   nearest real dates are 2026-07-26 and 2026-08-02). Its own
   "Migration Guide" section internally references "v2.4.0" as a
   *before* state and "v1.10.0" as the *after* state, but the real
   version sequence has 2.4.0 releasing *later* (2026-08-10) than
   1.9.x, and no version 1.10.0 exists anywhere in the real history —
   so this entry's version references are internally inconsistent
   with the real release sequence, not just misdated. It could not be
   confidently reassigned to a specific real version and is **omitted
   above** rather than attached to a guess. If you have the original
   source (a GitHub release, a commit, an internal doc) for this
   SNAPHU/urban-InSAR unwrapping work, it should be re-added under
   whichever real version actually shipped it.
4. **The "1.9.2 — 2026-06-15" entry** doesn't match either: no release
   exists on that date, and the real 1.9.2 (2026-07-26) has a fully
   different, independently-verified feature set (see above). Its
   content was tentatively reassigned to 1.0.2 (2026-06-28, the
   closest real date) above, but flagged as an unverified guess rather
   than a confirmed correction.
5. **"1.9.1 — 2026-05-20" and "1.9.0 — 2026-04-25"** in the previous
   file also don't match real dates for those version numbers (real
   1.9.1 released 2026-07-25; there is no pygeofetch release at all
   on 2026-04-25 — that predates even the real first release,
   2026-05-23). This content wasn't reassigned anywhere, for the same
   reason as #3: no confident match to a real version was found.

**49 of the 54 real releases have no independently-verified detailed
notes** in this file — PyPI's release-history page doesn't expose
per-version notes, and confirming each one would require fetching and
diffing all 54 individual version snapshots. The version numbers and
dates for all 54 are real and verified; only the narrative content is
incomplete. If per-version detail matters going forward, the fix is
process, not research: tag real GitHub releases with real notes at
release time, so this file (or an auto-generated one) has a genuine
source to pull from instead of reconstructing after the fact.