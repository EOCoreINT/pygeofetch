<div align="center">

[![PyPI version](https://badge.fury.io/py/pygeofetch.svg)](https://pypi.org/project/pygeofetch/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pygeofetch.svg)](https://pypi.org/project/pygeofetch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22087230.svg)](https://doi.org/10.5281/zenodo.22087230)
[![Tests](https://github.com/EOCoreINT/pygeofetch/actions/workflows/tests.yml/badge.svg)](https://github.com/EOCoreINT/pygeofetch/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/appiahkubis14/pygeofetch/graph/badge.svg?token=2PS2A2VZL6)](https://codecov.io/gh/appiahkubis14/pygeofetch)

**A unified satellite data and geospatial processing platform — one CLI, one Python API, 24 providers, and a fully verified InSAR chain.**

[Quick Start](#-quick-start) · [Mexico City Case Study](#-case-study-mapping-mexico-city-land-subsidence-with-insar) · [Case Study Repo ↗](https://github.com/EOCoreINT/mexico-subsidence-project) · [Documentation](https://appiahkubis14.github.io/pygeofetch-docs/) · [Notebooks](#-notebooks)

<br>
<img src="icon/concept_a.png" alt="PyGeoFetch Logo" width="350">
<br>

</div>

---

## Overview

pygeofetch is a production-ready framework for acquiring and processing Earth observation data. It provides authenticated, unified access to **24 satellite repositories** — Sentinel, Landsat, Planet, Maxar, Airbus, Copernicus, USGS, NASA, JAXA, and more — through a single CLI and Python API, and layers a complete geospatial processing engine on top.

Where pygeofetch goes further than most data-access libraries is in its **InSAR chain**: burst-aware coregistration, real flat-earth and topographic phase removal, ERA5 tropospheric and ionospheric correction, phase unwrapping, and SBAS time-series inversion — each stage independently verified against synthetic ground truth or real, published deformation studies, not assumed correct from theory alone. The [Mexico City case study](#-case-study-mapping-mexico-city-land-subsidence-with-insar) below walks through that chain end to end on real Sentinel-1 data.

**Core capabilities:**

| | |
|---|---|
| 🔐 **Authenticated access** | 24 providers, credentials stored via system keyring (Keychain / Credential Manager / Secret Service) |
| 🔍 **Federated search** | One query across all providers → STAC 1.0 GeoJSON / GeoParquet / CSV, with real footprint geometry where the provider supplies it |
| 📥 **Resilient downloads** | Parallel, resumable, checksum-verified, band-selective, atomic writes |
| ⚙️ **Preprocessing** | Atmospheric correction, cloud masking, reprojection, resampling, pan-sharpening, mosaicking |
| 📊 **17 spectral indices** | NDVI, EVI, SAVI, NDWI, MNDWI, NDBI, LST, Albedo, dNBR, GLCM texture, and more |
| 🌐 **Verified InSAR chain** | Coregistration → phase removal → atmospheric/ionospheric correction → unwrapping → SBAS inversion |
| 📋 **YAML pipelines** | Chainable, schedulable, repeatable workflows with full run history |

---

<p align="center">
  <img src="https://raw.githubusercontent.com/EOCoreINT/pygeofetch/refs/heads/main/docs/static/download.png" width="48%" />
  <img src="https://raw.githubusercontent.com/EOCoreINT/pygeofetch/refs/heads/main/docs/static/download%20(1).png" width="48%" />
</p>
<p align="center"><em>NDVI trend (2018–2024) and severity classification for the Obuasi Municipal District, Ghana — computed end-to-end with pygeofetch from boundary-clipped USGS Landsat data.</em></p>

Across 2018–2024, 32.0% of the district shows measurable vegetation decline (9.2% strong, 22.8% moderate), against 57.5% stable and 10.5% increasing. The decline is spatially concentrated west of −1.70° longitude, consistent with the district's documented small-scale mining activity — though an NDVI trend alone can't isolate cause from correlated signals like logging or agricultural clearing without ground verification. The eastern two-thirds of the district is overwhelmingly stable, which is itself informative: it suggests a localized driver rather than a district-wide seasonal artifact.

---

## Why pygeofetch

Satellite data access is fragmented — every provider has its own auth scheme, query API, and file format, and most tools that unify search stop well short of processing. pygeofetch is the only package that combines broad provider coverage with a full processing engine and a genuinely verified InSAR chain in one place.

| Feature | pygeofetch | EODAG | pystac-client | satpy | sentinelsat |
|---|---|---|---|---|---|
| Providers | **24** | 10+ | STAC only | Limited | Sentinel only |
| Processing engine | ✅ Full | ❌ | ❌ | Partial | ❌ |
| Spectral indices | ✅ 17+ | ❌ | ❌ | ❌ | ❌ |
| Full InSAR chain | ✅ Verified | ❌ | ❌ | ❌ | ❌ |
| Real footprint geometry | ✅ | ❌ | ✅ (STAC only) | ❌ | ❌ |
| YAML pipelines + scheduling | ✅ | ❌ | ❌ | ❌ | ❌ |
| Commercial providers | ✅ Planet/Maxar | ❌ | ❌ | ❌ | ❌ |

---

## Providers

**11 open-access providers**, no login required — including `planetary_computer`, `aws_earth`, `element84`, `noaa_big_data`, `esa_scihub`, `jaxa_earth`, and `geoserver_generic` for any OGC WMS/WFS/WCS endpoint.

**13 authenticated providers** — USGS, Copernicus CDSE, NASA Earthdata (+ Cloud), Alaska SAR Facility, OpenTopography, Planet Labs, Sentinel Hub, Maxar GBDX, Airbus OneAtlas, Google Earth Engine, TerraBotics, and Earth Explorer.

Every provider has been directly verified to return at minimum a correct bounding box; several are further confirmed to return real, precise footprint geometry against a live API response (marked in `pygeofetch providers info PROVIDER`). Run `pygeofetch providers list` for the full, current table.

---

## Installation

```bash
pip install pygeofetch                 # core — free providers work immediately
pip install "pygeofetch[geo]"          # + rasterio, geopandas, shapely
pip install "pygeofetch[insar]"        # + native SBAS inversion, phase unwrapping
pip install "pygeofetch[insar-full]"   # + For Linux Users
pip install "pygeofetch[all]"          # everything
```

Requires Python 3.9+. Run `pygeofetch doctor` to verify your install and provider connectivity.

---

## ⚡ Quick Start

```python
from pygeofetch import PyGeoFetch
from pygeofetch.models.search_query import SearchQuery, BoundingBox
from pygeofetch.models.download_task import DownloadOptions

client = PyGeoFetch()
client.add_credentials("copernicus", username="you@example.com", password="...")

results = client.search(
    SearchQuery(bbox=BoundingBox.from_string("-74.1,40.6,-73.7,40.9"),
                start_date="2024-01-01", cloud_cover_max=20),
    providers=["copernicus", "planetary_computer", "aws_earth"],
)

downloads = client.download(results[:5], destination="./data/",
                             options=DownloadOptions(parallel=4, bands=["B02", "B03", "B04"]))

ndvi = client.indices.ndvi(red="B04.tif", nir="B08.tif")
```

The equivalent is available as a CLI (`pygeofetch search run ...` / `pygeofetch download run ...`) and as chainable, schedulable YAML pipelines — see the [full CLI reference](#cli-reference) and [notebooks](#-notebooks).

---

## 🌐 Case Study: Mapping Mexico City Land Subsidence with InSAR

**Full project, notebook, and data:** [github.com/EOCoreINT/mexico-subsidence-project](https://github.com/EOCoreINT/mexico-subsidence-project)

Mexico City sits on a drained lakebed and is subsiding at some of the fastest rates on Earth as its aquifer is depleted. It's a demanding, real-world test for an InSAR chain: a huge, largely incoherent urban scene, a fragmented acquisition history, and a deformation signal large enough to make phase unwrapping genuinely hard. What follows is a condensed walkthrough of the [full project above](https://github.com/EOCoreINT/mexico-subsidence-project) — the pygeofetch InSAR chain run end to end against real Sentinel-1 data over Iztapalapa, no synthetic shortcuts. For the complete methodology, every intermediate number, and an unusually candid validation discussion, read the project README itself.

**1. Authenticated federated search.** `client.search()` against Copernicus Data Space for SLC scenes over the city (bbox `-99.183, 19.278, -99.003, 19.438`, July 2016 – September 2017) returns a stack of Sentinel-1A/1B candidates in seconds, filterable straight down to a single consistent track.

<p align="center"><img src="https://raw.githubusercontent.com/EOCoreINT/mexico-subsidence-project/main/analysis/data-search.png" width="85%" /></p>

Results carry real footprint geometry and drop straight onto a map via `MapViewer`, with hover info for scene ID, date, satellite, and provider — useful for spotting swath overlap before committing to a download.

<p align="center">
  <img src="https://raw.githubusercontent.com/EOCoreINT/mexico-subsidence-project/main/analysis/search_1.png" width="48%" />
  <img src="https://raw.githubusercontent.com/EOCoreINT/mexico-subsidence-project/main/analysis/search.png" width="48%" />
</p>

**2. DEM acquisition.** A matching search against OpenTopography returns seven DEM products for the AOI; pygeofetch downloads and clips the selected one (SRTM 30 m) automatically for topographic phase removal.

<p align="center"><img src="https://raw.githubusercontent.com/EOCoreINT/mexico-subsidence-project/main/analysis/dem_acquisition.png" width="85%" /></p>

**3. Preflight validation.** Before anything downloads, the `PreflightGate` checks search truncation, AOI coverage, temporal network connectivity, and burst-timing family compatibility from lightweight annotation XMLs — catching acquisitions that would waste compute before they cost bandwidth. This run avoided roughly 360 GB of downloads by excluding 48 of 115 candidate scenes at this stage alone.

<p align="center"><img src="https://raw.githubusercontent.com/EOCoreINT/mexico-subsidence-project/main/analysis/preflight-log.png" width="85%" /></p>

Real orbit state vectors then pre-filter the candidate pair list against Sentinel-1's own mission thresholds, rejecting pairs for excess burst-sync error, temporal baseline, or spatial baseline before any interferogram is attempted.

<p align="center"><img src="https://raw.githubusercontent.com/EOCoreINT/mexico-subsidence-project/main/analysis/pre-filter-for-interferogram.png" width="85%" /></p>

**4. Interferogram formation.** Of 2,211 theoretically possible pairs across 67 downloaded scenes, real orbit-based coregistration, per-burst Enhanced Spectral Diversity, deburst, and flat-earth/topographic phase removal reduce that to 78 valid pairs — the dense urban fringe pattern below is raw wrapped phase, before any correction.

<p align="center"><img src="https://raw.githubusercontent.com/EOCoreINT/mexico-subsidence-project/main/analysis/wrapped_phase.png" width="85%" /></p>

**5. Coherence.** Amplitude and coherence are estimated together for every pair; the urban core holds noticeably higher coherence (0.614) than the scene-wide average (0.508) — exactly the contrast expected between stable built structures and vegetated or agricultural surrounds.

<p align="center"><img src="https://raw.githubusercontent.com/EOCoreINT/mexico-subsidence-project/main/analysis/comparism.png" width="85%" /></p>

**6. Phase unwrapping.** SNAPHU (Chen & Zebker, 2001) resolves each wrapped interferogram into continuous phase, both as a raw array and reprojected onto the real city footprint. Reliable coverage is intentionally sparse (~0.1–0.2%) — dense vegetation cover means most of the scene decorrelates between 18–24 day passes, so the pipeline marks those pixels NaN rather than interpolating across them.

<p align="center">
  <img src="https://raw.githubusercontent.com/EOCoreINT/mexico-subsidence-project/main/analysis/unwrapped_phase.png" width="48%" />
  <img src="https://raw.githubusercontent.com/EOCoreINT/mexico-subsidence-project/main/analysis/unwrap.png" width="48%" />
</p>

**7. SBAS time-series inversion.** The full 58-date descending stack, once low-coherence and unreliable-reference-pixel pairs are excluded, fractures into 17 disconnected network "islands" — a genuine finding about this AOI's data quality, not a bug. The largest connected island (10 dates, 11 pairs) carries the final inversion, referenced near Cerro de la Estrella.

<p align="center"><img src="https://raw.githubusercontent.com/EOCoreINT/mexico-subsidence-project/main/analysis/1.png" width="85%" /></p>

The resulting velocity field, uncertainty map, reliability mask, and displacement time series recover a 2nd–98th-percentile vertical velocity of **−35.5 to −2.9 cm/year** where reliable pixels exist.

<p align="center"><img src="https://raw.githubusercontent.com/EOCoreINT/mexico-subsidence-project/main/analysis/island1_sbas_full_analysis.png" width="90%" /></p>

That result is **directionally consistent** — same sign, same order of magnitude — with the peer-reviewed peak of −39.1 cm/year reported for Iztapalapa by Cigna & Tapete (2021, *Remote Sensing of Environment*), a study built from 300+ scenes across six years. It is explicitly *not* a validated replication of that study: this run used a fraction of the data volume, and the project's own README is deliberately careful not to overclaim the comparison — worth reading in full if you want the honest version of what this number does and doesn't prove.

The full, runnable notebook — real search and download throughout, no synthetic substitutions — is [`mexico_city_full_confirmed (2).ipynb`](<https://github.com/EOCoreINT/mexico-subsidence-project/blob/main/mexico_city_full_confirmed%20(2).ipynb>) in the [mexico-subsidence-project](https://github.com/EOCoreINT/mexico-subsidence-project) repo; see also `piton_fournaise_full.ipynb` and `amatrice_full.ipynb` in [`notebooks/`](notebooks/) for the same chain applied to the 2021 Piton de la Fournaise eruption and the 2016 Amatrice earthquake.

---

## Processing Engine

**Preprocessing** (`client.preprocess`): `atmos` (DOS1/DOS2, Sen2Cor, FLAASH, 6S, iCOR) · `cloud_mask` · `cloud_fill` · `topo_correct` · `clip` · `reproject` · `resample` · `pansharpen` · `tile` · `mosaic` · `composite`

**Spectral indices** (`client.indices`): `ndvi` · `evi` · `savi` · `ndwi` · `mndwi` · `ndbi` · `ndsi` · `ndmi` · `nbr` / `dnbr` · `tct` · `pca` · `texture` (GLCM) · `lst` · `albedo` · `band_math`

**Post-processing** (`client.post`): `vectorize` → `smooth` → `regularize` → `zonal_stats` → `buffer` → `centroids` → `compress` → `cog`

**SAR** (`client.sar`): `despeckle` (Lee, Enhanced Lee, Frost, Gamma MAP) · `calibrate` (σ0/γ0/β0) · `flood_map` · `coherence`

**InSAR** (`pygeofetch.insar`): `SLCExtractor` · `InterferogramGenerator` · `AtmosphericCorrector` (elevation-correlated or real ERA5) · `IonosphericCorrector` (real IONEX/CDDIS) · `PhaseUnwrapper` (SNAPHU) · `SBASTimeSeries` · `DataValidator` · `InSARProject` (high-level, search-to-interferogram wrapper)

Full method signatures and options are in the [documentation](https://appiahkubis14.github.io/pygeofetch-docs/) and [`pygeofetch/insar/README.md`](pygeofetch/insar/README.md).

---

## YAML Pipelines

```yaml
name: weekly-sentinel2-ndvi
schedule: "0 6 * * 1"
steps:
  - search: {providers: [copernicus, aws_earth], date_range: last_7_days, cloud_cover: "0-10"}
  - filter: {expression: "data.cloud_cover < 5"}
  - download: {parallel: 4, output: ./raw/, bands: [B04, B08]}
  - ndvi: {red: B04.tif, nir: B08.tif}
  - vectorize: {threshold: 0.3, format: geojson}
  - cog: {compress: deflate}
```

```bash
pygeofetch proc-pipeline run weekly-sentinel2.yaml --input scene.tif
pygeofetch pipeline schedule weekly-sentinel2.yaml --name ndvi-weekly
```

Six built-in templates ship out of the box: `ndvi` · `change_detection` · `flood_map` · `urban_mapping` · `sar_analysis` · `land_cover`.

---

## CLI Reference

Every capability above is also exposed as a CLI command, grouped by area:

| Group | Example | Purpose |
|---|---|---|
| `auth` | `pygeofetch auth add copernicus --username U --password P` | Manage per-provider credentials |
| `providers` | `pygeofetch providers list --capabilities sar` | Discover and inspect providers |
| `search` | `pygeofetch search run --bbox "..." --providers aws_earth,copernicus` | Federated search, 7 output formats |
| `download` | `pygeofetch download run --from-search results.geojson --parallel 4` | Parallel, resumable downloads |
| `preprocess` | `pygeofetch preprocess clip scene.tif --bbox "..."` | Atmospheric correction, clipping, reprojection, etc. |
| `index` | `pygeofetch index ndvi --red B04.tif --nir B08.tif` | Spectral index computation |
| `post` | `pygeofetch post cog scene.tif --compress deflate` | Vectorization, zonal stats, COG conversion |
| `sar` | `pygeofetch sar coherence slc1.tif slc2.tif` | Despeckling, calibration, coherence, flood mapping |
| `proc-pipeline` / `pipeline` | `pygeofetch pipeline schedule weekly.yaml` | YAML pipeline execution and scheduling |
| `cache` / `config` | `pygeofetch cache prune --max-size 1GB` | Cache and configuration management |

Run `pygeofetch --help` or any subcommand with `--help` for the complete, current option list, or see the [CLI reference docs](https://appiahkubis14.github.io/pygeofetch-docs/).

---

## 📚 Citation

If you use PyGeoFetch in your research or operational work, please cite it using the following:

### Software Citation
```bibtex
@software{pygeofetch_2026,
  author       = {Appiah Kubi, Samuel},
  title        = {PyGeoFetch: A Unified Python Framework for Multi-Provider Satellite Data Acquisition, Pre-download Quality Control, and Geospatial Processing},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {2.6.2.1},
  doi          = {10.5281/zenodo.22087230},
  url          = {https://doi.org/10.5281/zenodo.22087230}
}
```

## Documentation

Full documentation: **https://appiahkubis14.github.io/pygeofetch-docs/** — CLI reference, provider auth guides, pipeline configuration, post-processing catalogue, and InSAR verification methodology.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Good first issues include extending real footprint geometry to remaining bbox-only providers, wiring stub providers to full API integrations, and adding new post-processing actions.

```bash
git clone git@github.com:EOCoreINT/pygeofetch.git && cd pygeofetch
pip install -e ".[dev,all]" && pytest tests/unit/ -v


## License

MIT — see [LICENSE](LICENSE). © 2026 Samuel Appiah Kubi. Part of the **PyGeoVision** platform, alongside [pygeofetch](https://github.com/appiahkubis14/pygeofetch) itself.
