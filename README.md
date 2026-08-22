<div align="center">

<!-- <img src="" alt="pygeofetch Logo" width="200"/> -->
![Software Logo](icon/logo.png)

# pygeofetch 🛰️

[![PyPI version](https://badge.fury.io/py/pygeofetch.svg)](https://pypi.org/project/pygeofetch/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pygeofetch.svg)](https://pypi.org/project/pygeofetch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/appiahkubis14/pygeofetch/actions/workflows/tests.yml/badge.svg)](https://github.com/appiahkubis14/pygeofetch/actions)
[![Coverage](https://codecov.io/gh/appiahkubis14/pygeofetch/branch/main/graph/badge.svg)](https://codecov.io/gh/appiahkubis14/pygeofetch)

**Universal satellite data pipeline + geospatial processing platform — unified access to 24 satellite repositories, 17 spectral indices, a full verified InSAR chain, and chainable YAML pipelines. One CLI, one Python API.**

</div>

---

## 📖 Introduction

pygeofetch is a **production-ready satellite data acquisition and processing framework** that provides unified, authenticated access to 24 Earth observation repositories — including Sentinel, Landsat, Planet, Maxar, Airbus, Copernicus, USGS, NASA, JAXA, and more — through a single consistent CLI and Python API.

The package abstracts away the authentication complexity, API fragmentation, and format inconsistencies of individual satellite providers, and adds a complete geospatial processing engine on top, including a real, independently-verified InSAR pipeline built and validated against real Sentinel-1 data, not just synthetic test cases. pygeofetch provides seven core capabilities:

1. **Authenticated access** to 24 providers, with secure credential storage via system keyring (macOS Keychain, Windows Credential Manager, Linux Secret Service).
2. **Unified federated search** across all providers, returning standardized STAC 1.0 GeoJSON, GeoParquet, or CSV results sortable by cloud cover, date, or relevance score — with real, verified footprint geometry (not just bounding boxes) for providers whose APIs actually supply it.
3. **Resilient parallel downloads** with band selection, checksum verification, resume support, exponential backoff, and atomic writes.
4. **Preprocessing engine** — atmospheric correction, cloud masking, reprojection, resampling, pan-sharpening, mosaicking, and compositing.
5. **17 spectral indices** — NDVI, EVI, SAVI, NDWI, MNDWI, NDBI, TCT, PCA, LST, Albedo, dNBR, GLCM texture, and more.
6. **A full, verified InSAR chain** — burst-aware coregistration, real flat-earth and topographic phase removal, real ERA5 tropospheric and ionospheric correction, phase unwrapping, and SBAS time series inversion, each stage independently verified against synthetic ground truth or real, published deformation studies.
7. **YAML pipeline orchestration** with cron scheduling, batch processing, and full execution history — enabling repeatable, automated geospatial workflows.

---



<p align="center">
  <img src="https://raw.githubusercontent.com/EOCoreINT/pygeofetch/main/docs/images/trend_map.png" width="48%" />
  <img src="https://raw.githubusercontent.com/EOCoreINT/pygeofetch/main/docs/images/trend_classification.png" width="48%" />
</p>
<p align="center"><em>NDVI trend (2018–2024) and severity classification for the Obuasi Municipal District, computed end-to-end with PyGeoFetch — real USGS Landsat data, boundary-clipped before processing.</em></p>

32.0% of the Obuasi Municipal District shows measurable vegetation decline over 2018–2024 (9.2% strong decline + 22.8% moderate decline), against 57.5% stable and 10.5% showing an increasing trend.

That headline number matters less than where it's concentrated. The decline isn't scattered randomly across the district — it forms a clear, spatially coherent cluster in the western portion of the AOI (roughly west of -1.70° longitude), visible as a dense, contiguous red-and-orange zone in both the trend map and the classification map. Real land degradation signals tend to look like this — clustered along mining concessions, roads, or river corridors — whereas sensor noise or processing artifacts tend to scatter more randomly across the whole scene. The fact that this pattern holds together spatially is a reasonable indicator that it reflects something real on the ground, not just pixel-level noise.

That said, an NDVI trend map on its own can't distinguish why vegetation declined — illegal small-scale mining (galamsey), selective logging, agricultural land clearing, and urban expansion can all produce a similar signature. Given Obuasi's well-documented galamsey activity specifically concentrated in this district, the spatial pattern here is consistent with mining-driven clearance — but confirming that specific cause would need either higher-resolution imagery, ground verification, or cross-referencing against known concession boundaries, not this analysis alone.

Two things worth flagging honestly rather than glossing over:

The scattered dark-blue linear and blob features running through the map (most visibly the river-like feature crossing the upper-central area) are very likely water bodies, not vegetation gain. NDVI trend over open water isn't a meaningful signal — surface reflectance there is driven by turbidity and water level, not plant health — so those features should be excluded from the "increasing" interpretation, not read as reforestation.
The small white gaps scattered through the trend map (several distinct circular patches) are nodata — pixels that didn't have enough valid, cloud-free observations across all six dates to fit a reliable trend. Worth being upfront that the 32%/57.5%/10.5% breakdown is computed over the valid pixels only, not the full district area including these gaps.

The eastern two-thirds of the district, by contrast, is overwhelmingly stable (dominant green in the classification map), which is itself a useful negative result — it suggests the decline is genuinely localized to a specific area rather than reflecting a district-wide drought or seasonal artifact that would show up everywhere.


## 📝 Statement of Need

Accessing satellite data at scale is surprisingly fragmented. Each provider — USGS, Copernicus, Planet, Maxar, NASA — exposes a different authentication scheme, a different query API, a different download protocol, and a different file format. Researchers and engineers working across multiple providers must maintain a patchwork of custom scripts, scattered credentials, and ad hoc download logic, making workflows difficult to reproduce and brittle to maintain.

Existing tools address parts of this problem: EODAG supports several providers but lacks pipeline orchestration and commercial coverage; `pystac-client` handles STAC-compliant endpoints only; `sentinelsat` is Sentinel-specific; ISCE2/MintPy provide a genuine InSAR chain but no unified data access layer. No single tool covers the full breadth of providers, processing, InSAR, and automation needed for operational geospatial workflows.

| Feature | pygeofetch | EODAG | pystac-client | satpy | sentinelsat |
|---|---|---|---|---|---|
| **Providers** | **24** | 10+ | STAC only | Limited | Sentinel only |
| **Processing Engine** | ✅ Full | ❌ | ❌ | Partial | ❌ |
| **Spectral Indices** | ✅ 17+ | ❌ | ❌ | ❌ | ❌ |
| **Full InSAR Chain** | ✅ Verified | ❌ | ❌ | ❌ | ❌ |
| **Real Footprint Display** | ✅ | ❌ | ✅ (STAC only) | ❌ | ❌ |
| **YAML Pipelines** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Auth Management** | ✅ Keyring | Partial | ❌ | ❌ | ✅ |
| **STAC 1.0 Output** | ✅ Native | ❌ | ✅ | ❌ | ❌ |
| **Cron Scheduling** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Commercial Providers** | ✅ Planet/Maxar | ❌ | ❌ | ❌ | ❌ |

---

## 🚀 Key Features

### 🛰️ 24 Satellite Providers

**Open access — no login required (11):**

| Provider ID | Satellites | Capabilities |
|---|---|---|
| `planetary_computer` | Sentinel-1/2, Landsat 8/9, MODIS, NAIP, ALOS DEM | STAC, SAR, real footprint geometry |
| `aws_earth` | Sentinel-2 COG, Landsat C2, NAIP | STAC, real footprint geometry |
| `element84` | Sentinel-2 L2A, Landsat C2, Sentinel-1 RTC, COP-DEM | STAC, SAR, real footprint geometry |
| `noaa_big_data` | GOES-16/17/18, NEXRAD radar | Weather |
| `esa_scihub` | Sentinel-1/2/3/5P (public mirrors) | SAR |
| `jaxa_earth` | ALOS 30m DSM, PALSAR-2 | SAR |
| `isro_bhuvan` | ResourceSat-2/2A (5.8m), Cartosat-1 (2.5m) | — |
| `inpe_cbers` | CBERS-4, CBERS-4A | — |
| `digitalglobe` | WorldView open disaster response | <1m VHR |
| `geoserver_generic` | Any OGC WMS/WFS/WCS endpoint | Generic |
| `eodag` | Sentinel-1/2, Landsat (via EODAG's own providers) | SAR, real footprint geometry (Shapely-derived) |

**Authenticated providers (13):** USGS (real footprint geometry, confirmed) · Copernicus CDSE (real footprint geometry, confirmed against a live search) · NASA Earthdata (real footprint geometry, confirmed) · NASA Earthdata Cloud · Alaska SAR Facility · OpenTopography · Planet Labs (real footprint geometry, confirmed) · Sentinel Hub (real footprint geometry via STAC) · Maxar GBDX · Airbus OneAtlas · Google Earth Engine · TerraBotics · Earth Explorer

Every provider above has been individually, directly verified to populate at minimum a correct bounding box. For providers marked "confirmed," real, precise footprint geometry has been independently verified against the provider's own live API response or documented spec. For the remaining providers, geometry parsing is real and correctly wired — it will surface real, precise footprint shape the moment the provider's live API returns one — but that specific live return hasn't been independently confirmed for each of them; a bounding-box rectangle is what you'll see today. See [Real Footprint Display](#-real-footprint-display) below.

### 🔍 Unified Search
- Federated query across multiple providers simultaneously with deduplicated results
- Filter by bbox, geometry file, date range, cloud cover, resolution, processing level, and CQL2 expressions
- 7 output formats: `table` · `json` · `stac` · `geojson` · `geoparquet` · `csv` · `ids`

### 🗺️ Real Footprint Display

Every real search result can be shown directly on an interactive map, with real, useful hover info (scene ID, date, satellite, provider, cloud cover) — not just printed to a table.

```python
from pygeofetch.viz.map import MapViewer

mv = MapViewer(center=(19.36, -99.09), zoom=9)
mv.add_basemap("SATELLITE")
mv.add_search_results(search_results)   # real footprints, real hover info
mv.show()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/EOCoreINT/pygeofetch/main/docs/images/search.png" width="100%" />
</p>

`add_search_results()` uses each result's real, provider-supplied geometry when available, and falls back automatically to a bounding-box rectangle when it isn't — every provider is safe to call this against, none will silently show nothing or crash on an empty result set.

### 📥 Resilient Downloads
- Adaptive parallel downloads with configurable concurrency and real-time progress
- Band selection (e.g. `B02,B03,B04` → download 150 MB instead of 600 MB full scene)
- SHA256 checksum verification, resume support, exponential-backoff retries
- Atomic writes — no partial files ever written to disk

### ⚙️ Preprocessing Engine (`client.preprocess`)

| Method | Description |
|---|---|
| `atmos()` | Atmospheric correction: DOS1, DOS2, Sen2Cor, FLAASH, 6S, iCOR |
| `cloud_mask()` | Cloud masking: SCL, FMask, threshold, NDSI |
| `cloud_fill()` | Fill cloud gaps from time-series |
| `topo_correct()` | Topographic correction: cosine, Minnaert, C-correction |
| `clip()` | Clip to bounding box or GeoJSON polygon |
| `reproject()` | Reproject to any CRS (EPSG:4326, UTM, etc.) |
| `resample()` | Change resolution: nearest, bilinear, cubic, lanczos |
| `pansharpen()` | Pan-sharpening: Brovey, IHS, Gram-Schmidt |
| `tile()` | Split into overlapping tiles for AI inference |
| `mosaic()` | Merge scenes: first, last, min, max |
| `composite()` | Multi-temporal: median, mean, max, best-pixel |

### 📊 Spectral Indices (`client.indices`)

| Index | Formula | Use Case |
|---|---|---|
| `ndvi` | (NIR−Red)/(NIR+Red) | Vegetation health |
| `evi` | G·(NIR−Red)/(NIR+C1·Red−C2·Blue+L) | Dense canopy |
| `savi` | (NIR−Red)/(NIR+Red+L)·(1+L) | Sparse vegetation |
| `ndwi` | (Green−NIR)/(Green+NIR) | Water bodies |
| `mndwi` | (Green−SWIR1)/(Green+SWIR1) | Urban water |
| `ndbi` | (SWIR1−NIR)/(SWIR1+NIR) | Built-up areas |
| `ndsi` | (Green−SWIR1)/(Green+SWIR1) | Snow / ice |
| `ndmi` | (NIR−SWIR1)/(NIR+SWIR1) | Canopy moisture |
| `nbr` / `dnbr` | (NIR−SWIR2)/(NIR+SWIR2) | Burn severity |
| `tct` | Matrix coefficients | Brightness, Greenness, Wetness |
| `pca` | Eigen decomposition | Dimensionality reduction |
| `texture` | GLCM | Contrast, homogeneity, energy |
| `lst` | Thermal → Kelvin / Celsius | Land surface temperature |
| `albedo` | Narrowband→broadband (Liang 2001) | Surface reflectance |
| `band_math` | Arbitrary expression on B[i] | Custom indices |

### 🔧 Post-Processing (`client.post`)

`vectorize` → `smooth` → `regularize` → `zonal_stats` → `buffer` → `centroids` → `compress` → `cog`

### 📡 SAR Processing (`client.sar`)

| Method | Description |
|---|---|
| `despeckle()` | Lee, Enhanced Lee, Frost, Gamma MAP, Boxcar |
| `calibrate()` | DN → sigma0 / gamma0 / beta0 (dB or linear) |
| `flood_map()` | Threshold or change-based flood detection |
| `coherence()` | Interferometric coherence (stable surface / change) |

### 📋 YAML Pipeline Orchestration
- Define search → filter → download → process → export workflows in YAML
- Chain any preprocessing, index, or post-processing step
- Schedule with cron expressions, run history, and retry
- 6 built-in templates: `ndvi` · `change_detection` · `flood_map` · `urban_mapping` · `sar_analysis` · `land_cover`

### 🔐 Security by Default
- Credentials stored in system keyring — never logged or written to disk in plaintext
- TLS 1.2+ enforced, SSL verification always on, no telemetry, no analytics

---

## 📦 Installation

```bash
# Core — free providers work immediately, no extras needed
pip install pygeofetch

# + Raster/vector processing (rasterio, geopandas, shapely)
pip install "pygeofetch[geo]"

# + Cloud provider S3 access
pip install "pygeofetch[cloud]"

# + Cron scheduling
pip install "pygeofetch[schedule]"

# + Full InSAR chain (SBAS inversion, phase unwrapping)
pip install "pygeofetch[insar]"

# + Advanced InSAR corrections (MintPy passthrough)
pip install "pygeofetch[insar-full]"

# Everything
pip install "pygeofetch[all]"
```

**Requirements:** Python 3.9+

Verify your installation:
```bash
pygeofetch doctor
# ✓ Python 3.11   ✓ httpx   ✓ pydantic   ✓ rich
# ✓ AWS Earth Search: HTTP 200
# ✓ Planetary Computer: HTTP 200
# ✓ Element 84: HTTP 200
```

---

## ⚡ Quick Start

### CLI

```bash
# Add credentials (free providers need no credentials at all)
pygeofetch auth add usgs --username USER --password PASS
pygeofetch auth add copernicus --username email@example.com --password PASS
pygeofetch auth add planet --api-key YOUR_KEY

# Search (free — no login)
pygeofetch search run \
    --bbox "-74.1,40.6,-73.7,40.9" \
    --start-date 2024-01-01 \
    --cloud-cover 0-15 \
    --providers planetary_computer,aws_earth \
    --format table \
    --output results.geojson

# Download with band selection
pygeofetch download run \
    --from-search results.geojson \
    --output ./data/ \
    --parallel 4 \
    --bands "B02,B03,B04" \
    --max-items 3

# Download with full post-processing chain
pygeofetch download run \
    --from-search results.geojson \
    --output ./data/ \
    --parallel 4 \
    --verify-checksum \
    --post-process "unzip,reproject:EPSG:4326,compress:lzw,cog"
```

### Python API

```python
from pathlib import Path
from pygeofetch import PyGeoFetch
from pygeofetch.models.search_query import SearchQuery, BoundingBox
from pygeofetch.models.download_task import DownloadOptions

client = PyGeoFetch()

# Credentials
client.add_credentials("usgs",       username="user", password="pass")
client.add_credentials("copernicus", username="email@example.com", password="pass")
client.add_credentials("planet",     api_key="PL_KEY")

# Search
results = client.search(
    SearchQuery(
        bbox=BoundingBox.from_string("-74.1,40.6,-73.7,40.9"),
        start_date="2024-01-01",
        end_date="2024-06-01",
        cloud_cover_max=20,
        sort_by="cloud_cover",
        sort_ascending=True,
    ),
    providers=["usgs", "copernicus", "planetary_computer", "aws_earth"],
)

# See what you found, on a real map, before downloading anything
from pygeofetch.viz.map import MapViewer
mv = MapViewer(center=(40.7, -74.0), zoom=9)
mv.add_basemap("SATELLITE")
mv.add_search_results(results)
mv.show()

# Download
downloads = client.download(
    results[:5],
    destination=Path("./data/"),
    options=DownloadOptions(
        parallel=4,
        verify_checksum=True,
        resume=True,
        bands=["B02", "B03", "B04"],
    ),
)

for dr in downloads:
    if dr.success:
        print(f"✓ {dr.data_id} ({dr.bytes_downloaded // 1024 // 1024:.1f} MB)")
    else:
        print(f"✗ {dr.data_id}: {dr.error}")

# Process
ndvi   = client.indices.ndvi(red="B04.tif", nir="B08.tif")
clipped = client.preprocess.clip("scene.tif", bbox=(-74.1, 40.6, -73.7, 40.9))
cog     = client.post.cog("ndvi.tif", compress="deflate")

# End-to-end pipeline
result = (
    client.pipeline("sentinel2-ndvi")
    .atmos(method="dos1")
    .cloud_mask(method="scl", scl_band="SCL.tif")
    .clip(bbox=(-74.1, 40.6, -73.7, 40.9))
    .ndvi(red="B04.tif", nir="B08.tif")
    .vectorize(threshold=0.3)
    .cog(compress="deflate")
    .run(input="scene.tif", output_dir="./processed/")
)
print(f"Pipeline: {result.success} in {result.duration_seconds:.1f}s")
```

### YAML Pipeline

```yaml
name: weekly-sentinel2-ndvi
schedule: "0 6 * * 1"   # Every Monday 06:00 UTC
description: Weekly NDVI monitoring — search, download, process, export

steps:
  - search:
      providers: [copernicus, aws_earth, planetary_computer]
      bbox: "-74.1,40.6,-73.7,40.9"
      date_range: last_7_days
      cloud_cover: "0-10"
      max_results: 20

  - filter:
      expression: "data.cloud_cover < 5"

  - download:
      parallel: 4
      output: ./raw/
      verify_checksum: true
      bands: [B04, B08]

  - ndvi:
      red: B04.tif
      nir: B08.tif

  - vectorize:
      threshold: 0.3
      format: geojson

  - cog:
      compress: deflate
```

```bash
# Validate without running
pygeofetch proc-pipeline validate weekly-sentinel2.yaml

# Run once
pygeofetch proc-pipeline run weekly-sentinel2.yaml --input scene.tif

# Schedule (recurring)
pygeofetch pipeline schedule weekly-sentinel2.yaml --name "ndvi-weekly"

# Generate a starter template
pygeofetch proc-pipeline template ndvi
pygeofetch proc-pipeline template flood_map
pygeofetch proc-pipeline template change_detection
```

---

## 🌐 InSAR Processing (`pygeofetch.insar`)

A full, independently-verified Interferometric SAR chain for Sentinel-1 — search through SBAS time series inversion, pure Python, no external InSAR software required beyond `snaphu-py`'s bundled unwrapper. Every stage below has been verified against either synthetic ground truth (closed-loop, known-answer tests) or real, published deformation studies — not assumed correct from theory alone.

```bash
pip install "pygeofetch[insar]"        # native SBAS inversion
pip install "pygeofetch[insar-full]"   # + MintPy passthrough for advanced corrections
```

### Core components

| Component | Description |
|---|---|
| `SLCExtractor` | Sub-swath matching and AOI-cropped extraction from raw SAFE archives; `.show_on_map()` for real amplitude display |
| `InterferogramGenerator` | Real orbit-based coregistration (falls back cleanly to shape-based when orbit files aren't supplied) + real per-burst-overlap Enhanced Spectral Diversity (ESD, verified against Prats-Iraola 2012) + real TOPS deburst (verified against ESA's own algorithm) + real flat-earth phase removal (orbit-geometry-derived, verified to 0.01% against independent computation) + topographic phase removal + Goldstein filtering |
| `AtmosphericCorrector` | Two real, independently-verified methods: elevation-correlated circular regression, and real ERA5 tropospheric delay correction via `pyaps3` and CDS, with automatic per-date differencing (not a naive single-date subtraction) |
| `IonosphericCorrector` | Real ionospheric pierce-point (IPP) geometry (ICD-GPS-200/Klobuchar model) and real CDDIS/IONEX TEC data, with both a scalar and a full per-pixel correction mode — closed-loop verified to recover a known, spatially-varying synthetic dispersive signal exactly |
| `PhaseUnwrapper` | SNAPHU (Chen & Zebker 2001) via the official `snaphu-py` bindings — the same algorithm used by ASF, ISCE2/3, GAMMA, and SNAP |
| `SBASTimeSeries` | Small BAseline Subset displacement/velocity inversion (Berardino et al. 2002) |
| `DataValidator` | SLC, coherence, and SBAS-network sanity checks, wired in at every real pipeline entry point |
| `InSARProject` | A high-level workflow wrapper — real search, download, extraction, and interferogram formation in a handful of calls, each with automatic map display, built on top of the fully-verified lower-level pieces above |

Real orbit-based coregistration computes genuine per-pixel offsets from actual satellite orbit state vectors and acquisition timing (not a shape-matching guess) — supply a DEM, both SAFE archives, and both `.EOF` orbit files, and it's used automatically; otherwise falls back cleanly to shape-based resampling.

Every stage supports automatic visualization (`auto_visualize=True`) and optional GPU acceleration (`use_gpu=True`, CuPy) for coherence estimation and SBAS inversion on large scenes.

See [`pygeofetch/insar/README.md`](pygeofetch/insar/README.md) for the full processing chain, verification methodology, and current limitations.

### Full pipeline, real orbit-based coregistration

```python
from pygeofetch.insar import InterferogramGenerator

gen = InterferogramGenerator(
    use_gpu=False,
    use_real_burst_processing=True,   # real per-burst ESD + real deburst
    remove_flat_earth_phase=True,     # real, orbit-geometry-derived correction
)
result = gen.process_pair(
    reference="slc_ref.tif",
    secondary="slc_sec.tif",
    dem="dem.tif",
    reference_safe_zip="S1A_..._ref.SAFE.zip",
    secondary_safe_zip="S1A_..._sec.SAFE.zip",
    reference_orbit_file="ref.EOF",
    secondary_orbit_file="sec.EOF",
    reference_date="2024-11-08",
    secondary_date="2024-11-20",
    apply_goldstein_filter=True,
)
result.save("./output", auto_visualize=True)
result.show_on_map(band="wrapped_phase")   # real, cyclic-colormap display
```

### Without the orbit-based inputs (falls back safely)

```python
result = gen.process_pair(reference="slc_ref.tif", secondary="slc_sec.tif", dem="dem.tif")
# Logs: "Using shape-based coregistration fallback ..."
```

### Real ERA5 tropospheric correction

```python
from pygeofetch.insar.atmosphere import AtmosphericCorrector

# One-time: writes the real ~/.cdsapirc CDS reads from directly
atm = AtmosphericCorrector(method="era5", cds_api_key="your-cds-key")
corrected_phase, meta = atm.correct(
    wrapped_phase, dem="dem.tif",
    reference_datetime="2024-11-08T12:34:39",
    secondary_datetime="2024-11-20T12:34:38",
    return_metadata=True,
)
```

### Real ionospheric correction (per-pixel)

```python
from pygeofetch.insar.ionosphere import IonosphericCorrector

# One-time: writes the real ~/.netrc Earthdata Login reads from directly
iono = IonosphericCorrector(
    ionex_dir="./ionex",
    earthdata_username="your-username", earthdata_password="your-password",
)
corrected_phase = iono.correct_per_pixel(
    wrapped_phase,
    reference_datetime="2024-12-26T12:34:35", secondary_datetime="2025-01-07T12:34:34",
    lat_grid=lat_grid, lon_grid=lon_grid,
    reference_orbit_file="ref.EOF", secondary_orbit_file="sec.EOF",
)
```

### High-level workflow — search to interferogram in a handful of calls

```python
from pygeofetch.insar import InSARProject
from pygeofetch.models import BoundingBox

project = InSARProject(
    name="my_aoi", aoi=BoundingBox(min_lon=-99.183, max_lon=-99.003, min_lat=19.278, max_lat=19.438),
    output_dir="data/my_aoi_insar",
)
project.search(start_date="2024-11-01", end_date="2025-01-15")       # real search, real map
project.download_and_extract(max_scenes=6)                            # real download + AOI crop
project.form_all_interferograms()                                     # full verified pipeline
project.summary()
```

### Full chain, search to SBAS

```python
from pathlib import Path
import numpy as np
from pygeofetch import PyGeoFetch
from pygeofetch.models.search_query import SearchQuery, BoundingBox
from pygeofetch.models.download_task import DownloadOptions
from pygeofetch.processing.preprocessor import Preprocessor
from pygeofetch.core.orbits import fetch_orbit_file
from pygeofetch.insar import (
    SLCExtractor, InterferogramGenerator, AtmosphericCorrector,
    PhaseUnwrapper, SBASTimeSeries, DataValidator, multilook,
)
from pygeofetch.insar.timeseries import InterferogramPair

client = PyGeoFetch()
client.add_credentials("copernicus", username="you@example.com", password="...")
client.add_credentials("opentopography", api_key="...")

aoi = BoundingBox(min_lon=-1.75, max_lon=-1.63, min_lat=6.15, max_lat=6.24)
output_dir = Path("./insar_output")

# Search — real geometry/bbox filtering
scenes = {}
for start, end in [("2026-01-01", "2026-01-08"), ("2026-01-13", "2026-01-20")]:
    results = client.search(
        SearchQuery(bbox=aoi, start_date=start, end_date=end,
                    product_type="SLC", polarisation="VV", max_results=1),
        providers=["copernicus"],
    )
    if results:
        scenes[str(results[0].datetime.date())] = results[0]

# Download
downloads = {
    label: client.download([scene], destination=output_dir / "raw" / label,
                            options=DownloadOptions(resume=True))[0]
    for label, scene in scenes.items()
}

# Real orbit files — use the real product name, not the catalog ID
orbits = {
    label: fetch_orbit_file(
        product_name=scene.properties.get("name", scene.id),
        output_dir=str(output_dir / "orbits"), orbit_type="precise",
    )
    for label, scene in scenes.items()
}

# Real DEM, clipped to the AOI
dem_result = client.download(
    client.search(SearchQuery(bbox=aoi), providers=["opentopography"])[:1],
    destination=output_dir / "dem",
)[0]
dem_path = Preprocessor().clip(dem_result.output_path, bbox=aoi,
                                output=str(output_dir / "dem" / "clipped.tif")).output_path

# AOI-cropped extraction (not the full sub-swath)
extractor = SLCExtractor(polarisation="VV")
slcs = {
    label: extractor.extract_scene(dl.output_path, aoi=aoi,
                                    output_dir=output_dir / "slc" / label, label=label)
    for label, dl in downloads.items()
}
dates = sorted(slcs)

# Interferogram formation — real orbit-based coregistration when all four
# inputs are available; falls back to shape-based resampling otherwise
gen = InterferogramGenerator(coherence_window=5, esd_enabled=True, use_gpu=False)
d1, d2 = dates[0], dates[1]
result = gen.process_pair(
    reference=slcs[d1], secondary=slcs[d2], dem=dem_path,
    reference_date=d1, secondary_date=d2,
    reference_safe_zip=downloads[d1].output_path, secondary_safe_zip=downloads[d2].output_path,
    reference_orbit_file=orbits[d1], secondary_orbit_file=orbits[d2],
)
result.save(output_dir / "interferograms", auto_visualize=True)

# Atmospheric correction — return_metadata=True to know what actually happened
atm = AtmosphericCorrector(method="elevation")
corrected_phase, atm_meta = atm.correct(
    np.angle(result.interferogram), dem=dem_path, return_metadata=True
)

# Unwrapping — multilook first; wrapped_phase must be explicit (True for
# phase, False for coherence — never guessed from dtype)
phase_ml = multilook(corrected_phase, 4, 1, wrapped_phase=True)
coherence_ml = multilook(result.coherence, 4, 1, wrapped_phase=False)
unwrapper = PhaseUnwrapper(cost_mode="defo", init_method="mcf")
unwrapped, conncomp = unwrapper.unwrap(phase_ml, coherence_ml, nlooks=4.0)

# SBAS inversion
pairs = [InterferogramPair(d1, d2, unwrapped, coherence_ml)]
network_check = DataValidator.validate_sbas_network(pairs, dates)
sbas = SBASTimeSeries(wavelength_m=0.05546576, reference_date=dates[0], use_gpu=False)
ts_result = sbas.invert(pairs, reference_pixel=(0, 0))
ts_result.save(output_dir / "timeseries", auto_visualize=True)
print(f"Mean velocity: {ts_result.velocity.mean()*1000:.1f} mm/year")
```

<!-- A real, runnable two-date example — extend the search date list and repeat
the interferogram/unwrapping steps per consecutive pair for a full SBAS
network. See the notebooks table below for complete, real, multi-date
worked examples of this exact chain, including two real, published
volcanic and seismic deformation events. -->

---

## 🖥️ Complete CLI Reference

```
SYSTEM
  pygeofetch doctor                     diagnose installation + connectivity
  pygeofetch status [--json]            provider and cache overview
  pygeofetch version

AUTH
  pygeofetch auth add PROVIDER [--username U] [--password P] [--api-key K]
  pygeofetch auth login PROVIDER        interactive prompt
  pygeofetch auth list [--json]
  pygeofetch auth test PROVIDER
  pygeofetch auth remove PROVIDER [--yes]
  pygeofetch auth export [--output FILE]

PROVIDERS
  pygeofetch providers list [--auth|--no-auth] [--capabilities sar] [--json]
  pygeofetch providers info PROVIDER
  pygeofetch providers search "TERM"

SEARCH
  pygeofetch search run \
    --bbox "minlon,minlat,maxlon,maxlat"   or  --geometry-file area.geojson
    --start-date YYYY-MM-DD  --end-date YYYY-MM-DD
    --cloud-cover 0-20
    --providers aws_earth,copernicus
    --satellites Sentinel-2
    --sort-by cloud_cover  --sort-order asc
    --max-results 50
    --cql2 "eo:cloud_cover < 5"
    --format table|json|stac|geojson|geoparquet|csv|ids
    --output results.geojson
    --no-cache  --timeout 60

DOWNLOAD
  pygeofetch download run \
    --from-search results.geojson
    --output ./data/
    --parallel 4  --retry 5
    --bands "B02,B03,B04"
    --verify-checksum  --resume
    --bandwidth-limit 10MB
    --post-process "reproject:EPSG:4326,compress:lzw,cog"
    --on-failure skip
    --max-items 10
    --notify webhook:https://hooks.slack.com/YOUR/WEBHOOK
    --json

PREPROCESSING
  pygeofetch preprocess atmos         scene.tif --method dos1|sen2cor|flaash|6s
  pygeofetch preprocess cloud-mask    scene.tif --method scl|fmask|threshold
  pygeofetch preprocess cloud-fill    cloudy.tif t1.tif t2.tif
  pygeofetch preprocess topo-correct  scene.tif dem.tif --method cosine
  pygeofetch preprocess clip          scene.tif --bbox "..." | --geometry file.geojson
  pygeofetch preprocess reproject     scene.tif --crs EPSG:4326
  pygeofetch preprocess resample      scene.tif --resolution 30
  pygeofetch preprocess pansharpen    pan.tif ms.tif --method brovey
  pygeofetch preprocess mosaic        s1.tif s2.tif --method first|last|min|max
  pygeofetch preprocess composite     *.tif --method median|mean|max|best_pixel
  pygeofetch preprocess tile          scene.tif --tile-size 512 --overlap 64

SPECTRAL INDICES
  pygeofetch index ndvi    --red B04.tif --nir B08.tif
  pygeofetch index evi     --blue B02.tif --red B04.tif --nir B08.tif
  pygeofetch index savi    --red B04.tif --nir B08.tif --soil-l 0.5
  pygeofetch index ndwi    --green B03.tif --nir B08.tif
  pygeofetch index mndwi   --green B03.tif --swir1 B11.tif
  pygeofetch index ndbi    --nir B08.tif --swir1 B11.tif
  pygeofetch index ndsi    --green B03.tif --swir1 B11.tif
  pygeofetch index ndmi    --nir B08.tif --swir1 B11.tif
  pygeofetch index nbr     --nir B08.tif --swir2 B12.tif
  pygeofetch index dnbr    --pre-nir B08.tif --pre-swir2 B12.tif \
                            --post-nir B08_post.tif --post-swir2 B12_post.tif
  pygeofetch index tct     --blue B02.tif --green B03.tif --red B04.tif \
                            --nir B08.tif --swir1 B11.tif --swir2 B12.tif
  pygeofetch index pca     B02.tif B03.tif B04.tif B08.tif --components 3
  pygeofetch index texture  B08.tif --window 7 --features contrast,homogeneity
  pygeofetch index lst     B10.tif --emissivity 0.97 --sensor landsat8
  pygeofetch index albedo  B02.tif B03.tif B04.tif B08.tif B11.tif B12.tif
  pygeofetch index band-math B04.tif B08.tif --expr "(B[1]-B[0])/(B[1]+B[0]+1e-6)"
  pygeofetch index stack   B02.tif B03.tif B04.tif

POST-PROCESSING
  pygeofetch post vectorize       ndvi.tif --threshold 0.3 --format geojson
  pygeofetch post smooth          polygons.geojson --tolerance 0.5
  pygeofetch post regularize      buildings.geojson
  pygeofetch post zonal-stats     ndvi.tif parcels.geojson --output stats.csv
  pygeofetch post buffer          roads.geojson --distance 15
  pygeofetch post centroids       polygons.geojson
  pygeofetch post geometry-metrics polygons.geojson
  pygeofetch post compress        scene.tif --method lzw|deflate|zstd
  pygeofetch post cog             scene.tif --compress deflate --blocksize 512

SAR PROCESSING
  pygeofetch sar despeckle  sar.tif --filter lee|enhanced_lee|frost|gamma --window 7
  pygeofetch sar calibrate  sar.tif --output-type sigma0|gamma0|beta0 --db
  pygeofetch sar flood-map  post.tif --threshold -15 [--reference pre.tif]
  pygeofetch sar coherence  slc1.tif slc2.tif --window 7

PROCESSING PIPELINES
  pygeofetch proc-pipeline template ndvi|change_detection|flood_map|urban_mapping|...
  pygeofetch proc-pipeline validate FILE
  pygeofetch proc-pipeline run FILE [--input scene.tif] [--output-dir ./out/]

DATA PIPELINES (search + download)
  pygeofetch pipeline run|validate|schedule|list-scheduled|unschedule|history

CACHE
  pygeofetch cache stats|clear|ttl [show|set N]|location|prune --max-size 1GB

CONFIG
  pygeofetch config show|get KEY|set KEY VALUE|path|reset

COMPLETION
  pygeofetch --install-completion bash|zsh|fish
```

## 📚 Notebooks

| Notebook | Topics |
|---|---|
| `01_getting_started.ipynb` | Install, doctor, first search, first download |
| `02_authentication_and_providers.ipynb` | All 24 providers, credentials, capability filters |
| `03_advanced_search.ipynb` | Federated search, CQL2 filters, 7 output formats, caching |
| `04_download_and_postprocessing.ipynb` | Band selection, parallel downloads, post-processing |
| `05_pipelines_and_scheduling.ipynb` | YAML pipelines, scheduling, Python builder API |
| `06_real_world_workflows.ipynb` | NDVI time series, change detection, multi-sensor fusion |
| `07_copernicus_and_authenticated_providers.ipynb` | Copernicus, USGS, NASA, Planet, ASF, OpenTopography |
| `08_cli_complete_reference.ipynb` | Every CLI command with runnable examples |
| `09_processing_complete.ipynb` | Full processing engine: preprocessing, indices, SAR, pipelines |
| `piton_fournaise_full.ipynb` | Real, verified InSAR pipeline against the real April 2021 Piton de la Fournaise eruption (Réunion Island) — real pygeofetch search/download throughout, full burst-aware coregistration, real flat-earth and ERA5 correction |
| `amatrice_full.ipynb` | Real, verified InSAR pipeline against the real 2016 Amatrice, Italy earthquake, using the exact real interferometric pair dates from a published processing archive |
| `provider_search_footprint_test.ipynb` | Real search and footprint-map display validation across all 24 providers, one fully explicit, independently-runnable cell per provider |

```bash
cd notebooks/
jupyter lab
```

---

## 📋 Documentation

Full documentation: **https://appiahkubis14.github.io/pygeofetch-docs/**

Covers: CLI reference · provider auth guides · pipeline configuration · post-processing catalogue · InSAR verification methodology · contributing guide.

---

## 🤝 Contributing

Contributions of all kinds are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

Good first issues include implementing stub providers to full API integrations, extending real footprint geometry support to the remaining bbox-only providers, improving test coverage, and adding new post-processing actions.

```bash
git clone git@github.com:EOCoreINT/pygeofetch.git
cd pygeofetch
pip install -e ".[dev,all]"
pytest tests/unit/ -v
```

---

## 📄 License

pygeofetch is free and open source software, licensed under the [MIT License](LICENSE).

© 2026 Samuel Appiah Kubi. Part of the **PyGeoVision** platform — [pygeofetch](https://github.com/appiahkubis14/pygeofetch) (data + processing) — complete Earth observation pipeline.