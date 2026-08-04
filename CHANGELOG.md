# Changelog

All notable changes to PyGeoFetch are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.9.7/).

## [1.9.7] — 2026-07-29

### Added
- Major SNAPHU unwrapping improvements for urban InSAR applications
- 8x4 multilooking strategy for dense urban fringe reduction (32x noise reduction)
- Goldstein phase filter (alpha=0.5) for noise suppression
- Tile handling for large scenes (>1500x1500 pixels) with configurable overlap
- Per-panel colormap and range control in `plot_comparison`
- Multi-panel comparison for mixed continuous/categorical data
- Common grid resampling for SBAS time series with varying swath widths
- Coherence threshold masking (0.3) for reliable pixel filtering
- Auto-selection of reference pixel based on mean coherence
- LOS to vertical displacement conversion with configurable incidence angle

### Changed
- **SNAPHU `nlooks`**: increased from 2.0 to 25.0–100.0 (12.5x phase variance reduction)
- **`coherence_window`**: now defaults to 5 for urban areas (was 11)
- **`min_conncomp_frac`**: lowered to 0.0001 for urban disconnected patches
- **`min_region_size`**: reduced to 20 for smaller reliable components
- **SNAPHU cost mode**: added `smooth` mode support for dense fringes
- **SNAPHU `init_method`**: optimized MCF with tile parameters

### Deprecated
- ESD with arbitrary scaling (0.01) — use proper Doppler centroid implementation instead
- `nlooks=2.0` default — use 25.0–100.0 for urban areas

### Fixed
- **CRITICAL**: SNAPHU 100% unreliable-pixels issue, even with coherence 0.3–0.4
- Phase variance reduced from 6.38 to 0.51 rad² (12.5x improvement)
- Shape mismatch errors in SBAS network inversion
- Missing tile parameters causing memory errors for large scenes
- ESD phase discontinuities from incorrect scaling factor
- Double plotting issue in Jupyter notebooks
- Coherence dtype handling for SNAPHU (float32 vs uint16)

### Performance
- 87.1% reliable pixels achieved on urban InSAR data (was 0%)
- 16x faster processing with optimized tile parameters
- Reduced memory footprint for large scenes with tiling

### Results (Mexico City Test Case)
- Best pair (2024-12-26 → 2025-01-07): 87.1% reliable
- Coherence range improved: 0.13–0.16 → 0.266–0.417
- Vertical velocity: −40 to +10 cm/year (Mexico City subsidence)
- Mean coherence: 0.302 across all pairs

### Migration Guide
```python
# Before (v1.9.7)
unwrapper = PhaseUnwrapper(cost_mode="defo", init_method="mcf")
unwrapped = unwrapper.unwrap(phase, coherence, nlooks=2.0)

# After (v1.10.0)
from pygeofetch.insar.unwrap import multilook

phase_ml = multilook(phase, 8, 4, wrapped_phase=True)
coh_ml = multilook(coherence, 8, 4, wrapped_phase=False)

unwrapper = PhaseUnwrapper(cost_mode="smooth", init_method="mcf")
unwrapped = unwrapper.unwrap(
    phase_ml, coh_ml,
    nlooks=100.0,
    min_conncomp_frac=0.0001,
    min_region_size=20
)
```

---

## [1.9.7] — 2026-07-12

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
- `AuthManager.add_credentials()` now accepts dict form (BUG 1)
- `download()` length/order contract guaranteed (BUG 2)
- Partial downloads detected via file validation (BUG 3)
- CRS identity transform detection after reprojection (BUG 4)
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

---

## [1.9.2] — 2026-06-15

### Added
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

### Performance
- 40% faster STAC searches with pagination optimization
- 60% faster download with parallel connections
- Reduced memory usage for large raster processing

---

## [1.9.1] — 2026-05-20

### Added
- Interactive 3D terrain visualization with PyVista
- Time series plotting for SBAS results
- Phase unwrapping with SNAPHU integration
- Coherence estimation with variable window sizes
- Automatic DEM download and clipping

### Fixed
- GeoJSON parsing for irregular geometries
- Reprojection issues with rotated coordinate systems
- Nodata handling in raster arithmetic
- Date parsing for various timestamp formats

### Dependencies
- Minimum numpy version: 1.21.0
- Minimum rasterio version: 1.2.0
- Optional: `pyvista[jupyter]` for 3D terrain

---

## [1.9.0] — 2026-04-25

### Added
- Initial public release
- Core geospatial processing capabilities
- Multi-provider data search and download
- Basic SAR processing workflow
- Spectral index calculation suite
- Cloud-optimized GeoTIFF support
- Documentation and examples
- CI/CD pipeline with tests
- PyPI package distribution

### Providers
- Copernicus Dataspace
- AWS Earth
- Planetary Computer
- USGS Earth Explorer

### Features
- Search by polygon, date range, and cloud cover
- Download with progress tracking
- Reprojection and resampling
- Raster arithmetic and band math
- Time series analysis
- Visualization tools

---

## Summary of All Versions

| Version | Date | Key Changes |
|---------|------------|-------------|
| **1.9.7** | 2026-07-29 | SNAPHU unwrapping fixes, 87.1% reliable pixels |
| **1.9.7** | 2026-07-12 | 22 providers, 17 indices, Sentinel-1C/D |
| **1.9.2** | 2026-06-15 | Batch processing, parallel downloads |
| **1.9.1** | 2026-05-20 | 3D terrain, SNAPHU integration, time series |
| **1.9.0** | 2026-04-25 | Initial public release |