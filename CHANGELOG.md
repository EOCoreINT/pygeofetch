# Changelog

All notable changes to PyGeoFetch are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.9.2/).

## [1.9.2] — 2026-07-12

### Added
- Federated search across 22+ satellite data providers
- Sentinel-1C and Sentinel-1D constellation support (active since May/April 2026)
- SLC product type search with automatic provider routing
- Precise orbit file management (POEORB/RESORB) for InSAR workflows
- 17 spectral indices: NDVI, EVI, SAVI, NDWI, MNDWI, NDBI, NDSI, NDMI,
  NBR, dNBR, TCT, PCA, Texture, LST, Albedo, Band Math, Stack
- SAR processing: despeckle, calibrate, flood mapping, coherence
- 41-step chainable processing pipeline builder
- YAML pipeline definitions with cron scheduling
- Cloud Optimized GeoTIFF output
- Clean search result tables with ANSI-aware column alignment
- Live download progress with Jupyter notebook HTML widget support
- Credential redaction in all log output

### Fixed
- AuthManager.add_credentials() now accepts dict form (BUG 1)
- download() length/order contract guaranteed (BUG 2)
- Partial downloads detected via file validation (BUG 3)
- CRS identity transform detection after reprojection (BUG 4)
- resolve_band_keys missing import in aws_earth provider
- Band alias conflict between Sentinel-2 and Landsat naming conventions
- Copernicus product_type OData filter now correctly applied
- USGS M2M API authentication payload (authType, catalogId fields)
- ZIP archive validation no longer attempts rasterio open on archives

### Providers
- Copernicus Dataspace (Sentinel-1/2/3/5P, SLC + GRD)
- AWS Earth (Sentinel-2, Landsat via STAC)
- Planetary Computer (Sentinel-2, Landsat, MODIS)
- Element84 Earth Search (STAC)
- USGS Earth Explorer (Landsat 1-9, ASTER, SRTM)
- NASA Earthdata (MODIS, ICESat-2, GEDI)
- Alaska SAR Facility (Sentinel-1 SLC, ALOS PALSAR)
- Planet Labs, Sentinel Hub, Maxar, Airbus, OpenTopography and more
