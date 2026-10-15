# Glossary

**AOI (area of interest).** The geographic region you care about, usually a bounding box
`(min_lon, min_lat, max_lon, max_lat)` in EPSG:4326.

**Asset.** One file belonging to a scene, such as a band, a thumbnail, metadata or an archive.
Modelled by `SatelliteAsset`.

**Baseline (processing baseline).** The version of the Sentinel-2 processing chain. From 04.00,
L2A reflectance carries a +1000 digital-number offset.

**Burst.** One of the sub-swaths-in-time of a Sentinel-1 TOPS acquisition. Burst-level
synchronisation matters for InSAR stacks.

**Circuit breaker.** A guard that stops calling a provider after repeated failures and retries
after a recovery timeout. See [Error handling](error-handling.md#circuit-breaker).

**COG (Cloud-Optimized GeoTIFF).** A GeoTIFF with internal tiling and overviews, so a client
can read part of it with HTTP range requests. See [Streaming](../data-access/streaming.md).

**Coherence.** A 0 to 1 measure of phase similarity between two SAR acquisitions. Low
coherence means the surface changed or decorrelated.

**Cosine correction.** A simple topographic illumination correction that divides reflectance by
the cosine of the solar incidence angle.

**CFAR.** Constant false alarm rate, a thresholding method for detecting bright targets in SAR
imagery.

**DEM.** Digital elevation model.

**DoD (DEM of Difference).** The subtraction of two DEMs, used to measure volume change.

**DN (digital number).** The raw integer pixel value before scaling to reflectance.

**GRD.** Ground Range Detected: Sentinel-1 amplitude imagery without phase.

**InSAR.** Interferometric SAR: using the phase difference between two acquisitions to measure
surface deformation.

**Interferogram.** The phase difference image formed from two SLCs.

**L1C / L2A.** Sentinel-2 top-of-atmosphere (L1C) and surface reflectance (L2A) products.

**Landsat Collection 2.** USGS's reprocessed Landsat archive, delivered at Level-1 and Level-2.

**NDVI.** Normalized Difference Vegetation Index, `(NIR - Red) / (NIR + Red)`.

**Offset tracking.** Measuring displacement by cross-correlating image patches. Works on optical
and SAR amplitude imagery.

**Post-processing action.** A step run after a download, such as `unzip`, clip, reproject or
COG conversion.

**Provider.** An adapter for one data source. See [Providers](../data-access/providers.md).

**SAFE.** The Sentinel product directory format, usually delivered as `.SAFE.zip`.

**SAR.** Synthetic aperture radar, an active microwave sensor that images day and night through
cloud.

**SBAS.** Small BAseline Subset: a time-series InSAR method that combines many short-baseline
interferograms.

**SLC.** Single Look Complex: SAR imagery that preserves phase, needed for InSAR.

**STAC.** SpatioTemporal Asset Catalog: a standard JSON format for describing geospatial
assets, used by many providers.

**Streaming.** Reading only the pixels you need from a remote file instead of downloading all
of it.

**TOPS.** Terrain Observation by Progressive Scans, the Sentinel-1 wide-swath acquisition mode.

**Unwrapping.** Converting wrapped interferometric phase (modulo 2 pi) to continuous phase;
pygeofetch uses `snaphu`.
