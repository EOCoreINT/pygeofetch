# Case studies

End-to-end sketches showing how the pieces fit together. Each uses commands and functions
documented elsewhere in this site.

:::{important}
These are workflow outlines, not benchmarked results. The processing steps were tested on
synthetic and sample data. Search and download calls were **not** run against live providers
while this documentation was written, so check provider names, collection IDs and credentials
against your account before relying on them.
:::

## Monitoring artisanal mining expansion

**Goal.** Detect new bare ground and disturbed vegetation around a concession over time.

1. **Find scenes.** Search Sentinel-2 L2A with low cloud cover over a fixed AOI, for each season.
   See [Searching](../data-access/search.md).
2. **Get only the AOI.** Use [`stream()`](../data-access/streaming.md) with a Planetary Computer or Element84 source so you do not download full tiles.
3. **Correct the offset.** For Sentinel-2 L2A at baseline 04.00 or later, set `optical_offset=-1000` and `optical_scale=10000` where the pipeline asks for it. See [Multi-sensor pipelines](../processing/multisensor.md).
4. **Compute indices.** NDVI for vegetation loss and a bare-soil index for exposed ground. See [Spectral indices](../processing/optical/spectral-indices.md).
5. **Add radar.** Use the NDVI plus SAR coherence pipeline to separate cloud-affected gaps from real disturbance, since coherence is unaffected by cloud.
6. **Map and report.** Use [Visualization](../visualization/plotter.md) to produce maps and change summaries.

**Caveats.** Mining detection from spectral change is an indicator, not evidence of activity. Field validation is needed before drawing conclusions about specific locations.

## Mineral exploration screening

**Goal.** Highlight alteration zones for follow-up fieldwork.

1. Download Sentinel-2 or Landsat scenes for the licence area ([Landsat](../processing/optical/landsat.md)).
2. Compute the geology indices (iron oxide, clay, ferrous minerals, and similar) from [Spectral indices](../processing/optical/spectral-indices.md).
3. Classify with [Classification](../processing/optical/classification.md) using training areas from known occurrences.
4. Add terrain context with [Terrain](../processing/optical/terrain.md).

Spectral indices screen; they do not identify mineralogy. Treat the result as a ranking of
places to visit.

## Drought and vegetation time series

**Goal.** Track seasonal vegetation condition over several years for a district.

1. Define the AOI and a monthly date loop.
2. Search and stream or download one low-cloud scene per month.
3. Compute NDVI per scene and stack with [Time series](../processing/optical/timeseries.md).
4. Compare each month with its multi-year mean to find anomalies.
5. Schedule the acquisition with a [YAML pipeline](pipelines.md) so new scenes are fetched automatically.

## Flood extent after a storm

**Goal.** Map flooding quickly, including under cloud.

1. Search Sentinel-1 GRD before and after the event ([SAR](../processing/sar-insar/sar.md)).
2. Run the flood-mapping SAR pipeline for a radar-only extent.
3. If cloud-free optical data exists, use the SAR plus optical flood pipeline to refine it ([Multi-sensor pipelines](../processing/multisensor.md)).
4. Export the flood mask and compute affected area.

## Ground deformation with InSAR

**Goal.** Measure subsidence or slope movement.

Follow the worked example in [InSAR Mexico City tutorial](../processing/sar-insar/insar-mexico-city-tutorial.md),
then see [InSAR](../processing/sar-insar/insar.md) for SBAS and the preflight checks. Note that
Copernicus delivers SLC as zip archives, so [streaming](../data-access/streaming.md) does not
apply and you must download full products.
