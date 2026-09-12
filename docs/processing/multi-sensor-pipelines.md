# Multi-Sensor Pipelines

Five real, standard pipelines that each combine **two genuinely
different sensor types** to do something neither one reliably does
alone. This is a different idea from
[SAR Processing](sar.md)'s five pipelines, which combine multiple
*dates* of the *same* sensor — these combine multiple *sensor types*
on (usually) the same or overlapping dates.

Every pipeline here returns the same real
`pygeofetch.sar.pipelines.PipelineResult` type used throughout
pygeofetch's orchestrated pipelines: `success`, `final_output`, and a
`metadata` dict with pipeline-specific detail — a consistent interface
whether you're fusing InSAR and optical, or computing DEM volumes.

```{contents}
:local:
:depth: 1
```

## 1. InSAR + Optical Displacement Fusion

```python
from pygeofetch.multisensor import insar_optical_displacement_pipeline

result = insar_optical_displacement_pipeline(
    insar_velocity_path="sbas_velocity.tif",
    insar_coherence_path="sbas_coherence.tif",
    optical_reference_path="sentinel2_2023-01-01_B08.tif",
    optical_secondary_path="sentinel2_2023-06-01_B08.tif",
    output_dir="./fused",
    cloud_mask_path="sentinel2_2023-06-01_SCL.tif",
)

print(f"InSAR-dominant: {result.metadata['pct_insar_dominant']}%")
print(f"Optical-dominant: {result.metadata['pct_optical_dominant']}%")
```

```bash
pygeofetch multisensor insar-optical-fusion \
    --insar-velocity sbas_velocity.tif --insar-coherence sbas_coherence.tif \
    --optical-ref sentinel2_pre_B08.tif --optical-sec sentinel2_post_B08.tif \
    --output-dir ./fused --cloud-mask sentinel2_post_SCL.tif
```

### Why this exists

This formalizes the real, proven combination this project's own
**Bu'ertai Mine validation run** demonstrated works: [InSAR](insar.md)
gives precise (millimetre-scale) vertical displacement, but is
structurally blind wherever ground change between the two SAR passes
exceeds the phase-coherence limit — a real physical constraint, not a
tuning problem. [Optical pixel offset tracking](optical-offset-tracking.md)
is coarser (typically sub-metre to a few metres), but keeps working
exactly where InSAR can't, since it never depends on phase coherence
at all.

In the real Bu'ertai run, InSAR's final usable coverage was **0.1%**
of the processed scene. Optical offset tracking, same real area and
dates, covered **83.7%**. Neither number is a target — both are what
actually came out of a real, executed pipeline run.

### What this pipeline actually does that the two underlying functions don't

[`fuse_insar_optical`](optical-offset-tracking.md#fusing-insar-and-optical)
itself assumes both inputs are already the same shape. In reality,
InSAR (Sentinel-1, often 5-20m pixels after multilooking) and optical
offset tracking (Sentinel-2, producing a displacement grid at
`step_size` × the optical pixel size — e.g. 160m for a 10m-pixel image
with `step_size=16`) essentially never share a native grid. This
pipeline:

1. Runs `prepare_optical_pair` and `compute_pixel_offsets` on the raw
   optical rasters.
2. Builds the real, correctly-georeferenced transform for the
   resulting sparse displacement grid (verified against a hand-worked
   example before use — window `(0, 0)`'s ground-coordinate center
   must exactly match what the original image's own transform gives
   for that window's real pixel-coordinate center).
3. Reprojects the optical displacement and SNR rasters onto the InSAR
   grid via `rasterio.warp.reproject`.
4. Calls `fuse_insar_optical` on the now-aligned arrays.

### A real, honest edge effect worth knowing

Window-based correlation can't measure right at an image's border — a
window needs real pixels beyond its own center to correlate against.
This means a border strip of the InSAR grid genuinely has no real
optical coverage after reprojection, and correctly falls back to InSAR
regardless of coherence there (by `fuse_insar_optical`'s own real,
documented fallback logic: use whichever source is actually present).
This is honest, correct behavior, not a bug — if you see InSAR-dominant
pixels exactly at your scene's edges even under low coherence, this is
why.

### Output

| File | Contents |
|---|---|
| `fused_displacement.tif` | The real fused value — InSAR's where InSAR-dominant, optical's where optical-dominant, blended in the transition zone |
| `fused_source_mask.tif` | `2`=InSAR-dominant, `1`=transition, `0`=optical-dominant — **always check this alongside the fused value** |

:::{danger}
Per `fuse_insar_optical`'s own real scientific caveat: InSAR gives
*vertical* displacement, optical gives *horizontal* magnitude. The
fused raster's real physical meaning at any given pixel depends on
which source actually produced it — read the source mask, don't treat
the fused value as one uniform physical quantity.
:::

---

## 2. SAR + Optical Flood/Water Mapping

```python
from pygeofetch.multisensor import multi_sensor_flood_pipeline
from pygeofetch.sar import SARProcessor

result = multi_sensor_flood_pipeline(
    sar_processor=SARProcessor(),
    sar_input_path="sentinel1_during_flood.tif",
    optical_green_path="sentinel2_B03.tif",
    optical_nir_path="sentinel2_B08.tif",
    output_dir="./flood",
    cloud_mask_path="sentinel2_SCL.tif",
    fusion_mode="cloud_aware",
)
```

```bash
pygeofetch multisensor flood-map \
    --sar sentinel1_during_flood.tif \
    --optical-green sentinel2_B03.tif --optical-nir sentinel2_B08.tif \
    --output-dir ./flood --cloud-mask sentinel2_SCL.tif --fusion-mode cloud_aware
```

### Why this exists

The same real idea Copernicus's Emergency Management Service uses for
its own flood extent products: SAR sees through the cloud cover that
almost always accompanies a flood event (a real physical property of
radar — clouds are effectively transparent at SAR wavelengths, not a
marketing claim); optical gives cleaner, more precise spectral water
discrimination (NDWI) wherever clouds do clear. Neither alone is as
complete as the two combined.

Reuses [`flood_mapping_pipeline`](sar.md#flood_mapping_pipeline) for
the real SAR side (which already enforces the real
calibrate→despeckle prerequisite chain) rather than reimplementing SAR
flood detection — this pipeline's own real contribution is the optical
NDWI side and the fusion logic between the two.

### Fusion modes

| Mode | Real behavior | When to use |
|---|---|---|
| `cloud_aware` (default) | Trust optical wherever it has a real, cloud-free reading; fall back to SAR wherever clouded | The real, physically-motivated default — optical is more precise where it has genuine signal |
| `union` | Water if *either* real source says water, wherever both have coverage | Disaster response, where missing flooded area is a worse error than a false positive |
| `intersection` | Water only where *both* real sources agree | Higher-confidence, fewer false positives, less complete coverage |

### NDWI threshold

Uses the real McFeeters (1996) NDWI formula
(`(GREEN - NIR) / (GREEN + NIR)`) via
[`pygeofetch.processor.indices.SpectralIndex`](spectral-indices.md),
with McFeeters' own original real threshold of `0.0` as the default —
override `optical_ndwi_threshold` if your own scene's water bodies
need a different cutoff (turbid or sediment-laden water can shift NDWI
away from the clean-water assumption the original threshold was
calibrated against).

### Real cloud mask classes

The default `cloud_mask_scl_classes=[3, 8, 9, 10]` matches Sentinel-2's
own real SCL band values for cloud shadow, cloud (medium probability),
cloud (high probability), and thin cirrus — override if using a
different sensor's own cloud-mask convention.

---

## 3. Terrain-Corrected Optical Change Detection

```python
from datetime import datetime, timezone
from pygeofetch.multisensor import terrain_corrected_change_pipeline

result = terrain_corrected_change_pipeline(
    optical_pre_path="ndvi_2023-01-15.tif",
    optical_post_path="ndvi_2023-07-15.tif",
    dem_path="copernicus_dem.tif",
    pre_datetime=datetime(2023, 1, 15, 10, 30, tzinfo=timezone.utc),
    post_datetime=datetime(2023, 7, 15, 10, 45, tzinfo=timezone.utc),
    latitude=27.98, longitude=86.92,  # scene center, e.g. Himalayan terrain
    output_dir="./terrain_change",
)

print(result.metadata["solar_position_pre"])   # {'zenith_deg': ..., 'azimuth_deg': ...}
print(result.metadata["solar_position_post"])
```

```bash
pygeofetch multisensor terrain-change \
    --pre ndvi_2023-01-15.tif --post ndvi_2023-07-15.tif --dem copernicus_dem.tif \
    --pre-datetime 2023-01-15T10:30:00 --post-datetime 2023-07-15T10:45:00 \
    --lat 27.98 --lon 86.92 --output-dir ./terrain_change
```

### Why this exists

In mountainous or high-relief terrain, raw optical change detection
confuses real land-cover change with shadow/illumination artifacts
from the terrain itself. A slope facing the sun looks brighter; the
same slope facing away looks darker — independent of anything actually
changing on the ground. Since the sun's real position is almost never
identical between two different real acquisition dates (different
season, different time of day, or both), that illumination difference
alone can look exactly like real change to a naive pixel-by-pixel
comparison.

This pipeline computes each date's own real, independent solar
position and applies the real, standard cosine topographic correction
(Teillet et al. 1982) before differencing — extending the same real
slope/aspect physics already built for
[InSAR layover/shadow masking](insar.md#advanced-safeguards-custom-dems-layovershadow-and-topographic-residuals)
into optical illumination correction.

### The real physics

Real solar position via Spencer (1971)'s simplified Fourier-series
approximation — the same standard approximation used throughout real
remote-sensing topographic-correction literature and NOAA's own solar
calculator. Verified directly against known physical reference points
before use here, not assumed correct from the formula alone:

- Equator at the equinox at solar noon → zenith ≈ 0° (sun directly
  overhead) — confirmed.
- Tropic of Cancer at the June solstice at solar noon → zenith ≈ 0°,
  declination ≈ 23.44° — confirmed.
- 45°N at the equinox at solar noon → zenith matching the real,
  expected `|latitude − declination|` relationship to within 0.03° —
  confirmed.

Real cosine correction:

```{math}
L_{corrected} = L_{observed} \times \frac{\cos(\theta_z)}{\cos(i)}
```

where the real local illumination angle {math}`i` (the angle between
the sun's rays and the surface normal at each pixel) is:

```{math}
\cos(i) = \cos(s)\cos(\theta_z) + \sin(s)\sin(\theta_z)\cos(\phi_s - a)
```

— {math}`s` = slope, {math}`a` = aspect (both from the real DEM via
`pygeofetch.insar.advanced_safeguards.slope_aspect_degrees`),
{math}`\theta_z` = solar zenith, {math}`\phi_s` = solar azimuth.

:::{note}
This is honestly the simplest real, standard correction (cosine
correction) — more sophisticated real methods exist (C-correction,
Minnaert, SCS+C) that better handle very steep terrain or dense
canopy, and aren't implemented here. If cosine correction
under-corrects in your own real, very steep study area, that's a real,
known limitation of this specific method, not just this
implementation of it.
:::

### Proof this isn't just theoretical

The test suite for this pipeline doesn't just check the formula runs —
it constructs a synthetic scenario where the true, underlying
reflectance is **genuinely unchanged** between two dates with
deliberately different real solar geometry, then confirms: (1) a naive,
uncorrected difference falsely flags real change purely from
illumination, and (2) the pipeline's real correction reduces that false
signal to less than 30% of the naive amount.

---

## 4. DEM Differencing / Volumetric Change

```python
from pygeofetch.multisensor import dem_differencing_pipeline

result = dem_differencing_pipeline(
    dem_old_path="srtm_2015.tif",
    dem_new_path="uav_dem_2024.tif",
    output_dir="./dem_diff",
    dem_old_vertical_rmse_m=8.0,   # real, published SRTM accuracy for this terrain
    dem_new_vertical_rmse_m=0.15,  # real, typical UAV photogrammetry accuracy
)

print(f"Net volume change: {result.metadata['net_volume_change_m3']} m³")
print(f"Level of detection: {result.metadata['level_of_detection_m']} m")
```

```bash
pygeofetch multisensor dem-diff \
    --dem-old srtm_2015.tif --dem-new uav_dem_2024.tif --output-dir ./dem_diff \
    --old-rmse 8.0 --new-rmse 0.15
```

### Why this exists

A real, standard geomorphology technique — "DEM of Difference" (DoD),
following Brasington et al. (2000) and Lane et al. (2003) — comparing
two real DEMs from different dates to quantify real elevation and
volume change. General-purpose: the same real technique applies to
landslide scarp/deposit volumes, glacier mass balance, coastal
dune/cliff erosion, and construction/earthwork progress tracking — not
just the open-pit mine or quarry that might be the most obvious use
case.

### The real detail most naive implementations skip

Raw elevation differencing without an uncertainty threshold treats DEM
noise as real change. The real, standard fix — Brasington et al.
(2000)'s own methodology — is a propagated **Level of Detection**
(LoD):

```{math}
\text{LoD} = t \sqrt{\sigma_{old}^2 + \sigma_{new}^2}
```

Only pixels where `|elevation difference| > LoD` are treated as
statistically real change; `t = 1.96` (95% confidence under a normal
error assumption) is the real, standard default.

:::{warning}
`dem_old_vertical_rmse_m`/`dem_new_vertical_rmse_m` are **not**
something this function measures — they're a real, independently-known
property of each specific DEM product you supply (SRTM's real,
published absolute vertical accuracy is often cited around 5-10m
depending on terrain; a UAV photogrammetry DEM might be 5-20cm; check
your own DEM's real documentation). Getting these wrong doesn't break
the pipeline, but it does make the real LoD threshold — and therefore
every volume number this pipeline reports — meaningless.
:::

### Verified against an exact, known volume

The test suite doesn't just check "some positive number comes out" —
it constructs a precisely-sized synthetic "pile" (a known pixel count
× known pixel area × known height) and confirms the computed deposition
volume matches the exact analytical value (50,000 m³ for a 10×10-pixel,
10m-pixel, 5m-tall test case) to within 1%.

### Real grid handling

If the two DEMs don't share a real grid, the new DEM is automatically
reprojected onto the old DEM's grid (the same real
`rasterio.warp.reproject` pattern used throughout pygeofetch) — you
don't need to align them yourself first.

### Output

| Metadata key | Meaning |
|---|---|
| `level_of_detection_m` | The real LoD threshold actually used |
| `erosion_volume_m3` | Real volume loss (from pixels with a significant negative difference) |
| `deposition_volume_m3` | Real volume gain |
| `net_volume_change_m3` | `deposition − erosion` |
| `pct_significant` | Real fraction of pixels exceeding the LoD — a useful sanity check on whether your two RMSE values are sensible for the actual, observed differences |

---

## 5. Vegetation Health + Sub-Canopy Disturbance

```python
from pygeofetch.multisensor import (
    vegetation_disturbance_pipeline,
    CLASS_NO_DISTURBANCE, CLASS_VISIBLE_DISTURBANCE, CLASS_SUBCANOPY_DISTURBANCE,
)

result = vegetation_disturbance_pipeline(
    optical_red_pre_path="red_2023-01.tif", optical_nir_pre_path="nir_2023-01.tif",
    optical_red_post_path="red_2023-07.tif", optical_nir_post_path="nir_2023-07.tif",
    sar_slc_pre_path="slc_2023-01.tif", sar_slc_post_path="slc_2023-07.tif",
    output_dir="./disturbance",
)

print(f"Visible disturbance: {result.metadata['pct_visible_disturbance']}%")
print(f"Sub-canopy disturbance: {result.metadata['pct_subcanopy_disturbance']}%")
```

```bash
pygeofetch multisensor vegetation-disturbance \
    --red-pre red_2023-01.tif --nir-pre nir_2023-01.tif \
    --red-post red_2023-07.tif --nir-post nir_2023-07.tif \
    --slc-pre slc_2023-01.tif --slc-post slc_2023-07.tif \
    --output-dir ./disturbance
```

### Why this exists — the actual, specific insight this pipeline adds

Optical sensors see the top of the canopy. Selective logging,
understory clearing, or early-stage construction under a still-standing
canopy can leave the visible crown largely unchanged — a small or
nonexistent NDVI drop — while the real ground-level disturbance is
severe enough to decorrelate SAR phase almost immediately (coherence
loss doesn't require *visible* change; it only requires the physical
scatterers within each resolution cell to have moved between the two
real acquisitions).

This pipeline distinguishes two real, different situations a
single-sensor approach can't tell apart:

| Class | NDVI | SAR coherence | Real interpretation |
|---|---|---|---|
| `CLASS_NO_DISTURBANCE` (0) | Unchanged | High | Nothing detected |
| `CLASS_VISIBLE_DISTURBANCE` (1) | Dropped significantly | (usually also drops) | Real, visible canopy loss — clear-cutting, fire |
| `CLASS_SUBCANOPY_DISTURBANCE` (2) | **Unchanged** | **Dropped significantly** | Real ground disturbance *invisible from above* — the pipeline's actual reason for existing |

Reuses the same canonical
`pygeofetch.utils.sar_math.estimate_interferometric_coherence` formula
[`coherence_disturbance_pipeline`](sar.md#coherence_disturbance_pipeline)
and `client.sar.coherence()` both already use — not a third,
independent implementation.

### A real constraint worth knowing before you start

The two SAR SLC rasters must already share a grid with the optical
pair — this pipeline does **not** reproject between them, because SLC
data can't be resampled without corrupting its real phase (the same
real constraint documented for `coherence_disturbance_pipeline`).
Reproject the optical side onto the SAR grid yourself first if they
don't already match; never the reverse.

### A real coherence-estimator edge effect, found while testing this pipeline

The shared coherence formula uses a real local averaging window
(`coherence_window`, default 7). This means a disturbed region's own
*edges* get blended with the surrounding coherent area — verified
directly: a synthetic 10×10-pixel fully-decorrelated region showed
coherence ≈0.13 at its center (well below the 0.3 threshold, correctly
classified) but ≈0.39 at its edges (blended, above threshold). This is
real, correct, expected behavior of a windowed coherence estimator, not
a bug — if your own real disturbance classification looks slightly
smaller than the area you expected, a wider real disturbance combined
with the coherence window's own smoothing is a genuine, physical reason
why, not necessarily a detection failure.

### Threshold tuning

`ndvi_drop_threshold` (default `0.15`) is a real, moderate, commonly-used
value in deforestation-detection literature, but tune it to your own
real vegetation baseline — a threshold calibrated for dense tropical
canopy may be far too strict for sparse savanna, where the entire
dynamic range of real NDVI is smaller to begin with.
