# Preprocessing Engine

The foundational optical-data toolkit underneath the InSAR/SAR/
Landsat/TimeSeries modules — the steps you almost always run *before*
computing an index or doing analysis: correcting for atmosphere and
terrain shadow, masking clouds, clipping/reprojecting to your AOI, and
combining multiple scenes into one.

```python
from pygeofetch import PyGeoFetch
client = PyGeoFetch()

corrected = client.preprocess.atmos("scene.tif", method="dos1")
masked = client.preprocess.cloud_mask("scene.tif", method="scl", scl_band="SCL.tif")
clipped = client.preprocess.clip("scene.tif", geometry="study_area.geojson")
reproj = client.preprocess.reproject("scene.tif", crs="EPSG:4326")
```

All methods live on `client.preprocess` (`pygeofetch.processing.preprocessor.Preprocessor`),
read via a block-by-block fallback (works on tiled/COG/compressed
inputs without a full-scene decode), and return a `ProcessingResult`
whose `.output_path` is the new raster on disk.

## Quick reference

| Method | Options | What it's for |
|---|---|---|
| `atmos()` | dos1, dos2, sen2cor, flaash, 6s, icor | Remove atmospheric haze/scattering before comparing reflectance across dates |
| `topo_correct()` | cosine, minnaert, c-correction | Remove terrain-shadow brightness differences (steep sunlit vs. shaded slopes) |
| `cloud_mask()` | scl, fmask, threshold, ndsi | Set cloud (and optionally shadow/snow) pixels to NoData |
| `cloud_fill()` | — | Fill cloud gaps using a multi-date time series |
| `clip()` | bbox or GeoJSON polygon | Crop to your AOI |
| `reproject()` | any target CRS | Change coordinate reference system |
| `resample()` | nearest, bilinear, cubic, lanczos | Change spatial resolution |
| `tile()` | — | Split a large raster into overlapping tiles |
| `pansharpen()` | brovey, ihs, gram-schmidt | Sharpen multispectral bands using a higher-resolution panchromatic band |
| `mosaic()` | first, last, min, max, sum | Merge scenes covering **different, adjacent areas** into one seamless image |
| `composite()` | median, mean, max, min, best_pixel | Merge scenes covering the **same area on different dates** into one cloud-free image |

```{tip}
**`mosaic()` vs. `composite()` — the distinction that trips people
up**: `mosaic()` stitches spatially *adjacent, non-overlapping* tiles
into one larger image (e.g. two Sentinel-2 tiles covering the east
and west halves of your AOI). `composite()` stacks the *same* area
across *multiple dates* to build one cloud-free image (e.g. 12 monthly
scenes of one tile, taking the per-pixel median to erase clouds that
happened to be over different parts on different dates). If your
inputs are "same place, different time," you want `composite()`, not
`mosaic()`.
```

## Atmospheric correction — `atmos()`

```python
result = client.preprocess.atmos("scene.tif", method="dos1")
```

**Why this matters**: sunlight passing through the atmosphere twice
(down to the surface, back up to the sensor) gets scattered and
absorbed — the raw digital numbers a satellite records aren't pure
surface reflectance, they include this atmospheric contribution. If
you're comparing one date's index values to another's (change
detection, time series), atmospheric differences between the two
acquisitions can masquerade as real surface change unless corrected
for first.

```{danger}
**Honest, verified limitation**: only `"dos1"`/`"dos2"` are genuine,
complete implementations. `"sen2cor"` is a **simplified** L1C→L2A
reflectance conversion (divide by 10000, the standard Sentinel-2
quantification value) — not the real, published Sen2Cor algorithm
(which does full atmospheric radiative transfer modeling with aerosol
and water-vapor retrieval). `"flaash"`, `"6s"`, and `"icor"` are
**placeholders** — calling any of them logs a warning and silently
falls back to DOS1, since those methods genuinely require an external
tool/executable this package doesn't bundle. If your workflow needs
one of those specific published algorithms rather than DOS1's dark-
object-subtraction approach, that real gap needs to be filled with the
actual external tool, not assumed to be running here.
```

**What DOS1/DOS2 actually do** (the two real implementations): find
the darkest 1st-percentile pixel value in each band (the "dark
object" — assumed to be a surface that should reflect near-zero, like
deep clear water or shadow) and subtract it from every pixel in that
band. DOS2 additionally applies a small path-radiance correction on
top. This is a real, standard, genuinely useful technique — just a
simpler one than full radiative-transfer atmospheric correction.

**Use it**: before any multi-date comparison (change detection, time
series indices); skip it for single-date, single-scene analysis where
absolute reflectance accuracy doesn't matter (e.g. just running NDVI
on one date to look at relative vegetation patterns within that one
scene).

## Topographic correction — `topo_correct()`

```python
result = client.preprocess.topo_correct("scene.tif", dem="dem.tif", method="cosine")
```

**Why this matters**: in mountainous terrain, a slope facing the sun
looks artificially bright and a shaded slope looks artificially dark
— purely because of illumination geometry, not because the actual
land cover is different. This can badly distort vegetation indices
and classification in hilly terrain if uncorrected. Needs a real DEM
covering the same AOI (see {doc}`/processing/terrain` for how to get
one via `client.search(..., providers=["opentopography"])`).

Three real methods: `"cosine"` (the simplest, based purely on the
illumination angle), `"minnaert"` (adds an empirical correction for
how "shiny vs. matte" the surface is), `"c-correction"` (adds an
empirical per-band correction term, generally the most accurate of
the three but needs enough valid pixels to fit that term reliably).

**Use it**: any AOI with meaningful topographic relief where you're
computing vegetation indices or doing classification. Skip it for
flat terrain (the correction would have negligible effect anyway).

## Cloud masking — `cloud_mask()`

```python
# Best option when you have it: real Sentinel-2 SCL classification band
result = client.preprocess.cloud_mask("scene.tif", method="scl", scl_band="SCL.tif")

# No SCL band available
result = client.preprocess.cloud_mask("scene.tif", method="fmask")
```

| `method=` | How it works | When to use it |
|---|---|---|
| `"scl"` | Reads a real Sentinel-2 Scene Classification Layer band and masks the real classified cloud/shadow classes (default: `[3,8,9,10,11]`) | **Best choice whenever you have an SCL band** — it's an official, per-pixel ESA classification, not a heuristic |
| `"fmask"` | A simplified brightness+whiteness heuristic (not the real, full FMask algorithm, which does multi-temporal + physical TOA tests) — needs at least 4 bands in blue/green/red/nir order | Reasonable fallback when no SCL band exists |
| `"threshold"` | Flags band-1 pixels above a brightness cutoff as cloud | Crude, single-band fallback — only reliable for obviously bright, thick cloud |
| `"ndsi"` | Computes NDSI internally and masks high-snow pixels — **requires bands in exactly `[blue, green, red, swir1]` order** (uses index 1 for green, index 3 for SWIR1) | Snow/ice removal, not actually cloud removal despite living on this method |

```{warning}
`"ndsi"` and `"fmask"` both assume a **specific band order** in your
input raster, not just a minimum band count — passing the same 4
bands in the wrong order silently produces a wrong mask rather than
an error. If your raster was stacked via `client.indices.stack()` or
{doc}`/processing/postprocessing`, double-check the order you passed
matches what these methods expect (blue, green, red, nir, ...).
```

`cloud_classes` (for `method="scl"`) lets you override which SCL class
values count as "cloud" — the real Sentinel-2 SCL legend is: 0=nodata,
1=saturated, 2=dark area, 3=cloud shadow, 4=vegetation, 5=bare soil,
6=water, 7=unclassified, 8=cloud medium probability, 9=cloud high
probability, 10=thin cirrus, 11=snow/ice. The default `[3,8,9,10,11]`
masks shadow + all cloud classes + cirrus + snow; pass your own list
if, e.g., you want to keep thin cirrus pixels rather than discard
them.

## `cloud_fill()` — filling gaps with other dates

```python
result = client.preprocess.cloud_fill(
    "scene_cloudy.tif",
    time_series=["scene_jan.tif", "scene_mar.tif"],
    method="interpolate",
)
```

Fills cloud/nodata gaps in `input` (a single target scene) using
valid pixels from `time_series` (other dates covering the same AOI).
`method="interpolate"` (default) linearly interpolates between the
nearest valid dates on either side of a gap; `method="nearest"` just
copies the value from whichever supplied date is temporally closest.
Real, useful when you want one specific date's raster filled in,
anchored to that date's actual values wherever they're valid — unlike
`composite()`, which blends every date together rather than treating
one as primary.

## Geometry operations

```python
clipped = client.preprocess.clip("scene.tif", bbox=(-74.1, 40.6, -73.7, 40.9))
clipped = client.preprocess.clip("scene.tif", geometry="study_area.geojson")
reproj = client.preprocess.reproject("scene.tif", crs="EPSG:4326")
resampled = client.preprocess.resample("scene.tif", resolution=10.0, method="bilinear")
```

`resample()` has three real, mutually-exclusive ways to specify the
target: `resolution=` (an absolute value in the raster's own CRS
units — degrees for geographic CRS, metres for projected), `scale_factor=`
(relative — `0.5` = half the current resolution, `2.0` = double), or
`reference=` (match another raster's grid *exactly* — same shape,
transform, *and* CRS, not just the same resolution number).

```{tip}
**Use `reference=`, not two separate `resolution=` calls, when you
need two rasters to align pixel-for-pixel** (e.g. before
`client.indices.ndvi()`, which needs its bands on the same grid).
Two rasters independently resampled to "the same resolution" can
still end up with different origins/extents and fail to overlay
correctly — `reference=` guarantees an exact match.
```

```{note}
**`clip()` CRS handling, verified:** a WGS84 boundary polygon (the
normal format for AOI GeoJSON) clipped against a raster in its native
UTM projection (the normal delivery format for real satellite
imagery) is automatically reprojected to match before masking —
confirmed against a real UTM Zone 30N test raster. This previously
failed silently with a near-empty intersection window rather than a
clear CRS error.
```

**Resampling method choice matters**: `"nearest"` preserves exact
original values (use for categorical/classified data — resampling a
land-cover class map with `"bilinear"` would create meaningless
fractional class values); `"bilinear"`/`"cubic"` smooth continuous
data like reflectance or elevation; `"lanczos"` is sharper but can
introduce ringing artifacts near hard edges — generally reserve it for
final visual output, not analysis inputs.

## `tile()` — splitting a large raster for chunked processing

```python
result = client.preprocess.tile("scene.tif", tile_size=256, overlap=32)
tiles = result.metadata["tile_paths"]
```

Splits a raster into overlapping square tiles (`tile_size` pixels,
`overlap` pixels shared between adjacent tiles) — useful for feeding
memory-constrained processing steps or a model that expects
fixed-size input chunks. `min_coverage` (default `0.1`) skips tiles
where less than that fraction of the tile is valid (non-nodata) data,
so edge tiles that are almost entirely nodata aren't written out.

## `pansharpen()` — multispectral + panchromatic → sharper multispectral

```python
sharp = client.preprocess.pansharpen("multispectral.tif", pan="panchromatic.tif", method="brovey")
```

Many satellites deliver a lower-resolution color image alongside a
higher-resolution grayscale ("panchromatic") band — pansharpening
combines them to approximate a higher-resolution color image. Three
real methods: `"brovey"` (fast, simple ratio-based, can shift colors
slightly on very bright targets), `"ihs"` (Intensity-Hue-Saturation
transform, generally better color preservation), `"gram-schmidt"`
(usually the best spectral fidelity of the three, at more compute
cost).

## Mosaic vs. composite — full detail

```python
# mosaic(): different areas, same time -- stitch tiles into one scene
result = client.preprocess.mosaic(["tile1.tif", "tile2.tif", "tile3.tif"], method="first")

# composite(): same area, different times -- build one cloud-free image
result = client.preprocess.composite(
    inputs=["jan.tif", "feb.tif", "mar.tif", ..., "dec.tif"],
    method="median",
)
```

`mosaic()` is a real, direct wrapper over `rasterio.merge` — a
well-tested library function, not a homegrown reimplementation.
`method` controls how overlapping pixels between adjacent tiles are
resolved: `"first"` (first input with valid data wins — fast, default),
`"last"`, `"min"`, `"max"`, `"sum"`.

`composite()` stacks every input date, then reduces per-pixel across
the time dimension: `"median"` (the standard choice — robust against
transient clouds/outliers without needing an explicit cloud mask),
`"mean"`, `"max"` (commonly used for max-NDVI compositing — pass NDVI
rasters as `inputs` to get the greenest pixel per location across the
whole period), `"min"`, `"best_pixel"` (picks the least-cloudy pixel
per location using `cloud_masks` if you supply them, rather than a
blind statistical reduction).

```{tip}
**Real, common workflow**: cloud-mask each date individually first
(`cloud_mask()`), *then* composite with `method="median"` — the
median naturally ignores the NaN/NoData gaps left by masking, giving
you a clean, cloud-free multi-month view without needing every single
input date to itself be cloud-free.
```

```{note}
Terrain-specific operations (`terrain_derivatives()`,
`topographic_wetness_index()`, `curvature()`,
`terrain_ruggedness_index()`, `identify_depressions()`,
`extract_drainage_network()`) also live on this same `Preprocessor`
class, reachable the same way (`client.preprocess.terrain_derivatives(...)`)
— documented separately on {doc}`/processing/terrain` since they form
a coherent topic of their own (DEM/DSM/DTM analysis) rather than
general-purpose optical preprocessing.
```
