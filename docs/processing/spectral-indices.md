# Spectral Indices

```bash
pip install "pygeofetch[processor]"
```

## What a spectral index actually is

If you're new to remote sensing: a satellite band is just one number
per pixel, measuring how much light bounced back at one specific
wavelength (e.g. "red light" or "near-infrared light"). A **spectral
index** is a simple formula that combines two or more bands into a
*new* single number per pixel, chosen specifically because that
combination correlates with something you actually care about —
"is this pixel healthy vegetation," "is this pixel open water," "did
this pixel just burn." You're not measuring vegetation or water
directly; you're measuring how surfaces made of vegetation, water, or
bare soil reflect light differently across bands, and exploiting that
difference.

The workhorse example: healthy leaves strongly reflect near-infrared
light (NIR) but absorb most red light (chlorophyll uses it for
photosynthesis). Bare soil or dead vegetation reflects red and NIR
about equally. So `(NIR - Red) / (NIR + Red)` — NDVI — comes out high
for healthy vegetation and low (or negative) for soil, water, or
built-up surfaces. Nearly every index on this page follows the same
pattern: pick two bands where a target surface type behaves very
differently, normalize the difference so it always falls in a
predictable range.

```{danger}
**Two separate classes exist, and only one is reachable via
`PyGeoFetch`** — the same real duplication pattern documented on
{doc}`/processing/sar`:

- **`client.indices`** (via `PyGeoFetch()`) is
  `pygeofetch.processing.indices.SpectralIndices` — one **dedicated
  method per index** (`client.indices.ndvi(red=..., nir=...)`), 17
  indices total, always available with no extra dependency. **This is
  the one almost every real workflow should use**, and everything on
  this page documents it.
- **`from pygeofetch.processor.indices import SpectralIndex`** is a
  *different* class with a generic `compute(index, **band_arrays)` /
  `from_files(index, **band_paths)` interface, and can reach 232+
  indices when `spyndex` is installed. Not accessible as
  `client.indices` — see the bottom of this page.

If in doubt, use `client.indices` — everything below is written
against it.
```

## Quick start

```python
from pygeofetch import PyGeoFetch

client = PyGeoFetch()

# Every method reads real band files directly and writes a real GeoTIFF
result = client.indices.ndvi(red="B04.tif", nir="B08.tif")

print(result.output_path)   # -> the new NDVI raster on disk
print(result.success)        # -> True
```

Every method:
- Accepts **file paths**, not in-memory arrays — pass the actual
  `.tif` files for each band (typically what `pf.download()` or
  {doc}`/processing/preprocessing` just produced).
- Reads via a block-by-block fallback, so it works directly on
  tiled/COG/compressed inputs without a full-scene decode crashing on
  large files.
- Returns a `ProcessingResult` — `.output_path` is a float32,
  DEFLATE-compressed, COG-tiled GeoTIFF. `NaN` marks nodata/invalid
  pixels (e.g. division by zero at a masked edge), not a garbage
  number.
- Bands don't need to already be the same shape/resolution — each
  method resamples secondary bands onto the first band's grid
  automatically.

## Finding your bands

Different providers name bands differently. For Sentinel-2 (the most
common source), the mapping most indices below need is:

| Common name | Sentinel-2 band | Wavelength | What it "sees" |
|---|---|---|---|
| Blue | B02 | ~490 nm | Water, atmosphere, shorelines |
| Green | B03 | ~560 nm | Vegetation vigor, water turbidity |
| Red | B04 | ~665 nm | Chlorophyll absorption |
| NIR | B08 | ~842 nm | Vegetation structure/health, water boundaries |
| SWIR1 | B11 | ~1610 nm | Moisture content, built-up areas, snow/cloud discrimination |
| SWIR2 | B12 | ~2190 nm | Burn severity, mineral/soil composition |

If you downloaded via `pf.download(results, "./data", bands=["B02","B03","B04","B08"])`
(see {doc}`/core-features/download`), you already have exactly the
files you need, named by their real band codes.

## Vegetation indices — "how healthy/dense is the plant life here"

### NDVI — the one to reach for first

```python
ndvi = client.indices.ndvi(red="B04.tif", nir="B08.tif")
```

`(NIR - Red) / (NIR + Red)`. The single most widely used vegetation
index in remote sensing — start here unless you have a specific
reason not to.

**Reading the values** (range always −1 to +1):

| Value | Meaning |
|---|---|
| < 0 | Water, clouds, snow |
| 0 to 0.2 | Bare soil, rock, sand, urban surfaces |
| 0.2 to 0.3 | Sparse/stressed vegetation, grassland |
| 0.3 to 0.6 | Moderate, healthy vegetation |
| > 0.6 | Dense, vigorous vegetation (forest, irrigated crops at peak growth) |

**Use it for**: crop health monitoring, deforestation detection
(watch NDVI drop over time in one area), drought stress screening,
general "how much live green vegetation is here" questions.

**Known limitation**: NDVI saturates at high vegetation density — a
sparse young forest and a dense old-growth forest can both read
"~0.8," so it's poor at distinguishing *among* already-healthy
canopies. It's also sensitive to bare soil showing through sparse
canopy (see SAVI below for a fix).

### EVI — NDVI's fix for dense canopy and atmospheric noise

```python
evi = client.indices.evi(blue="B02.tif", red="B04.tif", nir="B08.tif", G=2.5, C1=6.0, C2=7.5, L=1.0)
```

`G * (NIR-Red) / (NIR + C1*Red - C2*Blue + L)`. Uses the blue band to
correct for atmospheric scattering and canopy background noise that
NDVI doesn't account for. Better than NDVI specifically over dense
canopy (tropical forest, closed-canopy crops) where NDVI has already
saturated. The default coefficients (`G=2.5, C1=6.0, C2=7.5, L=1.0`)
are the standard MODIS EVI algorithm constants — only change them if
you have a specific, published reason to.

**Use it for**: dense forest canopy monitoring where NDVI has
plateaued; anywhere atmospheric haze is a real concern.

### SAVI — NDVI's fix for visible soil background

```python
savi = client.indices.savi(red="B04.tif", nir="B08.tif", L=0.5)
```

`(NIR-Red)/(NIR+Red+L) * (1+L)`. NDVI over sparse vegetation gets
pulled around by how much bare soil is visible between plants — SAVI
corrects for that with a soil-brightness constant `L` (default `0.5`,
the standard value for "intermediate" vegetation density; use lower
`L` for denser cover, higher for sparser).

**Use it for**: arid/semi-arid regions, early-season crops, rangeland
— anywhere a meaningful fraction of each pixel is exposed soil, not
just canopy.

## Water indices — "is this pixel open water"

### NDWI — the default choice

```python
ndwi = client.indices.ndwi(green="B03.tif", nir="B08.tif")
```

`(Green - NIR) / (Green + NIR)` (McFeeters 1996). Water strongly
absorbs NIR, so NDWI comes out positive over open water and negative
over vegetation/soil.

**Reading the values**: positive = water, negative = land. The exact
threshold for "definitely water" varies by scene, but `> 0` is a
reasonable starting cutoff.

**Known limitation**: NDWI often misclassifies built-up areas as
water, because urban materials can also produce a positive value —
see MNDWI below if your AOI includes cities.

### MNDWI — better in urban/built-up scenes

```python
mndwi = client.indices.mndwi(green="B03.tif", swir1="B11.tif")
```

`(Green - SWIR1) / (Green + SWIR1)` (Xu 2006). Swaps NIR for SWIR1,
which meaningfully improves separation between water and built-up
surfaces — use this instead of NDWI whenever your AOI has cities,
towns, or other built infrastructure near the water you're mapping.

**Use it for**: flood mapping near urban areas, reservoir monitoring
close to settlements, coastal change detection where shoreline
development is present.

## Built-up & bare-soil indices

### NDBI — built-up area extraction

```python
ndbi = client.indices.ndbi(nir="B08.tif", swir1="B11.tif")
```

`(SWIR1 - NIR) / (SWIR1 + NIR)` (Zha 2003). Positive over urban/
built-up surfaces, negative over vegetation. Note this is the mirror
image of NDVI's band pair (SWIR1 in place of Red, NIR still NIR) —
built-up materials reflect SWIR1 strongly and NIR weakly, the
opposite pattern from healthy vegetation.

**Use it for**: urban growth monitoring over time, impervious-surface
mapping, distinguishing built-up land from bare soil (which NDVI
alone can't reliably separate).

## Fire & burn severity

### NBR and dNBR — the standard fire-mapping pair

```python
nbr = client.indices.nbr(nir="B08.tif", swir2="B12.tif")

# For actual burn severity, compute pre- and post-fire NBR, then difference:
dnbr = client.indices.dnbr(
    pre_nir="pre_B08.tif", pre_swir2="pre_B12.tif",
    post_nir="post_B08.tif", post_swir2="post_B12.tif",
)
```

`NBR = (NIR - SWIR2) / (NIR + SWIR2)`. Healthy vegetation has high
NIR and low SWIR2 reflectance; burned areas invert this (charred
material and exposed soil both raise SWIR2, ash and canopy loss drop
NIR). `dNBR = NBR_pre - NBR_post` — a single call that reads all four
bands and does the pre/post subtraction for you, so you don't need to
call `nbr()` twice and subtract manually.

**Reading dNBR values** (the real USGS burn-severity classification):

| dNBR | Severity |
|---|---|
| < −0.25 | Regrowth (vegetation increased since the reference date — not a burn signal at all) |
| −0.25 to 0.1 | Unburned |
| 0.1 to 0.27 | Low severity |
| 0.27 to 0.44 | Moderate-low severity |
| 0.44 to 0.66 | Moderate-high severity |
| > 0.66 | High severity |

**Use it for**: post-fire burn severity mapping, fire perimeter
delineation, forest recovery monitoring (watch dNBR trend back toward
zero over subsequent years).

## Snow & moisture

```python
ndsi = client.indices.ndsi(green="B03.tif", swir1="B11.tif")
ndmi = client.indices.ndmi(nir="B08.tif", swir1="B11.tif")
```

- **NDSI** (Hall 1995): `(Green - SWIR1) / (Green + SWIR1)`. Snow is
  highly reflective in visible light but absorbs SWIR strongly —
  values above `0.4` typically indicate snow cover. Also useful for
  discriminating snow from clouds (both are bright in visible light,
  but clouds don't show the same SWIR absorption).
- **NDMI** (Wilson & Sader 2002): `(NIR - SWIR1) / (NIR + SWIR1)`.
  Sensitive to canopy water content — positive values indicate moist,
  well-watered vegetation; useful for drought stress monitoring
  alongside NDVI (a canopy can still look "green" on NDVI while
  already water-stressed on NDMI).

## Transforms and general-purpose tools

```python
tct = client.indices.tct(blue, green, red, nir, swir1, swir2, sensor="sentinel2")
pca_result = client.indices.pca(inputs=[b02, b03, b04, b08], n_components=3)
texture = client.indices.texture(input="B08.tif", window=5, features=["contrast", "homogeneity"])
lst = client.indices.lst(thermal="B10.tif", emissivity=0.97, sensor="landsat8")
albedo = client.indices.albedo(inputs=[b02, b03, b04, b08, b11, b12], sensor="sentinel2")
```

| Method | What it's for | Real basis |
|---|---|---|
| `tct()` | **Tasseled Cap Transformation** — 3-band Brightness/Greenness/Wetness summary, a classic land-cover-change screening tool | Nedkov (2017) for Sentinel-2, Baig et al. (2014) for Landsat-8 — real published coefficients |
| `pca()` | **Principal Component Analysis** — compress N correlated bands into fewer components that capture most of the variance, useful before classification | Real PCA over an arbitrary input band list |
| `texture()` | **GLCM texture features** — captures spatial pattern (roughness, uniformity), not just spectral value; useful for distinguishing surfaces that look similar spectrally but differ in texture (e.g. urban vs. bare soil) | `contrast`, `dissimilarity`, `homogeneity`, `energy`, `correlation`, `ASM` — via scipy, not a slow Python loop |
| `lst()` | **Land Surface Temperature** from a thermal band, in real Kelvin/Celsius | Real Landsat 8/9 Band 10 thermal constants (K1=774.8853, K2=1321.0789) |
| `albedo()` | **Narrowband-to-broadband surface albedo** | Liang (2001) published coefficients |

See {doc}`/processing/spectral-indices`'s full method reference
(below) for every parameter of each.

### General-purpose escape hatches

```python
custom = client.indices.band_math(
    inputs=[red_path, nir_path],
    expression="(B[1] - B[0]) / (B[1] + B[0] + 1e-6)",
)
stacked = client.indices.stack(inputs=[b02, b03, b04, b08])   # multi-band GeoTIFF
```

`band_math()` lets you compute anything not already covered — `B[0]`,
`B[1]`, etc. refer to your `inputs` list in order, `np` is available
for any numpy function.

```{danger}
`band_math()`'s `expression` is evaluated with Python's `eval()`
(`B` and `np` are the only names exposed). Fine for expressions you
write yourself; **never pass an `expression` string from untrusted
user input** — it is not sandboxed against arbitrary code execution.
```

## Common pitfalls

- **Mismatched band resolutions.** Sentinel-2's bands aren't all the
  same resolution (B02/B03/B04/B08 are 10m, B11/B12 are 20m). Every
  method here resamples automatically onto the first band's grid, so
  mixing resolutions "just works" — but be aware you're implicitly
  either upsampling the 20m bands or losing the extra detail in the
  10m ones, depending on argument order.
- **NaN, not zero, at invalid pixels.** A masked/nodata pixel becomes
  `NaN` in the output, not `0`. If you're computing statistics
  downstream, use NaN-aware functions (`np.nanmean`, not `np.mean`) or
  you'll silently get wrong numbers.
- **Comparing indices across dates without atmospheric correction.**
  Raw reflectance (and therefore any index computed from it) shifts
  with atmospheric conditions, sun angle, and sensor calibration
  drift — comparing NDVI from two dates processed differently can show
  "change" that's really just atmospheric noise. See
  {doc}`/processing/preprocessing`'s `atmos()` step if you're doing
  real change detection, not just a single-date snapshot.
- **`dNBR`'s sign convention.** It's `pre - post`, so a *positive*
  value means the surface got *less* vegetated (burned); this is the
  opposite sign convention from indices like NDVI change, where you'd
  naturally compute `post - pre`. Double-check which direction you
  actually subtracted before interpreting "positive = worse."

## Full method reference

| Method | Formula | Citation |
|---|---|---|
| `ndvi(red, nir)` | `(NIR-Red)/(NIR+Red)` | — |
| `evi(blue, red, nir, G=2.5, C1=6.0, C2=7.5, L=1.0)` | `G*(NIR-Red)/(NIR+C1*Red-C2*Blue+L)` | Standard MODIS EVI constants |
| `savi(red, nir, L=0.5)` | `(NIR-Red)/(NIR+Red+L)*(1+L)` | — |
| `ndwi(green, nir)` | `(Green-NIR)/(Green+NIR)` | McFeeters 1996 |
| `mndwi(green, swir1)` | `(Green-SWIR1)/(Green+SWIR1)` | Xu 2006 |
| `ndbi(nir, swir1)` | `(SWIR1-NIR)/(SWIR1+NIR)` | Zha 2003 |
| `ndsi(green, swir1)` | `(Green-SWIR1)/(Green+SWIR1)` | Hall 1995 |
| `ndmi(nir, swir1)` | `(NIR-SWIR1)/(NIR+SWIR1)` | Wilson & Sader 2002 |
| `nbr(nir, swir2)` | `(NIR-SWIR2)/(NIR+SWIR2)` | — |
| `dnbr(pre_nir, pre_swir2, post_nir, post_swir2)` | `NBR_pre - NBR_post` | USGS burn-severity scale |
| `tct(blue, green, red, nir, swir1, swir2, sensor="sentinel2")` | 3-band linear transform | Nedkov 2017 / Baig et al. 2014 |
| `pca(inputs, n_components=3)` | Principal component analysis | — |
| `texture(input, window=5, features=None)` | GLCM texture features | — |
| `lst(thermal, emissivity=0.97, sensor="landsat8")` | Thermal band → real Kelvin/Celsius | Real Landsat thermal constants |
| `albedo(inputs, sensor="sentinel2")` | Narrowband-to-broadband | Liang 2001 |
| `band_math(inputs, expression)` | Arbitrary expression | — |
| `stack(inputs)` | Multi-band GeoTIFF | — |

## The standalone, `spyndex`-backed `SpectralIndex`

```python
from pygeofetch.processor.indices import SpectralIndex

si = SpectralIndex()
ndvi = si.compute("NDVI", RED=red_array, NIR=nir_array)          # in-memory arrays
ndvi = si.from_files("NDVI", red="B04.tif", nir="B08.tif", output="ndvi.tif")  # files, like client.indices

si.available()   # -> list of all available index names
si.info("NDVI")   # -> formula, required bands, valid range
```

Without `spyndex` installed, 17 built-in formulae (matching
`client.indices`'s coverage, though via a different, generic call
shape) still work — `NDVI`, `EVI`, `SAVI`, `NDWI`, `MNDWI`, `NDBI`,
`NDSI`, `NDMI`, `NBR`, `dNBR`, `BSI`, `ARVI`, `GNDVI`, `RVI`, `VCI`,
`CRI1`, `PSRI`. With `spyndex` installed (`pip install
"pygeofetch[processor]"` already includes it), `si.available()`
returns spyndex's much larger published catalogue (232+ indices)
instead, transparently.

Band names are matched via a real alias table (`RED`/`R`, `NIR`/`N`,
`B04`/`R`, `B08`/`N`, etc.), so both spyndex's short codes and common
long-form names work as keyword arguments.

Reach for this instead of `client.indices` when you need one of
spyndex's less-common 200+ indices, or want in-memory-array input
without writing to a file first.
