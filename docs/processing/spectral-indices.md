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

:::{danger}

**Two separate classes exist, and only one is reachable via
`PyGeoFetch`** — the same real duplication pattern documented on
[SAR Processing](sar.md):

- **`client.indices`** (via `PyGeoFetch()`) is
  `pygeofetch.processing.indices.SpectralIndices` — one **dedicated
  method per index** (`client.indices.ndvi(red=..., nir=...)`), 18
  indices total, always available with no extra dependency. **This is
  the one almost every real workflow should use**, and everything on
  this page documents it.
- **`from pygeofetch.processor.indices import SpectralIndex`** is a
  *different* class with a generic `compute(index, **band_arrays)` /
  `from_files(index, **band_paths)` interface, and can reach 280
  indices when `spyndex` is installed (confirmed directly against
  spyndex's real, current catalogue — not the 232 an earlier pass
  of this page assumed), plus 6 real geology/mineral-exploration
  indices spyndex itself doesn't have. Not accessible as
  `client.indices` — see the bottom of this page.

If in doubt, use `client.indices` — everything below is written
against it.
:::
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
  [Preprocessing Engine](preprocessing.md) just produced).
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
(see [Downloading Satellite Data](../core-features/download.md)), you already have exactly the
files you need, named by their real band codes.

### One merged multi-band image? Skip finding separate files entirely

Drone multispectral orthomosaics (MicaSense RedEdge/Altum, DJI P4
Multispectral, and similar) typically arrive as one file with every
band already merged — not separate per-band GeoTIFFs like the
Sentinel-2 workflow above. Every method that needs two or more bands
can read all of them from that single file directly:

```python
# Same path passed to every role -- pygeofetch reads the file's own
# real, embedded band descriptions and resolves the correct band for
# each one automatically.
ndre = client.indices.ndre(rededge="drone_orthomosaic.tif", nir="drone_orthomosaic.tif")
evi = client.indices.evi(blue="drone.tif", red="drone.tif", nir="drone.tif")
```

This works by matching the file's real band descriptions (e.g.
`"Red edge"`, `"NIR"`) against a built-in synonym table covering
common sensor naming conventions. Check what your file actually
reports before relying on this:

```python
from pygeofetch.processing.base import detect_band_roles

print(detect_band_roles("drone_orthomosaic.tif"))
# -> {'blue': 1, 'green': 2, 'red': 4, 'rededge': 5, 'nir': 6}
```

If a role is missing from the result, your file's descriptions don't
match a known synonym for it (or the file has no embedded
descriptions at all — common; many drone processing tools don't write
them). Calling an index method with the same path for two roles in
that case fails honestly — `result.success` is `False` and
`result.error` names exactly which role(s) couldn't be resolved —
rather than silently reading the wrong band. Pass separately
pre-extracted single-band files instead, or pass an explicit
`(path, band_index)` tuple for any role if you already know the real
band number:

```python
ndre = client.indices.ndre(rededge=("drone.tif", 5), nir=("drone.tif", 6))
```

This auto-detection currently covers `ndvi`, `ndre`, `evi`, `savi`,
`ndwi`, `mndwi`, `ndbi`, `ndsi`, `ndmi`, `nbr`, and `band_math` — see
[Processing large rasters](#processing-large-rasters-chunked-mode)
below for the other real addition that goes with it.

### Clipping the result to an AOI

The same 11 methods also accept `bbox`, `geometry`, and `geometry_crs`
— if either is given, the computed result is clipped afterward, using
the same real bbox/GeoJSON-file/dict/shapely-geometry handling and
automatic CRS reprojection as
[`Preprocessor.clip()`](preprocessing.md). A no-op if neither is
supplied — the common case, where results stay at their input bands'
full extent.

```python
ndvi = client.indices.ndvi(
    red="B04.tif", nir="B08.tif",
    bbox=(-74.05, 40.68, -73.95, 40.78),   # minx, miny, maxx, maxy, WGS84 by default
)

# Or a real polygon boundary instead of a rectangle
ndvi = client.indices.ndvi(
    red="B04.tif", nir="B08.tif",
    geometry="farm_boundary.geojson",
)
```

Set `geometry_crs` if your bbox/geometry coordinates are already in a
projected CRS matching the raster, rather than WGS84 lat/lon (the
default). This combines with `chunked` and the multi-band
auto-detection above — clipping always happens as the last step,
after the index itself is fully computed.

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

### NDRE — NDVI's fix for saturation in dense canopy

```python
ndre = client.indices.ndre(rededge="rededge_band.tif", nir="B08.tif")
```

`(NIR - RedEdge) / (NIR + RedEdge)` (Barnes et al. 2000). Uses the red
edge band (~705-750nm, the steep reflectance transition between red
absorption and NIR reflectance) instead of red — this makes NDRE
meaningfully less prone to the saturation problem that limits NDVI in
dense canopy, since the red edge keeps responding to chlorophyll
changes past the point where red has already bottomed out.

Most satellites that carry NDVI's Red/NIR pair don't carry a red edge
band at all (Landsat doesn't; Sentinel-2 does, as B05/B06/B07). Red
edge sensors are far more common on **drone multispectral payloads**
built specifically for precision agriculture — MicaSense RedEdge/
Altum, DJI P4 Multispectral, and similar — which is where NDRE sees
most of its real, practical use.

**Use it for**: crop stress and chlorophyll content monitoring in
precision agriculture, especially from drone imagery; anywhere NDVI
has already saturated but you need finer discrimination within
already-dense canopy.

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

See [Spectral Indices](spectral-indices.md)'s full method reference
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

:::{danger}

`band_math()`'s `expression` is evaluated with Python's `eval()`
(`B` and `np` are the only names exposed). Fine for expressions you
write yourself; **never pass an `expression` string from untrusted
user input** — it is not sandboxed against arbitrary code execution.
:::
## Processing large rasters (chunked mode)

Every method's default path reads full bands into memory — fine for
typical satellite scenes, but a real problem for large drone
orthomosaics, which can exceed both available RAM and the standard
GeoTIFF format's hard 4GB file-size limit (internal byte offsets are
32-bit). Writing a large result the normal way fails with a raw,
unhelpful `TIFFAppendToStrip: Maximum TIFF file size exceeded` error
that gives no hint what actually went wrong.

`chunked=True` processes the raster in fixed-size tiles instead of
one full-array read/write, and writes a properly tiled, BigTIFF-safe
output:

```python
ndvi = client.indices.ndvi(red="huge_drone_band.tif", nir="huge_drone_band2.tif",
                            chunked=True, tile_size=1024)
```

- **Default is `chunked="auto"`**, not always-on. On every call, this
  compares the real, on-disk size of the actual inputs against real
  available system memory (via `psutil`, if installed — falls back to
  a conservative 500MB total-input-size threshold otherwise) and
  decides automatically. Small, ordinary satellite scenes get the
  fast, direct path with no per-tile overhead; a large drone
  orthomosaic gets chunked without you needing to know the parameter
  exists. Pass `chunked=True` or `chunked=False` explicitly to force
  one path regardless of size.
- **A real, deliberate correction worth stating plainly**: chunking's
  proven, reliable benefit is memory safety, not speed. Direct
  measurement in this project found chunking roughly a wash on
  processing time for a single-threaded run — sometimes marginally
  faster, sometimes slower, never reliably faster. If you need this
  to run faster, not just handle a bigger file, that requires genuine
  multi-core parallelism, which chunking alone does not provide.
- **`tile_size`** (default `1024`) controls memory usage, not output
  quality — smaller tiles use less RAM per step at the cost of more
  Python-level iterations. It does not need to be a multiple of 16;
  the underlying GeoTIFF block size is validated and rounded
  separately, automatically.
- Chunked and non-chunked paths are numerically identical — verified
  directly against a non-chunked reference computation, including at
  tile boundaries with a deliberately non-clean tile size, with zero
  difference.
- **Every real output raster now writes with `BIGTIFF=YES`
  unconditionally** — even without `chunked=True` — since this has no
  real downside for small files (GDAL only uses the 64-bit offset
  format when a file actually needs it) and fully prevents the 4GB
  crash on its own.

Combine chunking with the auto-detection above for the real, common
drone workflow — one large merged image, one method call, no manual
band splitting and no memory blowup:

```python
ndre = client.indices.ndre(rededge="huge_drone_orthomosaic.tif",
                            nir="huge_drone_orthomosaic.tif",
                            chunked=True)
```

**Currently supported**: `ndvi`, `ndre`, `evi`, `savi`, `ndwi`,
`mndwi`, `ndbi`, `ndsi`, `ndmi`, `nbr`, `band_math`. **Not yet
extended**: `dnbr`, `tct`, `pca`, `texture`, `lst`, `albedo` — these
still read full bands into memory and are unaffected by `chunked`
(the parameter doesn't exist on them). `texture` specifically needs
tile overlap to compute its neighborhood-based GLCM features
correctly at tile boundaries, not just the same fixed-tile approach
used here — naively chunking it the same way would produce wrong
values along every tile seam.

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
  [Preprocessing Engine](preprocessing.md)'s `atmos()` step if you're doing
  real change detection, not just a single-date snapshot.
- **`dNBR`'s sign convention.** It's `pre - post`, so a *positive*
  value means the surface got *less* vegetated (burned); this is the
  opposite sign convention from indices like NDVI change, where you'd
  naturally compute `post - pre`. Double-check which direction you
  actually subtracted before interpreting "positive = worse."

## Full method reference

| Method | Formula | Citation | `chunked`/auto-detect/clip |
|---|---|---|---|
| `ndvi(red, nir)` | `(NIR-Red)/(NIR+Red)` | — | ✅ |
| `ndre(rededge, nir)` | `(NIR-RedEdge)/(NIR+RedEdge)` | Barnes et al. 2000 | ✅ |
| `evi(blue, red, nir, G=2.5, C1=6.0, C2=7.5, L=1.0)` | `G*(NIR-Red)/(NIR+C1*Red-C2*Blue+L)` | Standard MODIS EVI constants | ✅ |
| `savi(red, nir, L=0.5)` | `(NIR-Red)/(NIR+Red+L)*(1+L)` | — | ✅ |
| `ndwi(green, nir)` | `(Green-NIR)/(Green+NIR)` | McFeeters 1996 | ✅ |
| `mndwi(green, swir1)` | `(Green-SWIR1)/(Green+SWIR1)` | Xu 2006 | ✅ |
| `ndbi(nir, swir1)` | `(SWIR1-NIR)/(SWIR1+NIR)` | Zha 2003 | ✅ |
| `ndsi(green, swir1)` | `(Green-SWIR1)/(Green+SWIR1)` | Hall 1995 | ✅ |
| `ndmi(nir, swir1)` | `(NIR-SWIR1)/(NIR+SWIR1)` | Wilson & Sader 2002 | ✅ |
| `nbr(nir, swir2)` | `(NIR-SWIR2)/(NIR+SWIR2)` | — | ✅ |
| `dnbr(pre_nir, pre_swir2, post_nir, post_swir2)` | `NBR_pre - NBR_post` | USGS burn-severity scale | ❌ |
| `tct(blue, green, red, nir, swir1, swir2, sensor="sentinel2")` | 3-band linear transform | Nedkov 2017 / Baig et al. 2014 | ❌ |
| `pca(inputs, n_components=3)` | Principal component analysis | — | ❌ |
| `texture(input, window=5, features=None)` | GLCM texture features | — | ❌ |
| `lst(thermal, emissivity=0.97, sensor="landsat8")` | Thermal band → real Kelvin/Celsius | Real Landsat thermal constants | ❌ |
| `albedo(inputs, sensor="sentinel2")` | Narrowband-to-broadband | Liang 2001 | ❌ |
| `band_math(inputs, expression)` | Arbitrary expression | — | ✅ |
| `stack(inputs)` | Multi-band GeoTIFF | — | ❌ |

Every method marked ✅ accepts `chunked=True, tile_size=1024` for
large rasters, auto-detects bands when the same path is passed for
more than one role, and accepts `bbox`/`geometry`/`geometry_crs` to
clip the result afterward — see

[Processing large rasters](#processing-large-rasters-chunked-mode),
[Finding your bands](#finding-your-bands), and
[Clipping the result to an AOI](#clipping-the-result-to-an-aoi) above.
Methods marked ❌ don't have any of these parameters yet.

## CLI reference

Every ✅ method above is also available from the command line, under
`pygeofetch index`:

```bash
pygeofetch index ndvi --red B04.tif --nir B08.tif
pygeofetch index ndre --rededge rededge_band.tif --nir nir_band.tif

# Same merged image passed to both roles -- auto-detected exactly like the Python API
pygeofetch index ndre --rededge drone.tif --nir drone.tif

# Large raster, memory-safe tiled processing
pygeofetch index ndvi --red huge.tif --nir huge2.tif --chunked --tile-size 1024

# Clip the result to an AOI -- bbox or a GeoJSON file
pygeofetch index ndvi --red B04.tif --nir B08.tif --bbox -74.05,40.68,-73.95,40.78
pygeofetch index ndvi --red B04.tif --nir B08.tif --geometry farm_boundary.geojson
```

`--chunked` is a flag (no value); `--tile-size` takes an integer and
is only used when `--chunked` is set. `--bbox` takes exactly 4
comma-separated numbers (`minx,miny,maxx,maxy`) — anything else fails
immediately with a clear error, before any processing starts.
`--geometry` takes a GeoJSON file path instead. `--geometry-crs`
(default `EPSG:4326`) sets the CRS those coordinates are in. Run
`pygeofetch index COMMAND --help` for the full option list of any
individual command — e.g. `pygeofetch index evi --help`.

## The standalone, `spyndex`-backed `SpectralIndex`

```python
from pygeofetch.processor.indices import SpectralIndex

si = SpectralIndex()
ndvi = si.compute("NDVI", RED=red_array, NIR=nir_array)          # in-memory arrays
ndvi = si.from_files("NDVI", red="B04.tif", nir="B08.tif", output="ndvi.tif")  # files, like client.indices

si.available()   # -> list of all available index names
si.info("NDVI")   # -> formula, required bands, valid range
```

Without `spyndex` installed, **23** built-in formulae still work — the
original 17 (`NDVI`, `EVI`, `SAVI`, `NDWI`, `MNDWI`, `NDBI`, `NDSI`,
`NDMI`, `NBR`, `dNBR`, `BSI`, `ARVI`, `GNDVI`, `RVI`, `VCI`, `CRI1`,
`PSRI`), plus 6 real geology/mineral-exploration indices added this
pass (see below). With `spyndex` installed (`pip install
"pygeofetch[processor]"` already includes it), `si.available()`
returns the real union of both — confirmed directly against a real
bug this used to have: it previously returned *either* spyndex's list
*or* the built-in list, never both, silently hiding the 6 geology
indices below from discovery whenever spyndex happened to be
installed, even though `compute()` could still run them.

Band names are matched via a real alias table (`RED`/`R`, `NIR`/`N`,
`B04`/`R`, `B08`/`N`, etc.), so both spyndex's short codes and common
long-form names work as keyword arguments.

Reach for this instead of `client.indices` when you need one of
spyndex's less-common indices (spyndex's real, current catalogue has
**280** — checked directly, not the 232 an earlier pass of this page
assumed), or want in-memory-array input without writing to a file
first.

:::{admonition} A real, confirmed bug in `RVI` was fixed here
:class: danger

Every built-in formula's result used to be clipped to `[-1, 1]`,
correct for a normalized-difference index like `NDVI` but **wrong**
for a genuine ratio index like `RVI` (Ratio Vegetation Index =
`NIR / RED`), which has no such bound. Confirmed directly: for
healthy vegetation (`NIR=0.4`, `RED=0.1`), the real RVI value is
`4.0` — the old code silently returned `1.0` instead, for any pixel
where the numerator exceeded the denominator, which is essentially
all healthy vegetation. `RVI` is now correctly exempted from the
`[-1, 1]` clip, alongside `DNBR` (which was already, correctly,
exempted) and the 6 new geology indices below.
:::
### Geology & mineral-exploration indices

Six real, verified indices from
[Geopera's spectral indices reference](https://docs.geopera.com/spectral-indices)
(CC-BY-4.0), added because neither the built-in 17 nor spyndex's own
280-index catalogue covered this domain at all — confirmed directly,
not assumed.

| Index | Formula | Sensor requirement |
|---|---|---|
| `FOX` (Ferric Oxides) | `NIR / RED` | Sentinel-2 / Landsat — computable directly |
| `AMP` (Amphibole) | `SWIR1 / SWIR2` | Sentinel-2 / Landsat — computable directly |
| `AKP` (Alunite/Kaolinite/Pyrophylite) | `(SWIR1 + SWIR3) / SWIR2` | Needs a real, distinct `SWIR3` band — **not** available on Sentinel-2 or Landsat (both carry only two SWIR bands total) |
| `ALT` (Alteration) | `SWIR3 / SWIR5` | Needs real, distinct `SWIR3` and `SWIR5` — ASTER-class sensor required |
| `FEI` (Ferrous Iron) | `(SWIR5 / RED) + (NIR1 / GREEN)` | Needs real, distinct `SWIR5` and `NIR1` — a sensor with multiple NIR/SWIR channels (e.g. WorldView-3) required |
| `GOS` (Gossan) | `SWIR4 / RED` | Needs a real, distinct `SWIR4` band — ASTER-class sensor required |

Only `FOX` and `AMP` work with the SWIR bands Sentinel-2/Landsat
actually provide. The other four are real, correct formulas, but
genuinely need a sensor with more distinct SWIR/NIR channels than the
most commonly free-available sensors carry — calling them with
Sentinel-2 data raises a clear error naming exactly which band is
missing, rather than silently computing something wrong with a
substituted band:

```python
si.compute("GOS", RED=red_array)
# ValueError: Missing band for index GOS: 'SWIR4'. Provided: ['RED']. GOS needs: SWIR4, RED
```

`si.info("AKP")` returns the real formula, required bands, and a
`sensor_note` explaining the same real constraint in prose.

