# Spectral Indices

```bash
pip install "pygeofetch[processor]"
```

```{danger}
**Same real duplication pattern as** {doc}`/processing/sar`: there are
two separate, differently-shaped classes, and only one is reachable
via `PyGeoFetch`.

- **`client.indices`** (via `PyGeoFetch()`) is
  `pygeofetch.processing.indices.SpectralIndices` — one **dedicated
  method per index** (`client.indices.ndvi(red=..., nir=...)`), 17
  indices total, always available with no extra dependency. **This is
  the one almost every real workflow should use.**
- **`from pygeofetch.processor.indices import SpectralIndex`** is a
  *different* class with a generic `compute(index, **band_arrays)` /
  `from_files(index, **band_paths)` interface, and can reach 232+
  indices when `spyndex` is installed. Not accessible as
  `client.indices`.

Both are real and independently useful — pick based on whether you
want `client.indices.ndvi(...)`-style discoverability (17 indices,
zero extra deps) or `spyndex`'s much larger catalogue behind one
generic call.
```

## `client.indices` — the primary, wired-in engine

```python
from pygeofetch import PyGeoFetch

client = PyGeoFetch()

ndvi = client.indices.ndvi(red="B04.tif", nir="B08.tif")
evi = client.indices.evi(blue="B02.tif", red="B04.tif", nir="B08.tif")
ndwi = client.indices.ndwi(green="B03.tif", nir="B08.tif")
```

Every method accepts per-band **file paths** (not arrays), reads them
via the same block-by-block fallback used throughout pygeofetch's
processing layer (so it works on tiled/COG/compressed inputs), and
returns a `ProcessingResult` whose `output_path` is a float32,
DEFLATE-compressed, COG-tiled GeoTIFF. NaN marks nodata/invalid
pixels.

### Vegetation & general-purpose indices

| Method | Formula | Notes |
|---|---|---|
| `ndvi(red, nir)` | `(NIR-Red)/(NIR+Red)` | Range −1..+1; values > 0.3 typically indicate healthy vegetation |
| `evi(blue, red, nir, G=2.5, C1=6.0, C2=7.5, L=1.0)` | `G*(NIR-Red)/(NIR+C1*Red-C2*Blue+L)` | Reduces atmospheric/canopy-background effects vs. NDVI |
| `savi(red, nir, L=0.5)` | Soil-adjusted NDVI variant | Better than NDVI over sparse vegetation with exposed soil |
| `gndvi` — *not present as a dedicated method here*; use `band_math()` (see below) or the standalone `SpectralIndex` class | | |

### Water & moisture indices

| Method | Formula |
|---|---|
| `ndwi(green, nir)` | `(Green-NIR)/(Green+NIR)` — open water |
| `mndwi(green, swir1)` | `(Green-SWIR1)/(Green+SWIR1)` — better than NDWI in built-up areas |
| `ndmi(nir, swir1)` | `(NIR-SWIR1)/(NIR+SWIR1)` — vegetation moisture content |

### Built-up & bare-soil

| Method | Formula |
|---|---|
| `ndbi(swir1, nir)` | `(SWIR1-NIR)/(SWIR1+NIR)` — built-up areas |
| `ndsi(green, swir1)` | `(Green-SWIR1)/(Green+SWIR1)` — snow cover |

### Fire & burn severity

| Method | Formula |
|---|---|
| `nbr(nir, swir2)` | `(NIR-SWIR2)/(NIR+SWIR2)` |
| `dnbr(nir_pre, swir2_pre, nir_post, swir2_post, output=None)` | pre-fire NBR minus post-fire NBR |

### Transforms

```python
tct = client.indices.tct(blue, green, red, nir, swir1, swir2, sensor="sentinel2")
```

**Tasseled Cap Transformation** — real, published coefficients, not
placeholders: Nedkov (2017) for `sensor="sentinel2"` (default), Baig
et al. (2014) for `sensor="landsat8"`. Produces a 3-band Brightness /
Greenness / Wetness output.

```python
pca_result = client.indices.pca(inputs=[b02, b03, b04, b08], n_components=3)
```

Real PCA (not a stand-in) over an arbitrary list of input bands.

```python
texture = client.indices.texture(
    input="B08.tif", window=5,
    features=["contrast", "homogeneity", "energy", "correlation"],
)
```

**GLCM texture features** via `scipy.ndimage` (not a pure-Python
loop, so it stays fast on real raster sizes). Six real features
available: `contrast`, `dissimilarity`, `homogeneity`, `energy`,
`correlation`, `ASM`; default computes all six as separate output
bands.

```python
lst = client.indices.lst(thermal="B10.tif", emissivity=0.97, sensor="landsat8")
```

**Land Surface Temperature** from a real thermal band — uses the
actual Landsat 8/9 Band 10 thermal constants (K1=774.8853,
K2=1321.0789), not approximated values. `sensor` also accepts
`"landsat9"` or `"modis"`. Output is 2-band: Kelvin and Celsius.
`emissivity` defaults to `0.97` (typical vegetation); use `0.98` for
water surfaces.

```python
albedo = client.indices.albedo(inputs=[b02, b03, b04, b08, b11, b12], sensor="sentinel2")
```

**Narrowband-to-broadband surface albedo** using Liang (2001)'s
published coefficients. Band order matters and is sensor-specific:
Sentinel-2 expects `[B02, B03, B04, B08, B11, B12]`; Landsat-8 expects
`[B2, B3, B4, B5, B6, B7]`.

### General-purpose escape hatches

```python
custom = client.indices.band_math(
    inputs=[red_path, nir_path],
    expression="(B[1] - B[0]) / (B[1] + B[0] + 1e-6)",
)
```

```{danger}
`band_math()`'s `expression` is evaluated with Python's `eval()`
(`B` and `np` are the only names exposed). Fine for expressions you
write yourself; **never pass an `expression` string from untrusted
user input** — it is not sandboxed against arbitrary code execution.
```

```python
stacked = client.indices.stack(inputs=[b02, b03, b04, b08])   # multi-band GeoTIFF, one input per band
```

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
