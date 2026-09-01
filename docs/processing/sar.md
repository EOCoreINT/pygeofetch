# SAR Processing

```bash
pip install "pygeofetch[sar]"
```

Speckle filtering, radiometric calibration, flood mapping, and
interferometric coherence — genuine, from-scratch implementations, not
thin wrappers around an external SAR toolkit.

```{danger}
**Two separate `SARProcessor` classes exist in the codebase, and only
one is reachable via `PyGeoFetch`.** This is a real architectural
duplication, not a typo in this page:

- **`client.sar`** (via `PyGeoFetch()`) is
  `pygeofetch.processing.sar.SARProcessor` — no backend concept,
  five real despeckle filters, calibration, flood mapping, and
  coherence, all implemented directly. **This is the one almost every
  real workflow should use.**
- **`from pygeofetch.sar import SARProcessor`** is a *different*,
  standalone class (`pygeofetch.sar.processor.SARProcessor`) with a
  pluggable `backend=` parameter (`"native"`/`"sarxarray"`/`"ost"`)
  and an additional `terrain_correct()` method — genuinely useful for
  the `sarxarray`/OST backends specifically, but **not accessible as
  `client.sar`** and its own `"native"` backend is a third, separate
  implementation from `client.sar`'s despeckle/calibrate/flood_map,
  not a delegation to it.

If you only need despeckling, calibration, flood mapping, or
coherence and don't need `sarxarray`/OST, use `client.sar` — it needs
no extra imports and is what the rest of pygeofetch's own workflow
(e.g. the InSAR chain) is built around. Reach for the standalone
`pygeofetch.sar.SARProcessor` only when you specifically want the
`sarxarray` or `ost` backend.
```

## `client.sar` — the primary, wired-in processor

```python
from pygeofetch import PyGeoFetch

client = PyGeoFetch()

despeckled = client.sar.despeckle("sentinel1_vv.tif", filter="enhanced_lee")
calibrated = client.sar.calibrate("sentinel1_dn.tif", output_type="sigma0", in_db=True)
flood = client.sar.flood_map("post_event.tif", reference="pre_event.tif", detect_direction="both")
coh = client.sar.coherence("slc_20260601.tif", "slc_20260613.tif")
```

All four methods read via a block-by-block fallback, so they work
directly on tiled/COG/compressed GeoTIFFs without a full-scene decode
crashing on large files.

### Despeckling

```python
client.sar.despeckle(input, filter="lee", window=5, num_looks=1, output=None)
```

| Filter | Real algorithm |
|---|---|
| `"lee"` (default) | Classic adaptive Lee filter — local mean/variance weighting |
| `"enhanced_lee"` | Coefficient-of-variation-thresholded Lee, sharper edge preservation |
| `"frost"` | Exponentially-weighted local averaging, adaptive to local CV |
| `"gamma"` | Gamma-MAP-style adaptive filter |
| `"boxcar"` | Plain uniform-window averaging (fastest, least edge-preserving) |

`window` must be odd (default 5). `num_looks` affects the Lee/Gamma
noise-variance threshold — set it to the real number of looks in your
input if known; the default of 1 is conservative (least aggressive
smoothing).

### Radiometric calibration

```python
client.sar.calibrate(input, output_type="sigma0", in_db=True, output=None)
```

Converts SAR digital numbers (DN) to backscatter coefficients:
`sigma0 = DN² / A²`.

```{warning}
**Honest, documented limitation**: the calibration constant `A` is
fixed at `1.0` (identity) — this is *not* a real, per-scene
calibration LUT read from the Sentinel-1 annotation XML, which is
what real radiometric calibration requires for absolute accuracy.
`gamma0`/`beta0` similarly use a **fixed nominal incidence angle**
(38°), not the real per-pixel local incidence angle from a DEM. This
is adequate for *relative* comparisons within one scene (e.g. flood
detection, change detection) but **not** for absolute, cross-scene
radiometric accuracy work. A real implementation would need to parse
the actual Sentinel-1 calibration vectors and use per-pixel incidence
angle from a real terrain model — track this as a known gap if your
use case needs true absolute calibration.
```

### Flood mapping

```python
client.sar.flood_map(
    input, threshold=-15.0, output=None,
    reference=None, detect_direction="decrease",
)
```

Two real modes:
- **Simple threshold** (no `reference`): backscatter below `threshold`
  dB flags water (open water is a near-specular reflector at typical
  incidence angles, so it returns very little energy to the sensor).
- **Change detection** (`reference` given): compares pre/post-event
  backscatter, sensitivity `abs(threshold * 0.5)`.

`detect_direction` matters more than it might look:

| Value | Detects | Why |
|---|---|---|
| `"decrease"` (default) | Backscatter dropping | The correct signature for **open water** — a newly-flooded field becomes a smooth, near-specular surface |
| `"increase"` | Backscatter rising | The correct signature for **flooded urban/built-up areas** — water at a building's base creates a double-bounce (ground-wall-sensor) reflection stronger than dry ground alone |
| `"both"` | Either direction | The robust choice when an AOI mixes open water and dense urban flooding — a one-directional threshold **structurally cannot** detect the other pattern at all, not just detect it poorly |

### Interferometric coherence

```python
client.sar.coherence(image1, image2, window=7, output=None)
```

Real formula: `coherence = |<s1*s2*>| / sqrt(<|s1|^2><|s2|^2>)`, range
`[0, 1]`. Both inputs must be **co-registered complex SLC** rasters
(`complex64` GeoTIFFs, or real-valued rasters treated as zero-phase).
High coherence indicates a stable surface between the two acquisition
dates; low coherence indicates change or temporal decorrelation
(vegetation growth, surface disturbance, etc.).

```{note}
For the *specific* case of Sentinel-1 InSAR coherence as part of a
full interferogram (not a standalone two-image comparison), see
{doc}`/processing/insar` -- `InterferogramGenerator` computes
coherence as part of real burst-aware, orbit-coregistered
interferogram formation, which is a substantially more involved
pipeline than this standalone `coherence()` method.
```

## The standalone, backend-pluggable `pygeofetch.sar.SARProcessor`

```python
from pygeofetch.sar import SARProcessor

# Native backend -- always available, no extra deps
proc = SARProcessor()   # backend="native" by default
result = proc.despeckle("s1_vv.tif", filter="lee")

# sarxarray backend -- richer xarray/Dask-native workflow for large-scale processing
proc = SARProcessor(backend="sarxarray")
result = proc.calibrate("s1_dn.tif", output_type="sigma0")

# OST/SNAP backend -- production-grade Range-Doppler terrain correction, requires SNAP
proc = SARProcessor(backend="ost")
result = proc.terrain_correct("s1_cal.tif", dem="srtm")
```

| Backend | Requires | Best for |
|---|---|---|
| `"native"` (default) | Nothing extra | Despeckle, calibrate, flood map, coherence -- its own separate implementation from `client.sar`, not a delegation |
| `"sarxarray"` | `pygeofetch[sar]` | xarray/Dask-native large-scale processing |
| `"ost"` | `pygeofetch[ost]` + a working SNAP install | Production Range-Doppler terrain correction -- the only place `terrain_correct()` is available |

Constructing with an unrecognised `backend` raises `ValueError`
immediately, not a delayed failure on first use.
