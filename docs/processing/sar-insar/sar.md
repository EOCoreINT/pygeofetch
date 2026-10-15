# SAR Processing

```bash
pip install "pygeofetch[sar]"
```

:::{seealso}
Looking to combine SAR with a *different* sensor type (optical, DEM)
rather than multiple dates of SAR alone? See
[Multi-Sensor Pipelines](../multisensor.md) — flood mapping,
vegetation disturbance, and InSAR+optical fusion all live there.
:::

Speckle filtering, radiometric calibration, flood mapping,
interferometric coherence, and five standard end-to-end
processing pipelines built on top of them. These are from-scratch implementations, not
wrappers around an external SAR toolkit.

:::{admonition} Two `SARProcessor` classes exist
:class: note

`client.sar` (from `PyGeoFetch()`) is `pygeofetch.processing.sar.SARProcessor`.
`from pygeofetch.sar import SARProcessor` is a separate class
(`pygeofetch.sar.processor.SARProcessor`) with the same methods and its own implementation.
They are not kept in sync: for example, only the standalone class reads complex SLC input
correctly for `coherence()`.
:::
## `client.sar`: the built-in processor

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

:::{warning}

**Limitation.** The calibration constant `A` is fixed at `1.0` (identity). It is not a
per-scene calibration look-up table read from the Sentinel-1 annotation XML, which absolute
radiometric accuracy requires. `gamma0` and `beta0` use a **fixed nominal incidence angle**
(38°) rather than the per-pixel local incidence angle from a DEM. This is adequate for
*relative* comparisons within one scene (flood detection and change detection, which is what
every pipeline below uses it for). It is **not** adequate for absolute, cross-scene
radiometric work. That would need the Sentinel-1 calibration vectors and a terrain model.
:::
### Flood mapping

```python
client.sar.flood_map(
    input, threshold=-15.0, output=None,
    reference=None, detect_direction="decrease",
)
```

Two modes:
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

:::{admonition} flood_map() alone is not a complete workflow
:class: note

Calling `flood_map()` directly on raw DN values gives poor,
unreliable results — its dB-scale threshold assumes calibrated
input. See [`flood_mapping_pipeline`](#flood_mapping_pipeline)
below, which handles the prerequisite chain for you.
:::
### Interferometric coherence

```python
client.sar.coherence(image1, image2, window=7, output=None)
```

Formula: `coherence = |<s1·conj(s2)>| / sqrt(<|s1|^2><|s2|^2>)`,
range `[0, 1]`. Both inputs must be **co-registered complex SLC**
rasters (`complex64` GeoTIFFs). High coherence indicates a stable
surface between the two acquisition dates; low coherence indicates
change or temporal decorrelation (vegetation growth, surface
disturbance, etc.).

:::{admonition} `client.sar.coherence()` fails on complex SLC input
:class: warning

In the current code, `client.sar.coherence()` (and therefore `pygeofetch sar coherence`) reads
complex bands through a shared loader that casts to real `float32`, discarding the phase, and
then fails with `ValueError: cannot reshape array of size ... into shape ...`. The same method
on the standalone class, `pygeofetch.sar.SARProcessor`, reads complex bands directly with
`rasterio` and works. Until `client.sar` is fixed, use the standalone class for coherence:

```python
from pygeofetch.sar import SARProcessor

result = SARProcessor().coherence("slc_a.tif", "slc_b.tif", window=7, output="coherence.tif")
```

The coherence formula itself lives in `pygeofetch.utils.sar_math.estimate_interferometric_coherence`
and is shared by the standalone class and the multi-sensor vegetation-disturbance pipeline.
See [Known issues](../../reference/known-issues.md#sar-coherence-fails-on-complex-input).
:::

## Five standard SAR processing pipelines

`pygeofetch.sar.pipelines` orchestrates the atomic operations above
into five widely-used end-to-end workflows — because remembering
and correctly ordering the prerequisite steps each analysis
actually needs is easy to get wrong (e.g. flood mapping on raw,
uncalibrated DN gives poor results, a common mistake).

Every pipeline takes a `pygeofetch.sar.SARProcessor` instance (the standalone class, not
`client.sar`).

```python
from pygeofetch.sar import SARProcessor
from pygeofetch.sar.pipelines import standard_grd_preprocessing_pipeline

proc = SARProcessor()
result = standard_grd_preprocessing_pipeline(proc, "s1_grd_dn.tif")

if result.success:
    print(f"Ready for analysis: {result.final_output}")
else:
    print(f"Failed: {result.error}")  # names exactly which stage failed
```

Every pipeline returns a `PipelineResult`: `success`, `final_output`,
`stages` (every intermediate `ProcessingResult`, not just the
final one — useful for inspecting exactly what each step produced),
and `metadata` (pipeline-specific extras, like a change mask path or
pixel counts).

### `standard_grd_preprocessing_pipeline`

The standard Sentinel-1 GRD chain nearly every SAR analysis
starts with: calibrate (DN → sigma0/gamma0/beta0), then despeckle — in
that order, not the reverse. Despeckling raw, uncalibrated DN conflates
backscatter with the sensor's raw digital-number scaling, and
filtering in dB space (calibrate's default) is standard practice since
speckle noise behaves closer to additive once log-converted, which
Lee/boxcar-style filters (designed for additive noise) handle better.

```python
from pygeofetch.sar.pipelines import standard_grd_preprocessing_pipeline

result = standard_grd_preprocessing_pipeline(
    proc, "s1_grd_dn.tif", output_type="sigma0", despeckle_filter="lee",
)
```

### `flood_mapping_pipeline`

Calibrates and despeckles the flood-event image (and a pre-event
reference, if doing change-based detection) before running
`flood_map()` — enforcing the prerequisite chain rather than
leaving it to the caller.

```python
from pygeofetch.sar.pipelines import flood_mapping_pipeline

result = flood_mapping_pipeline(
    proc, "during_flood.tif", reference_path="before_flood.tif", threshold=-15.0,
)
```

### `change_detection_pipeline`

standard bi-temporal amplitude change detection: calibrate and
despeckle both dates, then flag pixels whose backscatter changed by
more than `change_threshold_db`. Since both calibrated rasters are
already in dB, the log-ratio change metric reduces to a simple
subtraction (`post_dB - pre_dB`) — standard math, not an
approximation.

General-purpose beyond flood mapping: burn-scar mapping, deforestation
and clear-cut detection, construction/urban change, and disaster damage
assessment (collapsed structures show a sharp backscatter change).

```python
from pygeofetch.sar.pipelines import change_detection_pipeline

result = change_detection_pipeline(proc, "pre.tif", "post.tif", change_threshold_db=3.0)
print(f"{result.metadata['pct_changed']}% of pixels changed")
```

### `coherence_disturbance_pipeline`

Computes interferometric coherence between two co-registered SLC
acquisitions, then flags low-coherence pixels as a disturbance
mask. Complementary to `change_detection_pipeline` above:
that one needs GRD (amplitude) data and detects backscatter magnitude
change; this one needs SLC (phase-preserving) data and detects
phase-stability loss. common uses: deforestation and selective
logging (vegetation disturbance decorrelates radar phase almost
immediately), and general surface disturbance monitoring.

```python
from pygeofetch.sar.pipelines import coherence_disturbance_pipeline

result = coherence_disturbance_pipeline(proc, "slc_pre.tif", "slc_post.tif", coherence_threshold=0.3)
```

### `bright_target_detection_pipeline`

standard CA-CFAR (cell-averaging Constant False Alarm Rate)
bright-target detection: calibrate — but deliberately **do not
despeckle**, since speckle filtering would smooth away exactly the
small, bright targets this pipeline looks for — then flag pixels
significantly brighter than their local background, estimated
from a annulus of pixels between a guard window and a larger
background window (excluding the guard region so a target's own bright
halo doesn't inflate its own background estimate).

common uses: ship/vessel detection (maritime surveillance,
illegal fishing monitoring), and more generally any small,
strongly radar-reflective object against a comparatively uniform
background (metal structures, vehicles, equipment) — a real,
complementary technique to optical detection for conditions optical
can't cover: cloud cover and night-time passes.

```python
from pygeofetch.sar.pipelines import bright_target_detection_pipeline

result = bright_target_detection_pipeline(
    proc, "scene.tif", guard_window=5, background_window=21, k_factor=3.0,
)
print(f"{result.metadata['n_candidate_target_pixels']} candidate targets")
```

## The standalone `pygeofetch.sar.SARProcessor`

```python
from pygeofetch.sar import SARProcessor

proc = SARProcessor()                      # takes no arguments
result = proc.despeckle("s1_vv.tif", filter="lee")
result = proc.calibrate("s1_dn.tif", output_type="sigma0")
result = proc.coherence("slc_a.tif", "slc_b.tif", window=7)
```

`SARProcessor` has the same methods as `client.sar` (`despeckle`, `calibrate`, `flood_map`,
`coherence`) and is what the five pipelines above expect.

:::{admonition} The `backend=` parameter is not implemented
:class: warning

The `pygeofetch.sar` module docstring and the `sar` and `ost` extras describe a pluggable
backend: `SARProcessor(backend="sarxarray")` or `SARProcessor(backend="ost")`. In the current
code `SARProcessor` has no `__init__`, so `SARProcessor(backend="sarxarray")` raises
`TypeError: SARProcessor() takes no arguments`. The helper modules `pygeofetch/sar/_native.py`,
`_sarxarray.py` and `_ost.py` exist but nothing imports them. Terrain correction
(`terrain_correct`) and the `sarxarray` workflow are therefore not reachable through the public
API. See [Known issues](../../reference/known-issues.md#sar-backends-are-not-wired-in).
:::
