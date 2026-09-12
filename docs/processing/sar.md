# SAR Processing

```bash
pip install "pygeofetch[sar]"
```

Speckle filtering, radiometric calibration, flood mapping,
interferometric coherence, and five real, standard end-to-end
processing pipelines built on top of them — genuine, from-scratch
implementations, not thin wrappers around an external SAR toolkit.

!!! note "Two `SARProcessor` classes exist — corrected here after direct verification"

    `client.sar` (via `PyGeoFetch()`) is `pygeofetch.processing.sar.SARProcessor`.
    `from pygeofetch.sar import SARProcessor` is a separate, standalone class
    (`pygeofetch.sar.processor.SARProcessor`) with a pluggable `backend=`
    parameter (`"native"`/`"sarxarray"`/`"ost"`).

    An earlier version of this page said the standalone class's `"native"`
    backend was "a third, separate implementation... not a delegation."
    That was checked directly against the actual code and **isn't
    accurate**: `pygeofetch.sar._native.NativeSARBackend` constructs a
    real `pygeofetch.processing.sar.SARProcessor` internally and calls
    straight through to it for despeckle, calibrate, flood mapping, and
    coherence. For the native backend specifically, these are the same
    real implementation reached through two different import paths, not
    two copies to keep in sync.

    The real, narrower duplication that *did* exist — and has been
    fixed — was the interferometric coherence *formula* itself, which
    was independently implemented a second time inside
    `pygeofetch.insar.interferogram` (the full InSAR pipeline's own
    coherence step). Both were verified mathematically correct, but as
    two copies. The standalone version now delegates to one shared,
    canonical implementation in `pygeofetch.utils.sar_math` — see
    below. The `insar` package's own internal version deliberately
    stays separate: it's tightly coupled to a large, already
    real-world-validated class with a real chunked variant for
    memory-safe processing of full-scene Sentinel-1 SLC rasters (which
    can be multiple gigabytes) and real GPU acceleration support —
    refactoring that file to depend on this one was judged too risky
    for the marginal benefit versus the chance of regressing already-proven code.

    Use `client.sar` for despeckling, calibration, flood mapping, or
    coherence if you don't need the `sarxarray`/OST backends — it needs
    no extra imports. Reach for the standalone `pygeofetch.sar.SARProcessor`
    when you specifically want the `sarxarray` or `ost` backend, or when
    calling one of the five pipelines below (which use the pluggable
    facade so they work with any backend).

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

!!! warning

    **Honest, documented limitation**: the calibration constant `A` is
    fixed at `1.0` (identity) — this is *not* a real, per-scene
    calibration LUT read from the Sentinel-1 annotation XML, which is
    what real radiometric calibration requires for absolute accuracy.
    `gamma0`/`beta0` similarly use a **fixed nominal incidence angle**
    (38°), not the real per-pixel local incidence angle from a DEM. This
    is adequate for *relative* comparisons within one scene (e.g. flood
    detection, change detection — which is exactly what every pipeline
    below uses it for) but **not** for absolute, cross-scene radiometric
    accuracy work. A real implementation would need to parse the actual
    Sentinel-1 calibration vectors and use per-pixel incidence angle
    from a real terrain model — track this as a known gap if your use
    case needs true absolute calibration.

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

!!! note "flood_map() alone is not a complete workflow"

    Calling `flood_map()` directly on raw DN values gives poor,
    unreliable results — its dB-scale threshold assumes calibrated
    input. See [`flood_mapping_pipeline`](#flood_mapping_pipeline)
    below, which handles the real prerequisite chain for you.

### Interferometric coherence

```python
client.sar.coherence(image1, image2, window=7, output=None)
```

Real formula: `coherence = |<s1·conj(s2)>| / sqrt(<|s1|^2><|s2|^2>)`,
range `[0, 1]`. Both inputs must be **co-registered complex SLC**
rasters (`complex64` GeoTIFFs). High coherence indicates a stable
surface between the two acquisition dates; low coherence indicates
change or temporal decorrelation (vegetation growth, surface
disturbance, etc.).

!!! danger "A real, serious bug was fixed here — verify you're on a current version"

    A previous version of `coherence()` read complex SLC data through a
    shared helper that unconditionally casts everything to real
    float32 (discarding the imaginary/phase part), then tried to
    reinterpret that already-real data as complex via a raw memory
    `.view()`. **This made `coherence()` completely non-functional for
    its entire stated purpose** — genuine complex SLC input — before
    the fix. This wasn't found by code review; it was caught by an
    actual test using a real synthetic complex64 GeoTIFF, which crashed
    with a shape-mismatch error rather than silently returning
    corrupted output. Fixed by reading complex bands directly via
    rasterio, never through the lossy real-valued helper.

    The underlying formula now lives in one shared, canonical place —
    `pygeofetch.utils.sar_math.estimate_interferometric_coherence` —
    used by both `client.sar.coherence()` and the pipelines below.

!!! note

    For the *specific* case of Sentinel-1 InSAR coherence as part of a
    full interferogram (not a standalone two-image comparison), see
    [InSAR Processing](insar.md) -- `InterferogramGenerator` computes
    coherence as part of real burst-aware, orbit-coregistered
    interferogram formation, which is a substantially more involved
    pipeline than this standalone `coherence()` method.

## Five real, standard SAR processing pipelines

`pygeofetch.sar.pipelines` orchestrates the atomic operations above
into five real, widely-used end-to-end workflows — because remembering
and correctly ordering the real prerequisite steps each analysis
actually needs is easy to get wrong (e.g. flood mapping on raw,
uncalibrated DN gives poor results, a real, common mistake).

Every pipeline takes a `pygeofetch.sar.SARProcessor` instance (the
standalone, backend-pluggable facade, not `client.sar`), so it works
transparently with whichever backend you configure — though the native
backend (no extra deps) is the sensible default for most use.

```python
from pygeofetch.sar import SARProcessor
from pygeofetch.sar.pipelines import standard_grd_preprocessing_pipeline

proc = SARProcessor()  # native backend
result = standard_grd_preprocessing_pipeline(proc, "s1_grd_dn.tif")

if result.success:
    print(f"Ready for analysis: {result.final_output}")
else:
    print(f"Failed: {result.error}")  # names exactly which stage failed
```

Every pipeline returns a `PipelineResult`: `success`, `final_output`,
`stages` (every real intermediate `ProcessingResult`, not just the
final one — useful for inspecting exactly what each step produced),
and `metadata` (pipeline-specific extras, like a change mask path or
pixel counts).

### `standard_grd_preprocessing_pipeline`

The real, standard Sentinel-1 GRD chain nearly every real SAR analysis
starts with: calibrate (DN → sigma0/gamma0/beta0), then despeckle — in
that order, not the reverse. Despeckling raw, uncalibrated DN conflates
real backscatter with the sensor's raw digital-number scaling, and
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
`flood_map()` — enforcing the real prerequisite chain rather than
leaving it to the caller.

```python
from pygeofetch.sar.pipelines import flood_mapping_pipeline

result = flood_mapping_pipeline(
    proc, "during_flood.tif", reference_path="before_flood.tif", threshold=-15.0,
)
```

### `change_detection_pipeline`

Real, standard bi-temporal amplitude change detection: calibrate and
despeckle both dates, then flag pixels whose backscatter changed by
more than `change_threshold_db`. Since both calibrated rasters are
already in dB, the log-ratio change metric reduces to a simple
subtraction (`post_dB - pre_dB`) — real, standard math, not an
approximation.

General-purpose beyond flood mapping: burn-scar mapping, deforestation
and clear-cut detection, construction/urban change, and disaster damage
assessment (collapsed structures show a real, sharp backscatter change).

```python
from pygeofetch.sar.pipelines import change_detection_pipeline

result = change_detection_pipeline(proc, "pre.tif", "post.tif", change_threshold_db=3.0)
print(f"{result.metadata['pct_changed']}% of pixels changed")
```

### `coherence_disturbance_pipeline`

Computes interferometric coherence between two co-registered SLC
acquisitions, then flags low-coherence pixels as a real disturbance
mask. Genuinely complementary to `change_detection_pipeline` above:
that one needs GRD (amplitude) data and detects backscatter magnitude
change; this one needs SLC (phase-preserving) data and detects
phase-stability loss. Real, common uses: deforestation and selective
logging (vegetation disturbance decorrelates radar phase almost
immediately), and general surface disturbance monitoring.

```python
from pygeofetch.sar.pipelines import coherence_disturbance_pipeline

result = coherence_disturbance_pipeline(proc, "slc_pre.tif", "slc_post.tif", coherence_threshold=0.3)
```

### `bright_target_detection_pipeline`

Real, standard CA-CFAR (cell-averaging Constant False Alarm Rate)
bright-target detection: calibrate — but deliberately **do not
despeckle**, since speckle filtering would smooth away exactly the
small, bright targets this pipeline looks for — then flag pixels
significantly brighter than their real local background, estimated
from a real annulus of pixels between a guard window and a larger
background window (excluding the guard region so a target's own bright
halo doesn't inflate its own background estimate).

Real, common uses: ship/vessel detection (maritime surveillance,
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

## The standalone, backend-pluggable `pygeofetch.sar.SARProcessor`

```python
from pygeofetch.sar import SARProcessor

# Native backend -- always available, no extra deps, delegates to client.sar's own implementation
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
| `"native"` (default) | Nothing extra | Despeckle, calibrate, flood map, coherence -- delegates directly to `client.sar`'s own real implementation |
| `"sarxarray"` | `pygeofetch[sar]` | xarray/Dask-native large-scale processing |
| `"ost"` | `pygeofetch[ost]` + a working SNAP install | Production Range-Doppler terrain correction -- the only place `terrain_correct()` is available |

Constructing with an unrecognised `backend` raises `ValueError`
immediately, not a delayed failure on first use.
