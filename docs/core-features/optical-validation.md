# Optical Data Validation & Preflight

A pre-download data-quality gate for optical imagery (Sentinel-2,
Landsat, and any other provider's `SatelliteData` results) — the same
position in the pipeline, and the same philosophy, as InSAR's own
`pygeofetch.insar.preflight`: catch real, confirmed acquisition-
planning mistakes *before* bandwidth is spent, distinguish HARD
failures (reject the scene) from WARNINGS (log and proceed), and make
every check independently toggleable.

```{note}
`pygeofetch.validation` requires `shapely` (the `geo` or `insar`
extra). It's imported lazily — a base `pip install pygeofetch` never
requires it, and neither does `PyGeoFetch()` unless optical validation
is actually turned on.
```

## Quick start — wired into search() and download()

The simplest way to use this is the built-in toggle on `PyGeoFetch`
itself — no manual wiring needed:

```python
from pygeofetch import PyGeoFetch
from pygeofetch.models import SearchQuery

pf = PyGeoFetch(validate_optical=True)   # every search()/download() call validates automatically

results = pf.search(SearchQuery(
    bbox=(-74.1, 40.6, -73.7, 40.9),
    start_date="2024-06-01",
    end_date="2024-08-01",
))
# `results` already excludes anything with 0% AOI coverage, missing
# bands, the wrong processing level, or an out-of-range date --
# the AOI is derived automatically from the query's own bbox/geometry.

downloads = pf.download(results, destination="./data")
```

Toggle per call instead of (or in addition to) at construction time:

```python
pf = PyGeoFetch()  # validation off by default

results = pf.search(query, validate_optical=True)     # on for just this call
downloads = pf.download(results, "./data", validate_optical=True)  # a final gate before spending bandwidth
```

```{warning}
The default `required_bands` (`B02, B03, B04, B08, SCL`) are optical
band names. Don't enable `validate_optical` for SAR/InSAR searches —
every SAR scene would fail the missing-bands check. Either leave it
off for those, or pass a config with `check_required_bands=False`.
```

## Direct use — as its own module

For more control (custom configs per AOI, validating scenes from a
saved GeoJSON file that never went through `search()`, or just
wanting the full per-scene report rather than a filtered list):

```python
from shapely.geometry import box
from pygeofetch.validation import OpticalPreflightValidator, OpticalValidationConfig

validator = OpticalPreflightValidator(OpticalValidationConfig())
aoi = box(-74.1, 40.6, -73.7, 40.9)

safe_results = validator.run_preflight(results, aoi, start_date="2024-06-01", end_date="2024-08-01")
```

`run_preflight()` accepts real `SatelliteData` objects (what
`PyGeoFetch.search()` returns) *or* plain STAC-like dicts, and returns
them in the same representation and order they arrived in — no
conversion needed either side, so it slots directly between `search()`
and `download()`.

## Configuration — every check independently toggleable

```python
from pygeofetch.validation import OpticalValidationConfig

cfg = OpticalValidationConfig(
    max_cloud_cover_pct=10.0,
    check_snow_ice_cover=True,
    min_coverage_ratio=0.95,
)
```

| Field | Default | Description |
|---|---|---|
| `check_aoi_coverage` | `True` | Reject scenes whose footprint doesn't sufficiently overlap the AOI. |
| `min_coverage_ratio` | `0.8` | Minimum fraction of the AOI's area the scene must cover, `[0, 1]`. |
| `check_cloud_cover` | `True` | Flag scenes exceeding `max_cloud_cover_pct`. |
| `max_cloud_cover_pct` | `20.0` | Cloud cover threshold, percent. |
| `cloud_cover_is_hard_failure` | `False` | If `True`, exceeding the threshold rejects the scene instead of just warning. |
| `check_snow_ice_cover` | `False` | Off by default — most providers don't report it consistently, and it's irrelevant to most AOIs. |
| `max_snow_ice_pct` | `10.0` | Snow/ice cover threshold, percent. |
| `check_required_bands` | `True` | Reject scenes missing any `required_bands`. |
| `required_bands` | `["B02","B03","B04","B08","SCL"]` | Matched case-insensitively against the scene's asset keys. |
| `check_processing_level` | `True` | Reject scenes not at `expected_level`. |
| `expected_level` | `"Level-2A"` | Matched loosely — `"L2A"`, `"Level-2A"`, `"S2MSI2A"`, and pygeofetch's own `ProcessingLevel.L2A` all compare equal. |
| `check_nodata_margins` | `False` | Heuristic: flag scenes where the AOI sits mostly in the scene's edge margin. Off by default to avoid false positives on legitimately edge-adjacent AOIs — see the note below. |
| `nodata_margin_buffer_deg` | `0.01` | How far to shrink the footprint inward (degrees, ~1 km at the equator) before checking AOI overlap. |
| `nodata_margin_min_aoi_fraction` | `0.5` | Minimum fraction of the AOI that must fall within the shrunk footprint to be considered safe. |
| `check_temporal_bounds` | `True` | Reject scenes outside `[start_date, end_date]`. |
| `treat_warnings_as_errors` | `False` | If `True`, any WARNING (not just ERROR) excludes the scene from the returned list. |

The most consequential checks (AOI coverage, cloud cover, required
bands, processing level, temporal bounds) default **on**. Heavier or
more niche checks (snow/ice cover, nodata margins) default **off** — a
disabled check costs nothing; it's simply skipped.

```{note}
**`validate_nodata_margins` is a real but inherently approximate
heuristic**, not a raster-level check — it can't see actual per-pixel
no-data masks, since those don't exist until the file is downloaded.
It shrinks the scene footprint inward by `nodata_margin_buffer_deg`
and checks how much of the AOI still falls within that shrunk shape.
It catches the common "AOI clips the corner of the swath" case, not
cloud-masked interior gaps — that's not a margin issue and isn't this
check's job.
```

## Hard failures vs. warnings

| | Behavior | Examples |
|---|---|---|
| **Hard failure** (`ERROR`) | Scene excluded from `run_preflight()`'s returned list; logged as an error | Missing required band, 0% AOI coverage, wrong processing level, out-of-range date |
| **Warning** (`WARNING`) | Scene kept; logged as a warning | Cloud cover above threshold (unless `cloud_cover_is_hard_failure=True`), snow/ice above threshold, nodata-margin risk |

```python
report = validator.validate_scene(scene, aoi)

report.passed      # bool
report.errors      # list[ValidationIssue] -- ERROR severity only
report.warnings     # list[ValidationIssue] -- WARNING severity only
report.metrics      # {"aoi_coverage": 0.94, "cloud_cover_pct": 12.5, ...}
```

## Custom exception

```python
from pygeofetch.validation import OpticalValidationError

try:
    validator.validate_bands(["B02", "B03"])  # missing B04, B08, SCL
except OpticalValidationError as exc:
    print(f"Scene {exc.scene_id} rejected [{exc.code}]: {exc.reason}")
    # Scene <pending> rejected [MISSING_BANDS]: missing required band(s): B04, B08, SCL
```

`OpticalValidationError` is a `ValueError` subclass, so existing
`except ValueError` call sites keep working unchanged. It's raised
directly by the low-level `validate_bands()` method; `run_preflight()`
itself never raises per-scene — hard failures are filtered out and
logged, not raised, so a few bad scenes in a large search never abort
the whole run.

## Full method reference

| Method | Returns | Notes |
|---|---|---|
| `validate_aoi_coverage(footprint, aoi)` | `float` | Fraction of the AOI's area covered by the scene footprint, `[0, 1]`. Coverage of the AOI, not a symmetric IoU. |
| `validate_cloud_cover(scene)` | `float` | Cloud cover percentage, `0.0` if unreported. |
| `validate_snow_ice_cover(scene)` | `float` | Snow/ice cover percentage, `0.0` if unreported. |
| `validate_bands(available_assets)` | `list[str]` | Present required bands; raises `OpticalValidationError` if any are missing. |
| `validate_processing_level(scene)` | `bool` | Loose alias matching against `expected_level`. |
| `validate_nodata_margins(footprint, aoi)` | `bool` | The geometry-only heuristic described above. |
| `validate_temporal_bounds(dt, start, end)` | `bool` | Inclusive range check; missing dates are not treated as failures. |
| `validate_scene(scene, aoi, start_date=None, end_date=None)` | `SceneValidationReport` | Runs every enabled check against one scene. `aoi` may be `None` — AOI-dependent checks are skipped gracefully rather than raising. |
| `run_preflight(catalog_results, aoi, start_date=None, end_date=None)` | `list[SatelliteData \| dict]` | The main orchestrator — validates every scene, logs issues, returns the safe subset. |

## Real footprint geometry, with a bbox fallback

Like the rest of pygeofetch's provider layer, `validate_aoi_coverage`
uses a scene's real footprint geometry when the provider supplies it,
falling back to a bbox rectangle when it doesn't — see
{doc}`/core-features/providers` for which providers return precise
footprint geometry today.
