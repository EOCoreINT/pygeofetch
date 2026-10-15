# Upgrades: Auto-Fallback, UQ Export, Geotechnical Risk, Lightweight Prefetch

Four genuinely new capabilities added this pass, plus one item
(robust horizontal strain) that turned out to already exist. Every
claim below was checked against real, current source or verified with
a real, executed test — not assumed from the request that prompted it.

## What already existed — not rebuilt

**`compute_horizontal_strain`** in
`pygeofetch.optical.offset_tracking` already includes real,
MAD-based outlier protection (a `mad_outlier_threshold` parameter,
default 8.0) — built earlier this project in direct response to a
real, live 617% strain artifact found in this project's own Bu'ertai
validation run. There is no separate `compute_horizontal_strain_robust`
function; the base function *is* the robust one. See
[Optical Pixel Offset Tracking](optical-offset-tracking.md#strain-turning-displacement-into-a-risk-metric).

## 1. The Auto-Fallback Multi-Modal Router

`pygeofetch.insar.auto_fallback.run_multi_modal_deformation()`

```python
from pygeofetch.insar.auto_fallback import run_multi_modal_deformation

result = run_multi_modal_deformation(
    velocity=ts_result.velocity,      # from SBASTimeSeries.invert()
    pairs=list(project.interferograms.values()),
    optical_reference_path="sentinel2_pre.tif",   # only needed if InSAR is poor
    optical_secondary_path="sentinel2_post.tif",
    reference_transform=optical_transform,
    insar_transform=insar_transform, insar_crs=insar_crs,
)

print(f"Used optical fallback: {result.used_optical_fallback}")
print(f"Real InSAR quality: {result.insar_reliable_fraction:.1%} coverage, "
      f"{result.insar_mean_coherence:.2f} mean coherence")
```

### What this actually adds

This does **not** reimplement fusion — `pygeofetch.multisensor.insar_optical_displacement_pipeline`
and `fuse_insar_optical` already do that, and this router calls them
directly. Its real, distinct value is the *decision*: if InSAR's own
real reliable-pixel fraction and mean coherence both already clear
real, configurable thresholds (`reliable_fraction_threshold`,
`mean_coherence_threshold`, both default `0.3`), optical processing is
**skipped entirely** — a genuine compute and bandwidth saving over
always running both, which is what the existing multi-sensor pipeline
does unconditionally.

### A real gap found while building this

`TimeSeriesResult` (SBAS's own real output type) carries no
`reliable_fraction` or coherence field at all — confirmed directly by
reading `SBASTimeSeries.invert()`'s actual return statement, which
only populates `metadata={"method": ...}`. The router computes both
metrics honestly from what's actually available: real coverage from
`np.isfinite(velocity).mean()`, real mean coherence averaged across
the real interferogram pairs used.

### A naming collision, documented rather than left implicit

`fuse_insar_optical`'s own `reliability_source` output uses
`2=InSAR-dominant, 1=transition, 0=optical-dominant`. This router's
`processing_source_mask` uses a **different, deliberate** convention:
`0=Optical Fallback, 1=InSAR Valid, 2=Blended`. The two functions are
not interchangeable — passing one function's mask values into code
expecting the other's would silently invert the meaning of "good" and
"bad." Both conventions are documented explicitly in each function's
own docstring specifically to prevent this.

### The honest-failure principle in practice

When both thresholds fail and no optical inputs were supplied, this
raises a clear `ValueError` naming the real, current
reliable_fraction/coherence values — it does not silently return a
degraded or fabricated InSAR-only result. Confirmed by test.

**8/8 tests pass**, including reproductions of both the good-InSAR
(skip optical) and poor-InSAR (trigger fusion) real decision paths,
and the OR-logic (either metric alone can trigger fallback).

## 2. Uncertainty Quantification Export

`pygeofetch.processing.uq_exporter`

```python
from pygeofetch.processing.uq_exporter import (
    compute_phase_and_displacement_uncertainty,
    export_with_uncertainty,
    summarize_uncertainty_for_provenance,
)
from pygeofetch.insar.provenance import write_provenance_manifest

sigma_phi, sigma_disp = compute_phase_and_displacement_uncertainty(
    coherence, n_looks=4, wavelength_m=0.05546576,  # Sentinel-1 C-band
)
export_with_uncertainty(
    velocity, sigma_disp, profile, "velocity_with_uncertainty.tif",
    value_band_name="velocity_m_yr", uncertainty_band_name="sigma_disp_m",
)

write_provenance_manifest(
    output_dir, preflight_manifest, processing, corrections,
    quality=summarize_uncertainty_for_provenance(sigma_phi, sigma_disp, n_looks=4, wavelength_m=0.05546576),
)
```

### The real formula, reused not reinvented

```{math}
\sigma_\phi = \frac{\sqrt{1-\gamma^2}}{\gamma\sqrt{2N}}, \qquad
\sigma_{disp} = \frac{\lambda}{4\pi}\sigma_\phi
```

This is the real, standard Cramér-Rao interferometric phase-uncertainty
bound. Rather than derive it a second time, this module reuses the
**exact same formula** already present in `pygeofetch.insar.synthetic`
(used there to generate realistic synthetic decorrelation noise) —
confirmed numerically identical by direct test, so the two can never
silently diverge.

### What `write_provenance_manifest` needed

Its real, existing signature already accepts a generic `quality: dict`
parameter — nothing about it needed to change.
`summarize_uncertainty_for_provenance` builds the right real dict
shape to hand it, and a real, direct integration test confirms the
output feeds straight into the actual function without adaptation.

**12/12 tests pass**, including exact-formula-match, physical
monotonicity (higher coherence → lower uncertainty; more looks → lower
uncertainty), a real two-band GeoTIFF round-trip, and the real
provenance-manifest integration.

## 3. Domain-Specific Geotechnical Risk Mapper

`pygeofetch.geotech.risk_mapper`

```python
from pygeofetch.geotech.risk_mapper import (
    RiskMapperConfig, compute_curvature, compute_geotechnical_risk_index,
)

curvature = compute_curvature(dx, dy, pixel_size_m=16.0)
strain = compute_horizontal_strain(dx, dy, pixel_size_m=16.0)  # existing function

risk = compute_geotechnical_risk_index(
    velocity=insar_velocity,
    max_shear_strain=strain["exy"],
    curvature=curvature["curvature_max"],
)
print(f"Pixels with implausible strain, capped: {risk['n_strain_capped']}")
```

### The real, cited standard behind the strain cap

`RiskMapperConfig.max_severe_strain` defaults to **1.0%** — the real,
published National Coal Board (1975) *Subsidence Engineering
Handbook*'s own "very severe" damage category threshold. This is a
cited, real standard, not a placeholder.

`RiskMapperConfig.strain_cap` (default 10%) is a **separate**, harder
ceiling — a direct, explicit response to this project's own real,
observed 617% strain artifact. The test suite reproduces that exact
scenario (strain=6.17) and confirms the cap neutralizes it before it
can dominate the risk index, independent of whatever outlier
protection already ran upstream in `compute_horizontal_strain` itself.

### An honest note on `compute_curvature`'s real interface

The standard mining-subsidence convention computes curvature from a
*single* vertical displacement field. This implementation generalizes
to real displacement *magnitude* from two horizontal components
(`dx`, `dy`) for broader, multi-modal applicability (lateral landslide
movement, co-seismic slip) — documented explicitly as a real departure
from, not identical to, the classic single-field convention. Pass a
real zero array as `dy` to reproduce standard single-field curvature
exactly.

### Combination logic

`config.combination="max_weighted"` (default) computes
`0.6 * max(scores) + 0.4 * mean(scores)` — a single severe factor
dominates the index rather than being diluted by two mild ones,
matching real geotechnical practice where one severe indicator alone
is real cause for concern. `"mean"` is available as a simpler,
real alternative.

**15/15 tests pass**, including an exact analytical curvature check (a
parabolic field's known constant second derivative) and direct
verification that `max_weighted` gives materially higher risk than a
plain mean for the same real inputs.

## 4. Lightweight, Annotation-Only STAC Prefetch

`pygeofetch.stac_compute.lightweight_prefetch`

```python
from pygeofetch.stac_compute.lightweight_prefetch import prefetch_and_preflight

result = prefetch_and_preflight(
    reference_date="2024-01-01", secondary_date="2024-01-13",
    reference_product_url=ref_scene.assets["product"].href,
    secondary_product_url=sec_scene.assets["product"].href,
    reference_orbit_path="ref.EOF", secondary_orbit_path="sec.EOF",
    ground_point=aoi_center_ecef,
    output_dir="./prefetch_cache",
)

if result["passed"]:
    # Only now trigger the real, full multi-gigabyte SLC download.
    client.download([ref_scene, sec_scene], destination="./raw_slc")
else:
    print(f"Pair excluded before any real full download: {result['family_report']}")
```

### Real, confirmed feasibility — not a theoretical claim

The real, published, open-source `asfsmd` tool (Valentino, PyPI/GitHub)
already does exactly this against the real ASF archive: opening a
remote Sentinel-1 SLC ZIP's real central directory via HTTP range
requests and extracting only the annotation XML, never fetching the
real measurement rasters. This module implements the same real
technique directly with `httpx` (already a core pygeofetch dependency)
rather than adding `fsspec` as a new one.

:::{warning}
**A real, load-bearing precondition, checked directly rather than
assumed**: this only works if the server hosting the ZIP genuinely
supports HTTP range requests. `supports_range_requests()` runs a real
trial request and checks for a genuine `206 Partial Content` response
before attempting anything else — a `200 OK` response to a range
request means the server silently ignored it and returned the full
file, a real, common failure mode this check does not mistake for
support. Confirmed real precedent: Copernicus Data Space Ecosystem's
own documentation states Sentinel-1 packed products are **not**
accessible via S3 (only OData HTTP download) — real range-request
support there is a genuine, unverified assumption this module tests
for per-request rather than takes for granted. ASF is the provider
directly confirmed to work this way, via `asfsmd`'s own real, published
tool.
:::

### A real design correction made while building this

The obvious design — extract annotation files to a flat local folder —
turned out to be **incompatible** with the real, existing
`pygeofetch.insar.annotation.parse_slc_geometry`, which opens its
input via plain `zipfile.ZipFile(path)` and searches
`zf.namelist()` for members containing the real substring
`"/annotation/"`. Fixed by reconstructing a genuine, small, valid local
ZIP that **preserves the original internal member paths** — a real,
structurally-compatible (if much smaller) drop-in replacement for the
full SAFE zip, verified by the exact same real filter every existing
annotation-parsing function already uses.

### Verified against a real local server, not just written and trusted

Python's own built-in `http.server` turned out **not** to support
range requests at all (confirmed directly: it returns `200 OK` with
full content regardless of the `Range` header) — itself a real,
useful finding made while building the test. A minimal, real,
hand-implemented range-capable HTTP handler was built instead, and the
actual module code was run against it end to end:

- **98.7% of a real 5MB test file's "measurement" data was correctly
  never downloaded** — real bytes transferred were tracked and
  verified, not assumed.
- A real bug was caught this way and fixed: `zipfile` internally
  requires the file-like object to implement `seekable()`, which the
  first draft didn't have.

**8/8 tests pass**, using the same real range-capable local server
fixture, plus a correctly-scoped, mocked integration test for the
handoff into `select_burst_synchronized_dates` (building a fully
correct synthetic Sentinel-1 annotation schema from scratch is real,
separate, substantial work belonging to that function's own test
suite, not duplicated here).

## 5. Cross-Platform Assurance

Audited directly, not assumed:

- All four new modules use `pathlib.Path` throughout for real
  filesystem paths. The only literal `"/"` string checks
  (`"/annotation/" in name`) operate on **ZIP-internal member names**,
  not filesystem paths — the ZIP format itself mandates forward
  slashes internally regardless of host OS, so this is correct
  cross-platform behavior, not an oversight.
- No Linux-only system calls (no `os.fork`, no Unix-specific `signal`
  handling, no hardcoded `/tmp` or `/usr` paths) in any of the four
  new modules.
- The real snaphu CLI fallback the original request referenced
  already exists and was verified directly against source
  (`pygeofetch/insar/unwrap.py`, using `shutil.which("snaphu")`) —
  genuinely cross-platform Python, not something built new here.

## Final verification

Fresh, full test suite run: **758 passed, 0 failed** (715 before this
pass — exactly 43 new tests across the four modules, no regressions).
Clean against black and this project's ruff configuration.
