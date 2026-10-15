# Advanced InSAR modules

Four modules that build on the InSAR and optical chains:

1. An **auto-fallback router** that decides between InSAR and optical displacement.
2. **Uncertainty export** for phase and displacement rasters.
3. **Geotechnical risk mapping** from an SBAS time series.
4. **Lightweight prefetch**: read Sentinel-1 annotation metadata from a remote SLC ZIP without downloading the measurement data.

`compute_horizontal_strain` in `pygeofetch.optical.offset_tracking` already includes MAD-based
outlier protection (`mad_outlier_threshold`, default 8.0). It was added after a 617% strain
artifact appeared in the Bu'ertai validation run. There is no separate robust variant: the base
function is the robust one. See
[Optical offset tracking](optical-offset-tracking.md#strain-turning-displacement-into-a-risk-metric).

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
directly. Its distinct value is the *decision*: if InSAR's own
reliable-pixel fraction and mean coherence both already clear
configurable thresholds (`reliable_fraction_threshold`,
`mean_coherence_threshold`, both default `0.3`), optical processing is
**skipped entirely** — a compute and bandwidth saving over
always running both, which is what the existing multi-sensor pipeline
does unconditionally.

### Reliability metrics

`TimeSeriesResult` (SBAS's own output type) carries no
`reliable_fraction` or coherence field: `SBASTimeSeries.invert()` only populates
`metadata={"method": ...}`. The router computes both metrics from what is available: coverage from
`np.isfinite(velocity).mean()`, mean coherence averaged across
the interferogram pairs used.

### Mask value conventions

`fuse_insar_optical`'s own `reliability_source` output uses
`2=InSAR-dominant, 1=transition, 0=optical-dominant`. This router's
`processing_source_mask` uses a **different, deliberate** convention:
`0=Optical Fallback, 1=InSAR Valid, 2=Blended`. The two are not
interchangeable: passing one function's mask into code that expects the other's would invert
the meaning of "good" and "bad". Each function's docstring documents its own convention.

### Failure behaviour

When both thresholds fail and no optical inputs were supplied, the router raises a `ValueError`
naming the current `reliable_fraction` and coherence values. It does not return a degraded
InSAR-only result. Either metric alone can trigger the fallback (OR logic).

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

### The formula, reused not reinvented

```{math}
\sigma_\phi = \frac{\sqrt{1-\gamma^2}}{\gamma\sqrt{2N}}, \qquad
\sigma_{disp} = \frac{\lambda}{4\pi}\sigma_\phi
```

This is the standard Cramér-Rao interferometric phase-uncertainty
bound. Rather than derive it a second time, this module reuses the
**exact same formula** already present in `pygeofetch.insar.synthetic`
(used there to generate realistic synthetic decorrelation noise) —
confirmed numerically identical by direct test, so the two can never
silently diverge.

### What `write_provenance_manifest` needed

Its existing signature already accepts a generic `quality: dict`
parameter — nothing about it needed to change.
`summarize_uncertainty_for_provenance` builds the right dict
shape to hand it, and a direct integration test confirms the
output feeds straight into the actual function without adaptation.

**12/12 tests pass**, including exact-formula-match, physical
monotonicity (higher coherence → lower uncertainty; more looks → lower
uncertainty), a two-band GeoTIFF round-trip, and the real
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

### The cited standard behind the strain cap

`RiskMapperConfig.max_severe_strain` defaults to **1.0%** — the real,
published National Coal Board (1975) *Subsidence Engineering
Handbook*'s own "very severe" damage category threshold. This is a
cited, standard, not a placeholder.

`RiskMapperConfig.strain_cap` (default 10%) is a **separate**, harder
ceiling — a direct, explicit response to this project's own real,
observed 617% strain artifact. The test suite reproduces that exact
scenario (strain=6.17) and confirms the cap neutralizes it before it
can dominate the risk index, independent of whatever outlier
protection already ran upstream in `compute_horizontal_strain` itself.

### An note on `compute_curvature`'s interface

The standard mining-subsidence convention computes curvature from a
*single* vertical displacement field. This implementation generalizes
to displacement *magnitude* from two horizontal components
(`dx`, `dy`) for broader, multi-modal applicability (lateral landslide
movement, co-seismic slip) — documented explicitly as a departure
from, not identical to, the classic single-field convention. Pass a
zero array as `dy` to reproduce standard single-field curvature
exactly.

### Combination logic

`config.combination="max_weighted"` (default) computes
`0.6 * max(scores) + 0.4 * mean(scores)` — a single severe factor
dominates the index rather than being diluted by two mild ones,
matching geotechnical practice where one severe indicator alone
is cause for concern. `"mean"` is available as a simpler,
alternative.

**15/15 tests pass**, including an exact analytical curvature check (a
parabolic field's known constant second derivative) and direct
verification that `max_weighted` gives materially higher risk than a
plain mean for the same inputs.

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
    # Only now trigger the full multi-gigabyte SLC download.
    client.download([ref_scene, sec_scene], destination="./raw_slc")
else:
    print(f"Pair excluded before any real full download: {result['family_report']}")
```

### Background

The published, open-source `asfsmd` tool (Valentino, PyPI/GitHub)
already does exactly this against the ASF archive: opening a
remote Sentinel-1 SLC ZIP's central directory via HTTP range
requests and extracting only the annotation XML, never fetching the
measurement rasters. This module implements the same real
technique directly with `httpx` (already a core pygeofetch dependency)
rather than adding `fsspec` as a new one.

:::{warning}
**This works only if the server supports HTTP range requests.** `supports_range_requests()`
sends a trial range request and checks for `206 Partial Content`. A `200 OK` means the server
ignored the range and returned the whole file, which this check does not accept as support. ASF
works this way (as the `asfsmd` tool shows). Copernicus Data Space documents Sentinel-1 packed
products as not accessible through S3, so range support there is tested per request, not assumed.
:::

### Local ZIP layout

Extracting annotation files into a flat folder would be incompatible with
`pygeofetch.insar.annotation.parse_slc_geometry`, which opens its
input via plain `zipfile.ZipFile(path)` and searches
`zf.namelist()` for members containing the substring
`"/annotation/"`. The module therefore
reconstructs a small, valid local ZIP that **preserves the original internal member paths**,
as a drop-in replacement for the full SAFE zip.

### Tests

The tests run against a local HTTP handler that implements range requests (Python's built-in
`http.server` ignores the `Range` header and returns `200 OK` with the full content). In the
tests, 98.7% of a 5 MB file's measurement data was never transferred, with bytes counted by the
server. A mocked test covers the handoff into `select_burst_synchronized_dates`.

## 5. Platform notes

- The modules use `pathlib.Path` for filesystem paths. The only literal `"/"` checks
 (`"/annotation/" in name`) are on ZIP member names, which always use forward slashes
 regardless of the host OS.
- No Linux-only system calls (`os.fork`, Unix `signal` handling) and no hardcoded `/tmp` or `/usr` paths.
- The `snaphu` command-line fallback in `pygeofetch/insar/unwrap.py` locates the executable with `shutil.which("snaphu")`.

