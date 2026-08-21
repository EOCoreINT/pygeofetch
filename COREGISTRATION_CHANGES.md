# Coregistration upgrades — cross-correlation refinement, robust fitting, residual QA

Addresses the gap between this pipeline's orbit/DEM-based coregistration and
SNAP's CrossCorrelationOp + WarpOp, cross-checked directly against ESA's own
docs:
- https://step.esa.int/main/wp-content/help/versions/13.0.0/snap-toolboxes/eu.esa.microwavetbx.sar.op.insar.ui/operators/CrossCorrelationOp.html
- https://step.esa.int/main/wp-content/help/versions/13.0.0/snap-toolboxes/eu.esa.microwavetbx.sar.op.insar.ui/operators/WarpOp.html

## What changed

### `insar/coregister.py`

- **`refine_offsets_by_coherence()`** (new) — SNAP CrossCorrelationOp's own
  two-stage GCP refinement: coarse integer-pixel cross-correlation of
  imagettes, then sub-pixel coherence maximization via Powell's method
  (`scipy.optimize.minimize(method="Powell")`), with a coherence threshold
  that drops unreliable GCPs. This is what closes the gap between the
  orbit/DEM model's ~1-pixel-accurate first estimate and the sub-pixel
  accuracy interferometry actually needs.
- **`fit_offset_polynomial_robust()`** (new) — SNAP WarpOp's iterative
  mean-RMS GCP outlier rejection (fit → residuals → drop anything above
  mean RMS → repeat up to 2×→ optional final absolute threshold), returning
  a `CoregistrationQuality` dataclass with per-fit RMS/residual statistics
  instead of just a bare polynomial with no way to sanity-check it.
- **`CoregistrationQuality`** (new dataclass) — GCP counts, RMS mean/std,
  row/col residual mean/std, iterations used, `.is_reliable()`, `.to_dict()`.
- **`fit_offset_polynomial()`** — now supports degree 1, 2, *and 3* (was
  1/2 only), refactored to a generic exponent-based design matrix so all
  three degrees share one code path.
- **`compute_offset_field()`** — now logs a runtime warning pointing callers
  at `compute_offset_field_from_dem()` instead (this function still uses
  `solve_ground_point()`, which has a documented reliability gap).

### `insar/interferogram.py`

- `InterferogramGenerator._orbit_based_coregister()` now runs the full
  4-stage chain: DEM/orbit estimate → coherence refinement (optional, on by
  default) → robust polynomial fit → resample. Returns
  `(resampled, coreg_metadata)` instead of a bare array.
- `process_pair()` gained three new optional params: `coregistration_refine_by_coherence`
  (bool, default `True`), `coregistration_degree` (1/2/3, default `1`),
  `coregistration_rms_threshold` (float or `None`).
- `InterferogramResult.metadata` now includes `coregistration_method`,
  `coregistration_refined_by_coherence`, `coregistration_gcps_initial`,
  `coregistration_gcps_final`, `coregistration_rms_mean_px`,
  `coregistration_mean_coherence`.

### `insar/__init__.py`

- Exports `fit_offset_polynomial_robust`, `refine_offsets_by_coherence`,
  `CoregistrationQuality`.

## What did *not* change (and why)

SNAP's **CreateStack** (initial raster collocation before GCP selection)
doesn't have a direct equivalent here, deliberately. CreateStack matters in
SNAP because coregistration there starts from a raster-alignment
assumption. This pipeline instead computes each GCP's offset directly from
orbit/timing physics (`geodetic_to_ecef` + `find_zero_doppler_time`), so a
reference and secondary crop with different geometries are already handled
through the `ref_row_offset`/`sec_row_offset` full-scene coordinate
correction in `resample_with_offset_field()` — there's no raster-grid
assumption to collocate away. Bolting on a literal CreateStack step would
duplicate what the orbit-based path already does more precisely.

## Update: SNAP CreateStack-equivalent (raster collocation)

Added a second, selectable coregistration strategy that mirrors SNAP's
actual CreateStack → CrossCorrelationOp → WarpOp chain, rather than only
the physics-first approach above.

### `insar/coregister.py`

- **`collocate_by_geocoding()`** (new) — faithful implementation of
  CreateStack's documented behavior: resamples the secondary directly
  onto the reference's geographic raster using each file's own embedded
  CRS + affine transform (via `rasterio.warp.reproject`, real/imag split
  for phase-preserving complex resampling). Returns
  `(collocated, coverage_fraction, valid_mask)` — `coverage_fraction` is
  an explicit number (SNAP's UI only exposes this implicitly via its
  Reference/Maximum/Minimum extent option), so a near-zero-overlap
  coordinate-frame bug is caught immediately instead of surfacing later
  as unexplained low coherence.
- **`_regular_grid_points()`** (new) — uniform GCP-candidate grid used to
  seed refinement after collocation, where (unlike the DEM-driven path)
  there's no ground-point-derived grid to start from.

### `insar/interferogram.py`

- **`_raster_collocation_coregister()`** (new) — the full chain:
  `collocate_by_geocoding()` → `refine_offsets_by_coherence()` (wider
  `coarse_search_radius=12`, vs. the orbit/DEM path's default 4 — see
  precision note below) → `fit_offset_polynomial_robust()` → final
  resample. Same `(resampled, coreg_metadata)` contract as
  `_orbit_based_coregister()`, plus a `collocation_coverage_fraction` key.
- `process_pair()` gained `coregistration_method: str = "orbit_dem"`
  (new value: `"raster_collocation"`) to select between the two
  strategies. Only needs a real CRS + transform on both files — no DEM,
  SAFE zips, or orbit files required.
- **Bug fix (prerequisite for the above):** `process_pair()` was reading
  the secondary with `_read_complex(secondary, ref_shape=ref_complex.shape)`,
  which (a) silently discarded the secondary's own real CRS/transform,
  and (b) ran a naive, ungeoreferenced shape-only zoom resample
  *unconditionally, before any real coregistration path ever ran*,
  whenever ref/sec crop shapes genuinely differed. Both the orbit/DEM
  path's own crop-offset handling and the new raster-collocation path
  are explicitly built to handle differing ref/sec geometry correctly —
  this early resample was silently working around them. Now reads the
  secondary at its own native shape/profile; shape reconciliation
  happens only within (or as the deliberate final fallback after) real
  coregistration.

### `insar/__init__.py`

- Exports `collocate_by_geocoding`.

## Important precision caveat (found while building this, not fixed here)

SNAP's CreateStack collocates using each product's own **precise**
geocoding. Checked directly against `extraction.py`: pygeofetch's
`SLCExtractor` assigns its extracted GeoTIFFs a transform fitted via
`rasterio.transform.from_gcps()` — a single global affine fit to the
SAFE product's sparse embedded GCPs, explicitly documented there as an
approximation (with a 15%+ safety margin baked into AOI cropping
specifically because of it), not verified to sub-pixel or even
few-pixel accuracy, and structurally unable to capture SAR's real
range-dependent nonlinear geocoding distortion the way true per-pixel
or dense tie-point geocoding would.

`collocate_by_geocoding()` itself is fully correct and general — tested
against an accurate transform (Scenario A below) with excellent results.
The caveat is specific to feeding it pygeofetch's *current* extraction
output. Concretely measured via `test_raster_collocation_integration.py`:

- **Accurate transform:** raw coherence 0.006 → 0.964 after
  `raster_collocation` (96% real coverage, refinement barely needed —
  RMS 0.0005 px).
- **Imprecise, GCP-fit-style transform** (~6 px translation + slight
  rotation error injected into the declared transform, array data left
  at its true position): raw coherence 0.003 → only 0.176 from
  collocation alone → **0.782** once cross-correlation refinement runs
  on top. Same "stage 1 rough, stage 2 fixes it" pattern already proven
  for the orbit/DEM path, and why `_raster_collocation_coregister()`
  uses a wider default search radius than that path does.

**Bottom line:** `raster_collocation` is a genuine, working
CreateStack-equivalent, but with today's `SLCExtractor` transforms its
first stage alone is materially less precise than `orbit_dem`'s; it
depends on the refinement stage to reach comparable final quality. If
`orbit_dem`'s required inputs (DEM + SAFE zips + orbit files) are
available, that path's first-stage estimate is more reliable on its
own. Getting `raster_collocation` to true SNAP-equivalent precision at
the collocation stage itself would mean replacing `extraction.py`'s
global GCP-fit with genuine per-pixel or dense tie-point geocoding
(reusing the same `annotation.py`/`geolocation.py` physics
`compute_offset_field_from_dem()` already trusts) — a separate, larger
piece of work, not undertaken here.

## Validation (this update)

- `test_raster_collocation_integration.py` (new) — both scenarios above,
  run through the real, public `process_pair()` API end to end
  (dispatch, `sec_profile` reading, collocation, refinement,
  interferogram formation, coherence estimation, metadata) — not a
  reimplementation or a private-method-only test.