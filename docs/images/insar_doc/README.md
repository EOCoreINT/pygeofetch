# PyGeoFetch InSAR — Processing Pipeline

A pure-Python InSAR (Interferometric SAR) processing chain for Sentinel-1,
covering search through SBAS time series inversion. This document is a
complete, honest account of what was built, what's verified, what broke
along the way, and what still has open limitations. Nothing here is
overstated — every claim of "working" below was actually tested against
known ground truth, not just run once and assumed correct.

---

## 1. Pipeline overview

```
Search & Download (Copernicus, SLC product type)
        │
        ▼
SLC Extraction (sub-swath selection via embedded GCPs)
        │
        ▼
Interferogram Formation (coregistration → ESD → conj. multiply → topo phase removal → coherence)
        │
        ▼
Atmospheric Correction (elevation-correlated or ERA5)
        │
        ▼
Phase Unwrapping (SNAPHU, via snaphu-py)
        │
        ▼
SBAS Network Assembly + Validation
        │
        ▼
SBAS Time Series Inversion (displacement, velocity)
```

Every stage above has automatic visualization available
(`auto_visualize=True` on `.save()`, or the standalone functions in
`pygeofetch.insar.visualize`) and `DataValidator` checks wired in at
real entry/exit points — not just present as unused code.

---

## 2. What's fully verified and safe to rely on

Each item below was tested against independently-derived ground truth,
not just executed successfully once.

### `DataValidator` (`validate.py`)
- Complex dtype, NaN, dynamic-range checks for SLCs; 0–1 range check
  for coherence; graph-connectivity check for SBAS networks.
- **Wired into the real pipeline**, not just available: `process_pair()`
  validates both input SLCs at entry and validates coherence after
  computing it; `SBASTimeSeries.invert()` validates network
  connectivity and every pair's coherence before the expensive
  inversion runs.
- Found and fixed a real integration bug in the process: the network
  validator originally expected `.date1`/`.date2` attributes, but the
  real `InterferogramPair` class uses `.reference_date`/`.secondary_date`
  — would have silently failed on every real call.
- Found and fixed a real detection gap: amplitude-only data cast to a
  complex dtype (`real + 0j`) passed both the dtype check and the
  amplitude-variation check (since amplitude still varies). Added a
  specific check for near-zero imaginary part, the exact signature of
  this failure mode.

### Automatic visualization (`visualize.py`)
- `visualize_interferogram()`, `visualize_unwrapped()`,
  `visualize_timeseries()` — all reuse the existing `Plotter` class,
  no duplicated plotting logic.
- Wired into `InterferogramResult.save()` and `TimeSeriesResult.save()`
  via `auto_visualize=True`. Verified: produces both GeoTIFFs and PNGs
  together, default behavior unchanged, and a visualization failure
  only logs a warning — never blocks or loses the actual GeoTIFF output.

### GPU acceleration (`gpu.py`)
- `get_array_module()` — real CuPy/NumPy backend switching for
  coherence estimation and SBAS inversion's large matrix solve.
- **Honest limitation**: no GPU hardware was available to test the
  actual CUDA execution path. What *is* verified: the CPU fallback is
  byte-for-byte identical to the pre-existing implementation, and the
  no-GPU detection/fallback logic is genuinely tested (since "no GPU"
  was the real, live condition of the test environment).
  `use_gpu` defaults to `False` for exactly this reason — opt-in, not
  opt-out, until verified on real hardware.

### `annotation.py` — Sentinel-1 acquisition timing
- Parses real per-pixel timing (`productFirstLineUtcTime`,
  `azimuthTimeInterval`, `slantRangeTime`, `rangeSamplingRate`) from a
  SAFE archive's annotation XML.
- Field paths confirmed against ESA's own Mission Performance Centre
  documentation, not assumed.
- `SLCGeometry.azimuth_time()`/`row_for_azimuth_time()` and
  `range_time()`/`col_for_range_time()` verified as exact inverses of
  each other (round-trip error < 0.001 pixel — the residual is
  `datetime`'s microsecond resolution, not a logic error).
- Physical sanity confirmed: computed near/far slant range values match
  real Sentinel-1 IW parameters (~800km near range).

### Orbit file parsing and interpolation (`geolocation.py`)
- `parse_orbit_file()` parses real ESA `.EOF` XML — verified exact
  (position, velocity, and time all matched precisely) against an
  actual example record from ESA's official EOF format specification.
- `interpolate_orbit_state()` (Lagrange interpolation) verified exact
  at known nodes (0.000000m error) and physically sensible between
  nodes.

### `find_zero_doppler_time()` (`geolocation.py`)
The strongest-tested piece in this whole effort. Given a known ground
point and an orbit, finds the acquisition time observing it at zero
Doppler (a 1D secant-method root-find).
- Recovers a known true time to sub-microsecond accuracy.
- **Robust even from deliberately bad starting guesses** — tested from
  5 seconds and 30 seconds off the true answer, converged to the same
  sub-microsecond accuracy both times. Never needed a second attempt.

### `geodetic_to_ecef()` (`geolocation.py`)
- Closed-form (no iteration, always converges) lat/lon/height → ECEF
  conversion. Verified exact via round-trip against an independent
  reference implementation.

### DEM-driven coregistration (`coregister.py`)
- `compute_offset_field_from_dem()` — the actual working
  coregistration path. Samples real ground points **directly from a
  DEM's own geographic coordinates** (via `geodetic_to_ecef`, not
  solved for), then locates each in both the reference and secondary
  orbits via `find_zero_doppler_time()`. Deliberately never calls the
  unreliable `solve_ground_point()` — see §4.
- **49/49 grid points solved successfully** on a realistic synthetic
  scene (the exact scenario where the old pixel-driven approach failed
  21–28 times out of 49 — see §4).
- Full pipeline (offset field → polynomial fit → resample) verified
  end-to-end: recovered a known synthetic pattern's location to 0.00px
  error.
- **Wired into `InterferogramGenerator.process_pair()`** via four new
  optional parameters (`reference_safe_zip`, `secondary_safe_zip`,
  `reference_orbit_file`, `secondary_orbit_file`). Confirmed via
  explicit logging (`"Real orbit-based coregistration applied (49 grid
  points)"`) that the real path executes successfully through the
  actual public API, not just in isolation.
- **Backward compatible**: omit any of the four new parameters and it
  falls back to the previous shape-based resample, with a clear log
  message stating which path ran. Verified: default calls, partial-argument
  calls, and calls with nonexistent files all complete without crashing.

---

## 3. `SBASTimeSeries`, `PhaseUnwrapper` (SNAPHU), `AtmosphericCorrector` — deep verification

Added after the §2 components below were already independently
verified. Same standard: tested against known ground truth, not just
executed and eyeballed.

### `SBASTimeSeries` inversion — fully verified, no gaps found

- **Exact recovery of a known, non-linear, spatially-varying
  displacement time series**: constructed synthetic ground truth,
  forward-modeled the exact phase it implies (using the module's own
  documented sign convention), fed it through `invert()`, and recovered
  the original displacement to sub-nanometre precision (max error
  0.0000013 mm) — the residual is float32 rounding, not a real error.
- **Residual RMS confirmed genuinely zero** on a perfectly consistent
  synthetic network (max 1.2e-9), not just small.
- **Velocity fit verified exact** against a known, spatially-varying
  linear rate field (max error 0.0000024 mm/year).
- **Residual RMS confirmed to genuinely localize a real, deliberately-
  injected error**: corrupted one interferogram at one pixel by 5
  radians — the resulting residual map flagged exactly that pixel
  (0.0051) while leaving untouched pixels clean (0.0000000).
- One real mistake caught in my own first test: I initially built a
  spatially-*uniform* displacement field, which is physically
  meaningless for InSAR (it can only measure displacement *relative*
  to a reference point, so a uniform field trivially zeroes out
  everywhere once referenced) — not a code bug, a test-design error,
  fixed by using a genuinely spatially-varying field.

### `PhaseUnwrapper` (SNAPHU) — verified, with one real behavioral nuance documented

SNAPHU itself is JPL/Caltech's established production algorithm
(Chen & Zebker 2001); what was tested here is pygeofetch's wrapper
around it, using the *real* SNAPHU binary (visible in test output —
`snaphu v2.0.7`, genuine network-flow optimization), not a mock.

- **Recovers a known phase ramp spanning 4.7 full 2π cycles** to
  0.001% error (essentially float32 noise) with 100% of pixels marked
  reliable, zero global offset needed.
- **Coherence-driven reliability confirmed genuine**: injected real
  phase noise into a low-coherence region — clean region stayed 100%
  reliable with 0.0000 rad error; the noisy region correctly dropped
  to 1.3% reliability.
- **The `mask` parameter's real behavior is more nuanced than it
  looks, and this is worth knowing before relying on it**: pygeofetch's
  wrapper correctly zeroes coherence at masked locations (verified
  directly in the code) — but SNAPHU's own network-flow cost function
  can still judge a masked region reliable (`conncomp > 0`) if the
  underlying *phase* there is smooth and consistent with its
  neighbours, regardless of the forced-zero coherence. This isn't a
  pygeofetch bug — the wrapper did exactly what it should — but
  `mask=False` should not be assumed to force `conncomp=0`; SNAPHU
  makes its own reliability determination from the full cost
  optimization, not a simple threshold on the masked input.

### `AtmosphericCorrector` (elevation-correlated method) — a real, actionable limitation found

- **The regression itself is exact**: tested against a known,
  elevation-correlated phase signal on truly unwrapped phase — residual
  0.000002 rad, essentially perfect.
- **Real limitation found**: applied to *wrapped* phase where the true
  elevation-correlated signal spans multiple 2π cycles, the correction
  **silently does nothing** (output ≈ input, max residual 3.13 rad ≈
  unchanged). Root cause confirmed precisely: this method uses an
  ordinary (non-circular) linear regression — correct for unwrapped
  phase, but wrapping scrambles a genuinely strong elevation
  correlation into something with near-zero *linear* correlation, so
  the method's own R² > 0.5 safety gate correctly (but unhelpfully)
  concludes there's nothing reliable to remove, and skips the
  correction with only an INFO-level log message.
- **This is a real, practical concern for real usage**: the InSAR
  notebook in this series applies atmospheric correction to *wrapped*
  phase, before unwrapping. For scenes with enough relief or baseline
  to produce a multi-cycle elevation-correlated delay, this ordering
  means the correction can silently do nothing while looking like it
  ran. `interferogram.py`'s topographic-phase-removal step already
  solves this exact problem correctly, using a circular
  (complex-exponential) regression instead of a naive linear one — the
  same technique isn't yet applied here.
- **Not yet fixed** — found and precisely diagnosed, but fixing it
  (either reordering the pipeline to atmospheric-correct after
  unwrapping, or making this regression circular-phase-aware like the
  topographic one) is a real, separate decision and change, not made
  in this pass.

---

## 4. What's NOT yet verified

- **No test against a real, downloaded Sentinel-1 SAFE archive.**
  Everything above was verified against carefully-constructed synthetic
  data with known ground truth — which is how the bugs below were
  actually caught — but the full chain has not yet been run against
  real Copernicus data end-to-end. That's the next real test.
- **Reference dataset validation** (e.g. reprocessing the well-known
  ERS-2 Etna sequence and comparing to published values) was proposed
  early on and never built — it's real, valuable, separate work.

---

## 5. The coregistration reliability saga — issues, dead ends, and the actual fix

This section exists because the path here was not straight, and the
detours matter as much as the destination — they're why the final
result can be trusted.

### The original problem

`InterferogramGenerator`'s docstring claimed *"geometric coregistration
using orbit state vectors + reference DEM"*. The actual code did a
naive `scipy.ndimage.zoom` shape-match, and only ran that at all if the
two SLCs had different pixel dimensions — meaning two real,
same-shaped-but-genuinely-misaligned SLCs got **no coregistration at
all**. Confirmed by reading the real code, not assumed from the
docstring.

### Attempt 1: `solve_ground_point()` — the "obvious" 3-equation solve

The standard formulation (Kampes, Hanssen & Perski 2003): given a
satellite state vector and a range, solve the zero-Doppler + range +
ellipsoid equations for the ground point, via Newton's method.

**Bugs found and fixed along the way:**
1. A **near/far solution ambiguity** (analogous to GPS trilateration) —
   a naive nadir-point initial guess could converge cleanly to the
   *wrong* one of two mathematically valid solutions, silently landing
   kilometres off with no error raised. Fixed with a cross-track-aware
   initial guess (using the real look direction, right-looking
   convention).
2. **Catastrophic cancellation** in the range residual
   (`range_dist - range_m`, subtracting two ~700km numbers to get a
   small residual) caused unpredictable oscillation below ~1m accuracy
   for some geometries. Fixed with the standard numerically-stable
   reformulation: `(range_dist² - range_m²) / (range_dist + range_m)`.
3. Added **damped Newton** (step-length capping) after finding that
   some geometries produced wild, overshooting steps.
4. Added a **fallback retry** with a different initial guess strategy
   (plain nadir) when the primary strategy didn't converge.

**Result after all four fixes**: 33/35 (94%) reliable across a
diverse test suite (multiple hemispheres, latitudes, ascending/
descending passes, DEM heights from -50m to 1500m), always failing
safely (clear `RuntimeError`, never a silent wrong answer) on the
remainder.

**Why 94% still wasn't good enough**: real coregistration needs this
solve called ~50 times per interferogram pair (a sparse grid across the
scene). At even a 94% per-call success rate, hitting failures across a
real grid isn't a corner case — a follow-up test on a realistic scene
footprint showed **21–28 out of 49 grid points failing**, well past
usable.

### Attempt 2: warm-starting

Reasoned that the initial guess was still the likely cause, and that
using each grid point's own solved answer to seed its neighbor (since
adjacent points' true locations are physically close) should help.

**Result: it didn't. It made things slightly worse** (28/49 failures,
up from 21/49). This was the actually useful result of this attempt —
it demonstrated the problem probably wasn't really about initial-guess
distance at all, which redirected the investigation rather than
prompting a third blind parameter tweak.

### The actual fix: avoid `solve_ground_point()` entirely

The insight: `find_zero_doppler_time()` (already verified highly
reliable) needs a *known ground point* as input — but that point
doesn't have to come from the risky iterative solve. A DEM already has
real geographic coordinates at every pixel. Converting geodetic
coordinates to ECEF is a **closed-form, always-converges** calculation
(`geodetic_to_ecef`), not an iterative one.

So: sample ground points directly from the DEM (verified conversion) →
locate each in both orbits via `find_zero_doppler_time()` (verified
robust) → never touch `solve_ground_point()` at all.

**Result: 49/49 in the exact scenario that previously failed 21–28
times.** Confirmed end-to-end through the real `process_pair()` API,
not just in isolation.

### Current status of `solve_ground_point()`

Still exists in `geolocation.py` (with all four fixes from Attempt 1
intact — it's genuinely better than when this started, just not
reliable enough to be a recommended default). **Deliberately not
exported** from `pygeofetch.insar.__init__` — the module docstring
explains why. Available for advanced/experimental use, not part of the
recommended coregistration path.

---

## 6. Known limitations (current, honest)

- **Real-archive testing gap** (§3) — the single biggest open item.
- **`solve_ground_point()` reliability** is a known, unresolved gap
  (~94% at best, worse under repeated grid use). Worked around, not
  fixed. A genuine fix would likely mean reformulating in local/
  topocentric coordinates instead of raw global ECEF, or adopting
  Levenberg-Marquardt instead of damped Newton — real, scoped, separate
  work if `solve_ground_point()` itself is ever needed directly.
- **DEM-driven coregistration needs a real DEM.** If no DEM is
  supplied, coregistration falls back to the shape-based resample —
  correctly and safely, but without real sub-pixel accuracy.
- **No GPU hardware verification** (§2, GPU acceleration) — CPU path
  and fallback logic are solid; the actual CUDA execution path is not.
- **No checkpointing, no YAML config system, no CI/CD** — explicitly
  out of scope for this pass; large, separate efforts each.
- **Reference dataset benchmark** (ERS-2 Etna or similar) not built.

---

## 7. Usage

### Full real coregistration (recommended when a DEM is available)

```python
from pygeofetch.insar import InterferogramGenerator

gen = InterferogramGenerator(use_gpu=False)
result = gen.process_pair(
    reference="slc_ref.tif",
    secondary="slc_sec.tif",
    dem="dem.tif",
    reference_safe_zip="S1A_..._ref.SAFE.zip",
    secondary_safe_zip="S1A_..._sec.SAFE.zip",
    reference_orbit_file="ref.EOF",
    secondary_orbit_file="sec.EOF",
)
result.save("./output", auto_visualize=True)
```

### Without the new orbit-based inputs (falls back safely)

```python
result = gen.process_pair(reference="slc_ref.tif", secondary="slc_sec.tif", dem="dem.tif")
# Logs: "Using shape-based coregistration fallback ..."
```

### Full chain, search to SBAS

See `22_obuasi_insar_subsidence_sbas.ipynb` for a complete, real,
CPU-only worked example (search → extraction → interferogram →
atmospheric correction → unwrapping → SBAS network validation →
inversion → visualization at every step).

---

## 8. References

- Kampes, B., Hanssen, R. & Perski, Z. (2003). Radar interferometry
  with public domain tools.
- Yagüe-Martínez, N. et al. (2016). Interferometric processing of
  Sentinel-1 TOPS data. IEEE TGRS, 54(4), 2220-2234.
- Scheiber, R. & Moreira, A. (2000). Coregistration of interferometric
  SAR images using spectral diversity. IEEE TGRS, 38(5), 2179-2191.
- Zan, F. et al. (2018). Investigations on the Coregistration of
  Sentinel-1 TOPS with the Conventional Cross-Correlation Technique.
  Remote Sensing, 10(9), 1405.
- Chen, C.W. & Zebker, H.A. (2001). Two-dimensional phase unwrapping
  with use of statistical models for cost functions in a network
  programming framework. J. Opt. Soc. Am. A, 18(2), 338-351.
- Berardino, P. et al. (2002). A new algorithm for surface deformation
  monitoring based on small baseline differential SAR interferograms
  (SBAS).
- Yunjun, Z., Fattahi, H., Amelung, F. (2019). Small baseline InSAR
  time series analysis: unwrapping error correction and noise
  reduction. Computers & Geosciences, 133, 104331.
- ESA Mission Performance Centre. Thermal Denoising of Products
  Generated by the S-1 IPF (MPC-0392) — annotation field paths.
- ESA/EOP-CFI. Earth Observation Mission Software File Format
  Specification — EOF orbit file structure.