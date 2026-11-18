# pygeofetch: gap register and build plan (SAR / InSAR direction)

Direction: **feasibility-first Sentinel-1 InSAR.** Search, screen on physics before download, download only what can work, hand off to established engines. Optical is frozen except where it supports SAR (offset tracking, multi-modal fallback).

Rules for every gap below:
- No gap is closed until its **acceptance test** passes on saved result files, not on a screenshot.
- Numbers in the paper are read from those result files (paired bootstrap, 5000 resamples, 95% interval where a comparison is made).
- A closed gap is moved to "Closed" with the commit and the result file. A gap that cannot be closed stays here and is stated in the paper's limitations.
- Claims about other tools come from their documentation pages (dated); "not described" is not "absent".

Status key: OPEN, IN PROGRESS, CLOSED. Effort: S = days, M = 1-3 weeks, L = more than a month.

## How the evidence below was gathered
Read from the code in `pygeofetch.zip` (28 Oct 2026) and from the Silesia run logs. Nothing here was re-run. Where I could not confirm something it says "not found", meaning a search of the package found nothing.

---

## G1. Burst-timing check is per family before download, per pair only after processing
- **Evidence.** `preflight.py::_screen_burst_families` groups scenes into families from annotation XML before download. `esd.py::compute_burst_synchronization` gives the pair offset after processing. Silesia: S1B-S1B 12-day pairs measured 5.6 ms and 12.8 ms although the family check passed. ESA requires under 5 ms.
- **Why it matters.** This is the main claim of the package. A pass that does not hold at pair level is a false assurance.
- **First step (must come first).** Verify the offset calculation against the burst start times in the SAFE annotation files for 3-5 pairs, by hand. If the numbers disagree, fix the calculation before anything else.
- **Build.** A pair-level screen that reads annotation timing only (no full download) and returns an offset per candidate pair, with the 5 ms flag.
- **Acceptance test.** For the Q3 2017 track 175 stack, the pre-download offsets equal the post-processing offsets stored by `process_pair` within a stated tolerance, for every pair. Reported as agreement count and maximum difference.
- **Depends on.** Nothing. **Effort.** S-M. **Risk.** The offsets may be a measurement artefact; that is itself a result.
- **Progress (28 Oct 2026).** Step 1, partly done. (a) The arithmetic in `esd.compute_burst_synchronization` was tested against a synthetic orbit and burst grids with known offsets (0, 1, 3, 5.6, 12.8 ms, both signs, 1 s, 12- and 60-day gaps, 1 microsecond cycle mismatch): exact recovery, consistent sign, no wrap-around error. (b) By reading the code, both timing inputs are consistent: burst `azimuthTime` from the annotation and the orbit `UTC` tag, both UTC. So the 5.6 and 12.8 ms values are not explained by the formula or the time scales. **Still open:** the hand-check against real annotation files for 3-5 pairs (needs the SAFE annotation XML, not available in the workspace), and the pair-level pre-download screen.
- **Status.** IN PROGRESS

## G2. Coregistration accuracy is not validated and its reliability test is not enforced
- **Evidence.** `coregister.py` (offset field on sparse points; `is_reliable()` requires at least 6 GCPs and RMS at most 1.0 px). `is_reliable` is not called from `interferogram.py`. Silesia pairs ended with 4 final GCPs, RMS 0.07 px. ESA requires azimuth coregistration better than 0.001 samples for under 3 degrees of phase error. No check against that figure was found.
- **Build.** (a) Enforce or at least report `is_reliable` in `process_pair` results. (b) A residual azimuth-misregistration estimate per pair (from ESD overlaps) compared with 0.001 samples. (c) Increase the GCP count where the offset field allows.
- **Acceptance test.** Every pair carries a coregistration quality record; the Silesia stack reports how many pairs meet 6 GCPs and the 0.001-sample figure, with the distribution.
- **Depends on.** None. **Effort.** M. **Risk.** Many pairs may fail honestly; the paper then says so.
- **Progress (28 Oct 2026).** (a) done: every `process_pair` result now carries `metadata["quality"]` with the coregistration fit checked against 6 GCPs and 1.0 px (flag `coreg_unreliable`), and hard limits `min_coreg_gcps`, `max_coreg_rms_px` that reject a pair. (b) done in code: after the ESD shift is applied it is re-estimated (`esd_residual_after_correction_px`) and the spread across burst overlaps is stored (`esd_overlap_std_px`); both are compared with 0.001 samples (flags `azimuth_misregistration_above_requirement`, `esd_overlaps_inconsistent`). The re-estimate checks that the shift was applied consistently; the spread is the better indicator of real uncertainty. Not yet run on real data. (c) open.
- **Status.** IN PROGRESS

## G3. No head-to-head benchmark against an established engine
- **Evidence.** No comparison with HyP3, ISCE or SNAP outputs was found. Existing tests use synthetic data.
- **Build.** A benchmark script that takes the same pairs processed by pygeofetch and by HyP3 (burst InSAR, ISCE2) and reports coherence difference, wrapped-phase RMS difference after removing a plane, and the share of pixels agreeing within a stated tolerance, with paired bootstrap intervals.
- **User action needed.** Submit the same 3-5 pairs to HyP3 and place the outputs in a folder (I cannot reach HyP3 from the workspace).
- **Acceptance test.** One results file per pair; the paper table is generated from it.
- **Depends on.** G1, G2 (so the comparison is on a defensible pipeline). **Effort.** M. **Risk.** Differences may be large; they are the finding.
- **Status.** OPEN

## G4. Exporters for MintPy and dolphin
- **Evidence.** MintPy appears only in docstrings in `timeseries.py` and `__init__.py`; no export function found. dolphin's documentation says it starts from coregistered SLCs.
- **Build.** `export_mintpy()` (interferogram stack plus geometry in MintPy's input layout) and `export_dolphin()` (coregistered SLC stack plus metadata). Start with the one that needs the least geometry.
- **Acceptance test.** The exported stack is read without error by the target tool on the Silesia pairs, and the tool's own output is saved.
- **Depends on.** G5 (geometry/geocoding). **Effort.** M. **Risk.** Required geometry files may not exist in the pipeline yet.
- **Status.** OPEN

## G5. Output is in radar geometry with an approximate affine transform
- **Evidence.** `interferogram.py` fits an approximate affine transform from GCPs when the SLC has none (around lines 389-402). No proper geocoding step was found in the processing chain (`coregister.py::collocate_by_geocoding` exists but is a collocation helper).
- **Build.** Geocode products with a documented method, and write lookup tables (latitude, longitude, incidence, heading) that exporters need.
- **Acceptance test.** A geocoded coherence map overlays a reference basemap or the DEM with a measured position error stated.
- **Depends on.** Nothing, but G4 depends on it. **Effort.** M-L.
- **Status.** OPEN

## G6. No phase linking for distributed scatterers
- **Evidence.** No EVD or EMI code found in `insar/`. `ps_selection.py` has ADI and temporal coherence only. Silesia mean coherence is about 0.27.
- **Build, in this order.** (1) Do G4 first so dolphin can be used as the reference. (2) Implement a small, tested phase-linking function (EVD, then EMI) for homogeneous patches. (3) Compare against dolphin on the same stack.
- **Acceptance test.** On a stack with a known reference result (dolphin output), the pygeofetch estimate agrees within a stated phase RMS; on Silesia, report the coherent-pixel share before and after.
- **Depends on.** G4. **Effort.** L. **Risk.** High; it may be wiser to document dolphin as the recommended back-end and stop there.
- **Status.** OPEN. Decision needed after G4: build, or recommend dolphin.

## G7. Unwrapping is spatial only
- **Evidence.** `unwrap.py` wraps SNAPHU; pyunwrap is the second method. No temporal or 3D unwrapping found.
- **Build.** Optional hand-off to a 3D unwrapper after G4 rather than writing one.
- **Acceptance test.** Documented hand-off works on the Silesia pairs.
- **Depends on.** G4. **Effort.** S as a hand-off, L if built.
- **Status.** OPEN

## G8. pyunwrap pads small grids heavily
- **Evidence.** Silesia grid is 67 x 847 on a tile of 256: about three quarters of what the network sees is padding (figure V6b).
- **Build.** Choose tile size from the grid, or run on less aggressive multilooking; report both.
- **Acceptance test.** Padding share and agreement with SNAPHU reported for both settings.
- **Depends on.** None. **Effort.** S.
- **Status.** OPEN

## G9. Scale and hardware claims are unverified
- **Evidence.** `gpu.py` states the CuPy path has not been run on real CUDA hardware. No parallel execution was found in `insar/`.
- **Build.** Run the existing GPU path once on real hardware and record the result; add per-pair parallelism only if a benchmark shows it is the bottleneck.
- **Acceptance test.** A timing table on stated hardware.
- **Effort.** S for the check, M for parallelism.
- **Status.** OPEN. Until closed, the paper makes no GPU or scale claim.

## G10. Split-spectrum ionosphere is a diagnostic, not a correction
- **Evidence.** `splitspectrum_diagnostic.py` says so itself.
- **Decision.** Keep as a diagnostic; do not build unless a Silesia result needs it.
- **Status.** OPEN (deferred by decision)

---

## Build order
1. **G1** (check offsets, then pair-level screen) and **G2** in parallel. Both are small and both change what the Silesia run can claim.
2. Rerun the Q3 2017 track 175 stack, then answer the burst-timing against temporal-decorrelation question from saved files.
3. **G5** geocoding, then **G4** exporters.
4. **G3** benchmark against HyP3 using the user-supplied outputs.
5. **G6** decision point: build or recommend dolphin.
6. **G8**, **G9**, **G7** as time allows.

Nothing in steps 3-6 starts until step 2's result is on file, because it decides how much of the package's story rests on the burst-timing gate.

## G11. No generic stack builder for persistent scatterers (found 28 Oct 2026)
- **Evidence.** `ps_selection.py` needs a co-registered amplitude stack and nothing in pygeofetch built one. ps-gnn's `real_stack.py` built it with its own image-based registration (phase correlation, no orbit or ESD), which duplicated pygeofetch's registration and relaxed it.
- **Built (code written, tested on scratch data only).** `insar/ps_stack.py`: `PSStack`, `build_ps_stack` / `build_stack` (master-referenced stack from `process_pair`, exact amplitude recovery, rejection of every untrusted date with the reason), the selector protocol (`ADISelector` built in; any model plugs in through `name` and `select(stack)`), `compare_selections`, `holdout_stability`. `insar/ps_eval.py`: `boost_coherence_with_ps`, `compare_velocity_maps`. ps-gnn keeps only its model-specific adapter.
- **Acceptance test.** (1) The Silesia Q3 2017 stack builds from real `process_pair` output and the report lists every rejection; (2) amplitude recovered from the interferogram matches the amplitude read directly from the coregistered SLC on one date; (3) ADI and PS-GNN compared with `holdout_stability` on real data, intervals by paired bootstrap.
- **Status.** IN PROGRESS. Open points: real-data run, check (2), and ps-gnn's registration code is still in its repo (to delete after (1) passes).

## Changes made to pygeofetch defaults (28 Oct 2026) -- behaviour change, read before re-running old notebooks
- `InterferogramGenerator(remove_flat_earth_phase=None)`: flat-earth phase is now removed **automatically** when both SAFE zips and both orbit files are given. It used to be off unless asked. Results of existing notebooks will change (the full-width range ramp disappears). Pass `False` for the old behaviour.
- `process_pair` now returns `perpendicular_baseline_m` (from the orbits, at the AOI centre's zero-Doppler time) and `temporal_baseline_days`. Not yet tested against real orbit files.
- A registration fallback (shape-based resampling), a failed or missing flat-earth removal and a mixed-satellite or mixed-track pair now raise `RegistrationFallbackWarning`, `FlatEarthWarning` and `MixedSatelliteWarning` (they were only logged). The result metadata records `coregistration_fallback`, `flat_earth_phase_requested`, the platforms and tracks, `cross_satellite`, `cross_track`, and the looks and Goldstein settings.

## G12. Topographic phase was a regression against the DEM, not a simulation (found 28 Oct 2026)
- **Was.** `_remove_topographic_phase*` fitted phase against DEM height and removed it only when R2 >= 0.5. That removes whatever correlates with height (including atmosphere and slope deformation) and nothing where the fit is weak.
- **Built (28 Oct, tested on synthetic geometry only).** `insar/radar_geometry.py` (exact range-Doppler geometry on a 9 x 9 node grid in the coordinates of the output interferogram, with crop, deburst and multilook offsets applied) and `insar/topo.py` (iterative terrain geocoding from the DEM, geometric phase = flat-earth + k1*h + k2*h^2/2 per pixel, removed at full resolution). `process_pair` uses it whenever both SAFE zips and both orbit files are given. The regression stays only as a flagged fallback (`topo_regression_fallback`) when orbit geometry is missing.
- **Measured (synthetic, 27-30 degrees incidence, 53 m baseline, 28 rad of topographic phase):** simulated minus exact per-pixel phase, rms 0.0015 rad, max 0.007 rad; through `process_pair` with 4 x 8 looks and a crop offset the residual phase is 0.001 rad rms. The quadratic height term matters: linear alone is off by 0.14 rad at 1000 m.
- **Limits, stated.** DEM heights are used as ellipsoidal heights unless `dem_geoid_offset_m` is set (about +40 m in Silesia: nearly constant phase offset, and ~N/tan(incidence) geocoding shift). Layover/shadow pixels get flat-earth phase only and are flagged. Geocoding precision rests on the DEM and the orbit. Never run on real orbit files or a real DEM.
- **Acceptance test (open).** On the Silesia stack: residual phase-height correlation before and after (should fall), and the per-date slope test below (baseline-proportional slope = DEM error).
- **Status.** BUILT, NOT VALIDATED ON REAL DATA

## G15. Atmosphere: review and rebuild (28 Oct 2026)
- **Found in `atmosphere.py`:** ERA5/GACOS maps were resized to the interferogram's shape (`zoom`), assuming the DEM or model grid covers the same footprint in the same orientation as the radar crop; one fixed 38 degree incidence for the whole swath; ERA5 hour taken by truncation; a spatial filter that averaged wrapped angles and filled NaNs with zero; a height fit without a ramp term and with an error estimate that treats pixels as independent; no check that a correction helped.
- **Built (tested on synthetic data and a fake pyaps; ERA5/GACOS never run on real model data).** `insar/atmosphere_core.py` (robust height fit with ramp and block-jackknife error, structure-function validation, per-date network solve with a DEM-error test) and the rewritten `AtmosphericCorrector.correct` (geocoding by `geometry` or profile, never by shape; per-pixel incidence; ERA5 hour interpolation; relief, significance and variance-reduction gates; `AtmosphereError` or `on_error="skip"`).
- **Measured (synthetic):** with exact geometry the 15.5 rad GACOS-type delay is removed to the 0.10 rad noise floor and a 3 rad deformation bowl is preserved; with an approximate affine geocoding and a fixed 38 degree incidence the residual is 0.75 rad. A ramp makes a height fit without a ramp term read 8.5 instead of 6.4 rad/km; excluding the deformation area gives 6.1 (truth 6.0). The variance-reduction gate rejects wrong-sign and wrong-date model data. The DEM-scale test detects a 10 % error and misses 2 % with 10 dates.
- **Not done.** (a) Time-series APS filtering (high-pass in time, low-pass in space) for SBAS and PS: not in `timeseries.py` yet. (b) Ionosphere: `ionosphere.py` (IONEX thin shell) and the split-spectrum diagnostic were not reviewed. (c) Real ERA5 download path (needs CDS credentials and pyaps3). (d) Independent validation against GNSS ZTD.
- **Status.** BUILT, NOT VALIDATED ON REAL DATA

## G13. Real burst processing is off by default (found 28 Oct 2026)
- **Evidence.** `InterferogramGenerator(use_real_burst_processing=False)`: no per-burst ESD and no deburst unless asked. Results carry the flag `no_deburst`.
- **Decision needed after G1/G2 are checked on real data:** make it automatic when both SAFE zips are given (as was done for flat-earth), or keep it opt-in. Not changed yet because the burst path has not been run on real data.
- **Status.** OPEN

## G14. ESD applies one global azimuth shift
- **Evidence.** The per-overlap estimates are reduced to their median and applied to the whole secondary. ISCE2 and SNAP apply the correction per burst. The spread is now stored (`esd_overlap_std_px`), so the size of the problem will be visible from the Silesia run.
- **Status.** OPEN (decide from the stored spread)

## Interferogram formation changes (28 Oct 2026) -- results of existing runs will change
- **Multilooking order (bug fixed).** `process_pair` multilooked the two SLCs and multiplied the averages, which leaves the phase noise at the single-look level whatever the number of looks. On simulated speckle (coherence 0.8, 32 looks): phase std 0.92 rad that way, 0.097 rad when the interferogram is formed first and then averaged (measured through `process_pair`: 0.915 and 0.097). It now forms the interferogram at full resolution, removes the geometric phase there, then takes the complex mean. Every multilooked interferogram produced before this was far noisier than necessary. Coherence was always estimated on the full-resolution data and is unchanged.
- **Amplitude** of a multilooked result is now the root-mean-square over the window (it was the log of the averaged complex value, biased low by about 2.5 dB for speckle). Unchanged at 1 x 1.
- **Flat-earth phase (bug fixed).** The old polynomial was fitted in full-swath pixel coordinates but evaluated at the cropped, multilooked array's indices, so the crop offset was never applied and a multilooked product got a range ramp compressed by the number of looks. Replaced by the geometry above.
- **`result.geometry`** (per-pixel longitude, latitude, incidence, baseline on the output grid) and `metadata["reference_time_utc"]`, `["secondary_time_utc"]`, `["perpendicular_baseline_signed_mean_m"]`, `["incidence_range_deg"]`, `["topographic_phase_method"]` are new.
- **Still open (G16):** `result.profile["transform"]` is the native-resolution GCP-fitted affine and is not scaled for multilooked results; consumers that map pixels to coordinates from it are off by the look factor.

## Error handling in `process_pair` (28 Oct 2026) -- behaviour change
Before: nine places caught an exception and returned a weaker product that looked normal (orbit coregistration, raster collocation, burst ESD/deburst, topographic phase, flat-earth phase, AOI crop, burst-sync diagnostic, baseline, swath check). Now, following ISCE2/GAMMA/SNAP/HyP3 (stop when a requested step fails) and dolphin (validate inputs first):
- `InterferogramGenerator(on_error="raise")` is the default. A step that was requested or whose inputs were supplied and that fails raises a `ProcessingError` subclass (`InputError`, `CoregistrationError`, `BurstProcessingError`, `FlatEarthError`, `TopographicPhaseError`, `CropError`), all `ValueError`s, with the stage named and the cause chained. `on_error="degrade"` keeps the old fallbacks but grades the result `degraded`.
- A step that cannot run because an input was never supplied is recorded as skipped and the result is graded `quicklook`; only a pair with nothing skipped, flagged or failed is `science`. Everything is in `result.metadata["quality"]`.
- New hard limits on `process_pair`: `min_coreg_gcps`, `max_coreg_rms_px`, `max_burst_sync_ms`. Left as None they only report and flag.
- Different sub-swaths for the two SLCs, or an AOI crop that cannot be done, now raise.
- `build_stack` rejects `degraded` pairs always and `quicklook` pairs with `require_science_grade=True`.

## Closed
(none yet)
