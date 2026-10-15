# InSAR Processing

```bash
pip install "pygeofetch[insar]"
```

:::{tip}

Looking for a complete, real, cell-by-cell worked example rather than
an API reference? See
[Complete Worked Example: Mexico City Subsidence](insar-mexico-city-tutorial.md) — a full search-to-
validated-subsidence-map run, cross-referenced against a published
result (Cigna & Tapete 2021).
:::

:::{seealso}
Need to combine InSAR with optical data specifically? See
[Optical Pixel Offset Tracking](optical-offset-tracking.md)'s fusion
section, or the formalized, reusable
[InSAR + Optical Displacement Fusion pipeline](multi-sensor-pipelines.md#1-insar--optical-displacement-fusion).
:::
:::{note}

**Re-verified against a fresh source upload after this page and the
tutorial were originally written**: the InSAR module had substantial
internal changes since then (900+ diff lines in `interferogram.py`
alone; large diffs across nearly every file in `pygeofetch/insar/`).
Directly re-checked the real signatures of every function/class this
page and the tutorial document —
`search_and_select_consistent_stack`, `PreflightGate`,
`select_burst_synchronized_dates`, `SLCExtractor.extract_consistent_stack`,
`InterferogramGenerator.process_pair`, `PhaseUnwrapper.unwrap_pair`,
`build_sbas_network`, `select_reliable_reference_pixel`,
`bridge_unwrap_regions`, `SBASTimeSeries.invert`,
`RiskMapper.compute_risk` — against the fresh source. All matched
exactly; the large diffs were internal refactoring/implementation
changes, not breaking changes to the public API documented here.
:::
Coregistration, interferogram formation, phase unwrapping, and SBAS
time series inversion, in pure Python. No SNAP or ISCE required for
the core pipeline.

## The four-step chain

| Step | What it does |
|---|---|
| 1. SLC Extraction | Sub-swath extraction from `.SAFE` archives via embedded GCP matching |
| 2. Interferogram | Coregistration with ESD, formation, topographic phase removal |
| 3. Unwrapping | SNAPHU-based phase unwrapping via `snaphu-py` |
| 4. Time Series | SBAS inversion with reference-pixel normalization |

## End-to-end example

```python
from pygeofetch.insar import SLCExtractor, InterferogramGenerator, PhaseUnwrapper

extractor = SLCExtractor(polarisation="VV")
ref_tif, sec_tif = extractor.extract_pair(
    download_results[0], download_results[1],
    aoi=aoi, output_dir="./data",
)

gen = InterferogramGenerator()
result = gen.process_pair(ref_tif, sec_tif, dem="dem.tif")
print(f"Mean coherence: {result.coherence.mean():.3f}")

unwrapper = PhaseUnwrapper(cost_mode="defo")
unwrapped, conncomp = unwrapper.unwrap(result.interferogram, result.coherence)
```

## SLC extraction

Sentinel-1 IW SLC spans 3 sub-swaths per polarisation; your AOI
typically falls in just one, and which one varies per scene.
`SLCExtractor` reads each sub-swath's embedded GCPs via rasterio and
picks the one that overlaps your AOI.

Passing a `DownloadResult` object (rather than a raw path) is
preferred — `extract_pair()` reads `.output_path` directly, avoiding
filename/subfolder mismatch bugs.

## Interferogram generation

`InterferogramGenerator.process_pair()` handles GDAL's `complex_int16`
dtype — the actual format real Sentinel-1 SLC TIFFs use, not just
`complex64`/`complex128`. Missing this silently discards phase data.

Topographic phase removal is R²-gated (> 0.5) so genuine deformation
signal isn't mistaken for terrain phase.

## Phase unwrapping

SNAPHU (Chen & Zebker 2001) via `snaphu-py` — the same algorithm ASF's
On-Demand InSAR and ISCE2/3 use in production.

## SBAS time series

Berardino et al. (2002) SBAS inversion, with optional MintPy delegation
for the full correction chain.

:::{warning}

**The reference pixel matters more than almost anything else here.**
Phase unwrapping only recovers phase relative to an arbitrary
per-interferogram offset. Combining unwrapped interferograms without a
common, stable reference pixel corrupts the entire result. In one
verification run, referencing inside a synthetic subsidence bowl gave
103 mm/yr RMSE against a 100 mm/yr true signal; a verified-stable
reference gave 8.84 mm/yr RMSE. Always pass an explicit,
independently-verified `reference_pixel`.
:::
## Advanced safeguards: custom DEMs, layover/shadow, and topographic residuals

`pygeofetch.insar.advanced_safeguards` adds three real, general-purpose
capabilities — general-purpose meaning exactly that: the physics here
applies identically to a volcanic edifice, mountainous landslide
terrain, a high-relief urban area, or an open-pit mine. None of it is
mining-specific, even though this project's own validation work happens
to use a mine as its stress test.

These three functions belong to genuinely different stages of the real
InSAR chain, stated explicitly because conflating them is an easy
mistake:

### `prepare_custom_dem` — before interferogram generation

Aligns a custom, high-resolution DEM (UAV photogrammetry, airborne
LiDAR) to a reference SAR raster's exact grid via
`rasterio.warp.reproject`, for real topographic-phase removal during
interferogram formation. Useful when a coarse global DEM (SRTM,
Copernicus DEM) under-resolves real terrain — a UAV DEM over an active
volcanic crater, LiDAR over a landslide body, or a high-resolution
urban DSM.

```python
from pygeofetch.insar.advanced_safeguards import prepare_custom_dem

prepare_custom_dem(
    custom_dem_path="uav_dem.tif",
    reference_raster_path="reference_slc.tif",
    output_path="aligned_dem.tif",
)
```

### `generate_insar_mask` — before interferogram generation

Real, geometry-based layover/shadow masking from a DEM and the real
acquisition geometry (incidence angle, heading, look side) — flags
which pixels are structurally unusable for InSAR *before* any
interferogram is even formed, independent of coherence.

```python
from pygeofetch.insar.advanced_safeguards import generate_insar_mask

safe_mask = generate_insar_mask(
    dem=dem_array, pixel_size_m=10.0,
    incidence_angle_deg=35.0, heading_deg=193.0, look_side="right",
)
# True = geometrically safe for InSAR; False = layover or shadow —
# mask out, or fall back to optical offset tracking there instead.
```

The real physics: a pixel is in **layover** when local terrain slope
toward the radar is steep enough that the local incidence angle drops
to zero or below (the near-range part of the slope arrives at the
sensor before the far-range part, folding the image over itself), and
in **shadow** when slope facing away from the radar pushes the local
incidence angle to 90° or beyond (the radar beam physically can't reach
it). Verified against a synthetic cone DEM: a steep cone (76° slope vs.
35° incidence) correctly shows layover on the radar-facing flank *and*
shadow on the opposite flank simultaneously; a gentler cone (31° slope,
below the incidence angle) correctly shows neither.

### `build_sbas_design_matrix_with_topo` — during SBAS inversion, after unwrapping

This is real, standard residual DEM-error co-estimation (Berardino et
al. 2002; Fattahi & Amelung 2013), applied to the *already unwrapped*
phase stack — not topographic phase removal, which already happened
earlier using whatever DEM the interferogram stage used. This corrects
for the *residual* error left over in that earlier DEM, a real,
separate, later correction.

```python
from pygeofetch.insar.advanced_safeguards import build_sbas_design_matrix_with_topo

design_matrix, pair_order = build_sbas_design_matrix_with_topo(
    dates=all_dates,
    reference_date=reference_date,
    pair_dates=network_pairs,
    perpendicular_baselines=baseline_dict,   # {(date1, date2): B_perp_metres}
    wavelength_m=0.05546576,                  # Sentinel-1 C-band
    slant_range_m=850000.0,
    incidence_angle_deg=35.0,
)
```

Real physics: a residual DEM error `epsilon` contributes phase
`(4π/λ) × (B_perp / (R·sin θ)) × epsilon`, linear in `epsilon` exactly
like the familiar velocity term is linear in time — which is why both
can be solved for jointly in one real linear least-squares system per
pixel. The topographic-residual column added here has been checked
against this exact analytical formula to machine precision, not just
"does it run."

## Atmospheric correction

Elevation-correlated (no extra deps, same R²-gating), or ERA5-based via
PyAPS (`pygeofetch[insar-full]`, needs free CDS API credentials).

## Orbit files

```python
orbit_path = client.fetch_orbit_file(
    product_name=scene.properties.get("name", scene.id),
    orbit_type="precise",   # "precise" (21-day delay) or "restituted" (~3hr)
)
```

Served by ESA as `.EOF.zip`; extraction is automatic and always
returns a directly-usable `.EOF` path.

## Data validation

`DataValidator` runs automatically at real pipeline entry points, not
just available-but-unused: input SLC sanity checks (complex dtype,
NaN, dynamic range) at the start of every `process_pair()` call,
coherence range checks after estimation, and SBAS network connectivity
checks before the expensive inversion runs.

```python
from pygeofetch.insar import DataValidator

result = DataValidator.validate_slc(slc_array, name="reference SLC")
result.raise_if_invalid()  # a clear ValueError, not a downstream numerical artifact
```

Catches a real failure mode a naive dtype check misses: amplitude-only
data cast to a complex dtype (`real + 0j`) passes both the dtype check
and the amplitude-variation check — it's genuinely `complex64`, and
amplitude does vary. The validator specifically checks for near-zero
imaginary part everywhere, which real SAR phase never has.

## Real orbit-based coregistration

TOPS mode needs ~0.001-pixel coregistration accuracy (Yagüe-Martínez et
al. 2016) — orbit-based geometric coregistration is the proven method
for reaching it, not a shape-matching guess. Ground points are sampled
directly from a real DEM's own geographic coordinates (closed-form,
always converges) and located in both orbits via a real zero-Doppler
time solve.

```python
gen = InterferogramGenerator(esd_enabled=True, use_gpu=False)
result = gen.process_pair(
    reference="slc_ref.tif",
    secondary="slc_sec.tif",
    dem="dem.tif",
    reference_safe_zip="ref.SAFE.zip",
    secondary_safe_zip="sec.SAFE.zip",
    reference_orbit_file=fetch_orbit_file("S1A_..._ref"),
    secondary_orbit_file=fetch_orbit_file("S1A_..._sec"),
)
```

Supply all four (plus a DEM) and orbit-based coregistration is used
automatically. Omit any of the four and it falls back cleanly to
shape-based resampling, with a clear log line stating which path ran.

:::{note}

**Honest, documented limitation:** the lower-level
`solve_ground_point()` (an alternative, pixel-driven geolocation solve)
has a known reliability gap and is deliberately not exported as a
primary API. It always fails safely, but isn't recommended for
unattended use.
:::
## LOS-to-vertical conversion

InSAR measures line-of-sight range change — the true 3D displacement
vector projected onto the satellite's single viewing direction. With
one geometry, that's one equation and three unknowns.
`los_to_vertical_displacement()` applies the standard literature
technique (Fialko et al. 2001; Hooper et al. 2012): assume horizontal
motion is negligible.

```python
from pygeofetch.insar import los_to_vertical_displacement

vertical_velocity = los_to_vertical_displacement(
    ts_result.velocity, incidence_angle_deg=39.0,
)
```

This is an assumption, not a measurement — defensible for
vertically-dominated sources like mining/groundwater subsidence,
actively wrong for landslides or fault creep with a real lateral
component. Sentinel-1's near-polar orbit is also nearly blind to
north-south motion regardless of this conversion.

## Automatic visualization

```python
result.save("./output", auto_visualize=True)
# wrapped_phase.png, coherence.png, amplitude.png — alongside the GeoTIFFs
```

A visualization failure only logs a warning — it never blocks or loses
the actual GeoTIFF output.

## GPU acceleration

```python
gen = InterferogramGenerator(use_gpu=True)  # auto-detects; falls back to CPU cleanly
```

Optional CuPy backend for coherence estimation and SBAS inversion's
large matrix solve. `use_gpu` defaults to `False` — opt-in, not
opt-out.

## Risk mapping with real uncertainty quantification

```python
from pygeofetch.insar.analysis import RiskMapper

mapper = RiskMapper(ts_result)   # a TimeSeriesResult from SBASTimeSeries.invert()
risk_map = mapper.compute_risk(method="bayesian", confidence_level=0.95)

mapper.plot_risk_map(risk_map, output="risk_map.png")
mapper.export_geotiff("risk_map.tif")
mapper.export_uncertainty("risk_uncertainty.tif")

metrics = mapper.validate_risk_map(risk_map, validation_data=ground_truth_array)
# {"rmse": ..., "mae": ..., "r2": ..., "coverage": ..., "sharpness": ...}
```

Four real, distinct uncertainty-quantification methods, not one
formula presented four ways:

| `method=` | Real basis |
|---|---|
| `"bayesian"` (default) | Genuine conjugate Normal-Normal Bayesian update on the trend/slope |
| `"monte_carlo"` | Monte Carlo simulation, `n_simulations` draws |
| `"bootstrap"` | **Residual** bootstrap resampling — not naive time-index resampling, which would double-count temporal autocorrelation in the residuals |
| `"analytical"` | Closed-form analytical uncertainty propagation |

`risk_function` defaults to a trend-magnitude-over-variability ratio
but accepts any `(data_array, time_years) -> risk_array` callable, so
a domain-specific risk definition (e.g. weighting recent
acceleration more heavily) can be substituted directly.

:::{note}

**Real bug fixed**: an earlier version resolved the input time series
by *mutating the caller's own `ts_result` object* (setting new
`.data`/`.times` attributes on it as a side effect of just
constructing a `RiskMapper`) — a real risk of silently corrupting the
caller's own code if that same `ts_result` object was reused
elsewhere afterward. `RiskMapper.__init__` no longer mutates its
input.
:::
`RiskMapper` accepts a `TimeSeriesResult` from `SBASTimeSeries.invert()`
directly (matches its `.displacement`/`.dates` attributes), or any
object/dict exposing a 3D `(time, y, x)` array under one of
`data`/`displacement`/`deformation`/`timeseries`/`stack` and a
matching per-time-step label under one of
`times`/`dates`/`time`/`date`/`acquisition_dates`/`date_list`.

`validate_risk_map()` compares a computed risk map against real
ground truth or reference data — `rmse`, `mae`, `r2` for accuracy,
plus `coverage` and `sharpness` for calibration quality (does the
stated confidence interval actually contain the true value the stated
fraction of the time, and how tight is it).

## InSARProject — search to interferogram in a handful of calls

A high-level workflow wrapper for the full search → download → extract
→ interferogram chain, built on the verified lower-level pieces above.

```python
from pygeofetch.insar import InSARProject
from pygeofetch.models import BoundingBox

project = InSARProject(
    name="my_aoi",
    aoi=BoundingBox(min_lon=-99.183, max_lon=-99.003, min_lat=19.278, max_lat=19.438),
    output_dir="data/my_aoi_insar",
)
project.search(start_date="2024-11-01", end_date="2025-01-15")
project.download_and_extract(max_scenes=6)
project.form_all_interferograms()
project.summary()
```

If you want every burst-aware ESD, flat-earth, and unwrapping step
fully explicit instead of wrapped inside this convenience layer, use
`SLCExtractor` / `InterferogramGenerator` directly.
