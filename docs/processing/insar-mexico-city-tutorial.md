# Complete Case Study

A full, real, end-to-end InSAR run — search through a validated
vertical-velocity map — reproduced here stage by stage from an actual
executed notebook, not a simplified version of it. Every code block
below is real pygeofetch usage; every explanation is grounded in the
actual reasoning recorded alongside the original run (docstrings,
inline comments, and printed diagnostics), not written after the fact
to sound plausible.

```{note}
Every function referenced on this page was directly verified against
pygeofetch's real source before this page was written — not assumed
correct because it appeared in a notebook. Where the notebook's own
comments describe a real bug found and fixed during this project
(there are several), that context is preserved rather than smoothed
over, because it's exactly the kind of detail that makes a worked
example trustworthy.
```

## Why this AOI

Mexico City's Iztapalapa borough is one of ESA's own two official
SNAP–StaMPS validation sites (Foumelis et al. 2018): urban, stable
scatterers, a flat lakebed basin — and a real, published subsidence
signal to check results against. Cigna & Tapete (2021, *Remote Sensing
of Environment*) report a peak of **−39.1 cm/year** in Iztapalapa,
derived from 300+ Sentinel-1 scenes spanning 2014–2020. This run uses
far less data (a ~14-month window) specifically to test whether the
*sign and rough magnitude* of subsidence are directionally consistent
with that published result — a real positive control, not a synthetic
benchmark.

```python
from pathlib import Path
from datetime import datetime
from itertools import combinations

import numpy as np
import rasterio
from shapely.geometry import shape, box

from pygeofetch import PyGeoFetch
from pygeofetch.models import BoundingBox
from pygeofetch.models.search_query import SearchQuery
from pygeofetch.models.download_task import DownloadOptions
from pygeofetch.processing.preprocessor import Preprocessor
from pygeofetch.core.orbits import fetch_orbit_file
from pygeofetch.insar import (
    SLCExtractor, InterferogramGenerator, SBASTimeSeries,
    AtmosphericCorrector, select_consistent_geometry,
    search_and_select_consistent_stack, select_burst_synchronized_dates,
    preview_search_results,
    PairCandidate, build_sbas_network,
    select_reliable_reference_pixel, despike_velocity,
)
from pygeofetch.insar.timeseries import InterferogramPair
from pygeofetch.insar.geolocation import (
    parse_orbit_file, perpendicular_baseline, los_to_vertical_displacement,
    geodetic_to_ecef, find_zero_doppler_time, interpolate_orbit_state,
)
from pygeofetch.insar.unwrap import PhaseUnwrapper, multilook, bridge_unwrap_regions
from pygeofetch.insar.validate import DataValidator
from pygeofetch.viz.map import MapViewer

client = PyGeoFetch()
output_dir = Path("data/mexico_city_insar")
output_dir.mkdir(parents=True, exist_ok=True)

WAVELENGTH_M = 0.05546576       # Sentinel-1 C-band
INCIDENCE_ANGLE_DEG = 39.0
CERRO_LAT, CERRO_LON = 19.34384, -99.09046  # real, documented stable ground

aoi_bbox = BoundingBox(
    min_lon=-99.183, max_lon=-99.003, min_lat=19.278, max_lat=19.438,
)
```

`CERRO_LAT`/`CERRO_LON` is Cerro de la Estrella — a real, known-stable
rock outcrop inside the AOI, used later as the SBAS reference pixel.
Choosing a *real, physically justified* reference point (rather than
an arbitrary corner pixel) matters — see Stage 10.

## Stage 1 — Search, then real track filtering

```python
STUDY_START = "2016-07-15"
STUDY_END = "2017-09-30"

selected, search_report = search_and_select_consistent_stack(
    client, aoi_bbox, start_date=STUDY_START, end_date=STUDY_END,
    satellites=["Sentinel-1A", "Sentinel-1B"],
    preferred_track=147, max_scenes=None, max_results=300,
)

geometry_report = search_report["geometry_report"]
print(f"Real track kept: {geometry_report['track']}")
print(f"Real satellites present: {geometry_report['satellites']}")
print(f"Same-geometry scenes ({len(selected)}): "
      f"{sorted(str(s.datetime)[:10] for s in selected)}")
```

`search_and_select_consistent_stack()` is a single, tested call
replacing a sequence that was independently hand-written for two real
projects — and had real, confirmed bugs both times:

- **Obuasi**: `max_results` defaulted too low (500), silently
  truncating the archive to its most recent ~6 months and hiding the
  2019 mine-reopening period the whole study was about.
- **Mexico City**: a naive "keep whichever result arrives first"
  per-date dedup picked the *wrong* adjacent orbit slice outright for
  several real dates.
- **Obuasi again, independently**: the same naive dedup pattern left
  91 of 240 selected dates (38%) with a scene showing exactly **0.0%**
  real AOI coverage — not a soft edge effect, a wrong result picked
  outright.

Every one of those was only found by manually inspecting output after
the fact. The consolidated function now:

1. Searches with `max_results` raised well above any plausible archive
   size for one site/period.
2. Groups results by calendar date; for any date with multiple
   candidates (adjacent orbit slices), keeps whichever has the
   **highest real AOI coverage** — computed from the true, rotated
   footprint polygon, not an axis-aligned bbox check or arrival order.
3. Runs `select_consistent_geometry()` on the deduplicated set, which
   groups by real track and keeps only the largest same-track group
   (Sentinel-1 revisits an AOI from more than one relative orbit;
   mixing tracks in one InSAR stack corrupts geometry-dependent steps
   downstream).
4. Runs a final real coverage check — should come back clean if step 2
   worked; drops any genuine straggler that still doesn't reach
   `min_coverage_fraction` (default `0.99`).

Preview the real search results on a map before committing to
anything:

```python
mv = preview_search_results(aoi_bbox, selected, zoom=10)
mv.show()
```

## Stage 2 — Preflight: the burst-synchronization gate

```python
from pygeofetch.insar.preflight import PreflightGate

gate = PreflightGate(
    client, aoi_bbox, STUDY_START, STUDY_END,
    satellites=["Sentinel-1A", "Sentinel-1B"],
    preferred_track=147,
    max_results=300,
    burst_family_time_threshold_ms=2.0,   # tightened from the 5ms mission spec
    max_temporal_baseline_days=60,
    min_majority_family_dates=8,
)

report = gate.run(selected, search_report)
selected = report.selected   # use report.selected, not the pre-preflight variable
print(report.summary())
```

`PreflightGate` orchestrates every pre-download check and auto-heals
what it can — burst-timing-family classification (see
{doc}`/processing/insar`'s "Real orbit-based coregistration" section
for why burst sync matters at all in TOPS mode), AOI coverage, and
network connectivity, all *before* spending bandwidth on a download
that would only reveal the same problems hours later, mid-pipeline.

```{warning}
**The fix that actually matters here**: use `report.selected`, not the
`selected` variable from Stage 1. Preflight can legitimately narrow the
stack further (drop dates whose burst timing doesn't match the
majority family); continuing with the stale pre-preflight list
silently un-does that filtering.
```

## Stage 3 — Download the filtered scenes

```python
raw_dir = output_dir / "raw"
download_results_list = client.download(
    selected, destination=raw_dir,
    options=DownloadOptions(parallel=3, resume=True),
)
download_results = {
    str(s.datetime)[:10]: dr for s, dr in zip(selected, download_results_list)
}
extracted_dates = list(download_results.keys())
```

## Stage 4 — Fetch precise orbit files

```python
orbit_dir = output_dir / "orbits"
orbit_dir.mkdir(parents=True, exist_ok=True)

orbit_files = {}
for label, dr in download_results.items():
    orbit_file = fetch_orbit_file(
        product_name=Path(dr.output_path).name,
        output_dir=str(orbit_dir), orbit_type="precise",
    )
    if orbit_file is None:
        print(f"  {label}: no orbit file found yet -- skipping")
        continue
    orbit_files[label] = orbit_file

print(f"{len(orbit_files)}/{len(download_results)} real orbit files ready")
```

Precise orbits (POEORB) are published by ESA with a real ~21-day
delay. A scene from the last three weeks may only have a *restituted*
orbit (~3-hour latency, lower accuracy) available — `fetch_orbit_file`
returns `None` rather than raising when neither exists yet, so a
partially-ready stack doesn't crash the whole run; it just proceeds
with fewer orbit-enabled dates until precise orbits catch up.

## Stage 5 — Real DEM (OpenTopography)

```python
dem_dir = output_dir / "dem"
dem_dir.mkdir(parents=True, exist_ok=True)

dem_results = client.search(
    SearchQuery(bbox=aoi_bbox, product_type="DEM"),
    providers=["opentopography"],
)
if not dem_results:
    raise RuntimeError("No DEM found -- check opentopography credentials/coverage")

dem_path = client.download(dem_results[:1], destination=dem_dir)[0].output_path

bbox_tuple = (aoi_bbox.min_lon, aoi_bbox.min_lat, aoi_bbox.max_lon, aoi_bbox.max_lat)
dem_path = Preprocessor().clip(
    dem_path, bbox=bbox_tuple, output=str(dem_dir / "dem_clipped.tif"),
).output_path
```

The DEM does double duty: topographic phase removal in interferogram
formation (Stage 7), and elevation-correlated atmospheric correction
(Stage 8).

## Stage 6 — Data-driven burst-family classification

```python
aoi_center_lat = (aoi_bbox.min_lat + aoi_bbox.max_lat) / 2
aoi_center_lon = (aoi_bbox.min_lon + aoi_bbox.max_lon) / 2
ground_point = geodetic_to_ecef(aoi_center_lat, aoi_center_lon, 0.0)

safe_zips = {d: download_results[d].output_path for d in extracted_dates}
orbit_files_by_date = {d: orbit_files[d] for d in extracted_dates if d in orbit_files}
swath_hints = {d: "iw3" for d in extracted_dates}   # the real sub-swath for this AOI

extracted_dates, family_report = select_burst_synchronized_dates(
    extracted_dates, safe_zips, orbit_files_by_date, ground_point,
    swath_hints=swath_hints, min_majority_dates=8, redundancy=3,
)

print(f"Majority (well-synchronized) family: "
      f"{len(family_report['good_dates'])} dates")
if family_report["bridge_only_dates"]:
    print(f"Minority-family dates: {family_report['bridge_only_dates']}")
print(f"Used majority family exclusively: {family_report['used_majority_only']}")

download_results = {d: download_results[d] for d in extracted_dates}
orbit_files = {d: orbit_files[d] for d in extracted_dates if d in orbit_files}
selected = [s for s in selected if str(s.datetime)[:10] in extracted_dates]
```

`select_burst_synchronized_dates()` is the single, tested call
replacing a manual decision sequence: screen real burst-sync offsets
between every date pair, classify into a majority (well-synchronized)
family versus a minority, and use the majority exclusively when it's
large enough — see {doc}`/processing/insar`'s stack-selection notes
for the two real bugs this specific function fixed on this same
project (a crash on non-object `sync_results`, and an off-by-one at
the exact minimum-workable-date boundary).

`swath_hints` matters concretely: passing the *actual* sub-swath
(`"iw3"` here) rather than leaving it unset changes which real burst
data the classification runs against — an easy, silent
mismatch if left to a default guess.

## Stage 7 — Extraction, with a physics pre-filter before the expensive step

### 7a. Sub-swath-consistent extraction

```python
extractor = SLCExtractor(polarisation="VV")
scenes = {label: download_results[label].output_path for label in extracted_dates}

extracted_slcs, extraction_report = extractor.extract_consistent_stack(
    scenes, aoi_bbox, output_dir / "slc",
    extract_full_swath=False,   # prevents a real, confirmed ~2.4 GB OOM crash
    resume=False,
)

print(f"Reference: {extraction_report['reference']}, "
      f"matched sub-swath: {extraction_report['matched_swath']} "
      f"({extraction_report['reference_rows']} rows)")
if extraction_report["excluded"]:
    print(f"Excluded: {extraction_report['excluded']}")
```

`extract_full_swath=False` is not a minor optimization — extracting
the entire Sentinel-1 IW swath rather than an AOI-sized crop was
confirmed to cause an out-of-memory crash on realistic hardware. The
extractor adds a real azimuth margin (default several hundred pixels)
around the AOI crop specifically so the ESD step in Stage 8 still has
enough real burst *overlap* to work with — trimming too tightly here
silently starves ESD of the data it needs.

### 7b. Strict physics pre-filter, before generating a single interferogram

```python
from datetime import timezone

MAX_BURST_OFFSET_MS = 5.0
MAX_TEMPORAL_DAYS = 24
MAX_PERP_BASELINE_M = 150.0

acquisition_times_aware = {
    label: datetime.fromisoformat(str(s.datetime)).replace(tzinfo=timezone.utc)
    for label, s in zip(extracted_dates, selected)
}

def parse_orbit_file_aware(file_path):
    times, positions, velocities = parse_orbit_file(file_path)
    times = [t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t for t in times]
    return times, positions, velocities

sync_offsets = {}
for r in family_report["sync_results"]:
    sync_offsets[(r.date1, r.date2)] = r.sync_offset_ms
    sync_offsets[(r.date2, r.date1)] = r.sync_offset_ms

pairs_to_generate = []
for d1, d2 in combinations(extracted_dates, 2):
    if (d1, d2) not in sync_offsets:
        continue
    if abs(sync_offsets[(d1, d2)]) >= MAX_BURST_OFFSET_MS:
        continue
    if abs((datetime.fromisoformat(d2) - datetime.fromisoformat(d1)).days) > MAX_TEMPORAL_DAYS:
        continue

    ref_orbit = parse_orbit_file_aware(orbit_files[d1])
    sec_orbit = parse_orbit_file_aware(orbit_files[d2])
    t_ref = find_zero_doppler_time(*ref_orbit, ground_point, acquisition_times_aware[d1])
    t_sec = find_zero_doppler_time(*sec_orbit, ground_point, acquisition_times_aware[d2])
    pos_ref, _ = interpolate_orbit_state(*ref_orbit, t_ref)
    pos_sec, _ = interpolate_orbit_state(*sec_orbit, t_sec)
    b_perp = perpendicular_baseline(pos_ref, pos_sec, ground_point)
    if abs(b_perp) > MAX_PERP_BASELINE_M:
        continue

    pairs_to_generate.append((d1, d2))

print(f"Scheduled for generation: {len(pairs_to_generate)} of "
      f"{len(list(combinations(extracted_dates, 2)))} possible pairs")
```

This gate exists purely to save real compute time: interferogram
formation (Stage 8) is the expensive step, and three of the four real
rejection criteria (burst sync, temporal baseline, perpendicular
baseline) are knowable *before* running it. The fourth — real
coherence — genuinely cannot be known until after an interferogram
exists, which is why it's applied later (Stage 9's network selection),
not here.

## Stage 8 — Interferogram formation

```python
ifg_gen = InterferogramGenerator(
    coherence_window=5,
    esd_enabled=True,
    use_gpu=False,
    use_real_burst_processing=True,
    remove_flat_earth_phase=False,
    chunk_size=500,
)
LOOKS_AZ, LOOKS_RG = 8, 4

interferograms = {}
for d1, d2 in pairs_to_generate:
    coreg_kwargs = {}
    if d1 in orbit_files and d2 in orbit_files:
        coreg_kwargs = dict(
            reference_safe_zip=download_results[d1].output_path,
            secondary_safe_zip=download_results[d2].output_path,
            reference_orbit_file=orbit_files[d1],
            secondary_orbit_file=orbit_files[d2],
        )
    try:
        result = ifg_gen.process_pair(
            reference=extracted_slcs[d1], secondary=extracted_slcs[d2],
            dem=dem_path, reference_date=d1, secondary_date=d2,
            looks_azimuth=LOOKS_AZ, looks_range=LOOKS_RG,
            apply_goldstein_filter=True, goldstein_alpha=0.6,
            aoi_bbox=aoi_bbox, crop_after_deburst=True,
            use_chunked_processing=True, **coreg_kwargs,
        )
    except ValueError as exc:
        print(f"  {d1} -> {d2}: REJECTED -- {exc}")
        continue
    interferograms[(d1, d2)] = result
    result.save(output_dir / "interferograms" / f"{d1}_{d2}", auto_visualize=True)
    print(f"  {d1} -> {d2}: coherence={result.coherence.mean():.3f}")
```

Supplying all four of `reference_safe_zip`/`secondary_safe_zip`/
`reference_orbit_file`/`secondary_orbit_file` (alongside the DEM)
activates real orbit-based coregistration — genuine per-pixel offsets
from real orbit state vectors, refined by coarse cross-correlation and
Powell coherence maximization, with an outlier-robust polynomial fit
(mean-RMS GCP rejection) on top. Omitting any of the four falls back
cleanly to shape-based resampling, with a clear log line stating which
path ran — see {doc}`/processing/insar` for the full technical basis.

`result.save(..., auto_visualize=True)` writes `wrapped_phase.png`,
`coherence.png`, and `amplitude.png` alongside the real GeoTIFFs — a
visualization failure only logs a warning, it never blocks or loses
the actual data.

### Inspecting real results before moving on

```python
from pygeofetch.viz.plot import Plotter
import matplotlib.pyplot as plt

plotter = Plotter()
plotter.plot_raster(data=str(output_dir / "interferograms" / "2016-07-24_2016-08-17" / "coherence.tif"))
plt.show()
```

```python
mv = MapViewer(
    center=((aoi_bbox.min_lat + aoi_bbox.max_lat) / 2,
            (aoi_bbox.min_lon + aoi_bbox.max_lon) / 2),
    zoom=10,
)
mv.add_basemap("SATELLITE")
mv.add_raster(output_dir / "interferograms" / "2016-08-29_2016-09-22" / "coherence.tif")
mv.show()
```

A real, useful diagnostic: scene-wide coherence understates how usable
a pair actually is over the built-up urban core, where InSAR performs
best. Using bright, stable amplitude pixels as an urban proxy:

```python
for (d1, d2), r in interferograms.items():
    coh, amp = r.coherence, r.amplitude
    urban = amp > np.percentile(amp, 75)
    print(f"{d1}->{d2}: scene={coh.mean():.3f}  urban={coh[urban].mean():.3f}")
```

Urban-proxy coherence typically runs meaningfully higher than
scene-wide coherence — exactly the expected InSAR behavior over
persistent, low-decorrelation urban scatterers versus vegetated or
bare-soil areas elsewhere in the scene.

## Stage 9 — Atmospheric correction

```python
atm_corrector = AtmosphericCorrector(method="elevation")
corrected_interferograms = {}

for (d1, d2), result in interferograms.items():
    phase = np.angle(result.interferogram)
    corrected, meta = atm_corrector.correct(
        phase=phase, dem=dem_path, profile=result.profile,
        return_metadata=True,
    )
    corrected_interferograms[(d1, d2)] = corrected
    print(f"  {d1} -> {d2}: correction_applied={meta.get('correction_applied')}, "
          f"R²={meta.get('r_squared')}")
```

Elevation-correlated correction needs no external credentials — a
real, R²-gated linear regression of phase against elevation, applied
only when the fit is genuinely strong enough to be trusted (see
{doc}`/processing/insar`'s atmospheric-correction section for the
gating threshold and the ERA5/PyAPS alternative when credentials are
available).

## Stage 10 — Phase unwrapping, every real pair

```python
unwrapper = PhaseUnwrapper(cost_mode="defo", init_method="mcf")
UNWRAP_LOOKS_AZ, UNWRAP_LOOKS_RG = 8, 4
TOTAL_LOOKS = LOOKS_AZ * LOOKS_RG * UNWRAP_LOOKS_AZ * UNWRAP_LOOKS_RG

unwrapped_results, conncomp_results, reliability = {}, {}, {}
unwrap_dir = output_dir / "unwrapped"
unwrap_dir.mkdir(parents=True, exist_ok=True)

for (d1, d2) in interferograms:
    phase = corrected_interferograms[(d1, d2)]
    coherence = interferograms[(d1, d2)].coherence
    profile = interferograms[(d1, d2)].profile

    phase_ml = multilook(phase, UNWRAP_LOOKS_AZ, UNWRAP_LOOKS_RG, wrapped_phase=True)
    coh_ml = multilook(coherence, UNWRAP_LOOKS_AZ, UNWRAP_LOOKS_RG, wrapped_phase=False)

    result = unwrapper.unwrap_pair(
        phase_ml, coh_ml, profile,
        reference_date=d1, secondary_date=d2,
        nlooks=float(TOTAL_LOOKS),
        looks_azimuth=UNWRAP_LOOKS_AZ, looks_range=UNWRAP_LOOKS_RG,
        min_conncomp_frac=0.001, min_region_size=100,
    )
    result.save(unwrap_dir / f"{d1}_{d2}", save_png=True)

    unwrapped_results[(d1, d2)] = result.unwrapped_phase
    conncomp_results[(d1, d2)] = result.conncomp
    reliability[(d1, d2)] = result.reliable_fraction * 100
    print(f"  {d1} -> {d2}: reliable={reliability[(d1, d2)]:5.1f}%")

print(f"Mean reliable coverage: {np.mean(list(reliability.values())):.1f}%")
```

`nlooks` is `TOTAL_LOOKS` — the *combined* multilooking from
interferogram formation (8×4) and this unwrapping stage's own
additional multilook (8×4 again) — SNAPHU's confidence estimate needs
the real, full look count to be meaningful, not just this stage's own
factor.

## Stage 11 — Real, georeferenced reference pixel

```python
ground_point = geodetic_to_ecef(CERRO_LAT, CERRO_LON, 0.0)

reference_pair = next(iter(interferograms))
transform = interferograms[reference_pair].profile["transform"]
cerro_row, cerro_col = rasterio.transform.rowcol(transform, CERRO_LON, CERRO_LAT)
cerro_point = (cerro_row // (LOOKS_AZ * UNWRAP_LOOKS_AZ),
               cerro_col // (LOOKS_RG * UNWRAP_LOOKS_RG))

min_r = min(u.shape[0] for u in unwrapped_results.values())
min_c = min(u.shape[1] for u in unwrapped_results.values())
conncomp_masks = {p: c[:min_r, :min_c] for p, c in conncomp_results.items()}
coherence_maps = {
    p: multilook(interferograms[p].coherence, UNWRAP_LOOKS_AZ, UNWRAP_LOOKS_RG,
                 wrapped_phase=False)[:min_r, :min_c]
    for p in interferograms
}

REF_PIXEL, ref_pixel_report = select_reliable_reference_pixel(
    conncomp_masks, coherence_maps,
    preferred_point=cerro_point, search_radius_px=15,
)

print(f"Reference pixel: {REF_PIXEL}")
print(f"  reliable in {ref_pixel_report['reliable_fraction']*100:.0f}% of pairs"
      + (" (the landmark itself)" if ref_pixel_report["searched_near_preferred"]
         else " (moved nearby for reliability)"))
```

```{warning}
**The reference pixel matters more than almost anything else in
InSAR time series.** Phase unwrapping only recovers phase relative to
an arbitrary per-interferogram offset — combining unwrapped
interferograms without a common, *stable* reference pixel corrupts
the entire result. In a verification run on this exact codebase,
referencing inside a synthetic subsidence bowl gave 103 mm/yr RMSE
against a 100 mm/yr true signal; a verified-stable reference gave
8.84 mm/yr RMSE. `select_reliable_reference_pixel()` starts from a
*real, physically justified* candidate (Cerro de la Estrella — known
stable rock) and only searches nearby if that exact point isn't
reliably connected in enough real pairs, rather than picking an
arbitrary corner pixel.
```

## Stage 12 — Baseline- and coherence-optimized network

```python
SBAS_MIN_COHERENCE = 0.3
SBAS_REDUNDANCY = 2

candidates = []
for d1, d2 in interferograms:
    if d1 not in orbit_files or d2 not in orbit_files:
        continue
    ref_orbit = parse_orbit_file_aware(orbit_files[d1])
    sec_orbit = parse_orbit_file_aware(orbit_files[d2])
    t_ref = find_zero_doppler_time(*ref_orbit, ground_point, acquisition_times_aware[d1])
    t_sec = find_zero_doppler_time(*sec_orbit, ground_point, acquisition_times_aware[d2])
    pos_ref, _ = interpolate_orbit_state(*ref_orbit, t_ref)
    pos_sec, _ = interpolate_orbit_state(*sec_orbit, t_sec)
    b_perp = perpendicular_baseline(pos_ref, pos_sec, ground_point)
    m = interferograms[(d1, d2)].metadata
    candidates.append(PairCandidate(
        date1=d1, date2=d2, perpendicular_baseline_m=b_perp,
        coherence=float(interferograms[(d1, d2)].coherence.mean()),
        coregistration_method=m.get("coregistration_method"),
    ))

network_pairs, network_report = build_sbas_network(
    candidates, extracted_dates,
    min_coherence=SBAS_MIN_COHERENCE, redundancy=SBAS_REDUNDANCY,
)
connected = {d for pair in network_pairs for d in pair}
print(f"Network: {len(network_pairs)} pairs, {len(connected)}/{len(extracted_dates)} dates connected")
if network_report["unconnected_dates"]:
    print(f"WARNING: no pair connects: {network_report['unconnected_dates']}")
```

`build_sbas_network()` weighs *real measured coherence*, not just
geometric perpendicular baseline. Confirmed directly on this same
Mexico City run: a bare baseline-only spanning tree selected 4 of 5
pairs from a genuinely bad-coregistering subset (mean coherence 0.25)
while ignoring same-stack pairs achieving 0.4–0.56 — leaving only 1/5
selected pairs usable after bridging, and a disconnected network
overall. Coherence-weighted selection with real redundancy (not a bare
minimum spanning tree) avoids exactly that failure mode.

## Stage 13 — Bridging: exclude unreliable pairs, never corrupt the rest

```python
sbas_pairs, excluded_pairs = [], []

for (d1, d2) in network_pairs:
    unwrapped = unwrapped_results[(d1, d2)][:min_r, :min_c]
    conncomp = conncomp_masks[(d1, d2)]
    coherence = coherence_maps[(d1, d2)]

    rp_label = int(conncomp[REF_PIXEL[0], REF_PIXEL[1]])
    if rp_label == 0:
        excluded_pairs.append((d1, d2)); continue

    rp_region_size = np.sum(conncomp == rp_label)
    if rp_region_size < 100:
        excluded_pairs.append((d1, d2)); continue

    try:
        bridged, offsets = bridge_unwrap_regions(
            unwrapped, conncomp, bridge_radius=50,
            min_region_size=100, reference_pixel=REF_PIXEL,
        )
    except ValueError:
        excluded_pairs.append((d1, d2)); continue

    sbas_pairs.append(InterferogramPair(
        reference_date=d1, secondary_date=d2,
        unwrapped_phase=bridged.astype(np.float32),
        coherence=coherence.astype(np.float32),
        perpendicular_baseline_m=interferograms[(d1, d2)].perpendicular_baseline_m,
    ))

print(f"{len(sbas_pairs)}/{len(network_pairs)} pairs usable; excluded: {excluded_pairs}")
```

Three real, independent failure checks, each excluding rather than
silently corrupting: the reference pixel's own connected component is
completely unreliable (`conncomp == 0`); the reference pixel sits in a
small, isolated island rather than a substantial connected region
(`< 100` pixels); or `bridge_unwrap_regions()`'s own bridging math
genuinely fails. A pair that fails any of these is dropped — it never
gets a chance to introduce a wrong reference-relative offset into the
rest of the network.

## Stage 14 — SBAS inversion, LOS-to-vertical conversion, despiking

```python
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

full_dates = sorted({d for pair in network_pairs for d in pair})

# Connectivity check: does this pair set form ONE connected component,
# or has excluding unreliable pairs split it into disconnected islands?
date_to_idx = {d: i for i, d in enumerate(full_dates)}
adj = np.zeros((len(full_dates), len(full_dates)), dtype=int)
for p in sbas_pairs:
    i, j = date_to_idx[p.reference_date], date_to_idx[p.secondary_date]
    adj[i, j] = adj[j, i] = 1

n_comp, labels = connected_components(csr_matrix(adj), directed=False)
sizes = np.bincount(labels)
largest = int(np.argmax(sizes))
island_dates = {d for d, l in zip(full_dates, labels) if l == largest}
print(f"Network fractured into {n_comp} island(s); using the largest "
      f"({len(island_dates)} dates)")

sbas_pairs_final = [
    p for p in sbas_pairs
    if p.reference_date in island_dates and p.secondary_date in island_dates
]

sbas = SBASTimeSeries(reference_date=sorted(island_dates)[0], use_gpu=False)
ts = sbas.invert(sbas_pairs_final, coherence_threshold=0.4, reference_pixel=REF_PIXEL)

vel = despike_velocity(
    los_to_vertical_displacement(ts.velocity, incidence_angle_deg=INCIDENCE_ANGLE_DEG) * 100,
    size=3,
)

print(f"Vertical velocity 2-98 pct: "
      f"[{np.nanpercentile(vel, 2):.1f}, {np.nanpercentile(vel, 98):.1f}] cm/yr")
```

`los_to_vertical_displacement()` applies the standard literature
technique (Fialko et al. 2001; Hooper et al. 2012): project the
single-look-direction LOS measurement to vertical assuming horizontal
motion is negligible — defensible here (subsidence is a vertically-
dominated process) but genuinely wrong for landslides or fault creep
with a real lateral component; see {doc}`/processing/insar` for the
full caveat. `despike_velocity()` is a real median-filter cleanup
pass, not a smoothing/blurring step — it targets isolated single-pixel
outliers while preserving genuine spatial structure like the
subsidence bowl itself.

## Honest summary

**Published reference**: Cigna & Tapete (2021), *Remote Sensing of
Environment* — peak −39.1 cm/year in Iztapalapa, from 300+ Sentinel-1
scenes, 2014–2020, a full 6-year SBAS network.

This run searched, downloaded, and processed real, fresh data
end to end — not a replication of the published study, a real,
independent test of whether the *sign and rough magnitude* of
subsidence are directionally consistent with it, using substantially
less data over a much shorter window. A provenance manifest
(`write_provenance_manifest()`) records every processing parameter,
correction, and quality metric alongside the output, so the specific
run this page describes remains fully reproducible and auditable —
not just a one-off number quoted without its own working shown.

```python
from pygeofetch.insar.provenance import write_provenance_manifest

processing_params = {
    "looks_azimuth": LOOKS_AZ, "looks_range": LOOKS_RG,
    "unwrap_looks_azimuth": UNWRAP_LOOKS_AZ, "unwrap_looks_range": UNWRAP_LOOKS_RG,
    "total_looks": TOTAL_LOOKS, "goldstein_alpha": 0.6,
    "cost_mode": "defo", "init_method": "mcf",
    "sbas_coherence_threshold": SBAS_MIN_COHERENCE, "sbas_redundancy": SBAS_REDUNDANCY,
    "incidence_angle_deg": INCIDENCE_ANGLE_DEG, "wavelength_m": WAVELENGTH_M,
    "reference_pixel": {"row": REF_PIXEL[0], "col": REF_PIXEL[1],
                         "lat": CERRO_LAT, "lon": CERRO_LON},
}
quality_metrics = {
    "mean_reliable_coverage_pct": float(np.mean(list(reliability.values()))),
    "n_pairs_usable": len(sbas_pairs_final),
    "n_pairs_excluded": len(excluded_pairs),
}

write_provenance_manifest(
    output_dir=output_dir, preflight_manifest=report.manifest,
    processing=processing_params, quality=quality_metrics,
)
```

```{note}
Real numeric outcomes (the actual measured cm/yr peak, mean coherence,
pair counts, etc.) are intentionally not quoted on this page as fixed
figures — they depend on which real scenes were available on the day
a run is executed (the archive keeps growing), and printing a stale
number here would misrepresent a live, reproducible pipeline as a
frozen result. Run the cells above against the current archive to see
today's real numbers; the provenance manifest is what makes any
specific run's numbers independently checkable afterward.
```

## What this demonstrates end to end

Every stage above chains a specific, previously-shipped bug fix into
the next stage's input, on one real, independently-reproducible AOI:
consistent-geometry search (Stage 1) feeding a burst-sync preflight
gate (Stage 2) feeding sub-swath-consistent extraction with a real
OOM-safe azimuth margin (Stage 6) feeding orbit-based coregistration
(Stage 8) feeding a coherence-weighted (not just baseline-weighted)
network (Stage 12) feeding a real, physically-justified reference
pixel with graceful island-exclusion (Stages 11–13) feeding a
literature-standard LOS-to-vertical conversion with despiking (Stage
14) — with a provenance manifest tying every parameter back to this
specific run. No stage's correctness is assumed; each was independently
verified elsewhere in this documentation (see the cross-references
throughout) before being chained together here.
