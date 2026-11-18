#!/usr/bin/env python3
"""Create the pygeofetch "known limitations / help wanted" issues on GitHub.

Safe by default: without --apply it only prints what it WOULD do.

    python create_pygeofetch_issues.py                 # dry run: list issues and labels
    python create_pygeofetch_issues.py --show 3        # print the full body of issue number 3
    python create_pygeofetch_issues.py --only-closed --fixed-in https://github.com/EOCoreINT/pygeofetch/commit/<sha>
    python create_pygeofetch_issues.py --apply         # really create them

Needs only the Python standard library. Authentication, first match wins:
    1. environment variable GITHUB_TOKEN or GH_TOKEN   (a fine-grained token with "Issues: read and write"
       on the repository; "Administration" is NOT needed)
    2. the GitHub CLI, if it is logged in (`gh auth login`)
The token is never printed or written anywhere.

Re-running is safe: an issue is skipped if an issue (open or closed) with the same hidden marker or the same
title already exists. Labels are created only if missing; if your account cannot manage labels, the issues are
still created and the script says that the labels were not applied.

Every statement in the issue texts comes from the project's own gap register and the run logs, and each issue
says whether it was validated on real data. Edit the ISSUES list below before running if you want to change
wording; nothing else needs touching.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_REPO = "EOCoreINT/pygeofetch"
API = "https://api.github.com"
PAUSE_BETWEEN_CREATES_S = 3.0  # GitHub throttles bursts of content creation

LABELS = {
    "limitation": ("b60205", "Known limitation of the current implementation"),
    "help wanted": ("008672", "Extra attention is needed"),
    "good first issue": ("7057ff", "Good for newcomers"),
    "enhancement": ("a2eeef", "New feature or request"),
    "bug": ("d73a4a", "Something is not working"),
    "documentation": ("0075ca", "Improvements or additions to documentation"),
    "needs expertise: InSAR": ("5319e7", "Needs SAR / InSAR domain expertise"),
    "needs real-data validation": ("fbca04", "Implemented and tested on synthetic data only"),
    "needs GPU hardware": ("c5def5", "Needs access to a CUDA GPU"),
    "validation": ("0e8a16", "Comparison against an independent tool or ground truth"),
    "area: insar": ("1d76db", "InSAR processing chain"),
    "area: unwrapping": ("1d76db", "Phase unwrapping"),
    "area: atmosphere": ("1d76db", "Atmospheric / ionospheric correction"),
    "area: streaming": ("1d76db", "Streaming / partial download"),
    "area: performance": ("1d76db", "Memory, speed, hardware"),
    "area: testing": ("1d76db", "Tests and fixtures"),
}


def body(summary, now, needed, acceptance, skills, status, ref):
    return (
        f"{summary}\n\n"
        f"### Current behaviour\n{now}\n\n"
        f"### What is needed\n{needed}\n\n"
        f"### How we would know it is done\n{acceptance}\n\n"
        f"### Skills that help\n{skills}\n\n"
        f"### Status (as of 2026-11-02)\n{status}\n\n"
        f"_Contributions of any size are welcome, including a comment saying what you know about this "
        f"topic. Please say here before you start, so work is not duplicated._\n\n"
        f"<!-- pygeofetch-issue-id: {ref} -->\n"
    )


ISSUES = [
    # ------------------------------------------------------------------ InSAR processing
    dict(
        ref="burst-sync-pair-level",
        title="Burst-synchronisation check runs per scene family before download, per pair only after processing",
        labels=["limitation", "help wanted", "needs expertise: InSAR", "area: insar"],
        text=body(
            "The package's main promise is to screen Sentinel-1 scenes on physics before downloading. The "
            "burst-timing screen only holds per *family* of scenes; the pair-level offset is known only after "
            "processing.",
            "`preflight.py::_screen_burst_families` groups scenes from annotation XML before download. "
            "`esd.py::compute_burst_synchronization` gives the pair offset after processing. In a Silesia "
            "test, two 12-day S1B-S1B pairs measured 5.6 ms and 12.8 ms although the family check passed; ESA "
            "asks for under 5 ms.",
            "1. Hand-check the offset calculation against burst start times in the SAFE annotation files for "
            "3-5 pairs. The arithmetic was tested on synthetic orbits and burst grids (exact recovery, correct "
            "sign), so a disagreement on real files would be a finding in itself.\n"
            "2. Add a pair-level screen that reads annotation timing only (no full download) and returns an "
            "offset per candidate pair with the 5 ms flag.",
            "For a real stack, the pre-download offsets equal the post-processing offsets stored by "
            "`process_pair` within a stated tolerance for every pair; report the agreement count and the "
            "maximum difference.",
            "Sentinel-1 TOPS knowledge; reading SAFE annotation XML; Python.",
            "Arithmetic tested on synthetic data. The real-annotation hand-check and the pair-level screen "
            "are not done.",
            "burst-sync-pair-level",
        ),
    ),
    dict(
        ref="coreg-accuracy-validation",
        title="Coregistration accuracy is not compared with the 0.001-sample requirement on real data",
        labels=["limitation", "help wanted", "needs expertise: InSAR", "needs real-data validation",
                "area: insar"],
        text=body(
            "Interferometric phase needs azimuth coregistration better than about 0.001 samples (ESA) for "
            "under 3 degrees of phase error. The pipeline reports quality flags, but this requirement has not "
            "been checked on real data.",
            "`process_pair` records a coregistration quality record (GCP count, RMS, flags such as "
            "`coreg_unreliable` and `azimuth_misregistration_above_requirement`) and re-estimates the ESD "
            "shift after correction. These were written and tested on synthetic data; they have not been run "
            "on a real stack to see how many pairs meet the figure.",
            "Run the quality record on a real stack, report the distribution of residual azimuth "
            "misregistration and ESD overlap spread, and increase the number of usable GCPs where the offset "
            "field allows.",
            "A table of how many pairs meet 6 GCPs and the 0.001-sample figure, with the distribution, "
            "generated from saved result files.",
            "InSAR coregistration (ESD, offset fields); Sentinel-1 data access.",
            "Code built; not run on real data.",
            "coreg-accuracy-validation",
        ),
    ),
    dict(
        ref="geometry-first-real-data",
        title="Geometry-first coregistration is validated by simulation only",
        labels=["limitation", "help wanted", "needs real-data validation", "area: insar"],
        text=body(
            "Coregistration now uses the dense orbit/DEM offset field as the base and lets cross-correlation "
            "only correct it. This fixed pairs that the old rule rejected, but accuracy of the result for "
            "phase has not been verified.",
            "In a simulation of 60 cases the new fit had the lower error in all of them. Its weak spot is 2-3 "
            "cross-correlation points with 20 % gross outliers (95th-percentile error about 0.9 px). On a "
            "real 31-pair stack the old polynomial rule rejected 9 pairs; the new rule rejected none, and "
            "mean coherence over 22 shared pairs rose by about 0.011 (95 % interval about 0.001 to 0.023; "
            "driven by three pairs). Many coherences are near 0.3, so low-coherence pairs rely on few "
            "cross-correlation points.",
            "A test on real SLCs that compares phase quality (not only coherence) between "
            "`coregistration_fit=\"polynomial\"` and `\"geometry_first\"`, ideally against an independent "
            "processor.",
            "Phase RMS difference to a reference processor on the same pairs, with paired bootstrap "
            "intervals.",
            "InSAR coregistration; statistics.",
            "Simulation-tested. One real-data comparison exists (coherence only).",
            "geometry-first-real-data",
        ),
    ),
    dict(
        ref="esd-per-burst",
        title="ESD applies one global azimuth shift instead of a correction per burst",
        labels=["limitation", "help wanted", "needs expertise: InSAR", "area: insar"],
        text=body(
            "ISCE2 and SNAP apply the ESD correction per burst; pygeofetch reduces the per-overlap estimates "
            "to their median and applies one shift to the whole secondary image.",
            "The spread across burst overlaps is stored in `esd_overlap_std_px`, so the size of the problem "
            "is visible, but it is not corrected.",
            "Apply a per-burst (or smoothly varying) azimuth correction when the stored spread shows it "
            "matters.",
            "On a real stack, a lower residual phase discontinuity at burst boundaries than the global shift, "
            "reported with intervals.",
            "TOPS ESD theory; ISCE2 / SNAP implementations as reference.",
            "Open. Decision should follow from the stored spread on real data.",
            "esd-per-burst",
        ),
    ),
    dict(
        ref="real-burst-default",
        title="Real burst processing (per-burst ESD and deburst) is off by default",
        labels=["limitation", "enhancement", "help wanted", "needs real-data validation", "area: insar"],
        text=body(
            "`InterferogramGenerator(use_real_burst_processing=False)` is the default, so results carry the "
            "flag `no_deburst` unless the option is set.",
            "Flat-earth removal became automatic when both SAFE zips and orbit files are given. The burst "
            "path has not been made automatic because it has not been run on enough real data.",
            "Decide, with real-data evidence, whether to make the burst path automatic when the inputs allow "
            "it, and document the choice.",
            "A documented decision backed by a real-stack comparison of results with and without the burst "
            "path.",
            "TOPS processing; Python.",
            "Open (decision needed).",
            "real-burst-default",
        ),
    ),
    dict(
        ref="coherence-before-flattening",
        title="Coherence is estimated before the geometric phase is removed (biased low)",
        labels=["bug", "help wanted", "needs expertise: InSAR", "area: insar"],
        text=body(
            "Coherence estimation runs on the raw reference and secondary images, before flat-earth and "
            "topographic phase are removed, so any phase gradient inside the estimation window lowers it.",
            "By calculation, for a true coherence of 1 the estimate falls by about 8 % at 0.3 rad/pixel "
            "along range and about 23 % at 0.5 rad/pixel, and more in steep terrain. ISCE and GAMMA estimate "
            "coherence on the flattened interferogram. All coherence values (and anything gated on them) are "
            "affected. The change was not made because it alters every coherence value.",
            "Estimate the numerator from the flattened full-resolution interferogram and the denominator "
            "from the local powers of the two SLCs, behind a `PIPELINE_VERSION` bump so resume does not mix "
            "old and new results.",
            "Synthetic test with a known phase ramp shows no bias; a real-pair comparison before/after is "
            "saved; documentation notes that old results change.",
            "InSAR coherence estimation; numpy.",
            "Open.",
            "coherence-before-flattening",
        ),
    ),
    dict(
        ref="dem-geoid-default",
        title="DEM heights are used as ellipsoidal heights unless dem_geoid_offset_m is set",
        labels=["limitation", "help wanted", "needs real-data validation", "area: insar"],
        text=body(
            "Most DEMs (including Copernicus DEM) are orthometric (EGM2008), but the geometry needs "
            "ellipsoidal heights.",
            "`dem_geoid_offset_m` defaults to 0.0, so the DEM is used as is. In Silesia the difference is "
            "about +40 m, which gives a nearly constant phase offset and a geocoding shift of about "
            "N/tan(incidence).",
            "Apply a geoid model (e.g. EGM2008) automatically when the DEM is known to be orthometric, and "
            "record which vertical datum was used in the result metadata. Until then the default should be "
            "stated prominently in the docs.",
            "A test that a DEM of known datum gives the same geocoding as an ellipsoidal reference, and the "
            "datum appears in `metadata`.",
            "Geodesy (geoid models); rasterio / GDAL vertical CRS handling.",
            "Open. Comparison runs against other processors must treat the DEM identically.",
            "dem-geoid-default",
        ),
    ),
    dict(
        ref="transform-multilook",
        title="result.profile[\"transform\"] is not scaled for multilooked results",
        labels=["bug", "good first issue", "help wanted", "area: insar"],
        text=body(
            "The affine transform stored with a multilooked result describes the native-resolution grid.",
            "`result.profile[\"transform\"]` is the native-resolution GCP-fitted affine. Consumers that map "
            "pixels to coordinates from it are off by the look factor.",
            "Scale the transform by the azimuth and range looks (or document clearly that it is native "
            "resolution and provide a scaled one).",
            "A unit test: multilook by (a, r), then pixel-to-coordinate round trip matches the native grid "
            "within one multilooked pixel.",
            "Python, rasterio affine transforms.",
            "Open.",
            "transform-multilook",
        ),
    ),
    dict(
        ref="geocoding",
        title="Output is in radar geometry with an approximate affine transform; no geocoding step",
        labels=["limitation", "enhancement", "help wanted", "needs expertise: InSAR", "area: insar"],
        text=body(
            "Products stay in radar coordinates. Where the SLC has no transform, one is fitted from GCPs.",
            "No geocoding step exists in the processing chain (`coregister.py::collocate_by_geocoding` is a "
            "helper for collocation). Per-pixel longitude, latitude, incidence and baseline are available in "
            "`result.geometry`, but there is no lookup-table output or geocoded GeoTIFF.",
            "A documented geocoding method plus lookup tables (latitude, longitude, incidence, heading) that "
            "exporters need.",
            "A geocoded coherence map overlaid on a reference basemap or the DEM with a measured position "
            "error.",
            "Range-Doppler geocoding; GDAL/rasterio warping.",
            "Open. The exporters (MintPy, dolphin) depend on this.",
            "geocoding",
        ),
    ),
    dict(
        ref="exporters",
        title="Exporters for MintPy and dolphin",
        labels=["enhancement", "help wanted", "needs expertise: InSAR", "area: insar"],
        text=body(
            "pygeofetch should hand its stacks to established time-series engines.",
            "MintPy appears only in docstrings; no export function exists. dolphin starts from coregistered "
            "SLCs.",
            "`export_mintpy()` (interferogram stack plus geometry in MintPy's input layout) and "
            "`export_dolphin()` (coregistered SLC stack plus metadata). Start with the one that needs the "
            "least geometry.",
            "The exported stack is read without error by the target tool on a real stack, and the tool's own "
            "output is saved.",
            "MintPy and/or dolphin file layouts; HDF5.",
            "Open. Needs the geocoding / lookup tables issue first for MintPy.",
            "exporters",
        ),
    ),
    dict(
        ref="phase-linking",
        title="No phase linking (EVD/EMI) for distributed scatterers",
        labels=["limitation", "enhancement", "help wanted", "needs expertise: InSAR", "area: insar"],
        text=body(
            "At low coherence (Silesia mean about 0.27) persistent-scatterer selection alone leaves few "
            "usable pixels; phase linking is the standard way to use distributed scatterers.",
            "No EVD or EMI code exists in `insar/`; `ps_selection.py` has ADI and temporal coherence only.",
            "Either a small tested phase-linking function (EVD, then EMI) for homogeneous patches, or a "
            "documented recommendation to use dolphin as the back-end (after the exporter exists).",
            "On a stack with a known reference result (e.g. dolphin output) the estimate agrees within a "
            "stated phase RMS; coherent-pixel share before and after is reported.",
            "Phase linking (EVD/EMI), statistics of distributed scatterers.",
            "Open. Decision needed after the dolphin exporter: build or recommend dolphin.",
            "phase-linking",
        ),
    ),
    dict(
        ref="temporal-unwrapping",
        title="Unwrapping is spatial only; no temporal or 3D unwrapping hand-off",
        labels=["limitation", "enhancement", "help wanted", "needs expertise: InSAR", "area: unwrapping"],
        text=body(
            "`unwrap.py` wraps SNAPHU; pyunwrap is a second method. Both unwrap each interferogram on its "
            "own.",
            "No temporal or 3D unwrapping exists.",
            "An optional hand-off to an existing 3D unwrapper (after the exporter exists) rather than "
            "writing one from scratch.",
            "The hand-off runs on a real stack and its output is saved and read back.",
            "Time-series unwrapping tools; Python.",
            "Open.",
            "temporal-unwrapping",
        ),
    ),
    dict(
        ref="topo-real-data",
        title="Topographic phase simulation has only been validated on synthetic geometry",
        labels=["limitation", "help wanted", "needs real-data validation", "needs expertise: InSAR",
                "area: insar"],
        text=body(
            "Topographic phase is simulated per pixel from range-Doppler geometry and the DEM, replacing an "
            "older height regression. Accuracy is known only on synthetic data.",
            "Synthetic test (27-30 degrees incidence, 53 m baseline, 28 rad of topographic phase): "
            "simulated minus exact phase rms 0.0015 rad, max 0.007 rad. No residual statistics from a real "
            "stack have been computed yet. Layover/shadow pixels get flat-earth phase only and are "
            "flagged. Geocoding precision rests on the DEM and the orbit.",
            "Run on a real stack and report the residual phase-height correlation before and after, plus a "
            "per-date slope test (a baseline-proportional slope indicates DEM error).",
            "Result files with those statistics for a stated stack.",
            "InSAR geometry; DEM handling.",
            "Built, not validated on real data.",
            "topo-real-data",
        ),
    ),
    dict(
        ref="ps-stack-real-data",
        title="Persistent-scatterer stack builder (ps_stack) has not been checked against a direct SLC amplitude",
        labels=["limitation", "help wanted", "needs real-data validation", "area: insar"],
        text=body(
            "`build_ps_stack` builds a master-referenced stack from `process_pair` output and rejects "
            "untrusted dates with the reason.",
            "Amplitude is recovered from the interferogram. A direct check that it matches the amplitude "
            "read from the coregistered SLC on a real date is still open. Held-out comparison of selection "
            "methods (ADI vs a learned selector) on real data is also open.",
            "(1) Compare recovered and directly read amplitude on one real date. (2) Run "
            "`holdout_stability` for the selectors on a real stack with bootstrap intervals.",
            "Both results saved as files and referenced in the docs.",
            "PS-InSAR; Python.",
            "Tested on stand-ins and a small real run only.",
            "ps-stack-real-data",
        ),
    ),
    # ------------------------------------------------------------------ unwrapping
    dict(
        ref="pyunwrap-padding",
        title="pyunwrap pads small grids heavily; tile size is not chosen from the grid",
        labels=["limitation", "help wanted", "area: unwrapping"],
        text=body(
            "On small multilooked grids most of what the network sees is padding.",
            "A 67 x 847 grid on a 256-pixel tile is about three quarters padding. `LearnedUnwrapper` "
            "measures the share and raises the flag `heavy_padding` above 50 %, but does not pick a better "
            "tile.",
            "Choose the tile from the grid, or run with less aggressive multilooking, and report both.",
            "Padding share and agreement with SNAPHU reported for both settings on the same pairs.",
            "PyTorch inference; tiling.",
            "Open.",
            "pyunwrap-padding",
        ),
    ),
    dict(
        ref="learned-unwrap-validation",
        title="Learned and hybrid unwrapping are not validated on real data; the checkpoint config is not shipped",
        labels=["limitation", "help wanted", "needs real-data validation", "needs expertise: InSAR",
                "area: unwrapping"],
        text=body(
            "pygeofetch can load a pyunwrap checkpoint and run it, and combine it with SNAPHU in "
            "`HybridUnwrapper`. Evidence that this helps on real data is missing.",
            "On the training run's own synthetic test the network alone is below SNAPHU (88.2 / 87.1 / 49.5 % "
            "against 100 / 97.9 / 63.2 %) and the tuned hybrid equals SNAPHU (63.3 against 63.2 on hard "
            "scenes). In one single-site 78-pair test, the network's phase-closure was worse than SNAPHU's on "
            "most triplets (better on 8 of 36), and agreement with SNAPHU followed coherence. There is no "
            "ground truth, so this only shows consistency, not accuracy. The loader refuses to guess its "
            "configuration: a checkpoint without `config.json` raises `ModelLoadError`, and the origin of "
            "the configuration values used as a fallback is not documented.",
            "(1) Ship or document the `config.json` that belongs to each released checkpoint. (2) Evaluate "
            "on more sites and a range of coherence. (3) Keep the hybrid only if its triplet closure beats "
            "SNAPHU's. (4) Integrate the training run's correction stages, which are not in the package.",
            "Closure and agreement tables for several sites with paired bootstrap intervals.",
            "Deep-learning phase unwrapping; InSAR validation design.",
            "Built, not validated on real data or on the real checkpoint beyond the one test above.",
            "learned-unwrap-validation",
        ),
    ),
    # ------------------------------------------------------------------ atmosphere
    dict(
        ref="atmosphere-real-data",
        title="ERA5/GACOS atmospheric correction has never run on real model data",
        labels=["limitation", "help wanted", "needs real-data validation", "needs expertise: InSAR",
                "area: atmosphere"],
        text=body(
            "`AtmosphericCorrector.correct` was rewritten (geometry-based geocoding, per-pixel incidence, "
            "robust height fit, validation gates). It was tested on synthetic data and a fake pyaps only.",
            "The real ERA5 download path needs CDS credentials and `pyaps3` and was never exercised. No "
            "independent validation exists.",
            "Run the ERA5 and GACOS paths on real scenes and compare corrected delays with GNSS "
            "zenith-total-delay where available.",
            "Variance reduction and GNSS agreement reported for a stated stack.",
            "Tropospheric delay modelling; pyaps3; GNSS ZTD data.",
            "Built, not validated on real data.",
            "atmosphere-real-data",
        ),
    ),
    dict(
        ref="aps-time-series",
        title="No time-series filtering of atmospheric phase screens for SBAS / PS",
        labels=["limitation", "enhancement", "help wanted", "needs expertise: InSAR", "area: atmosphere"],
        text=body(
            "The standard approach to atmospheric phase screens in time series (high-pass in time, low-pass "
            "in space) is not implemented.",
            "`timeseries.py` has no APS filtering step.",
            "Add it with a documented, tested implementation.",
            "A synthetic test recovers a known deformation signal in the presence of a simulated APS.",
            "Time-series InSAR (SBAS/PS).",
            "Open.",
            "aps-time-series",
        ),
    ),
    dict(
        ref="ionosphere",
        title="Ionosphere: split-spectrum is a diagnostic only; ionosphere.py was not reviewed",
        labels=["limitation", "help wanted", "needs expertise: InSAR", "area: atmosphere"],
        text=body(
            "Ionospheric phase is not corrected.",
            "`splitspectrum_diagnostic.py` says it is a diagnostic. `ionosphere.py` (IONEX thin shell) was "
            "not reviewed in the same audit as the troposphere code.",
            "Review the IONEX path; decide whether to build a split-spectrum correction (deferred unless a "
            "real result needs it).",
            "Review notes in the issue; if built, a synthetic and a real-data test.",
            "L-band / C-band ionosphere in InSAR; IONEX products.",
            "Open (deferred by decision).",
            "ionosphere",
        ),
    ),
    # ------------------------------------------------------------------ validation
    dict(
        ref="benchmark-snap-hyp3",
        title="No head-to-head benchmark against SNAP, HyP3 or ISCE on the same pairs",
        labels=["limitation", "help wanted", "validation", "needs expertise: InSAR", "area: insar"],
        text=body(
            "There is no comparison of pygeofetch interferograms with an established engine on identical "
            "pairs, so agreement with accepted processors is unknown. This is the largest missing piece of "
            "external validation.",
            "Existing tests use synthetic data and one single-site run.",
            "A benchmark script that takes the same pairs processed by pygeofetch and by SNAP (graph "
            "processing), HyP3 (INSAR_GAMMA) or ISCE2, on a common grid with matched settings (DEM, orbits, "
            "looks, filter, coherence window), and reports coherence correlation and difference, "
            "wrapped-phase difference after plane removal, and closure for each tool, with paired bootstrap "
            "intervals (5000 resamples, 95 %).",
            "One result file per pair; tables generated from those files.",
            "SNAP / HyP3 / ISCE2 experience; InSAR comparison design; statistics.",
            "Open. A SNAP comparison on a small number of pairs is being prepared by the maintainers; help "
            "with more pairs, other processors and other sites is very welcome.",
            "benchmark-snap-hyp3",
        ),
    ),
    dict(
        ref="real-data-test-fixtures",
        title="Most validation is synthetic; a small real-data test fixture is needed",
        labels=["limitation", "help wanted", "good first issue", "area: testing"],
        text=body(
            "Many of the InSAR modules were tested only against synthetic data or stand-ins, so regressions "
            "against real Sentinel-1 data would not be caught.",
            "No small real burst pair with orbit files and a DEM clip is part of the tests.",
            "A tiny fixture (one pair of single-burst crops, orbit files, a DEM clip) small enough for CI, "
            "plus tests that check geometry, coregistration records and result metadata against stored "
            "reference values. Please check and state the data licence and source before adding any real "
            "data to the repository.",
            "CI runs the real-data tests in minutes and fails when a result changes beyond a stated "
            "tolerance.",
            "Python testing; Sentinel-1 data handling.",
            "Open.",
            "real-data-test-fixtures",
        ),
    ),
    # ------------------------------------------------------------------ performance
    dict(
        ref="gpu-real-hardware",
        title="GPU path has never been run on real CUDA hardware, and only covers coherence",
        labels=["limitation", "help wanted", "needs GPU hardware", "area: performance"],
        text=body(
            "`use_gpu=True` is supported by the chunked coherence estimation via CuPy.",
            "It was tested with a stand-in CuPy only. Coregistration cross-correlation, resampling, "
            "topographic phase and Goldstein filtering stay on the CPU, so the speed-up is limited. "
            "`gpu.gpu_info()` reports why the GPU is unavailable.",
            "Run on real CUDA hardware, record timings and results against the CPU path, and port other "
            "stages only where a benchmark shows they dominate.",
            "A timing table on stated hardware and identical numbers (to a stated tolerance) CPU vs GPU.",
            "CuPy / CUDA; profiling.",
            "Open. No GPU or scale claim is made until this is closed.",
            "gpu-real-hardware",
        ),
    ),
    dict(
        ref="parallel-pairs",
        title="No parallel execution across pairs",
        labels=["limitation", "enhancement", "help wanted", "area: performance"],
        text=body(
            "Pairs are processed one at a time.",
            "`process_pairs` runs sequentially (with resume). No parallelism exists in `insar/`.",
            "Per-pair parallelism with a memory-aware worker count, added only if a benchmark shows the "
            "per-pair loop is the bottleneck.",
            "A timing table; no change in per-pair results.",
            "Python multiprocessing; memory profiling.",
            "Open.",
            "parallel-pairs",
        ),
    ),
    dict(
        ref="memory-model-real-scenes",
        title="Memory estimate factors come from a synthetic probe; real full-size scenes are unprofiled",
        labels=["limitation", "help wanted", "area: performance"],
        text=body(
            "`memory_guard` compares an estimated peak with available memory before reading SLCs.",
            "The factors (8x the larger SLC chunked, 14x unchunked) come from a 1 x 1-look synthetic probe "
            "that measured 10.9x on a small image. They are conservative but not measured on real scenes. "
            "A 16 GB machine still ran out of memory on large single-look pairs in practice. Processing a "
            "burst-aligned AOI window instead of the whole sub-swath would use about 8x less memory but "
            "changes the extraction and therefore the numbers.",
            "Profile real full-size pairs (tracemalloc and OS peak), correct the factors, and reduce the "
            "peak in the coregistration read and deburst stages.",
            "A table of measured peaks for stated image sizes and the guard's estimates.",
            "Memory profiling; numpy.",
            "Open.",
            "memory-model-real-scenes",
        ),
    ),
    dict(
        ref="geometry-pickle",
        title="Resume stores geometry as a pickle (geometry.pkl); replace with a safe format",
        labels=["enhancement", "good first issue", "help wanted", "area: insar"],
        text=body(
            "Saved pair folders contain `geometry.pkl`.",
            "Loading a pickle runs arbitrary code, so loading someone else's output folder is unsafe "
            "(the docs say to load only your own outputs).",
            "Store the per-pixel geometry arrays in `.npz` or GeoTIFF/HDF5 and keep reading old folders for "
            "one release.",
            "Round-trip test; `result.json` fingerprint unchanged by the format.",
            "Python serialization.",
            "Open.",
            "geometry-pickle",
        ),
    ),
    dict(
        ref="resume-fingerprint-cause",
        title="Resume: cause of pairs being regenerated on one user's machine was never confirmed",
        labels=["limitation", "help wanted", "area: testing", "area: insar"],
        text=body(
            "Resume decides whether a saved pair is current from a fingerprint of the pipeline version, "
            "generator settings, input identities and `process_pair` arguments.",
            "One user reported pairs regenerated on every run. An end-to-end test resumed correctly, and the "
            "most likely cause (input GeoTIFFs rewritten with identical content) was addressed by "
            "identifying GeoTIFFs by name and raster header. The cause on that machine was not confirmed. "
            "Also, `PIPELINE_VERSION` must be bumped by hand whenever a change alters numbers; nothing "
            "enforces it.",
            "A test that fails when numeric output changes but `PIPELINE_VERSION` does not (e.g. a stored "
            "reference result), and a clearer message naming what changed.",
            "CI check passes when versions are bumped properly and fails when they are not.",
            "Python testing.",
            "Open.",
            "resume-fingerprint-cause",
        ),
    ),
    # ------------------------------------------------------------------ streaming
    dict(
        ref="streaming-real-providers",
        title="Streaming savings were measured on one local COG; real-provider benchmarks are missing",
        labels=["limitation", "help wanted", "validation", "area: streaming"],
        text=body(
            "`read_window()` reads only the tiles that cover an AOI from a COG, over HTTP range requests.",
            "The reported savings (99.70 % of the file avoided for a 1 km AOI, 94.8 % for 20 km) come from "
            "one 10980 x 10980 uint16 COG with 512 x 512 tiles on a local server that supports range "
            "requests. Real savings depend on tile layout, compression and the provider.",
            "Repeat the measurement on real COGs from several providers (different tile sizes and "
            "latency), and report bytes read and time against a full download.",
            "Table of bytes and time per provider and AOI size, generated from saved result files.",
            "Cloud-optimised GeoTIFF; HTTP range requests; GDAL.",
            "Open.",
            "streaming-real-providers",
        ),
    ),
    dict(
        ref="s1-partial-download",
        title="Sentinel-1 SLC needs a full zip download before the AOI crop",
        labels=["limitation", "enhancement", "help wanted", "needs expertise: InSAR", "area: streaming"],
        text=body(
            "Streaming works for COGs. Sentinel-1 SLC is delivered as a zip archive, which is refused for "
            "range-request streaming.",
            "pygeofetch downloads the complete SAFE zip (several GB per scene) and then crops the AOI "
            "locally from the archive. For stacks of dozens of dates this dominates transfer volume.",
            "Investigate whether any provider offers partial or burst-level access that can be used safely "
            "(for example selecting bursts or byte ranges inside the archive), and implement it behind a "
            "checked provider capability. Preflight screening already avoids downloading scenes that cannot "
            "work.",
            "A measured reduction in downloaded bytes for an AOI on a real provider, with identical cropped "
            "SLCs.",
            "Sentinel-1 data access (CDSE, ASF); zip / SAFE structure.",
            "Open. Whether this is possible depends on the provider; no claim is made yet.",
            "s1-partial-download",
        ),
    ),
    # ------------------------------------------------------------------ docs
    dict(
        ref="docs-behaviour-changes",
        title="Document behaviour changes that alter earlier results (migration notes)",
        labels=["documentation", "good first issue", "help wanted"],
        text=body(
            "Several fixes change results produced by earlier versions, and users need one place to read "
            "what changed.",
            "The changes are described only in internal notes: flat-earth removal is automatic when both "
            "SAFE zips and orbit files are given; multilooking order was fixed (phase noise was far higher "
            "than necessary before); amplitude of multilooked results is now the root-mean-square; "
            "`on_error=\"raise\"` is the default and pairs carry a quality grade (`science`, `quicklook`, "
            "`degraded`); geometry-first coregistration changes coherence of existing pairs; resume "
            "recomputes old folders once.",
            "A `CHANGELOG` / migration page that lists each change, how to restore the old behaviour where "
            "possible, and which results must be regenerated.",
            "Every default change in the code appears in the page.",
            "Technical writing; reading the code history.",
            "Open.",
            "docs-behaviour-changes",
        ),
    ),
]


# ----------------------------------------------------------------------------------- already fixed issues
# These are created and then closed straight away, so the history shows what was found and fixed. "Fixed" means
# fixed in the maintainers' development code; where something is still unvalidated on real data, the issue says
# so and points to the open issue that tracks it.
FIXED_NOTE = "{FIXED_IN}"


def fixed_body(problem, fix, evidence, remaining, ref):
    return (
        f"### Problem\n{problem}\n\n"
        f"### Resolution\n{fix}\n\n"
        f"### Evidence\n{evidence}\n\n"
        f"### What this does not cover\n{remaining}\n\n"
        f"{FIXED_NOTE}\n\n"
        f"<!-- pygeofetch-issue-id: {ref} -->\n"
    )


CLOSED = [
    dict(
        ref="fixed-multilook-order",
        title="Multilooked interferograms were far noisier than necessary (multilooking order)",
        labels=["bug", "area: insar"],
        text=fixed_body(
            "`process_pair` multilooked the two SLCs and multiplied the averages. That leaves the phase noise "
            "at the single-look level whatever the number of looks.",
            "The interferogram is now formed at full resolution, the geometric phase is removed there, and "
            "the complex mean is taken afterwards. Coherence was always estimated at full resolution and is "
            "unchanged.",
            "Simulated speckle at coherence 0.8 with 32 looks: phase standard deviation 0.92 rad before, "
            "0.097 rad after (measured through `process_pair`: 0.915 and 0.097).",
            "Every multilooked interferogram produced before this change was noisier than needed and should "
            "be regenerated.",
            "fixed-multilook-order",
        ),
    ),
    dict(
        ref="fixed-flat-earth-ramp",
        title="Flat-earth phase ramp wrong for cropped and multilooked products",
        labels=["bug", "area: insar"],
        text=fixed_body(
            "The old flat-earth polynomial was fitted in full-swath pixel coordinates but evaluated at the "
            "indices of the cropped, multilooked array. The crop offset was never applied, so a multilooked "
            "product got a range ramp compressed by the number of looks.",
            "Replaced by the exact range-Doppler geometry evaluated on the output grid (crop, deburst and "
            "multilook offsets applied). Flat-earth removal is now automatic when both SAFE zips and both "
            "orbit files are given (`remove_flat_earth_phase=None`); pass `False` for the old behaviour.",
            "Tested on synthetic geometry (see the open topographic-phase validation issue for the numbers).",
            "Not yet validated against an independent processor on real data.",
            "fixed-flat-earth-ramp",
        ),
    ),
    dict(
        ref="fixed-multilook-amplitude",
        title="Amplitude of multilooked results was biased low by about 2.5 dB",
        labels=["bug", "area: insar"],
        text=fixed_body(
            "Amplitude of a multilooked result was the log of the averaged complex value, which is biased "
            "low for speckle.",
            "It is now the root-mean-square over the look window. Unchanged at 1 x 1 looks.",
            "The bias of about 2.5 dB for speckle was derived for the old estimator; the new value is "
            "checked on simulated speckle.",
            "Results produced before the change should be regenerated.",
            "fixed-multilook-amplitude",
        ),
    ),
    dict(
        ref="fixed-topo-regression",
        title="Topographic phase was a regression against the DEM, not a simulation",
        labels=["bug", "area: insar"],
        text=fixed_body(
            "`_remove_topographic_phase*` fitted phase against DEM height and removed it only when R2 >= 0.5. "
            "That removes whatever correlates with height (atmosphere, slope deformation) and nothing where "
            "the fit is weak.",
            "Replaced by `insar/radar_geometry.py` and `insar/topo.py`: exact range-Doppler geometry on a "
            "node grid, iterative terrain geocoding from the DEM, and per-pixel geometric phase "
            "(flat-earth + k1*h + k2*h^2/2) removed at full resolution. The regression remains only as a "
            "flagged fallback (`topo_regression_fallback`) when orbit geometry is missing.",
            "Synthetic geometry (27-30 degrees incidence, 53 m baseline, 28 rad of topographic phase): "
            "simulated minus exact phase rms 0.0015 rad, max 0.007 rad; through `process_pair` with 4 x 8 "
            "looks and a crop offset the residual is 0.001 rad rms.",
            "Not validated on real data; that is tracked in the open topographic-phase issue. DEM heights "
            "are used as ellipsoidal unless `dem_geoid_offset_m` is set (also tracked).",
            "fixed-topo-regression",
        ),
    ),
    dict(
        ref="fixed-silent-fallbacks",
        title="process_pair returned weaker products silently when a step failed",
        labels=["bug", "area: insar"],
        text=fixed_body(
            "Nine places caught an exception and returned a weaker product that looked normal (orbit "
            "coregistration, raster collocation, burst ESD/deburst, topographic phase, flat-earth phase, AOI "
            "crop, burst-sync diagnostic, baseline, swath check).",
            "`InterferogramGenerator(on_error=\"raise\")` is now the default: a requested step that fails "
            "raises a `ProcessingError` subclass naming the stage, with the cause chained. "
            "`on_error=\"degrade\"` keeps the old fallbacks but grades the result `degraded`. Every result "
            "carries `metadata[\"quality\"]` (`science`, `quicklook`, `degraded`). Hard limits "
            "`min_coreg_gcps`, `max_coreg_rms_px`, `max_burst_sync_ms` can reject a pair. Fallbacks and "
            "mixed-satellite or mixed-track pairs now raise `RegistrationFallbackWarning`, "
            "`FlatEarthWarning` and `MixedSatelliteWarning` instead of only being logged. `build_stack` "
            "rejects `degraded` pairs always.",
            "Behaviour follows what ISCE2, GAMMA, SNAP and HyP3 do when a requested step fails; tests cover "
            "the error paths.",
            "Existing notebooks that relied on silent fallbacks will now stop or warn.",
            "fixed-silent-fallbacks",
        ),
    ),
    dict(
        ref="fixed-coreg-geometry",
        title="Coregistration discarded the orbit/DEM geometry and rejected good pairs",
        labels=["bug", "area: insar"],
        text=fixed_body(
            "From a real 31-pair Silesia log: after cross-correlation only 7-27 points survived at coherence "
            "about 0.3. The fit then rejected every point above the mean RMS on each pass, which removes "
            "about half the points whatever their quality. 22 accepted pairs were fitted through 3-5 points "
            "(a plane through 3 points has zero residual) and 9 pairs were rejected outright. The dense "
            "orbit/DEM field was discarded after the first step.",
            "`coregister.fit_geometry_first_offsets`: the base field is a polynomial through all orbit/DEM "
            "points; cross-correlation offsets only correct it (median + 3 MAD outlier rejection; 8 or more "
            "points give a linear correction, 2-7 a constant, fewer use geometry only). "
            "`process_pair(coregistration_fit=\"geometry_first\"|\"polynomial\")`; \"polynomial\" restores "
            "the old behaviour.",
            "Simulation of 60 cases: the new field has the lower error in all of them (e.g. 5 points, noise "
            "0.25 px: 0.16 px against 0.29-0.82 px). On the real stack the old rule rejected 9 of 31 pairs "
            "and the new one 0 of 31.",
            "Simulation-tested; validation against an independent processor on real data is tracked in an "
            "open issue. Coherence values of existing pairs change.",
            "fixed-coreg-geometry",
        ),
    ),
    dict(
        ref="fixed-coreg-quality-record",
        title="Coregistration reliability was computed but never reported",
        labels=["bug", "area: insar"],
        text=fixed_body(
            "`is_reliable()` (at least 6 GCPs, RMS at most 1.0 px) existed but was not called from "
            "`interferogram.py`; pairs with 4 final GCPs passed silently.",
            "Every `process_pair` result now carries `metadata[\"quality\"]` with the fit checked against "
            "6 GCPs and 1.0 px (flag `coreg_unreliable`), plus the ESD residual and overlap spread compared "
            "with 0.001 samples.",
            "GCP counts and RMS values appear in the pair logs of real runs; the quality grading itself was tested on synthetic pairs.",
            "Whether pairs meet the 0.001-sample requirement on real data is tracked in an open issue.",
            "fixed-coreg-quality-record",
        ),
    ),
    dict(
        ref="fixed-resume",
        title="No resume for process_pair, and pairs were regenerated on every run",
        labels=["bug", "enhancement", "area: insar"],
        text=fixed_body(
            "Batch runs recomputed finished pairs, and when resume was added, pairs were still regenerated "
            "because the fingerprint identified input files by name and byte size, which change when a "
            "notebook re-extracts a file with identical content.",
            "`process_pair(output_dir=..., resume=True)` saves `result.json` (written last), the complex "
            "interferogram and geometry, and loads a pair only when its fingerprint matches. GeoTIFFs are "
            "identified by name and raster header. `InterferogramGenerator.process_pairs(...)` checks every "
            "pair before opening any SLC, prints one decision line per pair, returns small records, and "
            "continues past a failing pair. Results carry `metadata[\"resume_status\"]`.",
            "A real 78-pair re-run: \"78 skipped, 0 computed, 0 failed\"; an earlier run: 53 skipped, 25 "
            "computed.",
            "The cause of the original regeneration on one machine was never confirmed (tracked in an open "
            "issue), and `PIPELINE_VERSION` still has to be bumped by hand.",
            "fixed-resume",
        ),
    ),
    dict(
        ref="fixed-deburst-memory",
        title="Out-of-memory crash in the burst stage on large pairs",
        labels=["bug", "area: performance"],
        text=fixed_body(
            "Debursting built a second full-size array, so a large single-look pair ran the machine out of "
            "memory right after coregistration.",
            "`deburst_array(..., inplace=True)` compacts the kept rows inside the existing array; "
            "`plan_deburst` validates both images before either is changed.",
            "Output identical to the copying version (tested). Extra peak memory in the test: 391 MB to "
            "45 MB for a 432 MB image.",
            "Not profiled on the full-size real stack; the crash was inferred from the log. See the open "
            "memory-model issue.",
            "fixed-deburst-memory",
        ),
    ),
    dict(
        ref="fixed-ps-stack-crash",
        title="Persistent-scatterer stack build crashed the machine (no chunking, no resume)",
        labels=["bug", "area: performance"],
        text=fixed_body(
            "The stack build created its generator without `chunk_size`, so every single-look pair was read "
            "and processed whole, and a crash lost all finished dates.",
            "`build_ps_stack` now sets `chunk_rows=500` when the generator has none (results unchanged), "
            "saves each master pair as it finishes (`output_dir=..., resume=True`), and loads finished "
            "dates instead of recomputing. `insar/memory.py` and `InterferogramGenerator(memory_guard="
            "\"warn\"|\"raise\"|\"off\")` compare an estimated peak with available memory before the SLCs "
            "are read.",
            "Tested with a stand-in `process_pair` (crash on date 4, rerun processes only that date).",
            "The memory estimate factors come from a synthetic probe, and the resumable build has not been "
            "run on all real scenes. See the open memory-model issue.",
            "fixed-ps-stack-crash",
        ),
    ),
    dict(
        ref="fixed-ps-stack-builder",
        title="No generic stack builder for persistent scatterers",
        labels=["enhancement", "area: insar"],
        text=fixed_body(
            "`ps_selection.py` needed a co-registered amplitude stack and nothing in pygeofetch built one; "
            "an external project duplicated pygeofetch's registration with a relaxed version.",
            "`insar/ps_stack.py` (`PSStack`, `build_ps_stack`, selector protocol with `ADISelector` built "
            "in, `compare_selections`, `holdout_stability`) and `insar/ps_eval.py`. Untrusted dates are "
            "rejected with the reason.",
            "Used on a real master-referenced stack in the Silesia project; selector comparison produced "
            "real selection counts.",
            "Held-out comparison on real data and the amplitude check are tracked in an open issue.",
            "fixed-ps-stack-builder",
        ),
    ),
    dict(
        ref="fixed-pyunwrap-access",
        title="pyunwrap could only be used from notebook code, with a guessed configuration",
        labels=["enhancement", "area: unwrapping"],
        text=fixed_body(
            "pygeofetch had no way to load a pyunwrap checkpoint or run it. The notebook carried its own "
            "loader with `weights_only=False`, a hard-coded `k_max`, hand-typed tile size and overlap, and "
            "multilook scales guessed from the channel count. Averaging `k` across tiles with no "
            "whole-cycle alignment was also found to cost accuracy.",
            "`insar/learned_unwrap.py`: safe checkpoint loading, configuration from `config.json` and never "
            "guessed, strict state-dict check, device check and a synthetic self-test, aligned tile "
            "merging, reliability mask, quality record, and `unwrap_interferograms` for a stack. "
            "`insar/hybrid_unwrap.py`: SNAPHU and the network combined with components tied into one "
            "cycle frame. New errors `UnwrapError`, `ModelLoadError`, `UnwrapDependencyError`.",
            "Tiling, alignment and error paths tested with a stand-in network; the hybrid tested with real "
            "SNAPHU 2.0.7 on synthetic scenes.",
            "Not validated on real data (tracked in an open issue). The training run's correction stages "
            "are not integrated.",
            "fixed-pyunwrap-access",
        ),
    ),
    dict(
        ref="fixed-atmosphere-rewrite",
        title="Atmospheric correction geocoded by array shape and never checked that it helped",
        labels=["bug", "area: atmosphere"],
        text=fixed_body(
            "ERA5/GACOS maps were resized to the interferogram's shape, assuming the same footprint and "
            "orientation. It also used one fixed 38-degree incidence for the whole swath, truncated the "
            "ERA5 hour, averaged wrapped angles in a spatial filter, filled NaNs with zero, fitted heights "
            "without a ramp term, and had no check that a correction helped.",
            "`insar/atmosphere_core.py` and a rewritten `AtmosphericCorrector.correct`: geocoding by "
            "geometry or profile, per-pixel incidence, ERA5 hour interpolation, a robust height fit with a "
            "ramp term and block-jackknife error, relief, significance and variance-reduction gates, and "
            "`AtmosphereError` or `on_error=\"skip\"`.",
            "Synthetic: with exact geometry a 15.5 rad delay is removed to the 0.10 rad noise floor and a "
            "3 rad deformation bowl is preserved; the gate rejects wrong-sign and wrong-date model data.",
            "Never run on real ERA5/GACOS data (tracked in an open issue).",
            "fixed-atmosphere-rewrite",
        ),
    ),
    dict(
        ref="fixed-gpu-silent",
        title="use_gpu=True did nothing with chunked processing, and failures were hidden",
        labels=["bug", "area: performance"],
        text=fixed_body(
            "`use_gpu=True` was read only by the non-chunked coherence function; with chunking the CPU path "
            "ran without saying so. `gpu_available()` swallowed ImportError and driver errors at debug "
            "level.",
            "The chunked coherence runs on CuPy when possible; `gpu.gpu_info()` reports the device and the "
            "reason it is unavailable; the constructor warns once with the reason; a trivial CuPy "
            "computation must succeed before the GPU counts as available; `metadata[\"coherence_backend\"]` "
            "records what ran.",
            "Tested with a stand-in CuPy (same numbers as CPU).",
            "Never run on real CUDA hardware; only coherence uses the GPU (tracked in an open issue).",
            "fixed-gpu-silent",
        ),
    ),
    dict(
        ref="fixed-verbosity",
        title="Log output too noisy; coregistration summary logged twice",
        labels=["enhancement", "area: insar"],
        text=fixed_body(
            "Per-pair repeats (parsed geometry, orbit state vectors, saved plots) flooded the notebook "
            "output, and the coregistration summary line was logged twice per pair.",
            "`InterferogramGenerator(verbosity=\"quiet\"|\"normal\"|\"full\"|\"debug\")` and "
            "`pygeofetch.insar.set_verbosity(...)`; levels apply to the `pygeofetch.insar` and "
            "`pygeofetch.viz` loggers only, and warnings are never hidden. The summary is logged once.",
            "Tested on the synthetic end-to-end pair; the levels only change logger settings.",
            "None known.",
            "fixed-verbosity",
        ),
    ),
    dict(
        ref="fixed-memory-speed",
        title="process_pair peak memory and DEM height iteration cost",
        labels=["enhancement", "area: performance"],
        text=fixed_body(
            "A single synthetic pair peaked at 16.5 times one complex64 array, and `terrain_heights` ran "
            "all 10 DEM passes even after every pixel had settled.",
            "Resampling (3.1 to 1.3 arrays, output identical), azimuth shift (5.0 to 1.6, identical), "
            "interferogram formation in row blocks, and geometric phase in place give 10.4 times one "
            "array. `terrain_heights` stops when every pixel has settled and is 3.5 times faster.",
            "Heights differ by at most 1.4 mm on a test DEM and the interferogram by at most 3e-3 rad on the "
            "synthetic pair; coherence and amplitude are bit-identical.",
            "Measured on synthetic data only (tracked in the open memory-model issue).",
            "fixed-memory-speed",
        ),
    ),
    dict(
        ref="fixed-baseline-output",
        title="process_pair did not return baselines or acquisition times",
        labels=["enhancement", "area: insar"],
        text=fixed_body(
            "Results had no perpendicular or temporal baseline, and no per-pixel geometry for downstream "
            "steps.",
            "`process_pair` returns `perpendicular_baseline_m` (from the orbits, at the AOI centre's "
            "zero-Doppler time) and `temporal_baseline_days`, `result.geometry` (per-pixel longitude, "
            "latitude, incidence, baseline), and `metadata` entries for acquisition times, signed mean "
            "baseline, incidence range and the topographic phase method.",
            "Baselines appear in the QA tables of real runs.",
            "Accuracy of the baselines against another processor has not been compared.",
            "fixed-baseline-output",
        ),
    ),
    dict(
        ref="fixed-phase-colormap",
        title="Analysis figures used a different cyclic colormap from the generation figures",
        labels=["bug", "documentation", "area: insar"],
        text=fixed_body(
            "Generation PNGs used the cyclic `hsv` colormap while the analysis figures used `twilight`, "
            "which made wrapped-phase plots hard to compare.",
            "`viz/insar.py` now uses `hsv` for wrapped phase. Unwrapped phase and signed quantities keep "
            "their non-cyclic maps.",
            "Change made in the plotting code; no automated image test exists.",
            "None known.",
            "fixed-phase-colormap",
        ),
    ),
]
for _c in CLOSED:
    _c["closed"] = True
ISSUES = ISSUES + CLOSED

# --------------------------------------------------------------------------------------------------- GitHub
def get_token():
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(name):
            return os.environ[name].strip()
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=15)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def api(method, path, token, payload=None):
    url = path if path.startswith("http") else API + path
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "pygeofetch-issue-creator")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
                return resp.status, (json.loads(raw) if raw else None), resp.headers
        except urllib.error.HTTPError as err:
            raw = err.read().decode("utf-8", "replace")
            if err.code in (403, 429) and attempt < 3:
                wait = int(err.headers.get("Retry-After") or 0)
                reset = err.headers.get("X-RateLimit-Reset")
                if not wait and err.headers.get("X-RateLimit-Remaining") == "0" and reset:
                    wait = max(1, int(reset) - int(time.time()) + 1)
                wait = wait or 60
                print(f"  rate limited, waiting {wait}s ...", flush=True)
                time.sleep(min(wait, 900))
                continue
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = {"message": raw}
            return err.code, parsed, err.headers
    return 0, {"message": "gave up"}, {}


def paged(path, token):
    page = 1
    out = []
    while True:
        sep = "&" if "?" in path else "?"
        status, data, _ = api("GET", f"{path}{sep}per_page=100&page={page}", token)
        if status != 200:
            raise RuntimeError(f"GET {path} failed: {status} {data}")
        out.extend(data)
        if len(data) < 100:
            return out
        page += 1


def norm(text):
    return " ".join(text.lower().split())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"owner/name (default {DEFAULT_REPO})")
    ap.add_argument("--apply", action="store_true", help="really create labels and issues (default: dry run)")
    ap.add_argument("--only", default="", help="comma-separated issue numbers from the dry-run list")
    ap.add_argument("--show", type=int, default=0, help="print the full text of issue number N and exit")
    ap.add_argument("--skip-closed", action="store_true", help="do not create the already-fixed (closed) issues")
    ap.add_argument("--only-closed", action="store_true", help="create only the already-fixed (closed) issues")
    ap.add_argument("--fixed-in", default="", help="URL of the commit, pull request or release that contains the fixes; added to each closed issue")
    args = ap.parse_args()

    if args.show:
        if not 1 <= args.show <= len(ISSUES):
            sys.exit(f"--show must be 1..{len(ISSUES)}")
        issue = ISSUES[args.show - 1]
        print(f"# {issue['title']}\nlabels: {', '.join(issue['labels'])}\n\n{issue['text']}")
        return

    wanted = None
    if args.only:
        try:
            wanted = {int(x) for x in args.only.split(",") if x.strip()}
        except ValueError:
            sys.exit("--only takes numbers like 1,4,7")
        bad = [n for n in wanted if not 1 <= n <= len(ISSUES)]
        if bad:
            sys.exit(f"no such issue number(s): {bad}")

    unknown = sorted({l for i in ISSUES for l in i["labels"]} - set(LABELS))
    if unknown:
        sys.exit(f"internal error: labels without a definition: {unknown}")

    token = get_token()
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"{mode}: repository {args.repo}, {len(ISSUES)} issues defined\n")
    if args.apply and not token:
        sys.exit("No token found. Set GITHUB_TOKEN (or log in with `gh auth login`) and run again.")

    existing_titles, existing_refs, existing_labels = set(), set(), set()
    can_label = True
    if token:
        status, repo, _ = api("GET", f"/repos/{args.repo}", token)
        if status != 200:
            sys.exit(f"Cannot read {args.repo}: {status} {repo.get('message') if repo else ''}. "
                     "Check the repository name and that the token has access to it.")
        perms = repo.get("permissions") or {}
        can_label = bool(perms.get("push") or perms.get("triage") or perms.get("maintain") or perms.get("admin"))
        if not can_label:
            print("NOTE: this token cannot manage labels on the repository; issues will be created without "
                  "labels.\n")
        for it in paged(f"/repos/{args.repo}/issues?state=all", token):
            if "pull_request" in it:
                continue
            existing_titles.add(norm(it["title"]))
            for line in (it.get("body") or "").splitlines():
                if "pygeofetch-issue-id:" in line:
                    existing_refs.add(line.split("pygeofetch-issue-id:")[1].replace("-->", "").strip())
        existing_labels = {l["name"] for l in paged(f"/repos/{args.repo}/labels", token)}
    else:
        print("NOTE: no token found, so existing issues cannot be checked. Fine for a dry run; --apply "
              "needs a token.\n")

    plan = []
    for n, issue in enumerate(ISSUES, 1):
        if wanted is not None and n not in wanted:
            continue
        if args.skip_closed and issue.get("closed"):
            continue
        if args.only_closed and not issue.get("closed"):
            continue
        dup = issue["ref"] in existing_refs or norm(issue["title"]) in existing_titles
        plan.append((n, issue, dup))
        tag = "  [then closed]" if issue.get("closed") else ""
        print(f"{n:2d}. {'SKIP (exists)' if dup else 'create       '}  {issue['title']}{tag}")
        print(f"      labels: {', '.join(issue['labels'])}")

    todo = [p for p in plan if not p[2]]
    print(f"\n{len(todo)} to create, {len(plan) - len(todo)} already exist.")
    if not args.apply:
        print("Nothing was changed. Use --show N to read an issue, then --apply to create them.")
        return
    if not todo:
        return

    if can_label:
        needed = sorted({l for _, i, _ in todo for l in i["labels"]} - existing_labels)
        for name in needed:
            color, desc = LABELS[name]
            status, data, _ = api("POST", f"/repos/{args.repo}/labels", token,
                                  {"name": name, "color": color, "description": desc})
            print(f"label '{name}': {'created' if status == 201 else f'not created ({status})'}")

    created = failed = 0
    for n, issue, _ in todo:
        text = issue["text"]
        if issue.get("closed"):
            note = "Fixed in the maintainers' development code." + (f" See {args.fixed_in}" if args.fixed_in else "")
            text = text.replace(FIXED_NOTE, note)
        payload = {"title": issue["title"], "body": text}
        if can_label:
            payload["labels"] = issue["labels"]
        status, data, _ = api("POST", f"/repos/{args.repo}/issues", token, payload)
        if status == 201:
            created += 1
            if issue.get("closed"):
                cs, cd, _ = api("PATCH", f"/repos/{args.repo}/issues/{data['number']}", token,
                                {"state": "closed", "state_reason": "completed"})
                if cs != 200:
                    print(f"         created but NOT closed ({cs}): {(cd or {}).get('message')}")
            print(f"created #{data['number']}: {issue['title']}\n         {data['html_url']}")
        else:
            failed += 1
            print(f"FAILED ({status}) {issue['title']}: {(data or {}).get('message')}")
        time.sleep(PAUSE_BETWEEN_CREATES_S)
    print(f"\nDone: {created} created, {failed} failed. Re-running is safe; existing issues are skipped.")


if __name__ == "__main__":
    main()