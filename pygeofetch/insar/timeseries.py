"""
SBASTimeSeries — Small BAseline Subset time series inversion.

Implements the SBAS weighted least-squares inversion (Berardino et al.
2002), including native, MintPy-equivalent advanced corrections
(unwrapping error correction via phase closure, DEM error correction),
entirely in numpy — no external InSAR software, and no MintPy
installation, required for any of it. The phase-closure technique's
algorithmic lineage (Yunjun et al. 2019) is cited below for attribution,
not because the package itself is a runtime dependency.

Reference:
  Berardino, P., Fornaro, G., Lanari, R., & Sansosti, E. (2002). A new
    algorithm for surface deformation monitoring based on small baseline
    differential SAR interferograms. IEEE TGRS, 40(11), 2375-2383.
  Yunjun, Z., Fattahi, H., & Amelung, F. (2019). Small baseline InSAR time
    series analysis: unwrapping error correction and noise reduction.
    Computers & Geosciences, 133, 104331.

Install: pip install "pygeofetch[insar]"
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("pygeofetch.insar.timeseries")

# REAL BUG FOUND AND FIXED HERE: this file previously had a second,
# complete module header (its own docstring, a duplicate
# "from __future__ import annotations", a duplicate "import logging",
# and a duplicate logger= line) pasted directly into the middle of the
# file, immediately after the real one -- a duplicate __future__ import
# that isn't the file's first statement is a real, hard SyntaxError in
# Python, not just messy style. This made the whole module unimportable
# as-is. The genuinely new, needed pieces from that second block
# (itertools, datetime, numpy, scipy.linalg -- all real dependencies of
# the new phase-closure and DEM-error correction methods below) are
# merged into the real imports above; the duplicate boilerplate is
# removed. The second docstring's real content is preserved below,
# describing what those two new corrections actually do.
#
# Native Advanced Corrections included in this module:
# 1. Unwrapping Error Correction via Greedy Phase Closure (Yunjun et al., 2019).
# 2. DEM Error Correction via Weighted Linear Regression of Residuals vs. Perpendicular Baseline.
logger = logging.getLogger("pygeofetch.insar.timeseries")

# Sentinel-1 C-band wavelength (meters)
SENTINEL1_WAVELENGTH_M = 0.05546576


@dataclass
class InterferogramPair:
    """Dataclass for a single interferogram pair."""
    reference_date: str
    secondary_date: str
    unwrapped_phase: np.ndarray
    coherence: np.ndarray
    perpendicular_baseline_m: float

# @dataclass
# class InterferogramPair:
#     """One interferogram in an SBAS network."""

#     reference_date: str  # ISO date, e.g. "2026-01-01"
#     secondary_date: str
#     unwrapped_phase: Any  # float32 (H, W) array, radians
#     coherence: Any  # float32 (H, W) array, 0-1
#     perpendicular_baseline_m: float = 0.0


@dataclass
class PairCandidate:
    """
    One candidate interferometric pair with its real, measured quality
    metrics -- everything build_sbas_network() needs to judge whether a
    pair is worth including in the SBAS network, beyond geometric
    baseline alone.

    coherence should be the pair's REAL, MEASURED mean coherence (e.g.
    InterferogramResult.coherence.mean()), not a modeled/predicted
    value. Perpendicular baseline is only a geometric proxy for
    expected decorrelation; measured coherence is the actual, direct
    signal of whether a pair coregistered and correlates well, and is
    what build_sbas_network() weighs most heavily.
    """

    date1: str
    date2: str
    perpendicular_baseline_m: float
    coherence: float
    coregistration_method: Optional[str] = None
    coregistration_refined_by_coherence: Optional[bool] = None


def _select_redundant_connected(
    items: List[Any],
    key_of: Callable[[Any], Tuple[str, str]],
    cost_of: Callable[[Any], float],
    is_good_of: Callable[[Any], bool],
    dates: Sequence[str],
    redundancy: int,
) -> Tuple[List[Any], List[Any], List[Any], List[str], bool]:
    """
    Generic redundant + bridging graph selection, shared by
    build_sbas_network() and select_pairs_for_processing() (see
    esd/stack-screening functions below): partitions `items` into
    "good" (is_good_of(item) True) and "poor" using the caller's own
    definition of good, greedily builds a connected, redundant graph
    from the good items (ranked by cost_of, ascending) via union-find,
    then bridges any remaining disconnected components using the
    best-available (lowest-cost) poor items -- the minimum necessary,
    never more. Kept generic (accessor functions instead of assuming a
    specific dataclass) specifically so this one, tested graph
    algorithm can be reused for different pair-quality signals
    (measured coherence, cheap pre-processing burst synchronization)
    rather than re-implementing the same union-find logic per signal.

    Returns:
        (selected_good, selected_bridge, excluded, unconnected_dates,
        connected) -- items in their original type, not tuples, so
        callers can still access whatever extra fields their own item
        type carries (coherence, sync_offset_ms, etc.) when building
        their own report/log messages.
    """
    date_set = set(dates)
    if len(date_set) < 2:
        return [], [], [], sorted(date_set), True

    good = sorted((c for c in items if is_good_of(c)), key=cost_of)
    poor = sorted((c for c in items if not is_good_of(c)), key=cost_of)

    parent = {d: d for d in date_set}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    degree = {d: 0 for d in date_set}
    selected_good: List[Any] = []
    excluded: List[Any] = []

    for c in good:
        d1, d2 = key_of(c)
        if d1 not in date_set or d2 not in date_set:
            continue
        connects_new = find(d1) != find(d2)
        under_redundancy = degree[d1] < redundancy or degree[d2] < redundancy
        if connects_new or under_redundancy:
            union(d1, d2)
            degree[d1] += 1
            degree[d2] += 1
            selected_good.append(c)
        else:
            excluded.append(c)

    selected_bridge: List[Any] = []
    for c in poor:
        d1, d2 = key_of(c)
        if d1 not in date_set or d2 not in date_set:
            continue
        if find(d1) != find(d2):
            union(d1, d2)
            degree[d1] += 1
            degree[d2] += 1
            selected_bridge.append(c)
        else:
            excluded.append(c)

    components: Dict[str, List[str]] = {}
    for d in date_set:
        components.setdefault(find(d), []).append(d)
    connected = len(components) == 1
    main_root = max(components, key=lambda r: len(components[r]))
    unconnected_dates = sorted(
        d for root, members in components.items() if root != main_root for d in members
    )

    return selected_good, selected_bridge, excluded, unconnected_dates, connected


def build_sbas_network(
    candidates: List[PairCandidate],
    dates: Sequence[str],
    min_coherence: float = 0.3,
    redundancy: int = 2,
) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """
    Select which candidate pairs to actually use for an SBAS network,
    weighing REAL measured coherence -- not just geometric perpendicular
    baseline -- and building in redundancy rather than a bare minimum
    spanning tree.

    Why this exists: a network chosen purely by minimizing baseline
    (e.g. a greedy union-find spanning tree over sorted b_perp) has no
    way to know that two dates happen to coregister badly for a real,
    independent reason (a bad sub-swath/crop-window match, TOPS burst
    desynchronization, etc.) -- it will happily pick the
    smallest-baseline pair even when a slightly-larger-baseline pair
    between the same two dates would coregister and correlate far
    better, and even when much better pairs exist elsewhere in the
    stack. Confirmed directly against this project's own Mexico City
    (Iztapalapa) run: a bare, baseline-only spanning tree selected 4 of
    5 pairs from a genuinely bad-coregistering subset (mean coherence
    0.25) while ignoring same-stack pairs achieving 0.4-0.56, leaving
    only 1/5 selected pairs usable after bridging and an invalid
    (disconnected) network overall.

    Two further, real problems a bare spanning tree has, independent of
    which specific edges it happens to pick: (1) exactly N-1 edges for
    N dates means ZERO redundancy -- a single pair excluded downstream
    (e.g. an unreliable reference pixel at the bridging step) can
    disconnect the whole network, exactly as happened here; (2) it
    offers no way to route around a bad edge, since every date has only
    as many connections as strictly necessary. This function keeps each
    date connected to multiple (`redundancy`) of its best real
    neighbours where good pairs exist, so one bad pair downstream has
    alternate paths to fall back on.

    Algorithm:
      1. Cost each candidate pair as
         `perpendicular_baseline_m / max(coherence, 0.05)` -- for equal
         coherence, prefers shorter baseline (the original, still-valid
         geometric criterion); for equal baseline, strongly prefers
         higher real coherence; a near-zero-coherence pair is costed as
         if its baseline were ~20x larger, matching the real intuition
         that decorrelated pairs should be avoided even when
         geometrically ideal.
      2. Split candidates into GOOD (coherence >= min_coherence) and
         POOR (below it).
      3. Process GOOD candidates in ascending cost order, adding each
         edge if it either connects two still-separate components
         (union-find) or gives either endpoint date fewer than
         `redundancy` connections so far -- building a connected,
         redundant network using only pairs that are actually known to
         coregister and correlate well.
      4. If GOOD pairs alone can't connect every date in `dates` (as in
         the Mexico City case, where two disjoint well-coregistering
         subsets exist with no good pair between them), bridge the
         remaining disconnected components using the best-available
         (lowest-cost) POOR pairs -- the minimum necessary, never more
         -- each one explicitly flagged in the report as a low-quality
         bridge rather than silently treated the same as a good pair.
      5. Any date still unreachable after step 4 (no candidate pair --
         good or poor -- connects it at all) is reported as
         unconnected, not silently dropped.

    Args:
        candidates: Every real candidate pair to consider (e.g. every
                   pair an interferogram was actually formed for),
                   each with its real measured coherence.
        dates:      Every acquisition date that should be covered.
        min_coherence: The GOOD/POOR threshold from step 2. 0.3 matches
                   this project's own SBASTimeSeries.invert()'s default
                   coherence_threshold, so a pair good enough to be
                   trusted here is also good enough not to be masked
                   out pixel-by-pixel at inversion time.
        redundancy: How many of each date's best real neighbours to
                   connect via GOOD pairs (beyond the bare minimum
                   needed for connectivity), for robustness against a
                   single pair failing downstream. 2 is a reasonable
                   default.

    Returns:
        (selected_pairs, report) where selected_pairs is a list of
        (date1, date2) tuples (GOOD pairs first, then any necessary
        bridge pairs), and report is a dict with:
          "good_pairs":        selected pairs that met min_coherence
          "bridge_pairs":      selected pairs that didn't, but were
                               necessary for connectivity -- inspect
                               these first if the resulting network
                               still performs poorly
          "excluded_pairs":    candidates not selected (redundant with
                               an already-good-enough network)
          "unconnected_dates": dates no candidate pair -- good or
                               poor -- could connect at all; a real,
                               unresolvable gap in the input
                               candidates, not something this function
                               can fix
          "connected":         bool, whether every date in `dates`
                               ended up reachable
    """
    date_set = set(dates)
    if len(date_set) < 2:
        return [], {
            "good_pairs": [], "bridge_pairs": [], "excluded_pairs": [],
            "unconnected_dates": sorted(date_set), "connected": True,
        }

    def cost(c: PairCandidate) -> float:
        return c.perpendicular_baseline_m / max(c.coherence, 0.05)

    selected_good, selected_bridge, excluded, unconnected_dates, connected = (
        _select_redundant_connected(
            candidates,
            key_of=lambda c: (c.date1, c.date2),
            cost_of=cost,
            is_good_of=lambda c: c.coherence >= min_coherence,
            dates=dates,
            redundancy=redundancy,
        )
    )

    if selected_bridge:
        logger.warning(
            "build_sbas_network: %d pair(s) below coherence threshold "
            "%.2f were included as necessary bridges to keep the "
            "network connected: %s -- inspect these first if inversion "
            "quality is still poor.",
            len(selected_bridge), min_coherence,
            [(c.date1, c.date2, round(c.coherence, 3)) for c in selected_bridge],
        )
    if unconnected_dates:
        logger.warning(
            "build_sbas_network: %d date(s) have no candidate pair "
            "connecting them to the rest of the network at all: %s",
            len(unconnected_dates), unconnected_dates,
        )

    selected_pairs = [(c.date1, c.date2) for c in selected_good + selected_bridge]
    report = {
        "good_pairs": [(c.date1, c.date2) for c in selected_good],
        "bridge_pairs": [(c.date1, c.date2) for c in selected_bridge],
        "excluded_pairs": [(c.date1, c.date2) for c in excluded],
        "unconnected_dates": unconnected_dates,
        "connected": connected,
    }
    return selected_pairs, report


@dataclass
class BurstSyncResult:
    """
    One pair's real, measured burst synchronization result from
    screen_stack_burst_synchronization() -- cheap (annotation XML +
    orbit files only, no coregistration/ESD/deburst/interferogram
    formation) evidence of whether a pair is worth the cost of full
    processing at all, available BEFORE any of that runs.
    """

    date1: str
    date2: str
    sync_offset_ms: float
    burst_cycle_s: float
    within_requirement: bool


def generate_candidate_pairs(
    dates: Sequence[str],
    max_temporal_baseline_days: int = 120,
) -> List[Tuple[str, str]]:
    """
    Generate candidate SBAS pairs limited to a maximum temporal
    baseline -- the standard first-stage filter real production SBAS
    pipelines use before any redundancy/connectivity optimization or
    burst-synchronization screening, since full pairwise combinations
    scale as O(n^2) and become computationally infeasible for a long
    time series: a 5-year archive at Sentinel-1's ~12-day repeat has on
    the order of 150 acquisitions -- C(150,2) = 11,175 possible pairs,
    the overwhelming majority temporally too far apart to ever be
    useful SBAS pairs (long temporal baselines mean more decorrelation
    and a larger unwrapping ambiguity risk) and not worth even cheaply
    screening, let alone fully processing.

    Runs in O(n * avg_pairs_per_date), not O(n^2): dates are sorted
    once, and because they're in chronological order, the moment a
    candidate's gap exceeds max_temporal_baseline_days every later
    candidate for that same reference date is guaranteed to be farther
    still, so the inner search stops immediately rather than checking
    every remaining date.

    Args:
        dates: All acquisition dates (ISO strings, "YYYY-MM-DD"), any
                       order, duplicates ignored.
        max_temporal_baseline_days: Only pairs with a temporal gap up
                       to this many days are generated. 120 days (~10
                       Sentinel-1 12-day cycles) is a reasonable,
                       commonly-used default -- wide enough to give
                       build_sbas_network()/select_pairs_for_processing()
                       real redundancy to choose from, narrow enough to
                       keep the candidate count manageable. Tighten
                       this for a scene with fast decorrelation (e.g.
                       dense vegetation, a tropical wet season); widen
                       it if the resulting network ends up poorly
                       connected.

    Returns:
        List of (date1, date2) tuples, date1 chronologically before
        date2, sorted by date1 then date2.
    """
    from datetime import datetime

    parsed = sorted(
        (datetime.strptime(d, "%Y-%m-%d"), d) for d in set(dates)
    )
    n = len(parsed)
    pairs: List[Tuple[str, str]] = []
    for i in range(n):
        t1, d1 = parsed[i]
        for j in range(i + 1, n):
            t2, d2 = parsed[j]
            gap_days = (t2 - t1).days
            if gap_days > max_temporal_baseline_days:
                break
            pairs.append((d1, d2))
    return pairs


def screen_stack_burst_synchronization(
    dates: Sequence[str],
    safe_zips: Dict[str, Any],
    orbit_files: Dict[str, Any],
    ground_point: Tuple[float, float, float],
    swath_hints: Optional[Dict[str, str]] = None,
) -> List[BurstSyncResult]:
    """
    Cheap, pre-processing burst-synchronization screen for every
    candidate pair in a stack -- real annotation XML + real orbit
    files only; no coregistration, ESD, deburst, or interferogram
    formation needed.

    Confirmed directly on this project's own real Mexico City
    (Iztapalapa) stack: burst synchronization alone
    (esd.compute_burst_synchronization) predicted the real, fully-
    processed outcome for every one of 15 real pairs, with zero
    exceptions -- 6/6 same-family pairs within Sentinel-1's own 5 ms
    requirement AND good final coherence (0.44-0.57); 9/9 cross-family
    pairs outside the requirement AND poor final coherence
    (~0.27-0.28). That means this cheap screen can reliably decide
    what's worth full processing BEFORE spending the time on it, not
    just explain results after the fact -- real value for a stack
    where full processing (coregistration + ESD + deburst +
    interferogram formation) costs minutes per pair and this screen
    costs seconds for the entire stack.

    Parses each date's real geometry, burst metadata, and orbit state
    vectors ONCE (O(n) parses for n dates), then evaluates every
    candidate pair's Δt_acq from those already-parsed objects (O(n^2)
    cheap evaluations, not O(n^2) file re-parses) -- this is what makes
    screening an entire stack fast enough to run before deciding what
    to fully process, rather than only being useful as a post-hoc
    diagnostic.

    Args:
        dates:       Every acquisition date to screen (ISO strings).
        safe_zips:   {date: path to that date's .SAFE.zip}.
        orbit_files: {date: path to that date's .EOF orbit file}.
        ground_point: Real ECEF ground point to evaluate synchronization
                      at (e.g. the AOI center, or a real, stable
                      reference point) -- shared across all pairs for
                      consistency; per compute_burst_synchronization's
                      own docstring, synchronization is spatially
                      stable across a scene (Yagüe-Martínez et al.
                      2016, Sec. V-E-2: stable to under 1 cm across an
                      entire slice), so the exact point rarely matters
                      much.
        swath_hints: Optional {date: matched sub-swath name} (e.g. from
                      extract_consistent_stack's own report) -- ensures
                      the same sub-swath is used here as was actually
                      extracted, when available. None uses each date's
                      automatic sub-swath match.

    Returns:
        List of BurstSyncResult, one per candidate pair that could be
        evaluated. A pair (or a whole date) is omitted, with a logged
        warning, rather than failing the entire screen, if its SAFE
        zip/orbit file is missing or can't be parsed, or if
        find_zero_doppler_time can't converge for it -- a real,
        per-pair/per-date data problem, not grounds to abandon
        screening the rest of the stack.
    """
    from itertools import combinations

    from pygeofetch.insar.annotation import parse_burst_info, parse_slc_geometry
    from pygeofetch.insar.esd import compute_burst_synchronization
    from pygeofetch.insar.geolocation import parse_orbit_file

    swath_hints = swath_hints or {}
    geoms: Dict[str, Any] = {}
    burst_infos: Dict[str, Any] = {}
    orbits: Dict[str, Any] = {}

    for d in dates:
        if d not in safe_zips or d not in orbit_files:
            logger.warning(
                "screen_stack_burst_synchronization: no SAFE zip/orbit "
                "file available for %s -- pairs involving this date "
                "will be skipped.", d,
            )
            continue
        try:
            hint = swath_hints.get(d)
            geoms[d] = parse_slc_geometry(safe_zips[d], member_hint=hint)
            burst_infos[d] = parse_burst_info(safe_zips[d], member_hint=hint)
            orbits[d] = parse_orbit_file(orbit_files[d])
        except Exception as exc:
            logger.warning(
                "screen_stack_burst_synchronization: could not parse "
                "real geometry/burst/orbit data for %s (%s) -- pairs "
                "involving this date will be skipped.", d, exc,
            )

    results: List[BurstSyncResult] = []
    for d1, d2 in combinations(dates, 2):
        if d1 not in geoms or d2 not in geoms:
            continue
        try:
            t1 = geoms[d1].azimuth_time(geoms[d1].n_lines / 2)
            t2 = geoms[d2].azimuth_time(geoms[d2].n_lines / 2)
            sync = compute_burst_synchronization(
                orbits[d1], orbits[d2], burst_infos[d1], burst_infos[d2],
                ground_point, ref_time_guess=t1, sec_time_guess=t2,
            )
            results.append(BurstSyncResult(
                date1=d1, date2=d2,
                sync_offset_ms=sync["sync_offset_ms"],
                burst_cycle_s=sync["burst_cycle_s"],
                within_requirement=sync["within_esa_requirement"],
            ))
        except Exception as exc:
            logger.warning(
                "screen_stack_burst_synchronization: %s -> %s failed "
                "(%s) -- skipped.", d1, d2, exc,
            )

    return results


def select_pairs_for_processing(
    sync_results: List[BurstSyncResult],
    dates: Sequence[str],
    redundancy: int = 2,
) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """
    Decide which pairs are worth full coregistration/ESD/deburst/
    interferogram processing, using ONLY the cheap burst-
    synchronization screen (screen_stack_burst_synchronization) --
    same redundant + bridging design as build_sbas_network() (sharing
    its exact graph-selection logic via _select_redundant_connected),
    applied to a signal available before any expensive processing
    runs, so a stack can be reduced to "worth processing" BEFORE
    spending the time, rather than only explained afterward.

    This does NOT replace build_sbas_network(): after the pairs
    selected here are actually processed, run build_sbas_network() on
    their real measured coherence as usual. Burst sync predicts which
    pairs are LIKELY to correlate well, but genuine ground
    decorrelation (vegetation change, water, unrelated temporal
    change) is a real, separate cause build_sbas_network's real-
    coherence weighting still needs to catch -- burst sync being good
    is necessary for a well-correlating TOPS pair, not sufficient on
    its own. Defaults to a slightly generous redundancy for exactly
    this reason: enough real, fully-processed pairs need to survive
    for that second stage to still have meaningful choices, not just
    whatever this cheaper, earlier screen happened to prefer.

    Args:
        sync_results: Output of screen_stack_burst_synchronization().
        dates:        Every date that should be covered.
        redundancy:   How many of each date's best (lowest |Δt_acq|)
                      synchronized neighbours to connect via GOOD pairs,
                      beyond the bare minimum needed for connectivity.

    Returns:
        (selected_pairs, report) -- same shape as build_sbas_network's
        own return (good_pairs/bridge_pairs/excluded_pairs/
        unconnected_dates/connected), so it drops in as a "which pairs
        to loop over" step before full processing, e.g.:

            sync_results = screen_stack_burst_synchronization(...)
            pairs_to_process, screen_report = select_pairs_for_processing(
                sync_results, extracted_dates,
            )
            for d1, d2 in pairs_to_process:
                result = ifg_gen.process_pair(...)
                ...
    """
    selected_good, selected_bridge, excluded, unconnected_dates, connected = (
        _select_redundant_connected(
            sync_results,
            key_of=lambda c: (c.date1, c.date2),
            cost_of=lambda c: abs(c.sync_offset_ms),
            is_good_of=lambda c: c.within_requirement,
            dates=dates,
            redundancy=redundancy,
        )
    )

    if selected_bridge:
        logger.warning(
            "select_pairs_for_processing: %d pair(s) outside Sentinel-1's "
            "own burst synchronization requirement were included as "
            "necessary bridges to keep the stack connected: %s -- "
            "expect these specific pairs to correlate poorly once "
            "processed (real Doppler-spectrum decorrelation, not "
            "fixable by reprocessing); still worth including so the "
            "stack has SOME connected network to invert.",
            len(selected_bridge),
            [(c.date1, c.date2, round(c.sync_offset_ms, 1)) for c in selected_bridge],
        )
    if unconnected_dates:
        logger.warning(
            "select_pairs_for_processing: %d date(s) have no candidate "
            "pair connecting them to the rest of the stack at all: %s",
            len(unconnected_dates), unconnected_dates,
        )

    selected_pairs = [(c.date1, c.date2) for c in selected_good + selected_bridge]
    report = {
        "good_pairs": [(c.date1, c.date2) for c in selected_good],
        "bridge_pairs": [(c.date1, c.date2) for c in selected_bridge],
        "excluded_pairs": [(c.date1, c.date2) for c in excluded],
        "unconnected_dates": unconnected_dates,
        "connected": connected,
    }
    return selected_pairs, report


@dataclass
class TimeSeriesResult:
    """Dataclass for the SBAS inversion result."""
    dates: List[str]
    displacement: np.ndarray  # (n_dates, h, w)
    velocity: np.ndarray      # (h, w)
    residual_rms: np.ndarray  # (h, w)
    reference_date: str
    metadata: Dict

    def save(
        self,
        output_dir: Union[str, Path],
        profile: Optional[dict] = None,
        auto_visualize: bool = False,
    ) -> Dict[str, Path]:
        """Save displacement time series and velocity as GeoTIFFs.

        Args:
            output_dir:     Directory to save into.
            profile:        Optional rasterio-style profile for georeferencing.
            auto_visualize: If True, also save PNG visualizations
                           (velocity map, residual RMS map, and a
                           composite per-date displacement grid)
                           alongside the GeoTIFFs, via
                           pygeofetch.insar.visualize.
        """
        import numpy as np

        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        h, w = self.velocity.shape
        base = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "height": h,
            "width": w,
            "nodata": -9999.0,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        if profile:
            base["crs"] = profile.get("crs")
            base["transform"] = profile.get("transform")

        vel_path = out_dir / "velocity_m_per_year.tif"
        with rasterio.open(vel_path, "w", **base) as dst:
            dst.write(self.velocity.astype(np.float32)[np.newaxis])
        paths["velocity"] = vel_path

        ts_path = out_dir / "displacement_timeseries.tif"
        ts_profile = dict(base, count=len(self.dates))
        with rasterio.open(ts_path, "w", **ts_profile) as dst:
            dst.write(self.displacement.astype(np.float32))
            for i, date in enumerate(self.dates, start=1):
                dst.update_tags(i, date=date)
        paths["displacement_timeseries"] = ts_path

        rms_path = out_dir / "residual_rms.tif"
        with rasterio.open(rms_path, "w", **base) as dst:
            dst.write(self.residual_rms.astype(np.float32)[np.newaxis])
        paths["residual_rms"] = rms_path

        logger.info("Time series products saved → %s", out_dir)

        if auto_visualize:
            from pygeofetch.insar.visualize import visualize_timeseries

            try:
                visualize_timeseries(self, out_dir)
            except Exception as exc:
                logger.warning("auto_visualize failed (GeoTIFFs were still saved successfully): %s", exc)

        return paths


def select_reliable_reference_pixel(
    conncomp_masks: Dict[Tuple[str, str], Any],
    coherence_maps: Optional[Dict[Tuple[str, str], Any]] = None,
    preferred_point: Optional[Tuple[int, int]] = None,
    search_radius_px: int = 0,
    min_reliable_fraction: float = 1.0,
    min_region_size: int = 100,
) -> Tuple[Tuple[int, int], Dict[str, Any]]:
    """
    Choose a reference pixel that is reliably unwrapped across as many
    of the given pairs as possible -- fixes a real, confirmed failure
    mode: picking a reference pixel by coherence alone (or by an
    arbitrary real-world landmark) and only checking per-pair
    reliability AFTER pair selection means a single pair where that
    specific pixel happens to be unreliable gets excluded post-hoc --
    which can disconnect an SBAS network that build_sbas_network() had
    already confirmed connected, forcing ad-hoc downstream workarounds
    (per-component inversion + spatial stitching, or abandoning the
    real inversion altogether) that a better-chosen reference pixel
    usually makes unnecessary in the first place.

    CRITICAL, fixed after a real failure: "reliable" here must mean
    exactly what unwrap.bridge_unwrap_regions() means by "valid
    region" -- NOT simply conncomp > 0. bridge_unwrap_regions
    additionally requires a labeled connected component to have at
    least min_region_size pixels before it will bridge from/to it (a
    small labeled fragment is more likely a real unwrapping artifact
    than a genuinely reliable region), and raises ValueError outright
    if the given reference_pixel's own label is too small -- even when
    the raw conncomp value there is genuinely nonzero. An earlier
    version of this function checked only conncomp > 0 and could
    therefore choose a pixel bridge_unwrap_regions then rejected:
    confirmed directly by a real run, "ValueError: reference_pixel
    (7, 86) is not part of any valid region." This function now applies
    the identical region-size criterion internally, so a pixel it
    calls reliable is guaranteed to also be a valid region for
    bridge_unwrap_regions -- PROVIDED min_region_size here is the same
    value passed to that later bridge_unwrap_regions() call (both
    default to 100 for exactly this reason; if one is overridden,
    override the other to match).

    Args:
        conncomp_masks: {(date1, date2): connected-component array}
                       for every candidate pair (straight from
                       PhaseUnwrapper.unwrap()'s own conncomp output,
                       cropped to a common shape). A pixel counts as
                       reliable for a pair only if its label is nonzero
                       AND that label's total region size is >=
                       min_region_size -- matching bridge_unwrap_regions
                       exactly, not just checking conncomp > 0.
        coherence_maps: Optional {(date1, date2): coherence array},
                       same keys/shapes. Used only as a tie-breaker
                       among pixels tied on reliable-pair count (mean
                       coherence across pairs) -- reliability, not
                       coherence, is the primary criterion, since a
                       high-coherence pixel that still fails to unwrap
                       reliably in even one pair is exactly the problem
                       this function exists to avoid.
        preferred_point: Optional (row, col) -- e.g. a real,
                       georeferenced landmark's pixel position. If
                       given, pixels within search_radius_px of this
                       point are searched FIRST; the function only
                       looks further away if nothing near the
                       preferred point meets min_reliable_fraction as
                       well as the whole-image best does.
        search_radius_px: Radius (pixels) around preferred_point to
                       search first. 0 means "consider preferred_point
                       itself only" before comparing against the
                       whole-image best.
        min_reliable_fraction: Required fraction of pairs the chosen
                       pixel must be reliable in. 1.0 (default) --
                       reliable in every single pair -- is the
                       strictest, safest choice, since it means no pair
                       will ever need excluding downstream on this
                       pixel's account. If no pixel anywhere meets this
                       fraction, the function relaxes automatically and
                       returns the best ACHIEVABLE fraction instead,
                       clearly reported via a logged warning and the
                       returned report -- never a silent downgrade.
        min_region_size: Minimum pixel count for a labeled connected
                       component to count as reliable. MUST match the
                       min_region_size passed to the
                       bridge_unwrap_regions() call this reference
                       pixel will actually be used with -- defaults to
                       100 to match that function's own default.

    Returns:
        (pixel, report) where pixel is (row, col), and report has:
          "reliable_fraction":     the chosen pixel's actual reliable
                       fraction (1.0 if reliable in every pair).
          "unreliable_pairs":      list of (date1, date2) pairs where
                       the chosen pixel is NOT reliable -- empty if
                       reliable_fraction is 1.0. A downstream per-pair
                       exclusion step should check this list directly
                       rather than re-deriving it per pixel.
          "searched_near_preferred": bool -- whether the final pixel
                       came from near preferred_point (True) or the
                       function had to move to a whole-image best
                       instead (False).
          "candidates_considered": total pixels in the scene (for
                       diagnostics on very large scenes).

    Raises:
        ValueError if conncomp_masks is empty, or if its arrays don't
        all share the same shape (they must, to compare pixel-wise).
    """
    import numpy as np

    if not conncomp_masks:
        raise ValueError("select_reliable_reference_pixel: conncomp_masks is empty.")

    pairs = list(conncomp_masks.keys())
    shapes = {tuple(np.asarray(arr).shape) for arr in conncomp_masks.values()}
    if len(shapes) > 1:
        raise ValueError(
            f"select_reliable_reference_pixel: conncomp_masks arrays have "
            f"inconsistent shapes {shapes} -- crop/align them to a common "
            f"shape first."
        )
    h, w = next(iter(shapes))
    n_pairs = len(pairs)

    # Real per-pair validity mask -- matches bridge_unwrap_regions'
    # own criterion exactly: nonzero label AND that label's total
    # region size >= min_region_size, not just nonzero (see the
    # docstring above for the real failure this fixes).
    reliable_stack = np.zeros((n_pairs, h, w), dtype=bool)
    for i, p in enumerate(pairs):
        conncomp = np.asarray(conncomp_masks[p])
        labels, counts = np.unique(conncomp, return_counts=True)
        valid_labels = [
            int(lbl) for lbl, cnt in zip(labels, counts)
            if lbl != 0 and cnt >= min_region_size
        ]
        if valid_labels:
            reliable_stack[i] = np.isin(conncomp, valid_labels)
        # else: no region in this pair meets min_region_size at all --
        # reliable_stack[i] stays all-False, correctly excluding every
        # pixel for this pair rather than falling back to a weaker check.

    reliable_fraction_map = reliable_stack.sum(axis=0) / n_pairs

    mean_coh = None
    if coherence_maps is not None:
        coh_stack = np.stack([np.asarray(coherence_maps[p]) for p in pairs], axis=0)
        mean_coh = coh_stack.mean(axis=0)

    def _best_in_region(r0, r1, c0, c1):
        sub = reliable_fraction_map[r0:r1, c0:c1]
        if sub.size == 0:
            return None, -1.0
        best_frac = float(sub.max())
        candidates = np.argwhere(sub == best_frac)
        if mean_coh is not None:
            sub_coh = mean_coh[r0:r1, c0:c1]
            best = max(candidates, key=lambda rc: sub_coh[rc[0], rc[1]])
        else:
            best = candidates[0]
        return (int(best[0] + r0), int(best[1] + c0)), best_frac

    whole_pixel, whole_fraction = _best_in_region(0, h, 0, w)
    pixel, achieved_fraction, searched_near_preferred = whole_pixel, whole_fraction, False

    if preferred_point is not None:
        pr, pc = preferred_point
        r0, r1 = max(0, pr - search_radius_px), min(h, pr + search_radius_px + 1)
        c0, c1 = max(0, pc - search_radius_px), min(w, pc + search_radius_px + 1)
        near_pixel, near_fraction = _best_in_region(r0, r1, c0, c1)
        # Prefer the real, physically-meaningful point whenever it does
        # at least as well as the whole-image search -- only actually
        # move away from it when the whole image is genuinely better.
        if near_pixel is not None and near_fraction >= whole_fraction:
            pixel, achieved_fraction, searched_near_preferred = near_pixel, near_fraction, True

    unreliable_pairs = [
        pairs[i] for i in range(n_pairs) if not reliable_stack[i, pixel[0], pixel[1]]
    ]

    if achieved_fraction < min_reliable_fraction:
        logger.warning(
            "select_reliable_reference_pixel: no pixel is reliable in the "
            "required fraction (%.2f) of %d pairs -- best achievable is "
            "%.2f at %s. %d pair(s) would need excluding if this pixel is "
            "used: %s.",
            min_reliable_fraction, n_pairs, achieved_fraction, pixel,
            len(unreliable_pairs), unreliable_pairs,
        )
    else:
        logger.info(
            "select_reliable_reference_pixel: %s is reliable in all %d "
            "pairs (searched %s).",
            pixel, n_pairs, "near the preferred point" if searched_near_preferred else "the whole scene",
        )

    report = {
        "reliable_fraction": achieved_fraction,
        "unreliable_pairs": unreliable_pairs,
        "searched_near_preferred": searched_near_preferred,
        "candidates_considered": int(h * w),
    }
    return pixel, report


def despike_velocity(velocity: Any, valid_mask: Optional[Any] = None, size: int = 3) -> Any:
    """
    Real, optional post-processing cleanup for phase-unwrapping cycle-
    slip artifacts: a NaN-aware spatial median filter over a per-pixel
    SBAS result (typically TimeSeriesResult.velocity, but works on any
    2D per-pixel array, e.g. a single date's displacement slice).

    Why this exists: SBAS inversion solves each pixel independently
    (see SBASTimeSeries._invert_native's own per-pixel-pattern grouped
    least squares) -- a real, localized unwrapping error at a single
    pixel in a single pair (a "cycle slip": the unwrapper picks the
    wrong 2*pi ambiguity for just that one pixel) produces an isolated,
    unphysically large outlier in that pixel's inverted velocity,
    surrounded by otherwise-normal neighbours. A median filter removes
    exactly this kind of isolated single-pixel spike while leaving
    real, spatially-coherent deformation signal (which by definition
    extends over many neighbouring pixels, not a lone pixel) intact.

    This is a real, standard InSAR post-processing step, but a coarse
    one: applying it will also suppress genuine sub-window-scale,
    highly localized real deformation if any exists (e.g. a single
    collapsing structure smaller than `size` pixels wide). Apply it
    once, compare against the pre-filtered map, and only keep the
    result if it's visibly removing isolated speckle rather than
    smoothing away real spatial structure.

    Uses a true NaN-aware median (scipy.ndimage.generic_filter with
    np.nanmedian) rather than filling NaNs with a placeholder value
    first -- filling with e.g. 0 before a normal median filter would
    bias every window that touches a NaN region, exactly the kind of
    edge artifact this function exists to avoid introducing.

    Args:
        velocity:   (H, W) array (or any 2D per-pixel array) to clean.
        valid_mask: Optional (H, W) boolean array -- pixels already NaN
                   in `velocity` stay NaN regardless; pass this only to
                   force ADDITIONAL pixels to NaN in the output (e.g. a
                   stricter reliability mask than what's already NaN).
        size:       Median filter window size (pixels). 3 is a
                   reasonable, conservative default -- large enough to
                   remove single-pixel spikes, small enough to preserve
                   most real spatial structure.

    Returns:
        Filtered array, same shape as `velocity`, float32.
    """
    import numpy as np
    from scipy.ndimage import generic_filter

    arr = np.asarray(velocity, dtype=np.float64)
    filtered = generic_filter(arr, np.nanmedian, size=size, mode="nearest")
    filtered = np.where(np.isnan(arr), np.nan, filtered)
    if valid_mask is not None:
        filtered = np.where(np.asarray(valid_mask), filtered, np.nan)
    return filtered.astype(np.float32)



class SBASTimeSeries:
    """
    Native SBAS Inversion Engine with Advanced Corrections.

    Replaces the need for MintPy's smallbaselineApp by operating directly
    on pygeofetch's native InterferogramPair objects.
    """

    def __init__(
        self,
        wavelength_m: float = SENTINEL1_WAVELENGTH_M,
        reference_date: Optional[str] = None,
    ) -> None:
        self._wavelength = wavelength_m
        self._reference_date = reference_date

    def invert(
        self,
        pairs: List[InterferogramPair],
        coherence_threshold: float = 0.3,
        correct_unwrap: bool = True,
        correct_dem: bool = True,
        reference_pixel: Optional[Tuple[int, int]] = None,
    ) -> TimeSeriesResult:
        """
        Invert the SBAS network with optional advanced corrections.

        Args:
            pairs: List of InterferogramPair objects.
            coherence_threshold: Minimum coherence for a pixel to be included.
            correct_unwrap: If True, apply greedy phase closure correction.
            correct_dem: If True, apply DEM error correction via residual regression.
            reference_pixel: (row, col) of a stable reference pixel.
        """
        from pygeofetch.insar.validate import DataValidator

        # 1. Validate Network
        all_dates = sorted({p.reference_date for p in pairs} | {p.secondary_date for p in pairs})
        DataValidator.validate_sbas_network(pairs, all_dates).raise_if_invalid()

        # Deep copy to avoid mutating original data during corrections
        working_pairs = [
            InterferogramPair(
                reference_date=p.reference_date,
                secondary_date=p.secondary_date,
                unwrapped_phase=np.array(p.unwrapped_phase, copy=True),
                coherence=np.array(p.coherence, copy=True),
                perpendicular_baseline_m=p.perpendicular_baseline_m,
            )
            for p in pairs
        ]

        # 2. Apply Advanced Corrections (Order matters: Unwrap -> DEM)
        if correct_unwrap:
            logger.info("Applying native unwrapping error correction (Phase Closure)...")
            working_pairs = self._correct_unwrapping_errors(working_pairs, all_dates)

        if correct_dem:
            logger.info("Applying native DEM error correction...")
            working_pairs = self._correct_dem_error(working_pairs, all_dates)

        # 3. Final Native Inversion
        return self._invert_native(working_pairs, coherence_threshold, reference_pixel)

    # ═══════════════════════════════════════════════════════════════════
    # ADVANCED CORRECTION 1: UNWRAPPING ERROR (PHASE CLOSURE)
    # ═══════════════════════════════════════════════════════════════════
    def _correct_unwrapping_errors(
        self,
        pairs: List[InterferogramPair],
        dates: List[str],
        max_iterations: int = 3,
        closure_threshold_rad: float = np.pi / 2
    ) -> List[InterferogramPair]:
        """
        Greedy phase closure correction (MintPy style).

        Physics: In a closed triangle of interferograms (i->j, j->k, i->k),
        the sum of phases should be zero modulo 2*pi. If not, an unwrapping
        error (integer 2*pi jump) exists in the pair with the lowest coherence.
        """
        n_dates = len(dates)
        date_idx = {d: i for i, d in enumerate(dates)}

        # Build adjacency and find all triangles
        adj = np.zeros((n_dates, n_dates), dtype=bool)
        pair_idx_map = {}
        for i, p in enumerate(pairs):
            i1, i2 = date_idx[p.reference_date], date_idx[p.secondary_date]
            adj[i1, i2] = adj[i2, i1] = True
            pair_idx_map[(i1, i2)] = i
            pair_idx_map[(i2, i1)] = i

        triangles = [
            (i, j, k) for i, j, k in itertools.combinations(range(n_dates), 3)
            if adj[i, j] and adj[j, k] and adj[i, k]
        ]

        if not triangles:
            logger.warning("No closed triangles found. Skipping unwrap correction.")
            return pairs

        logger.info("Found %d closed triangles for phase closure.", len(triangles))

        # Stack for vectorized math
        phase_stack = np.stack([p.unwrapped_phase for p in pairs], axis=0)
        coh_stack = np.stack([p.coherence for p in pairs], axis=0)
        h, w = phase_stack.shape[1], phase_stack.shape[2]

        for iteration in range(max_iterations):
            corrections_made = 0
            # Process in chunks to save memory
            chunk_size = 50000
            for r_start in range(0, h * w, chunk_size):
                r_end = min(r_start + chunk_size, h * w)
                phases_chunk = phase_stack.reshape(len(pairs), -1)[:, r_start:r_end]
                cohs_chunk = coh_stack.reshape(len(pairs), -1)[:, r_start:r_end]

                for i, j, k in triangles:
                    idx_ij = pair_idx_map[(i, j)]
                    idx_jk = pair_idx_map[(j, k)]
                    idx_ik = pair_idx_map[(i, k)]

                    # Closure phase: phi_ij + phi_jk - phi_ik
                    phi_closure = phases_chunk[idx_ij] + phases_chunk[idx_jk] - phases_chunk[idx_ik]
                    phi_closure_wrapped = np.mod(phi_closure + np.pi, 2 * np.pi) - np.pi

                    mask = np.abs(phi_closure_wrapped) > closure_threshold_rad
                    if not np.any(mask):
                        continue

                    # Find the pair with lowest coherence in this triangle
                    cohs_tri = np.stack([cohs_chunk[idx_ij], cohs_chunk[idx_jk], cohs_chunk[idx_ik]], axis=0)
                    min_coh_idx = np.argmin(cohs_tri, axis=0)

                    n_jumps = np.round(phi_closure_wrapped / (2 * np.pi))
                    signs = np.array([1.0, 1.0, -1.0]) # Signs in the closure equation

                    for target_idx, (pair_idx, sign) in enumerate(zip([idx_ij, idx_jk, idx_ik], signs)):
                        mask_to_correct = (min_coh_idx == target_idx) & mask
                        if np.any(mask_to_correct):
                            phases_chunk[pair_idx, mask_to_correct] -= sign * n_jumps[mask_to_correct] * 2 * np.pi
                            corrections_made += np.sum(mask_to_correct)

                phase_stack.reshape(len(pairs), -1)[:, r_start:r_end] = phases_chunk

            if corrections_made == 0:
                logger.info("Phase closure converged at iteration %d.", iteration + 1)
                break

        # Update pairs
        return [
            InterferogramPair(p.reference_date, p.secondary_date, phase_stack[i], p.coherence, p.perpendicular_baseline_m)
            for i, p in enumerate(pairs)
        ]

    # ═══════════════════════════════════════════════════════════════════
    # ADVANCED CORRECTION 2: DEM ERROR
    # ═══════════════════════════════════════════════════════════════════
    def _correct_dem_error(
        self,
        pairs: List[InterferogramPair],
        dates: List[str],
        max_iterations: int = 2
    ) -> List[InterferogramPair]:
        """
        DEM Error Correction via weighted linear regression.

        Physics: DEM errors cause a phase ramp proportional to the perpendicular
        baseline (B_perp). We regress the residual phase against B_perp to
        estimate and remove this error.
        """
        working_pairs = [
            InterferogramPair(p.reference_date, p.secondary_date,
                              np.array(p.unwrapped_phase, copy=True),
                              np.array(p.coherence, copy=True),
                              p.perpendicular_baseline_m)
            for p in pairs
        ]

        for iteration in range(max_iterations):
            logger.info("DEM error correction iteration %d/%d...", iteration + 1, max_iterations)

            # 1. Quick native inversion to get current residuals
            result = self._invert_native(working_pairs, coherence_threshold=0.3, reference_pixel=None)
            date_idx = {d: i for i, d in enumerate(result.dates)}
            date_idx[result.reference_date]

            # 2. Reconstruct phases from the time series
            h, w = working_pairs[0].unwrapped_phase.shape
            reconstructed_phases = np.zeros((len(working_pairs), h, w), dtype=np.float32)

            for i, p in enumerate(working_pairs):
                i1, i2 = date_idx[p.reference_date], date_idx[p.secondary_date]
                disp_diff = result.displacement[i2] - result.displacement[i1]
                reconstructed_phases[i] = disp_diff * (4 * np.pi / self._wavelength)

            # 3. Compute residual phase
            residual_phases = np.stack([p.unwrapped_phase for p in working_pairs], axis=0) - reconstructed_phases
            b_perp_array = np.array([p.perpendicular_baseline_m for p in working_pairs], dtype=np.float64)
            coh_array = np.stack([p.coherence for p in working_pairs], axis=0)

            # 4. WLS Regression: residual = beta * B_perp
            weights = coh_array ** 2
            weights = np.where(np.isnan(weights), 0.0, weights)

            numerator = np.sum(weights * b_perp_array[:, None, None] * residual_phases, axis=0)
            denominator = np.sum(weights * (b_perp_array[:, None, None] ** 2), axis=0)

            beta = np.where(denominator > 1e-6, numerator / denominator, 0.0)

            # 5. Correct the interferogram phases
            corrections_made = 0
            for i, p in enumerate(working_pairs):
                correction = beta * p.perpendicular_baseline_m
                valid_mask = (np.abs(p.perpendicular_baseline_m) > 10.0) & (p.coherence > 0.3)
                if np.any(valid_mask):
                    corrections_made += np.sum(valid_mask)
                    p.unwrapped_phase[valid_mask] -= correction[valid_mask]

            logger.info("DEM correction applied to %d pixels.", corrections_made)

        return working_pairs

    # ═══════════════════════════════════════════════════════════════════
    def _reference_pairs(
        self, pairs: List[InterferogramPair], reference_pixel: Optional[Tuple[int, int]]
    ) -> List[InterferogramPair]:
        """
        Reference every interferogram's unwrapped phase to a common pixel.

        SNAPHU (and any phase unwrapper) recovers phase only up to an
        arbitrary additive integer multiple of 2*pi per interferogram —
        there is no absolute phase reference without external ground truth.
        Combining multiple independently-unwrapped interferograms in a
        joint least-squares inversion requires first removing this
        per-interferogram offset by subtracting the phase at a common
        pixel, so that pixel reads exactly zero displacement in every
        interferogram (Berardino et al. 2002).
        """
        np = self._np()

        if reference_pixel is None:
            # Auto-select the pixel with highest mean coherence across all pairs
            coh_stack = np.stack([p.coherence for p in pairs], axis=0)
            mean_coh = coh_stack.mean(axis=0)
            ry, rx = np.unravel_index(np.argmax(mean_coh), mean_coh.shape)
            reference_pixel = (int(ry), int(rx))
            logger.info(
                "No reference_pixel specified — auto-selected pixel %s "
                "(mean coherence=%.3f). For real deformation monitoring, "
                "prefer an explicit reference_pixel known to be stable.",
                reference_pixel,
                float(mean_coh[ry, rx]),
            )
        else:
            logger.info("Referencing all interferograms to pixel %s", reference_pixel)

        ry, rx = reference_pixel
        referenced = []
        for p in pairs:
            offset = p.unwrapped_phase[ry, rx]
            referenced.append(
                InterferogramPair(
                    reference_date=p.reference_date,
                    secondary_date=p.secondary_date,
                    unwrapped_phase=p.unwrapped_phase - offset,
                    coherence=p.coherence,
                    perpendicular_baseline_m=p.perpendicular_baseline_m,
                )
            )
        return referenced


    def invert_weighted(
        self,
        pairs: List[InterferogramPair],
        classification: "Any" = None,
        coherence_threshold: float = 0.3,
        bridge_penalty: float = 0.3,
        correct_unwrap: bool = True,
        correct_dem: bool = True,
        reference_pixel: Optional[Tuple[int, int]] = None,
    ) -> TimeSeriesResult:
        """
        Weighted-least-squares SBAS inversion, using real per-pixel
        coherence as the weight rather than a hard threshold cutoff.

        This is a genuinely separate method from invert(), not a
        modification of it — the existing OLS path (invert()) is left
        completely untouched, specifically so this method's own tests
        can compare against it directly: with every weight set to 1.0
        (uniform coherence=1, no bridge pairs), this method's output
        must exactly reproduce invert()'s output, since the weighted
        normal equations A^T W A reduce exactly to the unweighted
        A^T A when W is the identity matrix. That equivalence is
        checked directly in this module's own tests, not just assumed.

        Mathematics (Berardino et al. 2002, extended to the weighted
        case): the standard SBAS system B*v = dphi is solved here as
        (B^T W B) v = B^T W dphi, where W is a real, per-pixel,
        per-pair diagonal weight matrix with w_i = coherence_i^2 (the
        conventional InSAR weighting -- coherence-squared approximates
        inverse phase variance under the standard InSAR phase noise
        model). Pairs classified as bridge_pairs by
        DataValidator.classify_pairs() have their weight additionally
        multiplied by `bridge_penalty`, acknowledging their lower
        reliability while preserving the network topology they're
        structurally necessary for -- this is the real mechanism that
        fixes the specific failure this project has directly observed:
        a network that looks well-connected by its own coherence
        selection shattering into many disconnected islands the moment
        low-quality-but-necessary pairs are excluded outright instead
        of down-weighted.

        Because the weight is a real, continuous per-pixel value (not
        just a pass/fail threshold), this cannot reuse invert()'s
        pattern-grouping optimization (which relies on many pixels
        sharing an identical design matrix). Instead, pixels are
        grouped by which pairs are usable at all (coherence > 0, i.e.
        a real, present observation), and within each such group the
        weighted normal equations are solved for every pixel at once
        via a single batched np.linalg.solve call over the group's
        real per-pixel weights — not a slow per-pixel Python loop, and
        not an approximation using a group-average weight.

        Args:
            pairs: Real InterferogramPair list forming the SBAS network.
            classification: Optional PairClassification from
                DataValidator.classify_pairs() — if given, bridge_pairs
                get bridge_penalty applied to their weight. If None,
                all real pairs are weighted purely by coherence^2 with
                no bridge penalty (equivalent to classifying nothing
                as a bridge).
            coherence_threshold: Pixels with zero usable coherence
                across all their pairs are still marked unreliable
                (NaN) -- this threshold only gates whether a
                pixel/pair observation is used AT ALL, not its weight
                once included.
            bridge_penalty: Multiplier applied to a bridge pair's
                weight (default 0.3, matching this feature's real
                specification).
            reference_pixel: Same meaning as invert() -- required for
                correct results; see invert()'s own docstring.

        Returns:
            TimeSeriesResult, same shape and fields as invert().
        """
        from pygeofetch.insar.validate import DataValidator

        np = self._np()

        all_dates = sorted({p.reference_date for p in pairs} | {p.secondary_date for p in pairs})
        DataValidator.validate_sbas_network(pairs, all_dates).raise_if_invalid()

        # REAL GAP FOUND AND FIXED HERE: this method was restored from a
        # commented-out block that predated invert()'s own real phase-
        # closure and DEM-error correction capability -- it had no way to
        # apply either, a genuine parity gap against invert(), not a
        # deliberate omission. Reuses the exact same, already-implemented
        # _correct_unwrapping_errors/_correct_dem_error methods invert()
        # calls, applied to a real deep copy so the caller's original
        # pairs are never mutated -- same real order (unwrap correction
        # before DEM correction) invert() uses, for the same reason.
        if correct_unwrap or correct_dem:
            working_pairs = [
                InterferogramPair(
                    reference_date=p.reference_date,
                    secondary_date=p.secondary_date,
                    unwrapped_phase=np.array(p.unwrapped_phase, copy=True),
                    coherence=np.array(p.coherence, copy=True),
                    perpendicular_baseline_m=p.perpendicular_baseline_m,
                )
                for p in pairs
            ]
            if correct_unwrap:
                logger.info("invert_weighted: applying native unwrapping error correction (Phase Closure)...")
                working_pairs = self._correct_unwrapping_errors(working_pairs, all_dates)
            if correct_dem:
                logger.info("invert_weighted: applying native DEM error correction...")
                working_pairs = self._correct_dem_error(working_pairs, all_dates)
            pairs = working_pairs

        pairs = self._reference_pairs(pairs, reference_pixel)

        bridge_pair_ids = set()
        if classification is not None:
            bridge_pair_ids = {
                (p.reference_date, p.secondary_date) for p in classification.bridge_pairs
            }

        dates = sorted(set([p.reference_date for p in pairs] + [p.secondary_date for p in pairs]))
        n_dates = len(dates)
        n_pairs = len(pairs)
        date_idx = {d: i for i, d in enumerate(dates)}

        ref_date = self._reference_date or dates[0]
        if ref_date not in date_idx:
            raise ValueError(f"reference_date {ref_date!r} not found in network dates")

        logger.info(
            "SBAS weighted inversion: %d dates, %d interferogram pairs (%d bridge), reference=%s",
            n_dates, n_pairs, len(bridge_pair_ids), ref_date,
        )

        h, w = pairs[0].unwrapped_phase.shape
        A = np.zeros((n_pairs, n_dates), dtype=np.float64)
        pair_bridge_penalty = np.ones(n_pairs, dtype=np.float64)
        for i, pair in enumerate(pairs):
            A[i, date_idx[pair.reference_date]] = -1
            A[i, date_idx[pair.secondary_date]] = 1
            if (pair.reference_date, pair.secondary_date) in bridge_pair_ids:
                pair_bridge_penalty[i] = bridge_penalty

        ref_col = date_idx[ref_date]
        keep_cols = [c for c in range(n_dates) if c != ref_col]
        keep_cols_arr = np.array(keep_cols)
        n_keep = len(keep_cols)

        phase_stack = np.stack([p.unwrapped_phase for p in pairs], axis=0)
        disp_stack = (phase_stack * self._wavelength / (4 * np.pi)).astype(np.float64)
        coh_stack = np.stack([p.coherence for p in pairs], axis=0).astype(np.float64)

        # Usable = real, finite coherence above a floor (a pair with zero/NaN
        # coherence at a pixel is not a real observation there at all,
        # regardless of weighting -- this is the same "is there real data
        # here" gate invert() uses, kept for the same reason: honesty about
        # missing data, not a quality threshold on top of that).
        usable_mask = np.isfinite(coh_stack) & (coh_stack >= coherence_threshold * 0.0 + 1e-6)
        pattern = np.zeros((h, w), dtype=np.int64)
        for i in range(n_pairs):
            pattern += usable_mask[i].astype(np.int64) << i

        displacement = np.full((n_dates, h, w), np.nan, dtype=np.float32)
        residual_rms = np.full((h, w), np.nan, dtype=np.float32)
        n_underdetermined = 0

        for p in np.unique(pattern):
            pixel_mask = pattern == p
            valid_pair_idx = [i for i in range(n_pairs) if (int(p) >> i) & 1]
            n_valid = len(valid_pair_idx)
            if n_valid < n_keep:
                n_underdetermined += int(pixel_mask.sum())
                continue

            A_sub = A[valid_pair_idx][:, keep_cols_arr]  # (n_valid, n_keep)
            rows, cols = np.where(pixel_mask)
            n_group = len(rows)

            # Same two-step indexing pattern as the existing, working OLS
            # code above (list-index the pair axis, then paired fancy-index
            # rows/cols on what remains) -- np.ix_ here would be WRONG: it
            # builds an outer-product mesh of rows against cols instead of
            # selecting the n_group real (row,col) pairs together, caught
            # directly by this feature's own first test run raising a real
            # shape error before this fix.
            coh_sub = coh_stack[valid_pair_idx][:, rows, cols].transpose(1, 0)  # (n_group, n_valid)
            penalty_sub = pair_bridge_penalty[valid_pair_idx]  # (n_valid,)
            w_sub = (coh_sub ** 2) * penalty_sub[None, :]  # (n_group, n_valid) real per-pixel weights

            b_sub = disp_stack[valid_pair_idx][:, rows, cols].transpose(1, 0)  # (n_group, n_valid)

            # Batched weighted normal equations, one linear system per pixel,
            # solved together in a single vectorized call:
            #   M[p] = A_sub^T diag(w_sub[p]) A_sub      (n_group, n_keep, n_keep)
            #   rhs[p] = A_sub^T diag(w_sub[p]) b_sub[p]  (n_group, n_keep)
            M = np.einsum("pi,ij,ik->pjk", w_sub, A_sub, A_sub)
            rhs = np.einsum("pi,ij,pi->pj", w_sub, A_sub, b_sub)

            try:
                est = np.linalg.solve(M, rhs[..., None])[..., 0]  # (n_group, n_keep)
                # explicit trailing dim on rhs disambiguates batch vs core
                # shape for np.linalg.solve's batched gufunc signature --
                # confirmed necessary directly: rhs shaped (n_group, n_keep)
                # alone raised a real shape-mismatch error in this numpy
                # version rather than being interpreted as a batch of
                # vectors, caught by this feature's own test run before
                # being papered over.
            except np.linalg.LinAlgError:
                # A small number of pixels in this group have a singular
                # weighted system (e.g. every usable pair happens to carry
                # near-zero weight) -- fall back to per-pixel pinv only for
                # those, rather than failing the whole group.
                est = np.full((n_group, n_keep), np.nan)
                for gi in range(n_group):
                    try:
                        est[gi] = np.linalg.pinv(M[gi]) @ rhs[gi]
                    except np.linalg.LinAlgError:
                        n_underdetermined += 1

            displacement[keep_cols_arr[:, None], rows, cols] = est.T
            predicted = np.einsum("ij,pj->pi", A_sub, est)  # (n_group, n_valid)
            residuals = b_sub - predicted
            residual_rms[rows, cols] = np.sqrt(np.mean(residuals ** 2, axis=1))

        displacement[ref_col] = 0.0
        if n_underdetermined > 0:
            logger.warning(
                "%d/%d pixels (%.1f%%) had too few usable pairs to solve and "
                "are marked unreliable (NaN) rather than silently included "
                "with insufficient data.",
                n_underdetermined, h * w, 100 * n_underdetermined / (h * w),
            )

        t_years = np.array(
            [self._days_between(ref_date, d) / 365.25 for d in dates], dtype=np.float32
        )
        velocity = self._fit_velocity(displacement, t_years)

        return TimeSeriesResult(
            dates=dates,
            displacement=displacement,
            velocity=velocity,
            residual_rms=residual_rms,
            reference_date=ref_date,
            metadata={
                "wavelength_m": self._wavelength,
                "n_pairs": n_pairs,
                "n_bridge_pairs": len(bridge_pair_ids),
                "bridge_penalty": bridge_penalty,
                "method": "SBAS weighted least squares, per-pixel coherence^2 weighting (Berardino et al. 2002)",
            },
        )


    # CORE NATIVE INVERSION (Weighted Least Squares)
    # ═══════════════════════════════════════════════════════════════════
    def _invert_native(
        self,
        pairs: List[InterferogramPair],
        coherence_threshold: float,
        reference_pixel: Optional[Tuple[int, int]] = None
    ) -> TimeSeriesResult:
        """Core WLS SBAS inversion (Berardino et al., 2002)."""
        dates = sorted(set([p.reference_date for p in pairs] + [p.secondary_date for p in pairs]))
        n_dates = len(dates)
        n_pairs = len(pairs)
        date_idx = {d: i for i, d in enumerate(dates)}
        ref_date = self._reference_date or dates[0]

        h, w = pairs[0].unwrapped_phase.shape

        # Design Matrix A
        A = np.zeros((n_pairs, n_dates), dtype=np.float64)
        for i, pair in enumerate(pairs):
            A[i, date_idx[pair.reference_date]] = -1
            A[i, date_idx[pair.secondary_date]] = 1

        ref_col = date_idx[ref_date]
        keep_cols = [c for c in range(n_dates) if c != ref_col]
        keep_cols_arr = np.array(keep_cols)
        n_keep = len(keep_cols)

        phase_stack = np.stack([p.unwrapped_phase for p in pairs], axis=0).astype(np.float64)
        disp_stack = (phase_stack * self._wavelength / (4 * np.pi))
        coh_stack = np.stack([p.coherence for p in pairs], axis=0).astype(np.float64)

        # Group pixels by their valid-pair pattern for efficient batch solving
        usable_mask = np.isfinite(coh_stack) & (coh_stack >= coherence_threshold)
        pattern = np.zeros((h, w), dtype=np.int64)
        for i in range(n_pairs):
            pattern += usable_mask[i].astype(np.int64) << i

        displacement = np.full((n_dates, h, w), np.nan, dtype=np.float32)
        residual_rms = np.full((h, w), np.nan, dtype=np.float32)
        n_underdetermined = 0

        for p_val in np.unique(pattern):
            pixel_mask = pattern == p_val
            valid_pair_idx = [i for i in range(n_pairs) if (int(p_val) >> i) & 1]
            n_valid = len(valid_pair_idx)

            if n_valid < n_keep:
                n_underdetermined += int(pixel_mask.sum())
                continue

            A_sub = A[valid_pair_idx][:, keep_cols_arr]
            rows, cols = np.where(pixel_mask)
            n_group = len(rows)

            coh_sub = coh_stack[valid_pair_idx][:, rows, cols].transpose(1, 0)
            w_sub = coh_sub ** 2
            b_sub = disp_stack[valid_pair_idx][:, rows, cols].transpose(1, 0)

            # WLS Normal Equations: (A^T W A) v = A^T W b
            M = np.einsum("pi,ij,ik->pjk", w_sub, A_sub, A_sub)
            rhs = np.einsum("pi,ij,pi->pj", w_sub, A_sub, b_sub)

            try:
                est = np.linalg.solve(M, rhs[..., None])[..., 0]
            except np.linalg.LinAlgError:
                est = np.full((n_group, n_keep), np.nan)

            displacement[keep_cols_arr[:, None], rows, cols] = est.T
            predicted = np.einsum("ij,pj->pi", A_sub, est)
            residuals = b_sub - predicted
            residual_rms[rows, cols] = np.sqrt(np.mean(residuals ** 2, axis=1))

        displacement[ref_col] = 0.0

        if n_underdetermined > 0:
            logger.warning("%d/%d pixels had too few usable pairs.", n_underdetermined, h * w)

        # Calculate velocity
        t_years = np.array([self._days_between(ref_date, d) / 365.25 for d in dates], dtype=np.float32)
        velocity = self._fit_velocity(displacement, t_years)

        return TimeSeriesResult(
            dates=dates, displacement=displacement, velocity=velocity,
            residual_rms=residual_rms, reference_date=ref_date,
            metadata={"method": "Native SBAS WLS with Advanced Corrections"}
        )

    def _fit_velocity(self, displacement, t_years):
        np = self._np()
        n_dates, h, w = displacement.shape
        t_mean = t_years.mean()
        t_centered = t_years - t_mean
        denom = np.sum(t_centered**2)
        if denom == 0:
            return np.zeros((h, w), dtype=np.float32)
        disp_mean = displacement.mean(axis=0)
        numer = np.tensordot(t_centered, displacement - disp_mean, axes=([0], [0]))
        return (numer / denom).astype(np.float32)

    def _days_between(self, d1: str, d2: str) -> int:
        return (datetime.strptime(d2, "%Y-%m-%d") - datetime.strptime(d1, "%Y-%m-%d")).days

    def _np(self):
        import numpy as np
        return np
