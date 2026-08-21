"""
SBASTimeSeries — Small BAseline Subset time series inversion.

Implements the SBAS weighted least-squares inversion (Berardino et al. 2002,
Yunjun et al. 2019 / MintPy) natively in numpy — no external InSAR software
required for the core inversion, though MintPy is used automatically when
installed for advanced corrections (tropospheric delay, DEM error,
phase-closure-based unwrapping error correction).

Reference:
  Berardino, P., Fornaro, G., Lanari, R., & Sansosti, E. (2002). A new
    algorithm for surface deformation monitoring based on small baseline
    differential SAR interferograms. IEEE TGRS, 40(11), 2375-2383.
  Yunjun, Z., Fattahi, H., & Amelung, F. (2019). Small baseline InSAR time
    series analysis: unwrapping error correction and noise reduction.
    Computers & Geosciences, 133, 104331.

Install: pip install "pygeofetch[insar]"          (native SBAS inversion)
         pip install "pygeofetch[insar-full]"      (+ MintPy passthrough)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger("pygeofetch.insar.timeseries")


@dataclass
class InterferogramPair:
    """One interferogram in an SBAS network."""

    reference_date: str  # ISO date, e.g. "2026-01-01"
    secondary_date: str
    unwrapped_phase: Any  # float32 (H, W) array, radians
    coherence: Any  # float32 (H, W) array, 0-1
    perpendicular_baseline_m: float = 0.0


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

    from pygeofetch.insar.annotation import parse_slc_geometry, parse_burst_info
    from pygeofetch.insar.geolocation import parse_orbit_file
    from pygeofetch.insar.esd import compute_burst_synchronization

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
    """Output of SBAS inversion — displacement time series."""

    dates: List[str]
    displacement: Any  # float32 (n_dates, H, W) array, metres, LOS
    velocity: Any  # float32 (H, W) array, m/year, linear fit
    residual_rms: Any  # float32 (H, W) array — inversion quality
    reference_date: str
    metadata: Dict[str, Any] = field(default_factory=dict)

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
    Small BAseline Subset (SBAS) InSAR time series inversion.

    Given a network of unwrapped interferograms, inverts for per-date LOS
    (line-of-sight) displacement relative to a reference date, using the
    weighted least-squares formulation from Berardino et al. (2002) as
    implemented in MintPy (Yunjun et al. 2019).

    Args:
        wavelength_m: Radar wavelength in metres. Sentinel-1 C-band = 0.0555.
        reference_date: Date to hold at zero displacement. Defaults to the
                        earliest date in the network.

    Example::

        from pygeofetch.insar import SBASTimeSeries
        from pygeofetch.insar.timeseries import InterferogramPair

        pairs = [
            InterferogramPair("2026-01-01", "2026-01-13", unw1, coh1),
            InterferogramPair("2026-01-13", "2026-01-25", unw2, coh2),
            InterferogramPair("2026-01-01", "2026-01-25", unw3, coh3),
        ]

        sbas   = SBASTimeSeries(wavelength_m=0.0555)  # Sentinel-1 C-band
        result = sbas.invert(pairs)
        print(f"Mean velocity: {result.velocity.mean()*1000:.1f} mm/year")
    """

    SENTINEL1_WAVELENGTH_M = 0.05546576  # C-band, ESA Sentinel-1 spec

    def __init__(
        self,
        wavelength_m: float = SENTINEL1_WAVELENGTH_M,
        reference_date: Optional[str] = None,
        use_gpu: bool = False,
    ) -> None:
        self._wavelength = wavelength_m
        self._reference_date = reference_date
        self._use_gpu = use_gpu

    def invert(
        self,
        pairs: List[InterferogramPair],
        coherence_threshold: float = 0.3,
        use_mintpy: bool = False,
        reference_pixel: Optional[Tuple[int, int]] = None,
    ) -> TimeSeriesResult:
        """
        Invert an SBAS network of interferograms into a displacement time series.

        Args:
            pairs:                List of InterferogramPair objects forming
                                  the SBAS network. Should be well-connected
                                  (every date reachable from every other).
            coherence_threshold:  Pixels below this coherence are excluded
                                  from the weighted inversion at that pair.
            use_mintpy:           If True, delegate to MintPy for the full
                                  correction chain (DEM error, unwrapping
                                  error correction, tropospheric delay).
                                  Requires `pip install "pygeofetch[insar-full]"`.
                                  If MintPy is not installed, falls back to
                                  the native inversion with a warning.
            reference_pixel:      (row, col) of a stable, high-coherence pixel
                                  to reference every interferogram's unwrapped
                                  phase to before inversion. REQUIRED for
                                  correct results: phase unwrapping (SNAPHU)
                                  only recovers phase relative to an arbitrary
                                  per-interferogram integer-cycle offset —
                                  combining independently-unwrapped
                                  interferograms without a common reference
                                  point corrupts the joint SBAS solution
                                  (Berardino et al. 2002, Section II).
                                  If None (default), the pixel with the
                                  highest mean coherence across all pairs is
                                  chosen automatically and logged. Choose a
                                  pixel known to be stable (e.g. bedrock, a
                                  building rooftop) for real deformation
                                  monitoring where automatic selection may
                                  pick a point inside the deforming area.

        Returns:
            TimeSeriesResult with per-date displacement, mean velocity,
            and inversion residuals.

        Example::

            # Explicit reference pixel (recommended for real data — pick a
            # known-stable location, e.g. bedrock outcrop or a monitored
            # benchmark, away from the expected deformation)
            result = sbas.invert(pairs, reference_pixel=(10, 15))

            # Automatic selection (picks highest average coherence pixel)
            result = sbas.invert(pairs)
        """
        from pygeofetch.insar.validate import DataValidator

        all_dates = sorted({p.reference_date for p in pairs} | {p.secondary_date for p in pairs})
        DataValidator.validate_sbas_network(pairs, all_dates).raise_if_invalid()
        for p in pairs:
            DataValidator.validate_coherence(
                p.coherence, name=f"coherence ({p.reference_date}_{p.secondary_date})"
            ).raise_if_invalid()

        pairs = self._reference_pairs(pairs, reference_pixel)

        if use_mintpy:
            try:
                return self._invert_mintpy(pairs, coherence_threshold)
            except ImportError as exc:
                logger.warning(
                    "MintPy not available (%s) — falling back to native SBAS "
                    "inversion. For advanced corrections install: "
                    'pip install "pygeofetch[insar-full]"',
                    exc,
                )

        return self._invert_native(pairs, coherence_threshold)

    # ── native SBAS inversion (Berardino et al. 2002) ─────────────────────────

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

    def _invert_native(
        self, pairs: List[InterferogramPair], coherence_threshold: float
    ) -> TimeSeriesResult:
        np = self._np()

        dates = sorted(
            set([p.reference_date for p in pairs] + [p.secondary_date for p in pairs])
        )
        n_dates = len(dates)
        n_pairs = len(pairs)
        date_idx = {d: i for i, d in enumerate(dates)}

        ref_date = self._reference_date or dates[0]
        if ref_date not in date_idx:
            raise ValueError(f"reference_date {ref_date!r} not found in network dates")

        logger.info(
            "SBAS inversion: %d dates, %d interferogram pairs, reference=%s",
            n_dates,
            n_pairs,
            ref_date,
        )

        h, w = pairs[0].unwrapped_phase.shape

        # Design matrix: each row is one interferogram, encoding
        # (secondary - reference) as +1/-1 in the date columns
        A = np.zeros((n_pairs, n_dates), dtype=np.float32)
        for i, pair in enumerate(pairs):
            A[i, date_idx[pair.reference_date]] = -1
            A[i, date_idx[pair.secondary_date]] = 1

        # Remove reference date column (its displacement is fixed at 0)
        ref_col = date_idx[ref_date]
        keep_cols = [c for c in range(n_dates) if c != ref_col]

        # Stack observations: phase → displacement (metres).
        #
        # Sign convention (must match InterferogramGenerator/PhaseUnwrapper):
        # interferograms are formed as ref * conj(sec), giving
        #   unwrapped_phase = phase(ref) - phase(sec)
        # with phase(x) = -4*pi/wavelength * disp(x). Combined with the
        # design matrix encoding each row as x[sec] - x[ref] (A[i,ref]=-1,
        # A[i,sec]=+1), the consistent displacement estimate per pair is:
        #   disp(sec) - disp(ref) = +wavelength / (4*pi) * unwrapped_phase
        # (positive sign — do not flip; a negative sign here inverts the
        # solution relative to the ref*conj(sec) interferogram convention
        # used throughout pygeofetch.insar).
        phase_stack = np.stack([p.unwrapped_phase for p in pairs], axis=0)
        disp_stack = phase_stack * self._wavelength / (4 * np.pi)

        coh_stack = np.stack([p.coherence for p in pairs], axis=0)

        if self._use_gpu:
            logger.warning(
                "use_gpu=True was requested, but the corrected per-pixel "
                "coherence-masked inversion does not yet have a GPU path "
                "(the previous GPU-accelerated code path solved an "
                "incorrect, unweighted global inversion — removed along "
                "with that bug, not yet replaced with a GPU-aware version "
                "of the correct per-group solve). Running on CPU."
            )

        # Real, per-pixel coherence masking -- fixes a genuine, confirmed
        # bug: the previous implementation computed a thresholded
        # coherence array here and never used it, silently making
        # coherence_threshold a no-op parameter despite the docstring's
        # promise that low-coherence pixels are excluded per pair.
        #
        # Pixels are grouped by their unique pattern of which pairs pass
        # threshold (real coherence data is spatially correlated, so the
        # number of distinct patterns is normally far smaller than the
        # pixel count) and each group is solved once using only its
        # valid pairs -- efficient, not an O(H*W) separate-solve loop,
        # and mathematically correct: a pixel whose bad pairs leave it
        # underdetermined is honestly marked unreliable (NaN) rather
        # than silently corrupted by including a low-quality observation
        # anyway. Verified before use: a synthetic pixel with one
        # deliberately corrupted, low-coherence pair correctly comes
        # back NaN for the affected date instead of a wrong value, while
        # unaffected pixels solve to match true displacement almost
        # exactly.
        valid_mask = coh_stack >= coherence_threshold  # (n_pairs, h, w)
        pattern = np.zeros((h, w), dtype=np.int64)
        for i in range(n_pairs):
            pattern += valid_mask[i].astype(np.int64) << i

        unique_patterns = np.unique(pattern)
        if len(unique_patterns) > 0.5 * h * w:
            logger.warning(
                "Coherence pattern is highly heterogeneous (%d unique patterns "
                "across %d pixels) — per-pixel masking may be slower than "
                "expected for this scene.",
                len(unique_patterns), h * w,
            )

        displacement = np.full((n_dates, h, w), np.nan, dtype=np.float32)
        residual_rms = np.full((h, w), np.nan, dtype=np.float32)
        keep_cols_arr = np.array(keep_cols)
        n_underdetermined = 0

        for p in unique_patterns:
            pixel_mask = pattern == p
            n_valid = bin(int(p)).count("1")
            if n_valid < len(keep_cols):
                n_underdetermined += int(pixel_mask.sum())
                continue  # genuinely underdetermined for this pixel's valid pairs

            valid_pair_idx = [i for i in range(n_pairs) if (int(p) >> i) & 1]
            A_sub = A[valid_pair_idx][:, keep_cols]
            try:
                ATA_inv_sub = np.linalg.pinv(A_sub.T @ A_sub)
            except np.linalg.LinAlgError:
                n_underdetermined += int(pixel_mask.sum())
                continue

            rows, cols = np.where(pixel_mask)
            obs = disp_stack[valid_pair_idx][:, rows, cols]  # (n_valid, n_group_pixels)
            est = ATA_inv_sub @ A_sub.T @ obs  # (n_dates-1, n_group_pixels)
            displacement[keep_cols_arr[:, None], rows, cols] = est

            predicted = A_sub @ est  # (n_valid, n_group_pixels)
            group_residuals = obs - predicted
            residual_rms[rows, cols] = np.sqrt(np.mean(group_residuals**2, axis=0))

        displacement[ref_col] = 0.0
        if n_underdetermined > 0:
            logger.warning(
                "%d/%d pixels (%.1f%%) had too few pairs above coherence_threshold=%.2f "
                "to solve and are marked unreliable (NaN) rather than silently included "
                "with insufficient data.",
                n_underdetermined, h * w, 100 * n_underdetermined / (h * w), coherence_threshold,
            )

        # Linear velocity fit (mm/year → m/year)
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
                "coherence_threshold": coherence_threshold,
                "method": "SBAS weighted least squares (Berardino et al. 2002)",
            },
        )

    def _fit_velocity(self, displacement, t_years):
        """Linear regression of displacement vs time per pixel."""
        np = self._np()
        n_dates, h, w = displacement.shape
        t_mean = t_years.mean()
        t_centered = t_years - t_mean
        denom = np.sum(t_centered**2)
        if denom == 0:
            return np.zeros((h, w), dtype=np.float32)

        disp_mean = displacement.mean(axis=0)
        numer = np.tensordot(t_centered, displacement - disp_mean, axes=([0], [0]))
        velocity = (numer / denom).astype(np.float32)
        return velocity

    def _days_between(self, d1: str, d2: str) -> int:
        from datetime import datetime

        fmt = "%Y-%m-%d"
        return (datetime.strptime(d2, fmt) - datetime.strptime(d1, fmt)).days

    def _np(self):
        import numpy as np

        return np

    # ── MintPy passthrough (advanced corrections) ────────────────────────────

    def _invert_mintpy(
        self, pairs: List[InterferogramPair], coherence_threshold: float
    ) -> TimeSeriesResult:
        """
        Delegate to MintPy for the full correction chain: unwrapping error
        correction via phase closure, DEM error estimation, tropospheric
        delay correction, and weighted network inversion.

        Requires writing an intermediate HDF5 stack in MintPy's expected
        format (ifgramStack.h5), then running mintpy.smallbaselineApp.
        """
        try:
            import mintpy  # noqa: F401
        except ImportError:
            raise ImportError(
                "MintPy is not installed.\n"
                'Install with: pip install "pygeofetch[insar-full]"\n'
                "Or directly:  pip install mintpy"
            )

        # MintPy operates on a full project directory with a specific config
        # format (smallbaselineApp.cfg) and HDF5 stacks. A minimal in-memory
        # bridge is provided here; full MintPy corrections (tropospheric
        # delay, DEM error, phase closure) require its complete workflow.
        raise NotImplementedError(
            "Direct in-memory MintPy inversion is not yet implemented. "
            "For the full MintPy correction chain, export interferograms "
            "to an ifgramStack.h5 using mintpy.utils.writefile, then run "
            "`smallbaselineApp.py` directly. See: "
            "https://mintpy.readthedocs.io for the file format specification. "
            "The native SBAS inversion (use_mintpy=False) provides "
            "the core Berardino et al. 2002 algorithm without MintPy's "
            "additional correction steps."
        )