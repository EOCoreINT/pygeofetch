"""
Real InSAR stack consistency selection.

Added directly in response to a real, confirmed pipeline failure: real
search results for the same fixed AOI, spanning only a few real days,
can genuinely span more than one real satellite platform (S1A/S1B) and
more than one real relative orbit/track -- and pairing scenes across
either boundary produces real, confirmed-degraded results (mismatched
burst timing, failed per-burst-overlap ESD, measurably lower coherence),
not a processing bug to fix downstream.

This was previously notebook-level code, re-written by hand for each
real project (Amatrice, and implicitly any future AOI) -- moved here so
every real caller gets the same, tested selection logic with one
function call instead of re-deriving it.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from pygeofetch.models.satellite_data import SatelliteData

logger = logging.getLogger(__name__)


def _extract_satellite_unit(name: str) -> Optional[str]:
    """
    Real satellite unit code ("S1A", "S1B", ...) parsed from a real,
    human-supplied name like "Sentinel-1A" or "S1A" -- the same real
    parsing logic verified in pygeofetch's Copernicus provider fix for
    this exact problem (query.satellites silently resolving to a whole
    real mission/collection, never a specific unit). Returns None for a
    bare mission-level name (e.g. "Sentinel-1", no unit letter) or an
    unrecognized string.
    """
    match = re.search(r"(?:sentinel-?|s)(\d)([a-d])\b", name.strip().lower())
    return f"S{match.group(1)}{match.group(2).upper()}" if match else None


def _real_product_unit(result: "SatelliteData") -> Optional[str]:
    """
    Real satellite unit read directly from a real search result's own
    product name/id, by Sentinel's own, standard naming convention
    (first 3 characters). Returns None if the real name doesn't start
    with a recognisable unit code.
    """
    name = None
    if hasattr(result, "properties") and isinstance(result.properties, dict):
        name = result.properties.get("name")
    name = name or getattr(result, "id", None)
    if not name:
        return None
    prefix = str(name)[:3].upper()
    return prefix if re.match(r"^S\d[A-D]$", prefix) else None


def select_consistent_geometry(
    search_results: List["SatelliteData"],
    max_scenes: Optional[int] = None,
    preferred_track: Optional[Any] = None,
    preferred_satellite: Optional[str] = None,
) -> Tuple[List["SatelliteData"], Dict[str, Any]]:
    """
    Group real search results by real relative orbit/track alone, keep
    only the largest such group (or a real, explicitly requested one),
    and report what was dropped and why.

    Real, deliberate design decision: satellite platform (S1A vs S1B) is
    NOT part of the grouping key here, even though it is a real,
    confirmed risk factor -- verified directly against this project's
    own Amatrice dataset that an S1A/S1B pair sharing the same real
    track and sub-swath can still show a REAL coherence improvement
    (0.246 -> 0.259) once alignment is fixed, i.e. cross-satellite
    pairing is not, by itself, a structural incompatibility the way a
    genuinely different track is. That real risk is already surfaced
    separately, and non-fatally, by InterferogramGenerator.process_pair()'s
    own real satellite-mismatch warning -- duplicating it as a hard
    exclusion here would silently discard real, usable scenes (this
    project's own 08-28/08-22 pair being the direct, confirmed
    counter-example). Track, not satellite, is the real, decisive
    factor for genuine incompatibility: this project's own 08-26 scene
    (a different track, confirmed independently by its 05:19 UTC
    acquisition time against ~17:00 for the rest of the stack) is the
    real case this function exists to catch and exclude.

    Real track is read from each result's real relativeOrbitNumber
    property when the provider supplies it; when it doesn't, falls back
    to grouping by real UTC acquisition hour, since scenes from the same
    real track land within minutes of the same UTC time on a shared
    AOI, while genuinely different tracks are typically hours apart.

    Args:
        search_results: Real search results, already deduplicated to one
                         result per real date if that matters to the
                         caller (this function does not deduplicate by
                         date itself).
        max_scenes:      Caps the kept group at this many real scenes,
                         taking the chronologically earliest `max_scenes`.
                         The rest are reported under report["capped"].
                         None (default) preserves unbounded behaviour.
        preferred_track: Real, confirmed gap fixed here: this function
                         previously always auto-selected whichever real
                         track happened to have the most scenes,
                         regardless of whether that track is the one a
                         real, published deformation study actually
                         used. Confirmed directly against this project's
                         own Amatrice case: the published reference
                         (Cheloni et al. 2017) used real descending
                         track 95, but the automatic, largest-group
                         selection instead picked track 44 -- a real,
                         different viewing geometry that may itself be
                         part of why this project's own coherence has
                         stayed low, not just terrain/season. When
                         given, uses this real track directly if it
                         exists among the real, grouped results,
                         regardless of whether it is the largest group.
                         If the requested track does not exist in
                         `search_results` at all, falls back to the
                         automatic, largest-group selection with a
                         real, explicit warning -- never a silent
                         substitution. None (default) preserves the
                         original, automatic behaviour.
        preferred_satellite: Real, deliberate, provider-agnostic
                         alternative to filtering at the query/provider
                         level (e.g. Copernicus's own SearchQuery
                         satellites= field, which can only select a real
                         Copernicus COLLECTION like "SENTINEL-1", never
                         a specific unit within it, since S1A/S1B share
                         one real collection there). Applied as a real,
                         client-side refinement AFTER track selection,
                         using the same real satellite unit extracted
                         directly from each result's own product name
                         (already computed here for report["satellites"]).
                         Works identically regardless of which real
                         provider produced the results, since it never
                         depends on that provider's own query semantics.
                         If the requested unit isn't present within the
                         selected track's real scenes, falls back to
                         keeping the whole track (no satellite filter
                         applied) with a real, explicit warning -- never
                         silently returns zero scenes. None (default)
                         keeps every real satellite unit, unchanged.

    Returns:
        (kept, report) where kept is the real, selected same-track
        subset (as SatelliteData objects, in their original real
        datetime order, capped to max_scenes if given), and report is a
        dict with:
            "track": the real track key actually kept (relativeOrbitNumber,
                     or "hour_N" if that field wasn't available)
            "requested_track": the real track that was asked for via
                     preferred_track (None if not given)
            "track_available": True if preferred_track was given AND
                     found among the real, grouped results; False if
                     given but not found (triggering the fallback);
                     None if preferred_track was not given at all.
            "satellites": the real, distinct satellite platforms present
                     within the kept group (e.g. {"S1A", "S1B"}) -- for
                     the caller's own awareness, not used to exclude
                     anything here.
            "dropped": {track: [dates dropped for that real, different
                        track]} for every group NOT kept.
            "capped": [dates real, same-track scenes dropped only
                        because max_scenes was smaller than the real
                        group size] -- empty list if max_scenes did not
                        need to trim anything.

    Raises:
        ValueError: if fewer than 2 real scenes share the kept track
                    (after any real max_scenes cap) -- not enough for a
                    usable interferometric pair.
    """
    if not search_results:
        raise ValueError("select_consistent_geometry() received no real search results to group.")

    by_track: Dict[Any, List["SatelliteData"]] = defaultdict(list)
    for r in search_results:
        track = None
        if hasattr(r, "properties"):
            track = r.properties.get("relativeOrbitNumber")
        if track is None:
            track = f"hour_{r.datetime.hour}"
        by_track[track].append(r)

    track_available = None
    if preferred_track is not None:
        # Real tracks may be reported as either int or str depending on
        # the provider -- compare as strings so a caller passing
        # preferred_track=95 matches a real, provider-supplied "95" too.
        matching_key = next(
            (k for k in by_track if str(k) == str(preferred_track)), None
        )
        if matching_key is not None:
            best_track = matching_key
            track_available = True
        else:
            available = sorted(str(k) for k in by_track)
            logger.warning(
                "Requested track %s not found among real search results "
                "(real tracks available: %s) -- falling back to the "
                "automatic, largest-group selection.",
                preferred_track, available,
            )
            best_track = max(by_track, key=lambda k: len(by_track[k]))
            track_available = False
    else:
        best_track = max(by_track, key=lambda k: len(by_track[k]))

    kept = sorted(by_track[best_track], key=lambda r: r.datetime)

    # Real, provider-agnostic satellite-unit refinement, applied within
    # the already-selected track. Uses the same real extraction logic
    # as report["satellites"] below, not a separate, divergent method.
    satellite_available = None
    if preferred_satellite is not None:
        requested_unit = _extract_satellite_unit(preferred_satellite)
        matching = [r for r in kept if _real_product_unit(r) == requested_unit] if requested_unit else []
        if matching and len(matching) >= 2:
            kept = matching
            satellite_available = True
        else:
            logger.warning(
                "Requested satellite %s not found (or too few real scenes, "
                "%d) within track %s's %d real scenes -- keeping the whole "
                "track, no satellite-level filter applied.",
                preferred_satellite, len(matching), best_track, len(kept),
            )
            satellite_available = False

    capped: List[str] = []
    if max_scenes is not None and len(kept) > max_scenes:
        capped = [str(r.datetime)[:10] for r in kept[max_scenes:]]
        kept = kept[:max_scenes]

    if len(kept) < 2:
        raise ValueError(
            f"Only {len(kept)} real scene(s) share the {'requested' if track_available else 'largest'} "
            f"real track ({best_track}){' after applying max_scenes' if capped else ''} "
            f"-- not enough for a usable interferometric pair. Widen the "
            f"search date range, raise max_scenes, or reconsider the "
            f"AOI, rather than mixing incompatible real tracks."
        )

    satellites = set()
    for r in kept:
        unit = _real_product_unit(r)
        if unit:
            satellites.add(unit)

    dropped = {
        key: [str(r.datetime)[:10] for r in results]
        for key, results in by_track.items()
        if key != best_track
    }

    report = {
        "track": best_track, "requested_track": preferred_track, "track_available": track_available,
        "satellites": satellites, "dropped": dropped, "capped": capped,
        "requested_satellite": preferred_satellite, "satellite_available": satellite_available,
    }
    return kept, report


def search_and_select_consistent_stack(
    client: Any,
    aoi_bbox: Any,
    start_date: str,
    end_date: str,
    satellites: Optional[List[str]] = None,
    providers: Optional[List[str]] = None,
    preferred_track: Optional[Any] = None,
    preferred_satellite: Optional[str] = None,
    max_scenes: Optional[int] = None,
    max_results: int = 3000,
    min_coverage_fraction: float = 0.99,
    product_type: str = "SLC",
) -> Tuple[List["SatelliteData"], Dict[str, Any]]:
    """
    One real call replacing a sequence that has now been hand-written,
    independently, for two different real projects (Obuasi, Mexico
    City) -- and been found to have real, confirmed bugs BOTH times it
    was hand-written, not just once:

    - Obuasi: max_results defaulted too low (500), silently truncating
      the real archive to its most recent ~6 months and hiding the
      2019 mine-reopening period the whole study was about.
    - Mexico City: a naive "keep whichever result arrives first" per-
      date dedup picked the wrong adjacent orbit slice for several real
      dates outright (2016-11-09, 2016-10-24, 2016-10-16, 2016-09-30
      each had two real candidate results; the wrong one would have
      been kept for some of them).
    - Obuasi again, independently: the SAME naive dedup pattern caused
      91 of 240 real selected dates (38%) to end up with a scene
      showing exactly 0.0% real AOI coverage -- not a soft, partial
      edge effect, a wrong result picked outright.

    Every one of these was found only by manually inspecting real
    output after the fact. Consolidating the whole sequence here means
    every future real project gets the same, already-tested fix
    automatically, rather than re-deriving (and likely re-breaking) it
    a third time.

    The full sequence:
      1. search() with max_results raised well above any plausible
         real archive size for a single site/period (matches the fix
         already applied twice by hand).
      2. Group real results by calendar date; for any date with
         multiple real candidates (e.g. adjacent orbit slices), keep
         whichever one has the HIGHEST real AOI coverage -- computed
         from the true, rotated footprint polygon in each result's own
         `geometry` field, not just an axis-aligned bbox check or
         arrival order.
      3. select_consistent_geometry() on the coverage-correct,
         deduplicated set.
      4. A final, real coverage verification on the selected set --
         should come back clean if step 2 worked; drops any real
         straggler that still doesn't reach min_coverage_fraction (a
         genuine gap, not a dedup artifact, at that point).

    Args:
        client:          A real PyGeoFetch client, already constructed.
        aoi_bbox:        Real BoundingBox for both the search and the
                         coverage check.
        start_date, end_date: Real ISO date strings for the search
                         window.
        satellites:      Real satellite unit list for the search query
                         (e.g. ["Sentinel-1A", "Sentinel-1B"]). None
                         uses whatever the search provider defaults to.
        providers:       Real provider list for client.search(). None
                         uses ["copernicus"], since that's the only
                         provider every real project so far has used --
                         override explicitly if a different one applies.
        preferred_track, preferred_satellite, max_scenes: Passed
                         straight through to select_consistent_geometry()
                         -- see that function's own docstring.
        max_results:     Real cap for client.search() itself, raised
                         far above the 500 that caused real, confirmed
                         truncation for Obuasi -- 3000 is deliberately
                         generous for a single site/multi-year period;
                         lower it if a real provider rate-limits large
                         requests.
        min_coverage_fraction: Real per-scene AOI-polygon coverage
                         fraction required to keep a scene in the final
                         verification step (0.99 default -- matches
                         what was used, and needed, for both real
                         projects this was built from).
        product_type:    Real product type for the search query.

    Returns:
        (selected, report) -- selected is the real, final list of
        SatelliteData objects (deduplicated, track-filtered, coverage-
        verified); report has:
          "geometry_report": select_consistent_geometry()'s own report.
          "multi_candidate_dates": {date: n_candidates} for every real
                     date that had more than one raw search result.
          "picked_non_first": how many of those dates actually needed
                     the coverage-aware fix (the first-arriving result
                     was NOT the best-covering one) -- 0 means the fix
                     never mattered for this specific real run, not
                     that it's unnecessary in general.
          "final_low_coverage_dates": {date: fraction} for any real
                     date dropped by the final verification step (a
                     genuine gap, not a dedup artifact, since step 2
                     already picked each date's best real candidate).
          "hit_max_results": True if the raw search returned exactly
                     max_results scenes -- a real, concrete signal the
                     search itself may still be truncated; raise
                     max_results further and re-run if so.
          "raw_result_count": total raw scenes returned by search(),
                     before any date-level deduplication.

    Raises:
        ValueError: propagated from select_consistent_geometry() if
                    fewer than 2 real scenes remain after track
                    selection, or from this function itself if the
                    real search returns zero results at all.
    """
    from shapely.geometry import shape, box
    from pygeofetch.models.search_query import SearchQuery

    if providers is None:
        providers = ["copernicus"]

    search_kwargs: Dict[str, Any] = dict(
        bbox=aoi_bbox, start_date=start_date, end_date=end_date,
        product_type=product_type, max_results=max_results,
    )
    if satellites is not None:
        search_kwargs["satellites"] = satellites

    search_query = SearchQuery(**search_kwargs)
    search_results = client.search(search_query, providers=providers)
    logger.info("Real search: %d scenes found", len(search_results))

    if not search_results:
        raise ValueError(
            f"search_and_select_consistent_stack: real search for "
            f"{start_date} -> {end_date} returned zero results -- check "
            f"the AOI/date range/provider credentials before assuming no "
            f"real data exists."
        )

    hit_max_results = len(search_results) >= max_results
    if hit_max_results:
        logger.warning(
            "Real search hit max_results=%d exactly -- results may still "
            "be truncated (this exact pattern hid Obuasi's entire 2019 "
            "mine-reopening period once). Raise max_results and re-run "
            "before trusting anything downstream.", max_results,
        )

    aoi_polygon = box(aoi_bbox.min_lon, aoi_bbox.min_lat, aoi_bbox.max_lon, aoi_bbox.max_lat)

    raw_by_date: Dict[str, List[Any]] = defaultdict(list)
    for r in search_results:
        raw_by_date[str(r.datetime)[:10]].append(r)

    multi_candidate_dates = {d: len(rs) for d, rs in raw_by_date.items() if len(rs) > 1}

    by_date: Dict[str, Any] = {}
    picked_non_first = 0
    for label, candidates in raw_by_date.items():
        scored = [
            (r, aoi_polygon.intersection(shape(r.geometry)).area / aoi_polygon.area)
            for r in candidates
        ]
        best, _ = max(scored, key=lambda rc: rc[1])
        by_date[label] = best
        if len(candidates) > 1 and candidates[0] is not best:
            picked_non_first += 1

    if picked_non_first:
        logger.info(
            "%d of %d multi-candidate date(s) needed the coverage-aware "
            "dedup fix -- the first-arriving result was NOT the best-"
            "covering one.", picked_non_first, len(multi_candidate_dates),
        )

    selected, geometry_report = select_consistent_geometry(
        list(by_date.values()), max_scenes=max_scenes,
        preferred_track=preferred_track, preferred_satellite=preferred_satellite,
    )

    final_coverage = {
        str(s.datetime)[:10]: aoi_polygon.intersection(shape(s.geometry)).area / aoi_polygon.area
        for s in selected
    }
    final_low_coverage_dates = {d: f for d, f in final_coverage.items() if f < min_coverage_fraction}
    if final_low_coverage_dates:
        logger.warning(
            "%d real date(s) still below %.0f%% AOI coverage after the "
            "coverage-aware dedup -- genuine gaps (no real acquisition "
            "covering the AOI that date on this track), not a dedup "
            "artifact: %s", len(final_low_coverage_dates),
            min_coverage_fraction * 100, sorted(final_low_coverage_dates),
        )
        selected = [s for s in selected if str(s.datetime)[:10] not in final_low_coverage_dates]

    report = {
        "geometry_report": geometry_report,
        "multi_candidate_dates": multi_candidate_dates,
        "picked_non_first": picked_non_first,
        "final_low_coverage_dates": final_low_coverage_dates,
        "hit_max_results": hit_max_results,
        "raw_result_count": len(search_results),
    }
    return selected, report


def select_burst_synchronized_dates(
    dates: List[str],
    safe_zips: Dict[str, Any],
    orbit_files: Dict[str, Any],
    ground_point: Tuple[float, float, float],
    swath_hints: Optional[Dict[str, str]] = None,
    min_majority_dates: int = 8,
    min_workable_dates: int = 3,
    redundancy: int = 3,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Real, data-driven burst-timing family classification -- decide
    which real dates to actually use BEFORE full processing, rather
    than discovering a "cross-family" split only after building
    interferograms for all of them.

    Confirmed, concrete motivation: Mexico City's own real archive had
    exactly this pattern -- with only 6 real dates available (from an
    earlier, artificially-capped search), a roughly even 3/3 split
    between two real burst-timing families left no way to build a
    network without bridging directly across a real burst-
    desynchronization boundary (confirmed via
    esd.compute_burst_synchronization: same-family Δt_acq consistently
    <5ms, cross-family consistently >5ms, zero exceptions across all 15
    real pairs). Widening the real search later revealed 33 real,
    consistent-track dates were actually available all along -- more
    than enough to build the whole network from a single, well-
    synchronized family, never touching the minority one at all.

    REAL BUG FIXED, found from a real 6-date run and a direct question
    about it ("why keep dates we already know will decorrelate"): the
    exclusion decision used to gate purely on
    `len(good_dates) >= min_majority_dates`. That's a fixed, absolute
    count -- and for a real archive with only 6 total dates, majority
    can NEVER reach a default of 8, no matter how clean the actual
    split is. The minority-keeping fallback fired unconditionally for
    that whole search window, regardless of whether excluding the
    minority would actually have left a perfectly workable network.

    Fixed to decide on CONNECTIVITY instead: after finding good_dates,
    check directly whether they connect into one component using only
    the real sync data BETWEEN THEMSELVES (reusing the same tested
    select_pairs_for_processing() graph logic, restricted to just the
    majority dates). If they do, and there are at least
    min_workable_dates of them, exclude the minority family entirely --
    known-poor "chaff" gets dropped rather than kept "just in case,"
    regardless of whether the majority count happens to reach the
    (often unreachable, for a small real archive) min_majority_dates
    target. min_majority_dates still gets reported as an informational
    "widen your search for a more robust network" note either way, but
    no longer blocks a real, achievable exclusion.

    HONEST NUANCE, confirmed directly on a real 6-date run (5 real
    S1B dates + 1 real S1A date): excluding the single minority
    (cross-satellite) date does NOT automatically mean the remaining
    majority is bridge-free. In that real run, two of the five S1B
    dates 36 real days apart had their own real >5ms offset (genuine
    orbital-tube drift between two same-satellite passes, unrelated to
    the S1A date at all) -- the majority-only network still needed
    that pair as an internal bridge to stay connected. This function
    now surfaces that directly via the real
    "majority_self_connected"/"majority_internal_bridges" report
    fields, rather than implying exclusion alone guarantees a clean,
    bridge-free result.

    Args:
        dates:           Every real candidate date to classify.
        safe_zips:       {date: real SAFE zip path}.
        orbit_files:     {date: real orbit file path}.
        ground_point:    Real ECEF ground point (e.g. the AOI center).
        swath_hints:     Optional {date: matched sub-swath}, e.g. from
                         extract_consistent_stack()'s own report.
        min_majority_dates: Reported as an informational "a more
                         robust network would have at least this many
                         majority-family dates" note. No longer gates
                         the exclusion decision itself (see the real
                         bug note above for why).
        min_workable_dates: Real minimum majority-family date count
                         required before exclusion is even considered
                         -- below this, there's not enough left for a
                         meaningful SBAS time series regardless of
                         connectivity, so minority dates are kept.
        redundancy:      Passed through to select_pairs_for_processing()
                         for both the full-stack and majority-only
                         connectivity checks.

    Returns:
        (chosen_dates, report) -- chosen_dates is the real, final date
        list to actually use; report has:
          "good_dates": the real majority-family dates.
          "bridge_only_dates": real minority-family dates (only ever
                     needed as bridges).
          "used_majority_only": bool -- whether the minority family was
                     excluded entirely (True) or kept as bridges
                     (False, only when excluding it would have left the
                     majority-only network disconnected, or with fewer
                     than min_workable_dates dates).
          "majority_self_connected": bool -- whether the majority-only
                     dates connect into one component using real sync
                     data among just themselves, independent of the
                     minority dates.
          "majority_internal_bridges": real pairs, both endpoints
                     inside the majority family, that were still needed
                     as bridges to keep the majority-only network
                     connected -- empty if the majority family is
                     genuinely bridge-free on its own.
          "sync_results": the raw, real per-pair BurstSyncResult list,
                     for callers who want the full picture.
    """
    from pygeofetch.insar.timeseries import (
        screen_stack_burst_synchronization, select_pairs_for_processing,
    )

    dates_for_screening = sorted(set(safe_zips) & set(orbit_files) & set(dates))
    sync_results = screen_stack_burst_synchronization(
        dates_for_screening, safe_zips, orbit_files, ground_point, swath_hints=swath_hints,
    )

    _, family_report = select_pairs_for_processing(
        sync_results, dates_for_screening, redundancy=redundancy,
    )

    good_dates = sorted(set(d for pair in family_report["good_pairs"] for d in pair))
    bridge_only_dates = sorted(
        set(d for pair in family_report["bridge_pairs"] for d in pair) - set(good_dates)
    )

    # Does the majority family connect on its own, using only the
    # pairs already discovered by the full-stack screen above, i.e.
    # would excluding the minority actually leave a workable network?
    # This is answered directly from family_report's own good_pairs/
    # bridge_pairs -- restricted to pairs with BOTH endpoints inside
    # the majority family -- rather than by re-deriving it from
    # sync_results. REAL BUG FIXED: the previous version re-filtered
    # sync_results by attribute access (r.date1/r.date2) and re-called
    # select_pairs_for_processing a second time purely to recompute
    # something already fully determined by the first call's own
    # output, which broke the moment sync_results wasn't a real,
    # attribute-bearing BurstSyncResult list (e.g. under test doubles,
    # or any future caller of screen_stack_burst_synchronization that
    # returns a different real representation).
    majority_date_set = set(good_dates)

    def _internal_pairs(pairs):
        return [(d1, d2) for d1, d2 in pairs if d1 in majority_date_set and d2 in majority_date_set]

    def _connected(nodes, edges):
        if not nodes:
            return True
        parent = {d: d for d in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for d1, d2 in edges:
            if d1 in parent and d2 in parent:
                ra, rb = find(d1), find(d2)
                if ra != rb:
                    parent[ra] = rb
        return len({find(d) for d in nodes}) == 1

    internal_good = _internal_pairs(family_report["good_pairs"])
    internal_bridge = _internal_pairs(family_report["bridge_pairs"])

    majority_self_connected = _connected(majority_date_set, internal_good + internal_bridge)
    needed_internal_bridges = (
        not _connected(majority_date_set, internal_good) and majority_self_connected
    )

    majority_internal_bridges = []
    if needed_internal_bridges and internal_bridge:
        # REAL BUG CAUGHT BY A TEST BEFORE SHIPPING: select_pairs_for_
        # processing's own report["bridge_pairs"] is a list of plain
        # (date1, date2) tuples -- no offset included (see its own
        # report construction) -- not BurstSyncResult objects with a
        # .date1 attribute. Look the real offset up from sync_results
        # instead of assuming a richer object was returned, and only
        # do so when an internal bridge was genuinely needed, so a
        # non-standard sync_results representation never breaks the
        # majority-only-and-clean path.
        offset_by_pair = {(r.date1, r.date2): r.sync_offset_ms for r in sync_results}
        offset_by_pair.update({(r.date2, r.date1): r.sync_offset_ms for r in sync_results})
        majority_internal_bridges = [
            (d1, d2, round(offset_by_pair[(d1, d2)], 1)) for d1, d2 in internal_bridge
        ]

    if majority_self_connected and len(good_dates) > min_workable_dates:
        chosen_dates = good_dates
        used_majority_only = True
        below_recommended = len(good_dates) < min_majority_dates
        logger.info(
            "Burst-sync majority family (%d real dates) connects on its "
            "own -- excluding %d minority-family date(s) entirely rather "
            "than keeping known-poor chaff.%s%s",
            len(good_dates), len(bridge_only_dates),
            f" (below the recommended min_majority_dates={min_majority_dates}; "
            f"widen the search window for a more robust network if possible.)"
            if below_recommended else "",
            f" Note: the majority family itself still needed {len(majority_internal_bridges)} "
            f"internal bridge(s) to stay connected: {majority_internal_bridges} -- real "
            f"same-family drift, not related to the excluded minority date(s)."
            if majority_internal_bridges else "",
        )
    else:
        reason = (
            f"only {len(good_dates)} real dates (< min_workable_dates={min_workable_dates})"
            if len(good_dates) < min_workable_dates else
            "disconnected even among themselves"
        )
        logger.info(
            "Burst-sync majority family has %s -- keeping %d minority-family "
            "date(s) as necessary bridges rather than leaving the network "
            "unusable. Widen the real search window if more majority-family "
            "dates should be available.",
            reason, len(bridge_only_dates),
        )
        chosen_dates = sorted(good_dates + bridge_only_dates)
        used_majority_only = False

    report = {
        "good_dates": good_dates,
        "bridge_only_dates": bridge_only_dates,
        "used_majority_only": used_majority_only,
        "majority_self_connected": majority_self_connected,
        "majority_internal_bridges": majority_internal_bridges,
        "sync_results": sync_results,
    }
    return chosen_dates, report


def bbox_to_geojson_path(bbox: Any, name: str = "AOI") -> Path:
    """
    Real AOI bounding box, as a real GeoJSON Polygon FeatureCollection
    written to a real temp file -- the exact preparation
    MapViewer.add_vector() needs (it takes a real file path, not a
    geometry object directly), pulled out of hand-written notebook code
    so it isn't re-derived (with the closing-ring point easy to forget)
    every time a real AOI needs to go on a map.

    Args:
        bbox: Real BoundingBox with min_lon/min_lat/max_lon/max_lat.
        name: Real "name" property on the single Feature written.

    Returns:
        Real Path to the written GeoJSON file (in a fresh temp
        directory, so repeated calls never collide with each other).
    """
    import json
    import tempfile

    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": name},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [bbox.min_lon, bbox.min_lat], [bbox.max_lon, bbox.min_lat],
                    [bbox.max_lon, bbox.max_lat], [bbox.min_lon, bbox.max_lat],
                    [bbox.min_lon, bbox.min_lat],
                ]],
            },
        }],
    }
    path = Path(tempfile.mkdtemp()) / "aoi.geojson"
    path.write_text(json.dumps(geojson))
    return path


def preview_search_results(
    bbox: Any,
    results: List[Any],
    zoom: int = 10,
    basemap: str = "SATELLITE",
    aoi_layer_name: str = "AOI",
    aoi_style: Optional[Dict[str, Any]] = None,
    show: bool = True,
    map_viewer_cls: Optional[Any] = None,
) -> Any:
    """
    Real map preview of a real AOI plus real search-result footprints --
    the exact sequence (build the AOI GeoJSON, construct a MapViewer
    centered on the AOI, add a real satellite basemap, add the real
    search-result footprints, add the real AOI outline, show it) that
    was hand-written for every real project so far. Uses only
    MapViewer's own already-confirmed public methods (add_basemap,
    add_vector, add_search_results, show) -- nothing here depends on
    MapViewer's internals.

    Args:
        bbox:            Real AOI BoundingBox -- also used to center
                         the map.
        results:         Real search results (or any object
                         MapViewer.add_search_results() accepts).
        zoom:            Real initial zoom level.
        basemap:         Real basemap name, passed straight to
                         add_basemap().
        aoi_layer_name:  Real layer name for the AOI outline.
        aoi_style:       Real style dict for the AOI outline. None uses
                         {"color": "yellow", "fillOpacity": 0, "weight": 3}
                         -- the same style used in every real project
                         this was consolidated from.
        show:            Whether to call .show() before returning. True
                         by default; set False to add more real layers
                         (e.g. extra point markers) before showing.
        map_viewer_cls:  Real MapViewer class to construct. None (the
                         real default) uses pygeofetch.viz.map.MapViewer
                         -- overridable for testing with a fake, or for
                         a caller that wants a differently-configured
                         subclass.

    Returns:
        The real, constructed (and, unless show=False, already-shown)
        map viewer instance, so a caller can add further real layers
        (e.g. point markers for specific real locations) before calling
        .show() themselves.
    """
    if map_viewer_cls is None:
        from pygeofetch.viz.map import MapViewer as map_viewer_cls

    if aoi_style is None:
        aoi_style = {"color": "yellow", "fillOpacity": 0, "weight": 3}

    aoi_path = bbox_to_geojson_path(bbox, name=aoi_layer_name)

    center = ((bbox.min_lat + bbox.max_lat) / 2, (bbox.min_lon + bbox.max_lon) / 2)
    mv = map_viewer_cls(center=center, zoom=zoom)
    mv.add_basemap(basemap)
    mv.add_search_results(results)
    mv.add_vector(str(aoi_path), layer_name=aoi_layer_name, style=aoi_style)

    if show:
        mv.show()

    return mv