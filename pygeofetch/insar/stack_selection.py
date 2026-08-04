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