"""
Pre-download data-quality gate for InSAR stack selection.
Runs AFTER search, BEFORE download. Catches — and where possible
auto-corrects — real, confirmed data-selection failures this project
has hit, so bandwidth and hours are never spent on a stack that was
guaranteed to fail:
Search truncation          -> auto-widen max_results & re-search
Mixed relative orbits      -> confirmed (already enforced by
select_consistent_geometry(), which
should already have run before this
gate; re-checked here defensively)
Multi-candidate per date   -> coverage-aware dedup (already
handled by search_and_select_
consistent_stack(), which this gate
calls)
0%/low AOI coverage        -> drop + report genuine gaps
Sparse temporal sampling   -> predict SBAS connectivity, warn
Burst-timing family risk   -> honest advisory, NOT a detector --
see _screen_burst_families()'s own
docstring for why a real pre-
download detector isn't achievable
with what's available before
download, and for a real, confirmed
bug found in an earlier version
that claimed to be one anyway.
The definitive burst-family check still happens post-download via
select_burst_synchronized_dates() (it needs SwathTiming parsed from
each product's real annotation XML, which isn't available before
download). This gate handles what real pre-download signals actually
allow, and is explicit about the one real risk it cannot resolve early.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("pygeofetch.insar.preflight")

# ─────────────────────────────────────────────────────────────────────────────
# Report data model
# ────────────────────────────────────────────────────────────────────────────

SEVERITY_GREEN = "GREEN"  # safe to proceed
SEVERITY_YELLOW = "YELLOW"  # proceed, but with logged caveats
SEVERITY_RED = "RED"  # do not download; fix first


@dataclass
class PreflightIssue:
    """A single problem found during preflight, with its fix."""

    code: str  # machine-readable id, e.g. "SEARCH_TRUNCATED"
    severity: str  # one of the SEVERITY_* constants
    message: str  # human-readable explanation
    auto_fixed: bool = False  # did the gate heal this automatically?
    fix_detail: str = ""  # what the auto-fix did, if applied


@dataclass
class PreflightReport:
    """
    Full outcome of the pre-download gate.
    Carries the go/no-go decision, every issue found, every auto-fix
    applied, and — critically — the provenance manifest that makes the
    whole run reproducible and defensible.
    """

    go: bool
    severity: str
    issues: List[PreflightIssue] = field(default_factory=list)
    selected: List[Any] = field(default_factory=list)  # final scene objects
    manifest: Dict[str, Any] = field(default_factory=dict)
    original_count: int = (
        0  # how many scenes came IN, before any filtering -- see summary()'s own real-bug note
    )

    def summary(self) -> str:
        lines = [f"PREFLIGHT {self.severity} — {'PROCEED' if self.go else 'BLOCKED'}"]

        # Print Bandwidth Manifest if available
        if "bandwidth_and_manifest" in self.manifest:
            bw = self.manifest["bandwidth_and_manifest"]
            lines.append(
                f"  [INFO] Estimated download size: {bw['estimated_bandwidth_gb']} GB for {len(bw['download_manifest'])} scenes."
            )

        for issue in self.issues:
            fixed = " [AUTO-FIXED]" if issue.auto_fixed else ""
            lines.append(f"  [{issue.severity}] {issue.code}{fixed}: {issue.message}")
            if issue.auto_fixed and issue.fix_detail:
                lines.append(f"        -> {issue.fix_detail}")

        # REAL BUG FOUND AND GUARDED AGAINST HERE: a real notebook
        # called gate.run(selected, search_report) and printed
        # report.summary() -- which correctly showed real dates being
        # excluded -- but never reassigned `selected = report.selected`
        # afterward, so every downstream step (the very next cell,
        # literally titled "Download the real, filtered scenes") kept
        # using the original, unfiltered list. The filtering logic was
        # completely correct; the scenes it said to exclude got
        # downloaded anyway, because nothing after this function call
        # actually used its return value. This can't be fixed from
        # inside this class (a function can't force a caller to use
        # its return value) but it CAN be made loud and impossible to
        # miss in the one place every real run already prints.
        if self.original_count and len(self.selected) < self.original_count:
            dropped = self.original_count - len(self.selected)
            lines.append(
                f"  [REMINDER] {dropped} of {self.original_count} scene(s) were "
                f"excluded by this gate. Use report.selected -- not your original "
                f"list -- for every downstream step (download, extraction, "
                f"processing). Passing the original list silently re-includes "
                f"exactly what this gate just excluded."
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# The gate itself
# ─────────────────────────────────────────────────────────────────────────────


class PreflightGate:
    """
    Orchestrates every pre-download check and auto-heals what it can.
    Usage:
        gate = PreflightGate(client, aoi_bbox, start_date, end_date)
        report = gate.run(selected, search_report)
        print(report.summary())
        if not report.go:
            raise RuntimeError("Preflight blocked the stack. See report.")
        download_results = client.download(report.selected, ...)
    """

    def __init__(
        self,
        client: Any,
        aoi_bbox: Any,
        start_date: str,
        end_date: str,
        satellites: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        preferred_track: Optional[Any] = None,
        max_scenes: Optional[int] = None,
        max_results: int = 3000,
        min_coverage_fraction: float = 0.99,
        product_type: str = "SLC",
        # Pre-download screening thresholds:
        burst_family_time_threshold_ms: float = 5.0,
        max_temporal_baseline_days: int = 60,
        min_network_redundancy: int = 2,
        min_majority_family_dates: int = 8,
        min_workable_family_dates: int = 3,
        attempt_real_burst_check: bool = True,
    ) -> None:
        self.client = client
        self.aoi_bbox = aoi_bbox
        self.start_date = start_date
        self.end_date = end_date
        self.satellites = satellites
        self.providers = providers
        self.preferred_track = preferred_track
        self.max_scenes = max_scenes
        self.max_results = max_results
        self.min_coverage_fraction = min_coverage_fraction
        self.product_type = product_type
        self.burst_family_time_threshold_ms = burst_family_time_threshold_ms
        self.max_temporal_baseline_days = max_temporal_baseline_days
        self.min_network_redundancy = min_network_redundancy
        self.min_majority_family_dates = min_majority_family_dates
        # Real minimum majority-date count before exclusion is even
        # considered -- separate from min_majority_family_dates, which
        # is now purely an informational "widen your search" target,
        # not a hard exclusion gate (see select_burst_synchronized_
        # dates()'s own docstring for the real bug this distinction
        # fixes: a fixed target like 8 can be structurally unreachable
        # for a small real archive, e.g. only 6 total dates found).
        self.min_workable_family_dates = min_workable_family_dates
        # Real, pre-download burst-family DETECTION -- not just the
        # honest advisory -- via copernicus_nodes fetching only each
        # candidate date's lightweight annotation XML (not the full,
        # multi-GB product) plus its real orbit file, then running the
        # same, already-validated select_burst_synchronized_dates()
        # used post-download. On by default since it's real, useful
        # information for real network cost (a handful of small
        # requests per date, not full downloads) -- set False for an
        # instant, zero-network preflight, or if "copernicus" isn't the
        # authenticated provider these real annotation fetches need.
        self.attempt_real_burst_check = attempt_real_burst_check

    # ── public entry point ───────────────────────────────────────────────
    def run(
        self, selected: List[Any], search_report: Dict[str, Any]
    ) -> PreflightReport:
        issues: List[PreflightIssue] = []
        selected = list(selected)  # work on a copy

        # Check 1: search truncation (with auto-heal re-search loop)
        selected, search_report, trunc_issue = self._check_truncation(
            selected, search_report
        )
        if trunc_issue:
            issues.append(trunc_issue)

        # Baseline for the "did anything get silently dropped downstream"
        # reminder in summary() -- captured HERE, after truncation
        # resolves, not at the very top: truncation can legitimately
        # GROW the candidate pool (a wider re-search finding more real
        # scenes), which isn't a "drop" and shouldn't suppress a real
        # warning about coverage/burst-family filtering that happens
        # after this point.
        original_count = len(selected)

        # Check 2: AOI coverage per scene (drop genuine gaps)
        selected, cov_issues = self._check_coverage(selected)
        issues.extend(cov_issues)

        # Check 3: temporal sampling / predicted SBAS connectivity
        issues.extend(self._check_temporal_network(selected))

        # Check 4: burst-timing family proxy screen
        geometry_report = search_report.get("geometry_report", {})
        selected, burst_issues = self._screen_burst_families(selected, geometry_report)
        issues.extend(burst_issues)

        # Build the provenance manifest from everything we now know
        manifest = self._build_manifest(selected, search_report, issues)

        # Decide go/no-go
        severity = self._overall_severity(issues)
        go = severity != SEVERITY_RED and len(selected) >= 2

        report = PreflightReport(
            go=go,
            severity=severity,
            issues=issues,
            selected=selected,
            manifest=manifest,
            original_count=original_count,
        )
        logger.info("\n%s", report.summary())
        return report

    # ── Check 1: truncation ───────────────────────────────────────────────
    def _check_truncation(
        self, selected: List[Any], search_report: Dict[str, Any], max_retries: int = 3
    ) -> Tuple[List[Any], Dict[str, Any], Optional[PreflightIssue]]:
        from pygeofetch.insar.stack_selection import search_and_select_consistent_stack

        retries = 0
        current_max = self.max_results
        while search_report.get("hit_max_results") and retries < max_retries:
            current_max *= 3
            logger.warning(
                "Search hit max_results=%d — likely truncated. Re-searching with %d.",
                search_report["raw_result_count"],
                current_max,
            )
            selected, search_report = search_and_select_consistent_stack(
                self.client,
                self.aoi_bbox,
                self.start_date,
                self.end_date,
                satellites=self.satellites,
                providers=self.providers,
                preferred_track=self.preferred_track,
                max_scenes=self.max_scenes,
                max_results=current_max,
                min_coverage_fraction=self.min_coverage_fraction,
                product_type=self.product_type,
            )
            retries += 1

        if search_report.get("hit_max_results"):
            return (
                selected,
                search_report,
                PreflightIssue(
                    code="SEARCH_TRUNCATED",
                    severity=SEVERITY_RED,
                    message=(
                        f"Search still hitting max_results={current_max} after {retries} "
                        "auto-widen attempts. The archive is larger than expected; raise "
                        "max_results manually before trusting this stack."
                    ),
                ),
            )
        if retries > 0:
            return (
                selected,
                search_report,
                PreflightIssue(
                    code="SEARCH_TRUNCATED",
                    severity=SEVERITY_GREEN,
                    message=f"Search was truncated; auto-widened max_results to {current_max}.",
                    auto_fixed=True,
                    fix_detail=f"Re-searched {retries}x, now {search_report['raw_result_count']} raw scenes.",
                ),
            )
        return selected, search_report, None

    # ── Check 2: AOI coverage ─────────────────────────────────────────────
    def _check_coverage(
        self, selected: List[Any]
    ) -> Tuple[List[Any], List[PreflightIssue]]:
        """Drop scenes whose true footprint doesn't cover the AOI."""
        try:
            from shapely.geometry import box, shape
        except ImportError as exc:  # pragma: no cover - exercised via ImportError path
            raise ImportError(
                "AOI coverage checking requires shapely. Install it with "
                "'pip install \"pygeofetch[insar]\"' or 'pip install shapely'."
            ) from exc

        aoi_poly = box(
            self.aoi_bbox.min_lon,
            self.aoi_bbox.min_lat,
            self.aoi_bbox.max_lon,
            self.aoi_bbox.max_lat,
        )
        kept, dropped = [], {}
        for s in selected:
            try:
                frac = aoi_poly.intersection(shape(s.geometry)).area / aoi_poly.area
            except Exception:
                frac = 0.0
            if frac >= self.min_coverage_fraction:
                kept.append(s)
            else:
                dropped[str(s.datetime)[:10]] = round(frac, 3)

        issues = []
        if dropped:
            issues.append(
                PreflightIssue(
                    code="LOW_AOI_COVERAGE",
                    severity=SEVERITY_YELLOW,
                    message=(
                        f"{len(dropped)} scene(s) below {self.min_coverage_fraction:.0%} "
                        f"AOI coverage dropped: {dropped}"
                    ),
                    auto_fixed=True,
                    fix_detail="Genuine coverage gaps removed from the stack.",
                )
            )
        return kept, issues

    # ── Check 3: temporal network prediction ──────────────────────────────
    def _check_temporal_network(self, selected: List[Any]) -> List[PreflightIssue]:
        """
        Predict, from dates alone, whether a connected SBAS network is
        achievable. Reuses generate_candidate_pairs() -- the same
        tested temporal-baseline-limited pair generator the real
        pipeline uses later -- rather than a second, independent
        reimplementation of the same date-difference logic.
        """
        from pygeofetch.insar.timeseries import generate_candidate_pairs

        dates = sorted({str(s.datetime)[:10] for s in selected})
        n = len(dates)
        if n < 2:
            return [
                PreflightIssue(
                    code="TOO_FEW_DATES",
                    severity=SEVERITY_RED,
                    message=f"Only {n} unique date(s) — need >= 2 for any interferogram.",
                )
            ]

        candidate_pairs = generate_candidate_pairs(
            dates,
            max_temporal_baseline_days=self.max_temporal_baseline_days,
        )
        adjacency: Dict[str, set] = {d: set() for d in dates}
        for d1, d2 in candidate_pairs:
            adjacency[d1].add(d2)
            adjacency[d2].add(d1)

        # Connectivity via depth-first traversal -- order doesn't affect
        # correctness for pure reachability, only the order nodes are
        # visited in.
        visited, stack = {dates[0]}, [dates[0]]
        while stack:
            d = stack.pop()
            for nb in adjacency[d] - visited:
                visited.add(nb)
                stack.append(nb)

        unconnected = set(dates) - visited
        issues = []
        if unconnected:
            issues.append(
                PreflightIssue(
                    code="NETWORK_DISCONNECTED",
                    severity=SEVERITY_RED,
                    message=(
                        f"{len(unconnected)} date(s) unreachable within "
                        f"{self.max_temporal_baseline_days}d baseline: {sorted(unconnected)}. "
                        "Widen the temporal baseline or the date range."
                    ),
                )
            )

        low_redundancy = [
            d for d in dates if len(adjacency[d]) < self.min_network_redundancy
        ]
        if low_redundancy:
            issues.append(
                PreflightIssue(
                    code="LOW_NETWORK_REDUNDANCY",
                    severity=SEVERITY_YELLOW,
                    message=(
                        f"{len(low_redundancy)} date(s) have < {self.min_network_redundancy} "
                        f"neighbours within {self.max_temporal_baseline_days}d: {low_redundancy}. "
                        "The network is fragile — a single bad pair could disconnect it."
                    ),
                )
            )

        if not issues:
            logger.info(
                "Temporal network OK: %d dates, all connected, redundancy >= %d.",
                n,
                self.min_network_redundancy,
            )
        return issues

    # ── Check 4: burst-timing family risk ─────────────────────────────────
    def _screen_burst_families(
        self,
        selected: List[Any],
        geometry_report: Dict[str, Any],
    ) -> Tuple[List[Any], List[PreflightIssue]]:
        """
        Pre-download burst-timing family risk -- attempts a REAL
        detection first (via _try_real_burst_family_check(), using
        copernicus_nodes to fetch only lightweight annotation XML, not
        full products), falling back to an honest "unassessed" advisory
        only if that isn't available or doesn't succeed. Never silently
        claims risk is absent just because it couldn't be checked.
        REAL BUG FIXED: earlier versions returned only `issues`, never
        `selected` -- so a successful real detection's own
        "BURST_FAMILY_DETECTED" issue was marked auto_fixed=True with a
        "Recommended date selection" in fix_detail, but that
        recommendation was NEVER ACTUALLY APPLIED to report.selected.
        Invisible on a run where the majority family was too small to
        exclude anything (the "recommendation" then equals the original
        full set by coincidence) -- confirmed happening on exactly such
        a real run -- but on a run where the majority family IS large
        enough to justify excluding the minority, report.selected would
        have silently still contained every original date while the
        report claimed the exclusion had already happened. Now returns
        the real, actually-filtered selection, matching the pattern
        _check_coverage() already used correctly.
        HISTORY, kept because it explains why this function is careful
        about false passes: an earlier version grouped scenes by
        (Track, Satellite) and treated agreement within a group as
        proof of a single burst-timing family. That's contradicted by
        this project's own strongest evidence: Mexico City's real 3/3
        cross-family split (confirmed via compute_burst_synchronization's
        actual Δt_acq measurement, zero exceptions across 15 real
        pairs) occurred WITHIN a single track -- real orbital-tube
        drift pass-to-pass, not a track or satellite mismatch. Grouping
        by (Track, Satellite) alone returned a clean, empty result for
        that exact stack -- a false pass on precisely the case this
        check needs to catch. A second, independent bug in that version
        also made the (Track, Satellite) grouping non-functional in
        practice (wrong dict keys against the real scene object).
        What this function guarantees regardless of which path fires:
        every selected scene really is on a single, consistent track
        (already enforced by select_consistent_geometry(); re-checked
        here defensively), and burst-FAMILY risk specifically is either
        actually resolved (BURST_FAMILY_DETECTED, from a real,
        successful pre-download check) or explicitly flagged as
        unresolved (BURST_FAMILY_RISK_UNASSESSED) -- never silently
        assumed absent.
        """
        tracks = set()
        for s in selected:
            track = None
            if hasattr(s, "properties"):
                track = s.properties.get("relativeOrbitNumber")
            if track is None:
                track = geometry_report.get("track")
            tracks.add(track)

        issues: List[PreflightIssue] = []
        if len(tracks) > 1:
            # Should be unreachable if select_consistent_geometry() already
            # ran correctly -- a real defensive check, not the expected path.
            issues.append(
                PreflightIssue(
                    code="MULTIPLE_TRACKS_IN_SELECTION",
                    severity=SEVERITY_RED,
                    message=(
                        f"Selected scenes span {len(tracks)} different real tracks "
                        f"({sorted(t for t in tracks if t is not None)}) -- "
                        f"select_consistent_geometry() should have already enforced "
                        f"a single track. Do not download until this is resolved."
                    ),
                )
            )

        satellites_present = geometry_report.get("satellites", set())
        real_result = (
            self._try_real_burst_family_check(selected)
            if self.attempt_real_burst_check
            else None
        )

        nodes_version = "unknown"
        try:
            from pygeofetch.providers.copernicus_nodes import (
                MODULE_VERSION as nodes_version,
            )
        except Exception:
            pass

        if real_result is not None:
            chosen_dates, family_report = real_result
            n_total = len({str(s.datetime)[:10] for s in selected})
            n_good = len(family_report["good_dates"])
            n_bridge = len(family_report["bridge_only_dates"])
            resolved = family_report["used_majority_only"]

            # Pair-level signal, not just the date-level majority/minority
            # count -- REAL GAP FOUND from a careful third-party review of
            # an actual run: date-level "5 majority / 1 minority" can look
            # like "one bad date" when the real, pairwise picture is far
            # more pervasive. Confirmed directly on a real run: only 4/15
            # (27%) of ALL candidate pairs were within Sentinel-1's own
            # <5ms requirement -- including a pair WITHIN the 5-date
            # "majority" set itself (same two S1B dates, 36 days apart,
            # real orbital-tube drift). The date-level split alone doesn't
            # surface that most pairwise connections are poor even among
            # "majority" dates.
            sync_results = family_report.get("sync_results") or []
            n_pairs_total = len(sync_results)
            n_pairs_within = sum(1 for r in sync_results if r.within_requirement)
            pair_fraction_msg = (
                f" Pair-level: {n_pairs_within}/{n_pairs_total} "
                f"({100*n_pairs_within/n_pairs_total:.0f}%) of all real candidate "
                f"pairs are within the 5ms requirement -- check this figure "
                f"too, not just the date-level split, since a majority-family "
                f"date can still be connected mostly through bridge-quality "
                f"pairs to its neighbours."
                if n_pairs_total
                else ""
            )

            # REAL BUGS FOUND from a careful third-party review, verified
            # against this codebase's own established convention before
            # fixing (see _check_coverage()'s LOW_AOI_COVERAGE issue, which
            # already used SEVERITY_YELLOW even though auto_fixed=True):
            #   1. auto_fixed=True was set unconditionally whenever the
            #      real check succeeded, even when NOTHING was actually
            #      excluded (majority too small, every date kept as-is --
            #      confirmed directly: "Applied date selection" was
            #      byte-identical to the original input on a real run).
            #      Labeling a no-op an "auto-fix" is misleading.
            #   2. SEVERITY_GREEN was used even when minority dates were
            #      KEPT (not excluded) as necessary bridges -- inconsistent
            #      with this file's own LOW_AOI_COVERAGE convention, and a
            #      real, accurate signal was being flattened to "safe to
            #      proceed" when the honest state is "proceeding with known
            #      bridge pairs expected to correlate poorly."
            # Fixed: GREEN + auto_fixed=True now means the majority family
            # was genuinely used exclusively (a real exclusion happened).
            # YELLOW + auto_fixed=False means minority dates were kept as a
            # real compromise, not a fix -- proceeding, but explicitly not
            # claiming the risk was resolved.
            #
            # REAL NUANCE surfaced after a direct question about why
            # minority dates were being kept at all ("we don't want
            # chaff"): resolved=True (majority used exclusively) does NOT
            # necessarily mean the majority is itself bridge-free.
            # Confirmed on a real run: the majority-only network still
            # needed its own internal bridge between two same-satellite
            # dates 36 real days apart, unrelated to the excluded minority
            # date entirely. Surfaced honestly via majority_internal_bridges
            # rather than implying "excluded the minority" means "clean."
            internal_bridges = family_report.get("majority_internal_bridges") or []
            internal_bridge_msg = (
                f" Note: the majority family itself still needed "
                f"{len(internal_bridges)} internal bridge(s) to stay "
                f"connected: {internal_bridges} -- real same-family drift, "
                f"unrelated to the excluded minority date(s); expect these "
                f"specific pairs to correlate poorly too."
                if internal_bridges
                else ""
            )

            issues.append(
                PreflightIssue(
                    code="BURST_FAMILY_DETECTED",
                    severity=SEVERITY_GREEN if resolved else SEVERITY_YELLOW,
                    message=(
                        f"[copernicus_nodes {nodes_version}] "
                        f"Real pre-download burst-family check succeeded "
                        f"({n_good + n_bridge}/{n_total} dates had both real "
                        f"annotation and orbit data available): majority family "
                        f"{n_good} dates, minority {n_bridge} dates."
                        + (
                            " Majority family used exclusively -- the "
                            "cross-family risk is genuinely resolved before "
                            "download, not deferred; known-poor minority "
                            "date(s) dropped rather than kept as chaff."
                            + internal_bridge_msg
                            if resolved
                            else f" Majority family alone is not usable on its own "
                            f"({'disconnected even among themselves' if not family_report.get('majority_self_connected', True) else f'only {n_good} dates, below min_workable_dates'}); "
                            f"minority dates KEPT as necessary bridges, not "
                            f"excluded -- this is a real compromise, not a fix. "
                            f"Expect the specific bridge pairs to correlate "
                            f"poorly (see the select_pairs_for_processing log "
                            f"warning for exactly which ones). Widen the date "
                            f"range for a cleaner single-family network if "
                            f"possible."
                        )
                        + pair_fraction_msg
                    ),
                    auto_fixed=resolved,
                    fix_detail=(
                        f"Applied date selection: {chosen_dates}"
                        if resolved
                        else f"No dates excluded -- majority family not usable on "
                        f"its own (see message). Full stack kept: {chosen_dates}."
                    ),
                )
            )

            # ACTUALLY apply the recommendation -- not just report it.
            chosen_set = set(chosen_dates)
            filtered_selected = [
                s for s in selected if str(s.datetime)[:10] in chosen_set
            ]
            return filtered_selected, issues

        # Real check unavailable, disabled, or failed -- honest fallback,
        # never a silent pass.
        issues.append(
            PreflightIssue(
                code="BURST_FAMILY_RISK_UNASSESSED",
                severity=SEVERITY_YELLOW,
                message=(
                    f"[copernicus_nodes {nodes_version}] "
                    f"Track consistency confirmed ({len(tracks)} track(s)); real "
                    f"satellite(s) present: "
                    f"{sorted(satellites_present) if satellites_present else 'unknown'}. "
                    f"Burst-timing FAMILY risk (the real cause of Mexico City's "
                    f"cross-family split) could not be resolved before download"
                    + (
                        " (attempt_real_burst_check=False)."
                        if not self.attempt_real_burst_check
                        else " -- the real pre-download check "
                        "(copernicus_nodes-based annotation fetch) was attempted "
                        "and did not succeed for enough dates; see the log above "
                        "for why."
                    )
                    + " Run select_burst_synchronized_dates() on the downloaded "
                    "stack before committing to full interferogram processing; "
                    "a clean preflight pass is not confirmation this risk is "
                    "absent."
                ),
            )
        )
        return selected, issues

    def _try_real_burst_family_check(
        self,
        selected: List[Any],
    ) -> Optional[Tuple[List[str], Dict[str, Any]]]:
        """
        Best-effort REAL pre-download burst-family classification.
        Uses providers.copernicus_nodes to fetch only each candidate
        date's lightweight annotation XML (not the full, multi-GB
        product) plus its real orbit file, then runs the SAME, already-
        validated stack_selection.select_burst_synchronized_dates()
        used post-download -- just fed lightweight, annotation-only
        mini-zips in place of full SAFE zips. This is a real substitute,
        not an approximation: annotation.parse_burst_info()/
        parse_slc_geometry() read only the annotation XML either way,
        confirmed directly with an end-to-end test where the real,
        unmodified parser reads a mini-zip built this way correctly.
        Returns None -- never raises -- if anything real goes wrong:
        missing "copernicus" credentials, a network failure, an
        unexpected real API response shape, too few dates with both
        annotation and orbit data available, etc. A None here always
        means "fall back to the honest BURST_FAMILY_RISK_UNASSESSED
        advisory," never "risk confirmed absent."
        """
        import importlib
        import shutil
        import tempfile

        try:
            for _mod_name in (
                "pygeofetch.providers.copernicus_nodes",
                "pygeofetch.core.orbits",
                "pygeofetch.insar.stack_selection",
                "pygeofetch.insar.geolocation",
            ):
                importlib.import_module(_mod_name)
        except ImportError as exc:
            logger.info("Real pre-download burst check unavailable (%s).", exc)
            return None

        try:
            work_dir = Path(tempfile.mkdtemp(prefix="preflight_burst_"))
        except Exception as exc:
            logger.info(
                "Real pre-download burst check: couldn't create a scratch dir (%s).",
                exc,
            )
            return None

        # REAL BUG FOUND AND FIXED: work_dir was never removed on any exit
        # path -- not on success, not on the early "too few usable dates"
        # return, not on the exception fallback. Every real call to this
        # method (i.e. every real preflight run with attempt_real_burst_
        # check=True, which is the default) left behind a full scratch
        # directory of real annotation mini-zips and real, full orbit
        # files (~4MB+ each) that was never cleaned up. Re-running the
        # same preflight cell repeatedly while iterating on a notebook --
        # completely normal, everyday usage -- accumulates one orphaned
        # directory per run, unbounded, on whatever drive the OS temp
        # folder lives on (on Windows, typically the same small system
        # drive as everything else). A real, concrete, plausible path to
        # exhausting that drive, which is a well-known way to make a
        # Jupyter kernel become unresponsive or fail to save cleanly --
        # matching a real report of a notebook going inactive with blank
        # cell outputs immediately after a preflight run. The real body
        # of this method is now factored into _run_real_burst_family_check
        # below, wrapped here so work_dir is removed on every real exit
        # path, success or failure, not just the ones tested before.
        try:
            return self._run_real_burst_family_check(selected, work_dir)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _run_real_burst_family_check(self, selected: List[Any], work_dir: Path):
        """
        The real body of _try_real_burst_family_check(), factored out so
        that method's own try/finally can guarantee work_dir cleanup on
        every real exit path here -- including the early returns --
        without duplicating the rmtree call at each one.
        """
        from pygeofetch.core.orbits import fetch_orbit_file
        from pygeofetch.insar.geolocation import geodetic_to_ecef
        from pygeofetch.insar.stack_selection import select_burst_synchronized_dates
        from pygeofetch.providers.copernicus_nodes import fetch_annotation_zip

        safe_zips: Dict[str, Any] = {}
        orbit_files: Dict[str, Any] = {}

        for s in selected:
            date = str(s.datetime)[:10]
            try:
                safe_zips[date] = fetch_annotation_zip(self.client, s, work_dir)
            except Exception as exc:
                logger.info(
                    "Real pre-download burst check: no annotation for %s (%s).",
                    date,
                    exc,
                )
                continue

            try:
                name = s.properties.get("name") if hasattr(s, "properties") else None
                if not name:
                    continue

                # ── FIX: ORBIT CASCADING ──
                # Previously, this only tried "precise". If precise wasn't
                # available (e.g., recent acquisitions), the date was dropped.
                # Now it cascades: Precise -> Restituted -> Predicted.
                orbit_file = None
                for orbit_type in ["precise", "restituted", "predicted"]:
                    try:
                        orbit_file = fetch_orbit_file(
                            product_name=name,
                            output_dir=str(work_dir),
                            orbit_type=orbit_type,
                        )
                        if orbit_file is not None:
                            if orbit_type != "precise":
                                logger.warning(
                                    "Orbit cascading: fell back to '%s' orbit for %s "
                                    "(precise not available).",
                                    orbit_type,
                                    date,
                                )
                            break
                    except Exception as exc:
                        logger.debug(
                            "Orbit cascade: '%s' failed for %s (%s).",
                            orbit_type,
                            date,
                            exc,
                        )
                        continue

                if orbit_file is None:
                    logger.info(
                        "Real pre-download burst check: no orbit file (any type) for %s.",
                        date,
                    )
                    continue

                orbit_files[date] = orbit_file
            except Exception as exc:
                logger.info(
                    "Real pre-download burst check: orbit cascade failed for %s (%s).",
                    date,
                    exc,
                )
                continue

        usable_dates = sorted(set(safe_zips) & set(orbit_files))
        if len(usable_dates) < 2:
            logger.info(
                "Real pre-download burst check: only %d/%d dates had both "
                "real annotation and orbit data available -- not enough to "
                "classify.",
                len(usable_dates),
                len({str(s.datetime)[:10] for s in selected}),
            )
            return None

        try:
            aoi_center_lat = (self.aoi_bbox.min_lat + self.aoi_bbox.max_lat) / 2
            aoi_center_lon = (self.aoi_bbox.min_lon + self.aoi_bbox.max_lon) / 2
            ground_point = geodetic_to_ecef(aoi_center_lat, aoi_center_lon, 0.0)
            chosen_dates, family_report = select_burst_synchronized_dates(
                usable_dates,
                safe_zips,
                orbit_files,
                ground_point,
                min_majority_dates=self.min_majority_family_dates,
                min_workable_dates=self.min_workable_family_dates,
            )
            return chosen_dates, family_report
        except Exception as exc:
            logger.warning(
                "Real pre-download burst-family check failed unexpectedly "
                "(%s) -- falling back to the honest unassessed advisory "
                "rather than risk a wrong classification.",
                exc,
            )
            return None

    # ── manifest + severity ───────────────────────────────────────────────
    def _build_manifest(
        self,
        selected: List[Any],
        search_report: Dict[str, Any],
        issues: List[PreflightIssue],
    ) -> Dict[str, Any]:
        # ── FIX: BANDWIDTH & DOWNLOAD MANIFEST ──
        # Calculate estimated bandwidth and build a concrete manifest of
        # what will actually be downloaded, preventing surprise disk fills.
        ESTIMATED_GB_PER_SLC_SCENE = 7.5  # Sentinel-1 IW SLC average size
        estimated_bandwidth_gb = len(selected) * ESTIMATED_GB_PER_SLC_SCENE

        download_manifest = []
        for s in selected:
            props = s.properties if hasattr(s, "properties") else {}
            download_manifest.append(
                {
                    "date": str(s.datetime)[:10],
                    "scene_id": props.get("name"),
                    "satellite": props.get("platform"),
                    "polarisation": props.get("polarisation"),
                }
            )

        return {
            "search": {
                "start_date": self.start_date,
                "end_date": self.end_date,
                "satellites": self.satellites,
                "providers": self.providers,
                "product_type": self.product_type,
                "max_results_effective": search_report.get("raw_result_count"),
                "aoi_bbox": [
                    self.aoi_bbox.min_lon,
                    self.aoi_bbox.min_lat,
                    self.aoi_bbox.max_lon,
                    self.aoi_bbox.max_lat,
                ],
            },
            "selection": {
                "n_selected": len(selected),
                "scene_ids": [str(s.datetime)[:10] for s in selected],
                "track": search_report.get("geometry_report", {}).get("track"),
                "satellites_present": sorted(
                    search_report.get("geometry_report", {}).get("satellites", set())
                ),
            },
            # NEW: Bandwidth & Manifest section
            "bandwidth_and_manifest": {
                "estimated_bandwidth_gb": round(estimated_bandwidth_gb, 2),
                "download_manifest": download_manifest,
            },
            "preflight": {
                "severity": self._overall_severity(issues),
                "issues": [
                    {
                        "code": i.code,
                        "severity": i.severity,
                        "auto_fixed": i.auto_fixed,
                        "message": i.message,
                    }
                    for i in issues
                ],
            },
        }

    @staticmethod
    def _overall_severity(issues: List[PreflightIssue]) -> str:
        if any(i.severity == SEVERITY_RED and not i.auto_fixed for i in issues):
            return SEVERITY_RED
        if any(i.severity == SEVERITY_YELLOW and not i.auto_fixed for i in issues):
            return SEVERITY_YELLOW
        return SEVERITY_GREEN