"""
Real per-burst-overlap Enhanced Spectral Diversity (ESD), replacing the
previous whole-image approximation.

Formula confirmed against the real, correctly-identified Sentinel-1-
specific source (fetched and read directly, not just cited secondhand):
  Yagüe-Martínez, N., Prats-Iraola, P., Rodríguez González, F., et al.
    (2016), "Interferometric Processing of Sentinel-1 TOPS Data," IEEE
    TGRS 54(4), 2220-2234. https://elib.dlr.de/100050/1/07390052.pdf
    (NOTE: this is frequently miscited elsewhere -- including an
    earlier version of this docstring -- as "Prats-Iraola et al. 2012";
    that 2012 paper, Prats-Iraola et al., "TOPS interferometry with
    TerraSAR-X," IEEE TGRS 50(8), 3179-3188, is about TerraSAR-X, not
    Sentinel-1. Both share Prats-Iraola as an author, which is likely
    the source of the confusion.)
  Scheiber, R. & Moreira, A. (2000), "Coregistration of interferometric
    SAR images using spectral diversity", IEEE TGRS.

The azimuth shift (in seconds) is:
    delta_t = delta_phi / (2 * pi * delta_f_ovl)
where delta_phi is the double-difference phase between the forward-
and backward-looking interferograms in the overlap, and delta_f_ovl
is the Doppler centroid difference between the two looks.

SENTINEL1_IW_DELTA_F_OVL_HZ: precision history, now resolved against
the primary source directly. This module previously used 4000.0 Hz
(cited to Grandin et al. 2016, GRL, "~4 kHz for Sentinel IW"), then
briefly 1480.0 Hz (citing a mis-identified paper, see above -- reverted
for that reason, without yet having a precisely-sourced replacement).
Yagüe-Martínez et al. (2016) Section III-C states directly: "Considering
the maximum ΔfDC of 5.2 kHz..." -- ΔfDC there is the Doppler centroid
variation across a burst's azimuth dwell (used for their eq. 4
coregistration-sensitivity analysis), essentially the same physical
quantity as the overlap-region Doppler difference this constant
represents (the overlap sits at the burst's trailing/leading edges,
where that variation is most pronounced). Set to 5200.0 Hz accordingly
-- now backed by a direct primary-source figure, not an order-of-
magnitude estimate. (ESA's own eoPortal/SentiWiki cites 5.5 kHz for the
same general quantity, consistent with this value.)

SENTINEL1_BURST_SYNC_REQUIREMENT_MS = 5.0: Sentinel-1's own mission
specification (SentiWiki, "S1 Mission" page): "a requirement for
achieving a synchronization of less than 5 ms between corresponding
bursts." Used by compute_burst_synchronization() below to directly
check whether two acquisitions' burst timing is good enough for
TOPS interferometry to work well at all -- independent of pixel-level
coregistration accuracy, which cannot compensate for genuine burst
desynchronization (Yagüe-Martínez et al. 2016, Section II-B: "A lack
of spectral overlap due to burst mis-synchronization leads to
decorrelation").
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from pygeofetch.insar.annotation import SwathTiming

logger = logging.getLogger("pygeofetch.insar.esd")

# Real Sentinel-1 IW Doppler centroid frequency difference between
# forward and backward looks in the burst overlap region. Directly
# cited to Yagüe-Martínez et al. (2016) Section III-C ("the maximum
# ΔfDC of 5.2 kHz") -- see the module docstring above for the full
# citation history and reasoning.
SENTINEL1_IW_DELTA_F_OVL_HZ = 5200.0

# Sentinel-1's own mission specification for burst timing accuracy
# between two acquisitions (SentiWiki, "S1 Mission" page): "a
# requirement for achieving a synchronization of less than 5 ms between
# corresponding bursts." Used by compute_burst_synchronization() below.
SENTINEL1_BURST_SYNC_REQUIREMENT_MS = 5.0


def compute_overlap_row_ranges(
    swath_timing: SwathTiming,
    azimuth_time_interval_s: float,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Compute the row ranges for each burst-overlap region, directly in
    row-index space (as opposed to overlap_time_windows() below, which
    works in absolute azimuth-time space). Not currently used by the
    live ESD path (estimate_esd_shift_per_burst_overlap uses
    overlap_time_windows instead) -- kept as a alternate, row-index-
    based implementation of the same calculation, useful if a caller
    already has row-indexed data and wants to avoid the datetime
    round-trip.

    Returns a list of ((bw_row_start, bw_row_end), (fw_row_start, fw_row_end))
    tuples, one per adjacent burst pair. bw = backward-looking (end of
    burst i), fw = forward-looking (start of burst i+1).
    """
    overlaps = []
    lines_per_burst = swath_timing.lines_per_burst
    bursts = swath_timing.bursts

    for i in range(len(bursts) - 1):
        # Backward-looking: last rows of burst i
        bw_row_end = (i + 1) * lines_per_burst - 1
        bw_row_start = i * lines_per_burst

        # Forward-looking: first rows of burst i+1
        fw_row_start = (i + 1) * lines_per_burst
        fw_row_end = (i + 2) * lines_per_burst - 1

        # Compute overlap extent from timing
        burst_i_end_time = (
            bursts[i].azimuth_time.timestamp()
            + (lines_per_burst - 1) * azimuth_time_interval_s
        )
        burst_next_start_time = bursts[i + 1].azimuth_time.timestamp()

        overlap_seconds = burst_i_end_time - burst_next_start_time
        if overlap_seconds <= 0:
            continue

        overlap_lines = int(round(overlap_seconds / azimuth_time_interval_s))
        if overlap_lines < 4:
            continue

        # Backward view: last `overlap_lines` rows of burst i
        bw_row_start = bw_row_end - overlap_lines + 1
        # Forward view: first `overlap_lines` rows of burst i+1
        fw_row_end = fw_row_start + overlap_lines - 1

        n_bw = bw_row_end - bw_row_start + 1
        n_fw = fw_row_end - fw_row_start + 1

        if n_bw != n_fw:
            logger.warning(
                "Overlap %d<->%d: backward-view row count (%d) does not "
                "match forward-view row count (%d) -- unusual burst "
                "timing; using the shorter of the two.",
                i,
                i + 1,
                n_bw,
                n_fw,
            )
            n_common = min(n_bw, n_fw)
            bw_row_start = bw_row_end - n_common + 1
            fw_row_end = fw_row_start + n_common - 1

        overlaps.append(((bw_row_start, bw_row_end), (fw_row_start, fw_row_end)))

    return overlaps


def overlap_time_windows(
    swath_timing: SwathTiming, azimuth_time_interval_s: float
) -> List[Tuple[int, "datetime", "datetime"]]:
    """
    Absolute azimuth-time window of every adjacent-burst overlap -- the
    ground area imaged by BOTH burst i (its trailing lines) and burst
    i+1 (its leading lines) in THIS scene.

    Returns:
        List of (overlap_index, window_start, window_end) as absolute
        azimuth datetimes.
    """
    bursts = swath_timing.bursts
    lines_per_burst = swath_timing.lines_per_burst
    out = []
    for i in range(len(bursts) - 1):
        start = bursts[i + 1].azimuth_time
        end = bursts[i].azimuth_time + timedelta(
            seconds=(lines_per_burst - 1) * azimuth_time_interval_s
        )
        out.append((i, start, end))
    return out


def compute_common_ground_overlaps(
    ref_timing: SwathTiming,
    azimuth_time_interval_s: float,
) -> List[Dict]:
    """
    Per-reference-overlap diagnostic of how many azimuth lines of real,
    double-covered ground each of the REFERENCE's own burst overlaps
    spans.

    BUG THIS FUNCTION USED TO HAVE, fixed here: an earlier version took
    a second (secondary-date) SwathTiming and intersected the
    reference's overlap window against a SEPARATELY computed secondary
    overlap window, using each date's own absolute azimuthTime. Since
    reference and secondary are different acquisition DATES (weeks
    apart), their absolute azimuth-time windows can never overlap --
    max(ref_start, sec_start) is always later than min(ref_end,
    sec_end) whenever the two dates differ at all, making that
    intersection empty for every overlap of every pair, unconditionally
    -- confirmed directly: a real run showed 0 common lines for every
    single pair, including same-burst-count ("same-family") pairs whose
    overlaps should trivially have real content in common. That
    uniform-across-the-board failure is the tell this was a bug, not
    the "cross-family burst timing" physical limitation the old message
    claimed (a genuine physical limitation would still show a real,
    nonzero value for matching-burst-count pairs).

    The correct fix: by the time ESD runs, the secondary has already
    been resampled onto the REFERENCE's own pixel grid by
    coregistration (see estimate_esd_shift_per_burst_overlap's own
    docstring) -- so there is no independent secondary time axis left
    to intersect against in the first place. The reference's own
    overlap window IS the common ground for both (now identically-
    indexed) arrays; this function reports its length directly.

    Returns:
        List of dicts, one per reference overlap, each with:
          ref_overlap_index, ref_window, common_lines (the full length,
          in azimuth lines, of that reference overlap window).
    """
    dt = azimuth_time_interval_s
    ref_wins = overlap_time_windows(ref_timing, dt)

    report = []
    for i, r_start, r_end in ref_wins:
        n_lines = max(0, int(round((r_end - r_start).total_seconds() / dt)) + 1)
        report.append(
            {
                "ref_overlap_index": i,
                "ref_window": (r_start, r_end),
                "common_lines": n_lines,
            }
        )
    return report


def _mean_burst_cycle_s(swath_timing: SwathTiming) -> float:
    """
    Real, measured mean time between consecutive burst starts -- the
    burst repeat period T_cycle -- computed directly from this date's
    own parsed per-burst azimuthTime values, not assumed from a nominal
    overlap fraction. Needed to reduce a raw acquisition-time
    difference to the sub-cycle residual that actually matters for
    burst synchronization (see compute_burst_synchronization).
    """
    times = [b.azimuth_time for b in swath_timing.bursts]
    if len(times) < 2:
        raise ValueError(
            "_mean_burst_cycle_s: need at least 2 real bursts to "
            "measure a burst cycle period, got "
            f"{len(times)}."
        )
    diffs = [(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)]
    return sum(diffs) / len(diffs)


def compute_burst_synchronization(
    ref_orbit,
    sec_orbit,
    ref_burst_info: SwathTiming,
    sec_burst_info: SwathTiming,
    ground_point: Tuple[float, float, float],
    ref_time_guess: datetime,
    sec_time_guess: datetime,
) -> Dict:
    """
    Real burst synchronization check: computes Δt_acq, the actual
    physical quantity Sentinel-1's own mission specification requires
    be under 5 ms for two acquisitions to be well-suited to TOPS
    interferometry at all -- independent of how precisely their pixels
    end up coregistered.

    Why this exists: coregistration accuracy (RMS in pixels, GCP
    coherence) tells you whether the SOFTWARE found the right
    alignment. It cannot tell you whether the two acquisitions'
    ANTENNA BEAM TIMING was itself compatible -- if the satellite
    imaged the same ground point with a substantially different squint
    angle on the two passes (i.e. at different points within each
    acquisition's own burst cycle), the resulting Doppler spectra don't
    overlap enough to correlate well, and no amount of pixel-level
    registration fixes that: it's a property of the two acquisitions
    themselves, not of how the data was processed. This function
    measures that property directly, rather than inferring it from a
    coherence number.

    Method matches Yagüe-Martínez, Prats-Iraola et al. (2016),
    "Interferometric Processing of Sentinel-1 TOPS Data," IEEE TGRS
    54(4), Section III-A: "The possible azimuth whole-burst offset can
    be retrieved by performing a geolocation of an arbitrary
    slant-range point ... using the master orbit to obtain the position
    on ground. Afterward, an inverse geolocation of this point using
    the slave orbit provides the slant-range coordinates for the slave
    point. With the azimuth burst length and the subswath timing
    information, the whole-burst offset can be easily obtained." Their
    own real worked example (Section V-B, a Salar de Uyuni pair):
    "the along-track position mismatching in the middle of the scene is
    0.12 ms ... indicating excellent burst synchronization."

    The raw geolocation round-trip (reference orbit forward, secondary
    orbit backward) gives a time difference that includes both a
    (large, many-burst-cycles) whole-burst offset AND the fine residual
    -- what actually determines decorrelation is only the residual,
    isolated from that whole-cycle offset.

    IMPORTANT NUMERICAL-PRECISION NOTE (found and fixed after real-data
    testing): reducing the RAW t_ref - t_sec difference by naively
    taking it modulo a single averaged cycle estimate is numerically
    unsafe for real dates that are weeks apart. Two acquisitions 60
    days apart span roughly (60*86400)/2.76 ~= 1.9 MILLION burst
    cycles; a cycle-period estimate averaged from just the 7-8 real
    burst timestamps in one product (microsecond-precision ISO-8601
    values) carries sub-microsecond uncertainty, but that tiny relative
    error gets multiplied by the ~1.9 million cycle count -- confirmed
    empirically able to produce over a SECOND of spurious residual
    error, which scattered real, well-correlated pairs (good
    coregistration RMS, good final coherence) across the entire
    (-cycle/2, +cycle/2] range with no relation to their actual
    synchronization. Avoided here by never forming that huge raw
    difference at all: each date's "local phase" is computed as the
    offset from T_ref/T_sec to THAT DATE'S OWN nearest real burst start
    time (an exact, per-date value, not an average), so only the
    difference of two already-small numbers ever needs any modulo
    reduction -- eliminating the large-N amplification entirely.

    Args:
        ref_orbit, sec_orbit: parse_orbit_file() output for each date.
        ref_burst_info, sec_burst_info: parse_burst_info() output for
                       each date -- supplies each date's own real,
                       precise burst start times.
        ground_point:  Real ECEF ground point to evaluate
                       synchronization at. Yagüe-Martínez et al.
                       recommend an arbitrary subswath midpoint;
                       per their own spatial-consistency analysis
                       (Section V-E-2) synchronization is stable to
                       under 1 cm across an entire slice, so any
                       reasonable real AOI point works well -- this
                       does not need to be the exact scene center.
        ref_time_guess, sec_time_guess: Initial time estimates for the
                       zero-Doppler solve (each date's own scene-center
                       acquisition time is a good choice).

    Returns:
        dict with:
          ref_zero_doppler_time, sec_zero_doppler_time: the real,
                       solved acquisition times for ground_point in
                       each orbit.
          burst_cycle_s: the real, measured burst cycle period
                       (averaged from both dates' own parsed timing) --
                       used only to bound the already-small residual
                       into (-cycle/2, +cycle/2], not to reduce a large
                       raw difference.
          sync_offset_ms: Δt_acq in milliseconds, centered in
                       (-cycle/2, +cycle/2]; positive means the
                       reference observes this point later within its
                       nearest real burst than the secondary does.
          within_esa_requirement: bool, whether |sync_offset_ms| is
                       under Sentinel-1's own 5 ms specification.

    Raises:
        RuntimeError if find_zero_doppler_time can't converge for
        either orbit at this ground point (surfaced directly rather
        than silently skipped, matching that function's own contract).
    """
    import bisect

    from pygeofetch.insar.geolocation import find_zero_doppler_time

    t_ref = find_zero_doppler_time(*ref_orbit, ground_point, ref_time_guess)
    t_sec = find_zero_doppler_time(*sec_orbit, ground_point, sec_time_guess)

    def _local_burst_phase_s(t: datetime, burst_info: SwathTiming) -> float:
        """Offset (seconds) from t to the start of the nearest real
        burst in burst_info whose start time is <= t -- an exact,
        per-date value bounded to roughly [0, cycle), never an average
        over another date or a huge time span."""
        times = [b.azimuth_time for b in burst_info.bursts]
        i = bisect.bisect_right(times, t) - 1
        i = max(0, min(i, len(times) - 1))
        return (t - times[i]).total_seconds()

    phase_ref = _local_burst_phase_s(t_ref, ref_burst_info)
    phase_sec = _local_burst_phase_s(t_sec, sec_burst_info)

    ref_cycle = _mean_burst_cycle_s(ref_burst_info)
    sec_cycle = _mean_burst_cycle_s(sec_burst_info)
    cycle = (ref_cycle + sec_cycle) / 2.0

    # Both phase_ref and phase_sec are already small (bounded by ~cycle)
    # -- this modulo only needs to handle the case where they sit on
    # opposite sides of a burst boundary, not a huge raw time gap, so
    # any imprecision in `cycle` here has a harmless, small effect.
    raw_diff = phase_ref - phase_sec
    sync_offset_s = ((raw_diff + cycle / 2.0) % cycle) - cycle / 2.0
    sync_offset_ms = sync_offset_s * 1000.0

    result = {
        "ref_zero_doppler_time": t_ref,
        "sec_zero_doppler_time": t_sec,
        "burst_cycle_s": cycle,
        "sync_offset_ms": sync_offset_ms,
        "within_esa_requirement": abs(sync_offset_ms)
        < SENTINEL1_BURST_SYNC_REQUIREMENT_MS,
    }

    logger.info(
        "Burst synchronization: Δt_acq=%.3f ms (burst cycle %.4f s) -- "
        "%s Sentinel-1's own <%.0f ms requirement.",
        sync_offset_ms,
        cycle,
        "within" if result["within_esa_requirement"] else "OUTSIDE",
        SENTINEL1_BURST_SYNC_REQUIREMENT_MS,
    )
    if not result["within_esa_requirement"]:
        logger.warning(
            "Burst synchronization Δt_acq=%.3f ms exceeds Sentinel-1's "
            "own <%.0f ms mission requirement for this pair -- expect "
            "real, physical Doppler-spectrum decorrelation independent "
            "of coregistration accuracy (Yagüe-Martínez et al. 2016, "
            "Sec. II-B). No amount of pixel-level registration fixes "
            "this; it reflects the two acquisitions' own burst timing.",
            sync_offset_ms,
            SENTINEL1_BURST_SYNC_REQUIREMENT_MS,
        )

    return result


def estimate_esd_shift_per_burst_overlap(
    ref_complex,
    sec_complex,
    swath_timing,
    azimuth_time_interval_s,
    row_offset: int = 0,
    delta_f_ovl_hz=SENTINEL1_IW_DELTA_F_OVL_HZ,
    coherence_threshold: float = 0.2,
    min_common_lines: int = 3,
):
    """
    Real per-burst-overlap ESD: estimate azimuth misregistration from
    the actual burst overlap regions.

    ref_complex, sec_complex must already be coregistered (at least
    coarsely -- real orbit-based coregistration is sufficient) onto the
    SAME pixel grid before this runs; `swath_timing` (the REFERENCE's
    own burst metadata) is then used to locate the overlap rows for
    BOTH arrays, since the secondary shares the reference's row
    indexing post-coregistration. There is deliberately no separate
    secondary-timing parameter: an earlier version accepted one and
    intersected the two dates' independent absolute burst timings,
    which -- being different acquisition DATES -- never actually
    overlap in absolute time; see compute_common_ground_overlaps's
    docstring for the full explanation of that bug.

    Args:
        ref_complex, sec_complex: Coregistered complex SLC arrays, same
                       shape -- full burst stack, or an already-cropped
                       extract (see row_offset).
        swath_timing:  Real burst metadata from
                       annotation.parse_burst_info(), for the
                       REFERENCE date.
        azimuth_time_interval_s: Real per-line time spacing.
        row_offset:    Real full-scene row that ref_complex/sec_complex's
                       row 0 corresponds to, same convention as
                       deburst.deburst_array().
        delta_f_ovl_hz: Real Doppler frequency separation in the
                       overlap region.
        coherence_threshold: Windowed-coherence gate for accepting an
                       overlap's pixels into the phase estimate.
        min_common_lines: Overlaps shorter than this (in azimuth lines)
                       are skipped as too small to give a reliable
                       estimate.

    Returns:
        (combined_shift_s, per_overlap_shifts) -- combined_shift_s is
        the median azimuth timing shift (seconds) across all valid
        overlaps (None if none were usable); per_overlap_shifts has one
        entry per adjacent burst pair.
    """
    import numpy as np
    from scipy.ndimage import uniform_filter

    dt = azimuth_time_interval_s
    L = swath_timing.lines_per_burst
    ref_wins = overlap_time_windows(swath_timing, dt)

    per_overlap_shifts = []
    skip_reasons = []

    for i, r_start, r_end in ref_wins:
        # The reference's own overlap window is the common ground for
        # BOTH arrays: by the time ESD runs, sec_complex has already
        # been resampled onto the reference's pixel grid by
        # coregistration, so it is indexed at exactly the same rows as
        # ref_complex -- there is no separate secondary time axis left
        # to intersect against. (Fixed bug: a previous version
        # intersected this window against a SEPARATELY computed
        # secondary-date overlap window using each date's own absolute
        # azimuthTime -- since reference and secondary are different
        # acquisition DATES weeks apart, that intersection was always
        # empty, for every overlap of every pair, regardless of whether
        # the two dates' burst structures actually matched. See
        # compute_common_ground_overlaps's docstring for the full
        # explanation.)
        n_lines = int(round((r_end - r_start).total_seconds() / dt)) + 1

        if n_lines < min_common_lines:
            per_overlap_shifts.append(None)
            skip_reasons.append("overlap_too_short")
            continue

        bw_rows, fw_rows = [], []
        for k in range(n_lines):
            tt = r_start + timedelta(seconds=k * dt)
            bw_rows.append(
                i * L
                + round((tt - swath_timing.bursts[i].azimuth_time).total_seconds() / dt)
                - row_offset
            )
            fw_rows.append(
                (i + 1) * L
                + round(
                    (tt - swath_timing.bursts[i + 1].azimuth_time).total_seconds() / dt
                )
                - row_offset
            )

        bw = np.array([r for r in bw_rows if 0 <= r < ref_complex.shape[0]])
        fw = np.array([r for r in fw_rows if 0 <= r < ref_complex.shape[0]])
        if len(bw) < min_common_lines or len(fw) < min_common_lines:
            per_overlap_shifts.append(None)
            skip_reasons.append("outside_array_bounds")
            continue

        ref_bw, sec_bw = ref_complex[bw], sec_complex[bw]
        ref_fw, sec_fw = ref_complex[fw], sec_complex[fw]
        igram_bw = ref_bw * np.conj(sec_bw)
        igram_fw = ref_fw * np.conj(sec_fw)

        # Windowed coherence gate (same as before)
        n_common = len(bw)
        coh_window = min(5, n_common) if n_common >= 3 else 1
        if coh_window >= 3:
            num = np.abs(
                uniform_filter(igram_bw.real, size=coh_window)
                + 1j * uniform_filter(igram_bw.imag, size=coh_window)
            )
            denom = np.sqrt(
                uniform_filter(np.abs(ref_bw) ** 2, size=coh_window)
                * uniform_filter(np.abs(sec_bw) ** 2, size=coh_window)
                + 1e-10
            )
            valid = np.isfinite(num / denom) & ((num / denom) >= coherence_threshold)
        else:
            valid = np.ones(igram_bw.shape, dtype=bool)

        if valid.mean() < 0.5:
            per_overlap_shifts.append(None)
            skip_reasons.append(f"low_coherence({100*valid.mean():.0f}%)")
            continue

        double_diff = igram_fw * np.conj(igram_bw)
        mean_phase = np.angle(np.mean(double_diff[valid]))
        per_overlap_shifts.append(float(mean_phase / (2 * np.pi * delta_f_ovl_hz)))
        skip_reasons.append("used")

    valid_shifts = [s for s in per_overlap_shifts if s is not None]
    if not valid_shifts:
        n_outside = skip_reasons.count("outside_array_bounds")
        n_short = skip_reasons.count("overlap_too_short")
        n_low = sum(1 for r in skip_reasons if r.startswith("low_coherence"))
        logger.warning(
            "ESD: no usable overlaps of %d (%d too short, %d outside array "
            "bounds, %d low coherence)",
            len(ref_wins),
            n_short,
            n_outside,
            n_low,
        )
        return None, per_overlap_shifts

    combined = float(np.median(valid_shifts))
    logger.info(
        "ESD common-ground: %d/%d usable, shift=%.6f ms (%.4f px)",
        len(valid_shifts),
        len(ref_wins),
        combined * 1000,
        combined / dt,
    )
    return combined, per_overlap_shifts
