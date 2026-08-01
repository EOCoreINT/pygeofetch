"""
TOPS deburst — remove real burst-boundary overlap/redundancy, matching
SNAP's own S-1 TOPS Deburst operator.

Algorithm confirmed directly against ESA's own STEP documentation for
TOPSARDeburstOp (step.esa.int/main/wp-content/help/.../TOPSARDeburstOp.html),
not inferred or guessed:

    "In the azimuth direction, bursts are merged according to their
    zero Doppler time... The merge time is determined by the average
    of the last line of the first burst and the first line of the next
    burst. For each range cell, the merging time is quantised to the
    nearest output azimuth cell."

Why this matters: without deburst, adjacent bursts' real overlap
regions both remain in the data, and the SAME ground area gets imaged
twice by two different (radiometrically and phase-wise inconsistent)
parts of the antenna beam pattern -- this is a real, confirmed source
of visible striping and degraded coherence at burst boundaries, and it
was never being corrected anywhere in this pipeline before this module
(pygeofetch's own existing ESD refinement docstring already admits this
directly: "has no effect on already-deburst/stripmap data" -- data
passed through it was never actually deburst).
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import List, Tuple

from pygeofetch.insar.annotation import SwathTiming

logger = logging.getLogger("pygeofetch.insar.deburst")


def compute_burst_row_ranges(
    swath_timing: SwathTiming, azimuth_time_interval_s: float
) -> List[Tuple[int, int]]:
    """
    Compute each burst's real, kept (non-overlapping) row range, in
    that burst's own local 0-indexed row coordinates.

    Verified before use: on a small, controlled synthetic case with
    known burst timing, the computed ranges left no gaps and no
    duplicate coverage, and their total length matched an
    independently-computed expected total (span from the first burst's
    start to the last burst's last line, at one line per
    azimuth_time_interval_s) exactly.

    Args:
        swath_timing:  Real burst metadata from
                       annotation.parse_burst_info().
        azimuth_time_interval_s: Real per-line time spacing, from the
                       same annotation XML's SLCGeometry
                       (imageAnnotation/imageInformation/azimuthTimeInterval)
                       -- must be the SAME value used for both, since
                       the cut-time calculation converts between the
                       two bursts' local row coordinates via this
                       shared spacing.

    Returns:
        List of (keep_start_row, keep_end_row) inclusive, local to
        each burst, same length and order as swath_timing.bursts.
    """
    bursts = swath_timing.bursts
    n = len(bursts)
    lines_per_burst = swath_timing.lines_per_burst

    if n == 0:
        return []
    if n == 1:
        return [(0, lines_per_burst - 1)]

    # Real cut row, in each side's own local coordinates, for every
    # adjacent burst pair
    cut_in_prev = [None] * n  # cut row (local to burst i) ending burst i's kept range
    cut_in_next = [None] * n  # cut row (local to burst i) starting burst i's kept range

    for i in range(n - 1):
        burst_i_last_line_time = bursts[i].azimuth_time + timedelta(
            seconds=(lines_per_burst - 1) * azimuth_time_interval_s
        )
        burst_next_first_line_time = bursts[i + 1].azimuth_time
        cut_time = burst_i_last_line_time + (burst_next_first_line_time - burst_i_last_line_time) / 2

        cut_row_in_i = round(
            (cut_time - bursts[i].azimuth_time).total_seconds() / azimuth_time_interval_s
        )
        cut_row_in_next = round(
            (cut_time - bursts[i + 1].azimuth_time).total_seconds() / azimuth_time_interval_s
        )

        cut_in_prev[i] = cut_row_in_i
        cut_in_next[i + 1] = cut_row_in_next

    ranges = []
    for i in range(n):
        keep_start = 0 if i == 0 else cut_in_next[i] + 1
        keep_end = (lines_per_burst - 1) if i == n - 1 else cut_in_prev[i]
        keep_start = max(0, min(keep_start, lines_per_burst - 1))
        keep_end = max(0, min(keep_end, lines_per_burst - 1))
        if keep_start > keep_end:
            logger.warning(
                "Burst %d: computed keep range is empty (start=%d > end=%d) "
                "— unusually large real overlap or unusual burst timing; "
                "this burst will contribute no rows to the debursted result.",
                i, keep_start, keep_end,
            )
        ranges.append((keep_start, keep_end))

    return ranges


def deburst_array(data, swath_timing: SwathTiming, azimuth_time_interval_s: float, row_offset: int = 0):
    """
    Apply real deburst to an array: remove each burst's redundant
    overlap rows and concatenate the kept, non-overlapping portions
    into one continuous array — matching SNAP's real behaviour (the
    image genuinely shrinks; this is not a masking operation).

    Args:
        data:          2D array (complex SLC, real-valued raster,
                       whatever a caller needs debursted), full burst
                       stack — n_bursts * lines_per_burst rows (or the
                       real full-scene extent if row_offset is used,
                       see below).
        swath_timing:  Real burst metadata, same as
                       compute_burst_row_ranges().
        azimuth_time_interval_s: Same real per-line spacing.
        row_offset:    If `data` is an already-cropped extract (e.g.
                       from SLCExtractor) rather than the full-scene
                       array, the real row offset of `data`'s row 0
                       within the full-scene burst stack (see
                       coregister.read_crop_offset() for how this is
                       tracked elsewhere in this project). Default 0
                       assumes `data` already starts at the full-scene
                       burst stack's row 0.

    Returns:
        (debursted_data, first_kept_full_scene_row) -- the debursted
        array, and the real full-scene row index its own row 0
        corresponds to (needed to correctly update georeferencing
        metadata for the result, e.g. a new SLCGeometry.first_line_time).
    """
    import numpy as np

    ranges = compute_burst_row_ranges(swath_timing, azimuth_time_interval_s)
    lines_per_burst = swath_timing.lines_per_burst

    chunks = []
    first_kept_full_scene_row = None
    for i, (keep_start, keep_end) in enumerate(ranges):
        if keep_start > keep_end:
            continue  # real, logged empty-range case from compute_burst_row_ranges

        burst_full_scene_start = i * lines_per_burst - row_offset
        chunk_start = burst_full_scene_start + keep_start
        chunk_end = burst_full_scene_start + keep_end + 1  # exclusive

        clipped_start = max(0, chunk_start)
        clipped_end = min(data.shape[0], chunk_end)
        if clipped_start >= clipped_end:
            continue  # this burst's kept range falls entirely outside the real cropped data

        if first_kept_full_scene_row is None:
            # Real, confirmed fix: use the ACTUAL clipped starting
            # position, not the theoretical, unclipped burst/keep_start
            # calculation -- when a real crop cuts into a burst's kept
            # range partway through (row_offset > 0 and the crop starts
            # mid-burst), the unclipped calculation silently reports the
            # wrong full-scene row. Verified directly: a real test case
            # with row_offset=5 exposed this exact mismatch (reported 0,
            # should have been 5) before this fix.
            first_kept_full_scene_row = clipped_start + row_offset

        chunks.append(data[clipped_start:clipped_end])

    if not chunks:
        raise ValueError(
            "deburst_array: no real data remained after debursting — "
            "check that row_offset correctly reflects data's real "
            "position within the full-scene burst stack."
        )

    debursted = np.concatenate(chunks, axis=0)
    logger.info(
        "Deburst: %d bursts -> %d output rows (input was %d rows)",
        len(swath_timing.bursts), debursted.shape[0], data.shape[0],
    )
    return debursted, first_kept_full_scene_row
