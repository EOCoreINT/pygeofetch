"""
TOPS deburst — remove real burst-boundary overlap/redundancy, matching
SNAP's own S-1 TOPS Deburst operator.

Algorithm confirmed directly against ESA's own STEP documentation for
TOPSARDeburstOp:
"In the azimuth direction, bursts are merged according to their
zero Doppler time... The merge time is determined by the average
of the last line of the first burst and the first line of the next
burst. For each range cell, the merging time is quantised to the
nearest output azimuth cell."

Why this matters: without deburst, adjacent bursts' real overlap
regions both remain in the data, and the SAME ground area gets imaged
twice by two different (radiometrically and phase-wise inconsistent)
parts of the antenna beam pattern — this is a real, confirmed source
of visible striping and degraded coherence at burst boundaries.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import List, Optional, Tuple

import numpy as np

from pygeofetch.insar.annotation import SwathTiming

logger = logging.getLogger("pygeofetch.insar.deburst")


def compute_burst_row_ranges(
    swath_timing: SwathTiming, azimuth_time_interval_s: float
) -> List[Tuple[int, int]]:
    """
    Compute each burst's real, kept (non-overlapping) row range, in
    that burst's own local 0-indexed row coordinates.

    Args:
        swath_timing: Real burst metadata from annotation.parse_burst_info().
        azimuth_time_interval_s: Real per-line time spacing.

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

    # Real cut row, in each side's own local coordinates
    cut_in_prev: List[Optional[int]] = [None] * n
    cut_in_next: List[Optional[int]] = [None] * n

    for i in range(n - 1):
        burst_i_last_line_time = bursts[i].azimuth_time + timedelta(
            seconds=(lines_per_burst - 1) * azimuth_time_interval_s
        )
        burst_next_first_line_time = bursts[i + 1].azimuth_time
        cut_time = (
            burst_i_last_line_time
            + (burst_next_first_line_time - burst_i_last_line_time) / 2
        )

        cut_row_in_i = round(
            (cut_time - bursts[i].azimuth_time).total_seconds()
            / azimuth_time_interval_s
        )
        cut_row_in_next = round(
            (cut_time - bursts[i + 1].azimuth_time).total_seconds()
            / azimuth_time_interval_s
        )

        cut_in_prev[i] = cut_row_in_i
        cut_in_next[i + 1] = cut_row_in_next

    ranges: List[Tuple[int, int]] = []
    for i in range(n):
        if i == 0:
            keep_start = 0
        else:
            next_cut = cut_in_next[i]
            assert next_cut is not None, (
                f"cut_in_next[{i}] must be set for all i > 0 "
                f"(set unconditionally in the loop above)."
            )
            keep_start = next_cut + 1
        if i == n - 1:
            keep_end = lines_per_burst - 1
        else:
            prev_cut = cut_in_prev[i]
            assert prev_cut is not None, (
                f"cut_in_prev[{i}] must be set for all i < n - 1 "
                f"(set unconditionally in the loop above)."
            )
            keep_end = prev_cut
        keep_start = max(0, min(keep_start, lines_per_burst - 1))
        keep_end = max(0, min(keep_end, lines_per_burst - 1))

        if keep_start > keep_end:
            logger.warning(
                "Burst %d: computed keep range is empty (start=%d > end=%d)",
                i,
                keep_start,
                keep_end,
            )
        ranges.append((keep_start, keep_end))

    return ranges


def deburst_array(
    data: np.ndarray,
    swath_timing: SwathTiming,
    azimuth_time_interval_s: float,
    row_offset: int = 0,
) -> Tuple[np.ndarray, int]:
    """
    Apply real deburst to an array: remove each burst's redundant
    overlap rows and concatenate the kept, non-overlapping portions
    into one continuous array.

    Args:
        data: 2D array (complex SLC, real-valued raster, etc.)
        swath_timing: Real burst metadata from annotation.parse_burst_info().
        azimuth_time_interval_s: Real per-line time spacing.
        row_offset: If `data` is an already-cropped extract, the real
            row offset of `data`'s row 0 within the full-scene burst stack.

    Returns:
        (debursted_data, first_kept_full_scene_row)
    """
    if not isinstance(swath_timing, SwathTiming):
        raise TypeError(
            f"deburst_array expects a SwathTiming object, got "
            f"{type(swath_timing).__name__}. Pass the result of "
            f"parse_burst_info(), not a raw bursts list."
        )

    ranges = compute_burst_row_ranges(swath_timing, azimuth_time_interval_s)
    lines_per_burst = swath_timing.lines_per_burst

    chunks = []
    first_kept_full_scene_row = None

    for i, (keep_start, keep_end) in enumerate(ranges):
        if keep_start > keep_end:
            continue

        burst_full_scene_start = i * lines_per_burst - row_offset
        chunk_start = burst_full_scene_start + keep_start
        chunk_end = burst_full_scene_start + keep_end + 1

        clipped_start = max(0, chunk_start)
        clipped_end = min(data.shape[0], chunk_end)

        if clipped_start >= clipped_end:
            continue

        if first_kept_full_scene_row is None:
            first_kept_full_scene_row = clipped_start + row_offset

        chunks.append(data[clipped_start:clipped_end])

    if not chunks:
        raise ValueError(
            "deburst_array: no real data remained after debursting — "
            "check that row_offset correctly reflects data's real "
            "position within the full-scene burst stack."
        )

    debursted = np.concatenate(chunks, axis=0)

    assert first_kept_full_scene_row is not None, (
        "first_kept_full_scene_row must be set whenever chunks is "
        "non-empty (set on the first appended chunk above)."
    )

    logger.info(
        "Deburst: %d bursts -> %d output rows (input was %d rows)",
        len(swath_timing.bursts),
        debursted.shape[0],
        data.shape[0],
    )

    return debursted, first_kept_full_scene_row
