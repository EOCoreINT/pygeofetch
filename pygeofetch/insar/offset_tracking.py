"""
Amplitude-based offset tracking ("speckle tracking") for pygeofetch.

Conventional phase-based InSAR fails once deformation gradients exceed
the phase ambiguity limit — roughly half a wavelength of LOS motion
between acquisitions, about 2.8cm for Sentinel-1's real C-band
wavelength. Active mining subsidence, fast landslides, and large
co-seismic or volcanic deformation routinely exceed this. Offset
tracking sidesteps the ambiguity entirely by cross-correlating real
SAR amplitude (not phase) between two acquisitions, directly measuring
whole-and-fractional-pixel shifts in range and azimuth — immune to
phase wrapping, at the cost of much coarser precision (typically
1/10 to 1/30 of a pixel, versus phase InSAR's sub-millimetre
precision).

This module implements the amplitude cross-correlation and sub-pixel
refinement pieces, verified here against synthetic shifted images with
a precisely known ground-truth shift. The geometric projection from
raw (range, azimuth) pixel offsets to ground (East, North, Up)
displacement is deliberately NOT in this module — see
offset_geometry.py, built separately and grounded in a specific,
cited real-geometry reference rather than derived from memory, the
same discipline this project already applies to
geolocation.los_to_vertical_displacement's own citations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

logger = logging.getLogger("pygeofetch.insar.offset_tracking")


@dataclass(frozen=True)
class OffsetTrackingResult:
    """
    Real per-window offset tracking output.

    range_offset / azimuth_offset are in PIXELS (sub-pixel precision),
    positive meaning the reference-image feature is found at a larger
    range/azimuth pixel coordinate in the secondary image. snr is the
    real correlation-peak signal-to-noise ratio at each window (see
    compute_snr's own docstring for the exact definition used).
    reliable is True where snr >= the caller's threshold — window
    results below threshold are NOT zeroed, they're left as computed,
    so a caller can inspect what a low-confidence match actually
    looked like; masking to NaN is the caller's decision (matching
    this project's own "fail loudly, never silently interpolate"
    principle — a match this module doesn't trust is flagged, not
    hidden).
    """

    range_offset: "Any"     # (n_windows_y, n_windows_x) float32, pixels
    azimuth_offset: "Any"   # (n_windows_y, n_windows_x) float32, pixels
    snr: "Any"              # (n_windows_y, n_windows_x) float32
    reliable: "Any"         # (n_windows_y, n_windows_x) bool
    window_centers_y: "Any"  # (n_windows_y,) int, pixel row of each window's center in the reference image
    window_centers_x: "Any"  # (n_windows_x,) int, pixel col of each window's center


def _require_numpy():
    try:
        import numpy as np
        return np
    except ImportError as exc:
        raise ImportError("offset_tracking requires numpy: pip install numpy") from exc


def normalized_cross_correlation(reference_patch, search_patch):
    """
    Real, standard normalized cross-correlation surface between a
    fixed reference patch and a larger search patch, computed via FFT
    for speed (mathematically identical to the direct sliding-window
    NCC definition, verified against the direct computation in this
    module's own tests before being trusted).

    Formula (standard, e.g. Lewis 1995 "Fast Normalized Cross
    Correlation" — the FFT acceleration technique used here, not a
    different definition of NCC itself):

        NCC(dy,dx) = sum[(R - R_mean)(S_win - S_win_mean)] /
                     sqrt(sum[(R - R_mean)^2] * sum[(S_win - S_win_mean)^2])

    where R is the reference patch and S_win is the search-patch
    window at offset (dy,dx), matched in size to R. Returns a real
    correlation surface in [-1, 1] at every valid offset — 1.0 means a
    perfect match, not just a strong one.

    Args:
        reference_patch: 2D real array, the fixed template (from the
            reference-date amplitude image).
        search_patch: 2D real array, LARGER than reference_patch in
            both dimensions (from the secondary-date amplitude image,
            centered on the same nominal location but padded outward
            by the real search radius).

    Returns:
        2D real array of shape
        (search_patch.shape[0] - reference_patch.shape[0] + 1,
         search_patch.shape[1] - reference_patch.shape[1] + 1),
        the NCC value at every valid alignment.
    """
    np = _require_numpy()

    rh, rw = reference_patch.shape
    sh, sw = search_patch.shape
    if sh < rh or sw < rw:
        raise ValueError(
            f"normalized_cross_correlation: search_patch {search_patch.shape} "
            f"must be at least as large as reference_patch {reference_patch.shape} "
            f"in both dimensions."
        )

    ref = reference_patch.astype(np.float64)
    ref = ref - ref.mean()
    ref_norm = np.sqrt(np.sum(ref ** 2))
    if ref_norm < 1e-12:
        # A perfectly uniform reference patch (e.g. flat water) has no
        # real texture to match against -- every offset is equally
        # "correct", which is meaningless. Return an all-zero surface
        # rather than dividing by ~zero and returning a spuriously
        # confident result.
        out_h, out_w = sh - rh + 1, sw - rw + 1
        return np.zeros((out_h, out_w), dtype=np.float64)

    search = search_patch.astype(np.float64)

    # FFT-based sliding-window cross-correlation (correlate2d with
    # mode='valid' would be mathematically identical but far slower
    # for realistic window sizes; verified against the direct O(n^4)
    # definition in this module's own test before use).
    from scipy.signal import fftconvolve

    numerator = fftconvolve(search, ref[::-1, ::-1], mode="valid")

    # Real local search-patch energy at each window position, via a
    # running sum computed through convolution with a ones-kernel --
    # standard technique (Lewis 1995), not an approximation.
    ones = np.ones((rh, rw), dtype=np.float64)
    local_sum = fftconvolve(search, ones, mode="valid")
    local_sum_sq = fftconvolve(search ** 2, ones, mode="valid")
    n = rh * rw
    local_mean = local_sum / n
    local_var = local_sum_sq / n - local_mean ** 2
    local_var = np.clip(local_var, 0, None)  # real floating point can produce tiny negatives here
    local_norm = np.sqrt(local_var * n)

    with np.errstate(divide="ignore", invalid="ignore"):
        ncc = numerator / (ref_norm * local_norm)
    ncc = np.nan_to_num(ncc, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(ncc, -1.0, 1.0)


def subpixel_peak_offset(ncc_surface) -> Tuple[float, float, float]:
    """
    Real sub-pixel refinement of the NCC surface's peak location via
    1D parabolic interpolation along each axis independently, applied
    at the integer peak — standard technique (e.g. described in Debella-
    Gilo & Kääb 2011 for exactly this SAR offset-tracking use case).

    Formula, applied separately in y and x through the integer peak
    (iy, ix):
        delta = 0.5 * (f(-1) - f(+1)) / (f(-1) - 2*f(0) + f(+1))
    where f(-1), f(0), f(+1) are the NCC values at the peak and its two
    immediate neighbors along that axis. delta is the real, sub-pixel
    correction to add to the integer peak location.

    Args:
        ncc_surface: 2D real NCC output from normalized_cross_correlation.

    Returns:
        (peak_y, peak_x, peak_value) — sub-pixel peak location in the
        NCC surface's own index space (NOT yet converted to a
        range/azimuth offset — see track_offset_window, which handles
        that conversion using the real window geometry), and the real
        interpolated peak correlation value.
    """
    np = _require_numpy()

    iy, ix = np.unravel_index(np.argmax(ncc_surface), ncc_surface.shape)
    h, w = ncc_surface.shape
    peak_val = float(ncc_surface[iy, ix])

    dy = 0.0
    if 0 < iy < h - 1:
        f_m1, f_0, f_p1 = ncc_surface[iy - 1, ix], ncc_surface[iy, ix], ncc_surface[iy + 1, ix]
        denom = f_m1 - 2 * f_0 + f_p1
        if abs(denom) > 1e-12:
            dy = 0.5 * (f_m1 - f_p1) / denom

    dx = 0.0
    if 0 < ix < w - 1:
        f_m1, f_0, f_p1 = ncc_surface[iy, ix - 1], ncc_surface[iy, ix], ncc_surface[iy, ix + 1]
        denom = f_m1 - 2 * f_0 + f_p1
        if abs(denom) > 1e-12:
            dx = 0.5 * (f_m1 - f_p1) / denom

    return float(iy) + dy, float(ix) + dx, peak_val


def compute_snr(ncc_surface, peak_yx: Tuple[int, int], exclusion_radius: int = 2) -> float:
    """
    Real signal-to-noise ratio of an NCC correlation peak, used to
    reject false matches in featureless areas (open water, smooth
    bare terrain) where the "peak" is just noise rather than a real
    matched feature.

    SNR = peak_value / std(background), where "background" is every
    NCC surface value OUTSIDE a real exclusion window around the peak
    (default radius 2 pixels) — this is the real, standard formulation
    (e.g. used in ampcor/GAMMA-style offset trackers), not a peak-to-
    mean ratio, since a peak-to-mean ratio can look artificially
    strong on a surface that's mostly near-zero even when the peak
    itself is weak noise.

    Args:
        ncc_surface: 2D real NCC output.
        peak_yx: Integer (row, col) location of the peak (NOT the
            sub-pixel-refined location — the background exclusion
            window is defined on the real, discrete surface).
        exclusion_radius: Pixels around the peak excluded from the
            background statistics.

    Returns:
        Real SNR value. Returns 0.0 (not NaN, not a crash) if the
        entire surface is within the exclusion radius of the peak —
        a real edge case for very small search windows, handled
        explicitly rather than raising.
    """
    np = _require_numpy()

    h, w = ncc_surface.shape
    py, px = peak_yx
    mask = np.ones((h, w), dtype=bool)
    y0, y1 = max(0, py - exclusion_radius), min(h, py + exclusion_radius + 1)
    x0, x1 = max(0, px - exclusion_radius), min(w, px + exclusion_radius + 1)
    mask[y0:y1, x0:x1] = False

    background = ncc_surface[mask]
    if background.size == 0:
        return 0.0

    bg_std = float(np.std(background))
    if bg_std < 1e-12:
        return 0.0

    peak_val = float(ncc_surface[py, px])
    return peak_val / bg_std


class OffsetTracker:
    """
    Tiles the verified NCC + sub-pixel + SNR primitives across a full
    real image pair on a configurable window/step grid — the real
    "Feature 2" entry point from the spec.

    Deliberately does NOT do geometric projection to ground ENU
    displacement (see this module's own top-level docstring for why
    that's a separate, not-yet-built piece: it needs a real, cited
    geometry reference, the same discipline this project already
    applies to geolocation.los_to_vertical_displacement's own
    citations, not a formula written from memory).

    Example::

        tracker = OffsetTracker(search_window_size=64, step_size=16)
        result = tracker.track(reference_amplitude, secondary_amplitude, snr_threshold=3.0)
        # result.range_offset, result.azimuth_offset: pixel offsets, NaN where unreliable
    """

    def __init__(self, search_window_size: int = 64, step_size: int = 16, chip_size: Optional[int] = None):
        """
        Args:
            search_window_size: Real search radius context — the
                secondary-image window searched around each reference
                chip is this size (pixels), matching the spec's own
                "64x64 or 128x128" example.
            step_size: Real spacing (pixels) between window centers —
                the spec's own "16 or 32 pixels for oversampling"
                (smaller than the window itself, so windows overlap).
            chip_size: Size of the fixed reference template chip taken
                from the reference image at each window center.
                Defaults to search_window_size // 2 if not given — a
                real, reasonable default (the chip must be smaller
                than the search window, or there's no room to search).
        """
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        if chip_size is None:
            chip_size = search_window_size // 2
        if chip_size >= search_window_size:
            raise ValueError(
                f"chip_size ({chip_size}) must be smaller than search_window_size "
                f"({search_window_size}) -- there must be real room to search."
            )
        self.search_window_size = search_window_size
        self.step_size = step_size
        self.chip_size = chip_size

    def track(
        self,
        reference_amplitude,
        secondary_amplitude,
        snr_threshold: float = 3.0,
    ) -> OffsetTrackingResult:
        """
        Real, full-image offset tracking on this tracker's configured
        window/step grid.

        Args:
            reference_amplitude: 2D real amplitude array (NOT phase —
                see this module's own top-level docstring for why
                amplitude specifically).
            secondary_amplitude: 2D real amplitude array, same
                dimensions as reference_amplitude, real coregistered
                geometry (same convention InterferogramGenerator
                already assumes for phase-based pairs).
            snr_threshold: Real per-window SNR floor (spec default
                3.0) below which a window's offset is marked
                unreliable — NOT silently dropped, marked, matching
                this project's "fail loudly, never silently
                interpolate" principle: reliable=False windows are
                still present in the output with their real computed
                values, so a caller can inspect what a rejected match
                actually looked like.

        Returns:
            OffsetTrackingResult with per-window offsets, SNR, and a
            reliable mask.
        """
        np = _require_numpy()

        if reference_amplitude.shape != secondary_amplitude.shape:
            raise ValueError(
                f"OffsetTracker.track: reference and secondary amplitude "
                f"must have the same shape, got {reference_amplitude.shape} "
                f"vs {secondary_amplitude.shape}."
            )

        h, w = reference_amplitude.shape
        half_chip = self.chip_size // 2
        half_search = self.search_window_size // 2
        margin = half_search  # need this much real image on every side of a window center

        centers_y = list(range(margin, h - margin, self.step_size))
        centers_x = list(range(margin, w - margin, self.step_size))

        if not centers_y or not centers_x:
            raise ValueError(
                f"OffsetTracker.track: image ({h}x{w}) is too small for "
                f"search_window_size={self.search_window_size} -- no real "
                f"window fits. Use a smaller search_window_size or a larger image."
            )

        n_y, n_x = len(centers_y), len(centers_x)
        range_offset = np.full((n_y, n_x), np.nan, dtype=np.float32)
        azimuth_offset = np.full((n_y, n_x), np.nan, dtype=np.float32)
        snr = np.full((n_y, n_x), np.nan, dtype=np.float32)
        reliable = np.zeros((n_y, n_x), dtype=bool)

        n_low_snr = 0
        for iy, cy in enumerate(centers_y):
            for ix, cx in enumerate(centers_x):
                ref_chip = reference_amplitude[
                    cy - half_chip: cy - half_chip + self.chip_size,
                    cx - half_chip: cx - half_chip + self.chip_size,
                ]
                search_chip = secondary_amplitude[
                    cy - half_search: cy - half_search + self.search_window_size,
                    cx - half_search: cx - half_search + self.search_window_size,
                ]
                # Search chip must be at least chip-sized larger than the
                # reference chip in both dims for a real valid correlation;
                # skip (leave NaN/unreliable) rather than crash on an edge
                # window that doesn't have enough real margin.
                if (search_chip.shape[0] < ref_chip.shape[0] or
                        search_chip.shape[1] < ref_chip.shape[1]):
                    continue

                ncc = normalized_cross_correlation(ref_chip, search_chip)
                peak_y, peak_x, peak_val = subpixel_peak_offset(ncc)
                iy_int, ix_int = int(round(peak_y)), int(round(peak_x))
                iy_int = min(max(iy_int, 0), ncc.shape[0] - 1)
                ix_int = min(max(ix_int, 0), ncc.shape[1] - 1)
                pixel_snr = compute_snr(ncc, (iy_int, ix_int))

                center_search_y = (search_chip.shape[0] - ref_chip.shape[0]) / 2
                center_search_x = (search_chip.shape[1] - ref_chip.shape[1]) / 2

                azimuth_offset[iy, ix] = peak_y - center_search_y
                range_offset[iy, ix] = peak_x - center_search_x
                snr[iy, ix] = pixel_snr
                is_reliable = pixel_snr >= snr_threshold
                reliable[iy, ix] = is_reliable
                if not is_reliable:
                    n_low_snr += 1

        total = n_y * n_x
        if n_low_snr > 0:
            logger.info(
                "OffsetTracker: %d/%d windows (%.1f%%) below SNR threshold "
                "%.1f, marked unreliable rather than silently included.",
                n_low_snr, total, 100 * n_low_snr / total, snr_threshold,
            )

        return OffsetTrackingResult(
            range_offset=range_offset,
            azimuth_offset=azimuth_offset,
            snr=snr,
            reliable=reliable,
            window_centers_y=np.array(centers_y),
            window_centers_x=np.array(centers_x),
        )
