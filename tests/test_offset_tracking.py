"""
Validates the NCC + sub-pixel refinement + SNR pieces against
synthetic data with a precisely known ground truth, not just checking
that the code runs.
"""

import numpy as np
import pytest

scipy_ndimage = pytest.importorskip("scipy.ndimage")
from scipy.ndimage import shift as ndi_shift

from pygeofetch.insar.offset_tracking import (
    compute_snr,
    normalized_cross_correlation,
    subpixel_peak_offset,
)


def _make_textured_image(size=128, seed=0):
    """A real, textured synthetic image (not uniform) -- random speckle-like
    pattern with actual structure to correlate against, smoothed slightly
    to be more realistic than pure white noise."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 1, (size, size))
    from scipy.ndimage import gaussian_filter

    img = gaussian_filter(img, sigma=1.5)
    return img


def test_ncc_matches_direct_definition():
    print(
        "=== 1. FFT-based NCC matches the direct, textbook sliding-window definition ==="
    )
    ref = _make_textured_image(16, seed=1)
    search = _make_textured_image(24, seed=2)

    fast = normalized_cross_correlation(ref, search)

    # Direct O(n^4) reference implementation, computed independently
    rh, rw = ref.shape
    sh, sw = search.shape
    out_h, out_w = sh - rh + 1, sw - rw + 1
    direct = np.zeros((out_h, out_w))
    ref_c = ref - ref.mean()
    ref_norm = np.sqrt(np.sum(ref_c**2))
    for y in range(out_h):
        for x in range(out_w):
            win = search[y : y + rh, x : x + rw]
            win_c = win - win.mean()
            win_norm = np.sqrt(np.sum(win_c**2))
            direct[y, x] = np.sum(ref_c * win_c) / (ref_norm * win_norm)

    max_diff = np.max(np.abs(fast - direct))
    print(f"  max difference between FFT and direct NCC: {max_diff:.2e}")
    assert (
        max_diff < 1e-8
    ), "FFT-accelerated NCC must match the direct definition to numerical precision"
    print("  PASS\n")


def test_recovers_known_integer_shift():
    print("=== 2. Recovers a known, exact INTEGER pixel shift ===")
    # REAL BUG CAUGHT AND FIXED HERE: the first version of this test built
    # the "shifted" search patch via manual index arithmetic on the base
    # image, and got the sign of the shift backwards -- confirmed directly
    # by the recovered offset coming back as almost exactly the NEGATIVE of
    # the true shift, not randomly wrong. This version instead uses
    # scipy.ndimage.shift, the same independently-defined, unambiguous
    # method test 3 already uses correctly, rather than fragile manual
    # index math.
    base = _make_textured_image(80, seed=3)
    ref_patch = base[30:50, 30:50]  # 20x20 template

    true_dy, true_dx = 5, -3
    shifted = ndi_shift(base, shift=(true_dy, true_dx), order=1, mode="reflect")
    search_patch = shifted[30 - 8 : 50 + 8, 30 - 8 : 50 + 8]

    ncc = normalized_cross_correlation(ref_patch, search_patch)
    peak_y, peak_x, peak_val = subpixel_peak_offset(ncc)

    center_y = (search_patch.shape[0] - ref_patch.shape[0]) / 2
    center_x = (search_patch.shape[1] - ref_patch.shape[1]) / 2
    recovered_dy = peak_y - center_y
    recovered_dx = peak_x - center_x

    print(f"  true shift: ({true_dy}, {true_dx})")
    print(f"  recovered shift: ({recovered_dy:.3f}, {recovered_dx:.3f})")
    print(f"  peak correlation: {peak_val:.4f}")
    assert abs(recovered_dy - true_dy) < 0.15
    assert abs(recovered_dx - true_dx) < 0.15
    assert (
        peak_val > 0.90
    ), "an exact integer shift of identical texture should correlate almost perfectly"
    print("  PASS\n")


def test_recovers_known_subpixel_shift():
    print(
        "=== 3. Recovers a known, precise SUB-PIXEL shift via parabolic interpolation ==="
    )
    base = _make_textured_image(100, seed=4)
    ref_patch = base[40:64, 40:64]  # 24x24 template

    true_dy, true_dx = 3.4, -2.7
    # Use scipy's own sub-pixel image shift (spline interpolation) to build
    # a real secondary image with a precisely known, non-integer shift --
    # a genuinely independent method from the NCC/parabolic-fit code being
    # tested, not circular.
    shifted = ndi_shift(base, shift=(true_dy, true_dx), order=3, mode="reflect")
    search_patch = shifted[40 - 10 : 64 + 10, 40 - 10 : 64 + 10]

    ncc = normalized_cross_correlation(ref_patch, search_patch)
    peak_y, peak_x, peak_val = subpixel_peak_offset(ncc)

    center_y = (search_patch.shape[0] - ref_patch.shape[0]) / 2
    center_x = (search_patch.shape[1] - ref_patch.shape[1]) / 2
    recovered_dy = peak_y - center_y
    recovered_dx = peak_x - center_x

    print(f"  true sub-pixel shift: ({true_dy}, {true_dx})")
    print(f"  recovered shift: ({recovered_dy:.3f}, {recovered_dx:.3f})")
    err_y, err_x = abs(recovered_dy - true_dy), abs(recovered_dx - true_dx)
    print(f"  error: ({err_y:.3f}, {err_x:.3f}) pixels")
    assert (
        err_y < 0.15 and err_x < 0.15
    ), f"sub-pixel recovery error too large: ({err_y:.3f}, {err_x:.3f}) -- expected <0.15px"
    print("  PASS -- confirms real sub-pixel accuracy, not just whole-pixel matching\n")


def test_uniform_patch_returns_zero_not_a_false_peak():
    print(
        "=== 4. A featureless (uniform) reference patch returns zero, not a spuriously confident match ==="
    )
    ref = np.full((16, 16), 5.0)  # perfectly flat, like calm water
    search = np.random.default_rng(5).normal(5, 0.01, (24, 24))
    ncc = normalized_cross_correlation(ref, search)
    print(f"  max NCC value on a uniform reference: {ncc.max():.6f}")
    assert np.allclose(
        ncc, 0.0
    ), "a uniform reference patch has no real texture to match -- must not report a confident peak"
    print("  PASS\n")


def test_snr_correctly_distinguishes_real_match_from_noise():
    print(
        "=== 5. SNR is high for a real, distinct match and low for pure noise (no real feature) ==="
    )
    base = _make_textured_image(80, seed=6)
    ref_patch = base[30:50, 30:50]
    search_patch = base[22:58, 22:58]  # contains a real, exact match at zero shift

    ncc_real = normalized_cross_correlation(ref_patch, search_patch)
    py, px = np.unravel_index(np.argmax(ncc_real), ncc_real.shape)
    snr_real = compute_snr(ncc_real, (py, px))

    # Pure noise search patch -- no real feature to match, so any "peak"
    # found is just noise, and should have a real, low SNR
    noise_search = np.random.default_rng(7).normal(0, 1, (36, 36))
    ncc_noise = normalized_cross_correlation(ref_patch, noise_search)
    py2, px2 = np.unravel_index(np.argmax(ncc_noise), ncc_noise.shape)
    snr_noise = compute_snr(ncc_noise, (py2, px2))

    print(f"  SNR for a real, distinct match: {snr_real:.2f}")
    print(f"  SNR for a pure-noise (no real feature) search: {snr_noise:.2f}")
    # The claim that actually matters, matching the real spec directly:
    # a real match clears the spec's own SNR>=3.0 reliability threshold,
    # and pure noise does not. An earlier version of this test also
    # asserted snr_real > snr_noise * 2 -- an arbitrary margin I invented,
    # not something the spec requires, and not always true in general
    # (noise SNR is itself a random variable and can occasionally land
    # closer to 3.0 by chance). Removed rather than loosened arbitrarily,
    # since the real, meaningful claim is the threshold behavior below.
    assert (
        snr_real >= 3.0
    ), "a real, exact match should clear the real SNR threshold from the spec"
    assert (
        snr_noise < 3.0
    ), "pure noise (no real feature) should NOT clear the real SNR threshold"
    print(
        "  PASS -- confirms SNR thresholding correctly separates real matches from false ones at the spec's own threshold\n"
    )


def test_offset_tracker_recovers_spatially_varying_field():
    print(
        "=== 6. OffsetTracker recovers a real, SPATIALLY-VARYING displacement field across a full image ==="
    )
    from pygeofetch.insar.offset_tracking import OffsetTracker

    size = 300
    base = _make_textured_image(size, seed=42)

    # Real, spatially-varying displacement: simulates a mining subsidence
    # bowl -- larger shift near the center, tapering outward. Built as a
    # real 2D field, not a uniform shift, specifically to prove the tracker
    # recovers LOCAL offsets correctly, not just a single global shift.
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2, size / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_shift = 6.0
    shift_field_y = max_shift * np.exp(-(r**2) / (2 * 60**2))
    shift_field_x = (
        0.4 * shift_field_y
    )  # a smaller, correlated range-direction component

    # Build the real secondary image via a genuine spatially-varying warp
    # (independent of the tracker/NCC code under test).
    #
    # REAL BUG CAUGHT AND FIXED HERE: the first version of this test used
    # `yy + shift_field_y` for the sample coordinates, which is backwards.
    # Verified directly with an isolated single-bright-pixel check before
    # fixing: scipy.ndimage.shift (already trusted, since tests 2/3 pass)
    # uses output[i] = input[i - shift] to move real content by +shift.
    # map_coordinates samples input AT the given coordinates, so matching
    # that same convention requires coords = position - shift, not
    # position + shift. Using +shift produced a real warp in the mirror-
    # opposite direction, which is exactly what an 11.7px max error with
    # every window falsely marked "reliable" was actually showing --
    # confident, precise correlation against genuinely wrong ground truth.
    from scipy.ndimage import map_coordinates

    warped_yy = yy - shift_field_y
    warped_xx = xx - shift_field_x
    secondary = map_coordinates(base, [warped_yy, warped_xx], order=3, mode="reflect")

    tracker = OffsetTracker(search_window_size=48, step_size=24, chip_size=24)
    result = tracker.track(base, secondary, snr_threshold=3.0)

    print(
        f"  grid shape: {result.azimuth_offset.shape}, reliable windows: {result.reliable.sum()}/{result.reliable.size}"
    )

    # Compare recovered offsets against the TRUE field, sampled at each
    # real window center -- only over reliable windows, matching how this
    # would actually be used (unreliable windows are honestly excluded,
    # not silently averaged in).
    errors_y, errors_x = [], []
    for iy, cy_px in enumerate(result.window_centers_y):
        for ix, cx_px in enumerate(result.window_centers_x):
            if not result.reliable[iy, ix]:
                continue
            true_y = shift_field_y[cy_px, cx_px]
            true_x = shift_field_x[cy_px, cx_px]
            errors_y.append(abs(result.azimuth_offset[iy, ix] - true_y))
            errors_x.append(abs(result.range_offset[iy, ix] - true_x))

    errors_y, errors_x = np.array(errors_y), np.array(errors_x)
    print(
        f"  mean abs error: azimuth={errors_y.mean():.3f}px, range={errors_x.mean():.3f}px"
    )
    print(
        f"  max abs error:  azimuth={errors_y.max():.3f}px, range={errors_x.max():.3f}px"
    )

    assert (
        result.reliable.sum() > 0.7 * result.reliable.size
    ), "most windows over real texture should be reliable"
    assert (
        errors_y.mean() < 0.5
    ), f"mean azimuth error too large: {errors_y.mean():.3f}px"
    assert errors_x.mean() < 0.5, f"mean range error too large: {errors_x.mean():.3f}px"
    print(
        "  PASS -- confirms the full grid recovers a real, spatially-varying field, not just a single global shift\n"
    )


if __name__ == "__main__":
    test_ncc_matches_direct_definition()
    test_recovers_known_integer_shift()
    test_recovers_known_subpixel_shift()
    test_uniform_patch_returns_zero_not_a_false_peak()
    test_snr_correctly_distinguishes_real_match_from_noise()
    test_offset_tracker_recovers_spatially_varying_field()
    print("ALL TESTS PASSED")
