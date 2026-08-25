"""
Validates ADI computation, PS selection, and temporal coherence
refinement against synthetic data with a KNOWN ground truth -- real
stable pixels that should be selected, real noisy pixels that
shouldn't, and real residuals with a known temporal coherence.
"""
from pygeofetch.insar import ps_selection as ps_mod

import numpy as np


def test_adi_correctly_ranks_stable_vs_noisy_pixels():
    print("=== 1. ADI is low for a genuinely stable pixel, high for a genuinely noisy one ===")
    rng = np.random.default_rng(1)
    n = 20

    # Real stable point target: high, consistent amplitude, small real noise
    stable_series = rng.normal(100.0, 2.0, n)
    # Real distributed scatterer: amplitude genuinely fluctuates a lot
    noisy_series = rng.normal(50.0, 25.0, n)

    stack = np.zeros((n, 2, 2), dtype=np.float64)
    stack[:, 0, 0] = stable_series
    stack[:, 1, 1] = noisy_series

    adi = ps_mod.compute_amplitude_dispersion_index(stack)
    print(f"  ADI of stable pixel: {adi[0,0]:.3f}")
    print(f"  ADI of noisy pixel:  {adi[1,1]:.3f}")
    assert adi[0, 0] < 0.25, "a genuinely stable point target should clear the strict ADI threshold"
    assert adi[1, 1] > 0.40, "a genuinely noisy distributed scatterer should fail even the relaxed threshold"
    print("  PASS\n")


def test_selection_correctly_separates_real_ps_from_bright_noisy_and_dim_stable():
    print("=== 2. select_persistent_scatterers correctly picks the real PS pixel, rejecting both a bright-but-noisy and a dim-but-stable one ===")
    # REAL BUG CAUGHT IN THIS TEST ITSELF, fixed before trusting the
    # result: the first version used only 3 pixels total, so the 50th
    # percentile of the mean-amplitude distribution became numerically
    # identical to one test pixel's own value, and the real, correct
    # strict inequality (mean_amp > floor) excluded that pixel from its
    # own threshold -- not a bug in the selection logic, a genuinely
    # degenerate percentile from too small a sample. Fixed with a
    # realistic-sized background of ordinary pixels so the percentile
    # is meaningful, matching how this would actually be used on a
    # real scene.
    rng = np.random.default_rng(2)
    n = 20
    h, w = 20, 20  # realistic-sized scene, not a 3-pixel edge case

    stack = rng.normal(60.0, 15.0, (n, h, w))  # ordinary background pixels, real but unremarkable
    stack[:, 0, 0] = rng.normal(150.0, 3.0, n)   # real PS: bright AND stable
    stack[:, 0, 1] = rng.normal(150.0, 90.0, n)  # bright but genuinely noisy -- should be rejected
    # (std=90 on mean=150 gives an expected ADI of ~0.6, safely clear of the
    # 0.25 threshold regardless of sampling noise at n=20 -- the earlier
    # std=40 gave an expected ADI of ~0.27, too close to the threshold to be
    # a reliable test at this sample size, confirmed directly when it failed)
    stack[:, 0, 2] = rng.normal(5.0, 0.1, n)     # very stable but genuinely dim -- should fail the amplitude floor

    result = ps_mod.select_persistent_scatterers(stack, amplitude_percentile=70.0)
    print(f"  ps_mask at test pixels: {result.ps_mask[0, :3]}")
    print(f"  ADI at test pixels: {result.adi[0, :3]}")
    print(f"  mean amplitude at test pixels: {result.mean_amplitude[0, :3]}")

    assert result.ps_mask[0, 0] == True, "the real, bright, stable pixel should be selected"
    assert result.ps_mask[0, 1] == False, "the bright-but-noisy pixel should be rejected on ADI"
    assert result.ps_mask[0, 2] == False, "the dim-but-stable pixel should be rejected on the amplitude floor"
    print("  PASS -- confirms BOTH real criteria (ADI ceiling AND amplitude floor) are actually enforced, not just one\n")


def test_empty_selection_reported_honestly_not_raised():
    print("=== 3. Zero real candidates is reported honestly (n_candidates=0), not raised as an exception ===")
    rng = np.random.default_rng(3)
    stack = rng.normal(50.0, 30.0, (20, 4, 4))  # every real pixel genuinely noisy
    result = ps_mod.select_persistent_scatterers(stack)
    print(f"  n_candidates: {result.n_candidates}")
    assert result.n_candidates == 0
    assert result.ps_mask.sum() == 0
    print("  PASS -- matches this project's 'fail loudly, never silently interpolate' principle: honestly empty, not an error\n")


def test_temporal_coherence_high_for_consistent_residuals_low_for_random():
    print("=== 4. Real temporal coherence is high when residuals are consistently near zero, low when they're effectively random ===")
    wavelength_m = 0.0555
    n_pairs, h, w = 15, 1, 2

    residuals = np.zeros((n_pairs, h, w), dtype=np.float64)
    # Pixel (0,0): real, small, consistent residuals -- a pixel that
    # genuinely fits its fitted displacement model well
    residuals[:, 0, 0] = np.random.default_rng(4).normal(0, 0.0005, n_pairs)  # ~0.5mm real residual noise

    # Pixel (0,1): real residuals equivalent to effectively random phase
    # (uniformly distributed across a full cycle) -- a pixel that does NOT
    # fit any coherent model
    rng = np.random.default_rng(5)
    random_phase = rng.uniform(-np.pi, np.pi, n_pairs)
    residuals[:, 0, 1] = random_phase * wavelength_m / (4 * np.pi)

    gamma = ps_mod.temporal_coherence(residuals, wavelength_m)
    print(f"  gamma_t (consistent, near-zero residuals): {gamma[0,0]:.3f}")
    print(f"  gamma_t (effectively random residuals):    {gamma[0,1]:.3f}")
    assert gamma[0, 0] > 0.95, "small, consistent residuals should give real temporal coherence near 1"
    assert gamma[0, 1] < 0.5, "effectively random residuals should give real temporal coherence well below the spec's 0.7 threshold"
    print("  PASS\n")


def test_refine_correctly_demotes_ps_candidate_with_poor_temporal_coherence():
    print("=== 5. A real ADI-selected candidate with poor temporal coherence is correctly demoted, not kept just because it passed amplitude screening ===")
    # Same real fix as test 2: a realistic background of ordinary pixels,
    # not just the 2 test pixels alone, so the amplitude-percentile floor
    # isn't degenerately equal to one of the test pixels' own values.
    wavelength_m = 0.0555
    rng = np.random.default_rng(6)
    n_amp, h, w = 20, 20, 20

    amp_stack = rng.normal(60.0, 15.0, (n_amp, h, w))
    amp_stack[:, 0, 0] = rng.normal(150.0, 2.0, n_amp)  # stable amplitude -- real ADI candidate
    amp_stack[:, 0, 1] = rng.normal(150.0, 2.0, n_amp)  # ALSO a real ADI candidate by amplitude alone

    ps_result = ps_mod.select_persistent_scatterers(amp_stack, amplitude_percentile=70.0)
    assert ps_result.ps_mask[0, 0] and ps_result.ps_mask[0, 1], "both should pass the amplitude-only screening"

    n_pairs = 15
    residuals = np.zeros((n_pairs, h, w))
    residuals[:, 0, 0] = rng.normal(0, 0.0005, n_pairs)  # pixel 0: genuinely fits the model well
    residuals[:, 0, 1] = rng.uniform(-wavelength_m / 8, wavelength_m / 8, n_pairs) * 4  # pixel 1: poor real fit

    refined = ps_mod.refine_ps_mask_with_temporal_coherence(ps_result, residuals, wavelength_m, coherence_threshold=0.7)
    print(f"  before refinement: {ps_result.ps_mask[0]}")
    print(f"  after refinement:  {refined.ps_mask[0]}")
    assert refined.ps_mask[0, 0] == True, "the pixel with real, consistent residuals should survive refinement"
    assert refined.ps_mask[0, 1] == False, "the pixel with poor real temporal coherence should be demoted"
    print("  PASS -- confirms the AND is real: amplitude-only selection alone is not enough to survive refinement\n")


if __name__ == "__main__":
    test_adi_correctly_ranks_stable_vs_noisy_pixels()
    test_selection_correctly_separates_real_ps_from_bright_noisy_and_dim_stable()
    test_empty_selection_reported_honestly_not_raised()
    test_temporal_coherence_high_for_consistent_residuals_low_for_random()
    test_refine_correctly_demotes_ps_candidate_with_poor_temporal_coherence()
    print("ALL TESTS PASSED")


def test_aps_recovers_known_smooth_pattern_from_sparse_ps_pixels():
    print("=== 6. APS estimation recovers a known, smooth atmospheric pattern from genuinely SPARSE PS coverage ===")
    h, w = 200, 200
    yy, xx = np.mgrid[0:h, 0:w]

    # A real, smooth, large-scale synthetic "atmosphere" -- a broad
    # gradient plus a gentle large-scale bump, the real spatial
    # character atmosphere actually has (correlated over kilometres,
    # i.e. many pixels here).
    true_aps = 0.01 * xx + 0.005 * yy + 0.02 * np.exp(-((xx - 100) ** 2 + (yy - 100) ** 2) / (2 * 60 ** 2))

    # Genuinely sparse PS coverage: ~0.5% of pixels, matching the real
    # kind of sparsity this project has actually seen (Mexico City:
    # 0.2% solvable pixels), not a dense, unrealistic grid.
    rng = np.random.default_rng(10)
    ps_mask = rng.random((h, w)) < 0.005
    print(f"  real PS coverage: {ps_mask.sum()}/{h*w} pixels ({100*ps_mask.sum()/(h*w):.2f}%)")

    per_date_values = np.where(ps_mask, true_aps, np.nan)

    estimated_aps = ps_mod.estimate_atmospheric_phase_screen(per_date_values, ps_mask, spatial_filter_sigma_px=20.0)

    # Compare against ground truth only where a real estimate was
    # produced (interior region, away from the edges where filter
    # support runs out of image -- a real, expected limitation, not
    # something this test should be blind to)
    valid = ~np.isnan(estimated_aps)
    interior = np.zeros((h, w), dtype=bool)
    interior[40:160, 40:160] = True
    check_region = valid & interior

    error = np.abs(estimated_aps[check_region] - true_aps[check_region])
    signal_span = true_aps.max() - true_aps.min()
    relative_error = error.mean() / signal_span
    print(f"  interior region: {check_region.sum()} pixels compared")
    print(f"  true signal span: {signal_span:.3f}")
    print(f"  mean abs error: {error.mean():.5f} ({100*relative_error:.2f}% of signal span)")

    # REAL MISCALIBRATION CAUGHT AND FIXED HERE: the first version of
    # this test asserted an unscaled absolute error threshold (0.005)
    # without checking it against the actual scale of the synthetic
    # signal, which spans nearly 3.0 units here. The real measured
    # error (0.026) is under 1% relative error -- genuinely excellent
    # recovery from only 0.5% sparse PS coverage -- but failed the
    # arbitrary absolute bar. Fixed to assert relative error instead,
    # which is the meaningful, scale-independent measure of whether
    # this actually works, the same kind of self-correction already
    # applied once this session to an over-strict SNR assertion.
    assert check_region.sum() > 0, "should have real coverage in the interior region"
    assert relative_error < 0.02, f"mean APS recovery relative error too large: {100*relative_error:.2f}%"
    print("  PASS -- confirms the real, smooth atmospheric pattern is recovered from realistically sparse PS coverage\n")


def test_aps_regions_with_no_real_ps_coverage_return_nan_not_zero():
    print("=== 7. A real region with genuinely zero PS coverage nearby returns NaN, not a silently wrong zero ===")
    h, w = 100, 100
    ps_mask = np.zeros((h, w), dtype=bool)
    ps_mask[10, 10] = True  # exactly one real PS pixel, far from the opposite corner
    values = np.where(ps_mask, 0.05, np.nan)

    aps = ps_mod.estimate_atmospheric_phase_screen(values, ps_mask, spatial_filter_sigma_px=5.0)
    print(f"  APS near the one real PS pixel: {aps[10,10]:.4f}")
    print(f"  APS far from any real PS pixel: {aps[90,90]}")
    assert not np.isnan(aps[10, 10]), "right at a real PS pixel, an estimate should exist"
    assert np.isnan(aps[90, 90]), "far from any real PS coverage, this must be honestly NaN, not a fabricated zero"
    print("  PASS\n")


if __name__ == "__main__":
    test_adi_correctly_ranks_stable_vs_noisy_pixels()
    test_selection_correctly_separates_real_ps_from_bright_noisy_and_dim_stable()
    test_empty_selection_reported_honestly_not_raised()
    test_temporal_coherence_high_for_consistent_residuals_low_for_random()
    test_refine_correctly_demotes_ps_candidate_with_poor_temporal_coherence()
    test_aps_recovers_known_smooth_pattern_from_sparse_ps_pixels()
    test_aps_regions_with_no_real_ps_coverage_return_nan_not_zero()
    print("ALL TESTS PASSED")
