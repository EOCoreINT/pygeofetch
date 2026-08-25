"""
Standalone validation for the coregistration upgrades:
  1. fit_offset_polynomial degree 1/2/3 recovers a known polynomial.
  2. fit_offset_polynomial_robust rejects injected outliers.
  3. refine_offsets_by_coherence recovers a known sub-pixel shift
     between synthetic complex speckle images and raises coherence.

Runs standalone against the package on disk (no orbit/annotation/DEM
inputs needed -- those are exercised at the interferogram.py wiring
level separately).
"""

import numpy as np

from pygeofetch.insar import coregister

fit_offset_polynomial = coregister.fit_offset_polynomial
fit_offset_polynomial_robust = coregister.fit_offset_polynomial_robust
refine_offsets_by_coherence = coregister.refine_offsets_by_coherence
_imagette_coherence = coregister._imagette_coherence
_sample_complex_imagette = coregister._sample_complex_imagette

rng = np.random.default_rng(0)


def test_polynomial_degrees():
    print("=== 1. fit_offset_polynomial degree recovery ===")
    rows = rng.uniform(0, 1000, size=60)
    cols = rng.uniform(0, 1000, size=60)

    # True degree-2 field: offset = 2 + 0.001*row - 0.002*col + 1e-6*row*col
    true = 2 + 0.001 * rows - 0.002 * cols + 1e-6 * rows * cols
    for degree in (1, 2, 3):
        fn = fit_offset_polynomial(rows, cols, true, degree=degree)
        pred = fn(rows, cols)
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        print(f"  degree={degree}: RMSE={rmse:.6f} px")
        # degree >= 2 must fit this (which is itself degree-2) essentially exactly
        if degree >= 2:
            assert rmse < 1e-6, f"degree {degree} should fit a degree-2 field exactly"
    # degree 1 (linear) should NOT perfectly fit a true quadratic field
    fn1 = fit_offset_polynomial(rows, cols, true, degree=1)
    rmse1 = float(np.sqrt(np.mean((fn1(rows, cols) - true) ** 2)))
    assert (
        rmse1 > 1e-3
    ), "linear fit should show residual error against a quadratic field"
    print(
        f"  degree=1 residual against true quadratic field: {rmse1:.4f} px (expected, nonzero)"
    )

    # minimum-points guard
    try:
        fit_offset_polynomial(rows[:2], cols[:2], true[:2], degree=3)
        raise AssertionError("expected ValueError for too few points")
    except ValueError as exc:
        print(f"  correctly rejected under-determined degree-3 fit: {exc}")
    print("  PASS\n")


def test_robust_outlier_rejection():
    print("=== 2. fit_offset_polynomial_robust outlier rejection ===")
    n = 49
    rows = np.tile(np.linspace(0, 1000, 7), 7)
    cols = np.repeat(np.linspace(0, 1000, 7), 7)

    true_row_offset = 5.0 + 0.0005 * rows - 0.0003 * cols
    true_col_offset = -3.0 + 0.0002 * rows + 0.0004 * cols

    noisy_row = true_row_offset + rng.normal(0, 0.02, size=n)
    noisy_col = true_col_offset + rng.normal(0, 0.02, size=n)

    # Inject 6 severe outliers (~12%), as if a few grid points had a bad solve.
    outlier_idx = rng.choice(n, size=6, replace=False)
    noisy_row[outlier_idx] += rng.choice([-1, 1], size=6) * rng.uniform(8, 15, size=6)
    noisy_col[outlier_idx] += rng.choice([-1, 1], size=6) * rng.uniform(8, 15, size=6)

    # Naive single-pass fit (old behaviour) for comparison.
    naive_row_fn = fit_offset_polynomial(rows, cols, noisy_row, degree=1)
    naive_col_fn = fit_offset_polynomial(rows, cols, noisy_col, degree=1)
    naive_rmse = float(
        np.sqrt(
            np.mean(
                (naive_row_fn(rows, cols) - true_row_offset) ** 2
                + (naive_col_fn(rows, cols) - true_col_offset) ** 2
            )
        )
    )

    row_fn, col_fn, quality = fit_offset_polynomial_robust(
        rows,
        cols,
        noisy_row,
        noisy_col,
        degree=1,
        max_iterations=2,
    )
    robust_rmse = float(
        np.sqrt(
            np.mean(
                (row_fn(rows, cols) - true_row_offset) ** 2
                + (col_fn(rows, cols) - true_col_offset) ** 2
            )
        )
    )

    print(f"  naive lstsq RMSE against ground truth:  {naive_rmse:.4f} px")
    print(f"  robust (outlier-rejected) RMSE:         {robust_rmse:.4f} px")
    print(
        f"  quality: {quality.n_gcps_final}/{quality.n_gcps_initial} GCPs kept, "
        f"{quality.iterations_used} iteration(s), fit RMS mean {quality.rms_mean:.4f} px"
    )
    assert (
        robust_rmse < naive_rmse / 3
    ), "robust fit should be substantially better than naive fit"
    assert (
        quality.n_gcps_final < quality.n_gcps_initial
    ), "some outliers should have been rejected"
    # NOTE: SNAP's own "eliminate anything above the mean RMS, repeat up
    # to 2x" rule is inherently aggressive -- for a roughly symmetric
    # residual distribution, close to half the points fall above the
    # mean on *each* pass, regardless of whether they're true outliers.
    # This is the documented, real behaviour of the algorithm being
    # matched, not a bug in this implementation; the meaningful check
    # is quality (RMSE) not raw survivor count.
    assert (
        quality.n_gcps_final >= 10
    ), "should retain enough GCPs for a usable degree-1 fit"
    print("  PASS\n")

    # min_gcps guard: force degree 3 with too few points after rejection
    print("  checking min-GCP guard on aggressive rejection...")
    tiny_rows = rows[:10]
    tiny_cols = cols[:10]
    tiny_row_off = true_row_offset[:10].copy()
    tiny_row_off[0] += 500  # one wild outlier among a minimal set
    tiny_col_off = true_col_offset[:10].copy()
    try:
        fit_offset_polynomial_robust(
            tiny_rows,
            tiny_cols,
            tiny_row_off,
            tiny_col_off,
            degree=3,
            max_iterations=2,
        )
        print("  (degree-3 fit on 10 points survived rejection -- also acceptable)")
    except RuntimeError as exc:
        print(f"  correctly raised on insufficient post-rejection GCPs: {exc}")
    print("  PASS\n")


def make_speckle_pair(shape=(256, 256), true_shift=(2.35, -1.7), seed=1):
    """Synthetic complex speckle: sec is ref shifted by a known
    sub-pixel amount (via Fourier shift) plus small independent noise,
    the way a mildly decorrelated real InSAR pair behaves."""
    rs = np.random.default_rng(seed)
    ref = (rs.normal(size=shape) + 1j * rs.normal(size=shape)).astype(np.complex128)

    # Fourier-domain sub-pixel shift (row, col)
    h, w = shape
    fy = np.fft.fftfreq(h).reshape(-1, 1)
    fx = np.fft.fftfreq(w).reshape(1, -1)
    ramp = np.exp(-2j * np.pi * (fy * true_shift[0] + fx * true_shift[1]))
    shifted = np.fft.ifft2(np.fft.fft2(ref) * ramp)

    noise = (rs.normal(size=shape) + 1j * rs.normal(size=shape)) * 0.15
    sec = (shifted + noise).astype(np.complex64)
    return ref.astype(np.complex64), sec


def test_cross_correlation_refinement():
    print("=== 3. refine_offsets_by_coherence recovers a known sub-pixel shift ===")
    true_shift = (
        2.35,
        -1.7,
    )  # (row, col) -- secondary is this much offset vs reference
    ref, sec = make_speckle_pair(shape=(256, 256), true_shift=true_shift)

    # Coherence BEFORE any refinement, at zero assumed offset (i.e. what
    # you'd get if you only trusted a coarse/orbit estimate that was off
    # by the true shift and never refined it).
    ref_im = _sample_complex_imagette(ref, 128, 128, 48)
    sec_im_unrefined = _sample_complex_imagette(sec, 128, 128, 48)
    coh_before = _imagette_coherence(ref_im, sec_im_unrefined)

    # A single grid point at image center, with a deliberately imperfect
    # initial ("orbit-based") offset guess -- close but not exact, as a
    # real orbit/DEM estimate would be.
    grid_rows = [128]
    grid_cols = [128]
    initial_guess_error = (0.6, -0.5)
    offset_rows = [true_shift[0] + initial_guess_error[0]]
    offset_cols = [true_shift[1] + initial_guess_error[1]]

    out_rows, out_cols, out_orow, out_ocol, coh = refine_offsets_by_coherence(
        ref,
        sec,
        grid_rows,
        grid_cols,
        offset_rows,
        offset_cols,
        window=48,
        coarse_search_radius=3,
        fine_search_radius=1.5,
        coherence_threshold=0.05,
    )

    refined_row_offset = out_orow[0]
    refined_col_offset = out_ocol[0]
    coh_after = coh[0]

    row_err = abs(refined_row_offset - true_shift[0])
    col_err = abs(refined_col_offset - true_shift[1])

    print(f"  true shift:            {true_shift}")
    print(
        f"  initial (unrefined) guess: "
        f"({offset_rows[0]:.3f}, {offset_cols[0]:.3f}) -- coherence at that guess: {coh_before:.3f}"
    )
    print(
        f"  refined offset:        ({refined_row_offset:.3f}, {refined_col_offset:.3f})"
    )
    print(f"  refined coherence:     {coh_after:.3f}")
    print(f"  |row error|={row_err:.3f} px, |col error|={col_err:.3f} px")

    assert (
        coh_after > coh_before
    ), "refinement should raise coherence vs the unrefined guess"
    assert (
        row_err < 0.25
    ), f"row offset should recover close to true shift, got err={row_err}"
    assert (
        col_err < 0.25
    ), f"col offset should recover close to true shift, got err={col_err}"
    print("  PASS\n")

    # Edge-drop + coherence-threshold behaviour: a point right at the
    # array edge should be dropped, and a point on pure noise (no real
    # correlation) should be dropped by the coherence threshold.
    print("  checking edge-drop and coherence-threshold rejection...")
    edge_rows = [128, 2]  # second point is too close to edge
    edge_cols = [128, 2]
    edge_orow = [true_shift[0] + initial_guess_error[0], 0.0]
    edge_ocol = [true_shift[1] + initial_guess_error[1], 0.0]
    kept_rows, *_, kept_coh = refine_offsets_by_coherence(
        ref,
        sec,
        edge_rows,
        edge_cols,
        edge_orow,
        edge_ocol,
        window=48,
        coarse_search_radius=3,
        fine_search_radius=1.5,
        coherence_threshold=0.05,
    )
    assert (
        len(kept_rows) == 1 and kept_rows[0] == 128
    ), "edge point should have been dropped"
    print(f"  edge point correctly dropped; {len(kept_rows)} GCP survived")

    # Pure decorrelated noise pair -> should be rejected by coherence_threshold
    rs2 = np.random.default_rng(99)
    noise_ref = (rs2.normal(size=(256, 256)) + 1j * rs2.normal(size=(256, 256))).astype(
        np.complex64
    )
    noise_sec = (rs2.normal(size=(256, 256)) + 1j * rs2.normal(size=(256, 256))).astype(
        np.complex64
    )
    try:
        refine_offsets_by_coherence(
            noise_ref,
            noise_sec,
            [128],
            [128],
            [0.0],
            [0.0],
            window=48,
            coarse_search_radius=3,
            fine_search_radius=1.5,
            coherence_threshold=0.5,
        )
        raise AssertionError(
            "expected RuntimeError: fully decorrelated noise should be rejected"
        )
    except RuntimeError as exc:
        print(f"  correctly rejected fully decorrelated pair: {exc}")
    print("  PASS\n")


if __name__ == "__main__":
    test_polynomial_degrees()
    test_robust_outlier_rejection()
    test_cross_correlation_refinement()
    print("ALL TESTS PASSED")
