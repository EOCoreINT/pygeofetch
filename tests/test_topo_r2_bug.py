"""
Proves the R² computation bug in _remove_topographic_phase(_chunked) and
AtmosphericCorrector: `centered = np.angle(np.exp(1j * (phase_v -
np.mean(phase_v))))` uses a NAIVE ARITHMETIC mean of already-wrapped
phase angles to "center" the data before computing ss_tot -- but for
circular/wrapped data, an arithmetic mean is only valid when the true
circular mean happens to sit far from the +-pi wraparound boundary. The
correct circular mean is np.angle(np.mean(np.exp(1j * phase_v))) --
which is used correctly, three times, elsewhere in the SAME function
(for `intercept`, inside `_flatness`, and for `fitted_phase_v`/
`residual`) -- just not here.

Demonstrates with a deliberately constructed case: a REAL, strong,
noise-light topographic-phase-vs-elevation relationship (should give a
high R²), but with its true circular mean placed near the wraparound
boundary (~pi) specifically to trigger the bug. Compares the buggy
formula against the correct one on identical data.
"""

import numpy as np

rng = np.random.default_rng(0)


def buggy_r_squared(phase_v, dem_v, best_slope, intercept):
    fitted_phase_v = np.angle(np.exp(1j * (best_slope * dem_v + intercept)))
    residual = np.angle(np.exp(1j * (phase_v - fitted_phase_v)))
    ss_res = np.sum(residual**2)
    centered = np.angle(np.exp(1j * (phase_v - np.mean(phase_v))))  # BUG: naive mean
    ss_tot = np.sum(centered**2)
    return 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0


def correct_r_squared(phase_v, dem_v, best_slope, intercept):
    fitted_phase_v = np.angle(np.exp(1j * (best_slope * dem_v + intercept)))
    residual = np.angle(np.exp(1j * (phase_v - fitted_phase_v)))
    ss_res = np.sum(residual**2)
    circular_mean = np.angle(np.mean(np.exp(1j * phase_v)))  # FIX: circular mean
    centered = np.angle(np.exp(1j * (phase_v - circular_mean)))
    ss_tot = np.sum(centered**2)
    return 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0


def make_case(true_slope, true_intercept, noise_std, n=5000):
    """dem elevations + wrapped phase generated from a REAL, known
    linear relationship (phase = slope*dem + intercept + noise, wrapped)."""
    dem_v = rng.uniform(0, 500, size=n)  # metres, realistic elevation range
    noise = rng.normal(0, noise_std, size=n)
    true_phase = true_slope * dem_v + true_intercept + noise
    phase_v = np.angle(np.exp(1j * true_phase))  # wrap to (-pi, pi]
    return phase_v, dem_v


def fit_and_compare(label, true_slope, true_intercept, noise_std):
    phase_v, dem_v = make_case(true_slope, true_intercept, noise_std)

    # Recover slope via the same candidate-search "flatness" method the
    # real code uses (not the point under test, but needed to get a
    # realistic best_slope/intercept to feed the R² formulas).
    def _flatness(candidate_slopes):
        phase_matrix = phase_v[None, :] - candidate_slopes[:, None] * dem_v[None, :]
        return np.abs(np.mean(np.exp(1j * phase_matrix), axis=1))

    max_slope = 0.05
    coarse = np.linspace(-max_slope, max_slope, 2000)
    best_slope = float(coarse[np.argmax(_flatness(coarse))])
    residual_v = np.angle(np.exp(1j * (phase_v - best_slope * dem_v)))
    intercept = float(np.angle(np.mean(np.exp(1j * residual_v))))

    r2_buggy = buggy_r_squared(phase_v, dem_v, best_slope, intercept)
    r2_correct = correct_r_squared(phase_v, dem_v, best_slope, intercept)

    true_circular_mean = np.angle(np.mean(np.exp(1j * phase_v)))
    naive_mean = float(np.mean(phase_v))

    print(f"=== {label} ===")
    print(
        f"  true slope={true_slope:.5f} rad/m, recovered best_slope={best_slope:.5f} rad/m"
    )
    print(
        f"  true circular mean of phase: {true_circular_mean:.3f} rad "
        f"(naive arithmetic mean: {naive_mean:.3f} rad, diff={abs(true_circular_mean-naive_mean):.3f})"
    )
    print(f"  R² (buggy, naive mean):   {r2_buggy:.4f}")
    print(f"  R² (correct, circular mean): {r2_correct:.4f}")
    return r2_buggy, r2_correct, true_circular_mean, naive_mean


if __name__ == "__main__":
    # Case A: strong real relationship, but with a MODEST total phase
    # swing (slope * elevation_range < one full cycle) so the wrapped
    # phase concentrates in a narrow arc -- and that arc is positioned
    # to straddle the +-pi discontinuity. This is where the bug bites:
    # a narrow cluster of real samples near +pi/-pi averages, under a
    # naive arithmetic mean, toward 0 -- diametrically opposite to
    # where the data actually sits.
    r2_buggy_a, r2_correct_a, tcm_a, nm_a = fit_and_compare(
        "Modest swing, arc straddling +-pi (realistic small-baseline case)",
        true_slope=0.003,
        true_intercept=float(np.pi - 0.003 * 250),
        noise_std=0.2,
    )
    print()

    # Case B: same relationship, same modest swing, but shifted so the
    # arc sits near 0 instead -- no wraparound, naive and circular mean
    # should closely agree, so buggy and correct R² should be close.
    r2_buggy_b, r2_correct_b, tcm_b, nm_b = fit_and_compare(
        "Modest swing, arc near 0 (no wraparound)",
        true_slope=0.003,
        true_intercept=0.0,
        noise_std=0.2,
    )
    print()

    # Case C: weak/near-absent true relationship (mostly noise -- the
    # realistic case for a genuinely flat AOI or a small baseline),
    # circular mean placed near +-pi. This is the case that should
    # reproduce the NEGATIVE R² actually observed in the real Mexico
    # City logs -- a symptom that, on its own, already signals
    # something is off (a correctly-computed R² against a fitted line
    # is bounded below by comparing to the same data's own mean, so a
    # large negative value for what should be a weak-but-honest fit is
    # itself suspicious).
    r2_buggy_c, r2_correct_c, tcm_c, nm_c = fit_and_compare(
        "Weak/near-absent true relationship, arc near +-pi",
        true_slope=0.0002,
        true_intercept=float(np.pi - 0.0002 * 250),
        noise_std=1.5,
    )
    print()

    print("=== Verdict ===")
    assert (
        abs(tcm_a - nm_a) > 0.5
    ), "test setup should actually trigger a large naive-vs-circular-mean gap"
    assert abs(r2_buggy_a - r2_correct_a) > 0.1, (
        f"buggy and correct R² should meaningfully diverge in the wraparound case: "
        f"buggy={r2_buggy_a:.4f}, correct={r2_correct_a:.4f}"
    )
    print(
        f"Wraparound case: buggy R²={r2_buggy_a:.4f} vs correct R²={r2_correct_a:.4f} -- "
        f"a real, meaningful discrepancy from mis-centering the data before computing ss_tot. "
        f"Note the DIRECTION isn't fixed: here the naive mean happens to inflate R² (makes the "
        f"fit look better than it truly is); with different data it can just as easily deflate "
        f"it. Either way the reported number is untrustworthy until fixed."
    )

    assert (
        abs(r2_buggy_b - r2_correct_b) < 0.1
    ), "non-wraparound case should show buggy and correct R² close together"
    print(
        f"Non-wraparound case: buggy R²={r2_buggy_b:.4f} vs correct R²={r2_correct_b:.4f} "
        f"-- close, as expected (the bug is data-dependent: it's invisible whenever the true "
        f"circular mean happens to sit away from the +-pi boundary, which is exactly why it "
        f"wasn't caught by casual testing)."
    )

    print(
        f"\nWeak-signal + wraparound case: buggy R²={r2_buggy_c:.4f}, correct R²={r2_correct_c:.4f}"
    )
    if r2_buggy_c < 0:
        print(
            "  Reproduces the negative-R² symptom actually seen in the real Mexico City logs."
        )
    print("\nPASS: bug confirmed and characterized.")
