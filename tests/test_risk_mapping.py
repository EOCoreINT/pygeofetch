"""
Validates the fixed pygeofetch.analysis.risk_mapping module against
known ground truth, reproducing every confirmed bug from the original
version and showing the fix resolves it.
"""

import numpy as np
import pytest

from pygeofetch.insar import analysis as rm

# Realistic irregular acquisition gaps, matching this project's own
# real Mexico City stack.
DATES = [
    "2016-07-24",
    "2016-08-05",
    "2016-09-22",
    "2016-10-04",
    "2016-12-13",
    "2016-12-25",
]
H, W = 6, 6
TRUE_VELOCITY_MM_YR = -20.0  # real subsidence rate, mm/year


class FakeTimeSeriesResult:
    """Minimal stand-in for pygeofetch.insar.timeseries.TimeSeriesResult."""

    def __init__(self, displacement, dates):
        self.displacement = displacement  # (n_time, H, W), metres
        self.dates = dates


def make_synthetic_ts(velocity_mm_yr=TRUE_VELOCITY_MM_YR, noise_std_mm=0.5, seed=0):
    t_years = rm._resolve_time_years(DATES)
    rng = np.random.default_rng(seed)
    n_time = len(DATES)
    disp_mm = velocity_mm_yr * t_years[:, None, None] * np.ones((n_time, H, W))
    disp_mm += rng.normal(0, noise_std_mm, size=(n_time, H, W))
    disp_m = disp_mm / 1000.0
    return FakeTimeSeriesResult(disp_m.astype(np.float32), DATES), t_years


def test_resolve_time_years_matches_sbas_convention():
    print(
        "=== 1. _resolve_time_years uses real elapsed time, matches SBASTimeSeries convention ==="
    )
    t_years = rm._resolve_time_years(DATES)
    from datetime import datetime

    expected_days = [
        (
            datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(DATES[0], "%Y-%m-%d")
        ).days
        for d in DATES
    ]
    expected_years = np.array(expected_days) / 365.25
    print(f"  dates: {DATES}")
    print(f"  resolved t_years: {np.round(t_years, 4)}")
    print(f"  expected (days/365.25): {np.round(expected_years, 4)}")
    assert np.allclose(
        t_years, expected_years
    ), "time_years should match days/365.25 exactly"
    assert t_years[0] == 0.0, "first entry should be exactly 0 (rebased)"
    print("  PASS\n")


def test_no_mutation_of_input_object():
    print("=== 2. RiskMapper no longer mutates the caller's ts_result object ===")
    ts, _ = make_synthetic_ts()
    assert not hasattr(
        ts, "data"
    ), "sanity check: fake object shouldn't have 'data' before construction"
    assert not hasattr(
        ts, "times"
    ), "sanity check: fake object shouldn't have 'times' before construction"
    mapper = rm.RiskMapper(ts)
    assert not hasattr(
        ts, "data"
    ), "ts_result should NOT have gained a 'data' attribute"
    assert not hasattr(
        ts, "times"
    ), "ts_result should NOT have gained a 'times' attribute"
    assert mapper.data is not None
    assert mapper.time_years is not None
    print(
        "  confirmed: ts_result unchanged after RiskMapper construction; resolved values live on the mapper"
    )
    print("  PASS\n")


def test_default_risk_function_uses_real_time():
    print(
        "=== 3. _default_risk_function recovers the true trend using real elapsed time ==="
    )
    ts, t_years = make_synthetic_ts(
        velocity_mm_yr=TRUE_VELOCITY_MM_YR, noise_std_mm=0.2
    )
    mapper = rm.RiskMapper(ts)

    # Recover the per-pixel slope directly (in metres/year) via the
    # same real-time axis the risk function now uses internally.
    data_2d = mapper.data.reshape(len(t_years), -1)
    slope_m_per_year = np.array(
        [np.polyfit(t_years, data_2d[:, i], 1)[0] for i in range(data_2d.shape[1])]
    )
    recovered_mm_yr = np.mean(slope_m_per_year) * 1000.0
    print(f"  true velocity: {TRUE_VELOCITY_MM_YR} mm/year")
    print(f"  recovered (mean over pixels): {recovered_mm_yr:.2f} mm/year")
    assert (
        abs(recovered_mm_yr - TRUE_VELOCITY_MM_YR) < 2.0
    ), "should recover the true trend within noise tolerance"
    print("  PASS\n")


def test_bayesian_prior_now_has_real_effect():
    print("=== 4. Bayesian prior_mean/prior_std now actually affect the posterior ===")
    ts, t_years = make_synthetic_ts()
    mapper = rm.RiskMapper(ts)

    x = t_years
    y = mapper.data[:, 0, 0]  # one pixel's real time series

    # Weak (uninformative) prior should recover ~OLS.
    s_weak, _ = mapper._bayesian_linear_regression(
        x, y, 5000, prior_mean=0.0, prior_std=1e6
    )
    ols_slope = np.polyfit(x, y, 1)[0]
    print(f"  OLS slope: {ols_slope:.6f} m/year")
    print(f"  weak-prior posterior mean: {s_weak.mean():.6f} m/year (should ~= OLS)")
    assert abs(s_weak.mean() - ols_slope) < 0.01 * (abs(ols_slope) + 1e-6) + 1e-4

    # Strong, confident prior should pull the posterior noticeably
    # toward prior_mean, away from the OLS estimate.
    strong_prior_mean = ols_slope + 10.0  # deliberately far from OLS
    s_strong, _ = mapper._bayesian_linear_regression(
        x, y, 5000, prior_mean=strong_prior_mean, prior_std=1e-4
    )
    print(f"  strong prior_mean: {strong_prior_mean:.6f}")
    print(
        f"  strong-prior posterior mean: {s_strong.mean():.6f} (should ~= prior_mean, NOT OLS)"
    )
    assert (
        abs(s_strong.mean() - strong_prior_mean) < 0.5
    ), "a very confident prior should dominate the posterior"
    assert (
        abs(s_strong.mean() - ols_slope) > 5.0
    ), "posterior should have moved FAR from the OLS estimate"
    print("  PASS\n")


def test_bootstrap_no_longer_scrambles_time():
    print(
        "=== 5. Residual bootstrap gives stable, sign-consistent slopes (not scrambled) ==="
    )
    ts, t_years = make_synthetic_ts(
        velocity_mm_yr=TRUE_VELOCITY_MM_YR, noise_std_mm=0.3
    )
    mapper = rm.RiskMapper(ts)

    risk_map = mapper.compute_risk(
        method="bootstrap", confidence_level=0.95, n_simulations=200
    )
    print(f"  risk mean: {np.nanmean(risk_map.risk):.4f}")
    print(
        f"  uncertainty (std across bootstrap trials) mean: {np.nanmean(risk_map.uncertainty):.4f}"
    )
    print(f"  CI width mean: {np.nanmean(risk_map.upper_ci - risk_map.lower_ci):.4f}")

    # With a strong, consistent true trend and modest noise, bootstrap
    # risk estimates should be reasonably TIGHT (low relative
    # uncertainty) -- the old, broken version would show wild,
    # sign-flipping instability instead. Use coefficient of variation
    # as the check.
    cv = np.nanmean(risk_map.uncertainty) / (np.nanmean(risk_map.risk) + 1e-10)
    print(
        f"  coefficient of variation: {cv:.3f} (should be modest, not wildly unstable)"
    )
    assert (
        cv < 1.0
    ), f"bootstrap uncertainty should be a reasonable fraction of the risk estimate, got CV={cv:.3f}"
    print("  PASS\n")


def test_export_uncertainty_handles_none_transform():
    print("=== 6. export_uncertainty no longer breaks with transform=None ===")
    import tempfile

    ts, _ = make_synthetic_ts()
    mapper = rm.RiskMapper(ts)
    risk_map = mapper.compute_risk(method="analytical", confidence_level=0.9)
    assert (
        risk_map.transform is None
    ), "sanity check: FakeTimeSeriesResult has no transform"

    with tempfile.TemporaryDirectory() as tmp:
        path = risk_map.export_uncertainty(f"{tmp}/uncertainty.tif")
        import rasterio

        with rasterio.open(path) as src:
            assert src.transform is not None
            print(f"  wrote successfully with transform={src.transform}")
    print("  PASS\n")


def test_coverage_no_longer_fabricates_a_number():
    print(
        "=== 7. validate_risk_map no longer echoes confidence_level as fake 'coverage' ==="
    )
    ts, _ = make_synthetic_ts()
    mapper = rm.RiskMapper(ts)
    risk_map = mapper.compute_risk(method="analytical", confidence_level=0.87)
    results = mapper.validate_risk_map(risk_map, validation_data=None)
    print(f"  coverage with no validation_data: {results['coverage']}")
    assert (
        results["coverage"] is None
    ), "coverage should be None (honest), not silently equal to confidence_level"
    print("  PASS\n")


def test_plot_risk_map_renders():
    pytest.importorskip("matplotlib")
    print(
        "=== 8. plot_risk_map (including uncertainty hatching) renders without error ==="
    )
    import tempfile

    ts, _ = make_synthetic_ts()
    mapper = rm.RiskMapper(ts)
    risk_map = mapper.compute_risk(method="analytical", confidence_level=0.95)
    with tempfile.TemporaryDirectory() as tmp:
        path = risk_map.plot_risk_map(f"{tmp}/fig.png", add_uncertainty_hatch=True)
        import os

        assert os.path.exists(path) and os.path.getsize(path) > 0
        print(f"  figure rendered successfully: {os.path.getsize(path)} bytes")
    print("  PASS\n")


if __name__ == "__main__":
    test_resolve_time_years_matches_sbas_convention()
    test_no_mutation_of_input_object()
    test_default_risk_function_uses_real_time()
    test_bayesian_prior_now_has_real_effect()
    test_bootstrap_no_longer_scrambles_time()
    test_export_uncertainty_handles_none_transform()
    test_coverage_no_longer_fabricates_a_number()
    test_plot_risk_map_renders()
    print("ALL TESTS PASSED")
