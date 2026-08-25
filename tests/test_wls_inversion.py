"""
Validates SBASTimeSeries.invert_weighted() against the existing,
untouched OLS invert() -- the two most important real properties:
(1) with uniform weights, WLS must reproduce OLS exactly, and
(2) a down-weighted bridge pair must have measurably reduced
influence on the solved displacement compared to full weight.
"""
import numpy as np

from pygeofetch.insar import timeseries as ts_mod
from pygeofetch.insar import validate as validate_mod

SBASTimeSeries = ts_mod.SBASTimeSeries
InterferogramPair = ts_mod.InterferogramPair
DataValidator = validate_mod.DataValidator


def _build_pairs(true_disp_m, dates, edges, wavelength_m, h=3, w=3, coherence=0.8):
    """Builds real, self-consistent synthetic InterferogramPairs from a known
    true displacement history, inverting the real phase formula used
    throughout this codebase.

    REAL BUG CAUGHT AND FIXED HERE, a repeat of one already found and
    documented earlier in this same project: a spatially UNIFORM phase
    field (np.full, same value at every pixel) becomes exactly zero
    everywhere once referenced to any pixel, since every pixel already
    equals the reference by construction -- that's correct SBAS
    physics (a single interferogram only measures displacement
    RELATIVE to a reference), not a bug in the code under test, but it
    makes a uniform field useless for testing anything. pixel (0,0)
    here is the fixed, zero-displacement reference; pixel (1,1) is a
    distinct location carrying the real synthetic signal, matching the
    same fix already applied once this session -- reused incorrectly
    at first when writing this new test, caught by the all-zero
    displacement this produced before being fixed.
    """
    pairs = []
    for d1, d2 in edges:
        disp_diff = true_disp_m[d2] - true_disp_m[d1]
        phase = np.zeros((h, w), dtype=np.float64)
        phase[1, 1] = (4 * np.pi / wavelength_m) * disp_diff  # the real, measured pixel
        # phase[0,0] stays 0.0 -- the fixed, stable reference pixel
        coh = np.full((h, w), coherence, dtype=np.float32)
        pairs.append(InterferogramPair(reference_date=d1, secondary_date=d2,
                                        unwrapped_phase=phase, coherence=coh,
                                        perpendicular_baseline_m=0.0))
    return pairs


def test_wls_reduces_to_ols_with_uniform_weights():
    print("=== 1. WLS with uniform weights (coherence=1, no bridges) exactly reproduces the existing OLS invert() ===")
    wavelength_m = 0.0555
    dates = ["2024-01-01", "2024-01-13", "2024-01-25", "2024-02-06"]
    true_disp = {"2024-01-01": 0.0, "2024-01-13": -0.003, "2024-01-25": -0.007, "2024-02-06": -0.009}
    edges = [("2024-01-01", "2024-01-13"), ("2024-01-13", "2024-01-25"),
             ("2024-01-25", "2024-02-06"), ("2024-01-01", "2024-01-25")]
    pairs = _build_pairs(true_disp, dates, edges, wavelength_m, coherence=1.0)

    sbas = SBASTimeSeries(wavelength_m=wavelength_m, reference_date=dates[0])
    result_ols = sbas.invert(pairs, coherence_threshold=0.0, correct_unwrap=False, correct_dem=False, reference_pixel=(0, 0))
    result_wls = sbas.invert_weighted(pairs, classification=None, coherence_threshold=0.0,
                                        bridge_penalty=1.0, correct_unwrap=False, correct_dem=False, reference_pixel=(0, 0))

    print(f"  OLS displacement (date 2, real pixel): {result_ols.displacement[2, 1, 1]:.8f}")
    print(f"  WLS displacement (date 2, real pixel): {result_wls.displacement[2, 1, 1]:.8f}")
    print(f"  OLS displacement (date 2, reference pixel, should be 0): {result_ols.displacement[2, 0, 0]:.8f}")

    assert np.allclose(result_ols.displacement, result_wls.displacement, atol=1e-9, equal_nan=True), \
        "WLS with W=I must exactly reproduce OLS -- this is the core correctness guarantee"
    assert np.allclose(result_ols.velocity, result_wls.velocity, atol=1e-9, equal_nan=True)
    print("  PASS -- confirms (B^T W B)v = B^T W dphi reduces exactly to (B^T B)v = B^T dphi when W=I\n")


def test_bridge_penalty_reduces_influence_of_bad_pair():
    print("=== 2. A down-weighted bridge pair with a wrong observation has measurably LESS influence than at full weight ===")
    wavelength_m = 0.0555
    dates = ["2024-01-01", "2024-01-13", "2024-01-25"]
    true_disp = {"2024-01-01": 0.0, "2024-01-13": -0.003, "2024-01-25": -0.007}
    edges = [("2024-01-01", "2024-01-13"), ("2024-01-13", "2024-01-25"), ("2024-01-01", "2024-01-25")]
    pairs = _build_pairs(true_disp, dates, edges, wavelength_m, coherence=0.8)

    # Corrupt the redundant pair (01-01 -> 01-25) with a real, wrong observation
    # -- error injected only at the real, measured pixel (1,1); pixel (0,0)
    # stays the fixed, zero-displacement reference for both pairs.
    corrupted = list(pairs)
    bad_phase = pairs[2].unwrapped_phase.copy()
    bad_phase[1, 1] += (4 * np.pi / wavelength_m) * 0.05  # +5cm error injected
    corrupted[2] = InterferogramPair(
        reference_date="2024-01-01", secondary_date="2024-01-25",
        unwrapped_phase=bad_phase, coherence=np.full((3, 3), 0.15, dtype=np.float32),
        perpendicular_baseline_m=0.0,
    )

    class FakeClassification:
        bridge_pairs = [corrupted[2]]

    sbas = SBASTimeSeries(wavelength_m=wavelength_m, reference_date=dates[0])

    result_full_weight = sbas.invert_weighted(corrupted, classification=None,
                                                coherence_threshold=0.0, bridge_penalty=1.0,
                                                correct_unwrap=False, correct_dem=False,
                                                reference_pixel=(0, 0))
    result_penalized = sbas.invert_weighted(corrupted, classification=FakeClassification(),
                                              coherence_threshold=0.0, bridge_penalty=0.05,
                                              correct_unwrap=False, correct_dem=False,
                                              reference_pixel=(0, 0))

    true_final = true_disp["2024-01-25"]
    err_full = abs(result_full_weight.displacement[2, 1, 1] - true_final)
    err_penalized = abs(result_penalized.displacement[2, 1, 1] - true_final)

    print(f"  true final displacement: {true_final:.4f} m")
    print(f"  error with bad pair at FULL weight:      {err_full:.4f} m")
    print(f"  error with bad pair down-weighted 0.05x: {err_penalized:.4f} m")

    assert err_penalized < err_full, \
        "down-weighting the corrupted bridge pair must measurably reduce its distortion of the solution"
    print("  PASS -- confirms bridge_penalty genuinely reduces a bad pair's influence, not just labels it\n")


def test_invert_weighted_integrates_with_real_classify_pairs():
    print("=== 3. End-to-end: DataValidator.classify_pairs() output plugs directly into invert_weighted() ===")
    wavelength_m = 0.0555
    dates = ["2024-01-01", "2024-01-13", "2024-01-25", "2024-02-06", "2024-02-18", "2024-03-01"]
    A, B, C, D, E, F = dates
    true_disp = {A: 0.0, B: -0.002, C: -0.004, D: -0.006, E: -0.008, F: -0.010}
    edges = [(A, B), (B, C), (D, E), (E, F), (C, D)]  # C-D is the sole bridge
    pairs = _build_pairs(true_disp, dates, edges, wavelength_m, coherence=0.8)
    # make the bridge pair genuinely low-coherence, matching a real scenario
    pairs[-1] = InterferogramPair(
        reference_date=C, secondary_date=D,
        unwrapped_phase=pairs[-1].unwrapped_phase,
        coherence=np.full((3, 3), 0.12, dtype=np.float32),
        perpendicular_baseline_m=0.0,
    )

    classification = DataValidator.classify_pairs(pairs, dates, coherence_threshold=0.3)
    print(f"  {classification.summary()}")
    assert len(classification.bridge_pairs) == 1

    sbas = SBASTimeSeries(wavelength_m=wavelength_m, reference_date=A)
    result = sbas.invert_weighted(pairs, classification=classification,
                                    coherence_threshold=0.0, correct_unwrap=False, correct_dem=False,
                                    reference_pixel=(0, 0))
    recovered_f = result.displacement[dates.index(F), 1, 1]
    print(f"  recovered F displacement: {recovered_f:.4f} m (true: {true_disp[F]:.4f} m)")
    assert abs(recovered_f - true_disp[F]) < 0.001
    print("  PASS -- classify_pairs() output flows directly into invert_weighted(), full path works end to end\n")


if __name__ == "__main__":
    test_wls_reduces_to_ols_with_uniform_weights()
    test_bridge_penalty_reduces_influence_of_bad_pair()
    test_invert_weighted_integrates_with_real_classify_pairs()
    print("ALL TESTS PASSED")
