"""
Validates select_reliable_reference_pixel() and despike_velocity() --
the two new library functions replacing the notebook's manual SVD
reimplementation and networkx-based multi-component stitching with
correct fixes at the root cause (a reference pixel chosen to actually
be reliable across the network) plus one genuinely-missing real
capability (cycle-slip cleanup), both via the tested library instead
of ad-hoc notebook code.
"""
import numpy as np

from pygeofetch.insar import timeseries
from pygeofetch.insar import unwrap as unwrap_mod

select_reliable_reference_pixel = timeseries.select_reliable_reference_pixel
despike_velocity = timeseries.despike_velocity
bridge_unwrap_regions = unwrap_mod.bridge_unwrap_regions


# ── real bug regression: region-size mismatch with bridge_unwrap_regions ──

def test_reference_pixel_matches_bridge_unwrap_regions_criterion():
    print("=== 0. Real bug regression: conncomp>0 is not enough, must match "
          "bridge_unwrap_regions' min_region_size criterion ===")
    h, w = 30, 30
    min_region_size = 100

    # Pair A: a SMALL labeled region (label 3, only 20 px -- below
    # min_region_size) sitting at (7, 6), plus a real, large valid
    # region (label 1, 200 px) elsewhere. If reliability were checked
    # via conncomp > 0 alone (the old, buggy criterion), (7, 6) would
    # look reliable here even though bridge_unwrap_regions would reject
    # it outright as too small a region.
    conncomp_a = np.zeros((h, w), dtype=np.int32)
    conncomp_a[0:20, 0:10] = 1     # 200 px, real valid region
    conncomp_a[7, 6] = 3           # single pixel, "small region" label 3 -- but
    # give it a few more pixels so it's a real (if tiny) connected component,
    # not literally one isolated pixel:
    conncomp_a[6:9, 5:8] = 3       # 9 px total for label 3 -- still << 100

    # Pair B: same large valid region (label 1) reliable; the small
    # region from pair A doesn't even appear here (simulates a
    # different pair's real unwrapping outcome).
    conncomp_b = np.zeros((h, w), dtype=np.int32)
    conncomp_b[0:20, 0:10] = 1

    conncomp_masks = {("d1", "d2"): conncomp_a, ("d1", "d3"): conncomp_b}

    pixel, report = select_reliable_reference_pixel(
        conncomp_masks, min_region_size=min_region_size,
    )
    print(f"  chosen pixel: {pixel}, reliable_fraction={report['reliable_fraction']}")

    # The old (buggy) criterion would have found (7,6) or a nearby
    # label-3 pixel reliable in pair A (conncomp>0 there) -- confirm
    # the FIXED function does not choose anything inside that small
    # region.
    assert conncomp_a[pixel] != 3, (
        f"chosen pixel {pixel} falls in the small (9px < {min_region_size}) "
        f"region -- the exact real bug this fix addresses"
    )
    assert report["reliable_fraction"] == 1.0
    print("  correctly avoided the small (9px) region")

    # Close the loop: the chosen pixel must actually work with the
    # REAL downstream bridge_unwrap_regions() call, not raise.
    for pair_name, conncomp in [("d1->d2", conncomp_a), ("d1->d3", conncomp_b)]:
        unwrapped = np.random.default_rng(0).normal(size=(h, w)).astype(np.float32)
        try:
            bridge_unwrap_regions(
                unwrapped, conncomp, bridge_radius=50,
                min_region_size=min_region_size, reference_pixel=pixel,
            )
        except ValueError as exc:
            raise AssertionError(
                f"bridge_unwrap_regions rejected the chosen reference pixel "
                f"for {pair_name}: {exc} -- select_reliable_reference_pixel's "
                f"criterion still doesn't match bridge_unwrap_regions'"
            )
    print("  confirmed: bridge_unwrap_regions accepts the chosen pixel for every pair")
    print("  PASS\n")


# ── select_reliable_reference_pixel ────────────────────────────────────────

def test_finds_fully_reliable_pixel():
    print("=== 1. A pixel reliable in every pair should be found exactly ===")
    h, w = 20, 20
    rng = np.random.default_rng(0)
    pairs = [("d1", "d2"), ("d1", "d3"), ("d2", "d3"), ("d1", "d4")]
    conncomp = {}
    for p in pairs:
        mask = (rng.random((h, w)) > 0.4).astype(np.int32)  # mostly random reliability
        conncomp[p] = mask
    # Force pixel (5, 7) reliable in every single pair, by construction.
    for p in pairs:
        conncomp[p][5, 7] = 1

    pixel, report = select_reliable_reference_pixel(conncomp, min_region_size=1)
    print(f"  chosen pixel: {pixel}, reliable_fraction={report['reliable_fraction']}")
    assert report["reliable_fraction"] == 1.0
    assert report["unreliable_pairs"] == []
    print("  PASS\n")


def test_relaxes_gracefully_when_no_perfect_pixel_exists():
    print("=== 2. No pixel reliable everywhere -> graceful relaxation, honest report ===")
    h, w = 10, 10
    pairs = [("d1", "d2"), ("d1", "d3"), ("d2", "d3")]
    conncomp = {p: np.zeros((h, w), dtype=np.int32) for p in pairs}
    # Pixel (2,2) reliable in only 2/3 pairs; no pixel is reliable in all 3.
    conncomp[("d1", "d2")][2, 2] = 1
    conncomp[("d1", "d3")][2, 2] = 1
    # every other pixel reliable in at most 1 pair
    conncomp[("d2", "d3")][5, 5] = 1

    pixel, report = select_reliable_reference_pixel(conncomp, min_reliable_fraction=1.0, min_region_size=1)
    print(f"  chosen pixel: {pixel}, reliable_fraction={report['reliable_fraction']:.3f}, "
          f"unreliable_pairs={report['unreliable_pairs']}")
    assert report["reliable_fraction"] < 1.0
    assert pixel == (2, 2), f"expected the best-achievable pixel (2,2), got {pixel}"
    assert report["unreliable_pairs"] == [("d2", "d3")]
    print("  PASS\n")


def test_prefers_real_landmark_when_equally_good():
    print("=== 3. Preferred (real, georeferenced) point used when it's just as good ===")
    h, w = 15, 15
    pairs = [("d1", "d2"), ("d1", "d3")]
    conncomp = {p: np.zeros((h, w), dtype=np.int32) for p in pairs}
    # Two equally-good (fully reliable) pixels: (3,3) [the "preferred" real
    # landmark] and (10,10) [elsewhere in the scene].
    for p in pairs:
        conncomp[p][3, 3] = 1
        conncomp[p][10, 10] = 1

    pixel, report = select_reliable_reference_pixel(
        conncomp, preferred_point=(3, 3), search_radius_px=0, min_region_size=1,
    )
    print(f"  chosen pixel: {pixel}, searched_near_preferred={report['searched_near_preferred']}")
    assert pixel == (3, 3), "should prefer the real landmark when it's equally reliable"
    assert report["searched_near_preferred"] is True
    print("  PASS\n")

    # Now make the landmark genuinely WORSE than elsewhere -- should move.
    conncomp2 = {p: np.zeros((h, w), dtype=np.int32) for p in pairs}
    conncomp2[("d1", "d2")][3, 3] = 1  # landmark only reliable in 1/2 pairs
    for p in pairs:
        conncomp2[p][10, 10] = 1  # elsewhere reliable in both

    pixel2, report2 = select_reliable_reference_pixel(
        conncomp2, preferred_point=(3, 3), search_radius_px=0, min_region_size=1,
    )
    print(f"  chosen pixel: {pixel2}, searched_near_preferred={report2['searched_near_preferred']}")
    assert pixel2 == (10, 10), "should move away from a genuinely worse landmark"
    assert report2["searched_near_preferred"] is False
    print("  PASS\n")


def test_input_validation():
    print("=== 4. Input validation ===")
    try:
        select_reliable_reference_pixel({})
        raise AssertionError("expected ValueError for empty conncomp_masks")
    except ValueError as exc:
        print(f"  correctly rejected empty input: {exc}")

    try:
        select_reliable_reference_pixel({
            ("d1", "d2"): np.zeros((5, 5)),
            ("d1", "d3"): np.zeros((6, 6)),
        })
        raise AssertionError("expected ValueError for mismatched shapes")
    except ValueError as exc:
        print(f"  correctly rejected mismatched shapes: {exc}")
    print("  PASS\n")


# ── despike_velocity ────────────────────────────────────────────────────────

def test_despike_removes_isolated_spike():
    print("=== 5. despike_velocity removes a real, isolated single-pixel spike ===")
    h, w = 30, 30
    # Smooth, real-looking background trend (mimics genuine spatial
    # deformation gradient, e.g. a subsidence bowl).
    yy, xx = np.mgrid[0:h, 0:w]
    background = -0.1 * ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (w * h) * 50
    velocity = background.copy().astype(np.float32)

    # Inject one real, isolated cycle-slip-style spike.
    velocity[15, 15] += 500.0  # wildly larger than the smooth surrounding field

    filtered = despike_velocity(velocity, size=3)
    print(f"  original spike value: {velocity[15,15]:.2f}")
    print(f"  filtered value at spike: {filtered[15,15]:.2f}")
    print(f"  background value at that point: {background[15,15]:.2f}")
    assert abs(filtered[15, 15] - background[15, 15]) < 1.0, "spike should be removed, close to real background"

    # Real spatial structure elsewhere should be preserved (not smoothed
    # into meaninglessness) -- check a point away from the spike.
    away_diff = abs(filtered[5, 5] - background[5, 5])
    print(f"  difference from background away from the spike: {away_diff:.4f}")
    assert away_diff < 0.5, "smooth real spatial structure should survive mostly intact"
    print("  PASS\n")


def test_despike_nan_handling():
    print("=== 6. despike_velocity: NaN stays NaN, doesn't bias real neighbours ===")
    h, w = 20, 20
    velocity = np.full((h, w), 2.0, dtype=np.float32)
    velocity[10, 10] = np.nan  # one real "unreliable pixel" gap

    filtered = despike_velocity(velocity, size=3)
    assert np.isnan(filtered[10, 10]), "NaN pixel should stay NaN"
    # Neighbours right next to the NaN should still read the correct
    # constant value, not be pulled toward some filled-NaN placeholder.
    neighbour_vals = [filtered[9, 10], filtered[11, 10], filtered[10, 9], filtered[10, 11]]
    print(f"  neighbours of the NaN pixel: {neighbour_vals}")
    for v in neighbour_vals:
        assert abs(v - 2.0) < 1e-6, f"neighbour of a NaN pixel should be unaffected, got {v}"

    # valid_mask should force additional pixels to NaN.
    mask = np.ones((h, w), dtype=bool)
    mask[0, 0] = False
    filtered2 = despike_velocity(velocity, valid_mask=mask, size=3)
    assert np.isnan(filtered2[0, 0]), "valid_mask=False pixel should be forced to NaN"
    print("  PASS\n")


if __name__ == "__main__":
    test_reference_pixel_matches_bridge_unwrap_regions_criterion()
    test_finds_fully_reliable_pixel()
    test_relaxes_gracefully_when_no_perfect_pixel_exists()
    test_prefers_real_landmark_when_equally_good()
    test_input_validation()
    test_despike_removes_isolated_spike()
    test_despike_nan_handling()
    print("ALL TESTS PASSED")
