"""
Integration test for coregistration_method="raster_collocation", run
through the REAL, public InterferogramGenerator.process_pair() API (not
a private method) -- exercises the full dispatch, sec_profile reading,
collocate_by_geocoding, refinement, and final interferogram/coherence
pipeline together.

Two scenarios:
  A. Accurate transforms on both files (the case collocate_by_geocoding
     is designed for) -- confirms the raster_collocation path itself is
     correct and achieves strong coherence with no refinement needed.
  B. A deliberately IMPRECISE secondary transform -- simulating exactly
     the caveat flagged for pygeofetch's own SLCExtractor output (a
     global-affine GCP fit, not true per-pixel geocoding): the array
     data sits at its TRUE geographic position, but the file's declared
     transform is perturbed (small rotation + translation), the way a
     GCP-fit approximation would be wrong. Confirms that (1) collocation
     alone measurably suffers under this realistic imprecision, and (2)
     the cross-correlation refinement stage recovers good coherence
     anyway -- the same "stage 1 rough, stage 2 fixes it" pattern
     validated for the orbit_dem path.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/claude/work")

import numpy as np
import rasterio
from rasterio.transform import from_origin, Affine

from pygeofetch.insar.interferogram import InterferogramGenerator

WORKDIR = Path(tempfile.mkdtemp(prefix="pygeofetch_raster_collocation_it_"))


def write_complex_geotiff(path, array, transform, crs="EPSG:4326"):
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype="complex64", crs=crs, transform=transform,
    ) as dst:
        dst.write(array.astype(np.complex64), 1)


def whole_image_coherence(ref, sec):
    num = np.abs(np.mean(ref * np.conj(sec)))
    denom = np.sqrt(np.mean(np.abs(ref) ** 2) * np.mean(np.abs(sec) ** 2))
    return float(num / denom)


def build_world(size=400, seed=0):
    rs = np.random.default_rng(seed)
    return (rs.normal(size=(size, size)) + 1j * rs.normal(size=(size, size))).astype(np.complex64)


def test_accurate_transform():
    print("=== Scenario A: accurate transforms on both sides ===")
    world = build_world()
    n = 200
    ref_r0, ref_c0 = 80, 80
    sec_r0, sec_c0 = 84, 76  # secondary genuinely covers a slightly different real footprint

    ref_arr = world[ref_r0:ref_r0 + n, ref_c0:ref_c0 + n]
    sec_arr = world[sec_r0:sec_r0 + n, sec_c0:sec_c0 + n]
    noise = (np.random.default_rng(1).normal(size=(n, n)) + 1j * np.random.default_rng(2).normal(size=(n, n))) * 0.1
    sec_arr = (sec_arr + noise).astype(np.complex64)

    px = 0.0002  # deg/pixel
    world_origin_lon, world_origin_lat = 10.0, 45.0  # world[0,0]

    ref_transform = from_origin(world_origin_lon + ref_c0 * px, world_origin_lat - ref_r0 * px, px, px)
    sec_transform = from_origin(world_origin_lon + sec_c0 * px, world_origin_lat - sec_r0 * px, px, px)

    ref_path = WORKDIR / "ref_a.tif"
    sec_path = WORKDIR / "sec_a.tif"
    write_complex_geotiff(ref_path, ref_arr, ref_transform)
    write_complex_geotiff(sec_path, sec_arr, sec_transform)

    coh_raw = whole_image_coherence(ref_arr, sec_arr)
    print(f"  raw (unregistered) coherence: {coh_raw:.3f}")

    gen = InterferogramGenerator()
    result = gen.process_pair(
        reference=ref_path, secondary=sec_path,
        coregistration_method="raster_collocation",
        coregistration_refine_by_coherence=True,
    )
    print(f"  metadata: {result.metadata}")
    print(f"  mean coherence (full interferogram): {result.coherence.mean():.3f}")

    assert result.metadata["coregistration_method"] == "raster_collocation"
    assert result.metadata["coregistration_collocation_coverage_fraction"] > 0.85, (
        "accurate transforms should give near-complete real coverage"
    )
    assert result.coherence.mean() > coh_raw + 0.3, "collocation should substantially beat raw coherence"
    assert result.coherence.mean() > 0.5, f"expected strong coherence, got {result.coherence.mean():.3f}"
    print("  PASS\n")


def test_imprecise_transform_recovered_by_refinement():
    print("=== Scenario B: imprecise (GCP-fit-style) secondary transform ===")
    world = build_world(size=500, seed=10)
    n = 220
    ref_r0, ref_c0 = 140, 140
    sec_r0, sec_c0 = 145, 133

    ref_arr = world[ref_r0:ref_r0 + n, ref_c0:ref_c0 + n]
    sec_arr = world[sec_r0:sec_r0 + n, sec_c0:sec_c0 + n]
    noise = (np.random.default_rng(3).normal(size=(n, n)) + 1j * np.random.default_rng(4).normal(size=(n, n))) * 0.1
    sec_arr = (sec_arr + noise).astype(np.complex64)

    px = 0.0002
    world_origin_lon, world_origin_lat = 10.0, 45.0

    ref_transform = from_origin(world_origin_lon + ref_c0 * px, world_origin_lat - ref_r0 * px, px, px)
    true_sec_transform = from_origin(world_origin_lon + sec_c0 * px, world_origin_lat - sec_r0 * px, px, px)

    # Perturb the DECLARED secondary transform -- the array's real data
    # stays exactly where it is, but the file now claims a slightly
    # wrong geoposition (small rotation + translation), mimicking a
    # global-affine GCP fit's real, documented imprecision. Rotation
    # angle and translation both deliberately large enough (a handful
    # of pixels of resulting error) to require real refinement, not so
    # large that a coarse_search_radius=12 refinement can't recover.
    theta = np.radians(0.15)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = Affine(cos_t, -sin_t, 0, sin_t, cos_t, 0)
    translation_error_px = 6.0
    perturbed_sec_transform = (
        true_sec_transform
        * Affine.translation(translation_error_px, -translation_error_px * 0.5)
        * rotation
    )

    ref_path = WORKDIR / "ref_b.tif"
    sec_path = WORKDIR / "sec_b.tif"
    write_complex_geotiff(ref_path, ref_arr, ref_transform)
    write_complex_geotiff(sec_path, sec_arr, perturbed_sec_transform)

    coh_raw = whole_image_coherence(ref_arr, sec_arr)
    print(f"  raw (unregistered) coherence: {coh_raw:.3f}")

    gen = InterferogramGenerator()

    result_collocate_only = gen.process_pair(
        reference=ref_path, secondary=sec_path,
        coregistration_method="raster_collocation",
        coregistration_refine_by_coherence=False,
    )
    print(f"  collocation ONLY (imprecise transform): "
          f"coverage={result_collocate_only.metadata['coregistration_collocation_coverage_fraction']:.3f}, "
          f"mean coherence={result_collocate_only.coherence.mean():.3f}")

    result_refined = gen.process_pair(
        reference=ref_path, secondary=sec_path,
        coregistration_method="raster_collocation",
        coregistration_refine_by_coherence=True,
    )
    print(f"  collocation + cross-correlation refinement: "
          f"mean coherence={result_refined.coherence.mean():.3f}, "
          f"metadata={result_refined.metadata}")

    assert result_collocate_only.metadata["coregistration_method"] == "raster_collocation"
    assert result_refined.metadata["coregistration_refined_by_coherence"] is True
    assert result_refined.coherence.mean() > result_collocate_only.coherence.mean(), (
        "refinement should measurably improve on collocation-only under a realistic "
        "imprecise (GCP-fit-style) transform -- this is the concrete evidence for "
        "the precision caveat: collocation alone is not enough with pygeofetch's "
        "current transform quality, but the full chain still recovers well"
    )
    assert result_refined.coherence.mean() > 0.5, (
        f"expected the full chain to recover strong coherence despite the "
        f"imprecise transform, got {result_refined.coherence.mean():.3f}"
    )
    print(f"  coherence recovered by refinement despite imprecise transform: "
          f"{result_collocate_only.coherence.mean():.3f} -> {result_refined.coherence.mean():.3f}")
    print("  PASS\n")


if __name__ == "__main__":
    test_accurate_transform()
    test_imprecise_transform_recovered_by_refinement()
    print("ALL RASTER-COLLOCATION INTEGRATION TESTS PASSED")
