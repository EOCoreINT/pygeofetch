# """
# End-to-end integration test for _orbit_based_coregister's real pipeline:

#     compute_offset_field_from_dem()  (existing, unmodified)
#       -> refine_offsets_by_coherence()   (new)
#       -> fit_offset_polynomial_robust()  (new)
#       -> resample_with_offset_field()  (existing, unmodified)

# Builds a physically self-consistent synthetic SLC geometry + orbit pair
# (bypassing real SAFE/EOF file parsing via monkeypatching, since those
# parsers are independently tested elsewhere and not the subject of this
# change) plus a real, on-disk DEM GeoTIFF, then runs the actual
# InterferogramGenerator._orbit_based_coregister() method -- not a
# reimplementation of it -- and checks that:

#   1. The DEM/orbit-based offset field is computed successfully.
#   2. A deliberately injected sub-pixel misregistration bias (the kind
#      an orbit/DEM model alone can't capture) is corrected by the
#      cross-correlation refinement stage.
#   3. Coherence after full coregistration is substantially higher than
#      with refinement disabled -- the concrete "before/after" the
#      original bug report was about.
#   4. Cropped inputs (AOI-cropped reference/secondary, different crop
#      offsets from each other) are still handled correctly.
# """
# import math
# import sys
# import tempfile
# from datetime import datetime, timedelta
# from pathlib import Path


# import numpy as np
# import rasterio
# from rasterio.transform import from_origin

# from pygeofetch.insar.annotation import SLCGeometry
# from pygeofetch.insar.geolocation import geodetic_to_ecef
# from pygeofetch.insar.coregister import compute_offset_field_from_dem, fit_offset_polynomial_robust
# from pygeofetch.insar.interferogram import InterferogramGenerator
# import pygeofetch.insar.annotation as annotation_mod
# import pygeofetch.insar.geolocation as geolocation_mod

# WORKDIR = Path(tempfile.mkdtemp(prefix="pygeofetch_coreg_it_"))
# SPEED_OF_LIGHT = 299792458.0
# GM_EARTH = 3.986004418e14


# # ── 1. Build a physically self-consistent synthetic orbit + geometry ──────

# def build_scene(n=512, target_lat=45.0, target_lon=10.0, incidence_deg=35.0,
#                  perp_baseline_m=120.0, seed=0):
#     t0 = datetime(2026, 6, 1, 0, 0, 0)

#     target_ecef = np.array(geodetic_to_ecef(target_lat, target_lon, 0.0))
#     up = target_ecef / np.linalg.norm(target_ecef)
#     east = np.cross([0, 0, 1], target_ecef)
#     east = east / np.linalg.norm(east)
#     north = np.cross(up, east)
#     north = north / np.linalg.norm(north)

#     inc = math.radians(incidence_deg)
#     h_sat = 693000.0
#     slant_range = h_sat / math.cos(inc)
#     look_dir = math.cos(inc) * up + math.sin(inc) * east
#     look_dir = look_dir / np.linalg.norm(look_dir)

#     sat_pos_ref = target_ecef + slant_range * look_dir
#     # Small perpendicular-baseline offset for the secondary pass, along
#     # a direction orthogonal to both velocity (north) and look_dir --
#     # i.e. genuinely "perpendicular" in the InSAR sense.
#     perp_dir = np.cross(north, look_dir)
#     perp_dir = perp_dir / np.linalg.norm(perp_dir)
#     sat_pos_sec = sat_pos_ref + perp_baseline_m * perp_dir

#     orbital_speed = math.sqrt(GM_EARTH / (6378137.0 + h_sat))
#     velocity = orbital_speed * north  # broadside (zero-Doppler) at t=0 by construction

#     orbit_times = [t0 + timedelta(seconds=float(s)) for s in range(-10, 11)]

#     def straight_line_orbit(sat_pos_center):
#         positions = [tuple(sat_pos_center + velocity * (t - t0).total_seconds()) for t in orbit_times]
#         velocities = [tuple(velocity) for _ in orbit_times]
#         return orbit_times, positions, velocities

#     ref_orbit = straight_line_orbit(sat_pos_ref)
#     sec_orbit = straight_line_orbit(sat_pos_sec)

#     # SLCGeometry timing: centre row/col at t=0 / slant_range respectively.
#     az_interval = 0.002
#     range_sampling_rate = 6.0e7
#     range_time_center = 2 * slant_range / SPEED_OF_LIGHT

#     # The image's real ground footprint (approximate, flat-earth-local),
#     # used below to size the GeoTIFFs written for the reference/
#     # secondary arrays so their geographic bounds actually match what
#     # the synthetic geometry covers -- NOT the (much larger) DEM
#     # extent. Azimuth (row) direction runs along `north`; range (col)
#     # direction's ground-projected component runs along `east`.
#     az_half_extent_m = (n / 2) * az_interval * orbital_speed
#     ground_range_half_extent_m = (n / 2) / range_sampling_rate * (SPEED_OF_LIGHT / 2) / math.sin(inc)
#     # Exact (unmargined) footprint -- this MUST match what
#     # row 0..n-1 / col 0..n-1 correspond to geographically per
#     # SLCGeometry's own timing (az_interval/range_sampling_rate), since
#     # it's used below to size the GeoTIFF pixel grid. Any margin needed
#     # for DEM sample_bounds slack is applied separately, on top of
#     # these file bounds, not baked into the pixel grid itself.
#     img_half_extent_lat_deg = az_half_extent_m / 111320.0
#     img_half_extent_lon_deg = ground_range_half_extent_m / (111320.0 * math.cos(math.radians(target_lat)))
#     px_lon = 2 * img_half_extent_lon_deg / n  # degrees per column (full-scene)
#     px_lat = 2 * img_half_extent_lat_deg / n  # degrees per row (full-scene)
#     origin_lon = target_lon - img_half_extent_lon_deg  # full-scene upper-left
#     origin_lat = target_lat + img_half_extent_lat_deg

#     ref_geom = SLCGeometry(
#         first_line_time=t0 - timedelta(seconds=(n / 2) * az_interval),
#         azimuth_time_interval_s=az_interval,
#         near_range_time_s=range_time_center - (n / 2) / range_sampling_rate,
#         range_sampling_rate_hz=range_sampling_rate,
#         n_lines=n, n_columns=n,
#     )
#     # Secondary geometry: same convention (independent orbits can in
#     # principle use different timing/sampling, but re-using the same
#     # convention here isolates the geometric-offset effect we want to
#     # test, rather than also mixing in an arbitrary timing-grid change).
#     sec_geom = SLCGeometry(
#         first_line_time=t0 - timedelta(seconds=(n / 2) * az_interval),
#         azimuth_time_interval_s=az_interval,
#         near_range_time_s=range_time_center - (n / 2) / range_sampling_rate,
#         range_sampling_rate_hz=range_sampling_rate,
#         n_lines=n, n_columns=n,
#     )

#     # Real, on-disk DEM covering the target area with margin, flat
#     # elevation (500 m) -- keeps the geometry solve well-conditioned;
#     # the point of this test is the coregistration pipeline, not DEM
#     # relief handling (already covered by compute_offset_field_from_dem's
#     # own bounds-clamping logic).
#     dem_size = 700
#     half_extent_deg = 0.08
#     dem_transform = from_origin(
#         target_lon - half_extent_deg, target_lat + half_extent_deg,
#         2 * half_extent_deg / dem_size, 2 * half_extent_deg / dem_size,
#     )
#     dem_path = WORKDIR / "dem.tif"
#     with rasterio.open(
#         dem_path, "w", driver="GTiff", height=dem_size, width=dem_size,
#         count=1, dtype="float32", crs="EPSG:4326", transform=dem_transform,
#     ) as dst:
#         dst.write(np.full((dem_size, dem_size), 500.0, dtype=np.float32), 1)

#     return dict(
#         ref_geom=ref_geom, sec_geom=sec_geom,
#         ref_orbit=ref_orbit, sec_orbit=sec_orbit,
#         dem_path=dem_path, target_lat=target_lat, target_lon=target_lon,
#         half_extent_deg=half_extent_deg, n=n,
#         img_half_extent_lat_deg=img_half_extent_lat_deg,
#         img_half_extent_lon_deg=img_half_extent_lon_deg,
#         px_lon=px_lon, px_lat=px_lat, origin_lon=origin_lon, origin_lat=origin_lat,
#     )


# def write_geotiff(path, array, scene, crop_row_off=0.0, crop_col_off=0.0):
#     """Write a small real GeoTIFF for `array` covering its own real
#     ground footprint (the crop's sub-window of the full synthetic
#     scene, if crop_row_off/crop_col_off are given) -- used only so
#     _orbit_based_coregister can read real bounds (for sample_bounds)
#     and crop-offset tags from it, the same way it would from a real
#     SLCExtractor output. NOTE: deliberately NOT the DEM's (larger)
#     extent -- using the DEM's extent here would tell the real pipeline
#     to sample DEM ground points far outside where this synthetic image
#     actually has valid geometry, exactly the DEM-vs-crop-extent
#     mismatch compute_offset_field_from_dem's own sample_bounds
#     clamping exists to avoid."""
#     crop_origin_lon = scene["origin_lon"] + crop_col_off * scene["px_lon"]
#     crop_origin_lat = scene["origin_lat"] - crop_row_off * scene["px_lat"]
#     transform = from_origin(crop_origin_lon, crop_origin_lat, scene["px_lon"], scene["px_lat"])
#     with rasterio.open(
#         path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
#         count=1, dtype="complex64", crs="EPSG:4326", transform=transform,
#     ) as dst:
#         dst.write(array.astype(np.complex64), 1)
#         dst.update_tags(crop_row_off=crop_row_off, crop_col_off=crop_col_off)


# # ── 2. Run compute_offset_field_from_dem() for real, to get the "pure
# #    geometric" (baseline-only) offset field the orbit/DEM model predicts ──

# def get_geometric_offset_functions(scene):
#     margin = 1.2  # slack for this standalone helper's own DEM sampling
#     sample_bounds = (
#         scene["target_lon"] - margin * scene["img_half_extent_lon_deg"],
#         scene["target_lat"] - margin * scene["img_half_extent_lat_deg"],
#         scene["target_lon"] + margin * scene["img_half_extent_lon_deg"],
#         scene["target_lat"] + margin * scene["img_half_extent_lat_deg"],
#     )
#     grid_rows, grid_cols, off_rows, off_cols = compute_offset_field_from_dem(
#         scene["ref_geom"], scene["ref_orbit"], scene["sec_geom"], scene["sec_orbit"],
#         scene["dem_path"],
#         ref_scene_center_time=scene["ref_geom"].azimuth_time(scene["n"] / 2),
#         sec_scene_center_time=scene["sec_geom"].azimuth_time(scene["n"] / 2),
#         grid_points=7, sample_bounds=sample_bounds,
#     )
#     row_fn, col_fn, quality = fit_offset_polynomial_robust(
#         grid_rows, grid_cols, off_rows, off_cols, degree=1,
#     )
#     print(f"  [setup] pure geometric (baseline-only) offset field: "
#           f"{len(grid_rows)} GCPs, row_fn(center)={row_fn(scene['n']/2, scene['n']/2):.4f} px, "
#           f"col_fn(center)={col_fn(scene['n']/2, scene['n']/2):.4f} px")
#     return row_fn, col_fn


# def make_speckle_pair(scene, row_fn, col_fn, extra_bias, noise_level=0.1, seed=3):
#     """ref: random speckle. sec: ref resampled by the TRUE total offset
#     field (pure orbit/DEM geometry + an extra, deliberately unmodeled
#     sub-pixel bias, simulating real residual timing/orbit-precision
#     error) plus independent decorrelation noise."""
#     from scipy.ndimage import map_coordinates

#     n = scene["n"]
#     rs = np.random.default_rng(seed)
#     ref = (rs.normal(size=(n, n)) + 1j * rs.normal(size=(n, n))).astype(np.complex128)

#     row_idx, col_idx = np.mgrid[0:n, 0:n].astype(np.float64)
#     true_row_offset = row_fn(row_idx, col_idx) + extra_bias[0]
#     true_col_offset = col_fn(row_idx, col_idx) + extra_bias[1]
#     sample_rows = row_idx + true_row_offset
#     sample_cols = col_idx + true_col_offset

#     sec_real = map_coordinates(ref.real, [sample_rows, sample_cols], order=1, mode="constant")
#     sec_imag = map_coordinates(ref.imag, [sample_rows, sample_cols], order=1, mode="constant")
#     sec = sec_real + 1j * sec_imag
#     noise = (rs.normal(size=(n, n)) + 1j * rs.normal(size=(n, n))) * noise_level
#     sec = (sec + noise).astype(np.complex64)
#     return ref.astype(np.complex64), sec, true_row_offset, true_col_offset


# def whole_image_coherence(ref, sec):
#     num = np.abs(np.mean(ref * np.conj(sec)))
#     denom = np.sqrt(np.mean(np.abs(ref) ** 2) * np.mean(np.abs(sec) ** 2))
#     return float(num / denom)


# def run_coregister(scene, ref_complex, sec_complex, ref_path, sec_path, refine_by_coherence):
#     gen = InterferogramGenerator()
#     # Monkeypatch the file-parsing entry points only -- everything
#     # downstream (compute_offset_field_from_dem, refine_offsets_by_coherence,
#     # fit_offset_polynomial_robust, resample_with_offset_field) is the
#     # REAL, unmodified code under test.
#     annotation_mod.parse_slc_geometry = lambda path, member_hint=None: (
#         scene["ref_geom"] if "ref" in str(path) else scene["sec_geom"]
#     )
#     geolocation_mod.parse_orbit_file = lambda path: (
#         scene["ref_orbit"] if "ref" in str(path) else scene["sec_orbit"]
#     )
#     resampled, coreg_metadata = gen._orbit_based_coregister(
#         ref_complex, sec_complex, scene["dem_path"],
#         "ref_safe.zip", "sec_safe.zip", "ref_orbit.eof", "sec_orbit.eof",
#         reference_extracted_path=str(ref_path), secondary_extracted_path=str(sec_path),
#         refine_by_coherence=refine_by_coherence,
#     )
#     return resampled, coreg_metadata


# def test_full_pipeline_no_crop():
#     print("=== Integration test 1: full pipeline, uncropped ===")
#     scene = build_scene()
#     row_fn, col_fn = get_geometric_offset_functions(scene)

#     # A deliberate extra sub-pixel bias the orbit/DEM model does NOT
#     # know about -- exactly the residual refine_offsets_by_coherence is
#     # meant to close.
#     extra_bias = (0.8, -0.6)
#     ref_complex, sec_complex, true_row, true_col = make_speckle_pair(
#         scene, row_fn, col_fn, extra_bias, noise_level=0.1,
#     )

#     ref_path = WORKDIR / "ref_scene.tif"
#     sec_path = WORKDIR / "sec_scene.tif"
#     write_geotiff(ref_path, ref_complex, scene)
#     write_geotiff(sec_path, sec_complex, scene)

#     coh_raw = whole_image_coherence(ref_complex, sec_complex)
#     print(f"  raw (unregistered) whole-image coherence: {coh_raw:.3f}")

#     resampled_refined, meta_refined = run_coregister(
#         scene, ref_complex, sec_complex, ref_path, sec_path, refine_by_coherence=True,
#     )
#     coh_refined = whole_image_coherence(ref_complex, resampled_refined)
#     print(f"  WITH cross-correlation refinement: coherence={coh_refined:.3f}, "
#           f"metadata={meta_refined}")

#     resampled_unrefined, meta_unrefined = run_coregister(
#         scene, ref_complex, sec_complex, ref_path, sec_path, refine_by_coherence=False,
#     )
#     coh_unrefined = whole_image_coherence(ref_complex, resampled_unrefined)
#     print(f"  WITHOUT refinement (orbit/DEM estimate only): coherence={coh_unrefined:.3f}, "
#           f"metadata={meta_unrefined}")

#     assert meta_refined["method"] == "orbit_dem_based"
#     assert meta_refined["refined_by_coherence"] is True
#     assert meta_unrefined["refined_by_coherence"] is False
#     assert coh_refined > coh_raw, "coregistration should improve on raw coherence"
#     assert coh_refined > coh_unrefined, (
#         "cross-correlation refinement should recover the injected sub-pixel "
#         "bias and beat the orbit/DEM-only estimate"
#     )
#     assert coh_refined > 0.7, f"expected strong final coherence, got {coh_refined:.3f}"
#     print(f"  coherence improvement from refinement: {coh_unrefined:.3f} -> {coh_refined:.3f}")
#     print("  PASS\n")
#     return scene, ref_complex, sec_complex, true_row, true_col


# def test_cropped_inputs(scene):
#     print("=== Integration test 2: cropped reference/secondary, different crop offsets ===")
#     row_fn, col_fn = get_geometric_offset_functions(scene)
#     extra_bias = (0.5, 0.4)
#     full_ref, full_sec, _, _ = make_speckle_pair(scene, row_fn, col_fn, extra_bias, seed=7)

#     n = scene["n"]
#     crop = 300
#     # Reference crop starts at (80, 60); secondary crop starts at a
#     # DIFFERENT local origin (150, 110) -- the realistic case where two
#     # independent AOI extractions don't share the same crop window.
#     ref_r0, ref_c0 = 80, 60
#     sec_r0, sec_c0 = 150, 110
#     ref_crop = full_ref[ref_r0:ref_r0 + crop, ref_c0:ref_c0 + crop]
#     sec_crop = full_sec[sec_r0:sec_r0 + crop, sec_c0:sec_c0 + crop]

#     ref_path = WORKDIR / "ref_crop.tif"
#     sec_path = WORKDIR / "sec_crop.tif"
#     write_geotiff(ref_path, ref_crop, scene, crop_row_off=ref_r0, crop_col_off=ref_c0)
#     write_geotiff(sec_path, sec_crop, scene, crop_row_off=sec_r0, crop_col_off=sec_c0)

#     coh_raw = whole_image_coherence(ref_crop, sec_crop)
#     resampled, meta = run_coregister(
#         scene, ref_crop, sec_crop, ref_path, sec_path, refine_by_coherence=True,
#     )
#     coh_after = whole_image_coherence(ref_crop, resampled)
#     print(f"  raw coherence (crop, unregistered): {coh_raw:.3f}")
#     print(f"  coherence after full pipeline (mismatched crop offsets): {coh_after:.3f}")
#     print(f"  metadata: {meta}")

#     assert meta["method"] == "orbit_dem_based", "should not have fallen back to shape-based"
#     assert coh_after > coh_raw + 0.3, "coregistration must correctly handle differing crop offsets"
#     assert coh_after > 0.4, f"expected clearly improved coherence despite crop mismatch, got {coh_after:.3f}"
#     print("  PASS\n")


# if __name__ == "__main__":
#     scene, *_ = test_full_pipeline_no_crop()
#     test_cropped_inputs(scene)
#     print("ALL INTEGRATION TESTS PASSED")
