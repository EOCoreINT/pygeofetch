"""
Real orbit-based coregistration offset field.

Combines annotation.py (real per-pixel acquisition timing) and
geolocation.py (the orbit-based Doppler/Range/Ellipsoid solve) to
compute a genuine, physically-grounded offset field between a
reference and secondary SLC — replacing the naive shape-only resample
that was previously used regardless of whether the two images were
actually aligned.

Method (standard practice, e.g. Kampes, Hanssen & Perski 2003; matches
what the 2019 URSI paper and the current TOPS coregistration literature
both describe, and mirrors the stage structure of ESA SNAP's own
coregistration chain -- CreateStack / CrossCorrelationOp / WarpOp):

  1. compute_offset_field_from_dem(): a genuine, physically-grounded
     FIRST ESTIMATE of the offset at a sparse grid of points across
     the image (not every pixel — solving per pixel would be far too
     slow and offers no real accuracy benefit, since the offset field
     is smooth), using ground points sampled directly from a real DEM.
     Accurate to roughly a pixel, bounded by orbit/timing/DEM
     precision — analogous to SNAP's initial GCP geopositioning.
  2. refine_offsets_by_coherence() [optional but recommended]: SNAP
     CrossCorrelationOp's own two-stage refinement -- coarse integer-
     pixel cross-correlation of imagettes, then sub-pixel coherence
     maximization (Powell's method) -- run directly against the real
     image content to close the sub-pixel gap step 1 leaves. This is
     usually what separates chronically low coherence from a properly
     registered pair: even a fraction of a pixel of true residual
     misregistration decorrelates two otherwise-identical complex
     samples substantially.
  3. fit_offset_polynomial_robust(): SNAP WarpOp's iterative mean-RMS
     GCP outlier rejection, then a low-degree (1-3) 2D polynomial fit
     to the surviving grid -- rather than a single unweighted
     least-squares pass that a handful of bad points could dominate.
  4. resample_with_offset_field(): apply the fitted, dense field to
     resample the secondary image onto the reference grid.

References for the CrossCorrelationOp/WarpOp stage structure:
  https://step.esa.int/main/wp-content/help/versions/13.0.0/snap-toolboxes/eu.esa.microwavetbx.sar.op.insar.ui/operators/CrossCorrelationOp.html
  https://step.esa.int/main/wp-content/help/versions/13.0.0/snap-toolboxes/eu.esa.microwavetbx.sar.op.insar.ui/operators/WarpOp.html
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

logger = logging.getLogger("pygeofetch.insar.coregister")


@dataclass
class CoregistrationQuality:
    """
    Residual quality metrics for a fitted offset-field polynomial —
    the equivalent of SNAP WarpOp's "residual file" (RMS mean/std,
    row/column residual mean/std, GCP counts before and after
    filtering), returned so callers can actually validate whether
    coregistration succeeded instead of only getting back a callable
    polynomial with no way to tell if it was fit on good points.

    All residuals are in pixels, measured in the same (row, col) units
    as the offset field itself (i.e. against the reference image grid).
    """

    n_gcps_initial: int
    n_gcps_final: int
    n_gcps_rejected: int
    rms_mean: float
    rms_std: float
    row_residual_mean: float
    row_residual_std: float
    col_residual_mean: float
    col_residual_std: float
    per_point_rms: List[float]
    iterations_used: int
    degree: int
    # Populated by callers that ran refine_offsets_by_coherence() before
    # fitting; None if the field going into the fit was orbit/DEM-only.
    mean_coherence: Optional[float] = None
    # Set by fit_geometry_first_offsets(): how the final field was built
    # ("geometry_only", "geometry+constant" or "geometry+linear") and how many
    # orbit/DEM points defined the base field. None for the plain polynomial fit.
    mode: Optional[str] = None
    n_geometry_points: Optional[int] = None

    def log_summary(self, level: int = logging.INFO) -> None:
        coh_part = (
            f", mean fine-registration coherence {self.mean_coherence:.3f}"
            if self.mean_coherence is not None
            else ""
        )
        logger.log(
            level,
            "Coregistration quality: %d/%d GCPs kept after %d outlier-"
            "rejection iteration(s) (degree-%d warp) — RMS mean %.3f px, "
            "std %.3f px [row residual mean/std %.3f/%.3f px, col "
            "residual mean/std %.3f/%.3f px]%s",
            self.n_gcps_final,
            self.n_gcps_initial,
            self.iterations_used,
            self.degree,
            self.rms_mean,
            self.rms_std,
            self.row_residual_mean,
            self.row_residual_std,
            self.col_residual_mean,
            self.col_residual_std,
            coh_part,
        )

    def is_reliable(self, min_gcps: int = 6, max_rms_px: float = 1.0) -> bool:
        """
        A quick, conservative pass/fail gate — not a substitute for
        looking at the actual numbers, but useful for automated
        pipelines that need to decide whether to trust a pair's
        coregistration or flag it for review. SNAP doesn't impose a
        hard default here either (RMS Threshold is user-set); 1.0 px
        and 6 GCPs (enough to safely support a degree-2 fit with some
        redundancy) are reasonable, conservative starting points, not
        universal constants.
        """
        return self.n_gcps_final >= min_gcps and self.rms_mean <= max_rms_px

    def to_dict(self) -> dict:
        return {
            "n_gcps_initial": self.n_gcps_initial,
            "n_gcps_final": self.n_gcps_final,
            "n_gcps_rejected": self.n_gcps_rejected,
            "rms_mean": self.rms_mean,
            "rms_std": self.rms_std,
            "row_residual_mean": self.row_residual_mean,
            "row_residual_std": self.row_residual_std,
            "col_residual_mean": self.col_residual_mean,
            "col_residual_std": self.col_residual_std,
            "iterations_used": self.iterations_used,
            "degree": self.degree,
            "mean_coherence": self.mean_coherence,
            "mode": self.mode,
            "n_geometry_points": self.n_geometry_points,
        }


def _min_points_for_degree(degree: int) -> int:
    """Number of terms in a full 2D polynomial of the given degree —
    (degree+1)(degree+2)/2 — i.e. the minimum GCPs a fit needs to be
    non-degenerate. degree 1 -> 3, degree 2 -> 6, degree 3 -> 10."""
    return (degree + 1) * (degree + 2) // 2


def read_matched_swath(path: Union[str, Path]) -> Optional[str]:
    """
    Read back the matched sub-swath label (e.g. "iw3") recorded by
    SLCExtractor.extract_scene(). Returns None if not present, so
    callers can fall back to parse_slc_geometry()'s default (arbitrary
    first-match) behaviour rather than fail outright.
    """
    import rasterio

    try:
        with rasterio.open(path) as src:
            return src.tags().get("matched_swath") or None
    except Exception:
        return None


def read_crop_offset(path: Union[str, Path]) -> Tuple[float, float]:
    """
    Read back the crop offset recorded by SLCExtractor._crop_to_aoi() -
    the local array's origin in the original, uncropped scene's pixel
    coordinates. Returns (0.0, 0.0) if the file wasn't cropped (either
    it's the raw fallback extraction, or a file this metadata predates),
    which is the correct default: no crop means local and full-scene
    coordinates are already the same thing.
    """
    import math

    import rasterio

    try:
        with rasterio.open(path) as src:
            tags = src.tags()
            row = float(tags.get("crop_row_off", 0.0))
            col = float(tags.get("crop_col_off", 0.0))
    except Exception:
        return (0.0, 0.0)
    # Files cropped before 2026-11-09 may carry fractional offsets, but rasterio read them from the NEAREST whole
    # pixel (half rounds up). The data's true origin is that whole pixel, so use it: taking the fractional value
    # misregistered every pair by the difference of its two dates' fractions (up to ~0.5 px).
    snapped = (float(math.floor(row + 0.5)), float(math.floor(col + 0.5)))
    if snapped != (row, col):
        logger.info(
            "%s: fractional crop offset (%.3f, %.3f) in the tags; the data starts at whole pixel (%d, %d) "
            "(rasterio rounds window offsets), which is what the geometry now uses",
            Path(str(path)).name, row, col, int(snapped[0]), int(snapped[1]),
        )
    return snapped


def collocate_by_geocoding(
    secondary_data,
    secondary_profile,
    reference_shape,
    reference_profile,
    resampling: str = "bilinear",
):
    """
    SNAP CreateStackOp equivalent: resample the secondary image into
    the reference image's own geographic raster, using each product's
    real, embedded CRS + affine transform -- exactly what CreateStack
    does before handing off to CrossCorrelationOp:
    https://step.esa.int/main/wp-content/help/versions/13.0.0/snap-toolboxes/eu.esa.microwavetbx.sar.op.insar.ui/operators/CreateStackOp.html
    ("the pixel values of the secondary product are resampled into the
    geographical raster of the reference product... the geographical
    position of a reference sample is used to find the corresponding
    sample in the secondary raster").

    IMPORTANT PRECISION CAVEAT, specific to this project (not a SNAP
    limitation): SNAP's CreateStack collocates using each product's
    own precise geocoding, generally accurate to a small fraction of a
    pixel. If secondary_profile/reference_profile come from
    SLCExtractor's AOI-cropped output, their transform is instead a
    single GLOBAL AFFINE FIT to the SAFE product's embedded GCPs
    (rasterio.transform.from_gcps in extraction.py) -- explicitly
    documented there as an approximation good enough to safely window
    an AOI crop (with a 15%+ safety margin specifically because of its
    known error), NOT verified to sub-pixel or even few-pixel accuracy
    across a scene. A single global affine also cannot capture SAR's
    real range-dependent, non-linear geocoding distortion the way a
    proper per-pixel or dense tie-point geocoding would. This function
    is fully correct and general -- pass it any two profiles with a
    genuinely accurate CRS + transform (e.g. from a properly
    terrain-geocoded product) and it collocates as precisely as that
    input's own geopositioning allows. With pygeofetch's current
    SLCExtractor output specifically, treat its result as a coarse
    first alignment that still needs (and, in
    interferogram.py's _raster_collocation_coregister(), gets) real
    cross-correlation refinement afterward, not a finished
    coregistration on its own.

    Args:
        secondary_data:    Complex secondary SLC array.
        secondary_profile: rasterio-style profile for secondary_data;
                       must include "crs" and "transform".
        reference_shape:   (rows, cols) of the reference image -- the
                       output raster this collocates onto.
        reference_profile: rasterio-style profile for the reference
                       image; must include "crs" and "transform".
        resampling:    "nearest", "bilinear", or "cubic" -- the same
                       three methods SNAP's CreateStack offers (it
                       also offers "None", not meaningful here since
                       InSAR always needs an actual resample onto a
                       different grid).

    Returns:
        (collocated_data, coverage_fraction, valid_mask):
          - collocated_data: secondary resampled onto reference_shape,
            complex64; pixels with no real geographic counterpart in
            the secondary are 0+0j (consistent with
            resample_with_offset_field's own out-of-bounds convention).
          - coverage_fraction: fraction of reference pixels that found
            real secondary coverage. SNAP's own UI surfaces this only
            implicitly (via the Reference/Maximum/Minimum extent
            choice); here it's explicit so callers can catch a
            nearly-zero-overlap case -- most likely a genuine
            coordinate-frame problem, exactly the bug class this
            function exists to make visible -- immediately rather than
            discovering it later as inexplicably low coherence.
          - valid_mask: boolean array, True where collocation found
            real coverage -- lets callers exclude nodata pixels from
            downstream GCP selection or coherence estimation.

    Raises:
        ValueError if either profile is missing crs/transform --
        collocation is meaningless without real geopositioning on both
        sides, so this fails loudly rather than silently degrading to
        a shape-only assumption.
    """
    import numpy as np
    from rasterio.warp import Resampling, reproject

    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    if resampling not in resampling_map:
        raise ValueError(
            f"resampling must be one of {list(resampling_map)}, got {resampling!r}"
        )

    sec_crs = secondary_profile.get("crs")
    sec_transform = secondary_profile.get("transform")
    ref_crs = reference_profile.get("crs")
    ref_transform = reference_profile.get("transform")
    if sec_crs is None or sec_transform is None:
        raise ValueError(
            "collocate_by_geocoding: secondary_profile is missing "
            "crs/transform -- real geopositioning is required, this "
            "is not a shape-only resample."
        )
    if ref_crs is None or ref_transform is None:
        raise ValueError(
            "collocate_by_geocoding: reference_profile is missing "
            "crs/transform -- real geopositioning is required, this "
            "is not a shape-only resample."
        )

    rows, cols = reference_shape
    real_out = np.full((rows, cols), np.nan, dtype=np.float32)
    imag_out = np.full((rows, cols), np.nan, dtype=np.float32)

    # GDAL's warp doesn't resample complex dtypes directly -- reproject
    # real and imaginary components separately (phase-preserving,
    # same split already used elsewhere in this pipeline for resampling
    # complex data). dst_nodata=nan (rather than 0) is deliberate: it's
    # what lets valid_mask distinguish "genuinely reprojected to a
    # small value near zero" from "no source pixel existed here at all".
    reproject(
        source=np.ascontiguousarray(secondary_data.real.astype(np.float32)),
        destination=real_out,
        src_transform=sec_transform,
        src_crs=sec_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=resampling_map[resampling],
        dst_nodata=np.nan,
    )
    reproject(
        source=np.ascontiguousarray(secondary_data.imag.astype(np.float32)),
        destination=imag_out,
        src_transform=sec_transform,
        src_crs=sec_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=resampling_map[resampling],
        dst_nodata=np.nan,
    )

    valid_mask = np.isfinite(real_out) & np.isfinite(imag_out)
    coverage_fraction = float(np.mean(valid_mask))

    if coverage_fraction < 0.05:
        logger.warning(
            "collocate_by_geocoding: only %.1f%% of the reference raster "
            "found real secondary coverage. This almost always means a "
            "genuine coordinate-frame problem (wrong CRS, a transform "
            "that doesn't actually describe this array, or reference/"
            "secondary that don't really overlap on the ground) rather "
            "than the scene genuinely having near-zero overlap -- "
            "verify both profiles' transform/CRS before trusting this "
            "result.",
            coverage_fraction * 100,
        )
    elif coverage_fraction < 0.5:
        logger.info(
            "collocate_by_geocoding: %.1f%% coverage -- partial overlap "
            "(expected if reference/secondary crops only partially "
            "coincide; unexpected if they were meant to cover the same "
            "AOI).",
            coverage_fraction * 100,
        )

    collocated = (
        np.nan_to_num(real_out, nan=0.0) + 1j * np.nan_to_num(imag_out, nan=0.0)
    ).astype(np.complex64)

    return collocated, coverage_fraction, valid_mask


def _regular_grid_points(shape, grid_points: int, margin_frac: float = 0.08):
    """
    A uniformly-spaced (row, col) grid across `shape`, kept
    `margin_frac` away from each edge -- used to seed GCP candidates
    for cross-correlation refinement after raster collocation (where,
    unlike compute_offset_field_from_dem's DEM-driven sampling, there's
    no ground-point-derived grid to start from; the images are already
    nominally on the same raster, so any evenly-spread sample of
    reference pixels is a reasonable set of candidate GCPs).
    """
    rows, cols = shape
    row_lo, row_hi = rows * margin_frac, rows * (1 - margin_frac)
    col_lo, col_hi = cols * margin_frac, cols * (1 - margin_frac)
    row_samples = _linspace(row_lo, row_hi, grid_points)
    col_samples = _linspace(col_lo, col_hi, grid_points)
    return [(r, c) for r in row_samples for c in col_samples]


def compute_offset_field_from_dem(
    ref_geometry,
    ref_orbit,
    sec_geometry,
    sec_orbit,
    dem_path: Union[str, Path],
    ref_scene_center_time,
    sec_scene_center_time,
    grid_points: int = 7,
    sample_bounds: Optional[Tuple[float, float, float, float]] = None,
):
    """
    Compute a real, orbit-based offset field using ground points sampled
    directly from a real DEM's own geographic coordinates — not solved
    for via the unreliable solve_ground_point(). This is the
    recommended path: it uses only components verified reliable
    (geodetic_to_ecef, a closed-form conversion with no iteration, and
    find_zero_doppler_time, confirmed robust even from starting guesses
    tens of seconds off), never the 3-equation Doppler/Range/Ellipsoid
    solve that has a known, unresolved reliability gap.

    Args:
        ref_geometry, sec_geometry: SLCGeometry for each image.
        ref_orbit, sec_orbit: (times, positions, velocities) for each.
        dem_path:      A real DEM covering the scene, with real
                       geographic (lat/lon) referencing — the same DEM
                       already optionally supplied to process_pair()
                       for topographic phase removal.
        ref_scene_center_time, sec_scene_center_time: Approximate
                       acquisition time for each image (e.g. each
                       geometry's azimuth_time() at its own centre row)
                       — used as the starting guess for
                       find_zero_doppler_time(), which only needs to be
                       roughly right (confirmed robust to a 30-second-
                       off starting guess in testing).
        grid_points:   Sample grid resolution per axis.
        sample_bounds: Optional (min_lon, min_lat, max_lon, max_lat) to
                       restrict DEM sampling to the geographic area
                       actually covered by the crop being processed —
                       e.g. the extracted SLC's own real bounds. Without
                       this, grid points are sampled across the DEM's
                       FULL extent regardless of how much of it the
                       current pair's crop actually covers, which for a
                       DEM spanning a much larger region than a single
                       date's crop (confirmed real case: a full
                       administrative-region DEM vs a single sub-swath
                       crop covering a fraction of it) means most points
                       silently fall outside the real SLC extent and get
                       dropped, leaving a small, spatially-clustered set
                       of usable points and a poorly-conditioned fit.

    Returns:
        (grid_rows, grid_cols, offset_rows, offset_cols) — same shape
        as compute_offset_field(), ready for fit_offset_polynomial().

    Raises:
        RuntimeError if too many grid points fail (more than half) —
        a real, actionable signal (e.g. the DEM doesn't actually
        overlap the scene), not silently tolerated.
    """
    import numpy as np
    import rasterio

    from pygeofetch.insar.geolocation import find_zero_doppler_time, geodetic_to_ecef

    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        transform = src.transform
        dem_h, dem_w = dem.shape

        if sample_bounds is not None:
            min_lon, min_lat, max_lon, max_lat = sample_bounds
            # Convert the geographic bounds to DEM pixel row/col bounds,
            # then clamp to the DEM's own real extent.
            inv = ~transform
            c1, r1 = inv * (min_lon, max_lat)
            c2, r2 = inv * (max_lon, min_lat)
            row_lo = max(0, min(int(r1), int(r2)))
            row_hi = min(dem_h - 1, max(int(r1), int(r2)))
            col_lo = max(0, min(int(c1), int(c2)))
            col_hi = min(dem_w - 1, max(int(c1), int(c2)))
            if row_hi <= row_lo or col_hi <= col_lo:
                logger.warning(
                    "compute_offset_field_from_dem: sample_bounds do not "
                    "overlap the DEM extent — falling back to sampling the "
                    "full DEM."
                )
                row_lo, row_hi, col_lo, col_hi = 0, dem_h - 1, 0, dem_w - 1
        else:
            row_lo, row_hi, col_lo, col_hi = 0, dem_h - 1, 0, dem_w - 1

    dem_row_samples = [int(r) for r in _linspace(row_lo, row_hi, grid_points)]
    dem_col_samples = [int(c) for c in _linspace(col_lo, col_hi, grid_points)]

    grid_rows, grid_cols, offset_rows, offset_cols = [], [], [], []
    n_failed = 0
    n_out_of_bounds = 0
    n_total = 0

    for dem_row in dem_row_samples:
        for dem_col in dem_col_samples:
            n_total += 1
            try:
                lon, lat = transform * (dem_col + 0.5, dem_row + 0.5)
                height = float(dem[dem_row, dem_col])
                if not np.isfinite(height):
                    height = 0.0
                ground_point = geodetic_to_ecef(lat, lon, height)

                t_ref = find_zero_doppler_time(
                    ref_orbit[0],
                    ref_orbit[1],
                    ref_orbit[2],
                    ground_point,
                    ref_scene_center_time,
                )
                t_sec = find_zero_doppler_time(
                    sec_orbit[0],
                    sec_orbit[1],
                    sec_orbit[2],
                    ground_point,
                    sec_scene_center_time,
                )

                ref_row = ref_geometry.row_for_azimuth_time(t_ref)
                sec_row = sec_geometry.row_for_azimuth_time(t_sec)

                sat_pos_ref, _ = _interpolate(ref_orbit, t_ref)
                sat_pos_sec, _ = _interpolate(sec_orbit, t_sec)
                range_ref_time = 2 * _distance(sat_pos_ref, ground_point) / 299792458.0
                range_sec_time = 2 * _distance(sat_pos_sec, ground_point) / 299792458.0
                ref_col = ref_geometry.col_for_range_time(range_ref_time)
                sec_col = sec_geometry.col_for_range_time(range_sec_time)

                if not (
                    0 <= ref_row < ref_geometry.n_lines
                    and 0 <= ref_col < ref_geometry.n_columns
                ):
                    n_out_of_bounds += 1
                    continue  # this DEM point falls outside the actual SLC extent

                grid_rows.append(ref_row)
                grid_cols.append(ref_col)
                offset_rows.append(sec_row - ref_row)
                offset_cols.append(sec_col - ref_col)
            except RuntimeError as exc:
                n_failed += 1
                logger.debug(
                    "DEM-driven offset field: point (%d, %d) failed: %s",
                    dem_row,
                    dem_col,
                    exc,
                )

    if n_failed > n_total / 2:
        raise RuntimeError(
            f"DEM-driven offset field: {n_failed}/{n_total} points failed "
            f"— check that the DEM genuinely overlaps the scene and the "
            f"orbit files' validity periods cover both acquisition times."
        )
    if not grid_rows:
        raise RuntimeError(
            "DEM-driven offset field: no valid grid points fell within "
            "the actual SLC extent — check the DEM covers the scene."
        )

    if n_out_of_bounds > 0:
        logger.info(
            "DEM-driven offset field: %d/%d points solved successfully "
            "(%d fell outside the actual SLC extent%s, %d raised a real error)",
            len(grid_rows),
            n_total,
            n_out_of_bounds,
            " -- consider passing sample_bounds" if sample_bounds is None else "",
            n_failed,
        )
    else:
        logger.info(
            "DEM-driven offset field: %d/%d points solved successfully",
            len(grid_rows),
            n_total,
        )
    return grid_rows, grid_cols, offset_rows, offset_cols


def compute_offset_field(
    ref_geometry,
    ref_orbit,
    sec_geometry,
    sec_orbit,
    image_shape: Tuple[int, int],
    dem_height_m: float = 0.0,
    grid_points: int = 7,
):
    """
    Compute a real, orbit-based offset field between a reference and
    secondary SLC image, sampled on a sparse grid and ready for
    polynomial fitting.

    Args:
        ref_geometry, sec_geometry: SLCGeometry from
                       annotation.parse_slc_geometry() for each image.
        ref_orbit, sec_orbit: (times, positions, velocities) from
                       geolocation.parse_orbit_file() for each image.
        image_shape:   (rows, cols) of the reference image.
        dem_height_m:  Constant height above the ellipsoid to use for
                       the geolocation solve (0.0 = pure ellipsoid). A
                       real per-pixel DEM lookup would be more accurate
                       over rugged terrain but isn't wired in here yet.
        grid_points:   Number of sample points per axis (grid_points^2
                       total ground-point solves) — 7 is a reasonable
                       default balancing accuracy and speed; the offset
                       field is smooth, so a sparse grid captures it well.

    Returns:
        (grid_rows, grid_cols, offset_rows, offset_cols): four lists of
        equal length — the sample grid positions in the reference image
        and the real, solved offset (secondary - reference) at each,
        in pixels.

    Raises:
        RuntimeError if too many grid points fail to solve (more than
        half) — a real, actionable signal that the orbit data or
        geometry inputs are likely wrong, surfaced clearly rather than
        silently fitting a polynomial to mostly-garbage points.
    """
    logger.warning(
        "compute_offset_field() uses solve_ground_point(), which has a "
        "known, unresolved reliability gap (see geolocation.py). Prefer "
        "compute_offset_field_from_dem() -- it solves the same problem "
        "using only components independently verified reliable "
        "(geodetic_to_ecef + find_zero_doppler_time) and never calls "
        "solve_ground_point() at all. This function is kept only for "
        "the case where no real DEM is available."
    )
    from pygeofetch.insar.geolocation import find_zero_doppler_time, solve_ground_point

    rows, cols = image_shape
    row_samples = [int(r) for r in _linspace(0, rows - 1, grid_points)]
    col_samples = [int(c) for c in _linspace(0, cols - 1, grid_points)]

    grid_rows, grid_cols, offset_rows, offset_cols = [], [], [], []
    n_failed = 0
    n_total = 0
    last_ground_point = None  # warm-start seed, carried across grid points

    for row in row_samples:
        for col in col_samples:
            n_total += 1
            try:
                t_ref = ref_geometry.azimuth_time(row)
                range_time = ref_geometry.range_time(col)

                sat_pos, sat_vel = _interpolate(ref_orbit, t_ref)
                ground_point = solve_ground_point(
                    sat_pos,
                    sat_vel,
                    range_time,
                    dem_height_m=dem_height_m,
                    initial_guess=last_ground_point,
                )
                last_ground_point = ground_point  # seed the next grid point

                t_sec = find_zero_doppler_time(
                    sec_orbit[0], sec_orbit[1], sec_orbit[2], ground_point, t_ref
                )
                sec_row = sec_geometry.row_for_azimuth_time(t_sec)

                sat_pos_sec, _ = _interpolate(sec_orbit, t_sec)
                range_sec_m = _distance(sat_pos_sec, ground_point)
                range_time_sec = 2 * range_sec_m / 299792458.0
                sec_col = sec_geometry.col_for_range_time(range_time_sec)

                grid_rows.append(row)
                grid_cols.append(col)
                offset_rows.append(sec_row - row)
                offset_cols.append(sec_col - col)
            except RuntimeError as exc:
                n_failed += 1
                logger.debug(
                    "Offset field: grid point (%d, %d) failed: %s", row, col, exc
                )
                # Don't carry a failed point's (non-existent) result forward
                # as the next warm-start seed -- last_ground_point simply
                # stays at its last successful value, which is still a
                # reasonable seed for the next point.

    if n_failed > n_total / 2:
        raise RuntimeError(
            f"Offset field computation: {n_failed}/{n_total} grid points "
            f"failed to solve — this usually means the orbit data doesn't "
            f"actually correspond to this scene, or the annotation timing "
            f"is wrong, not a transient numerical issue. Check that the "
            f"orbit file's validity period actually covers this "
            f"acquisition's real timestamp."
        )
    if n_failed > 0:
        logger.warning(
            "Offset field: %d/%d grid points failed to solve (proceeding "
            "with the remaining %d) — the polynomial fit below is still "
            "valid, but based on fewer points than requested.",
            n_failed,
            n_total,
            n_total - n_failed,
        )

    return grid_rows, grid_cols, offset_rows, offset_cols


def _sample_complex_imagette(data, center_row: float, center_col: float, size: int,
                             phase_ramp=None, row_offset: float = 0.0, col_offset: float = 0.0):
    """
    With ``phase_ramp`` (a TOPS ramp, see resample_with_offset_field), the
    neighbourhood is deramped before the bilinear interpolation and the
    imagette is reramped at the sampled (full-scene) positions, so a
    sub-pixel shift of a TOPS burst keeps its phase; ``row_offset`` and
    ``col_offset`` are `data`'s origin in full-scene coordinates.

    Extract a `size` x `size` complex imagette centered at an
    arbitrary (possibly sub-pixel, possibly out-of-bounds) position via
    bilinear interpolation. Positions outside `data` read as 0+0j
    (scipy's mode="constant"), which naturally suppresses coherence
    for imagettes that fall mostly off the edge of the array rather
    than raising -- the coherence-threshold rejection downstream
    handles those the same way it handles any other low-coherence GCP.
    """
    import numpy as np
    from scipy.ndimage import map_coordinates

    half = size / 2.0
    row_idx, col_idx = np.mgrid[0:size, 0:size].astype(np.float64)
    row_idx = row_idx - half + center_row
    col_idx = col_idx - half + center_col

    if phase_ramp is not None:
        h, w = data.shape
        r0 = max(0, int(np.floor(row_idx.min())) - 1)
        r1 = min(h, int(np.ceil(row_idx.max())) + 2)
        c0 = max(0, int(np.floor(col_idx.min())) - 1)
        c1 = min(w, int(np.ceil(col_idx.max())) + 2)
        if r1 <= r0 or c1 <= c0:
            return np.zeros((size, size), dtype=np.complex64)
        block = data[r0:r1, c0:c1].astype(np.complex128)
        rows_g = np.arange(r0, r1, dtype=np.float64)[:, None] + row_offset
        cols_g = np.arange(c0, c1, dtype=np.float64)[None, :] + col_offset
        block = block * np.exp(-1j * phase_ramp(rows_g, cols_g))
        coords = [row_idx - r0, col_idx - c0]
        out = map_coordinates(block.real, coords, order=1, mode="constant", cval=0.0) + 1j * map_coordinates(
            block.imag, coords, order=1, mode="constant", cval=0.0
        )
        out = out * np.exp(1j * phase_ramp(row_idx + row_offset, col_idx + col_offset))
        return out.astype(np.complex64)

    real = map_coordinates(
        data.real, [row_idx, col_idx], order=1, mode="constant", cval=0.0
    )
    imag = map_coordinates(
        data.imag, [row_idx, col_idx], order=1, mode="constant", cval=0.0
    )
    return (real + 1j * imag).astype(np.complex64)


def _imagette_coherence(ref_imagette, sec_imagette) -> float:
    """
    Single scalar complex coherence between two same-shape imagettes —
    SNAP CrossCorrelationOp's "Method 1" (no internal sliding window;
    the whole imagette is treated as one coherence estimation window):
    https://step.esa.int/main/wp-content/help/versions/13.0.0/snap-toolboxes/eu.esa.microwavetbx.sar.op.insar.ui/operators/CrossCorrelationOp.html

    Deliberately NOT a windowed (uniform_filter) estimate averaged
    across the imagette. Measured directly, not assumed: with a small
    window (e.g. 5x5 = 25 samples), the maximum-likelihood coherence
    estimator carries substantial positive bias at low true coherence
    (Touzi, Lopes, Bruniquel & Vachon 1999) — empirically, true
    coherence 0.0 reads back ~0.19, true 0.3 reads back ~0.34. Since
    this value both gates coherence_threshold and is what the Powell
    fine-search below maximizes, that bias isn't just a cosmetic
    reporting issue: it lets genuinely poor GCPs pass the threshold and
    can pull the search toward a locally-higher-but-not-truly-better
    optimum. A single scalar over the WHOLE imagette (~1024 samples for
    the default 32px window) is close to unbiased at any true
    coherence, verified the same way (true 0.0 -> ~0.03, true 0.3 ->
    ~0.31).
    """
    import numpy as np

    num = abs(np.mean(ref_imagette * np.conj(sec_imagette)))
    denom = np.sqrt(
        np.mean(np.abs(ref_imagette) ** 2) * np.mean(np.abs(sec_imagette) ** 2)
    )
    if denom < 1e-12:
        return 0.0
    return float(np.clip(num / denom, 0.0, 1.0))


def refine_offsets_by_coherence(
    ref_complex,
    sec_complex,
    grid_rows,
    grid_cols,
    offset_rows,
    offset_cols,
    ref_row_offset: float = 0.0,
    ref_col_offset: float = 0.0,
    sec_row_offset: float = 0.0,
    sec_col_offset: float = 0.0,
    window: int = 32,
    coarse_search_radius: int = 4,
    fine_search_radius: float = 1.5,
    coherence_threshold: float = 0.3,
    ref_phase_ramp=None,
    sec_phase_ramp=None,
):
    """
    Refine an orbit/DEM-derived offset field against the actual image
    content, via the same two-stage process SNAP's CrossCorrelationOp
    uses for GCP selection: coarse integer-pixel cross-correlation of
    imagettes, then sub-pixel coherence maximization (Powell's method)
    for complex data:
    https://step.esa.int/main/wp-content/help/versions/13.0.0/snap-toolboxes/eu.esa.microwavetbx.sar.op.insar.ui/operators/CrossCorrelationOp.html

    Why this matters: the orbit/DEM path (compute_offset_field_from_dem)
    gives a smooth, physically-grounded FIRST ESTIMATE of the offset
    field, but it is only ever as accurate as the orbit state vectors,
    acquisition timing metadata, and DEM height/vertical datum feeding
    it -- typically good to within a pixel or so, not the sub-pixel
    accuracy interferometry needs. Coherence is a measure of how well
    two complex samples at the SAME ground point agree; even a
    fraction of a pixel of true residual misregistration decorrelates
    it substantially. This step is what actually closes that gap, by
    searching directly against the image data itself rather than
    trusting the orbit/DEM model alone.

    Args:
        ref_complex, sec_complex: the actual complex SLC arrays as
                       loaded (i.e. possibly AOI-cropped, NOT
                       necessarily full-scene).
        grid_rows, grid_cols, offset_rows, offset_cols: the initial
                       offset field in FULL-SCENE reference
                       coordinates, e.g. from
                       compute_offset_field_from_dem().
        ref_row_offset, ref_col_offset, sec_row_offset, sec_col_offset:
                       each array's local-crop origin in full-scene
                       coordinates (from read_crop_offset()); 0.0 if
                       uncropped. Needed to map the full-scene grid
                       onto each array's own local pixel indices.
        window:        Imagette size in pixels (SNAP's "Registration
                       Window"). Must be well within both arrays'
                       dimensions; 32 balances localization accuracy
                       against having enough independent samples for a
                       stable coherence estimate.
        coarse_search_radius: Integer-pixel half-width searched in the
                       coarse cross-correlation stage.
        fine_search_radius: Sub-pixel half-width (around the coarse
                       result) searched by the Powell coherence-
                       maximization stage.
        coherence_threshold: GCPs whose refined coherence is below
                       this are dropped as unreliable, rather than fed
                       into the polynomial fit -- SNAP's own
                       "Coherence Threshold" parameter.

    Returns:
        (grid_rows, grid_cols, offset_rows, offset_cols, coherences) --
        refined in place for GCPs that had a usable imagette pair and
        met the coherence threshold; GCPs that didn't (fell too close
        to an array edge, or never reached threshold coherence even
        after refinement) are dropped from all five returned lists.

    Raises:
        RuntimeError if every GCP is dropped -- a real signal that
        `ref_complex`/`sec_complex` and the supplied offset field
        don't actually correspond (e.g. wrong crop offsets passed),
        not something to silently paper over by returning an empty,
        useless refined field.
    """
    import numpy as np
    from scipy.optimize import minimize

    if window % 2 != 0:
        window += 1  # even size keeps the imagette centering symmetric

    ref_h, ref_w = ref_complex.shape
    sec_h, sec_w = sec_complex.shape
    half = window / 2.0
    # An imagette needs to be substantially inside the array to give a
    # meaningful (not mostly zero-padded) coherence estimate -- require
    # the nominal center to be at least half a window from every edge,
    # PLUS enough room for the Powell fine-search stage to still sample
    # fully in-bounds imagettes up to fine_search_radius beyond the
    # coarse result. Omitting fine_search_radius here lets edge-adjacent
    # GCPs pass this check and then have the fine-search silently read
    # zero-padded (out-of-bounds) data during refinement.
    margin = half + coarse_search_radius + fine_search_radius

    out_rows, out_cols, out_orow, out_ocol, out_coh = [], [], [], [], []
    n_total = len(grid_rows)
    n_edge_dropped = 0
    n_low_coherence = 0

    for row, col, orow, ocol in zip(grid_rows, grid_cols, offset_rows, offset_cols):
        ref_local_row = row - ref_row_offset
        ref_local_col = col - ref_col_offset
        sec_local_row = row + orow - sec_row_offset
        sec_local_col = col + ocol - sec_col_offset

        if not (
            margin <= ref_local_row <= ref_h - margin
            and margin <= ref_local_col <= ref_w - margin
            and margin <= sec_local_row <= sec_h - margin
            and margin <= sec_local_col <= sec_w - margin
        ):
            n_edge_dropped += 1
            continue

        ref_imagette = _sample_complex_imagette(
            ref_complex, ref_local_row, ref_local_col, window,
            phase_ramp=ref_phase_ramp, row_offset=ref_row_offset, col_offset=ref_col_offset,
        )

        # --- Coarse stage: integer-pixel cross-correlation search ---
        best_dr, best_dc, best_coh = 0, 0, -1.0
        for dr in range(-coarse_search_radius, coarse_search_radius + 1):
            for dc in range(-coarse_search_radius, coarse_search_radius + 1):
                sec_imagette = _sample_complex_imagette(
                    sec_complex, sec_local_row + dr, sec_local_col + dc, window,
                    phase_ramp=sec_phase_ramp, row_offset=sec_row_offset, col_offset=sec_col_offset,
                )
                coh = _imagette_coherence(ref_imagette, sec_imagette)
                if coh > best_coh:
                    best_dr, best_dc, best_coh = dr, dc, coh

        # --- Fine stage: sub-pixel coherence maximization (Powell) ---
        def _neg_coherence(
            shift, _row0=sec_local_row + best_dr, _col0=sec_local_col + best_dc
        ):
            sec_imagette = _sample_complex_imagette(
                sec_complex, _row0 + shift[0], _col0 + shift[1], window,
                phase_ramp=sec_phase_ramp, row_offset=sec_row_offset, col_offset=sec_col_offset,
            )
            return -_imagette_coherence(ref_imagette, sec_imagette)

        result = minimize(
            _neg_coherence,
            x0=np.array([0.0, 0.0]),
            method="Powell",
            bounds=[(-fine_search_radius, fine_search_radius)] * 2,
        )
        refined_dr = best_dr + float(result.x[0])
        refined_dc = best_dc + float(result.x[1])
        final_coh = float(-result.fun)

        if final_coh < coherence_threshold:
            n_low_coherence += 1
            continue

        out_rows.append(row)
        out_cols.append(col)
        out_orow.append(orow + refined_dr)
        out_ocol.append(ocol + refined_dc)
        out_coh.append(final_coh)

    if not out_rows:
        raise RuntimeError(
            f"refine_offsets_by_coherence: all {n_total} GCPs were "
            f"dropped ({n_edge_dropped} too close to an array edge, "
            f"{n_low_coherence} below coherence_threshold={coherence_threshold}). "
            f"This usually means ref_complex/sec_complex don't actually "
            f"correspond to the supplied offset field/crop offsets, or "
            f"window is too large for the array size -- not that the "
            f"scene genuinely has zero coherence everywhere."
        )

    if n_edge_dropped or n_low_coherence:
        logger.info(
            "Cross-correlation refinement: %d/%d GCPs refined "
            "(%d dropped: too close to an edge; %d dropped: below "
            "coherence threshold %.2f); mean coherence %.3f",
            len(out_rows),
            n_total,
            n_edge_dropped,
            n_low_coherence,
            coherence_threshold,
            float(np.mean(out_coh)),
        )
    else:
        logger.info(
            "Cross-correlation refinement: all %d GCPs refined, mean " "coherence %.3f",
            len(out_rows),
            float(np.mean(out_coh)),
        )

    return out_rows, out_cols, out_orow, out_ocol, out_coh


def _poly_term_exponents(degree: int):
    """(i, j) exponent pairs for every monomial row^i * col^j with
    i + j <= degree, ordered by total degree then i — a fixed,
    deterministic ordering so the design matrix and its evaluation
    always agree. degree=1 -> [(0,0),(1,0),(0,1)]; degree=2 adds
    (1,1),(2,0),(0,2); degree=3 (SNAP's max supported order) adds
    (2,1),(1,2),(3,0),(0,3)."""
    return [(i, total - i) for total in range(degree + 1) for i in range(total + 1)]


def fit_offset_polynomial(grid_rows, grid_cols, offsets, degree: int = 1):
    """
    Fit a low-degree 2D polynomial to a sparse offset grid — standard
    practice (Kampes, Hanssen & Perski 2003; matches the 2019 URSI
    paper's approach, and the same degree-1/2/3 range SNAP's WarpOp
    supports) rather than every-pixel solving, since the true offset
    field varies smoothly across a scene.

    Degree 1 (linear) is recommended for most cases. Higher orders
    (2, 3) should only be used when the image suffers a high level of
    distortion (e.g. TOPS data or a large perpendicular baseline) AND
    there are enough well-distributed GCPs to support them without
    over-fitting — a degree-3 fit needs at least 10 GCPs to be
    non-degenerate and considerably more than that to be well-
    conditioned; prefer fit_offset_polynomial_robust() over calling
    this directly, since it enforces a minimum point count per degree
    and rejects outliers before fitting.

    Returns a callable f(row, col) -> offset, evaluable at any pixel.
    """
    import numpy as np

    if degree not in (1, 2, 3):
        raise ValueError(f"Unsupported polynomial degree: {degree} (use 1, 2, or 3)")

    rows = np.asarray(grid_rows, dtype=np.float64)
    cols = np.asarray(grid_cols, dtype=np.float64)
    vals = np.asarray(offsets, dtype=np.float64)

    exponents = _poly_term_exponents(degree)
    n_terms = len(exponents)
    if rows.size < n_terms:
        raise ValueError(
            f"fit_offset_polynomial: degree {degree} needs at least "
            f"{n_terms} GCPs, got {rows.size}. Use a lower degree, a "
            f"denser grid, or fit_offset_polynomial_robust() which "
            f"enforces this before attempting the fit."
        )

    A = np.column_stack([rows**i * cols**j for (i, j) in exponents])
    coeffs, *_ = np.linalg.lstsq(A, vals, rcond=None)

    def evaluate(row, col):
        row = np.asarray(row, dtype=np.float64)
        col = np.asarray(col, dtype=np.float64)
        result = np.zeros_like(row + col, dtype=np.float64)
        for c, (i, j) in zip(coeffs, exponents):
            result = result + c * row**i * col**j
        return result

    return evaluate


def fit_offset_polynomial_robust(
    grid_rows,
    grid_cols,
    offset_rows,
    offset_cols,
    degree: int = 1,
    max_iterations: int = 2,
    rms_threshold: Optional[float] = None,
    outlier_rule: str = "mad",
    mad_k: float = 3.0,
    mad_floor_px: float = 0.05,
):
    """
    Fit row and column offset polynomials with SNAP WarpOp-style
    iterative GCP outlier rejection, rather than a single unweighted
    least-squares pass:

      1. Fit a degree-`degree` polynomial to the current GCP set.
      2. Map each GCP's (row, col) through the fit and compute its
         residual (RMS of the row- and column-offset error).
      3. Eliminate GCPs whose RMS exceeds the *mean* RMS.
      4. Repeat steps 1-3 up to `max_iterations` times (SNAP's default
         is 2), using only the surviving GCPs each time.
      5. If `rms_threshold` is given, do one final absolute-threshold
         filter (SNAP's user-set "RMS Threshold") and refit.

    A few bad offset estimates — from a low-coherence patch, a DEM
    artifact, or an orbit-solve glitch at one grid point — can
    otherwise corrupt the whole polynomial, since ordinary least
    squares has no protection against outliers pulling the fit toward
    them. This directly matches the "residual file" concept from
    SNAP's WarpOp documentation:
    https://step.esa.int/main/wp-content/help/versions/13.0.0/snap-toolboxes/eu.esa.microwavetbx.sar.op.insar.ui/operators/WarpOp.html

    Args:
        grid_rows, grid_cols: GCP positions (reference image coords).
        offset_rows, offset_cols: (secondary - reference) offsets at
                       each GCP, in pixels.
        degree:        Warp polynomial order (1, 2, or 3).
        max_iterations: Number of mean-RMS filtering passes (SNAP: 2).
        outlier_rule:  "mad" (default): reject GCPs whose RMS exceeds
                       median + mad_k * 1.4826 * MAD (scale floored at
                       mad_floor_px), and never reduce the set below the
                       minimum the degree needs. "mean": SNAP WarpOp's
                       rule (reject every GCP above the mean RMS). The mean
                       rule removes roughly half the points on every pass
                       even when all are good, which on a sparse
                       low-coherence GCP set leaves 3 points and a fit that
                       passes exactly through their noise.
        rms_threshold: Optional final absolute RMS cutoff, in pixels.
                       None (default) skips this and relies only on
                       the iterative mean-RMS filtering above, which
                       is self-calibrating and doesn't require knowing
                       a sensible absolute threshold in advance.

    Returns:
        (row_fn, col_fn, quality): the two fitted polynomials (each
        evaluable as f(row, col) -> offset, same as
        fit_offset_polynomial()'s return) plus a CoregistrationQuality
        with the final residual statistics.

    Raises:
        RuntimeError if outlier rejection would leave fewer GCPs than
        the chosen degree needs — a real, actionable signal (the
        initial offset field is unreliable, or the degree is too high
        for how many GCPs are available), not silently tolerated by
        fitting a degenerate or wildly underdetermined polynomial.
    """
    import numpy as np

    if degree not in (1, 2, 3):
        raise ValueError(f"Unsupported polynomial degree: {degree} (use 1, 2, or 3)")

    rows = np.asarray(grid_rows, dtype=np.float64)
    cols = np.asarray(grid_cols, dtype=np.float64)
    orow = np.asarray(offset_rows, dtype=np.float64)
    ocol = np.asarray(offset_cols, dtype=np.float64)
    n_initial = rows.size
    min_gcps = _min_points_for_degree(degree)

    if n_initial < min_gcps:
        raise RuntimeError(
            f"fit_offset_polynomial_robust: only {n_initial} GCPs "
            f"available, but a degree-{degree} fit needs at least "
            f"{min_gcps}. Use a lower degree, a denser initial grid, "
            f"or check upstream offset-field computation for a "
            f"systematic failure rather than isolated bad points."
        )

    keep = np.ones(n_initial, dtype=bool)
    row_fn = col_fn = None
    resid_row = resid_col = rms = np.empty(0)
    iterations_used = 0

    for iteration in range(max_iterations + 1):
        idx = np.nonzero(keep)[0]
        if idx.size < min_gcps:
            raise RuntimeError(
                f"fit_offset_polynomial_robust: outlier rejection left "
                f"only {idx.size} GCPs after {iteration} iteration(s), "
                f"below the {min_gcps} a degree-{degree} fit needs. "
                f"This means the offset field has more disagreement "
                f"than isolated noise -- try degree=1, inspect the "
                f"pre-filtering residuals, or check that the reference/"
                f"secondary geolocation inputs are actually consistent."
            )

        row_fn = fit_offset_polynomial(rows[idx], cols[idx], orow[idx], degree=degree)
        col_fn = fit_offset_polynomial(rows[idx], cols[idx], ocol[idx], degree=degree)

        resid_row = row_fn(rows[idx], cols[idx]) - orow[idx]
        resid_col = col_fn(rows[idx], cols[idx]) - ocol[idx]
        rms = np.sqrt(resid_row**2 + resid_col**2)
        iterations_used = iteration

        if iteration == max_iterations:
            break

        if outlier_rule == "mean":
            is_outlier = rms > float(np.mean(rms))
        elif outlier_rule == "mad":
            med = float(np.median(rms))
            scale = max(1.4826 * float(np.median(np.abs(rms - med))), mad_floor_px)
            is_outlier = rms > med + mad_k * scale
            # never reject below what the degree needs: drop the worst first
            if rms.size - int(np.count_nonzero(is_outlier)) < min_gcps:
                n_drop = max(rms.size - min_gcps, 0)
                worst = np.argsort(rms)[::-1][:n_drop]
                cand = np.zeros_like(is_outlier)
                cand[worst] = True
                is_outlier = cand & is_outlier
        else:
            raise ValueError(f"outlier_rule must be 'mad' or 'mean', got {outlier_rule!r}")
        if not np.any(is_outlier):
            break  # converged -- nothing left to reject

        keep = np.zeros(n_initial, dtype=bool)
        keep[idx[~is_outlier]] = True

    # Step 5: optional final absolute-threshold filter + refit.
    if rms_threshold is not None and np.any(rms > rms_threshold):
        idx = np.nonzero(keep)[0]
        survivors_local = rms <= rms_threshold
        if survivors_local.sum() >= min_gcps:
            keep = np.zeros(n_initial, dtype=bool)
            keep[idx[survivors_local]] = True
            idx = np.nonzero(keep)[0]
            row_fn = fit_offset_polynomial(
                rows[idx], cols[idx], orow[idx], degree=degree
            )
            col_fn = fit_offset_polynomial(
                rows[idx], cols[idx], ocol[idx], degree=degree
            )
            resid_row = row_fn(rows[idx], cols[idx]) - orow[idx]
            resid_col = col_fn(rows[idx], cols[idx]) - ocol[idx]
            rms = np.sqrt(resid_row**2 + resid_col**2)
        else:
            logger.warning(
                "fit_offset_polynomial_robust: rms_threshold=%.3f would "
                "leave only %d/%d GCPs (below the %d a degree-%d fit "
                "needs) -- skipping the absolute-threshold filter and "
                "keeping the iterative-mean-RMS result instead.",
                rms_threshold,
                int(survivors_local.sum()),
                idx.size,
                min_gcps,
                degree,
            )

    n_final = int(np.count_nonzero(keep))
    quality = CoregistrationQuality(
        n_gcps_initial=n_initial,
        n_gcps_final=n_final,
        n_gcps_rejected=n_initial - n_final,
        rms_mean=float(np.mean(rms)),
        rms_std=float(np.std(rms)),
        row_residual_mean=float(np.mean(resid_row)),
        row_residual_std=float(np.std(resid_row)),
        col_residual_mean=float(np.mean(resid_col)),
        col_residual_std=float(np.std(resid_col)),
        per_point_rms=rms.tolist(),
        iterations_used=iterations_used,
        degree=degree,
    )
    quality.log_summary(logging.DEBUG)   # the caller logs the one summary line that includes the coherence
    return row_fn, col_fn, quality


def fit_geometry_first_offsets(
    geom_rows,
    geom_cols,
    geom_offset_rows,
    geom_offset_cols,
    xc_rows,
    xc_cols,
    xc_offset_rows,
    xc_offset_cols,
    degree: int = 1,
    min_points_linear: int = 8,
    min_points_constant: int = 2,
    mad_k: float = 3.0,
    mad_floor_px: float = 0.05,
    max_iterations: int = 2,
):
    """
    Build the final offset field from the orbit/DEM geometry, corrected by
    image content only as far as the image content can support.

    The orbit/DEM field is dense (tens of points) and physically grounded
    but can carry a bias from orbit timing or DEM datum errors. Cross-
    correlation offsets are image-based but sparse and noisy at low
    coherence (at gamma 0.3 only a handful of windows pass the threshold).
    Fitting a plane through 3-5 of them, as a free polynomial, gives a
    field that passes through their noise and extrapolates it across the
    scene. Instead:

      1. Fit the geometry points with a degree-`degree` polynomial (g).
      2. Take d = xcorr offset - g at the cross-correlation points.
      3. Reject outliers in d with median + k*MAD (at most `max_iterations`
         passes, never below `min_points_constant`).
      4. Correct g by a model only as rich as the surviving points support:
           >= min_points_linear points -> degree-1 correction,
           >= min_points_constant      -> constant (median) correction,
           otherwise                  -> none (geometry only).

    Returns (row_fn, col_fn, quality): evaluable f(row, col) -> offset, and
    a CoregistrationQuality with .mode and .n_gcps_final = number of
    cross-correlation points that support the correction.
    """
    import numpy as np

    gr = np.asarray(geom_rows, dtype=np.float64)
    gc = np.asarray(geom_cols, dtype=np.float64)
    gor = np.asarray(geom_offset_rows, dtype=np.float64)
    goc = np.asarray(geom_offset_cols, dtype=np.float64)
    xr = np.asarray(xc_rows, dtype=np.float64)
    xcl = np.asarray(xc_cols, dtype=np.float64)
    xor_ = np.asarray(xc_offset_rows, dtype=np.float64)
    xoc = np.asarray(xc_offset_cols, dtype=np.float64)

    deg = degree
    while deg > 1 and gr.size < _min_points_for_degree(deg):
        deg -= 1
    if gr.size < _min_points_for_degree(deg):
        raise RuntimeError(
            f"fit_geometry_first_offsets: only {gr.size} orbit/DEM points; "
            f"a degree-{deg} geometry fit needs {_min_points_for_degree(deg)}."
        )
    g_row = fit_offset_polynomial(gr, gc, gor, degree=deg)
    g_col = fit_offset_polynomial(gr, gc, goc, degree=deg)

    n_xc = int(xr.size)
    dr = np.empty(0)
    dc = np.empty(0)
    keep = np.zeros(n_xc, dtype=bool)
    iterations_used = 0
    if n_xc:
        dr = xor_ - g_row(xr, xcl)
        dc = xoc - g_col(xr, xcl)
        keep = np.ones(n_xc, dtype=bool)
        for it in range(max_iterations):
            idx = np.nonzero(keep)[0]
            if idx.size <= min_points_constant:
                break
            mr, mc = float(np.median(dr[idx])), float(np.median(dc[idx]))
            dist = np.hypot(dr[idx] - mr, dc[idx] - mc)
            med = float(np.median(dist))
            scale = max(1.4826 * float(np.median(np.abs(dist - med))), mad_floor_px)
            bad = dist > med + mad_k * scale
            if not np.any(bad) or idx.size - int(bad.sum()) < min_points_constant:
                break
            keep[idx[bad]] = False
            iterations_used = it + 1

    idx = np.nonzero(keep)[0]
    n_support = int(idx.size)

    def _zero(r, c):
        return np.zeros_like(np.asarray(r, dtype=np.float64) + np.asarray(c, dtype=np.float64))

    corr_row = corr_col = _zero
    mode = "geometry_only"
    if n_support >= min_points_linear:
        try:
            corr_row = fit_offset_polynomial(xr[idx], xcl[idx], dr[idx], degree=1)
            corr_col = fit_offset_polynomial(xr[idx], xcl[idx], dc[idx], degree=1)
            mode = "geometry+linear"
        except Exception:
            corr_row = corr_col = _zero
    if mode == "geometry_only" and n_support >= min_points_constant:
        cr, cc_ = float(np.median(dr[idx])), float(np.median(dc[idx]))

        def corr_row(r, c, _v=cr):
            return np.full_like(np.asarray(r, dtype=np.float64) + np.asarray(c, dtype=np.float64), _v)

        def corr_col(r, c, _v=cc_):
            return np.full_like(np.asarray(r, dtype=np.float64) + np.asarray(c, dtype=np.float64), _v)

        mode = "geometry+constant"

    def row_fn(r, c):
        return g_row(r, c) + corr_row(r, c)

    def col_fn(r, c):
        return g_col(r, c) + corr_col(r, c)

    if n_support:
        er = row_fn(xr[idx], xcl[idx]) - xor_[idx]
        ec = col_fn(xr[idx], xcl[idx]) - xoc[idx]
        rms = np.sqrt(er**2 + ec**2)
        quality = CoregistrationQuality(
            n_gcps_initial=n_xc, n_gcps_final=n_support, n_gcps_rejected=n_xc - n_support,
            rms_mean=float(np.mean(rms)), rms_std=float(np.std(rms)),
            row_residual_mean=float(np.mean(er)), row_residual_std=float(np.std(er)),
            col_residual_mean=float(np.mean(ec)), col_residual_std=float(np.std(ec)),
            per_point_rms=rms.tolist(), iterations_used=iterations_used, degree=deg,
            mode=mode, n_geometry_points=int(gr.size))
    else:
        quality = CoregistrationQuality(
            n_gcps_initial=n_xc, n_gcps_final=0, n_gcps_rejected=n_xc,
            rms_mean=0.0, rms_std=0.0, row_residual_mean=0.0, row_residual_std=0.0,
            col_residual_mean=0.0, col_residual_std=0.0, per_point_rms=[],
            iterations_used=iterations_used, degree=deg, mode=mode,
            n_geometry_points=int(gr.size))
    logger.info(
        "Geometry-first coregistration: %s from %d orbit/DEM points (degree-%d); %d/%d "
        "cross-correlation points support the correction%s",
        mode, gr.size, deg, n_support, n_xc,
        f", residual RMS {quality.rms_mean:.3f} px" if n_support else " (image content did not constrain it)")
    return row_fn, col_fn, quality


def resample_with_offset_field(
    data,
    row_offset_fn,
    col_offset_fn,
    ref_row_offset: float = 0.0,
    ref_col_offset: float = 0.0,
    sec_row_offset: float = 0.0,
    sec_col_offset: float = 0.0,
    phase_ramp=None,
    output_shape=None,
):
    """
    Resample a complex array using a real, per-pixel offset field
    (from fit_offset_polynomial), rather than a naive shape-only zoom.

    Args:
        data:           Complex array to resample (the secondary SLC).
        row_offset_fn, col_offset_fn: Callables from fit_offset_polynomial(),
                       giving the real (secondary - reference) offset
                       at any (row, col) expressed in REFERENCE
                       full-scene coordinates (that's what
                       compute_offset_field_from_dem() fits them on).
        ref_row_offset, ref_col_offset: The reference array's origin in
                       full-scene coordinates (from read_crop_offset()
                       on the reference file, if cropped) -- needed to
                       correctly evaluate the offset functions, since
                       they were fit on full-scene coordinates, not the
                       reference crop's local 0-based ones.
        sec_row_offset, sec_col_offset: The secondary array's (this
                       function's `data`) origin in full-scene
                       coordinates -- needed to convert the computed
                       full-scene secondary sample position back into
                       `data`'s own local coordinates for indexing,
                       since the secondary crop can have a different
                       offset than the reference crop.
                       All four default to 0 for already-full-scene
                       (uncropped) data.
        phase_ramp:     Optional callable ``phase_ramp(rows, cols)`` giving
                       the azimuth phase ramp (rad) that `data` carries at
                       full-scene secondary positions -- for Sentinel-1
                       TOPS bursts, :meth:`pygeofetch.insar.tops.TopsRamp.phase`.
                       When given, `data` is deramped (multiplied by
                       exp(-i ramp)) before interpolation and the result is
                       reramped with the ramp at the positions sampled. A
                       TOPS burst's Doppler centroid sweeps by ~5 kHz, ten
                       times the azimuth sampling rate, so interpolating it
                       without deramping corrupts its phase (fringes along
                       azimuth that jump at burst edges) and its coherence.
                       `data` itself is not modified.
        output_shape:   (rows, cols) of the REFERENCE crop, i.e. of the
                       grid to resample onto. Defaults to `data.shape`,
                       which is only right when both crops have the same
                       size: the output rows/cols are reference-local
                       coordinates. With crops of different sizes the old
                       default produced an array of the secondary's size,
                       which the caller then zoomed to the reference's
                       size -- stretching the secondary along azimuth.

    Returns:
        The resampled complex array, on the reference image's grid.
    """
    import numpy as np
    from scipy.ndimage import map_coordinates

    # h, w = data.shape
    # out_real = np.empty((h, w), dtype=np.float32)
    # out_imag = np.empty((h, w), dtype=np.float32)
    # data_real = data.real
    # data_imag = data.imag

    # Process in row chunks rather than building ~10 full-resolution
    # float64 intermediate arrays (row_idx, col_idx, ref_global_row/col,
    # offset_row/col, sec_global_row/col, sample_rows/cols) all at once.
    # Confirmed real: at float64 this scales to several GB for large
    # crops (a 5000x15000 crop needs ~6GB in intermediates alone), which
    # is a genuine, reproducible OOM risk, not a theoretical one -- the
    # crash this was built to fix happened silently (bypassing this
    # function's own try/except in the caller, meaning it was a hard
    # memory/native failure, not a normal Python exception). float32 is
    # more than sufficient precision for sub-pixel offset resampling and
    # halves the footprint on its own; chunking bounds peak memory to
    # one row-block regardless of total crop size.

    h, w = (int(output_shape[0]), int(output_shape[1])) if output_shape is not None else data.shape
    # One complex64 output filled tile by tile. The previous version built two float32 planes and then
    # (out_real + 1j * out_imag).astype(complex64), which held about four extra full-size arrays at the end.
    out = np.empty((h, w), dtype=np.complex64)
    if phase_ramp is not None:
        # Deramp a copy of the secondary on its own grid (baseband), in row chunks.
        work = np.array(data, dtype=np.complex64, copy=True)
        sec_cols_global = np.arange(work.shape[1], dtype=np.float64)[None, :] + sec_col_offset
        for r0 in range(0, work.shape[0], 512):
            r1 = min(work.shape[0], r0 + 512)
            rows_global = np.arange(r0, r1, dtype=np.float64)[:, None] + sec_row_offset
            work[r0:r1] *= np.exp(-1j * phase_ramp(rows_global, sec_cols_global)).astype(np.complex64)
        data = work
    data_real = data.real
    data_imag = data.imag

    # Process in 2D tiles for cache locality and strict memory bounding
    tile_h, tile_w = 1024, 1024

    for r_start in range(0, h, tile_h):
        r_end = min(r_start + tile_h, h)
        for c_start in range(0, w, tile_w):
            c_end = min(c_start + tile_w, w)

            # Use ogrid to create broadcastable 1D arrays instead of full 2D mgrid matrices
            row_idx = np.arange(r_start, r_end, dtype=np.float32)[:, None]
            col_idx = np.arange(c_start, c_end, dtype=np.float32)[None, :]

            ref_global_row = row_idx + ref_row_offset
            ref_global_col = col_idx + ref_col_offset

            # Evaluate polynomial (broadcasts automatically, minimal memory)
            offset_row = row_offset_fn(ref_global_row, ref_global_col)
            offset_col = col_offset_fn(ref_global_row, ref_global_col)

            sample_rows = ref_global_row + offset_row - sec_row_offset
            sample_cols = ref_global_col + offset_col - sec_col_offset

            # Broadcast to 2D for map_coordinates
            sample_rows_2d = np.broadcast_to(
                sample_rows, (r_end - r_start, c_end - c_start)
            )
            sample_cols_2d = np.broadcast_to(
                sample_cols, (r_end - r_start, c_end - c_start)
            )

            tile = out[r_start:r_end, c_start:c_end]
            tile.real = map_coordinates(
                data_real,
                [sample_rows_2d, sample_cols_2d],
                order=1,
                mode="constant",
                cval=0.0,
            )
            tile.imag = map_coordinates(
                data_imag,
                [sample_rows_2d, sample_cols_2d],
                order=1,
                mode="constant",
                cval=0.0,
            )
            if phase_ramp is not None:
                tile *= np.exp(
                    1j * phase_ramp(sample_rows_2d + sec_row_offset, sample_cols_2d + sec_col_offset)
                ).astype(np.complex64)

            del sample_rows_2d, sample_cols_2d, offset_row, offset_col

    return out
    # chunk_rows = max(1, min(h, 2000))
    # for row_start in range(0, h, chunk_rows):
    #     row_end = min(row_start + chunk_rows, h)
    #     row_idx, col_idx = np.mgrid[row_start:row_end, 0:w].astype(np.float32)

    #     # row_idx/col_idx are the reference grid's local (0-based)
    #     # coordinates. Convert to full-scene coordinates to correctly
    #     # evaluate the fitted offset functions, which were built on real
    #     # annotation-derived full-scene coordinates, not any particular
    #     # crop's local ones.
    #     ref_global_row = row_idx + ref_row_offset
    #     ref_global_col = col_idx + ref_col_offset

    #     offset_row = row_offset_fn(ref_global_row, ref_global_col)
    #     offset_col = col_offset_fn(ref_global_row, ref_global_col)

    #     # Where this pixel's secondary counterpart is, in full-scene coordinates
    #     sec_global_row = ref_global_row + offset_row
    #     sec_global_col = ref_global_col + offset_col

    #     # Convert to the SECONDARY array's own local coordinates for
    #     # actually indexing into `data` -- the secondary crop can have a
    #     # different offset than the reference crop, so this is not the
    #     # same subtraction as ref_row_offset/ref_col_offset above.
    #     sample_rows = sec_global_row - sec_row_offset
    #     sample_cols = sec_global_col - sec_col_offset

    #     out_real[row_start:row_end] = map_coordinates(
    #         data_real, [sample_rows, sample_cols], order=1, mode="constant", cval=0.0
    #     )
    #     out_imag[row_start:row_end] = map_coordinates(
    #         data_imag, [sample_rows, sample_cols], order=1, mode="constant", cval=0.0
    #     )

    # return (out_real + 1j * out_imag).astype(np.complex64)


def _interpolate(orbit, t):
    from pygeofetch.insar.geolocation import interpolate_orbit_state

    return interpolate_orbit_state(orbit[0], orbit[1], orbit[2], t)


def _distance(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def _linspace(start, stop, n):
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]

class BurstTimingMap:
    """Exact azimuth row mapping between two TOPS SLCs in their bursted (TIFF) layout.

    In the measurement TIFF of an IW/EW SLC, burst ``b`` occupies rows ``b*L ... (b+1)*L-1`` and starts at its own
    annotated ``azimuthTime``. The secondary row that sees the same ground as reference row ``R`` is therefore a
    *staircase* in ``R``: within a burst it follows the (smooth) geometric time difference, but from burst to burst it
    jumps by the change in the two acquisitions' relative burst timing. A single polynomial in ``R`` cannot follow
    those steps; fitted through them it leaves a misregistration that ramps through every burst and jumps at burst
    edges -- for TOPS, a phase ramp of about ``2 pi k_t dt_az * error`` along azimuth (~3 rad/km per pixel).

    The map separates the two parts. Offsets are expressed as a time difference ``u`` (in reference rows) between
    the secondary's and the reference's acquisition of the same ground, which is smooth and can be fitted with a
    low-order polynomial; the burst bookkeeping is then applied exactly: reference row -> (burst b, time) -> same
    burst of the secondary (matched by time) -> secondary row.
    """

    def __init__(self, ref_geometry, sec_geometry):
        import numpy as np

        for g in (ref_geometry, sec_geometry):
            if not (getattr(g, "_has_burst_timing", lambda: False)()):
                raise ValueError("burst timing not available")
        self._np = np
        t0 = ref_geometry.burst_azimuth_times[0]
        self.ref_t = np.array([(t - t0).total_seconds() for t in ref_geometry.burst_azimuth_times])
        self.sec_t = np.array([(t - t0).total_seconds() for t in sec_geometry.burst_azimuth_times])
        self.ref_lpb, self.sec_lpb = int(ref_geometry.lines_per_burst), int(sec_geometry.lines_per_burst)
        self.ref_dt = float(ref_geometry.azimuth_time_interval_s)
        self.sec_dt = float(sec_geometry.azimuth_time_interval_s)
        self.match = None

    def _ref_time(self, R):
        np = self._np
        R = np.asarray(R, dtype=np.float64)
        b = np.clip(np.floor(R / self.ref_lpb).astype(np.int64), 0, len(self.ref_t) - 1)
        return self.ref_t[b] + (R - b * self.ref_lpb) * self.ref_dt, b

    def _sec_time(self, S):
        np = self._np
        S = np.asarray(S, dtype=np.float64)
        b = np.clip(np.floor(S / self.sec_lpb).astype(np.int64), 0, len(self.sec_t) - 1)
        return self.sec_t[b] + (S - b * self.sec_lpb) * self.sec_dt

    def to_u(self, R, S):
        """Time difference (in reference rows) between secondary row S and reference row R."""
        t_ref, _ = self._ref_time(R)
        return (self._sec_time(S) - t_ref) / self.ref_dt

    def set_matching(self, u_typical: float) -> None:
        """Pair every reference burst with the secondary burst covering the same ground (nearest centre time)."""
        np = self._np
        ref_mid = self.ref_t + 0.5 * self.ref_lpb * self.ref_dt + u_typical * self.ref_dt
        sec_mid = self.sec_t + 0.5 * self.sec_lpb * self.sec_dt
        self.match = np.array([int(np.argmin(np.abs(sec_mid - m))) for m in ref_mid])

    def steps_px(self) -> "list[float]":
        """Relative burst timing (reference minus matched secondary burst start), in rows, per reference burst."""
        d = (self.ref_t - self.sec_t[self.match]) / self.ref_dt
        return [float(x) for x in d - self._np.median(d)]

    def wrap(self, u_fn):
        """Row-offset function (secondary row - reference row) from a smooth time-difference function u(R, C)."""
        np = self._np
        if self.match is None:
            raise RuntimeError("call set_matching() first")

        def row_fn(R, C):
            R = np.asarray(R, dtype=np.float64)
            t_ref, b = self._ref_time(R)
            t = t_ref + np.asarray(u_fn(R, C), dtype=np.float64) * self.ref_dt
            bs = self.match[b]
            S = bs * self.sec_lpb + (t - self.sec_t[bs]) / self.sec_dt
            return (S - R).astype(np.float32)

        return row_fn
