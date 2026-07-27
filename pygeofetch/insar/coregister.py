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
both describe): compute the real offset at a sparse grid of points
across the image (not every pixel — solving per pixel would be far
too slow and offers no real accuracy benefit, since the offset field
is smooth), fit a low-degree 2D polynomial to that sparse grid, then
apply the fitted, dense field to resample the secondary image.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

logger = logging.getLogger("pygeofetch.insar.coregister")


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
    import rasterio

    try:
        with rasterio.open(path) as src:
            tags = src.tags()
            return (
                float(tags.get("crop_row_off", 0.0)),
                float(tags.get("crop_col_off", 0.0)),
            )
    except Exception:
        return (0.0, 0.0)


def compute_offset_field_from_dem(
    ref_geometry,
    ref_orbit,
    sec_geometry,
    sec_orbit,
    dem_path: Union[str, Path],
    ref_scene_center_time,
    sec_scene_center_time,
    grid_points: int = 7,
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

    from pygeofetch.insar.geolocation import geodetic_to_ecef, find_zero_doppler_time

    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        transform = src.transform
        dem_h, dem_w = dem.shape

    dem_row_samples = [int(r) for r in _linspace(0, dem_h - 1, grid_points)]
    dem_col_samples = [int(c) for c in _linspace(0, dem_w - 1, grid_points)]

    grid_rows, grid_cols, offset_rows, offset_cols = [], [], [], []
    n_failed = 0
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
                    ref_orbit[0], ref_orbit[1], ref_orbit[2],
                    ground_point, ref_scene_center_time,
                )
                t_sec = find_zero_doppler_time(
                    sec_orbit[0], sec_orbit[1], sec_orbit[2],
                    ground_point, sec_scene_center_time,
                )

                ref_row = ref_geometry.row_for_azimuth_time(t_ref)
                sec_row = sec_geometry.row_for_azimuth_time(t_sec)

                sat_pos_ref, _ = _interpolate(ref_orbit, t_ref)
                sat_pos_sec, _ = _interpolate(sec_orbit, t_sec)
                range_ref_time = 2 * _distance(sat_pos_ref, ground_point) / 299792458.0
                range_sec_time = 2 * _distance(sat_pos_sec, ground_point) / 299792458.0
                ref_col = ref_geometry.col_for_range_time(range_ref_time)
                sec_col = sec_geometry.col_for_range_time(range_sec_time)

                if not (0 <= ref_row < ref_geometry.n_lines and 0 <= ref_col < ref_geometry.n_columns):
                    continue  # this DEM point falls outside the actual SLC extent

                grid_rows.append(ref_row)
                grid_cols.append(ref_col)
                offset_rows.append(sec_row - ref_row)
                offset_cols.append(sec_col - ref_col)
            except RuntimeError as exc:
                n_failed += 1
                logger.debug("DEM-driven offset field: point (%d, %d) failed: %s", dem_row, dem_col, exc)

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

    logger.info(
        "DEM-driven offset field: %d/%d points solved successfully",
        len(grid_rows), n_total,
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
    from pygeofetch.insar.geolocation import solve_ground_point, find_zero_doppler_time

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
                    sat_pos, sat_vel, range_time, dem_height_m=dem_height_m,
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
                logger.debug("Offset field: grid point (%d, %d) failed: %s", row, col, exc)
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
            n_failed, n_total, n_total - n_failed,
        )

    return grid_rows, grid_cols, offset_rows, offset_cols


def fit_offset_polynomial(grid_rows, grid_cols, offsets, degree: int = 1):
    """
    Fit a low-degree 2D polynomial to a sparse offset grid — standard
    practice (Kampes, Hanssen & Perski 2003; matches the 2019 URSI
    paper's approach) rather than every-pixel solving, since the true
    offset field varies smoothly across a scene.

    Returns a callable f(row, col) -> offset, evaluable at any pixel.
    """
    import numpy as np

    rows = np.asarray(grid_rows, dtype=np.float64)
    cols = np.asarray(grid_cols, dtype=np.float64)
    vals = np.asarray(offsets, dtype=np.float64)

    if degree == 1:
        A = np.column_stack([np.ones_like(rows), rows, cols])
    elif degree == 2:
        A = np.column_stack([
            np.ones_like(rows), rows, cols, rows * cols, rows**2, cols**2,
        ])
    else:
        raise ValueError(f"Unsupported polynomial degree: {degree} (use 1 or 2)")

    coeffs, *_ = np.linalg.lstsq(A, vals, rcond=None)

    def evaluate(row, col):
        row = np.asarray(row, dtype=np.float64)
        col = np.asarray(col, dtype=np.float64)
        if degree == 1:
            return coeffs[0] + coeffs[1] * row + coeffs[2] * col
        return (
            coeffs[0] + coeffs[1] * row + coeffs[2] * col
            + coeffs[3] * row * col + coeffs[4] * row**2 + coeffs[5] * col**2
        )

    return evaluate


def resample_with_offset_field(
    data, row_offset_fn, col_offset_fn,
    ref_row_offset: float = 0.0, ref_col_offset: float = 0.0,
    sec_row_offset: float = 0.0, sec_col_offset: float = 0.0,
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

    Returns:
        The resampled complex array, on the reference image's grid.
    """
    import numpy as np
    from scipy.ndimage import map_coordinates

    h, w = data.shape
    out_real = np.empty((h, w), dtype=np.float32)
    out_imag = np.empty((h, w), dtype=np.float32)
    data_real = data.real
    data_imag = data.imag

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
    chunk_rows = max(1, min(h, 2000))
    for row_start in range(0, h, chunk_rows):
        row_end = min(row_start + chunk_rows, h)
        row_idx, col_idx = np.mgrid[row_start:row_end, 0:w].astype(np.float32)

        # row_idx/col_idx are the reference grid's local (0-based)
        # coordinates. Convert to full-scene coordinates to correctly
        # evaluate the fitted offset functions, which were built on real
        # annotation-derived full-scene coordinates, not any particular
        # crop's local ones.
        ref_global_row = row_idx + ref_row_offset
        ref_global_col = col_idx + ref_col_offset

        offset_row = row_offset_fn(ref_global_row, ref_global_col)
        offset_col = col_offset_fn(ref_global_row, ref_global_col)

        # Where this pixel's secondary counterpart is, in full-scene coordinates
        sec_global_row = ref_global_row + offset_row
        sec_global_col = ref_global_col + offset_col

        # Convert to the SECONDARY array's own local coordinates for
        # actually indexing into `data` -- the secondary crop can have a
        # different offset than the reference crop, so this is not the
        # same subtraction as ref_row_offset/ref_col_offset above.
        sample_rows = sec_global_row - sec_row_offset
        sample_cols = sec_global_col - sec_col_offset

        out_real[row_start:row_end] = map_coordinates(
            data_real, [sample_rows, sample_cols], order=1, mode="constant", cval=0.0
        )
        out_imag[row_start:row_end] = map_coordinates(
            data_imag, [sample_rows, sample_cols], order=1, mode="constant", cval=0.0
        )

    return (out_real + 1j * out_imag).astype(np.complex64)


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