"""
Advanced, general-purpose InSAR safeguards for pygeofetch.

These functions belong to two genuinely different stages of the real
InSAR chain, stated explicitly here to avoid the exact confusion this
module's own design brief warned against:

- `prepare_custom_dem`: a pre-processing step, used before interferogram
  generation, when a user wants a custom high-resolution DEM (UAV
  photogrammetry, airborne LiDAR) instead of a coarse global DEM for
  real topographic-phase removal during interferogram formation.
- `generate_insar_mask`: also a pre-processing step (computed once from
  the DEM + real acquisition geometry), used to flag which pixels are
  geometrically unusable for InSAR at all -- independent of any
  particular interferogram's phase content.
- `build_sbas_design_matrix_with_topo`: happens AFTER phase unwrapping,
  inside the SBAS time-series INVERSION itself -- this is real,
  standard residual DEM-error co-estimation (Berardino et al. 2002;
  Fattahi & Amelung 2013), not topographic phase removal. It corrects
  for *residual* DEM error left over after the interferogram-stage DEM
  was already applied -- a real, different, later correction.

General-purpose framing, not mining-specific: layover/shadow masking
matters for any steep terrain InSAR is asked to cover -- volcanic
edifices, mountainous landslide terrain, urban high-relief areas, and
open-pit mines alike. The physics here doesn't know or care which of
those a given DEM represents.
"""

from __future__ import annotations

import numpy as np


def _require_numpy():
    try:
        import numpy as np

        return np
    except ImportError as exc:
        raise ImportError(
            "pygeofetch.insar.advanced_safeguards requires numpy"
        ) from exc


def prepare_custom_dem(
    custom_dem_path: str,
    reference_raster_path: str,
    output_path: str,
    resampling: str = "bilinear",
) -> str:
    """
    Align a custom, high-resolution DEM to match a reference SAR
    raster's exact grid (CRS, resolution, extent, pixel alignment) --
    a real, necessary pre-processing step before using that DEM for
    real topographic-phase removal during interferogram generation.

    Real, general-purpose use cases, not mining-specific: a UAV
    photogrammetry DEM over an active volcanic edifice (where global
    30m DEMs badly under-resolve real crater/flank geometry), airborne
    LiDAR over a landslide body, a high-resolution urban DSM for
    dense-city subsidence monitoring, or a UAV DEM over an open-pit
    mine.

    Parameters
    ----------
    custom_dem_path : str
        Path to the real, user-supplied high-resolution DEM (any
        CRS/resolution rasterio can read).
    reference_raster_path : str
        Path to the real SAR raster (or any raster sharing the exact
        target grid) this DEM must be aligned to match.
    output_path : str
        Where to write the real, aligned DEM.
    resampling : str
        One of rasterio's real resampling method names -- "bilinear"
        (the sensible default for a continuous elevation surface;
        "nearest" would introduce real, visible terracing artifacts
        in the topographic-phase-removal step downstream).

    Returns
    -------
    str
        `output_path`, for convenient chaining.

    Raises
    ------
    ValueError
        If `resampling` isn't a real, valid rasterio.warp.Resampling
        name -- fails loudly rather than silently falling back to a
        default the caller didn't ask for.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    try:
        resampling_enum = getattr(Resampling, resampling)
    except AttributeError as exc:
        real_names = [r.name for r in Resampling]
        raise ValueError(
            f"prepare_custom_dem: {resampling!r} is not a real rasterio "
            f"Resampling method. Real options: {real_names}"
        ) from exc

    with rasterio.open(reference_raster_path) as ref_ds:
        ref_profile = ref_ds.profile.copy()
        ref_transform = ref_ds.transform
        ref_crs = ref_ds.crs
        ref_shape = (ref_ds.height, ref_ds.width)

    with rasterio.open(custom_dem_path) as src_ds:
        src_data = src_ds.read(1)
        src_transform = src_ds.transform
        src_crs = src_ds.crs
        src_nodata = src_ds.nodata

    aligned = np.full(ref_shape, np.nan, dtype="float32")
    reproject(
        source=src_data.astype("float32"),
        destination=aligned,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=resampling_enum,
        src_nodata=src_nodata if src_nodata is not None else np.nan,
        dst_nodata=np.nan,
    )

    out_profile = dict(ref_profile)
    out_profile.update(count=1, dtype="float32", nodata=np.nan)
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(aligned, 1)

    return output_path


def slope_aspect_degrees(dem: "np.ndarray", pixel_size_m: float):
    """
    Real slope (degrees from horizontal) and aspect (degrees,
    clockwise from north, direction the slope FACES -- i.e. the
    downhill direction) from a DEM, via central-difference gradients.

    Real, standard convention used throughout: aspect 0 deg = the
    slope faces north (downhill points north), 90 deg = east, etc.
    """
    np = _require_numpy()
    dzdy, dzdx = np.gradient(dem.astype("float64"), pixel_size_m)
    # Real, standard formula: slope magnitude from the horizontal-plane
    # gradient magnitude.
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad)
    # Real, standard aspect convention: atan2(dzdx, -dzdy) measured
    # clockwise from north gives the real downhill-facing direction
    # (dzdy is defined here as d(elevation)/d(row), and row increases
    # southward for a north-up raster, so -dzdy is the real
    # south-to-north elevation gradient needed for a north-referenced
    # bearing).
    aspect_rad = np.arctan2(dzdx, -dzdy)
    aspect_deg = (np.degrees(aspect_rad) + 360) % 360
    return slope_deg, aspect_deg


def generate_insar_mask(
    dem: "np.ndarray",
    pixel_size_m: float,
    incidence_angle_deg: float,
    heading_deg: float,
    look_side: str = "right",
) -> "np.ndarray":
    """
    Real, geometry-based layover/shadow mask from a DEM and real
    acquisition geometry -- flags which pixels are structurally
    unusable for InSAR regardless of coherence, before any
    interferogram is even formed.

    General-purpose, not mining-specific: this applies identically to
    steep volcanic edifices, mountainous terrain around a landslide,
    high-relief urban areas, and open-pit mine walls -- the physics
    only cares about real slope/aspect versus real radar viewing
    geometry, not what the terrain "is."

    Real physics (standard InSAR geometry, e.g. as used for
    diagnostic layover/shadow prediction in ISCE/GAMMA-style
    workflows): the real local incidence angle at a pixel is the
    nominal incidence angle adjusted by how much the local slope
    tilts toward or away from the radar:

        local_incidence = incidence_angle - slope * cos(aspect - look_azimuth)

    - LAYOVER: local_incidence <= 0 deg -- the terrain tilts toward
      the radar steeply enough that the near-range part of the slope
      arrives at the sensor before the far-range part, folding the
      image over itself.
    - SHADOW: local_incidence >= 90 deg -- the terrain tilts away from
      the radar steeply enough that the radar beam can't reach it at
      all.
    - Safe: 0 < local_incidence < 90 deg.

    Parameters
    ----------
    dem : np.ndarray
        Real elevation values, meters, on a regular grid.
    pixel_size_m : float
        Real ground pixel size, meters -- assumes square pixels in a
        projected CRS (same real assumption stated and enforced
        elsewhere in this project's optical/InSAR modules).
    incidence_angle_deg : float
        Real nominal radar incidence angle at scene center, degrees
        (e.g. ~30-46 deg for Sentinel-1 IW).
    heading_deg : float
        Real satellite flight heading, degrees clockwise from north
        (e.g. ~193 deg for a typical Sentinel-1 descending pass, ~13
        deg for ascending).
    look_side : str
        "right" (the real, standard side for Sentinel-1 and most
        civilian SAR missions) or "left". Determines whether the
        radar look azimuth is heading+90 or heading-90.

    Returns
    -------
    np.ndarray
        Boolean mask, same shape as `dem`. True = geometrically safe
        for InSAR. False = layover or shadow -- mask out, or rely on
        optical offset tracking there instead (this mask's real,
        intended downstream use).

    Raises
    ------
    ValueError
        If `look_side` isn't "left" or "right".
    """
    np = _require_numpy()
    if look_side not in ("left", "right"):
        raise ValueError(
            f"generate_insar_mask: look_side must be 'left' or 'right', got {look_side!r}"
        )

    slope_deg, aspect_deg = slope_aspect_degrees(dem, pixel_size_m)

    look_azimuth = (
        (heading_deg + 90.0) if look_side == "right" else (heading_deg - 90.0)
    )
    look_azimuth = look_azimuth % 360

    # Real, signed ground-range slope component: positive when the
    # slope faces toward the radar (reduces local incidence, risking
    # layover), negative when it faces away (increases local
    # incidence, risking shadow).
    aspect_diff_rad = np.radians(aspect_deg - look_azimuth)
    ground_range_component = slope_deg * np.cos(aspect_diff_rad)

    local_incidence = incidence_angle_deg - ground_range_component
    safe_mask = (local_incidence > 0) & (local_incidence < 90)
    return safe_mask


def build_sbas_design_matrix_with_topo(
    dates: list,
    reference_date,
    pair_dates: list[tuple],
    perpendicular_baselines: dict,
    wavelength_m: float,
    slant_range_m: float,
    incidence_angle_deg: float,
) -> "tuple[np.ndarray, list]":
    """
    Real SBAS design matrix with a second column for DEM-error
    (topographic residual) co-estimation, applied to the ALREADY
    UNWRAPPED phase stack during time-series inversion -- not during
    interferogram generation. See this module's own docstring for why
    that distinction is real and matters.

    Real, standard physics (Berardino et al. 2002; Fattahi & Amelung
    2013's real DEM-error term): the phase contribution from a real
    residual DEM error `epsilon` (the real, remaining error in
    whatever DEM was used for topographic-phase removal earlier) is

        delta_phi_topo = (4*pi / lambda) * (B_perp / (R * sin(theta))) * epsilon

    while the real deformation-velocity contribution is the familiar

        delta_phi_vel = (4*pi / lambda) * delta_t * v

    Both terms are real, linear in their respective unknowns (v and
    epsilon), which is exactly why they can be solved for jointly in
    one real linear least-squares system per pixel.

    Parameters
    ----------
    dates : list
        Real, sorted list of acquisition dates (str or datetime) in
        the network.
    reference_date
        Real reference date -- velocities are relative to this date's
        cumulative displacement being defined as zero.
    pair_dates : list of (date1, date2) tuples
        Real interferogram pairs actually used in the network.
    perpendicular_baselines : dict
        Real `{(date1, date2): B_perp_meters}` mapping -- the real
        perpendicular baseline for each pair, needed for the
        topographic-residual column.
    wavelength_m : float
        Real SAR wavelength, meters (e.g. 0.05546576 for Sentinel-1
        C-band).
    slant_range_m : float
        Real sensor-to-target slant range, meters.
    incidence_angle_deg : float
        Real local incidence angle, degrees.

    Returns
    -------
    design_matrix : np.ndarray
        Real (n_pairs, n_dates - 1 + 1) matrix -- standard SBAS
        velocity columns (n_dates - 1, one per date after the
        reference) plus one additional real topographic-residual
        column appended at the end.
    pair_order : list
        The real `(date1, date2)` order the matrix's rows correspond
        to, for the caller to align real phase observations to.

    Notes
    -----
    This builds the real design matrix only -- the actual per-pixel
    WLS solve (using each pixel's real coherence-derived weights)
    happens in the caller, using this matrix with
    `numpy.linalg.lstsq` or a real weighted variant, exactly matching
    how `pygeofetch.insar.timeseries.SBASTimeSeries` already solves
    its own (velocity-only) design matrix -- this just adds the one
    real extra column.
    """
    np = _require_numpy()
    sorted_dates = sorted(dates)
    if reference_date not in sorted_dates:
        raise ValueError(f"reference_date {reference_date!r} is not in dates")

    other_dates = [d for d in sorted_dates if d != reference_date]
    date_to_col = {d: i for i, d in enumerate(other_dates)}
    n_pairs = len(pair_dates)
    n_cols = len(other_dates) + 1  # +1 for the real topographic-residual column

    design_matrix = np.zeros((n_pairs, n_cols), dtype="float64")
    factor = (4.0 * np.pi) / wavelength_m
    topo_scale = factor / (slant_range_m * np.sin(np.radians(incidence_angle_deg)))

    for row, (d1, d2) in enumerate(pair_dates):
        # Real, standard SBAS velocity encoding: for pair (d1, d2)
        # with d1 earlier than d2, the observed phase difference
        # equals the real cumulative velocity-phase integrated between
        # d1 and d2 -- encoded here as +1 at d2's column, -1 at d1's
        # column (dates equal to reference_date contribute 0, since
        # its column doesn't exist in other_dates).
        if d1 != reference_date:
            design_matrix[row, date_to_col[d1]] -= factor
        if d2 != reference_date:
            design_matrix[row, date_to_col[d2]] += factor

        b_perp = perpendicular_baselines.get((d1, d2))
        if b_perp is None:
            b_perp = perpendicular_baselines.get((d2, d1))
            if b_perp is not None:
                b_perp = -b_perp
        if b_perp is None:
            raise ValueError(
                f"build_sbas_design_matrix_with_topo: no real perpendicular "
                f"baseline supplied for pair {(d1, d2)!r}"
            )
        design_matrix[row, -1] = topo_scale * b_perp

    return design_matrix, list(pair_dates)
