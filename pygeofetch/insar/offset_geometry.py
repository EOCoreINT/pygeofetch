"""
Geometric projection for SAR offset tracking.

Converts raw pixel offsets (range, azimuth) from amplitude-based speckle
tracking into ground displacement vectors (East, North, Up).

This module deliberately separates geometry from the signal processing
(NCC, sub-pixel refinement) to ensure all coordinate transformations are
grounded in established, peer-reviewed SAR geometry conventions.

Primary References:
1. Joughin, I. (2002). "Ice-sheet velocity mapping: a combined
   interferometric and speckle-tracking approach." Annals of Glaciology, 34, 195-201.
2. Wegmüller, U., et al. (2016). "SAR backscatter and offset tracking for
   the monitoring of surface deformation." IEEE JSTARS, 9(8), 3620-3629.
3. Fialko, Y., Simons, M., & Agnew, D. (2001). "The complete (3-D) surface
   displacement field... from space geodetic observations." GRL, 28(16), 3063-3066.

Convention Used:
- Heading angle (alpha_h): Azimuth of satellite velocity vector, clockwise
  from North (0 to 360 degrees). E.g., ~190° for Sentinel-1 descending.
- Incidence angle (theta_inc): Angle from vertical (nadir) to the line-of-sight.
- Range offset: Positive means the target moved to a higher range pixel
  index (increased slant range, i.e., away from the satellite).
- Azimuth offset: Positive means the target moved to a higher azimuth
  pixel index (in the same direction as the satellite's along-track motion).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("pygeofetch.insar.offset_geometry")


def pixel_to_physical_offsets(
    range_offset_px: np.ndarray,
    azimuth_offset_px: np.ndarray,
    range_spacing_m: float,
    azimuth_spacing_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel offsets to physical distances in meters.

    Args:
        range_offset_px: 2D array of range offsets in pixels.
        azimuth_offset_px: 2D array of azimuth offsets in pixels.
        range_spacing_m: Ground range pixel spacing in meters (e.g., ~2.3m for S1 IW).
        azimuth_spacing_m: Azimuth pixel spacing in meters (e.g., ~14.1m for S1 IW).

    Returns:
        Tuple of (range_offset_m, azimuth_offset_m) as float64 arrays.
    """
    if range_spacing_m <= 0 or azimuth_spacing_m <= 0:
        raise ValueError("Pixel spacings must be strictly positive.")

    range_offset_m = np.asarray(range_offset_px, dtype=np.float64) * range_spacing_m
    azimuth_offset_m = (
        np.asarray(azimuth_offset_px, dtype=np.float64) * azimuth_spacing_m
    )

    return range_offset_m, azimuth_offset_m


def solve_enu_displacement(
    range_offset_m: np.ndarray,
    azimuth_offset_m: np.ndarray,
    incidence_angle_deg: float,
    heading_angle_deg: float,
    vertical_displacement_m: Optional[np.ndarray] = None,
    assume_north_zero: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve for East, North, Up ground displacement from SAR offset tracking.

    This function solves the underdetermined 2-equation, 3-unknown system
    by leveraging either a known vertical displacement (e.g., from SBAS)
    or a reasonable geophysical assumption (e.g., North-South motion is
    negligible for E-W oriented faults).

    The forward model (Joughin 2002, Wegmüller 2016) for a right-looking
    SAR (like Sentinel-1) is:
        ΔR = -d_E * sin(θ) * cos(α) + d_N * sin(θ) * sin(α) - d_U * cos(θ)
        ΔA =  d_E * sin(α)          + d_N * cos(α)

    Where:
        ΔR = range offset (positive = increased range)
        ΔA = azimuth offset (positive = along-track direction)
        d_E, d_N, d_U = East, North, Up ground displacement
        θ = incidence angle
        α = satellite heading angle (clockwise from North)

    Args:
        range_offset_m: 2D array of range offsets in meters.
        azimuth_offset_m: 2D array of azimuth offsets in meters.
        incidence_angle_deg: Mean incidence angle at the target (degrees).
        heading_angle_deg: Satellite heading angle, clockwise from North (degrees).
            E.g., ~190° for Sentinel-1 descending, ~350° for ascending.
        vertical_displacement_m: Optional 2D array of Up displacement (meters).
            If provided, the system is fully constrained for d_E and d_N.
            Highly recommended to use the SBAS-derived vertical velocity/displacement.
        assume_north_zero: If True and vertical_displacement_m is None, assumes
            d_N = 0. Useful for E-W striking faults or when only E-W motion is
            expected. Warning: This is unstable if heading is near 0° or 180°
            (sin(α) ≈ 0).

    Returns:
        Tuple of (displacement_east_m, displacement_north_m, displacement_up_m)
        as float64 arrays. NaNs are propagated from unreliable input masks.
    """
    # Ensure inputs are float64 for numerical stability in matrix inversion
    range_offset_m = np.asarray(range_offset_m, dtype=np.float64)
    azimuth_offset_m = np.asarray(azimuth_offset_m, dtype=np.float64)

    theta = np.radians(incidence_angle_deg)
    alpha = np.radians(heading_angle_deg)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)

    # Initialize output arrays with NaNs to safely propagate unreliable masks
    shape = range_offset_m.shape
    d_E = np.full(shape, np.nan, dtype=np.float64)
    d_N = np.full(shape, np.nan, dtype=np.float64)
    d_U = np.full(shape, np.nan, dtype=np.float64)

    # Valid mask: ignore NaNs from the offset tracker's unreliable windows
    valid_mask = np.isfinite(range_offset_m) & np.isfinite(azimuth_offset_m)

    if not np.any(valid_mask):
        logger.warning("solve_enu_displacement: No valid offset data to project.")
        return d_E, d_N, d_U

    if vertical_displacement_m is not None:
        # ── MODE 1: Constrained by known vertical displacement (Recommended)
        # This leverages the SBAS result to solve the remaining 2x2 system for E and N.
        d_U[valid_mask] = vertical_displacement_m[valid_mask]

        # Rearrange the range equation to isolate the horizontal components
        # R' = ΔR + d_U * cos(θ)
        R_prime = range_offset_m[valid_mask] + d_U[valid_mask] * cos_theta
        A_prime = azimuth_offset_m[valid_mask]

        # 2x2 System:
        # [ R' ]   [ -sin(θ)cos(α)   sin(θ)sin(α) ] [ d_E ]
        # [ A' ] = [    sin(α)          cos(α)     ] [ d_N ]
        #
        # Determinant = -sin(θ)cos²(α) - sin(θ)sin²(α) = -sin(θ)
        # Since θ is typically 30-45°, sin(θ) is safely non-zero.

        det = -sin_theta

        d_E[valid_mask] = (cos_alpha * R_prime - sin_theta * sin_alpha * A_prime) / det
        d_N[valid_mask] = (-sin_alpha * R_prime - sin_theta * cos_alpha * A_prime) / det

        logger.info(
            "solve_enu_displacement: Solved for E/N using provided vertical displacement. "
            "Matrix determinant (sin(θ)) = %.4f (stable).",
            sin_theta,
        )

    elif assume_north_zero:
        # ── MODE 2: Assume North-South motion is zero (d_N = 0)
        # Useful for E-W deformation, but mathematically unstable if sin(α) ≈ 0.
        if abs(sin_alpha) < 0.1:
            logger.warning(
                "solve_enu_displacement: assume_north_zero is highly unstable for "
                "heading angles near 0° or 180° (sin(α) ≈ %.4f). "
                "Results for d_E will be noisy.",
                sin_alpha,
            )

        d_U[valid_mask] = 0.0  # Explicitly set to 0 for this mode
        d_N[valid_mask] = 0.0

        # From azimuth equation: ΔA = d_E * sin(α)  =>  d_E = ΔA / sin(α)
        d_E[valid_mask] = azimuth_offset_m[valid_mask] / sin_alpha

        # From range equation: ΔR = -d_E * sin(θ) * cos(α) - d_U * cos(θ)
        # Since d_U = 0, we can solve for it if we wanted, but we assume d_U is
        # coupled to d_E here. Actually, let's solve for d_U to be complete:
        # d_U = (-ΔR - d_E * sin(θ) * cos(α)) / cos(θ)
        d_U[valid_mask] = (
            -range_offset_m[valid_mask] - d_E[valid_mask] * sin_theta * cos_alpha
        ) / cos_theta

        logger.info(
            "solve_enu_displacement: Solved assuming d_N = 0. "
            "Note: This is an approximation; use vertical_displacement_m if available."
        )

    else:
        raise ValueError(
            "Underdetermined system: Must provide either `vertical_displacement_m` "
            "(recommended, e.g., from SBAS) or set `assume_north_zero=True`."
        )

    return d_E, d_N, d_U
