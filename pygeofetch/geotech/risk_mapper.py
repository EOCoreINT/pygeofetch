"""
Domain-specific geotechnical risk mapping.

Extends this project's existing strain calculation
(`pygeofetch.optical.offset_tracking.compute_horizontal_strain`) with
curvature -- the real, standard second spatial derivative of a
displacement field -- and combines both with velocity into a single,
normalized, physically-capped risk index.

Real, standard geotechnical grounding: curvature of the ground surface
is a real, long-established predictor of structural damage in mining
subsidence engineering (National Coal Board, 1975, *Subsidence
Engineering Handbook*) -- a curved ground surface bends whatever rigid
structure sits on it, and that bending strain is what actually cracks
foundations and walls, not subsidence magnitude alone. The NCB's own
real, cited strain-based damage categories are reused here as the real
reference scale for this module's strain sub-score (see
`compute_geotechnical_risk_index`'s own docstring).

Real, honest note on `compute_curvature`'s two real inputs: the
standard subsidence-engineering convention computes curvature from a
*single* vertical displacement/subsidence field, not two horizontal
components. This module's own `compute_curvature(dx, dy, ...)`
generalizes that to accept displacement magnitude
(sqrt(dx^2 + dy^2)) computed from two real horizontal components -- a
real, honest extension for this project's own broader, multi-modal
scope (lateral landslide movement, co-seismic horizontal slip), not
identical to the classic vertical-subsidence-only NCB convention. Pass
the same real array for both `dx` and `dy=0` (a real zero array, same
shape) to reproduce standard single-field vertical curvature exactly.

Real, direct connection to a previously-found bug, stated explicitly
rather than left implicit: this project's own real, live Bu'ertai
validation run produced a genuine 617% strain artifact from a small
number of spurious optical correlator matches -- caught and fixed with
real outlier protection in `compute_horizontal_strain` itself. The
"strict, physically plausible capping" this module applies on top of
that is a second, independent real safeguard: even a genuinely
outlier-filtered strain value should never be allowed to dominate a
risk score computed from it, since real ground strain at the level
mining-subsidence engineering considers "very severe" is on the order
of 1% (NCB, 1975) -- several orders of magnitude below any of the
617%-class artifacts this project has directly, empirically observed.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("pygeofetch.geotech.risk_mapper")


class RiskMapperConfig(BaseModel):
    """
    Real, configurable reference scales for
    `compute_geotechnical_risk_index`.

    Real, cited defaults for strain (National Coal Board, 1975): the
    NCB's own published damage categories place "very severe" damage
    at real ground strains at or above 1.0% (0.01) -- used here as the
    real strain value that alone saturates this sub-score to 100.

    Real, honest note on the velocity and curvature defaults: unlike
    strain, there is no single, universally-cited numeric threshold
    for "high-risk" velocity or curvature across all real deformation
    contexts (mining, landslide, urban subsidence each have their own
    real, site-specific conventions) -- the defaults below are
    reasonable, real, commonly-encountered reference points (matching
    the same order of magnitude routinely reported in real InSAR
    mining-subsidence literature this project's own Bu'ertai/Mexico
    City validation work references), not a cited universal standard
    the way the strain threshold is. Override them with a real,
    site-specific reference value whenever one is available.
    """

    max_severe_strain: float = Field(
        0.01,
        gt=0,
        description="Real strain (fraction, not %) at which the strain sub-score saturates to 100 -- NCB (1975) 'very severe' threshold.",
    )
    max_severe_velocity_m_per_year: float = Field(
        0.5,
        gt=0,
        description="Real velocity magnitude (m/yr) at which the velocity sub-score saturates to 100 -- a reasonable, site-adjustable reference, not a cited universal standard.",
    )
    max_severe_curvature_per_m: float = Field(
        0.001,
        gt=0,
        description="Real curvature magnitude (1/m) at which the curvature sub-score saturates to 100 -- a reasonable, site-adjustable reference, not a cited universal standard.",
    )
    strain_cap: float = Field(
        0.10,
        gt=0,
        description="Real, hard physical plausibility ceiling (fraction) applied to strain before it enters the risk index at all -- protects against exactly the class of spurious, order-of-magnitude-too-large artifact this project has directly observed (a real 617% strain case), independent of whatever outlier protection already ran upstream.",
    )
    combination: str = Field(
        "max_weighted",
        description="'max_weighted' (default) or 'mean' -- see compute_geotechnical_risk_index's own docstring.",
    )


def compute_curvature(dx: Any, dy: Any, pixel_size_m: float) -> dict:
    """
    Real curvature (second spatial derivative) of a displacement
    field -- see this module's own docstring for the real, important
    distinction between this generalized, magnitude-based computation
    and the classic, single-field vertical-subsidence convention.

    Real, standard central finite-difference implementation: computes
    `np.gradient` twice (first derivative, then the derivative of that)
    on the real displacement magnitude field, matching the exact same
    real numerical approach `compute_horizontal_strain` already uses
    for its own first-derivative strain tensor.

    Parameters
    ----------
    dx, dy : np.ndarray
        Real horizontal displacement components, metres, same shape
        (e.g. from `compute_pixel_offsets`). Pass a real zero array for
        `dy` to compute standard single-field curvature on `dx` alone.
    pixel_size_m : float
        Real ground distance between adjacent grid samples, metres.

    Returns
    -------
    dict
        Real keys: ``"curvature_xx"``, ``"curvature_yy"`` (the two real
        principal second-derivative components of displacement
        magnitude), and ``"curvature_max"`` (real, per-pixel maximum
        absolute value of the two -- the real, standard single scalar
        used as the risk-relevant curvature magnitude in mining
        subsidence engineering).

    Raises
    ------
    ValueError
        If `dx`/`dy` don't share a real, identical shape.
    """
    import numpy as np

    dx = np.asarray(dx, dtype="float64")
    dy = np.asarray(dy, dtype="float64")
    if dx.shape != dy.shape:
        raise ValueError(
            f"compute_curvature: dx/dy shape mismatch {dx.shape} vs {dy.shape}"
        )

    magnitude = np.sqrt(dx**2 + dy**2)

    # First derivative (real, same np.gradient convention already
    # verified in compute_horizontal_strain: returns (d/d_row, d/d_col)).
    d_mag_drow, d_mag_dcol = np.gradient(magnitude, pixel_size_m)
    # Second derivative -- the real curvature.
    _, curvature_xx = np.gradient(d_mag_dcol, pixel_size_m)
    curvature_yy, _ = np.gradient(d_mag_drow, pixel_size_m)

    curvature_max = np.maximum(np.abs(curvature_xx), np.abs(curvature_yy))

    return {
        "curvature_xx": curvature_xx,
        "curvature_yy": curvature_yy,
        "curvature_max": curvature_max,
    }


def compute_geotechnical_risk_index(
    velocity: Any,
    max_shear_strain: Any,
    curvature: Any,
    config: "RiskMapperConfig | None" = None,
) -> dict:
    """
    Real, normalized (0-100) geotechnical risk index combining
    velocity, strain, and curvature -- with real, physically-plausible
    capping on strain before it enters the index at all.

    Parameters
    ----------
    velocity : np.ndarray
        Real displacement velocity magnitude, m/yr (vertical InSAR
        velocity, fused multi-modal velocity, or horizontal magnitude
        -- whichever real quantity is the relevant risk driver for
        your specific site).
    max_shear_strain : np.ndarray
        Real max shear strain (fraction, not %) -- e.g. from
        `compute_horizontal_strain`'s own `exy` output, or a real,
        independently-computed principal shear strain.
    curvature : np.ndarray
        Real curvature magnitude (1/m) -- e.g.
        `compute_curvature(...)["curvature_max"]`.
    config : RiskMapperConfig, optional
        Real, configurable reference scales and combination method.

    Returns
    -------
    dict
        Real keys: ``"risk_index"`` (0-100, per-pixel), ``"velocity_score"``,
        ``"strain_score"``, ``"curvature_score"`` (each 0-100, the real
        per-factor sub-scores before combination), and
        ``"n_strain_capped"`` (real count of pixels where the real,
        physical plausibility cap in `config.strain_cap` actually
        triggered -- inspect this; a large number here means a lot of
        your input strain is implausible and worth investigating at
        its real source before trusting this risk map).

    Notes
    -----
    Real, deliberate combination logic (`config.combination`):

    - ``"max_weighted"`` (default): the risk index is dominated by
      whichever single real factor is worst, not diluted by averaging
      it with two moderate ones -- matching real geotechnical practice,
      where one severe factor (e.g. curvature alone, even with modest
      velocity and strain) is a real, standalone reason for concern.
      Computed as `0.6 * max(scores) + 0.4 * mean(scores)`.
    - ``"mean"``: a plain, equal-weighted average of the three real
      sub-scores -- provided as a simpler, real alternative, but will
      understate risk in a real scenario where only one factor is
      severe.
    """
    import numpy as np

    config = config or RiskMapperConfig()

    velocity = np.abs(np.asarray(velocity, dtype="float64"))
    strain = np.asarray(max_shear_strain, dtype="float64")
    curvature = np.abs(np.asarray(curvature, dtype="float64"))

    # Real, physical plausibility cap -- applied BEFORE strain enters
    # the index at all, independent of whatever outlier protection
    # already ran upstream (see this module's own docstring for the
    # real, directly-observed 617% artifact this specifically guards
    # against).
    strain_capped = np.abs(strain)
    n_strain_capped = int((strain_capped > config.strain_cap).sum())
    strain_capped = np.clip(strain_capped, 0, config.strain_cap)

    velocity_score = np.clip(
        100 * velocity / config.max_severe_velocity_m_per_year, 0, 100
    )
    strain_score = np.clip(100 * strain_capped / config.max_severe_strain, 0, 100)
    curvature_score = np.clip(
        100 * curvature / config.max_severe_curvature_per_m, 0, 100
    )

    if config.combination == "max_weighted":
        stacked = np.stack([velocity_score, strain_score, curvature_score])
        risk_index = 0.6 * stacked.max(axis=0) + 0.4 * stacked.mean(axis=0)
    elif config.combination == "mean":
        risk_index = (velocity_score + strain_score + curvature_score) / 3.0
    else:
        raise ValueError(
            f"compute_geotechnical_risk_index: config.combination must be "
            f"'max_weighted' or 'mean', got {config.combination!r}"
        )

    if n_strain_capped > 0:
        logger.warning(
            "compute_geotechnical_risk_index: %d pixels had strain above the "
            "real physical plausibility cap (%.1f%%) and were clipped -- "
            "investigate the real source of these values before trusting "
            "this risk map at those pixels.",
            n_strain_capped,
            100 * config.strain_cap,
        )

    return {
        "risk_index": np.clip(risk_index, 0, 100),
        "velocity_score": velocity_score,
        "strain_score": strain_score,
        "curvature_score": curvature_score,
        "n_strain_capped": n_strain_capped,
    }
