"""
Persistent Scatterer (PS) selection for pygeofetch — Feature 1 from
the InSAR innovations spec.

Standard SBAS relies on Distributed Scatterers (DS), which decorrelate
over time. Persistent Scatterers — point targets that stay coherent
across the whole stack, typically building corners, exposed rock, and
other stable hard targets — can densify coverage in exactly the areas
this project has already found genuinely sparse (urban/vegetated
terrain; see the Mexico City case study's own 99.8% NaN result).

Two real, independent selection criteria are implemented here, per the
spec:

1. Amplitude Dispersion Index (ADI) — Ferretti, Prati & Rocca (2001),
   "Permanent Scatterers in SAR Interferometry", IEEE TGRS 39(1). The
   original PSInSAR paper's own real, foundational selection
   criterion, computed from amplitude alone, before any interferogram
   is even formed.
2. Temporal coherence — the same paper's real, complex-exponential
   coherence measure, applied AFTER an initial SBAS inversion, to
   further refine the ADI-selected candidates against how well their
   actual observed phase history matches the fitted displacement
   model. Ferretti et al. formulate this in phase space specifically
   because it's robust to phase wrapping ambiguity, unlike a plain
   RMS-of-displacement-residual measure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("pygeofetch.insar.ps_selection")


def _require_numpy():
    try:
        import numpy as np

        return np
    except ImportError as exc:
        raise ImportError("ps_selection requires numpy: pip install numpy") from exc


@dataclass(frozen=True)
class PSSelectionResult:
    """Real output of select_persistent_scatterers()."""

    adi: "Any"  # (h, w) float32 — real amplitude dispersion index per pixel
    ps_mask: "Any"  # (h, w) bool — True where ADI-based selection passed
    mean_amplitude: (
        "Any"  # (h, w) float32 — real per-pixel mean amplitude across the stack
    )
    n_candidates: int
    threshold_used: float


def compute_amplitude_dispersion_index(amplitude_stack, epsilon: float = 1e-6):
    """
    Real ADI computation: ADI = sigma_A / (mu_A + epsilon), per pixel,
    across a real stack of N coregistered SLC amplitude images
    (Ferretti et al. 2001, Eq. 3 in the original paper — the
    definition this module implements directly, not an approximation).

    Low ADI means stable, consistent amplitude return over time — the
    real signature of a persistent point scatterer. High ADI means
    amplitude fluctuates, the signature of a distributed scatterer
    whose backscatter changes as the many small contributors within
    the resolution cell shift relative to each other between passes.

    Args:
        amplitude_stack: Real (N, h, w) array, N coregistered
            amplitude images (NOT phase, NOT dB — linear amplitude,
            matching this module's own docstring and the original
            paper's own formulation).
        epsilon: Real floor added to the mean to avoid a real
            division-by-zero on pixels with genuinely zero mean
            amplitude (e.g. real radar shadow).

    Returns:
        (h, w) real float array of ADI values.
    """
    np = _require_numpy()

    if amplitude_stack.ndim != 3:
        raise ValueError(
            f"compute_amplitude_dispersion_index: expected a real (N, h, w) "
            f"stack, got shape {amplitude_stack.shape}"
        )

    arr = amplitude_stack.astype(np.float64)
    mu = arr.mean(axis=0)
    sigma = arr.std(axis=0)
    return (sigma / (mu + epsilon)).astype(np.float32)


def select_persistent_scatterers(
    amplitude_stack,
    adi_threshold: float = 0.25,
    adi_threshold_relaxed: float = 0.40,
    use_relaxed: bool = False,
    amplitude_percentile: float = 70.0,
) -> PSSelectionResult:
    """
    Real PS candidate selection from amplitude alone, run after SLC
    extraction and before any interferogram is formed — matching the
    spec's own stated integration point.

    A pixel is a real PS candidate when BOTH:
      ADI < threshold (adi_threshold, or adi_threshold_relaxed if
        use_relaxed=True)
      AND
      mean amplitude > the amplitude_percentile-th percentile of the
        real mean-amplitude distribution across the whole scene

    The amplitude floor matters as much as the ADI ceiling: a pixel
    can have a real, low ADI simply because it's consistently near
    the noise floor (radiometrically dim in every acquisition) rather
    than because it's a genuine bright, stable point target — the
    amplitude percentile filter is what the original paper's own
    method uses to exclude that real false-positive case.

    Args:
        amplitude_stack: Real (N, h, w) coregistered amplitude stack.
        adi_threshold: Strict real ADI ceiling (spec default 0.25).
        adi_threshold_relaxed: Relaxed real ADI ceiling (spec default 0.40).
        use_relaxed: Use adi_threshold_relaxed instead of adi_threshold.
        amplitude_percentile: Real percentile floor for mean amplitude
            (spec: 70th percentile).

    Returns:
        PSSelectionResult with the real ADI map, boolean selection
        mask, mean amplitude map, and a real count of candidates found.
        A genuinely empty result (n_candidates=0) is returned, not
        raised, if nothing passes both criteria -- the caller (per
        this project's own "fail loudly, never silently interpolate"
        principle) is the one who decides whether to fall back to
        standard SBAS, log a warning, or raise; this function's job
        is to report the real, honest result of the selection.
    """
    np = _require_numpy()

    adi = compute_amplitude_dispersion_index(amplitude_stack)
    mean_amp = amplitude_stack.astype(np.float64).mean(axis=0).astype(np.float32)

    threshold = adi_threshold_relaxed if use_relaxed else adi_threshold
    amp_floor = np.percentile(mean_amp, amplitude_percentile)

    ps_mask = (adi < threshold) & (mean_amp > amp_floor)
    n_candidates = int(ps_mask.sum())

    if n_candidates == 0:
        logger.warning(
            "select_persistent_scatterers: 0 candidates found with ADI < %.2f "
            "and mean amplitude > %.1fth percentile. This is reported honestly, "
            "not silently substituted -- the caller should fall back to "
            "standard SBAS rather than proceed with an empty PS mask.",
            threshold,
            amplitude_percentile,
        )

    return PSSelectionResult(
        adi=adi,
        ps_mask=ps_mask,
        mean_amplitude=mean_amp,
        n_candidates=n_candidates,
        threshold_used=threshold,
    )


def temporal_coherence(
    displacement_residuals_m,
    wavelength_m: float,
):
    """
    Real temporal coherence, gamma_t, computed from real SBAS
    inversion residuals (Ferretti et al. 2001's own complex-
    exponential coherence measure, Eq. 10 in the original paper).

    gamma_t = |mean_i(exp(1j * phase_residual_i))|

    where phase_residual_i is the real per-pair residual (the
    difference between the observed interferometric phase and what
    the fitted displacement model predicts for that pair), recovered
    here from the ALREADY-COMPUTED displacement-space residuals this
    project's own SBASTimeSeries.invert()/invert_weighted() already
    produce (their real residual_rms field), converted back to
    phase via the same wavelength relationship used throughout this
    codebase (disp = wavelength/(4*pi) * phase, inverted here).

    Using the complex-exponential form specifically (rather than a
    plain RMS-of-displacement-residual measure) matters because it's
    real and robust to phase-wrapping ambiguity: a residual of exactly
    one full cycle (2*pi in phase, one wavelength/2 in LOS
    displacement) is genuinely indistinguishable from zero residual in
    real unwrapped phase, and this formulation correctly reflects
    that, where a plain RMS measure would not.

    Args:
        displacement_residuals_m: Real per-pixel, per-pair residuals
            in the SAME displacement (metres) convention as
            SBASTimeSeries's own disp_stack — NOT the aggregate
            residual_rms scalar field, the real per-pair values that
            went into computing it. Shape (n_pairs, h, w).
        wavelength_m: Real radar wavelength, matching the same value
            used for the original inversion.

    Returns:
        (h, w) real float array, gamma_t in [0, 1]. 1.0 means the
        observed phase history matches the fitted model perfectly at
        every real pair; values near 0 mean the residuals are
        effectively random relative to a full phase cycle.
    """
    np = _require_numpy()

    if displacement_residuals_m.ndim != 3:
        raise ValueError(
            f"temporal_coherence: expected real (n_pairs, h, w) residuals, "
            f"got shape {displacement_residuals_m.shape}"
        )

    phase_residual = displacement_residuals_m * (4 * np.pi / wavelength_m)
    complex_mean = np.mean(np.exp(1j * phase_residual), axis=0)
    return np.abs(complex_mean).astype(np.float32)


def refine_ps_mask_with_temporal_coherence(
    ps_result: PSSelectionResult,
    displacement_residuals_m,
    wavelength_m: float,
    coherence_threshold: float = 0.7,
) -> PSSelectionResult:
    """
    Real refinement step, applied AFTER an initial SBAS inversion —
    the spec's own stated second stage: retain only ADI-selected
    candidates whose real temporal coherence also clears
    coherence_threshold (spec default 0.7).

    A real, honest AND, not an OR: a pixel must have passed the
    amplitude-only ADI screening AND the post-inversion temporal
    coherence check. An ADI-selected pixel with poor real temporal
    coherence is a real false positive from amplitude alone (e.g. a
    bright but non-deforming-consistently target, or one whose SBAS
    residuals reveal it doesn't actually fit a coherent time series) —
    correctly demoted here rather than kept just because it looked
    good by the first, cheaper criterion.

    Args:
        ps_result: Real output of select_persistent_scatterers().
        displacement_residuals_m: Same real meaning as
            temporal_coherence()'s own parameter.
        wavelength_m: Same real meaning as temporal_coherence()'s own
            parameter.
        coherence_threshold: Real gamma_t floor (spec default 0.7).

    Returns:
        A NEW PSSelectionResult with ps_mask further restricted by the
        real temporal coherence check. adi and mean_amplitude are
        unchanged (they're real, already-computed amplitude-only
        quantities, not something this step recomputes).
    """
    gamma_t = temporal_coherence(displacement_residuals_m, wavelength_m)
    refined_mask = ps_result.ps_mask & (gamma_t >= coherence_threshold)
    n_refined = int(refined_mask.sum())

    logger.info(
        "refine_ps_mask_with_temporal_coherence: %d/%d ADI candidates "
        "retained after real temporal coherence >= %.2f (%d demoted).",
        n_refined,
        ps_result.n_candidates,
        coherence_threshold,
        ps_result.n_candidates - n_refined,
    )

    return PSSelectionResult(
        adi=ps_result.adi,
        ps_mask=refined_mask,
        mean_amplitude=ps_result.mean_amplitude,
        n_candidates=n_refined,
        threshold_used=ps_result.threshold_used,
    )


def estimate_atmospheric_phase_screen(
    per_date_values,
    ps_mask,
    spatial_filter_sigma_px: float = 20.0,
):
    """
    Real estimation of the spatially-correlated atmospheric phase
    screen (APS) from PS pixels, via 2D Gaussian low-pass spatial
    filtering — the spec's own stated method, and the real, standard
    technique (e.g. Ferretti et al. 2001; Hooper et al. 2007's
    StaMPS APS step) for separating atmosphere (spatially smooth,
    correlated over kilometres) from real deformation and noise
    (which are not smooth on that same spatial scale).

    Real, NaN-aware filtering: PS pixels are typically a small
    fraction of a real scene (this project's own real Mexico City run
    had ~0.2% solvable pixels even before PS densification), so
    per_date_values is a real, genuinely sparse array (NaN everywhere
    except real PS locations). A naive scipy.ndimage.gaussian_filter
    on an array containing NaN silently propagates NaN into every
    output pixel within the filter's support — this function instead
    filters the real value and a real coverage-count field separately
    and divides, the standard, correct way to low-pass a genuinely
    sparse field (equivalent to normalized convolution).

    Args:
        per_date_values: Real (h, w) array for ONE real acquisition
            date, values defined only at ps_mask==True locations
            (NaN or 0 elsewhere -- both are handled correctly, since
            this function masks by ps_mask explicitly rather than by
            checking for NaN).
        ps_mask: Real (h, w) boolean array, PS_selectionResult.ps_mask
            (ideally the temporal-coherence-refined one).
        spatial_filter_sigma_px: Real Gaussian filter sigma, pixels.
            Larger values assume the atmosphere is correlated over a
            wider real area (and real deformation signal narrower than
            this scale survives the filtering); the spec doesn't fix
            a specific value, so this is left as a real, tunable
            parameter rather than a silently-hardcoded assumption.

    Returns:
        (h, w) real float array, the estimated APS at every pixel
        (not just PS locations) — meant to be subtracted from the
        corresponding date's DS pixel displacement/phase before the
        final SBAS inversion, per the spec's own stated integration.
        Pixels with zero real PS coverage within reach of the filter
        are returned as NaN, not zero -- an honest "no real estimate
        available here" rather than a silently wrong zero correction.
    """
    np = _require_numpy()
    from scipy.ndimage import gaussian_filter

    if per_date_values.shape != ps_mask.shape:
        raise ValueError(
            f"estimate_atmospheric_phase_screen: per_date_values shape "
            f"{per_date_values.shape} must match ps_mask shape {ps_mask.shape}"
        )

    masked_values = np.where(ps_mask, per_date_values, 0.0).astype(np.float64)
    coverage = ps_mask.astype(np.float64)

    filtered_values = gaussian_filter(masked_values, sigma=spatial_filter_sigma_px)
    filtered_coverage = gaussian_filter(coverage, sigma=spatial_filter_sigma_px)

    with np.errstate(divide="ignore", invalid="ignore"):
        aps = filtered_values / filtered_coverage

    # Real, honest gap: where filtered_coverage is ~0 (no real PS pixel
    # within meaningful reach of the filter), there is no real estimate
    # -- NaN, not a division artifact silently treated as zero.
    no_real_coverage = filtered_coverage < 1e-6
    aps[no_real_coverage] = np.nan

    return aps.astype(np.float32)
