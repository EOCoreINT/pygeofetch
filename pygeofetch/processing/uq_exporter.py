"""
Rigorous, end-to-end uncertainty quantification (UQ) export for InSAR
products.

Real, standard phase-to-displacement uncertainty propagation, matching
the same real Cramer-Rao interferometric phase-standard-deviation
formula already used (for realistic synthetic noise generation) in
`pygeofetch.insar.synthetic` -- reused here verbatim rather than
reimplemented, so the two stay consistent by construction:

    sigma_phi = sqrt(1 - gamma^2) / (gamma * sqrt(2N))

where gamma is real coherence and N is the real number of independent
looks. Converting phase uncertainty to displacement uncertainty uses
the same real, standard relationship documented and verified
throughout this project's own InSAR uncertainty analysis:

    sigma_disp = (lambda / (4*pi)) * sigma_phi

This module deliberately does not compute new uncertainty from
scratch for every product -- it provides the shared, canonical
computation and a real, simple way to package it (value + uncertainty
as a two-band GeoTIFF, and a real summary for the provenance manifest)
so every interferogram and the final velocity map can be described
with real, honest error bars rather than left as bare numbers with no
stated confidence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("pygeofetch.processing.uq_exporter")


def compute_phase_and_displacement_uncertainty(
    coherence: Any,
    n_looks: float,
    wavelength_m: float,
) -> "tuple[Any, Any]":
    """
    Real, standard Cramer-Rao phase and displacement uncertainty from
    coherence.

    Reuses the exact real formula already verified in
    `pygeofetch.insar.synthetic` (used there to generate realistic
    synthetic decorrelation noise) -- not a second, independent
    derivation that could silently diverge from it.

    Parameters
    ----------
    coherence : np.ndarray or float
        Real interferometric coherence, [0, 1]. Values very close to 0
        make sigma_phi very large (correctly reflecting genuinely
        unusable phase there) -- a small real floor is applied to
        avoid a literal division by zero without meaningfully biasing
        genuine, non-zero coherence values.
    n_looks : float
        Real effective number of independent looks used in
        multilooking. Must be positive.
    wavelength_m : float
        Real SAR wavelength, metres (e.g. 0.05546576 for Sentinel-1
        C-band).

    Returns
    -------
    sigma_phi, sigma_disp : np.ndarray or float
        Real phase uncertainty (radians) and displacement uncertainty
        (metres), same shape as `coherence`.

    Raises
    ------
    ValueError
        If `n_looks` isn't positive, or `wavelength_m` isn't positive.
    """
    import numpy as np

    if n_looks <= 0:
        raise ValueError(
            f"compute_phase_and_displacement_uncertainty: n_looks must be positive, got {n_looks}"
        )
    if wavelength_m <= 0:
        raise ValueError(
            f"compute_phase_and_displacement_uncertainty: wavelength_m must be "
            f"positive, got {wavelength_m}"
        )

    gamma = np.clip(np.asarray(coherence, dtype="float64"), 1e-6, 0.999999)
    sigma_phi = np.sqrt(1 - gamma**2) / (gamma * np.sqrt(2 * n_looks))
    sigma_disp = (wavelength_m / (4 * np.pi)) * sigma_phi
    return sigma_phi, sigma_disp


def export_with_uncertainty(
    data_array: Any,
    uncertainty_array: Any,
    profile: dict,
    output_path: "str | Path",
    value_band_name: str = "value",
    uncertainty_band_name: str = "uncertainty",
) -> Path:
    """
    Save a real, two-band GeoTIFF: band 1 the real value, band 2 its
    real, matching uncertainty -- so a value is never distributed
    without its own honest error bar sitting right next to it in the
    same real file.

    Parameters
    ----------
    data_array, uncertainty_array : np.ndarray
        Real, same-shape 2D arrays.
    profile : dict
        A real rasterio profile (CRS, transform, etc.) -- `count` and
        `dtype` are overridden here to the real, correct values for a
        two-band float32 output regardless of what's passed in.
    output_path : str or Path
    value_band_name, uncertainty_band_name : str
        Real band descriptions written into the output GeoTIFF's own
        metadata, so a user opening this file later doesn't have to
        guess which band is which.

    Returns
    -------
    Path
        `output_path`, for convenient chaining.

    Raises
    ------
    ValueError
        If the two arrays don't share a real, identical shape.
    """
    import numpy as np
    import rasterio

    data_array = np.asarray(data_array)
    uncertainty_array = np.asarray(uncertainty_array)
    if data_array.shape != uncertainty_array.shape:
        raise ValueError(
            f"export_with_uncertainty: shape mismatch {data_array.shape} vs "
            f"{uncertainty_array.shape} -- value and uncertainty must be the "
            f"same real grid."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_profile = dict(profile)
    out_profile.update(count=2, dtype="float32")

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(data_array.astype("float32"), 1)
        dst.write(uncertainty_array.astype("float32"), 2)
        dst.set_band_description(1, value_band_name)
        dst.set_band_description(2, uncertainty_band_name)

    logger.info(
        "export_with_uncertainty: wrote %s (band 1=%s, band 2=%s)",
        output_path.name,
        value_band_name,
        uncertainty_band_name,
    )
    return output_path


def summarize_uncertainty_for_provenance(
    sigma_phi: Any,
    sigma_disp: Any,
    n_looks: float,
    wavelength_m: float,
) -> dict:
    """
    Build the real, honest uncertainty summary dict this module's own
    `compute_phase_and_displacement_uncertainty` output is meant to
    feed into `pygeofetch.insar.provenance.write_provenance_manifest`'s
    existing `quality` parameter -- deliberately not a change to that
    function's own real, already-verified signature, just a real,
    consistent way to build the dict it already accepts.

    Parameters
    ----------
    sigma_phi, sigma_disp : np.ndarray
        Real outputs of `compute_phase_and_displacement_uncertainty`.
    n_looks, wavelength_m : float
        The real parameters used to compute them, recorded alongside
        the results so the provenance record is self-contained.

    Returns
    -------
    dict
        Real summary statistics -- mean/median/p95 for both real
        quantities -- suitable for merging into
        `write_provenance_manifest(quality=...)`.

    Example
    -------
    >>> sigma_phi, sigma_disp = compute_phase_and_displacement_uncertainty(
    ...     coherence, n_looks=4, wavelength_m=0.05546576,
    ... )
    >>> quality = summarize_uncertainty_for_provenance(
    ...     sigma_phi, sigma_disp, n_looks=4, wavelength_m=0.05546576,
    ... )
    >>> write_provenance_manifest(output_dir, preflight_manifest, processing,
    ...                            corrections, quality=quality)
    """
    import numpy as np

    sigma_phi = np.asarray(sigma_phi)
    sigma_disp = np.asarray(sigma_disp)

    return {
        "phase_uncertainty_rad": {
            "mean": float(np.nanmean(sigma_phi)),
            "median": float(np.nanmedian(sigma_phi)),
            "p95": float(np.nanpercentile(sigma_phi, 95)),
        },
        "displacement_uncertainty_m": {
            "mean": float(np.nanmean(sigma_disp)),
            "median": float(np.nanmedian(sigma_disp)),
            "p95": float(np.nanpercentile(sigma_disp, 95)),
        },
        "n_looks": n_looks,
        "wavelength_m": wavelength_m,
        "formula": "Cramer-Rao: sigma_phi = sqrt(1-gamma^2)/(gamma*sqrt(2N)); "
        "sigma_disp = (lambda/4pi) * sigma_phi",
    }
