"""
pygeofetch.optical — optical pixel offset tracking (POT).

Complements pygeofetch.insar's phase-based InSAR chain: conventional
InSAR loses coherence once ground motion exceeds roughly half a
wavelength between acquisitions (~2.8cm for Sentinel-1 C-band), which
active mining subsidence, fast landslides, and large co-seismic
deformation routinely do. Optical offset tracking sidesteps this
entirely by cross-correlating optical band imagery (not phase, and not
SAR amplitude) between two dates, directly measuring 2D horizontal
ground displacement — coarser precision than phase InSAR, but able to
measure motion phase InSAR structurally cannot.

Built on top of pygeofetch.insar.offset_tracking's real, already-
verified NCC + sub-pixel + SNR core (the same correlation mathematics
apply regardless of whether the input pixels are SAR amplitude or
optical reflectance) rather than a second, independent reimplementation
of that same real math.
"""

from pygeofetch.optical.offset_tracking import (
    OpticalOffsetResult,
    compute_horizontal_strain,
    compute_pixel_offsets,
    fuse_insar_optical,
    prepare_optical_pair,
)

__all__ = [
    "OpticalOffsetResult",
    "compute_pixel_offsets",
    "prepare_optical_pair",
    "compute_horizontal_strain",
    "fuse_insar_optical",
]