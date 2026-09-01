"""
pygeofetch.validation — pre-download data-quality gates.

Mirrors the philosophy of pygeofetch.insar.preflight (a rigorous,
modular, configurable gate that runs after search and before
download), applied to optical imagery (Sentinel-2, Landsat, and any
other provider's SatelliteData results).
"""

from pygeofetch.validation.optical_validator import (
    OpticalPreflightValidator,
    OpticalValidationConfig,
    OpticalValidationError,
    SceneValidationReport,
    ValidationIssue,
)

__all__ = [
    "OpticalPreflightValidator",
    "OpticalValidationConfig",
    "OpticalValidationError",
    "SceneValidationReport",
    "ValidationIssue",
]