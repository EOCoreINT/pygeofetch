"""
OpenSARToolkit (OST) backend for PyGeoFetch SAR processing.

Requires: pip install "pygeofetch[ost]"
OST wraps ESA SNAP for production-grade SAR processing including
terrain correction via Range-Doppler geocoding with SRTM/Copernicus DEM.

IMPORTANT: OST requires SNAP to be installed separately.
Download SNAP from: https://step.esa.int/main/download/snap-download/
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("pygeofetch.sar.ost")


def _require_ost():
    try:
        import opensartoolkit as ost

        return ost
    except ImportError:
        raise ImportError(
            "OpenSARToolkit is not installed.\n"
            'Install with: pip install "pygeofetch[ost]"\n'
            "Or directly:  pip install opensartoolkit\n\n"
            "NOTE: OST also requires SNAP to be installed:\n"
            "      https://step.esa.int/main/download/snap-download/"
        )


class OSTBackend:
    """OST/SNAP-based SAR backend — production-grade terrain correction."""

    def despeckle(self, path: Path, filter="lee", window=5, output=None, **kw):
        ost = _require_ost()
        logger.info("OST despeckle: %s", path.name)
        out = Path(output) if output else path.parent / f"{path.stem}_despeckled.tif"
        try:
            ost.snap.speckle_filter(
                str(path),
                str(out),
                filter=filter.upper(),
                filter_x_size=window,
                filter_y_size=window,
            )
            logger.info("OST despeckle complete → %s", out.name)
            return out
        except Exception as exc:
            raise RuntimeError(f"OST despeckle failed: {exc}") from exc

    def calibrate(
        self, path: Path, output_type="sigma0", in_db=True, output=None, **kw
    ):
        ost = _require_ost()
        logger.info("OST calibrate: %s → %s", path.name, output_type)
        out = Path(output) if output else path.parent / f"{path.stem}_{output_type}.tif"
        try:
            ost.snap.calibration(
                str(path),
                str(out),
                polarizations=kw.get("polarizations", ["VV", "VH"]),
                output_sigma0=output_type == "sigma0",
                output_gamma0=output_type == "gamma0",
                output_beta0=output_type == "beta0",
                db=in_db,
            )
            logger.info("OST calibration complete → %s", out.name)
            return out
        except Exception as exc:
            raise RuntimeError(f"OST calibration failed: {exc}") from exc

    def terrain_correct(self, path: Path, dem="srtm", output=None, **kw):
        ost = _require_ost()
        logger.info("OST terrain correction: %s (DEM: %s)", path.name, dem)
        out = Path(output) if output else path.parent / f"{path.stem}_tc.tif"
        try:
            ost.snap.terrain_correction(
                str(path),
                str(out),
                dem=dem.upper(),
                pixel_spacing_m=kw.get("pixel_spacing_m", 10),
            )
            logger.info("Terrain correction complete → %s", out.name)
            return out
        except Exception as exc:
            raise RuntimeError(f"OST terrain correction failed: {exc}") from exc

    def flood_map(self, path: Path, threshold=-15.0, reference=None, output=None, **kw):
        from pygeofetch.sar._native import NativeSARBackend

        return NativeSARBackend().flood_map(path, threshold, reference, output, **kw)

    def coherence(self, image1: Path, image2: Path, window=7, output=None, **kw):
        from pygeofetch.sar._native import NativeSARBackend

        return NativeSARBackend().coherence(image1, image2, window, output, **kw)
