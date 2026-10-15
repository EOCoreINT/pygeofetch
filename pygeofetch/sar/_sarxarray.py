"""
sarxarray backend for PyGeoFetch SAR processing.

Requires: pip install "pygeofetch[sar]"
sarxarray provides xarray-native SAR processing compatible with Dask for
large-scale / out-of-memory processing.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("pygeofetch.sar.sarxarray")


def _require_sarxarray():
    """Import sarxarray, raising a clear ImportError if not installed or blocked."""
    import sys

    _msg = (
        "sarxarray is not installed.\n"
        'Install with: pip install "pygeofetch[sar]"\n'
        "Or directly:  pip install sarxarray"
    )
    # patch.dict("sys.modules", {"sarxarray": None}) blocks the import
    # by setting the entry to None — detect and raise clearly
    if "sarxarray" in sys.modules and sys.modules["sarxarray"] is None:
        raise ImportError(_msg)
    try:
        import sarxarray  # noqa: PLC0415

        return sarxarray
    except ImportError:
        raise ImportError(_msg)


class SARXarrayBackend:
    """sarxarray-based SAR backend — xarray-native, Dask-compatible."""

    def despeckle(self, path: Path, filter="lee", window=5, output=None, **kw):
        logger.info(
            "sarxarray despeckle: %s (filter=%s, window=%d)", path.name, filter, window
        )
        try:
            sx = _require_sarxarray()
            import rioxarray  # noqa: F401
            import xarray as xr

            da = xr.open_dataarray(str(path), engine="rasterio")
            if hasattr(sx, "speckle_filter"):
                filtered = sx.speckle_filter(da, filter_type=filter, window_size=window)
                if output:
                    filtered.rio.to_raster(output, compress="deflate", tiled=True)
                return filtered
        except Exception as exc:
            logger.warning(
                "sarxarray despeckle failed (%s), falling back to native", exc
            )
        from pygeofetch.sar._native import NativeSARBackend

        return NativeSARBackend().despeckle(path, filter, window, output, **kw)

    def calibrate(
        self, path: Path, output_type="sigma0", in_db=True, output=None, **kw
    ):
        _require_sarxarray()
        from pygeofetch.sar._native import NativeSARBackend

        return NativeSARBackend().calibrate(path, output_type, in_db, output, **kw)

    def terrain_correct(self, path: Path, dem="srtm", output=None, **kw):
        raise NotImplementedError(
            "Terrain correction is not available in the sarxarray backend. "
            "Use SARProcessor(backend='ost') with SNAP installed."
        )

    def flood_map(self, path: Path, threshold=-15.0, reference=None, output=None, **kw):
        from pygeofetch.sar._native import NativeSARBackend

        return NativeSARBackend().flood_map(path, threshold, reference, output, **kw)

    def coherence(self, image1: Path, image2: Path, window=7, output=None, **kw):
        from pygeofetch.sar._native import NativeSARBackend

        return NativeSARBackend().coherence(image1, image2, window, output, **kw)
