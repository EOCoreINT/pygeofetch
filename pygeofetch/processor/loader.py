"""
DataLoader — sensor-agnostic raster loading.

Uses EOReader when available (rich metadata, multi-sensor), falls back to
rasterio for lightweight operation. All outputs are xarray DataArrays when
xarray is available, otherwise numpy arrays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

logger = logging.getLogger("pygeofetch.processor.loader")


def _check_install(package: str, extra: str = "processor") -> None:
    raise ImportError(
        f"{package} is not installed.\n"
        f'Install with: pip install "pygeofetch[{extra}]"\n'
        f"Or directly:  pip install {package}"
    )


class DataLoader:
    """
    Sensor-agnostic data loader.

    Supports rasterio (always available with pygeofetch[geo]) and EOReader
    (optional, richer sensor metadata). Falls back gracefully.

    Args:
        use_eoreader: If True, require EOReader and use its sensor-aware loading.
                      If False (default), use rasterio with optional xarray output.
        use_xarray:   If True (default), return xarray DataArrays when possible.

    Example::

        from pygeofetch.processor import DataLoader

        # Simple load (rasterio backend)
        loader = DataLoader()
        data   = loader.load("B04.tif")                # numpy array
        data   = loader.load("B04.tif", as_xarray=True)  # xarray DataArray

        # Rich sensor-aware load (EOReader backend)
        loader = DataLoader(use_eoreader=True)
        data   = loader.load("S2B_MSIL2A_20240531...")  # auto-detects sensor
    """

    def __init__(self, use_eoreader: bool = False, use_xarray: bool = True) -> None:
        self._use_eoreader = use_eoreader
        self._use_xarray = use_xarray
        self._eoreader = None
        self._rioxarray = None

    # ── lazy loaders ──────────────────────────────────────────────────────────

    def _get_eoreader(self):
        if self._eoreader is None:
            try:
                import eoreader

                self._eoreader = eoreader
                logger.debug("EOReader %s loaded", eoreader.__version__)
            except ImportError:
                _check_install("eoreader")
        return self._eoreader

    def _get_rioxarray(self):
        if self._rioxarray is None:
            try:
                import rioxarray

                self._rioxarray = rioxarray
            except ImportError:
                _check_install("rioxarray")
        return self._rioxarray

    # ── public API ────────────────────────────────────────────────────────────

    def load(
        self,
        path: Union[str, Path],
        bands: Optional[List[str]] = None,
        as_xarray: Optional[bool] = None,
        band_index: int = 1,
    ) -> Any:
        """
        Load a raster file.

        Args:
            path:       Path to raster, or path to an EOReader product directory.
            bands:      Band names to load (EOReader backend only).
            as_xarray:  Return xarray DataArray. Defaults to self.use_xarray.
            band_index: Band to read (rasterio backend only, 1-based).

        Returns:
            numpy.ndarray or xarray.DataArray depending on backend and as_xarray.
        """
        use_xa = self._use_xarray if as_xarray is None else as_xarray

        if self._use_eoreader:
            return self._load_eoreader(Path(path), bands, use_xa)
        else:
            return self._load_rasterio(Path(path), band_index, use_xa)

    def _load_eoreader(self, path: Path, bands, use_xa: bool) -> Any:
        """Load with EOReader — sensor-aware, rich metadata."""
        from eoreader.reader import Reader

        reader = Reader()
        product = reader.open(path)

        if bands is None:
            # Load all available bands
            bands = product.get_existing_bands()
        else:
            # Resolve band names to EOReader band objects
            resolved = []
            for b in bands:
                try:
                    resolved.append(
                        getattr(product.get_existing_bands()[0].__class__, b.upper())
                    )
                except AttributeError:
                    logger.warning("Band %r not found in EOReader, skipping", b)
            bands = resolved or product.get_existing_bands()

        band_dict = product.load(bands)
        if use_xa:
            # Return dict of DataArrays
            return band_dict
        else:
            return {k: v.values for k, v in band_dict.items()}

    def _load_rasterio(self, path: Path, band_index: int, use_xa: bool) -> Any:
        """Load with rasterio, optionally wrapping in xarray."""
        try:
            import rasterio
        except ImportError:
            _check_install("rasterio", "geo")

        if use_xa:
            try:
                self._get_rioxarray()
                import rioxarray  # noqa: F401  — activates .rio accessor
                import xarray as xr

                da = xr.open_dataarray(str(path), engine="rasterio")
                return da
            except ImportError:
                pass  # fall back to numpy

        with rasterio.open(path) as src:
            import numpy as np

            if band_index > src.count:
                raise ValueError(
                    f"Band {band_index} requested but file has {src.count} bands"
                )
            data = src.read(band_index).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, float("nan"), data)
            return data

    def load_stack(self, paths: List[Union[str, Path]], as_xarray: bool = True) -> Any:
        """
        Load multiple single-band rasters and stack them.

        Returns an (n_bands, H, W) array or a time-stacked xarray Dataset.
        """
        import numpy as np

        arrays = [self._load_rasterio(Path(p), 1, False) for p in paths]
        stack = np.stack(arrays, axis=0)

        if as_xarray:
            try:
                import xarray as xr

                return xr.DataArray(
                    stack,
                    dims=["band", "y", "x"],
                    attrs={"source_files": [str(p) for p in paths]},
                )
            except ImportError:
                pass

        return stack

    @property
    def backend(self) -> str:
        return "eoreader" if self._use_eoreader else "rasterio"
