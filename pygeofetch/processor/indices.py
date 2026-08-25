"""
SpectralIndex — 232+ indices via spyndex, with numpy fallback for the 17 core ones.

Install spyndex for the full catalogue: pip install "pygeofetch[processor]"
Without spyndex, the 17 built-in indices (NDVI, EVI, NDWI, etc.) still work.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pygeofetch.processor.indices")

# 17 built-in formulae — no extra dependency required
_BUILTIN = {
    "NDVI": lambda b: (b["NIR"] - b["RED"]) / (b["NIR"] + b["RED"] + 1e-10),
    "EVI": lambda b: (
        2.5
        * (b["NIR"] - b["RED"])
        / (b["NIR"] + 6 * b["RED"] - 7.5 * b["BLUE"] + 1 + 1e-10)
    ),
    "SAVI": lambda b: (b["NIR"] - b["RED"]) / (b["NIR"] + b["RED"] + 0.5) * 1.5,
    "NDWI": lambda b: (b["GREEN"] - b["NIR"]) / (b["GREEN"] + b["NIR"] + 1e-10),
    "MNDWI": lambda b: (b["GREEN"] - b["SWIR1"]) / (b["GREEN"] + b["SWIR1"] + 1e-10),
    "NDBI": lambda b: (b["SWIR1"] - b["NIR"]) / (b["SWIR1"] + b["NIR"] + 1e-10),
    "NDSI": lambda b: (b["GREEN"] - b["SWIR1"]) / (b["GREEN"] + b["SWIR1"] + 1e-10),
    "NDMI": lambda b: (b["NIR"] - b["SWIR1"]) / (b["NIR"] + b["SWIR1"] + 1e-10),
    "NBR": lambda b: (b["NIR"] - b["SWIR2"]) / (b["NIR"] + b["SWIR2"] + 1e-10),
    "dNBR": lambda b: (
        ((b["NIR_PRE"] - b["SWIR2_PRE"]) / (b["NIR_PRE"] + b["SWIR2_PRE"] + 1e-10))
        - (
            (b["NIR_POST"] - b["SWIR2_POST"])
            / (b["NIR_POST"] + b["SWIR2_POST"] + 1e-10)
        )
    ),
    "BSI": lambda b: (
        ((b["SWIR1"] + b["RED"]) - (b["NIR"] + b["BLUE"]))
        / ((b["SWIR1"] + b["RED"]) + (b["NIR"] + b["BLUE"]) + 1e-10)
    ),
    "ARVI": lambda b: (
        (b["NIR"] - (2 * b["RED"] - b["BLUE"]))
        / (b["NIR"] + (2 * b["RED"] - b["BLUE"]) + 1e-10)
    ),
    "GNDVI": lambda b: (b["NIR"] - b["GREEN"]) / (b["NIR"] + b["GREEN"] + 1e-10),
    "RVI": lambda b: b["NIR"] / (b["RED"] + 1e-10),
    "VCI": lambda b: (b["NIR"] - b["RED"]) / (b["NIR"] + b["RED"] + b["BLUE"] + 1e-10),
    "CRI1": lambda b: (1 / (b["BLUE"] + 1e-10)) - (1 / (b["GREEN"] + 1e-10)),
    "PSRI": lambda b: (b["RED"] - b["BLUE"]) / (b["NIR"] + 1e-10),
}

# Band name aliases: common names → spyndex short codes
_ALIASES = {
    "RED": "R",
    "GREEN": "G",
    "BLUE": "B",
    "NIR": "N",
    "SWIR1": "S1",
    "SWIR2": "S2",
    "B02": "B",
    "B03": "G",
    "B04": "R",
    "B08": "N",
    "B11": "S1",
    "B12": "S2",
}


class SpectralIndex:
    """
    Compute spectral indices from raster bands.

    Without spyndex: 17 built-in indices available.
    With spyndex:    232+ indices from the published Awesome Spectral Indices catalogue.

    Args:
        prefer_spyndex: Use spyndex if installed (default True).

    Example::

        from pygeofetch.processor import SpectralIndex
        import numpy as np

        si   = SpectralIndex()
        red  = np.random.rand(100, 100).astype("float32")
        nir  = np.random.rand(100, 100).astype("float32")

        # Built-in index
        ndvi = si.compute("NDVI", RED=red, NIR=nir)

        # From file paths
        ndvi = si.from_files("NDVI", red="B04.tif", nir="B08.tif")

        # List available indices
        print(si.available())
    """

    def __init__(self, prefer_spyndex: bool = True) -> None:
        self._prefer_spyndex = prefer_spyndex
        self._spyndex = None

    def _get_spyndex(self):
        if self._spyndex is None:
            try:
                import spyndex

                self._spyndex = spyndex
                logger.debug(
                    "spyndex loaded — %d indices available", len(spyndex.indices)
                )
            except ImportError:
                self._spyndex = False
        return self._spyndex if self._spyndex is not False else None

    def available(self) -> List[str]:
        """Return list of all available index names."""
        sx = self._get_spyndex()
        if self._prefer_spyndex and sx:
            return sorted(sx.indices.keys())
        return sorted(_BUILTIN.keys())

    def info(self, index: str) -> Optional[Dict]:
        """Return metadata for an index (spyndex only)."""
        sx = self._get_spyndex()
        if sx and index in sx.indices:
            idx = sx.indices[index]
            return {
                "name": idx.long_name,
                "formula": idx.formula,
                "bands": idx.bands,
                "domain": idx.application_domain,
                "reference": getattr(idx, "reference", ""),
            }
        if index in _BUILTIN:
            return {
                "name": index,
                "formula": "built-in",
                "bands": [],
                "domain": "general",
            }
        return None

    def compute(self, index: str, **band_arrays: Any) -> Any:
        """
        Compute a spectral index from numpy arrays or xarray DataArrays.

        Args:
            index:        Index name (e.g. "NDVI", "EVI", "NDWI").
            **band_arrays: Band arrays keyed by band name.
                           Common names: RED, GREEN, BLUE, NIR, SWIR1, SWIR2.
                           Sentinel-2 codes: B02, B03, B04, B08, B11, B12.

        Returns:
            Same type as inputs (numpy array or xarray DataArray).

        Example::

            ndvi = si.compute("NDVI", RED=red_array, NIR=nir_array)
            evi  = si.compute("EVI",  RED=r, GREEN=g, BLUE=b, NIR=n)
        """
        import numpy as np

        # Normalise band names to uppercase
        bands = {k.upper(): v for k, v in band_arrays.items()}

        # Try spyndex first
        sx = self._get_spyndex()
        if self._prefer_spyndex and sx and index.upper() in sx.indices:
            # Map band names to spyndex short codes
            sx_bands = {}
            idx_info = sx.indices[index.upper()]
            for band in idx_info.bands:
                # Try direct match, then alias
                if band in bands:
                    sx_bands[band] = bands[band]
                else:
                    for user_key, sx_code in _ALIASES.items():
                        if sx_code == band and user_key in bands:
                            sx_bands[band] = bands[user_key]
                            break

            if len(sx_bands) == len(idx_info.bands):
                try:
                    return sx.computeIndex(index=index.upper(), params=sx_bands)
                except Exception as exc:
                    logger.warning(
                        "spyndex failed for %s: %s — using fallback", index, exc
                    )

        # Built-in fallback (never mutates _BUILTIN dict)
        _BUILTIN_UPPER = {k.upper(): v for k, v in _BUILTIN.items()}
        idx_key = index.upper()
        if idx_key in _BUILTIN_UPPER:
            # Build resolved band map — start with uppercased user keys
            mapped = {k.upper(): v for k, v in bands.items()}
            # Forward aliases: B04→RED, B08→NIR, B02→BLUE etc.
            aliases_upper = {k.upper(): v for k, v in _ALIASES.items()}
            for src_key, dst_key in list(aliases_upper.items()):
                if src_key in mapped and dst_key not in mapped:
                    mapped[dst_key] = mapped[src_key]
            # Reverse aliases: NIR→B08, RED→B04 etc.
            for src_key, dst_key in list(aliases_upper.items()):
                if dst_key in mapped and src_key not in mapped:
                    mapped[src_key] = mapped[dst_key]
            try:
                fn = _BUILTIN_UPPER[idx_key]
                result = fn(mapped).astype(np.float32)
                return result if idx_key == "DNBR" else np.clip(result, -1.0, 1.0)
            except KeyError as exc:
                raise ValueError(
                    f"Missing band for index {index}: {exc}. "
                    f"Provided: {sorted(bands.keys())}. "
                    f"Try: RED, GREEN, BLUE, NIR, SWIR1, SWIR2"
                ) from exc

        raise ValueError(
            f"Unknown spectral index: {index!r}. "
            f"Available: {self.available()[:20]}... "
            f'Install spyndex for 232+ indices: pip install "pygeofetch[processor]"'
        )

    def from_files(self, index: str, output: Optional[str] = None, **band_paths) -> Any:
        """
        Compute an index directly from raster file paths.

        Args:
            index:      Index name.
            output:     Optional output path for the result GeoTIFF.
            **band_paths: Band name → file path mappings.
                          E.g. red="B04.tif", nir="B08.tif"

        Returns:
            numpy array of the computed index.

        Example::

            ndvi = si.from_files("NDVI", red="B04.tif", nir="B08.tif",
                                 output="ndvi.tif")
        """
        from pathlib import Path

        import numpy as np

        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        band_arrays = {}
        profile = None
        ref_shape = None

        for name, path in band_paths.items():
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"Band file not found: {p}")
            with rasterio.open(p) as src:
                if profile is None:
                    profile = src.profile.copy()
                    ref_shape = (src.height, src.width)
                data = src.read(1).astype(np.float32)
                if ref_shape and data.shape != ref_shape:
                    from scipy.ndimage import zoom

                    zf = (ref_shape[0] / data.shape[0], ref_shape[1] / data.shape[1])
                    data = zoom(data, zf, order=1).astype(np.float32)
                nodata = src.nodata
                if nodata is not None:
                    data = np.where(data == nodata, np.nan, data)
                band_arrays[name.upper()] = data

        result = self.compute(index, **band_arrays)

        if output and profile:
            assert ref_shape is not None, (
                "ref_shape is always set together with profile in the "
                "band-loading loop above."
            )
            out_profile = {
                "driver": "GTiff",
                "dtype": "float32",
                "count": 1,
                "height": ref_shape[0],
                "width": ref_shape[1],
                "crs": profile.get("crs"),
                "transform": profile.get("transform"),
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "nodata": -9999.0,
            }
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **out_profile) as dst:
                arr = np.where(np.isnan(result), -9999.0, result).astype(np.float32)
                dst.write(arr[np.newaxis, :, :])
            logger.info("%s computed and saved → %s", index, out_path.name)

        return result
