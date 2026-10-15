"""
BandStacker — seamless multi-band and time-series stacking.

Uses rioxarray when available for full xarray/GDAL integration,
falls back to rasterio + numpy for lightweight operation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("pygeofetch.processor.stacker")


class BandStacker:
    """
    Stack multiple single-band rasters into multi-band or time-series arrays.

    Args:
        use_rioxarray: Use rioxarray for xarray output (default True if installed).

    Example::

        from pygeofetch.processor import BandStacker

        stacker = BandStacker()

        # Stack RGB bands
        rgb = stacker.stack(["B02.tif", "B03.tif", "B04.tif"],
                            band_names=["Blue", "Green", "Red"])

        # Time-series stack
        ts = stacker.time_stack(
            {"2024-01": "ndvi_jan.tif", "2024-06": "ndvi_jun.tif"}
        )
    """

    def __init__(self, use_rioxarray: bool = True) -> None:
        self._use_rioxarray = use_rioxarray
        self._rioxarray = None

    def _get_rioxarray(self):
        if self._rioxarray is None:
            try:
                import rioxarray

                self._rioxarray = rioxarray
            except ImportError:
                self._rioxarray = False
        return self._rioxarray if self._rioxarray is not False else None

    def stack(
        self,
        paths: List[Union[str, Path]],
        band_names: Optional[List[str]] = None,
        output: Optional[Union[str, Path]] = None,
        as_xarray: bool = True,
    ) -> Any:
        """
        Stack multiple single-band rasters into a multi-band array.

        Args:
            paths:       List of raster paths (one band each).
            band_names:  Names for each band (e.g. ["Blue","Green","Red"]).
            output:      Optional output path for multi-band GeoTIFF.
            as_xarray:   Return xarray DataArray (default True).

        Returns:
            (n_bands, H, W) numpy array or xarray DataArray.
        """
        import numpy as np

        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        bands = []
        profile = None
        ref_shape = None

        for i, p in enumerate(paths):
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(f"Band file not found: {p}")
            with rasterio.open(p) as src:
                if profile is None:
                    profile = src.profile.copy()
                    ref_shape = (src.height, src.width)
                assert ref_shape is not None, (
                    "ref_shape is always set on the first iteration above, "
                    "since profile starts as None."
                )
                data = src.read(1).astype(np.float32)
                if data.shape != ref_shape:
                    from scipy.ndimage import zoom

                    zf = (ref_shape[0] / data.shape[0], ref_shape[1] / data.shape[1])
                    data = zoom(data, zf, order=1).astype(np.float32)
                    logger.debug("Band %d resampled to %s", i + 1, ref_shape)
                bands.append(data)

        stack = np.stack(bands, axis=0)  # (n_bands, H, W)

        # Write output if requested
        if output and profile:
            out_profile = profile.copy()
            out_profile.update(
                count=len(bands),
                dtype="float32",
                compress="deflate",
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(stack)
                if band_names:
                    for i, name in enumerate(band_names[: len(bands)], start=1):
                        dst.update_tags(i, name=name)
            logger.info("Stacked %d bands → %s", len(bands), out_path.name)

        # Return as xarray if requested and possible
        if as_xarray:
            rx = self._get_rioxarray()
            if rx:
                import xarray as xr

                names = band_names or [f"band_{i + 1}" for i in range(len(bands))]
                da = xr.DataArray(
                    stack,
                    dims=["band", "y", "x"],
                    coords={"band": names},
                    attrs={
                        "source_files": [str(p) for p in paths],
                        "crs": str(profile.get("crs")) if profile else None,
                    },
                )
                # Set spatial reference via rioxarray
                if profile and profile.get("crs") and profile.get("transform"):
                    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
                    da = da.rio.write_crs(str(profile["crs"]))
                    da = da.rio.write_transform(profile["transform"])
                return da

        return stack

    def time_stack(
        self,
        time_map: Dict[str, Union[str, Path]],
        as_xarray: bool = True,
    ) -> Any:
        """
        Stack rasters from different dates into a time-series array.

        Args:
            time_map:  OrderedDict of {date_string: raster_path}.
                       E.g. {"2024-01": "ndvi_jan.tif", "2024-06": "ndvi_jun.tif"}
            as_xarray: Return xarray DataArray with time coordinate.

        Returns:
            (n_times, H, W) array with time coordinate when as_xarray=True.
        """
        import numpy as np

        dates = list(time_map.keys())
        paths = [Path(p) for p in time_map.values()]
        arrays = []
        profile = None

        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        for p in paths:
            with rasterio.open(p) as src:
                if profile is None:
                    profile = src.profile.copy()
                arrays.append(src.read(1).astype(np.float32))

        stack = np.stack(arrays, axis=0)  # (n_times, H, W)

        if as_xarray:
            try:
                import pandas as pd
                import xarray as xr

                time_idx = pd.to_datetime(dates)
                da = xr.DataArray(
                    stack,
                    dims=["time", "y", "x"],
                    coords={"time": time_idx},
                    attrs={"source_files": [str(p) for p in paths]},
                )
                return da
            except ImportError:
                pass

        return stack

    def mosaic(
        self,
        paths: List[Union[str, Path]],
        output: Union[str, Path],
        method: str = "first",
    ) -> Path:
        """
        Mosaic multiple rasters into one (handles overlapping areas).

        Args:
            paths:   Raster paths to mosaic.
            output:  Output path.
            method:  Merge method: "first", "last", "min", "max", "mean".

        Returns:
            Path to the mosaicked output file.
        """
        try:
            import rasterio
            from rasterio.merge import merge
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        import numpy as np

        method_map = {
            "first": "first",
            "last": "last",
            "min": "min",
            "max": "max",
            "mean": "sum",  # mean needs post-processing
        }
        merge_method = method_map.get(method, "first")

        datasets = [rasterio.open(p) for p in paths]
        try:
            mosaic_arr, mosaic_transform = merge(datasets, method=merge_method)
            if method == "mean":
                count_arr, _ = merge(
                    [rasterio.open(p) for p in paths],
                    method="count",
                )
                mosaic_arr = np.where(count_arr > 0, mosaic_arr / count_arr, mosaic_arr)

            out_profile = datasets[0].profile.copy()
            out_profile.update(
                height=mosaic_arr.shape[1],
                width=mosaic_arr.shape[2],
                transform=mosaic_transform,
                compress="deflate",
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(mosaic_arr)

            logger.info("Mosaicked %d files → %s", len(paths), out_path.name)
            return out_path
        finally:
            for ds in datasets:
                ds.close()
