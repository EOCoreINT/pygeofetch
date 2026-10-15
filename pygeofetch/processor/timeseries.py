"""
TimeSeriesAnalyzer — the missing link between per-date raster downloads and
actual time-series analysis.

Before this module, getting from "N dates of downloaded bands" to "NDVI
trend for my AOI" required manually looping: compute the index yourself
for each date, write each result to disk, call zonal_stats() once per
date, and hand-assemble the results into something plottable. The
existing pieces (SpectralIndex, BandStacker.time_stack(),
Plotter.plot_timeseries()) are each real and useful, but nothing tied
them together into an actual time-series workflow — BandStacker only
stacks rasters that are already single-band index values; it never
computes the index; Plotter only plots values that are already extracted
into a plain dict.

This module closes that gap:

    from pygeofetch.processor import TimeSeriesAnalyzer

    ts = TimeSeriesAnalyzer(index="NDVI")
    stack = ts.build_index_stack({
        "2024-01-15": {"RED": "jan_B04.tif", "NIR": "jan_B08.tif"},
        "2024-06-20": {"RED": "jun_B04.tif", "NIR": "jun_B08.tif"},
        "2024-11-03": {"RED": "nov_B04.tif", "NIR": "nov_B08.tif"},
    })

    trend = ts.trend(stack)                          # per-pixel slope/year
    df    = ts.zonal_timeseries(stack, "parcels.geojson")   # tidy DataFrame
    anom  = ts.anomaly(stack, baseline=["2024-01-15"])      # z-score vs baseline
    series = ts.zone_series(stack, "parcels.geojson", zone_id=3)  # -> Plotter.plot_timeseries()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("pygeofetch.processor.timeseries")


def _require_numpy():
    try:
        import numpy as np

        return np
    except ImportError:
        raise ImportError('numpy required: pip install "pygeofetch[geo]"')


def _require_rasterio():
    try:
        import rasterio

        return rasterio
    except ImportError:
        raise ImportError('rasterio required: pip install "pygeofetch[geo]"')


def _require_geopandas():
    try:
        import geopandas as gpd

        return gpd
    except ImportError:
        raise ImportError('geopandas required: pip install "pygeofetch[geo]"')


class IndexTimeStack:
    """
    Result of build_index_stack(): a (time, H, W) index array plus the
    georeferencing needed for zonal extraction, which BandStacker's plain
    xarray output does not itself carry.
    """

    def __init__(self, values: Any, dates: List[str], profile: dict, index_name: str):
        self.values = values  # (n_times, H, W) numpy array
        self.dates = dates  # sorted list of ISO date strings, same order as values
        self.profile = profile  # rasterio profile of the (common) grid
        self.index_name = index_name

    def __repr__(self) -> str:
        return (
            f"IndexTimeStack({self.index_name}, {len(self.dates)} dates, "
            f"shape={self.values.shape})"
        )

    def as_xarray(self) -> Any:
        """Convert to an xarray DataArray with a real time coordinate, CRS
        and transform preserved as attrs (unlike BandStacker.time_stack(),
        which only preserves attrs.source_files)."""
        import pandas as pd
        import xarray as xr

        return xr.DataArray(
            self.values,
            dims=["time", "y", "x"],
            coords={"time": pd.to_datetime(self.dates)},
            attrs={
                "index": self.index_name,
                "crs": str(self.profile.get("crs")),
                "transform": self.profile.get("transform"),
            },
        )


class TimeSeriesAnalyzer:
    """
    Compute a spectral index across multiple dates and analyze it as an
    actual time series — per-pixel trend, zonal (per-polygon) series
    extraction across dates, and anomaly detection against a baseline
    period.

    Args:
        index: Index name passed to SpectralIndex.compute() for each date
              (e.g. "NDVI", "NDWI", "NBR"). Ignored if you pass
              already-computed single-band rasters to build_index_stack()
              via the `precomputed=True` path.
    """

    def __init__(self, index: str = "NDVI") -> None:
        self.index_name = index

    # ── building the stack ───────────────────────────────────────────────────

    def build_index_stack(
        self,
        date_bands: Dict[str, Dict[str, Union[str, Path]]],
        precomputed: bool = False,
        align_grids: bool = True,
    ) -> IndexTimeStack:
        """
        Compute the configured index for every date and stack the results
        into a single (time, H, W) array with real georeferencing.

        Args:
            date_bands: {date_string: {band_name: path}} for each date, OR
                       (if precomputed=True) {date_string: single_raster_path}
                       when you already have per-date index rasters and
                       just want them stacked with proper CRS/transform
                       preserved (BandStacker.time_stack() does this too,
                       but without keeping the profile for zonal use).
            precomputed: Treat date_bands values as ready-made single-band
                       index rasters instead of raw band dicts.
            align_grids: If a date's raster doesn't share the first date's
                       grid (different shape, transform, or CRS), reproject
                       it onto the first date's grid automatically rather
                       than raising. Default True. This is a real, common
                       case: different acquisitions of the same AOI can
                       come from different satellite scene footprints
                       (tile boundaries, orbit tracks), especially for
                       elongated or irregularly-shaped AOIs, and end up
                       with slightly different pixel grids after clipping
                       even though they cover (nearly) the same area. Set
                       False to restore the old strict behaviour (raise on
                       any mismatch) if you specifically want that.

        Returns:
            IndexTimeStack — sorted by date, with .as_xarray() available.
        """
        np = _require_numpy()
        rasterio = _require_rasterio()

        from pygeofetch.processor.indices import SpectralIndex

        dates = sorted(date_bands.keys())
        arrays = []
        profile = None
        ref_shape = None
        ref_transform = None
        ref_crs = None

        si = SpectralIndex(prefer_spyndex=False) if not precomputed else None

        def _read_aligned(path, date, label):
            nonlocal profile, ref_shape, ref_transform, ref_crs
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
                if profile is None:
                    profile = src.profile.copy()
                    ref_shape = arr.shape
                    ref_transform = src.transform
                    ref_crs = src.crs
                    return arr

                same_grid = (
                    arr.shape == ref_shape
                    and src.transform == ref_transform
                    and src.crs == ref_crs
                )
                if same_grid:
                    return arr

                if not align_grids:
                    raise ValueError(
                        f"{date}/{label}: grid {arr.shape} doesn't match the "
                        f"first date's grid {ref_shape}. All dates must "
                        f"share the same grid, or call with align_grids=True "
                        f"(the default) to align automatically."
                    )

                from rasterio.warp import Resampling, reproject

                logger.info(
                    "%s/%s: grid %s differs from the first date's %s — "
                    "reprojecting onto the first date's grid automatically "
                    "(align_grids=True). This is expected when different "
                    "acquisitions come from different scene footprints over "
                    "the same AOI.",
                    date,
                    label,
                    arr.shape,
                    ref_shape,
                )
                aligned = np.full(ref_shape, np.nan, dtype=np.float32)
                reproject(
                    source=arr,
                    destination=aligned,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                )
                return aligned

        for date in dates:
            entry = date_bands[date]

            if precomputed:
                arr = _read_aligned(Path(entry), date, "index")  # type: ignore[arg-type]
            else:
                assert si is not None, (
                    "si is only None when precomputed=True, and this "
                    "branch only runs when precomputed is falsy."
                )
                band_arrays = {}
                for band_name, band_path in entry.items():
                    band_arrays[band_name.upper()] = _read_aligned(
                        Path(band_path), date, band_name
                    )
                arr = np.asarray(
                    si.compute(self.index_name, **band_arrays), dtype=np.float32
                )

            arrays.append(arr)

        stack = np.stack(arrays, axis=0)
        logger.info(
            "Built %s time stack: %d dates, shape %s",
            self.index_name,
            len(dates),
            stack.shape,
        )
        assert profile is not None, (
            "profile is set on the first _read_aligned() call inside the "
            "dates loop above, which always runs at least once for a "
            "non-empty dates list."
        )
        return IndexTimeStack(stack, dates, profile, self.index_name)

    # ── per-pixel trend ──────────────────────────────────────────────────────

    def trend(self, stack: IndexTimeStack) -> Any:
        """
        Per-pixel linear trend (slope per year), computed via vectorized
        least-squares — not a per-pixel Python loop, so this stays fast
        even on large stacks.

        Returns:
            (H, W) array of slope-per-year. Positive = increasing over
            time (e.g. vegetation recovery), negative = decreasing
            (e.g. degradation/clearance). NaN input pixels propagate as
            NaN in the output.
        """
        np = _require_numpy()

        import pandas as pd

        t_years = (
            pd.to_datetime(stack.dates) - pd.to_datetime(stack.dates[0])
        ).days / 365.25
        t_years = np.asarray(t_years, dtype=np.float64)

        n_times, h, w = stack.values.shape
        y = stack.values.reshape(n_times, -1)  # (n_times, H*W)

        valid = np.isfinite(y)
        # Vectorized per-pixel least-squares slope via the standard
        # covariance formula, masking NaNs per-pixel rather than requiring
        # every pixel to have data at every date.
        t_mean = np.where(valid, t_years[:, None], np.nan)
        t_mean = np.nanmean(t_mean, axis=0)
        y_mean = np.nanmean(np.where(valid, y, np.nan), axis=0)

        t_dev = t_years[:, None] - t_mean[None, :]
        y_dev = y - y_mean[None, :]
        t_dev = np.where(valid, t_dev, 0.0)
        y_dev = np.where(valid, y_dev, 0.0)

        numerator = np.nansum(t_dev * y_dev, axis=0)
        denominator = np.nansum(t_dev**2, axis=0)

        n_valid = valid.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(
                (denominator > 0) & (n_valid >= 2), numerator / denominator, np.nan
            )

        return slope.reshape(h, w)

    # ── zonal (per-polygon) time series ──────────────────────────────────────

    def zonal_timeseries(
        self,
        stack: IndexTimeStack,
        zones: Union[str, Path],
        zone_id_field: Optional[str] = None,
        stat: str = "mean",
    ) -> Any:
        """
        Extract a per-zone value at every date — the "give me the time
        series for this AOI" operation that previously required manually
        calling zonal_stats() once per date and assembling the results
        yourself.

        Args:
            stack:         Result of build_index_stack().
            zones:         Vector file with zone polygons.
            zone_id_field: Column to use as the zone identifier in the
                          output. Defaults to the row index if None.
            stat:          "mean", "median", "min", "max", or "std".

        Returns:
            A tidy pandas DataFrame with columns [zone_id, date, value] —
            one row per (zone, date) pair, ready for groupby/pivot,
            plotting, or export to CSV.
        """
        np = _require_numpy()
        _require_rasterio()  # dependency guard — clear error if rasterio missing
        gpd = _require_geopandas()
        import pandas as pd
        from rasterio.features import geometry_mask

        stat_fns = {
            "mean": np.nanmean,
            "median": np.nanmedian,
            "min": np.nanmin,
            "max": np.nanmax,
            "std": np.nanstd,
        }
        if stat not in stat_fns:
            raise ValueError(f"stat must be one of {list(stat_fns)}, got {stat!r}")
        stat_fn = stat_fns[stat]

        gdf = gpd.read_file(zones)
        target_crs = stack.profile.get("crs")
        if target_crs and gdf.crs and str(gdf.crs) != str(target_crs):
            gdf = gdf.to_crs(target_crs)

        transform = stack.profile["transform"]
        out_shape = stack.values.shape[1:]

        rows = []
        for i, geom in enumerate(gdf.geometry):
            zone_id = gdf.iloc[i][zone_id_field] if zone_id_field else i
            try:
                mask = ~geometry_mask(
                    [geom], out_shape=out_shape, transform=transform, invert=False
                )
            except Exception as exc:
                logger.warning("zonal_timeseries: skipping zone %s: %s", zone_id, exc)
                continue

            if not mask.any():
                logger.warning(
                    "zonal_timeseries: zone %s does not overlap the raster grid",
                    zone_id,
                )
                continue

            for t_idx, date in enumerate(stack.dates):
                pixel_values = stack.values[t_idx][mask]
                finite = pixel_values[np.isfinite(pixel_values)]
                value = float(stat_fn(finite)) if finite.size else float("nan")
                rows.append({"zone_id": zone_id, "date": date, "value": value})

        return pd.DataFrame(rows)

    def zone_series(
        self,
        stack: IndexTimeStack,
        zones: Union[str, Path],
        zone_id: Any,
        zone_id_field: Optional[str] = None,
        stat: str = "mean",
    ) -> Dict[str, float]:
        """
        Convenience wrapper around zonal_timeseries() for a single zone,
        returned as {date: value} — the exact shape Plotter.plot_timeseries()
        expects, so this plugs directly into existing visualization:

            series = ts.zone_series(stack, "parcels.geojson", zone_id=3)
            Plotter().plot_timeseries(series, title="NDVI — Parcel 3")
        """
        df = self.zonal_timeseries(stack, zones, zone_id_field=zone_id_field, stat=stat)
        zone_df = df[df["zone_id"] == zone_id].sort_values("date")
        if zone_df.empty:
            raise ValueError(f"No data found for zone_id={zone_id!r}")
        return dict(zip(zone_df["date"], zone_df["value"]))

    # ── anomaly detection ────────────────────────────────────────────────────

    def anomaly(
        self,
        stack: IndexTimeStack,
        baseline: List[str],
        target: Optional[str] = None,
    ) -> Any:
        """
        Per-pixel z-score of a target date against a baseline period's
        mean and standard deviation — flags where/how much a scene departs
        from what's "normal" for this location, rather than just showing
        raw index values.

        Args:
            stack:    Result of build_index_stack().
            baseline: Dates (must be in stack.dates) defining the baseline
                     "normal" period, e.g. several dry-season dates before
                     an event of interest.
            target:   Date to compute the anomaly for. Defaults to the
                     most recent date in the stack.

        Returns:
            (H, W) z-score array. |z| > ~2 is a common rule-of-thumb
            threshold for "notably anomalous", but treat this as a
            starting point, not a validated statistical claim for your
            specific index/region — always sanity-check against a map.
        """
        np = _require_numpy()

        missing = set(baseline) - set(stack.dates)
        if missing:
            raise ValueError(f"Baseline dates not found in stack: {missing}")

        baseline_idx = [stack.dates.index(d) for d in baseline]
        baseline_stack = stack.values[baseline_idx]

        target_date = target or stack.dates[-1]
        if target_date not in stack.dates:
            raise ValueError(f"Target date {target_date!r} not found in stack")
        target_arr = stack.values[stack.dates.index(target_date)]

        baseline_mean = np.nanmean(baseline_stack, axis=0)
        baseline_std = np.nanstd(baseline_stack, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(
                baseline_std > 0, (target_arr - baseline_mean) / baseline_std, np.nan
            )

        return z
