# Time Series Analysis

```python
from pygeofetch.processor import TimeSeriesAnalyzer
```

`TimeSeriesAnalyzer` is the link between "N dates of downloaded bands"
and actual time-series analysis: automated per-date index computation,
per-pixel trend fitting, zonal time series extraction, and anomaly
detection. Not wired into `PyGeoFetch` — always a standalone import.

```python
ts = TimeSeriesAnalyzer(index="NDVI")

stack = ts.build_index_stack({
    "2022-01-15": {"RED": "jan22_B04.tif", "NIR": "jan22_B08.tif"},
    "2023-01-15": {"RED": "jan23_B04.tif", "NIR": "jan23_B08.tif"},
    "2024-01-15": {"RED": "jan24_B04.tif", "NIR": "jan24_B08.tif"},
})

trend = ts.trend(stack)
df = ts.zonal_timeseries(stack, "parcels.geojson")
anom = ts.anomaly(stack, baseline=["2022-01-15"])
series = ts.zone_series(stack, "parcels.geojson", zone_id=3)
```

## `build_index_stack()`

```python
ts.build_index_stack(date_bands, precomputed=False, align_grids=True)
```

Computes the configured index for every date (via the standalone
`SpectralIndex` class internally — see {doc}`/processing/spectral-indices`
for how that differs from `client.indices`) and stacks the results
into a single `(time, H, W)` array with real georeferencing preserved.

- **`date_bands`**: `{date_string: {band_name: path}}` normally, or —
  if `precomputed=True` — `{date_string: single_raster_path}` when you
  already have per-date index rasters and just want them stacked with
  the profile preserved for later zonal use.
- **`align_grids`** (default `True`): if a date's raster doesn't share
  the first date's grid (different shape, transform, or CRS),
  reproject it onto the first date's grid automatically rather than
  raising. This is a real, common case — different acquisitions of
  the same AOI can come from different scene footprints (tile
  boundaries, orbit tracks), especially for elongated or irregular
  AOIs, and end up with slightly different pixel grids after clipping
  even though they cover nearly the same area. Set `False` to restore
  strict behavior (raise on any mismatch).

Returns an `IndexTimeStack` — sorted by date, with the real building
blocks the rest of this page's methods operate on.

### `IndexTimeStack`

```python
stack.values       # (n_times, H, W) numpy array
stack.dates         # sorted list of ISO date strings, same order as .values
stack.profile       # rasterio profile of the common grid
stack.index_name    # e.g. "NDVI"

xarr = stack.as_xarray()   # xr.DataArray with a real time coordinate, CRS/transform as attrs
```

`as_xarray()` genuinely preserves CRS and transform as attrs (and a
real pandas-datetime time coordinate) — the related
`pygeofetch.processor.stacker.BandStacker.time_stack()` utility
produces a similar-looking xarray stack but only preserves
`attrs.source_files`, not full georeferencing, so it isn't a
substitute when you need the CRS/transform downstream (e.g. for
`zonal_timeseries()`).

## Analysis methods

| Method | What it does |
|---|---|
| `trend(stack)` | Vectorized per-pixel least-squares slope-per-year — not a slow per-pixel Python loop |
| `zonal_timeseries(stack, zones_path)` | Per-zone value at every date as a tidy DataFrame (needs `geopandas` — `pip install "pygeofetch[geo]"`) |
| `zone_series(stack, zones_path, zone_id)` | Single-zone convenience wrapper, shaped exactly for `Plotter.plot_timeseries()` |
| `anomaly(stack, baseline)` | Per-pixel z-score of a target date vs. a baseline period's mean/std |

```python
# Feed a single zone's series straight into a plot
from pygeofetch.viz import Plotter

series = ts.zone_series(stack, "parcels.geojson", zone_id=3)
Plotter().plot_timeseries(series)
```

## Verification basis

Verified against synthetic data carrying a known ground-truth trend (a
declining region vs. a stable region):

- `trend()` correctly distinguished −0.158 NDVI/yr (declining) from
  0.000 (stable).
- `zonal_timeseries()` correctly tracked 0.429→0.111 (declining) vs.
  0.429→0.409 (stable).
- `anomaly()` correctly flagged a −10.3 z-score in the declining
  region, and safely returns NaN — not a garbage value — when
  baseline variance is genuinely zero.
