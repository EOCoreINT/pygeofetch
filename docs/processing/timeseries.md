# Time Series Analysis

`TimeSeriesAnalyzer` is the link between "N dates of downloaded bands"
and actual time-series analysis: automated per-date index computation,
per-pixel trend fitting, zonal time series extraction, and anomaly
detection.

```python
from pygeofetch.processor import TimeSeriesAnalyzer

ts = TimeSeriesAnalyzer(index="NDVI")

# Computes NDVI for every date automatically, stacks with real georeferencing
stack = ts.build_index_stack({
    "2022-01-15": {"RED": "jan22_B04.tif", "NIR": "jan22_B08.tif"},
    "2023-01-15": {"RED": "jan23_B04.tif", "NIR": "jan23_B08.tif"},
    "2024-01-15": {"RED": "jan24_B04.tif", "NIR": "jan24_B08.tif"},
})

trend = ts.trend(stack)                                    # per-pixel slope/year
df = ts.zonal_timeseries(stack, "parcels.geojson")           # tidy DataFrame, all zones × all dates
anom = ts.anomaly(stack, baseline=["2022-01-15"])             # z-score vs baseline
series = ts.zone_series(stack, "parcels.geojson", zone_id=3)   # -> Plotter.plot_timeseries() directly
```

## Methods

| Method | What it does |
|---|---|
| `build_index_stack()` | Computes the configured index for every date via `SpectralIndex`, stacks into a `(time, H, W)` array with real CRS/transform preserved |
| `trend()` | Vectorized per-pixel least-squares slope-per-year — not a slow per-pixel Python loop |
| `zonal_timeseries()` | Per-zone value at every date as a tidy DataFrame |
| `zone_series()` | Single-zone convenience wrapper, shaped exactly for `Plotter.plot_timeseries()` |
| `anomaly()` | Per-pixel z-score of a target date vs. a baseline period's mean/std |

Verified against synthetic data carrying a known ground-truth trend (a
declining region vs. a stable region): `trend()` correctly
distinguished −0.158 NDVI/yr (declining) from 0.000 (stable);
`zonal_timeseries()` correctly tracked 0.429→0.111 vs. 0.429→0.409;
`anomaly()` correctly flagged a −10.3 z-score in the declining region,
and safely returns NaN rather than a garbage value when baseline
variance is genuinely zero.
