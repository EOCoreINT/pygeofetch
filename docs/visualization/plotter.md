# Plotter

```bash
pip install "pygeofetch[viz]"
```

Static plotting for almost any raster output: indices, SAR intensity,
classifications, comparisons, and 3D terrain.

## quicklook — one call for almost anything

```python
from pygeofetch.viz import Plotter

pl = Plotter()
pl.quicklook(ndvi_array)              # index -> diverging colormap
pl.quicklook("sentinel1_sigma0.tif")   # SAR -> grayscale
pl.quicklook(flood_mask)               # categorical -> auto legend
pl.quicklook(download_result)           # DownloadResult resolved automatically
```

`quicklook()` uses value-range heuristics, not format detection: few
distinct values → categorical; mostly-negative dB range → SAR
grayscale; values within `[-1, 1]` → diverging index colormap;
otherwise → continuous. All overridable via `mode=`.

## Purpose-built plots

```python
pl.plot_comparison(
    {"Baseline": ndvi_before, "Recent": ndvi_after, "Change": ndvi_change},
    per_panel_cmap={"Change": "RdBu"},
    per_panel_range={"Change": (-0.5, 0.5)},
)

pl.plot_classification(
    classified_array,
    class_labels={0: "Stable", 1: "Moderate", 2: "Severe"},
    class_colors={0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"},
)
```

`plot_raster()` and `plot_rgb()` accept in-memory numpy arrays
directly — no round-trip through disk needed.

## Other plot types

| Method | Use |
|---|---|
| `plot_raster()` | Single raster, any colormap |
| `plot_rgb()` | 3-band composite |
| `plot_multi_panel_comparison()` | N-panel grid comparison |
| `plot_timeseries()` | Line plot over dates — pairs directly with `TimeSeriesAnalyzer.zone_series()` |
| `plot_histogram()` | Value distribution |
| `plot_terrain_summary()` | 2x2 terrain analysis figure (hillshade, elevation, slope, difference maps) — see below |
| `plot_3d_terrain()` | Fast, static hillshade-draped 3D surface, no extra dependency |
| `plot_3d_terrain_interactive()` | Real interactive mesh (PyVista), rotate/zoom/pan in-browser |

## Terrain summary — a real, purpose-built 2x2 figure

```python
pl.plot_terrain_summary(
    dem="dem.tif",
    hillshade="hillshade.tif",   # optional -- auto-generated from the DEM if omitted
    slope="slope.tif",            # optional -- auto-generated from the DEM if omitted
    diffs={"copernicus": "copernicus_minus_aster.tif"},
    summit_points={"Afadjato": (-0.45, 7.15), "Aduadu": (-0.44, 7.14)},
    primary_source="ASTER",
    steep_threshold=30.0,
    output="terrain_summary.png",
)
```

Lays out a real, purpose-built figure:

```
[ Hillshade      | Elevation      ]
[ Slope          | Difference     ]
```

Pass `hillshade`/`slope` directly if you've already computed them via
{doc}`/processing/terrain`'s `terrain_derivatives()`; omit either and
this method generates it from the DEM itself. `diffs` accepts one or
more `{label: raster}` pairs — e.g. comparing two different DEM
sources over the same AOI — rendered with a diverging colormap
(`vmin_diff`/`vmax_diff`, default ±20). `summit_points` overlays
labeled markers at real `(lon, lat)` coordinates. `steep_threshold`
(degrees) controls the "% steep terrain" statistic shown alongside the
slope panel.

## 3D terrain

```python
# Fast, static, no extra dependency
pl.plot_3d_terrain("dem.tif", drape=susceptibility, drape_colormap="Blues")

# Real interactive mesh (PyVista) — exported as a standalone HTML file
pl.plot_3d_terrain_interactive(
    "dem.tif", drape=twi, drape_colormap="YlGnBu",
    output="terrain_interactive.html",
)
```

`plot_3d_terrain_interactive()` needs `pip install "pyvista[jupyter]"`
— the `[jupyter]` extra specifically for HTML export.

```{note}
Both are hardened against a real failure mode: a near-uniform
elevation field (e.g. an AOI that barely overlaps real terrain)
previously rendered as a flat, confusingly "empty" plot with no
explanation. Now raises a clear warning identifying the actual cause.
```
