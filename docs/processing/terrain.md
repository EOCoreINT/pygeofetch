# Terrain Analysis (DEM / DSM / DTM)

Hydrological and terrain-shape analysis on any DEM — each method
individually verified against a known analytical or physically
realistic case, not just run once and eyeballed. All six methods live
on `client.preprocess` (the same `Preprocessor` class as
{doc}`/processing/preprocessing`, just documented separately since
terrain analysis is a coherent topic of its own).

```python
pp = client.preprocess
terrain = pp.terrain_derivatives("dem.tif")            # slope, aspect, hillshade
twi = pp.topographic_wetness_index("dem.tif")            # flood-susceptibility screening
curv = pp.curvature("dem.tif")                            # concave/convex -- water convergence vs. dispersion
tri = pp.terrain_ruggedness_index("dem.tif")              # Riley et al. 1999
sinks = pp.identify_depressions("dem.tif")                # enclosed basins with no downhill outlet
network = pp.extract_drainage_network("dem.tif")          # real channels from D8 flow accumulation
```

## `terrain_derivatives()` — slope, aspect, hillshade

```python
pp.terrain_derivatives(input, azimuth=315.0, altitude=45.0, output_dir=None)
```

Standard Horn-method gradient-based terrain analysis, computing all
three outputs from one DEM read. Correctly handles **geographic
(lat/lon) DEMs** by converting pixel size from degrees to metres at
the raster's actual latitude before computing slope — a common,
easy-to-miss source of wrong slope values when a DEM's pixel size in
degrees is used directly instead. `azimuth`/`altitude` control the
simulated sun position for hillshade (defaults: 315° = NW, 45°
altitude — the standard cartographic convention).

## `topographic_wetness_index()` — flood susceptibility

```python
pp.topographic_wetness_index(input, output=None)
```

`TWI = ln(specific catchment area / tan(slope))` — a standard, real
hydrological index (Beven & Kirkby, 1979) used in genuine flood
susceptibility and soil moisture mapping, not an approximate
substitute. High TWI marks flat, low-lying areas with a large upslope
contributing area — exactly where surface water accumulates and
standing floodwater is most likely to persist. Low TWI marks steep or
ridge terrain where water drains away quickly.

Uses a real D8 flow-direction and flow-accumulation algorithm
(O'Callaghan & Mark, 1984) — the same method GRASS GIS's
`r.watershed`, ArcGIS's Flow Accumulation, and TauDEM use — computed
directly rather than depending on an external hydrology library.

## `curvature()` — convergence vs. divergence

```python
pp.curvature(input, output=None)
```

General (Laplacian) surface curvature: `∇²z = d²z/dx² + d²z/dy²`.
Positive values mark concave (bowl-shaped) terrain, where water
converges and tends to collect; negative values mark convex
(dome/ridge-shaped) terrain, where water diverges and drains away.

```{note}
This is the simplified general/mean curvature, not the full
profile/plan curvature decomposition (Zevenbergen & Thorne, 1987) —
sufficient to identify convergence/divergence zones, but not a
substitute for that finer decomposition if your analysis specifically
needs profile vs. plan curvature separated.
```

## `terrain_ruggedness_index()` — micro-topographic complexity

```python
pp.terrain_ruggedness_index(input, output=None)
```

Riley et al. (1999)'s TRI: the root-sum-of-squares elevation
difference between each cell and its 8 neighbours. High TRI marks
complex, heterogeneous micro-topography; low TRI marks smooth, uniform
terrain.

## `identify_depressions()` — real, physically enclosed basins

```python
pp.identify_depressions(input, min_depth_m=0.1, output=None)
```

Identifies enclosed depressions (local basins with no downhill outlet)
via morphological reconstruction (fill-then-diff) — real locations
where standing water physically has nowhere to drain to, distinct from
TWI's flow-convergence zones (which still have an outlet, just a
large contributing area). `min_depth_m` filters out sub-metre
noise/artifacts rather than flagging every tiny numerical dip; output
is depression depth in metres, `0` where there's no depression.

## `extract_drainage_network()` — real stream channels

```python
pp.extract_drainage_network(input, accumulation_threshold=None, output=None)
```

Extracts a drainage/stream network by thresholding D8 flow
accumulation (the same algorithm `topographic_wetness_index()` uses)
— cells where enough upslope area drains through them to represent a
real channel, not diffuse overland flow. `accumulation_threshold` is
the minimum number of upslope cells required to be classified as a
channel; when left as `None` (default), it uses the 99th percentile
of flow accumulation across the AOI — an automatic threshold that
adapts to the DEM's actual resolution and catchment size rather than
a fixed, arbitrary cutoff.

## Verification basis

| Method | Basis | Verified against |
|---|---|---|
| `terrain_derivatives()` | Horn-method gradient | Known cone geometry — recovered slope exact |
| `topographic_wetness_index()` | Beven & Kirkby 1979, D8 flow accumulation | Sloped-plane flow accumulation exact; V-valley floor correctly ranked wetter than steep sides |
| `curvature()` | Laplacian (∇²z) | Known paraboloid bowl — constant analytical curvature recovered |
| `terrain_ruggedness_index()` | Riley et al. 1999 | Flat surface → exact 0; checkerboard → large, correct value |
| `identify_depressions()` | Morphological reconstruction (fill-then-diff) | Known synthetic 20m basin — exact depth recovered |
| `extract_drainage_network()` | D8 flow accumulation, thresholded | V-valley test — channel correctly concentrated at the true valley floor |

```{warning}
Open elevation products (SRTM, Copernicus DEM) are surface models, not
clean bare-earth DTMs — over forested or vegetated terrain, some
canopy height is baked into the "ground" elevation. A genuine Canopy
Height Model needs a real bare-earth DTM, typically from LiDAR, which
isn't available through pygeofetch's current open-data providers. This
isn't worked around by fabricating a DTM that doesn't exist — it's an
honest, stated limitation.
```
