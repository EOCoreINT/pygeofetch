# Terrain Analysis (DEM / DSM / DTM)

Hydrological and terrain-shape analysis on any DEM — each method
individually verified against a known analytical or physically
realistic case, not just run once and eyeballed.

```python
pp = client.preprocess
terrain = pp.terrain_derivatives("dem.tif")            # slope, aspect, hillshade
twi = pp.topographic_wetness_index("dem.tif")            # flood-susceptibility screening
curv = pp.curvature("dem.tif")                            # concave/convex — water convergence vs. dispersion
tri = pp.terrain_ruggedness_index("dem.tif")              # Riley et al. 1999
sinks = pp.identify_depressions("dem.tif")                # enclosed basins with no downhill outlet
network = pp.extract_drainage_network("dem.tif")          # real channels from D8 flow accumulation
```

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
