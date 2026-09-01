# MapViewer

Interactive maps built on `leafmap` — search footprints, rasters,
vectors, and split comparisons.

```python
from pygeofetch.viz import MapViewer

mv = MapViewer(center=(6.198, -1.693), zoom=12)
mv.add_vector("boundary.geojson", layer_name="Study Area", style={"color": "#2c3e50", "fillOpacity": 0})
mv.add_raster("ndvi_change.tif", colormap="RdBu", vmin=-0.5, vmax=0.5, opacity=0.75)
mv.add_basemap("SATELLITE")
mv.save("interactive_map.html")
```

`add_raster()` needs `rioxarray` and `localtileserver` alongside
`leafmap` — all three are declared in the `[viz]` extra.

## Search result footprints — see before you download

```python
results = client.search(query, providers=["copernicus", "aws_earth"])

mv = MapViewer(center=(6.198, -1.693), zoom=9)
mv.add_basemap("SATELLITE")
mv.add_search_results(results)   # real footprints, real hover info
mv.show()                         # in Jupyter — or mv.save("results.html") elsewhere
```

Uses each result's real, provider-supplied geometry when available,
and falls back automatically to a bounding-box rectangle when it
isn't — safe to call against any provider's results. See
{doc}`/core-features/providers` for which providers return precise
footprint geometry today.

## Split-panel comparison

Draggable side-by-side comparison of two rasters — e.g. a DSM against
a DTM, or two independent DEM sources — backed by leafmap's own
`split_map()`.

```python
mv.add_split_comparison(
    "dsm.tif", "dtm.tif",
    left_label="DSM", right_label="DTM",
)
```

```{note}
The interactive split view depends on `localtileserver` spinning up a
real local background HTTP server, which is known to fail in some
restricted/sandboxed network environments (firewalls, some corporate
VPNs) with a `ServerDownError`. When that happens,
`add_split_comparison()` automatically falls back to a static
side-by-side comparison that needs no server at all.
```
