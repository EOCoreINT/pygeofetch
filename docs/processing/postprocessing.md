# Postprocessing

Turn a raster analysis result into GIS-ready vectors and statistics.

```python
vectors = client.post.vectorize("ndvi.tif", threshold=0.3)
smoothed = client.post.smooth(vectors, tolerance=0.5)
regular = client.post.regularize(smoothed)
stats = client.post.zonal_stats("ndvi.tif", "parcels.geojson", output="stats.csv")
cog = client.post.cog("scene.tif", compress="deflate")
```

## Raster → vector

```python
result = client.post.vectorize(
    "classification.tif", band=1, threshold=0.5,
    format="geojson", min_area=100.0,
)
```

Converts a raster (e.g. a classification or binary mask) to vector
polygons. `threshold` applies a binary cutoff before vectorizing
(pixels ≥ threshold → 1, else 0) — omit it to vectorize the raster's
existing discrete values directly. `format` accepts `"geojson"`,
`"gpkg"`, or `"shp"`. `min_area` discards polygons smaller than that
area in CRS units² — useful for dropping single-pixel noise from a
noisy classification.

## Vector cleanup

| Method | What it does |
|---|---|
| `smooth(input, tolerance=1.0, method="simplify")` | Simplify/smooth boundaries — `"simplify"` (Douglas-Peucker) or `"buffer"` (buffer then un-buffer, rounds sharp corners) |
| `regularize(input, corner_threshold_deg=30.0)` | Orthogonalize building footprints or irregular polygons — snaps edges to right angles where the corner angle is within `corner_threshold_deg` of 90° |
| `buffer(input, distance=10.0, cap_style="round", join_style="round")` | Buffer geometries by `distance` (CRS units). `cap_style`: `"round"`/`"flat"`/`"square"`. `join_style`: `"round"`/`"mitre"`/`"bevel"` |
| `centroids(input)` | Extract centroid points from polygon/line geometries |
| `add_geometry_metrics(input)` | Adds `area_m2`, `perimeter_m`, and `compactness` (Polsby-Popper — 1.0 = a perfect circle) columns |

```python
# A realistic building-extraction cleanup chain
buildings = client.post.vectorize("building_mask.tif", threshold=0.5, min_area=20.0)
smoothed = client.post.smooth(buildings, tolerance=0.5)
regular = client.post.regularize(smoothed, corner_threshold_deg=25.0)
with_metrics = client.post.add_geometry_metrics(regular)
```

## Zonal statistics

```python
result = client.post.zonal_stats(
    "ndvi.tif", "parcels.geojson", output="stats.csv",
    stats=["mean", "median", "std"], band=1, all_touched=False,
)
```

Computes `mean`, `median`, `min`, `max`, `std`, `count` per zone by
default; pass `stats` to compute a subset. `all_touched=True` includes
any pixel touching a zone's boundary, not just pixels whose center
falls inside it — matters for small zones relative to pixel size.

## Compression & Cloud Optimized GeoTIFF

```python
compressed = client.post.compress("scene.tif", method="lzw", zlevel=6)
print(compressed.metadata["ratio"])   # real measured compression ratio, e.g. 2.4

cog_result = client.post.cog("scene.tif", compress="deflate")
```

`compress()` applies lossless GeoTIFF compression (`"lzw"`,
`"deflate"`, `"zstd"`, `"packbits"`) and reports the real, measured
compression ratio and file sizes in `result.metadata` — not an
estimate. `cog()` converts to a real Cloud Optimized GeoTIFF
(internal tiling + overviews), the format most cloud-native raster
tools (rasterio, GDAL VSI, STAC-based viewers) expect for efficient
partial reads.

## Vector ↔ vector overlay

Combine, merge, and relate two vector layers by their real geometry —
genuinely different from `zonal_stats` above, which relates a
*raster* to vector zones. These three are real, thin wrappers around
GeoPandas's own overlay engine, so behavior matches
`geopandas.overlay`/`dissolve`/`sjoin` if you already know them.

```python
# Which parcels does the flood extent actually touch?
flooded = client.post.overlay("flood_extent.geojson", "parcels.geojson",
                               output="flooded_parcels.geojson", how="intersection")

# Merge individual parcels into zones by land use
zones = client.post.dissolve("parcels.geojson", output="zones.geojson", by="land_use")

# Which zone does each sample point fall inside?
joined = client.post.spatial_join("sample_points.geojson", "zones.geojson",
                                   output="points_with_zone.geojson", predicate="within")
```

| Method | What it does |
|---|---|
| `overlay(left, right, output, how="intersection")` | Combine two layers by geometric relationship — `"intersection"`, `"union"`, `"difference"` (left minus right), `"symmetric_difference"`, or `"identity"` |
| `dissolve(input, output, by=None, aggfunc="first")` | Merge polygons sharing the same value in the `by` column into single geometries; omit `by` to merge everything into one |
| `spatial_join(left, right, output, how="inner", predicate="intersects")` | Attach `right`'s attributes to `left` by real spatial relationship (`"intersects"`, `"within"`, `"contains"`, `"touches"`, `"crosses"`, `"overlaps"`) instead of a shared key column |

All three reproject `right` to match `left`'s real CRS automatically
when the two differ — checked and handled, not left to silently
produce a wrong or empty result. All three require an explicit
`output` path, since the right vector format (GeoJSON, Shapefile,
GeoPackage) depends on what you're doing with the result, unlike
raster outputs which have a sensible default naming convention.

**A common mistake worth flagging directly**: `predicate="intersects"`
(the default for `spatial_join`) also matches features that merely
touch at an edge. If you need points genuinely *inside* a polygon,
not on its boundary, use `predicate="within"` instead.

