# Postprocessing

Turn a raster analysis result into GIS-ready vectors and statistics:
vectorize → smooth → regularize → zonal stats → buffer → centroids →
compress → COG.

```python
vectors = client.post.vectorize("ndvi.tif", threshold=0.3)
smoothed = client.post.smooth(vectors, tolerance=0.5)
stats = client.post.zonal_stats("ndvi.tif", "parcels.geojson", output="stats.csv")
cog = client.post.cog("scene.tif", compress="deflate")
```

## Methods

| Method | Description |
|---|---|
| `vectorize()` | Raster → vector polygons at a given threshold |
| `smooth()` | Simplify/smooth polygon boundaries |
| `zonal_stats()` | Per-polygon raster statistics (mean, min, max, etc.) |
| `buffer()` | Buffer vector geometries by a distance |
| `compress()` | Recompress a raster (LZW, DEFLATE, ZSTD) |
| `cog()` | Convert to Cloud Optimized GeoTIFF |
