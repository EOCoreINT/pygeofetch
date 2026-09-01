# Preprocessing Engine

Atmospheric correction, cloud masking, topographic correction,
geometric operations, pan-sharpening, and multi-scene compositing —
the foundational optical-data toolkit underneath the InSAR/SAR/
Landsat/TimeSeries modules.

```python
# Individual operations
corrected = client.preprocess.atmos("scene.tif", method="dos1")
masked = client.preprocess.cloud_mask("scene.tif", method="scl", scl_band="SCL.tif")
clipped = client.preprocess.clip("scene.tif", geometry="study_area.geojson")
reproj = client.preprocess.reproject("scene.tif", crs="EPSG:4326")
```

## Available operations

| Method | Options | Description |
|---|---|---|
| `atmos()` | dos1, dos2, sen2cor, flaash, 6s, icor | Atmospheric correction |
| `cloud_mask()` | scl, fmask, threshold, ndsi | Cloud masking |
| `cloud_fill()` | — | Fill cloud gaps using a multi-date time series |
| `topo_correct()` | cosine, minnaert, c-correction | Topographic (terrain illumination) correction |
| `clip()` | bbox or GeoJSON polygon | Clip to an AOI — auto-reprojects the AOI to the raster's own CRS if they differ |
| `reproject()` | any target CRS | Reproject (EPSG:4326, UTM zones, etc.) |
| `resample()` | nearest, bilinear, cubic, lanczos | Change spatial resolution |
| `tile()` | — | Split a large raster into overlapping tiles for chunked/tiled processing |
| `pansharpen()` | brovey, ihs, gram-schmidt | Pan-sharpen multispectral with a panchromatic band |
| `mosaic()` | first, last, min, max | Merge overlapping scenes |
| `composite()` | median, mean, max, best-pixel | Multi-temporal compositing |

```{note}
**`clip()` CRS handling, verified:** a WGS84 boundary polygon (the
normal format for AOI GeoJSON) clipped against a raster in its native
UTM projection (the normal delivery format for real satellite
imagery) is automatically reprojected to match before masking —
confirmed against a real UTM Zone 30N test raster. This previously
failed silently with a near-empty intersection window rather than a
clear CRS error.
```

## `tile()` — splitting a large raster for chunked processing

```python
result = client.preprocess.tile("scene.tif", tile_size=256, overlap=32)
tiles = result.metadata["tile_paths"]
```

Splits a raster into overlapping square tiles (`tile_size` pixels,
`overlap` pixels shared between adjacent tiles) — useful for feeding
memory-constrained processing steps or a model that expects
fixed-size input chunks. `min_coverage` (default `0.1`) skips tiles
where less than that fraction of the tile is valid (non-nodata) data,
so edge tiles that are almost entirely nodata aren't written out.

```{note}
Terrain-specific operations (`terrain_derivatives()`,
`topographic_wetness_index()`, `curvature()`,
`terrain_ruggedness_index()`, `identify_depressions()`,
`extract_drainage_network()`) also live on this same `Preprocessor`
class, reachable the same way (`client.preprocess.terrain_derivatives(...)`)
— documented separately on {doc}`/processing/terrain` since they form
a coherent topic of their own (DEM/DSM/DTM analysis) rather than
general-purpose preprocessing.
```
