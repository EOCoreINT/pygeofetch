# Searching Satellite Data

Federated search across multiple providers simultaneously. Results are
deduplicated, scored, and returned sorted.

## CLI examples

```bash
# Free providers — works immediately, no credentials
pygeofetch search run \
  --bbox "-74.1,40.6,-73.7,40.9" \
  --start-date 2024-01-01 --end-date 2024-03-01 \
  --cloud-cover 0-10 \
  --providers aws_earth,planetary_computer,element84 \
  --format table

# Save to GeoJSON for download, QGIS, or sharing
pygeofetch search run \
  --bbox "-74.1,40.6,-73.7,40.9" \
  --cloud-cover 0-5 \
  --providers aws_earth \
  --output results.geojson

# Specific satellite, sorted by cloud cover ascending
pygeofetch search run \
  --bbox "-10,35,10,55" \
  --start-date 2024-06-01 \
  --providers copernicus \
  --satellites Sentinel-2 \
  --cloud-cover 0-15 \
  --sort-by cloud_cover --sort-order asc \
  --max-results 50

# CQL2 advanced filter (Planetary Computer, Element84, AWS Earth)
pygeofetch search run \
  --bbox "-74.1,40.6,-73.7,40.9" \
  --providers planetary_computer \
  --cql2 "eo:cloud_cover < 5 AND platform = 'sentinel-2b'"

# Search using a GeoJSON geometry file
pygeofetch search run \
  --geometry-file my_area.geojson \
  --cloud-cover 0-10 \
  --providers aws_earth
```

## Search flags

| Flag | Type | Description |
|---|---|---|
| `--bbox` | string | `"minlon,minlat,maxlon,maxlat"`. Longitude first. Alternative to `--geometry-file`. |
| `--geometry-file` | path | GeoJSON file with a search AOI polygon; extracts bbox automatically. |
| `--start-date` / `--end-date` | date | `YYYY-MM-DD`. `--end-date` defaults to today if omitted. |
| `--cloud-cover` | range | `min-max` percent, e.g. `0-20`. |
| `--resolution` | range | `min-max` metres, e.g. `10-30`. |
| `--providers` | list | Comma-separated provider IDs. |
| `--satellites` | list | Comma-separated satellite names, e.g. `Sentinel-2,Landsat-8`. |
| `--processing-level` | string | e.g. `L2A`, `L2SP`. |
| `--max-results` | int | Default 100. |
| `--sort-by` | choice | `datetime`, `cloud_cover`, `score`, `satellite`. Default `datetime`. |
| `--sort-order` | choice | `asc` or `desc`. Default `desc`. |
| `--cql2` | string | CQL2 filter expression, sent as CQL2-JSON to STAC APIs. |
| `--output` | path | Save results to this file — needed before `download run --from-search`. |
| `--format` | choice | `table`, `json`, `stac`, `geojson`, `geoparquet`, `csv`, `ids`. |
| `--on-provider-failure` | choice | `skip`, `abort`, `retry`. |
| `--timeout` | int | Per-provider timeout in seconds. Default 60. |
| `--no-cache` | flag | Bypass the in-memory result cache. |
| `--validate-optical` | flag | Run optical preflight validation on results before returning them (AOI coverage, cloud cover, required bands, processing level, temporal bounds). See {doc}`/core-features/optical-validation`. Not appropriate for SAR/InSAR searches. |
| `--optical-max-cloud-cover` | float | Override the max cloud cover threshold used by `--validate-optical` (default 20.0). |
| `--optical-min-coverage` | float | Override the min AOI coverage ratio used by `--validate-optical`, 0-1 (default 0.8). |
| `--optical-required-bands` | string | Comma-separated required bands for `--validate-optical` (default `B02,B03,B04,B08,SCL`). |

## Output formats

| Format | Description |
|---|---|
| `table` | Pretty terminal table (default) — ID, date, cloud%, score, satellite |
| `json` | Full JSON array, all fields — good for scripting |
| `stac` | STAC 1.0 `ItemCollection`, compatible with `pystac-client` |
| `geojson` | `FeatureCollection` — open in QGIS, ArcGIS, Leaflet |
| `geoparquet` | GeoParquet file (requires geopandas) — best for large result sets |
| `csv` | id, provider, satellite, datetime, cloud_cover, bbox |
| `ids` | Scene IDs only, one per line — good for shell scripting |

## CQL2 filter examples

```bash
--cql2 "eo:cloud_cover < 10"
--cql2 "platform = 'sentinel-2b'"
--cql2 "eo:cloud_cover < 5 AND platform = 'sentinel-2b'"
--cql2 "s2:processing_level = 'L2A'"
--cql2 "landsat:collection_category = 'T1'"
```

CQL2 is supported by STAC-native providers: Planetary Computer,
Element84, AWS Earth.

## Python API

```python
from pygeofetch import PyGeoFetch
from pygeofetch.models import SearchQuery

pf = PyGeoFetch()

results = pf.search(
    SearchQuery(
        bbox=(-74.1, 40.6, -73.7, 40.9),   # tuple is fine — coerced to BoundingBox
        start_date="2024-01-01",
        end_date="2024-06-01",
        cloud_cover_max=20,
        max_results=50,
        sort_by="cloud_cover",
        sort_ascending=True,     # bool, not a "asc"/"desc" string
    ),
    providers=["usgs", "copernicus", "planetary_computer", "aws_earth"],
)

for r in results[:3]:
    print(r.id, r.datetime, r.cloud_cover)
```

Full `SearchQuery` field reference: {doc}`/reference/python-api`.
