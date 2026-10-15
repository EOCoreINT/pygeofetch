from pathlib import Path
from pygeofetch import PyGeoFetch
from pygeofetch.models import SearchQuery

client = PyGeoFetch()
bbox = (-1.80, 6.10, -1.70, 6.20)  # min_lon, min_lat, max_lon, max_lat

scenes = client.search(
    SearchQuery(
        bbox=bbox,
        start_date="2024-01-01",
        end_date="2024-03-01",
        cloud_cover_max=20,
        satellites=["sentinel-2"],
    ),
    providers=["planetary_computer"],
)

# Rank by cloud cover yourself: STAC providers ignore sort_by.
scenes = sorted(scenes, key=lambda s: s.cloud_cover if s.cloud_cover is not None else 100)

results = client.stream(
    scenes[:3],
    bbox=bbox,
    destination=Path("./aoi_data"),
    bands=["B02", "B03", "B04", "B08"],
    parallel=4,
)

for r in results:
    coverage = f"{r.aoi_coverage:.0%}" if r.aoi_coverage is not None else "n/a"
    print(r.data_id, r.status, f"AOI coverage: {coverage}")