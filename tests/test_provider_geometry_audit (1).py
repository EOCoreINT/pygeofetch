"""
Regression tests for the provider-wide geometry audit: verifying real
footprint geometry is correctly populated (or safely left as bbox-only
where no better field could be confirmed) across every provider that
constructs SatelliteData, plus a real, separate crash fix found along
the way in nasa_earthdata_cloud.

Shared-pattern providers (esa_scihub, inpe_cbers, jaxa_earth,
isro_bhuvan, alaska_satellite_facility, digitalglobe,
earth_explorer_additional, geoserver_generic, google_earth_engine,
maxar_gbdx, terrabotics) all received the identical fix and are tested
via one representative, parametrized case per real provider class to
keep this maintainable. airbus_oneatlas and noaa_big_data were later
given full, bespoke rewrites against their real APIs and are no longer
part of this shared group -- see test_airbus_oneatlas.py and
test_noaa_big_data.py.
"""

import pytest

REAL_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [
        [
            [-55.75, -21.28],
            [-55.66, -21.28],
            [-55.66, -21.19],
            [-55.75, -21.19],
            [-55.75, -21.28],
        ]
    ],
}


def _make_provider(module_name, class_name):
    import importlib

    module = importlib.import_module(f"pygeofetch.providers.{module_name}")
    cls = getattr(module, class_name)
    provider = cls.__new__(cls)
    provider.PROVIDER_ID = module_name
    provider.DISPLAY_NAME = module_name
    return provider


SHARED_PATTERN_PROVIDERS = [
    ("esa_scihub", "EsaScihubProvider"),
    ("inpe_cbers", "InpeCbersProvider"),
    ("jaxa_earth", "JaxaEarthProvider"),
    ("isro_bhuvan", "IsroBhuvanProvider"),
    ("alaska_satellite_facility", "AlaskaSatelliteFacilityProvider"),
    # airbus_oneatlas removed: it no longer uses this shared generic
    # template as of its own full rewrite against the real, verified
    # OneAtlas API (which returns geometry as a GeoJSON Polygon only,
    # never a flat top-level "bbox" array) -- see the dedicated
    # test_airbus_oneatlas.py for its real, bespoke parsing tests,
    # same precedent as terrabotics below.
    # airbus_oneatlas and noaa_big_data removed: both received full,
    # bespoke rewrites against their real APIs (OneAtlas opensearch;
    # real, listable S3 buckets for NOAA) and no longer use this
    # shared generic dict-based _parse_item() shape at all -- see
    # test_airbus_oneatlas.py and test_noaa_big_data.py respectively,
    # same precedent as terrabotics below.
    ("digitalglobe", "DigitalglobeProvider"),
    ("earth_explorer_additional", "EarthExplorerAdditionalProvider"),
    ("geoserver_generic", "GeoserverGenericProvider"),
    ("google_earth_engine", "GoogleEarthEngineProvider"),
    ("maxar_gbdx", "MaxarGbdxProvider"),
]


@pytest.mark.parametrize("module_name,class_name", SHARED_PATTERN_PROVIDERS)
def test_shared_pattern_provider_populates_real_geometry(module_name, class_name):
    """Every shared-pattern provider must correctly extract real
    geometry from a real, format-accurate GeoJSON-like item, not just
    bbox -- confirming the identical fix applied across all of them."""
    try:
        provider = _make_provider(module_name, class_name)
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"Could not import {module_name}.{class_name}: {exc}")

    item = {
        "id": "scene1",
        "bbox": [-55.75, -21.28, -55.66, -21.19],
        "geometry": REAL_GEOMETRY,
        "satellite": "SENTINEL-1B",
    }
    result = provider._parse_item(item)
    assert result.geometry == REAL_GEOMETRY
    assert result.bbox == (-55.75, -21.28, -55.66, -21.19)


@pytest.mark.parametrize("module_name,class_name", SHARED_PATTERN_PROVIDERS)
def test_shared_pattern_provider_handles_missing_geometry_gracefully(
    module_name, class_name
):
    """A real item with no geometry field at all must leave geometry
    as None without raising -- the same graceful degradation as before
    this fix, just now also correctly populating it when available."""
    try:
        provider = _make_provider(module_name, class_name)
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"Could not import {module_name}.{class_name}: {exc}")

    item = {"id": "scene1", "bbox": [-55.75, -21.28, -55.66, -21.19]}
    result = provider._parse_item(item)
    assert result.geometry is None
    assert result.bbox == (-55.75, -21.28, -55.66, -21.19)


def test_terrabotics_populates_real_geometry():
    """TerraBotics uses a slightly different item shape (top-level bbox
    key check) but the same real geometry fix."""
    provider = _make_provider("terrabotics", "TerraboticsProvider")
    item = {
        "id": "scene1",
        "bbox": [-55.75, -21.28, -55.66, -21.19],
        "geometry": REAL_GEOMETRY,
        "satellite": "SENTINEL-1B",
    }
    result = provider._parse_item(item)
    assert result.geometry == REAL_GEOMETRY


def test_planet_passes_through_already_extracted_geometry():
    """Real bug fixed: Planet's parser already extracted geom from the
    real API response (used to compute bbox) but never passed it
    through to SatelliteData itself."""
    provider = _make_provider("planet", "PlanetProvider")
    feature = {
        "id": "scene1",
        "geometry": REAL_GEOMETRY,
        "properties": {"item_type": "PlanetScope", "cloud_cover": 0.1},
    }
    result = provider._parse_feature(feature)
    assert result.geometry == REAL_GEOMETRY


def test_usgs_passes_through_already_extracted_spatial_geometry():
    """Real bug fixed: USGS's real, confirmed spatialBounds/
    spatialCoverage field was already parsed for bbox but never
    passed through as geometry itself."""
    provider = _make_provider("usgs", "USGSProvider")
    scene = {
        "entityId": "scene1",
        "displayId": "scene1_display",
        "spatialBounds": REAL_GEOMETRY,
        "cloudCover": 5.0,
        "browse": [],
        "temporalCoverage": {"startDate": "2021-04-18", "endDate": "2021-04-18"},
    }
    result = provider._scene_to_satellite_data(scene, "sentinel_1")
    assert result.geometry == REAL_GEOMETRY
    assert result.bbox == (-55.75, -21.28, -55.66, -21.19)


def test_eodag_converts_real_shapely_geometry_to_geojson():
    """Real fix: EODAG's own documented API gives a real Shapely
    geometry object, converted via shapely's own mapping() utility."""
    from shapely.geometry import Polygon

    provider = _make_provider("eodag_provider", "EODAGProvider")

    class FakeProduct:
        def __init__(self):
            self.properties = {
                "id": "scene1",
                "platform": "SENTINEL-1",
                "processingLevel": "L1",
            }
            self.geometry = Polygon(
                [
                    (-55.75, -21.28),
                    (-55.66, -21.28),
                    (-55.66, -21.19),
                    (-55.75, -21.19),
                ]
            )
            self.product_type = "S1_SAR_SLC"
            self.remote_location = "https://example.com/scene1"

    result = provider._eodag_to_satellite_data(FakeProduct())
    assert result.geometry is not None
    assert result.geometry["type"] == "Polygon"
    assert result.bbox == (-55.75, -21.28, -55.66, -21.19)


def test_nasa_earthdata_cloud_does_not_crash_on_real_granule():
    """Real, severe bug fixed: dict.get() was called with an invalid
    keyword argument, crashing every single search result from this
    provider. Confirmed here it no longer does."""
    provider = _make_provider("nasa_earthdata_cloud", "NASAEarthdataCloudProvider")
    entry = {
        "id": "G123456-TEST",
        "collection_concept_id": "C1234-TEST",
        "data_center": "NASA",
        "boxes": ["-21.28 -55.75 -21.19 -55.66"],
        "title": "test granule",
        "producer_granule_id": "test",
        "links": [],
    }
    result = provider._parse_granule(entry)
    assert result.collection == "C1234-TEST"
    assert result.bbox == (-55.75, -21.28, -55.66, -21.19)
