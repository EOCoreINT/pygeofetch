"""
Regression tests for the provider-wide geometry audit: verifying real
footprint geometry is correctly populated (or safely left as bbox-only
where no better field could be confirmed) across every provider that
constructs SatelliteData, plus a real, separate crash fix found along
the way in nasa_earthdata_cloud.

Shared-pattern providers still on the original template (esa_scihub,
inpe_cbers, isro_bhuvan, google_earth_engine) are tested via one
representative, parametrized case per real provider class to keep this
maintainable. airbus_oneatlas and noaa_big_data were the first to get
full, bespoke rewrites against their real APIs -- see
test_airbus_oneatlas.py and test_noaa_big_data.py. digitalglobe,
alaska_satellite_facility, inpe_cbers (kept in the shared list -- its
rewrite preserved the same _parse_item(item) shape), jaxa_earth,
isro_bhuvan, earth_explorer_additional, and geoserver_generic later
each received their own full, bespoke rewrites against their real,
individually-researched APIs too, each with a real, different
_parse_item signature/shape (a deterministic tile system for
jaxa_earth with no _parse_item at all; a subclassed USGSProvider
method for earth_explorer_additional; real STAC/WFS/GeoJSON feature
shapes, not this file's generic flat-dict fixture, for the rest) --
see their own dedicated test files (test_digitalglobe.py,
test_alaska_satellite_facility.py, test_jaxa_earth.py,
test_earth_explorer_additional.py, test_geoserver_generic.py) for
their real, bespoke geometry-parsing tests instead. maxar_gbdx also
received a full rewrite (GBDX itself was confirmed shut down in 2022;
it now targets the real, current Discovery API) -- see
test_maxar_gbdx.py.

eodag_provider and terrabotics were both removed entirely (not
rewritten): eodag_provider only ever delegated to the third-party
`eodag` package -- a separate multi-provider aggregator, not a native
connection -- and was never even wired into the real provider registry
in the first place. terrabotics targeted a company confirmed real, but
with no public, verifiable API documentation to check its endpoints
against (a bespoke, per-contract enterprise integration, not a
standardized, publicly-documented API like every other provider here)
-- its real, confirmed domain wasn't even the one previously used
(terrabotics.co.uk, not terrabotics.earth). Removed rather than
guessed at, consistent with every other real fix in this file.
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
    ("isro_bhuvan", "IsroBhuvanProvider"),
    ("google_earth_engine", "GoogleEarthEngineProvider"),
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
    # digitalglobe, earth_explorer_additional, and geoserver_generic
    # removed: each received its own full, bespoke rewrite against its
    # real, individually-researched API (see the module docstring
    # above) -- see test_digitalglobe.py, test_earth_explorer_additional.py,
    # and test_geoserver_generic.py for their real, bespoke tests instead.
    # maxar_gbdx removed: GBDX itself was confirmed shut down in 2022;
    # it now targets the real, current Discovery API -- see
    # test_maxar_gbdx.py.
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