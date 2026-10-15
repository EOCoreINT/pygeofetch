"""
Validates bbox_to_geojson_path() and preview_search_results() -- the
map-preview boilerplate consolidated out of every real project's
notebook (Mexico City, Obuasi both hand-wrote this identically).
"""

import json
from typing import ClassVar, List

from pygeofetch.insar import stack_selection


class FakeBBox:
    def __init__(self, min_lon, min_lat, max_lon, max_lat):
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = (
            min_lon,
            min_lat,
            max_lon,
            max_lat,
        )


def test_bbox_to_geojson_path_correct_geometry():
    print("=== 1. bbox_to_geojson_path produces a correct, closed polygon ring ===")
    bbox = FakeBBox(min_lon=-1.76, min_lat=6.09, max_lon=-1.61, max_lat=6.27)
    path = stack_selection.bbox_to_geojson_path(bbox, name="Test AOI")

    assert path.exists(), "file should actually be written"
    data = json.loads(path.read_text())
    print(f"  wrote: {path}")
    print(f"  content: {data}")

    assert data["type"] == "FeatureCollection"
    feature = data["features"][0]
    assert feature["properties"]["name"] == "Test AOI"
    ring = feature["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1], "ring must be closed (first point == last point)"
    assert (
        len(ring) == 5
    ), "a rectangle ring should have 5 points (4 corners + closing point)"

    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    assert min(lons) == bbox.min_lon and max(lons) == bbox.max_lon
    assert min(lats) == bbox.min_lat and max(lats) == bbox.max_lat
    print("  PASS\n")


def test_bbox_to_geojson_path_no_collisions():
    print("=== 2. Repeated calls never collide (fresh temp dir each time) ===")
    bbox = FakeBBox(-1.0, -1.0, 1.0, 1.0)
    p1 = stack_selection.bbox_to_geojson_path(bbox)
    p2 = stack_selection.bbox_to_geojson_path(bbox)
    assert p1 != p2
    assert p1.exists() and p2.exists()
    print(f"  {p1}\n  {p2}")
    print("  PASS\n")


class FakeMapViewer:
    """Records every call made to it, matching MapViewer's confirmed public API."""

    instances: ClassVar[List["FakeMapViewer"]] = []

    def __init__(self, center, zoom):
        self.center = center
        self.zoom = zoom
        self.calls = []
        FakeMapViewer.instances.append(self)

    def add_basemap(self, name):
        self.calls.append(("add_basemap", name))

    def add_search_results(self, results):
        self.calls.append(("add_search_results", results))

    def add_vector(self, path, layer_name, style):
        self.calls.append(("add_vector", path, layer_name, style))

    def show(self):
        self.calls.append(("show",))


def test_preview_search_results_orchestration():
    print(
        "=== 3. preview_search_results calls MapViewer's real, confirmed methods in the right order ==="
    )
    FakeMapViewer.instances.clear()
    bbox = FakeBBox(min_lon=-1.76, min_lat=6.09, max_lon=-1.61, max_lat=6.27)
    results = ["scene1", "scene2"]

    mv = stack_selection.preview_search_results(
        bbox,
        results,
        zoom=11,
        map_viewer_cls=FakeMapViewer,
    )

    assert mv is FakeMapViewer.instances[0], "should return the constructed viewer"
    expected_center = (
        (bbox.min_lat + bbox.max_lat) / 2,
        (bbox.min_lon + bbox.max_lon) / 2,
    )
    assert (
        mv.center == expected_center
    ), f"expected center {expected_center}, got {mv.center}"
    assert mv.zoom == 11

    call_names = [c[0] for c in mv.calls]
    print(f"  call sequence: {call_names}")
    assert call_names == [
        "add_basemap",
        "add_search_results",
        "add_vector",
        "show",
    ], "must call basemap -> search results -> AOI vector -> show, in that order"
    assert mv.calls[0][1] == "SATELLITE"
    assert mv.calls[1][1] == results
    assert mv.calls[2][2] == "AOI"
    assert mv.calls[2][3] == {"color": "yellow", "fillOpacity": 0, "weight": 3}
    print("  PASS\n")


def test_preview_search_results_show_false_lets_caller_extend():
    print("=== 4. show=False skips .show(), so a caller can add more layers first ===")
    FakeMapViewer.instances.clear()
    bbox = FakeBBox(-1.0, -1.0, 1.0, 1.0)

    mv = stack_selection.preview_search_results(
        bbox,
        [],
        show=False,
        map_viewer_cls=FakeMapViewer,
    )
    call_names = [c[0] for c in mv.calls]
    print(f"  call sequence: {call_names}")
    assert "show" not in call_names

    # Confirm the caller really can extend it afterward, exactly like
    # Obuasi's notebook adding mine/town point markers before showing.
    mv.add_vector("/fake/points.geojson", layer_name="Points of interest", style={})
    mv.show()
    assert [c[0] for c in mv.calls][-2:] == ["add_vector", "show"]
    print("  PASS\n")


def test_preview_search_results_custom_style():
    print("=== 5. Custom aoi_style overrides the default ===")
    FakeMapViewer.instances.clear()
    bbox = FakeBBox(-1.0, -1.0, 1.0, 1.0)
    custom_style = {"color": "red", "fillOpacity": 0.2, "weight": 1}

    mv = stack_selection.preview_search_results(
        bbox,
        [],
        aoi_style=custom_style,
        map_viewer_cls=FakeMapViewer,
    )
    assert mv.calls[2][3] == custom_style
    print("  PASS\n")


if __name__ == "__main__":
    test_bbox_to_geojson_path_correct_geometry()
    test_bbox_to_geojson_path_no_collisions()
    test_preview_search_results_orchestration()
    test_preview_search_results_show_false_lets_caller_extend()
    test_preview_search_results_custom_style()
    print("ALL TESTS PASSED")
