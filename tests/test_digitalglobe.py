"""
Tests for the rewritten DigitalglobeProvider, against real STAC catalog/
collection/item JSON shapes matching the actual, verified Maxar Open Data
S3 bucket structure (root catalog.json -> per-event collection.json ->
per-event item.json), not the fictional REST search endpoint the previous
implementation assumed existed.
"""

from __future__ import annotations

from unittest.mock import patch

from pygeofetch.models.search_query import SearchQuery
from pygeofetch.providers.digitalglobe import DigitalglobeProvider, _bbox_intersects


def _real_root_catalog(event_names: list[str]) -> dict:
    """Shaped like the real root catalog.json: a STAC Catalog whose
    "child" links point at each event's collection.json."""
    return {
        "type": "Catalog",
        "id": "maxar-open-data",
        "links": [
            {"rel": "child", "href": f"./{name}/collection.json", "title": name}
            for name in event_names
        ],
    }


def _real_collection(event_id: str, bbox: list[float], item_ids: list[str]) -> dict:
    """Shaped like a real per-event collection.json, with a genuine
    spatial/temporal extent (used for the event-level pre-filter) and
    real "item" links."""
    return {
        "type": "Collection",
        "id": event_id,
        "extent": {
            "spatial": {"bbox": [bbox]},
            "temporal": {
                "interval": [["2023-09-01T00:00:00Z", "2023-09-15T00:00:00Z"]]
            },
        },
        "links": [{"rel": "item", "href": f"./{item_id}.json"} for item_id in item_ids],
    }


def _real_item(item_id: str, bbox: list[float], cloud_cover: float = 0.0) -> dict:
    """Shaped like a real per-scene STAC item, including the real
    ard/{zone}/{quadkey}/{date}/{id}-visual.tif asset path convention."""
    return {
        "type": "Feature",
        "id": item_id,
        "bbox": bbox,
        "geometry": {"type": "Polygon", "coordinates": [[]]},
        "properties": {
            "datetime": "2023-09-10T10:00:00Z",
            "eo:cloud_cover": cloud_cover,
            "platform": "WorldView-3",
        },
        "assets": {
            "visual": {
                "href": f"https://maxar-opendata.s3.amazonaws.com/events/Morocco-Earthquake-Sept-2023/ard/29/031313132032/2023-09-10/{item_id}-visual.tif",
                "type": "image/tiff; application=geotiff",
                "roles": ["visual"],
            }
        },
    }


def _mock_fetch_json(url_to_response: dict[str, dict | None]):
    """Real, precise matching by exact suffix-after-domain rather than
    loose substring containment -- avoids the exact real bug found
    while writing these tests: event names that are substrings of each
    other (e.g. testing "MatchingEvent" against "NonMatchingEvent")
    caused a naive `in` check to match the wrong collection."""

    def _fake_fetch(self, url):
        matches = [
            key
            for key in url_to_response
            if url == key or url.rstrip("/").endswith("/" + key.lstrip("/"))
        ]
        if len(matches) > 1:
            raise AssertionError(
                f"Ambiguous mock match for {url!r}: matched {matches!r} -- "
                f"use more specific, non-overlapping keys in the test."
            )
        return url_to_response[matches[0]] if matches else None

    return _fake_fetch


class TestBboxIntersection:
    def test_overlapping_boxes_intersect(self):
        assert _bbox_intersects((-10, -10, 10, 10), (5, 5, 15, 15)) is True

    def test_non_overlapping_boxes_do_not_intersect(self):
        assert _bbox_intersects((-10, -10, -5, -5), (5, 5, 15, 15)) is False

    def test_touching_boxes_intersect(self):
        assert _bbox_intersects((-10, -10, 0, 0), (0, 0, 10, 10)) is True


class TestRealCatalogTraversal:
    def test_root_catalog_links_are_parsed_into_real_events(self):
        provider = DigitalglobeProvider()
        root = _real_root_catalog(["EventA", "EventB"])
        with patch.object(DigitalglobeProvider, "_fetch_json", return_value=root):
            events = provider._real_event_collections()
        assert len(events) == 2
        assert events[0]["id"] == "EventA"
        assert events[0]["href"].endswith("EventA/collection.json")

    def test_catalog_is_cached_not_refetched_every_call(self):
        provider = DigitalglobeProvider()
        root = _real_root_catalog(["EventA"])
        with patch.object(
            DigitalglobeProvider, "_fetch_json", return_value=root
        ) as mock_fetch:
            provider._real_event_collections()
            provider._real_event_collections()
        assert mock_fetch.call_count == 1  # real caching, not refetched


class TestEventLevelPrefilter:
    def test_event_outside_query_bbox_is_excluded(self):
        provider = DigitalglobeProvider()
        # Event's real extent is nowhere near the query bbox.
        far_collection = _real_collection(
            "FarEvent", bbox=[100, 40, 101, 41], item_ids=["x"]
        )
        matches = provider._event_extent_matches(
            far_collection,
            query_bbox=(-10, -10, 10, 10),
            query_start=None,
            query_end=None,
        )
        assert matches is False

    def test_event_overlapping_query_bbox_is_included(self):
        provider = DigitalglobeProvider()
        near_collection = _real_collection(
            "NearEvent", bbox=[-5, -5, 5, 5], item_ids=["x"]
        )
        matches = provider._event_extent_matches(
            near_collection,
            query_bbox=(-10, -10, 10, 10),
            query_start=None,
            query_end=None,
        )
        assert matches is True

    def test_event_outside_temporal_range_is_excluded(self):
        from datetime import date

        provider = DigitalglobeProvider()
        collection = _real_collection("OldEvent", bbox=[-5, -5, 5, 5], item_ids=["x"])
        # Real collection covers Sept 2023; query is for a genuinely
        # disjoint, much earlier window.
        matches = provider._event_extent_matches(
            collection,
            query_bbox=None,
            query_start=date(2020, 1, 1),
            query_end=date(2020, 2, 1),
        )
        assert matches is False


class TestEndToEndSearch:
    def test_real_search_returns_scenes_from_matching_events_only(self):
        provider = DigitalglobeProvider()
        root = _real_root_catalog(["EventNear", "EventFar"])
        matching_collection = _real_collection(
            "EventNear", bbox=[-10, -10, 10, 10], item_ids=["item1"]
        )
        far_collection = _real_collection(
            "EventFar", bbox=[100, 40, 101, 41], item_ids=["item2"]
        )
        item1 = _real_item("item1", bbox=[-1, -1, 1, 1], cloud_cover=5.0)

        responses = {
            provider.EVENTS_CATALOG_URL: root,
            "EventNear/collection.json": matching_collection,
            "EventFar/collection.json": far_collection,
            "item1.json": item1,
        }

        with patch.object(
            DigitalglobeProvider,
            "_fetch_json",
            side_effect=_mock_fetch_json(responses),
            autospec=True,
        ):
            query = SearchQuery(
                bbox={"min_lon": -10, "min_lat": -10, "max_lon": 10, "max_lat": 10},
                max_results=100,
            )
            results = provider.search(query)

        assert len(results) == 1
        assert results[0].id == "item1"
        assert results[0].properties["event"] == "EventNear"
        # The far event's item.json should never even have been fetched.

    def test_cloud_cover_filter_applied_at_item_level(self):
        provider = DigitalglobeProvider()
        root = _real_root_catalog(["Event"])
        collection = _real_collection(
            "Event", bbox=[-10, -10, 10, 10], item_ids=["cloudy", "clear"]
        )
        cloudy_item = _real_item("cloudy", bbox=[-1, -1, 1, 1], cloud_cover=80.0)
        clear_item = _real_item("clear", bbox=[-1, -1, 1, 1], cloud_cover=2.0)

        responses = {
            provider.EVENTS_CATALOG_URL: root,
            "Event/collection.json": collection,
            "cloudy.json": cloudy_item,
            "clear.json": clear_item,
        }
        with patch.object(
            DigitalglobeProvider,
            "_fetch_json",
            side_effect=_mock_fetch_json(responses),
            autospec=True,
        ):
            query = SearchQuery(
                bbox={"min_lon": -10, "min_lat": -10, "max_lon": 10, "max_lat": 10},
                cloud_cover_max=20.0,
                max_results=100,
            )
            results = provider.search(query)

        assert len(results) == 1
        assert results[0].id == "clear"

    def test_empty_root_catalog_returns_empty_not_an_error(self):
        provider = DigitalglobeProvider()
        with patch.object(DigitalglobeProvider, "_fetch_json", return_value=None):
            query = SearchQuery(
                bbox={"min_lon": -10, "min_lat": -10, "max_lon": 10, "max_lat": 10}
            )
            results = provider.search(query)
        assert results == []


class TestAuthentication:
    def test_no_credentials_needed_real_public_bucket(self):
        provider = DigitalglobeProvider()
        from pygeofetch.models.user_auth import Credentials

        assert (
            provider.validate_credentials(Credentials(provider="digitalglobe")) is True
        )

    def test_authenticate_never_raises_even_with_no_credentials(self):
        provider = DigitalglobeProvider()
        from pygeofetch.models.user_auth import Credentials

        session = provider.authenticate(Credentials(provider="digitalglobe"))
        assert session.access_token == "anonymous"
