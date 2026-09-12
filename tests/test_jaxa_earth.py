"""
Tests for the rewritten JaxaEarthProvider, against the real, confirmed
AW3D30 tile ID format and the free OpenTopography mirror -- not the
fictional REST search the previous version assumed, and not a
temporal/cloud-cover filter that never made sense for a static DSM.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pygeofetch.models.search_query import SearchQuery
from pygeofetch.providers.jaxa_earth import (
    JaxaEarthProvider,
    _tile_id,
    _tiles_covering_bbox,
)


class TestTileIdFormat:
    """Real, confirmed independently from two separate sources (a
    Microsoft AI for Earth storage doc and a real JAXA file-naming
    description) -- both 3-digit latitude AND longitude, which a first
    pass at this got wrong (2-digit latitude) before being caught here."""

    def test_northern_eastern_hemisphere(self):
        assert _tile_id(35, 138) == "N035E138"

    def test_northern_western_hemisphere(self):
        assert _tile_id(15, -11) == "N015W011"

    def test_southern_western_hemisphere(self):
        assert _tile_id(-57, -33) == "S057W033"

    def test_latitude_is_three_digits_not_two(self):
        """The specific real bug caught during development: an earlier
        version zero-padded latitude to 2 digits, producing 'N35E138'
        instead of the real, confirmed 'N035E138'."""
        tile_id = _tile_id(35, 138)
        assert len(tile_id) == 8  # 1 + 3 + 1 + 3
        assert tile_id[1:4] == "035"

    def test_equator_and_prime_meridian(self):
        assert _tile_id(0, 0) == "N000E000"


class TestTilesCoveringBbox:
    def test_single_tile_bbox(self):
        tiles = _tiles_covering_bbox(138.2, 35.2, 138.8, 35.8)
        assert tiles == ["N035E138"]

    def test_bbox_spanning_two_tiles_east_west(self):
        tiles = _tiles_covering_bbox(138.5, 35.2, 139.5, 35.8)
        assert set(tiles) == {"N035E138", "N035E139"}

    def test_bbox_spanning_four_tiles(self):
        tiles = _tiles_covering_bbox(138.5, 35.5, 139.5, 36.5)
        assert set(tiles) == {"N035E138", "N035E139", "N036E138", "N036E139"}


class TestSearchIsDeterministicNotAQuery:
    def test_search_requires_a_bbox(self):
        provider = JaxaEarthProvider()
        query = SearchQuery()
        results = provider.search(query)
        assert results == []

    def test_search_returns_real_tile_filenames(self):
        provider = JaxaEarthProvider()
        query = SearchQuery(
            bbox={"min_lon": 138.2, "min_lat": 35.2, "max_lon": 138.8, "max_lat": 35.8}
        )
        results = provider.search(query)
        assert len(results) == 1
        assert results[0].id == "N035E138"
        assert (
            results[0]
            .data_assets["dsm"]
            .href.endswith("N035E138/ALPSMLC30_N035E138_DSM.tif")
        )

    def test_no_date_or_cloud_cover_in_results_real_static_dataset(self):
        provider = JaxaEarthProvider()
        query = SearchQuery(
            bbox={"min_lon": 138.2, "min_lat": 35.2, "max_lon": 138.8, "max_lat": 35.8}
        )
        results = provider.search(query)
        assert results[0].datetime is None
        assert results[0].cloud_cover is None

    def test_no_network_call_made_for_search(self):
        """Real, deliberate design: search() is pure computation, no
        HTTP request at all -- confirmed by patching httpx.get and
        checking it's never called."""
        provider = JaxaEarthProvider()
        with patch("httpx.get") as mock_get:
            query = SearchQuery(
                bbox={
                    "min_lon": 138.2,
                    "min_lat": 35.2,
                    "max_lon": 138.8,
                    "max_lat": 35.8,
                }
            )
            provider.search(query)
        mock_get.assert_not_called()


class TestNoAuthRequired:
    def test_requires_auth_is_false(self):
        provider = JaxaEarthProvider()
        assert provider.REQUIRES_AUTH is False

    def test_validate_credentials_always_true(self):
        from pygeofetch.models.user_auth import Credentials

        provider = JaxaEarthProvider()
        assert provider.validate_credentials(Credentials(provider="jaxa_earth")) is True


class TestDownloadFallback:
    def test_download_tries_nested_path_first(self, tmp_path):
        provider = JaxaEarthProvider()
        query = SearchQuery(
            bbox={"min_lon": 138.2, "min_lat": 35.2, "max_lon": 138.8, "max_lat": 35.8}
        )
        scene = provider.search(query)[0]

        mock_resp = MagicMock(status_code=200)
        mock_resp.iter_bytes.return_value = [b"fake-tiff-data"]
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_resp
        mock_ctx.__exit__.return_value = False

        from pygeofetch.models.download_task import DownloadOptions

        with patch("httpx.stream", return_value=mock_ctx) as mock_stream:
            provider.download(scene, tmp_path, DownloadOptions())

        args, _ = mock_stream.call_args
        assert "N035E138/ALPSMLC30_N035E138_DSM.tif" in args[1]

    def test_download_falls_back_to_flat_layout_on_404(self, tmp_path):
        provider = JaxaEarthProvider()
        query = SearchQuery(
            bbox={"min_lon": 138.2, "min_lat": 35.2, "max_lon": 138.8, "max_lat": 35.8}
        )
        scene = provider.search(query)[0]

        not_found = MagicMock(status_code=404)
        not_found_ctx = MagicMock()
        not_found_ctx.__enter__.return_value = not_found
        not_found_ctx.__exit__.return_value = False

        found = MagicMock(status_code=200)
        found.iter_bytes.return_value = [b"fake-tiff-data"]
        found_ctx = MagicMock()
        found_ctx.__enter__.return_value = found
        found_ctx.__exit__.return_value = False

        from pygeofetch.models.download_task import DownloadOptions

        with patch(
            "httpx.stream", side_effect=[not_found_ctx, found_ctx]
        ) as mock_stream:
            result = provider.download(scene, tmp_path, DownloadOptions())

        assert (
            result.status.value == "completed"
            or result.status == "completed"
            or result.success
        )
        assert mock_stream.call_count == 2
