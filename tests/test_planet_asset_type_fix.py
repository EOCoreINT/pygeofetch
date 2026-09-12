"""
Tests for the real fix to PlanetProvider.download()'s asset type
selection: the previous code hardcoded "ortho_analytic_4b_sr" (a real
PSScene-specific asset) for every item_type, including SkySat --
confirmed against Planet's own SkySat documentation that no such asset
exists there at all (SkySat's real asset types are ortho_pansharpened,
ortho_visual, ortho_analytic, ortho_panchromatic).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pygeofetch.models.download_task import DownloadOptions
from pygeofetch.models.satellite_data import SatelliteData
from pygeofetch.providers.planet import PlanetProvider


def _authed_provider() -> PlanetProvider:
    provider = PlanetProvider()
    provider._session = MagicMock()
    provider._session.session_data = {"api_key": "real-planet-key"}
    return provider


def _scene(satellite: str) -> SatelliteData:
    return SatelliteData(
        id="20240615_test_scene", provider="planet", satellite=satellite
    )


class TestRealPerItemTypeAssetSelection:
    def test_psscene_uses_the_real_confirmed_default(self):
        assert PlanetProvider.DEFAULT_ASSET_TYPES["PSScene"] == "ortho_analytic_4b_sr"

    def test_skysat_does_not_use_the_psscene_specific_asset(self):
        """The specific real bug: SkySat has no ortho_analytic_4b_sr
        asset at all -- confirmed against Planet's own documentation."""
        assert (
            PlanetProvider.DEFAULT_ASSET_TYPES["SkySatScene"] != "ortho_analytic_4b_sr"
        )
        assert (
            PlanetProvider.DEFAULT_ASSET_TYPES["SkySatCollect"]
            != "ortho_analytic_4b_sr"
        )

    def test_skysat_uses_the_real_confirmed_ortho_pansharpened(self):
        assert PlanetProvider.DEFAULT_ASSET_TYPES["SkySatScene"] == "ortho_pansharpened"
        assert (
            PlanetProvider.DEFAULT_ASSET_TYPES["SkySatCollect"] == "ortho_pansharpened"
        )

    def test_download_activates_the_real_correct_asset_for_skysat(self, tmp_path):
        provider = _authed_provider()
        scene = _scene("SkySatCollect")

        with patch.object(
            provider, "_activate_asset", return_value=True
        ) as mock_activate, patch.object(
            provider, "_wait_for_activation", return_value=None
        ):
            provider.download(scene, tmp_path, DownloadOptions())

        args, _ = mock_activate.call_args
        # (item_id, item_type, asset_type, api_key)
        assert args[2] == "ortho_pansharpened"

    def test_download_activates_the_real_correct_asset_for_psscene(self, tmp_path):
        provider = _authed_provider()
        scene = _scene("PSScene")

        with patch.object(
            provider, "_activate_asset", return_value=True
        ) as mock_activate, patch.object(
            provider, "_wait_for_activation", return_value=None
        ):
            provider.download(scene, tmp_path, DownloadOptions())

        args, _ = mock_activate.call_args
        assert args[2] == "ortho_analytic_4b_sr"

    def test_config_override_takes_precedence_over_the_default_mapping(self, tmp_path):
        provider = _authed_provider()
        provider.config = {"asset_type": "ortho_udm2"}
        scene = _scene("PSScene")

        with patch.object(
            provider, "_activate_asset", return_value=True
        ) as mock_activate, patch.object(
            provider, "_wait_for_activation", return_value=None
        ):
            provider.download(scene, tmp_path, DownloadOptions())

        args, _ = mock_activate.call_args
        assert args[2] == "ortho_udm2"

    def test_unmapped_item_type_falls_back_to_the_original_default(self, tmp_path):
        provider = _authed_provider()
        scene = _scene("REOrthoTile")

        with patch.object(
            provider, "_activate_asset", return_value=True
        ) as mock_activate, patch.object(
            provider, "_wait_for_activation", return_value=None
        ):
            provider.download(scene, tmp_path, DownloadOptions())

        args, _ = mock_activate.call_args
        assert args[2] == "ortho_analytic_4b_sr"
