"""
Tests for the rewritten EarthExplorerAdditionalProvider -- a thin,
honest subclass of the already-verified USGSProvider M2M client,
rather than a second, parallel API implementation.
"""

from __future__ import annotations

from pygeofetch.models.search_query import SearchQuery
from pygeofetch.providers.earth_explorer_additional import (
    EarthExplorerAdditionalProvider,
)
from pygeofetch.providers.usgs import USGSProvider


class TestReusesRealUsgsInfrastructure:
    def test_is_a_real_usgs_provider_subclass(self):
        provider = EarthExplorerAdditionalProvider()
        assert isinstance(provider, USGSProvider)

    def test_authenticate_is_the_literal_inherited_method_not_reimplemented(self):
        """Confirms this doesn't duplicate the real, already-verified
        M2M auth flow -- it's the exact same method object."""
        provider = EarthExplorerAdditionalProvider()
        assert provider.authenticate.__func__ is USGSProvider.authenticate

    def test_uses_the_real_shared_m2m_base_url(self):
        provider = EarthExplorerAdditionalProvider()
        assert provider.BASE_URL == "https://m2m.cr.usgs.gov/api/api/json/stable"

    def test_search_and_download_are_inherited_not_reimplemented(self):
        provider = EarthExplorerAdditionalProvider()
        assert provider.search.__func__ is USGSProvider.search
        assert provider.download.__func__ is USGSProvider.download


class TestDatasetResolution:
    """Real dataset code resolution -- confirmed here empirically
    rather than assumed correct from writing the alias dict, which
    caught a real bug: 'kh9' is itself a substring of 'kh9hexagon',
    so dict ordering had to put the more specific key first or it
    would never be reachable at all."""

    def test_default_with_no_satellites_uses_the_confirmed_dataset(self):
        provider = EarthExplorerAdditionalProvider()
        assert provider._resolve_datasets(SearchQuery()) == ["declassii"]

    def test_kh9_resolves_to_the_confirmed_declassii(self):
        provider = EarthExplorerAdditionalProvider()
        query = SearchQuery(satellites=["KH9"])
        assert provider._resolve_datasets(query) == ["declassii"]

    def test_kh9_hexagon_resolves_to_declassiii_not_shadowed_by_kh9(self):
        """The specific real bug this locks in: without correct dict
        ordering, 'kh9' (checked first) would match 'kh9hexagon' as a
        substring and shadow the more specific, intended declassiii
        mapping entirely."""
        provider = EarthExplorerAdditionalProvider()
        query = SearchQuery(satellites=["KH9Hexagon"])
        assert provider._resolve_datasets(query) == ["declassiii"]

    def test_kh7_resolves_to_declassii(self):
        provider = EarthExplorerAdditionalProvider()
        query = SearchQuery(satellites=["KH-7"])
        assert provider._resolve_datasets(query) == ["declassii"]

    def test_corona_resolves_to_declassi(self):
        provider = EarthExplorerAdditionalProvider()
        query = SearchQuery(satellites=["Corona"])
        assert provider._resolve_datasets(query) == ["declassi"]

    def test_argon_and_lanyard_also_resolve_to_declassi(self):
        provider = EarthExplorerAdditionalProvider()
        assert provider._resolve_datasets(SearchQuery(satellites=["Argon"])) == [
            "declassi"
        ]
        assert provider._resolve_datasets(SearchQuery(satellites=["Lanyard"])) == [
            "declassi"
        ]

    def test_explicit_collections_bypass_alias_resolution(self):
        provider = EarthExplorerAdditionalProvider()
        query = SearchQuery(collections=["some_exact_dataset_code"])
        assert provider._resolve_datasets(query) == ["some_exact_dataset_code"]

    def test_default_does_not_fall_back_to_modern_landsat(self):
        """Real, deliberate override: USGSProvider's own default
        (unrelated to this provider) falls back to modern Landsat --
        not what a caller of THIS specifically-declassified-focused
        provider wants with no satellite specified."""
        provider = EarthExplorerAdditionalProvider()
        result = provider._resolve_datasets(SearchQuery())
        assert "landsat_ot_c2_l2" not in result
        assert "landsat_etm_c2_l2" not in result
