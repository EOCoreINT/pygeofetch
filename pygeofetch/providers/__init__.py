"""
Provider registry for PyGeoFetch.

22 satellite data providers registered here. Add new providers by:
1. Creating a module in pygeofetch/providers/
2. Implementing AbstractBaseProvider
3. Adding to PROVIDER_REGISTRY and PROVIDER_META below.
"""

from __future__ import annotations

from pygeofetch.providers.base import AbstractBaseProvider


def _lazy_load_providers() -> dict[str, type[AbstractBaseProvider]]:
    from pygeofetch.providers.airbus_oneatlas import AirbusOneatlasProvider
    from pygeofetch.providers.alaska_satellite_facility import (
        AlaskaSatelliteFacilityProvider,
    )
    from pygeofetch.providers.aws_earth import AWSEarthProvider
    from pygeofetch.providers.copernicus import CopernicusProvider
    from pygeofetch.providers.digitalglobe import DigitalglobeProvider
    from pygeofetch.providers.earth_explorer_additional import (
        EarthExplorerAdditionalProvider,
    )
    from pygeofetch.providers.element84 import Element84Provider
    from pygeofetch.providers.esa_scihub import EsaScihubProvider
    from pygeofetch.providers.geoserver_generic import GeoserverGenericProvider
    from pygeofetch.providers.google_earth_engine import GoogleEarthEngineProvider
    from pygeofetch.providers.inpe_cbers import InpeCbersProvider
    from pygeofetch.providers.isro_bhuvan import IsroBhuvanProvider
    from pygeofetch.providers.jaxa_earth import JaxaEarthProvider
    from pygeofetch.providers.maxar_gbdx import MaxarGbdxProvider
    from pygeofetch.providers.nasa_earthdata import NASAEarthdataProvider
    from pygeofetch.providers.nasa_earthdata_cloud import NASAEarthdataCloudProvider
    from pygeofetch.providers.noaa_big_data import NoaaBigDataProvider
    from pygeofetch.providers.opentopography import OpentopographyProvider
    from pygeofetch.providers.planet import PlanetProvider
    from pygeofetch.providers.planetary_computer import PlanetaryComputerProvider
    from pygeofetch.providers.sentinel_hub import SentinelHubProvider
    from pygeofetch.providers.usgs import USGSProvider

    return {
        "usgs": USGSProvider,
        "copernicus": CopernicusProvider,
        "aws_earth": AWSEarthProvider,
        "nasa_earthdata": NASAEarthdataProvider,
        "nasa_earthdata_cloud": NASAEarthdataCloudProvider,
        "planetary_computer": PlanetaryComputerProvider,
        "element84": Element84Provider,
        "opentopography": OpentopographyProvider,
        "planet": PlanetProvider,
        "sentinel_hub": SentinelHubProvider,
        "maxar_gbdx": MaxarGbdxProvider,
        "airbus_oneatlas": AirbusOneatlasProvider,
        "alaska_satellite_facility": AlaskaSatelliteFacilityProvider,
        "noaa_big_data": NoaaBigDataProvider,
        "google_earth_engine": GoogleEarthEngineProvider,
        "earth_explorer_additional": EarthExplorerAdditionalProvider,
        "jaxa_earth": JaxaEarthProvider,
        "isro_bhuvan": IsroBhuvanProvider,
        "inpe_cbers": InpeCbersProvider,
        "esa_scihub": EsaScihubProvider,
        "digitalglobe": DigitalglobeProvider,
        "geoserver_generic": GeoserverGenericProvider,
    }


_REGISTRY: dict[str, type[AbstractBaseProvider]] | None = None


def _get_registry() -> dict[str, type[AbstractBaseProvider]]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _lazy_load_providers()
    return _REGISTRY


def get_provider(provider_id: str, config: dict | None = None) -> AbstractBaseProvider:
    """
    Instantiate a provider by its ID.

    Args:
        provider_id: Provider identifier (e.g. 'usgs', 'copernicus', 'planetary_computer').
        config: Optional provider-specific configuration dict.

    Returns:
        Instantiated provider.

    Raises:
        ValueError: If provider_id is not registered.
    """
    registry = _get_registry()
    pid = provider_id.lower().replace("-", "_")
    if pid not in registry:
        available = ", ".join(sorted(registry.keys()))
        msg = f"Unknown provider {provider_id!r}. Available: {available}"
        raise ValueError(msg)
    return registry[pid](config=config)


def list_providers() -> list[str]:
    """Return sorted list of all registered provider IDs."""
    return sorted(_get_registry().keys())


def list_provider_info() -> list[dict]:
    """Return rich metadata for all registered providers."""
    registry = _get_registry()
    results = []
    for pid, cls in sorted(registry.items()):
        try:
            instance = cls()
            caps = instance.get_capabilities()
            results.append(
                {
                    "id": pid,
                    "name": caps.name or getattr(cls, "DISPLAY_NAME", pid),
                    "display_name": caps.name or getattr(cls, "DISPLAY_NAME", pid),
                    "description": caps.description or getattr(cls, "DESCRIPTION", ""),
                    "requires_auth": cls.REQUIRES_AUTH,
                    "auth_type": caps.auth_type,
                    "satellites": caps.satellites or getattr(cls, "SATELLITES", []),
                    "supports_sar": caps.supports_sar,
                    "supports_sub_meter": caps.supports_sub_meter,
                    "stac": caps.stac,
                    "regions": caps.regions,
                    "resolution_min_m": caps.resolution_min_m,
                    "resolution_max_m": caps.resolution_max_m,
                    "endpoint_url": caps.endpoint_url,
                    "docs_url": caps.docs_url,
                }
            )
        except Exception:
            results.append(
                {
                    "id": pid,
                    "name": getattr(cls, "DISPLAY_NAME", pid),
                    "display_name": getattr(cls, "DISPLAY_NAME", pid),
                    "description": getattr(cls, "DESCRIPTION", ""),
                    "requires_auth": cls.REQUIRES_AUTH,
                    "auth_type": "unknown",
                    "satellites": getattr(cls, "SATELLITES", []),
                    "supports_sar": False,
                    "supports_sub_meter": False,
                    "stac": False,
                    "regions": [],
                    "resolution_min_m": None,
                    "resolution_max_m": None,
                    "endpoint_url": "",
                    "docs_url": "",
                }
            )
    return results


def get_free_providers() -> list[str]:
    """Return IDs of providers that require no authentication."""
    return [pid for pid, cls in _get_registry().items() if not cls.REQUIRES_AUTH]


def get_providers_by_capability(
    sar: bool | None = None,
    sub_meter: bool | None = None,
    stac: bool | None = None,
    requires_auth: bool | None = None,
    region: str | None = None,
) -> list[str]:
    """
    Filter providers by capability flags.

    Args:
        sar: If True/False, filter by SAR support.
        sub_meter: If True/False, filter by sub-meter resolution.
        stac: If True/False, filter by STAC API support.
        requires_auth: If True/False, filter by auth requirement.
        region: Filter by region string (e.g. 'global', 'europe').

    Returns:
        List of matching provider IDs.
    """
    info = list_provider_info()
    results = []
    for item in info:
        if sar is not None and item.get("supports_sar") != sar:
            continue
        if sub_meter is not None and item.get("supports_sub_meter") != sub_meter:
            continue
        if stac is not None and item.get("stac") != stac:
            continue
        if requires_auth is not None and item.get("requires_auth") != requires_auth:
            continue
        if region is not None and region not in (item.get("regions") or []):
            continue
        results.append(item["id"])
    return results


__all__ = [
    "AbstractBaseProvider",
    "get_provider",
    "list_providers",
    "list_provider_info",
    "get_free_providers",
    "get_providers_by_capability",
]