"""
EODAG provider adapter — exposes all EODAG-supported providers via PyGeoFetch.

Install with: pip install "pygeofetch[providers]"

EODAG (Earth Observation Data Access Gateway) supports 20+ providers
including:
  - Theia (French national land surface data service)
  - PEPS (French Sentinel data service)
  - Mundi (European satellite data marketplace)
  - SOBLOO (Airbus Defence data service)
  - AWS S3 public datasets (alternative STAC endpoint)
  - Cop-Ads (Copernicus Alternative Data Service)

Usage::

    from pygeofetch.providers.eodag_provider import EODAGProvider

    p = EODAGProvider()
    p.set_preferred_provider("theia")
    results = p.search(query)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import yaml

from pygeofetch.models.download_task import (
    DownloadOptions,
    DownloadResult,
    DownloadStatus,
)
from pygeofetch.models.satellite_data import QuotaInfo, SatelliteData
from pygeofetch.models.search_query import SearchQuery
from pygeofetch.models.user_auth import AuthSession, Credentials
from pygeofetch.providers.base import (
    AbstractBaseProvider,
    AuthenticationError,
    SearchError,
)

logger = logging.getLogger("pygeofetch.providers.eodag")


def _require_eodag():
    try:
        from eodag import EODataAccessGateway

        return EODataAccessGateway
    except ImportError:
        raise ImportError(
            "eodag is not installed.\n"
            'Install with: pip install "pygeofetch[providers]"\n'
            "Or directly:  pip install eodag"
        )


class EODAGProvider(AbstractBaseProvider):
    """
    EODAG-backed provider — exposes 20+ additional data sources.

    Args:
        preferred_provider: EODAG provider ID to prefer (e.g. "theia", "peps").
                            EODAG auto-selects if None.

    Example::

        from pygeofetch.providers.eodag_provider import EODAGProvider
        from pygeofetch.models.search_query import SearchQuery, BoundingBox

        p = EODAGProvider(preferred_provider="theia")
        p.authenticate(credentials)

        results = p.search(SearchQuery(
            bbox=BoundingBox.from_string("-74.1,40.6,-73.7,40.9"),
            start_date="2024-01-01", end_date="2024-06-01",
        ))
    """

    PROVIDER_ID = "eodag"
    REQUIRES_AUTH = False  # EODAG can search without auth; download may require it
    SATELLITES: list = [
        "Sentinel-1",
        "Sentinel-2",
        "Sentinel-3",
        "Sentinel-5P",
        "Landsat-8",
        "Landsat-9",
        "MODIS",
        "Envisat",
    ]

    def __init__(self, preferred_provider: Optional[str] = None) -> None:
        super().__init__()
        self._preferred = preferred_provider
        self._gateway: Any = None

    def _get_gateway(self):
        if self._gateway is None:
            EODataAccessGateway = _require_eodag()
            self._gateway = EODataAccessGateway()
            if self._preferred:
                self._gateway.set_preferred_provider(self._preferred)
        return self._gateway

    def authenticate(self, credentials: Credentials) -> AuthSession:
        """Configure EODAG credentials."""
        try:
            gw = self._get_gateway()
            # EODAG uses a config dict per provider
            if credentials.username and credentials.get_password():
                provider = self._preferred or "theia"
                gw.update_providers_config(
                    yaml.dump(
                        {
                            provider: {
                                "auth": {
                                    "credentials": {
                                        "username": credentials.username,
                                        "password": str(credentials.get_password()),
                                    }
                                }
                            }
                        }
                    )
                )
            session = AuthSession(
                provider=self.PROVIDER_ID,
                access_token=None,
                session_data={"username": credentials.username},
            )
            self._session = session
            logger.info(
                "EODAG credentials configured for provider: %s", self._preferred
            )
            return session
        except Exception as exc:
            raise AuthenticationError(f"EODAG auth failed: {exc}") from exc

    def get_capabilities(self):
        """Return provider capabilities."""
        from pygeofetch.models.satellite_data import ProviderCapabilities

        return ProviderCapabilities(
            search=True,
            download=True,
            streaming=False,
            requires_auth=False,
            supports_cql2=False,
        )

    def get_quota_info(self) -> "QuotaInfo":
        """Return quota info (EODAG does not expose quota)."""
        from pygeofetch.models.satellite_data import QuotaInfo

        return QuotaInfo(provider=self.PROVIDER_ID)

    def validate_credentials(self, credentials) -> bool:
        """Validate by attempting a lightweight EODAG operation."""
        try:
            self._get_gateway()
            return True
        except Exception:
            return False

    def search(self, query: SearchQuery) -> List[SatelliteData]:
        """Search via EODAG — returns results from preferred/all providers."""
        gw = self._get_gateway()

        try:
            # Build EODAG search parameters
            params = {
                "start": str(query.start_date) if query.start_date else "1970-01-01",
                "end": (
                    str(query.end_date)
                    if query.end_date
                    else datetime.utcnow().strftime("%Y-%m-%d")
                ),
                "items_per_page": min(getattr(query, "max_results", 100), 500),
            }

            if query.bbox:
                params["geom"] = {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [query.bbox.min_lon, query.bbox.min_lat],
                            [query.bbox.max_lon, query.bbox.min_lat],
                            [query.bbox.max_lon, query.bbox.max_lat],
                            [query.bbox.min_lon, query.bbox.max_lat],
                            [query.bbox.min_lon, query.bbox.min_lat],
                        ]
                    ],
                }

            if query.cloud_cover_max is not None:
                params["cloudCover"] = query.cloud_cover_max

            # Determine product type from satellites
            product_type = self._resolve_product_type(query)
            logger.info("EODAG search: %s %s", product_type, params.get("start"))

            results, _ = gw.search(product_type=product_type, **params)
            return [self._eodag_to_satellite_data(r) for r in results]

        except Exception as exc:
            raise SearchError(f"EODAG search failed: {exc}") from exc

    def download(
        self,
        data: SatelliteData,
        destination: Path,
        options: DownloadOptions,
    ) -> DownloadResult:
        """Download via EODAG."""
        gw = self._get_gateway()
        destination.mkdir(parents=True, exist_ok=True)

        try:
            # Reconstruct the EODAG product from stored properties
            product_id = data.properties.get("eodag_product_id") or data.id
            results, _ = gw.search(
                id=product_id, product_type=data.collection or "S2_MSI_L2A"
            )
            if not results:
                return DownloadResult(
                    status=DownloadStatus.FAILED,
                    data_id=data.id,
                    provider=self.PROVIDER_ID,
                    error="Product not found by EODAG",
                )
            product = results[0]
            paths = gw.download(product, outputs_prefix=str(destination))
            out_paths = [
                Path(p) for p in (paths if isinstance(paths, list) else [paths])
            ]
            return DownloadResult(
                status=DownloadStatus.COMPLETED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                output_path=out_paths[0] if out_paths else None,
                output_paths=out_paths,
            )
        except Exception as exc:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                error=str(exc),
            )

    def _resolve_product_type(self, query: SearchQuery) -> str:
        """Map PyGeoFetch query to an EODAG product type string."""
        sats = [s.lower() for s in (query.satellites or [])]
        pt = (getattr(query, "product_type", None) or "").upper()

        if "sentinel-2" in " ".join(sats) or not sats:
            return "S2_MSI_L2A"
        if "sentinel-1" in " ".join(sats):
            return "S1_SAR_SLC" if pt == "SLC" else "S1_SAR_GRD"
        if "landsat" in " ".join(sats):
            return "LANDSAT_C2L2"
        return "S2_MSI_L2A"  # safe default

    def _eodag_to_satellite_data(self, product: Any) -> SatelliteData:
        """Convert an EODAG EOProduct to a PyGeoFetch SatelliteData."""
        props = product.properties or {}
        bbox = None
        geometry = None
        if hasattr(product, "geometry") and product.geometry:
            b = product.geometry.bounds
            if len(b) == 4:
                bbox = (b[0], b[1], b[2], b[3])
            # Real fix: EODAG's own documented API gives a real Shapely
            # geometry object here, not just bounds -- shapely's own
            # mapping() utility converts it to real GeoJSON directly.
            try:
                from shapely.geometry import mapping

                geometry = mapping(product.geometry)
            except Exception:
                geometry = None

        from pygeofetch.models.satellite_data import ProcessingLevel, SatelliteAsset

        raw_processing_level = props.get("processingLevel")
        processing_level = ProcessingLevel.UNKNOWN
        if raw_processing_level:
            for pl in ProcessingLevel:
                if pl.value == raw_processing_level:
                    processing_level = pl
                    break

        return SatelliteData(
            id=product.properties.get("id", str(product)),
            provider=self.PROVIDER_ID,
            collection=product.product_type,
            satellite=props.get("platform") or props.get("platformSerialIdentifier"),
            datetime=props.get("startTimeFromAscendingNode") or props.get("startDate"),
            bbox=bbox,
            geometry=geometry,
            cloud_cover=props.get("cloudCover"),
            processing_level=processing_level,
            assets={
                "download": SatelliteAsset(
                    key="download", href=product.remote_location or ""
                )
            },
            properties={"eodag_product_id": product.properties.get("id", ""), **props},
        )
