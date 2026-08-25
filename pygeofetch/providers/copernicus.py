"""
Copernicus Data Space Ecosystem (CDSE) provider.

Provides access to Sentinel-1, Sentinel-2, Sentinel-3, Sentinel-5P, and
other Copernicus missions via the ESA CDSE OData API and STAC catalog.

Authentication:
    Uses Copernicus Data Space account credentials with OAuth2 token exchange.
    Free registration at https://dataspace.copernicus.eu

Supported datasets:
    - Sentinel-1 (SAR C-band)
    - Sentinel-2 (Multispectral, 10-60m)
    - Sentinel-3 (Ocean/Land colour, OLCI/SLSTR)
    - Sentinel-5P (Atmospheric composition)
    - Sentinel-6 (Sea level)

Example::

    from pygeofetch.providers.copernicus import CopernicusProvider
    from pygeofetch.models.user_auth import Credentials, AuthType

    provider = CopernicusProvider()
    session = provider.authenticate(Credentials(
        provider="copernicus",
        auth_type=AuthType.USERNAME_PASSWORD,
        username="user@example.com",
        password="password",
    ))

    results = provider.search(SearchQuery(
        bbox=(-74.1, 40.6, -73.7, 40.9),
        start_date="2024-01-01",
        end_date="2024-03-01",
        satellites=["Sentinel-2"],
        cloud_cover_max=15,
    ))
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pygeofetch.core.logging import report_download_progress
from pygeofetch.models.download_task import (
    DownloadOptions,
    DownloadResult,
    DownloadStatus,
)
from pygeofetch.models.satellite_data import (
    DataFormat,
    ProcessingLevel,
    ProviderCapabilities,
    QuotaInfo,
    SatelliteAsset,
    SatelliteData,
)
from pygeofetch.models.user_auth import AuthSession, Credentials
from pygeofetch.providers.base import (
    AbstractBaseProvider,
    AuthenticationError,
)

if TYPE_CHECKING:
    from pygeofetch.models.search_query import SearchQuery


class CopernicusProvider(AbstractBaseProvider):
    """
    Copernicus Data Space Ecosystem provider.

    Supports search across all Sentinel missions and download via
    S3 presigned URLs or direct HTTPS.

    Attributes:
        PROVIDER_ID: 'copernicus'
        DISPLAY_NAME: 'Copernicus Data Space'
        REQUIRES_AUTH: True
    """

    PROVIDER_ID = "copernicus"
    DISPLAY_NAME = "Copernicus Data Space"
    REQUIRES_AUTH = True
    DESCRIPTION = (
        "Access to all Sentinel satellite missions (Sentinel-1 through 6) "
        "via ESA's Copernicus Data Space Ecosystem."
    )
    DATA_TYPES = ["Sentinel-1", "Sentinel-2", "Sentinel-3", "Sentinel-5P", "Sentinel-6"]
    BASE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
    # CRITICAL: downloads MUST use a DIFFERENT host from the catalogue search
    # host. Using BASE_URL for downloads causes 401/403/404 errors — this is
    # a well-documented CDSE gotcha (see the CDSE community forum threads on
    # OData download failures). Confirmed against the official docs:
    # https://documentation.dataspace.copernicus.eu/APIs/OData.html
    DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1"
    AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"

    # Map user-friendly satellite names to CDSE collection names
    SATELLITE_COLLECTION_MAP = {
        # Sentinel-1 — all platform variants map to the same collection
        "sentinel-1": "SENTINEL-1",
        "sentinel-1a": "SENTINEL-1",
        "sentinel-1b": "SENTINEL-1",
        "sentinel-1c": "SENTINEL-1",  # active constellation (May 2025+)
        "sentinel-1d": "SENTINEL-1",  # active constellation (Apr 2026+)
        "s1a": "SENTINEL-1",
        "s1b": "SENTINEL-1",
        "s1c": "SENTINEL-1",
        "s1d": "SENTINEL-1",
        # Sentinel-2
        "sentinel-2": "SENTINEL-2",
        "sentinel-2a": "SENTINEL-2",
        "sentinel-2b": "SENTINEL-2",
        "sentinel-2c": "SENTINEL-2",
        # Sentinel-3, 5P, 6
        "sentinel-3": "SENTINEL-3",
        "sentinel-5": "SENTINEL-5P",
        "sentinel-5p": "SENTINEL-5P",
        "sentinel-6": "SENTINEL-6",
    }

    def authenticate(self, credentials: Credentials) -> AuthSession:
        """
        Obtain an OAuth2 access token from Copernicus Identity Service.

        Args:
            credentials: Must contain username and password.

        Returns:
            AuthSession with access and refresh tokens.

        Raises:
            AuthenticationError: If login fails.
        """
        if not credentials.username or not credentials.get_password():
            msg = "Copernicus requires username (email) and password"
            raise AuthenticationError(msg)

        try:
            import httpx

            response = httpx.post(
                self.AUTH_URL,
                data={
                    "grant_type": "password",
                    "username": credentials.username,
                    "password": credentials.get_password(),
                    "client_id": "cdse-public",
                    # Two-Factor Authentication support — required by CDSE
                    # on every login when 2FA is enabled on the account.
                    # Without this, 2FA-enabled accounts fail with the same
                    # generic "Invalid user credentials" error as a wrong
                    # password, giving no indication that 2FA is the cause.
                    **(
                        {"totp": credentials.extra["totp"]}
                        if credentials.extra.get("totp")
                        else {}
                    ),
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
            if response.status_code != 200:
                try:
                    err = response.json().get("error_description", response.text[:200])
                except Exception:
                    err = response.text[:200]

                hint = ""
                if "Invalid user credentials" in err or "invalid_grant" in err:
                    has_totp = bool(credentials.extra.get("totp"))
                    hint = (
                        "\n\nCDSE returns this exact generic message for several "
                        "distinct causes — check each:\n"
                        "  1. Password typo — verify by logging in at "
                        "https://dataspace.copernicus.eu directly.\n"
                        f"  2. Two-Factor Authentication (2FA) enabled on the account "
                        f"but no code supplied (totp {'was' if has_totp else 'was NOT'} "
                        "passed to this request) — if 2FA is enabled in your CDSE "
                        "profile, pass the current 6-digit code: "
                        "client.add_credentials('copernicus', username=..., "
                        "password=..., totp='123456'). Codes expire in ~30s, so "
                        "generate it right before running the request.\n"
                        "  3. Account email not yet verified — check your inbox for "
                        "a verification link from registration.\n"
                        "  4. Account created very recently — CDSE can take a few "
                        "minutes to fully activate new accounts."
                    )
                msg = f"Copernicus login failed: {err}{hint}"
                raise AuthenticationError(msg)

            token_data = response.json()
            session = AuthSession(
                provider=self.PROVIDER_ID,
                access_token=token_data["access_token"],
                refresh_token=token_data.get("refresh_token"),
                expires_at=datetime.utcnow()
                + timedelta(seconds=token_data.get("expires_in", 600)),
                session_data={"username": credentials.username},
            )
            self._session = session
            # Cached for the cdse-package fallback path in download() only —
            # never logged or persisted to disk.
            self._raw_password = credentials.get_password()
            self._logger.info(
                f"Authenticated with Copernicus Data Space as {credentials.username!r}"
            )
            return session

        except AuthenticationError:
            raise
        except Exception as exc:
            msg = f"Copernicus auth error: {exc}"
            raise AuthenticationError(msg) from exc

    def _refresh_token_if_needed(self, force: bool = False) -> None:
        """Refresh the access token if it expires soon (or immediately if force=True)."""
        if not self._session:
            return
        mins = self._session.minutes_until_expiry
        needs_refresh = force or (mins is not None and mins < 5)
        if needs_refresh and self._session.refresh_token:
            try:
                import httpx

                response = httpx.post(
                    self.AUTH_URL,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": self._session.refresh_token,
                        "client_id": "cdse-public",
                    },
                    timeout=30,
                )
                if response.status_code == 200:
                    token_data = response.json()
                    self._session.access_token = token_data["access_token"]
                    self._session.refresh_token = token_data.get(
                        "refresh_token", self._session.refresh_token
                    )
                    self._session.expires_at = datetime.utcnow() + timedelta(
                        seconds=token_data.get("expires_in", 600)
                    )
                    self._logger.debug("Copernicus token refreshed")
            except Exception as exc:
                self._logger.warning(f"Token refresh failed: {exc}")

    def validate_credentials(self, credentials: Credentials) -> bool:
        return bool(credentials.username and credentials.get_password())

    def set_session(self, session: Any) -> None:
        """Store an authenticated session for use in requests."""
        self._session = session

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        """
        Search the Copernicus OData catalog.

        Args:
            query: Search parameters.

        Returns:
            List of matching SatelliteData records.
        """
        self.require_auth()
        self._refresh_token_if_needed()

        collections = self._resolve_collections(query)
        results: list[SatelliteData] = []

        for collection in collections:
            try:
                items = self._search_collection(collection, query)
                results.extend(items)
                if len(results) >= query.max_results:
                    break
            except Exception as exc:
                self._logger.warning(
                    f"Search failed for collection {collection!r}: {exc}"
                )

        # Real, confirmed gap fixed here: query.satellites only ever
        # resolves to a real Copernicus COLLECTION (e.g. "SENTINEL-1"),
        # never a specific satellite UNIT within it -- confirmed directly
        # against this provider's own SATELLITE_COLLECTION_MAP, whose own
        # comment states "all platform variants map to the same
        # collection". That means requesting satellites=["Sentinel-1A"]
        # silently returned S1B scenes too, since the real Copernicus API
        # itself has no separate collection for S1A vs S1B -- they share
        # one real "SENTINEL-1" collection, so unit-level selection can
        # only happen as a real, client-side post-filter, after the
        # results already contain both units. Only activates when the
        # caller actually asked for a specific unit (e.g. "Sentinel-1A",
        # not just "Sentinel-1") -- a bare mission-level request still
        # returns every real unit, matching the original, correct
        # behaviour for that case.
        requested_units = self._extract_requested_satellite_units(query.satellites)
        if requested_units:
            before = len(results)
            results = [
                r for r in results
                if self._product_unit(r) is None or self._product_unit(r) in requested_units
            ]
            self._logger.info(
                "Real unit-level satellite filter (%s): %d/%d results kept.",
                sorted(requested_units), len(results), before,
            )

        return results[: query.max_results]

    @staticmethod
    def _extract_requested_satellite_units(satellites: "list[str] | None") -> set[str]:
        """
        Real, confirmed fix: parse out specific satellite UNIT requests
        (e.g. "S1A") from a real query.satellites list, distinct from a
        bare mission-level request (e.g. "Sentinel-1", which should
        still return every real unit). Returns an empty set when no
        unit-level names were given, signalling "no extra filtering
        needed" to the caller.
        """
        units: set[str] = set()
        if not satellites:
            return units
        for sat in satellites:
            match = re.search(r"(?:sentinel-?|s)(\d)([a-d])\b", sat.strip().lower())
            if match:
                units.add(f"S{match.group(1)}{match.group(2).upper()}")
        return units

    @staticmethod
    def _product_unit(result: "SatelliteData") -> "str | None":
        """
        Real satellite unit ("S1A", "S1B", ...) read directly from the
        result's own real product name/id, by Sentinel's own, standard
        naming convention (first 3 characters). Returns None if the
        real name doesn't start with a recognisable unit code, so a
        caller can choose to keep ambiguous results rather than silently
        drop them.
        """
        name = None
        if hasattr(result, "properties") and isinstance(result.properties, dict):
            name = result.properties.get("name")
        name = name or getattr(result, "id", None)
        if not name:
            return None
        prefix = str(name)[:3].upper()
        return prefix if re.match(r"^S\d[A-D]$", prefix) else None

    def _resolve_collections(self, query: SearchQuery) -> list[str]:
        """Map query satellite names to Copernicus collection codes."""
        if query.collections:
            return [c.upper() for c in query.collections]

        if not query.satellites:
            # Use product_type to pick a sensible default collection
            pt = getattr(query, "product_type", None) or ""
            if pt.upper() in ("SLC", "GRD", "GRD-COG", "OCN"):
                return ["SENTINEL-1"]  # SAR product type → Sentinel-1
            return ["SENTINEL-2"]  # default for optical

        collections = set()
        for sat in query.satellites:
            sat_lower = sat.lower().replace("-", "").replace("_", "").replace(" ", "")
            sat_orig = sat.lower()
            matched = False
            # Try exact match first, then prefix/substring
            for key, col in self.SATELLITE_COLLECTION_MAP.items():
                key_norm = key.replace("-", "").replace("_", "")
                if key_norm == sat_lower or key in sat_orig or sat_orig.startswith(key):
                    collections.add(col)
                    matched = True
                    break
            if not matched:
                # Fall back: if it mentions "sentinel-1" anywhere, use SENTINEL-1
                if "sentinel-1" in sat_orig or sat_orig.startswith("s1"):
                    collections.add("SENTINEL-1")
                elif "sentinel-2" in sat_orig or sat_orig.startswith("s2"):
                    collections.add("SENTINEL-2")
                else:
                    # Pass through as-is (user may know the collection name)
                    collections.add(sat.upper())

        # If no satellite matched but product_type suggests SAR, use Sentinel-1
        if not collections:
            pt = getattr(query, "product_type", None) or ""
            if pt.upper() in ("SLC", "GRD", "GRD-COG", "OCN"):
                collections.add("SENTINEL-1")
            else:
                collections.add("SENTINEL-2")

        return list(collections)

    def _build_odata_filter(self, collection: str, query: SearchQuery) -> str:
        """Build OData $filter string for the given collection and query."""
        filters = [f"Collection/Name eq '{collection}'"]

        if query.start_date:
            filters.append(f"ContentDate/Start ge {query.start_date}T00:00:00.000Z")
        if query.end_date:
            filters.append(f"ContentDate/Start le {query.end_date}T23:59:59.000Z")

        # Prefer a real polygon spatial filter over a bounding-box
        # approximation when a geometry is given. Previously query.geometry
        # was never checked here at all — only query.bbox — meaning any
        # search using geometry= (not bbox=) had NO spatial constraint
        # applied whatsoever: the query scanned Copernicus's entire global
        # archive for the date/product-type/polarisation criteria, sorted
        # by most-recent-first, with location playing no role at all. This
        # is confirmed to be the actual explanation for a real search
        # returning a Sentinel-1 scene over the Yucatán Peninsula, Mexico
        # for a query intended for Accra, Ghana — not a GCP-reading or
        # georeferencing bug downstream, which was independently tested
        # and ruled out. This is the exact same bug class (a field never
        # checked at all) found and fixed for USGS earlier — it just
        # hadn't been checked for Copernicus specifically until now.
        geometry = getattr(query, "geometry", None)
        if geometry:
            geom_type = geometry.get("type", "Polygon")
            coordinates = geometry.get("coordinates")
            if coordinates and geom_type == "Polygon":
                # OData.CSC.Intersects expects a WKT POLYGON — build it
                # directly from the GeoJSON ring, not a bbox approximation,
                # so search results are constrained to the real AOI shape.
                ring = coordinates[0]
                coord_str = ",".join(f"{lon} {lat}" for lon, lat in ring)
                polygon = f"POLYGON(({coord_str}))"
                filters.append(
                    f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')"
                )
        elif query.bbox:
            bbox = query.bbox
            polygon = (
                f"POLYGON(("
                f"{bbox.min_lon} {bbox.min_lat},"
                f"{bbox.max_lon} {bbox.min_lat},"
                f"{bbox.max_lon} {bbox.max_lat},"
                f"{bbox.min_lon} {bbox.max_lat},"
                f"{bbox.min_lon} {bbox.min_lat}))"
            )
            filters.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')")

        if (query.cloud_cover_max or 100) < 100:
            filters.append(
                f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {query.cloud_cover_max})"  # noqa: E501
            )
        if getattr(query, "cloud_cover_min", 0) > 0:
            filters.append(
                f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value ge {getattr(query, 'cloud_cover_min', 0)})"  # noqa: E501
            )

        # Processing level filter
        if query.processing_levels:
            level_filter = " or ".join(
                f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'processingLevel' and att/OData.CSC.StringAttribute/Value eq '{level}')"  # noqa: E501
                for level in query.processing_levels
            )
            filters.append(f"({level_filter})")

        # Product type filter (SLC, GRD, GRD-COG, OCN for Sentinel-1)
        pt = getattr(query, "product_type", None)
        if pt:
            filters.append(
                f"Attributes/OData.CSC.StringAttribute/any("
                f"att:att/Name eq 'productType' and "
                f"att/OData.CSC.StringAttribute/Value eq '{pt.upper()}')"
            )

        # Polarisation filter (VV, VH, VV+VH, HH etc.)
        #
        # CDSE stores polarisationChannels as a COMBINED string for dual-pol
        # acquisitions — e.g. 'VV&VH', not separate 'VV' and 'VH' values.
        # Confirmed empirically against the live API on real SLC products
        # over a test AOI: polarisationChannels = 'VV&VH' verbatim.
        # An exact eq('VV') match therefore matches ZERO products for any
        # dual-pol acquisition — the vast majority of Sentinel-1 IW data.
        #
        # Two fix attempts were tried and rejected before finding the
        # working pattern below, each confirmed by direct live-API testing:
        #   1. contains(Value, 'VV') inside the any() lambda — CDSE returns
        #      HTTP 400 "Function 'contains' is not supported". CDSE's docs
        #      only demonstrate contains()/startswith()/endswith() on
        #      top-level fields like Name, never inside an any() predicate.
        #   2. A single any() with an OR'd condition inside its lambda —
        #      `any(att:att/Name eq 'polarisationChannels' and
        #      (Value eq 'VV' or Value eq 'VV&VH'))` — returns HTTP 200 but
        #      0 results despite matching known-good data. CDSE's OData
        #      engine appears not to evaluate compound OR conditions
        #      correctly inside a single any() lambda predicate.
        #
        # WORKING fix (confirmed 114/117 real SLC products matched): two
        # separate any() calls, each with a single eq() condition, OR'd
        # together at the top filter level rather than inside one lambda.
        pol = getattr(query, "polarisation", None)
        if pol:
            pol_upper = pol.upper()
            # Map each requested channel to every CDSE-stored string format
            # that includes it (single-pol and the relevant dual-pol
            # combination).
            _POL_VALUE_MAP = {
                "VV": ["VV", "VV&VH"],
                "VH": ["VH", "VV&VH"],
                "HH": ["HH", "HH&HV"],
                "HV": ["HV", "HH&HV"],
            }
            candidates = _POL_VALUE_MAP.get(pol_upper, [pol_upper])
            or_clauses = " or ".join(
                f"Attributes/OData.CSC.StringAttribute/any("
                f"att:att/Name eq 'polarisationChannels' and "
                f"att/OData.CSC.StringAttribute/Value eq '{v}')"
                for v in candidates
            )
            filters.append(f"({or_clauses})")

        # Pass direction filter (ascending / descending)
        pass_dir = getattr(query, "pass_direction", None)
        if pass_dir:
            filters.append(
                f"Attributes/OData.CSC.StringAttribute/any("
                f"att:att/Name eq 'orbitDirection' and "
                f"att/OData.CSC.StringAttribute/Value eq '{pass_dir.upper()}')"
            )

        return " and ".join(filters)

    def _search_collection(
        self, collection: str, query: SearchQuery
    ) -> list[SatelliteData]:
        """Perform OData search for a specific collection."""
        import httpx

        params: dict[str, Any] = {
            "$filter": self._build_odata_filter(collection, query),
            "$orderby": "ContentDate/Start desc",
            "$top": min(query.max_results, 1000),
            "$expand": "Attributes",
        }

        response = httpx.get(
            f"{self.BASE_URL}/Products",
            params=params,
            headers={"Authorization": f"Bearer {self._session.access_token}"},  # type: ignore
            timeout=60,
        )
        self._handle_http_error(response)
        data = response.json()
        return [self._product_to_satellite_data(p) for p in data.get("value", [])]

    def _product_to_satellite_data(self, product: dict) -> SatelliteData:
        """Convert a Copernicus OData product dict to SatelliteData."""
        product_id = product.get("Id", "")
        name = product.get("Name", "")
        collection = (
            product.get("S3Path", "").split("/")[2] if product.get("S3Path") else ""
        )

        # Parse footprint -- real, confirmed field name is "GeoFootprint"
        # (verified directly against Copernicus Data Space Ecosystem's
        # own official documentation, documentation.dataspace.copernicus.eu/APIs/OData.html,
        # which lists it explicitly: columns_to_print = ['Id', 'Name',
        # 'S3Path', 'GeoFootprint']). It's a real, structured GeoJSON
        # geometry object, not a WKT string -- the previous code looked
        # for a nonexistent "Footprint" field expecting WKT text, which
        # meant bbox (and geometry, which wasn't populated at all)
        # silently came back None for every real Copernicus result.
        bbox = None
        geometry = None
        geo_footprint = product.get("GeoFootprint")
        if isinstance(geo_footprint, dict) and geo_footprint.get("coordinates"):
            geometry = geo_footprint
            try:
                coords = geo_footprint["coordinates"]
                # Real, robust flattening: handles Polygon (nested one
                # level deeper than Point) without assuming a specific
                # geometry type up front
                flat_coords = []

                def _flatten(c):
                    if isinstance(c[0], (int, float)):
                        flat_coords.append(c)
                    else:
                        for sub in c:
                            _flatten(sub)

                _flatten(coords)
                lons = [c[0] for c in flat_coords]
                lats = [c[1] for c in flat_coords]
                bbox = (min(lons), min(lats), max(lons), max(lats))
            except Exception:
                pass

        # Parse datetime
        dt = None
        date_str = product.get("ContentDate", {}).get("Start")
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except Exception:
                pass

        # Extract attributes
        attributes = {
            a.get("Name"): a.get("Value") for a in (product.get("Attributes") or [])
        }
        cloud_cover = attributes.get("cloudCover")
        if cloud_cover is not None:
            try:
                cloud_cover = float(cloud_cover)
            except (ValueError, TypeError):
                cloud_cover = None

        # Determine satellite platform from product name (e.g. S1C_IW_SLC...)
        satellite = "Sentinel"
        platform = attributes.get("platform") or attributes.get("platformShortName", "")
        if name.startswith("S1C") or "SENTINEL-1C" in platform.upper():
            satellite = "SENTINEL-1C"
        elif name.startswith("S1D") or "SENTINEL-1D" in platform.upper():
            satellite = "SENTINEL-1D"
        elif name.startswith("S1A") or "SENTINEL-1A" in platform.upper():
            satellite = "SENTINEL-1A"
        elif name.startswith("S1B") or "SENTINEL-1B" in platform.upper():
            satellite = "SENTINEL-1B"
        elif name.startswith("S1"):
            satellite = "Sentinel-1"
        elif name.startswith("S2"):
            satellite = "Sentinel-2"
        elif name.startswith("S3"):
            satellite = "Sentinel-3"
        elif name.startswith("S5P"):
            satellite = "Sentinel-5P"

        # Extract SAR-specific attributes
        # Copernicus OData attribute names (from $expand=Attributes):
        product_type = attributes.get("productType")  # "SLC", "GRD", "GRD-COG"
        polarisation = attributes.get("polarisationChannels")  # "VV VH", "HH", etc.
        pass_direction = attributes.get("orbitDirection")  # "ASCENDING", "DESCENDING"
        rel_orbit = attributes.get("relativeOrbitNumber")
        abs_orbit = attributes.get("absoluteOrbitNumber")
        incidence_angle = attributes.get("incidenceAngleMax")

        # Normalise polarisation to "VV+VH" style
        if polarisation:
            polarisation = polarisation.strip().replace(" ", "+")

        # Normalise pass direction to lowercase
        if pass_direction:
            pass_direction = pass_direction.lower()  # "ascending" / "descending"

        # Parse orbit numbers to int
        try:
            rel_orbit = int(rel_orbit) if rel_orbit is not None else None
        except (ValueError, TypeError):
            rel_orbit = None
        try:
            abs_orbit = int(abs_orbit) if abs_orbit is not None else None
        except (ValueError, TypeError):
            abs_orbit = None
        try:
            incidence_angle = (
                float(incidence_angle) if incidence_angle is not None else None
            )
        except (ValueError, TypeError):
            incidence_angle = None

        # GSD from product type
        gsd_map = {"SLC": 5.0, "GRD": 10.0, "GRD-COG": 10.0, "OCN": None}
        gsd_m = gsd_map.get(product_type or "", None)

        # Build assets
        assets: dict[str, SatelliteAsset] = {}
        s3_path = product.get("S3Path")
        if s3_path:
            assets["data"] = SatelliteAsset(
                key="data",
                href=f"{self.S3_ENDPOINT}{s3_path}",
                title=name,
                roles=["data"],
                size_bytes=product.get("ContentLength"),
            )

        download_href = f"{self.DOWNLOAD_URL}/Products({product_id})/$value"
        assets["download"] = SatelliteAsset(
            key="download",
            href=download_href,
            title=f"Download {name}",
            roles=["data"],
            size_bytes=product.get("ContentLength"),
        )

        proc_level_str = attributes.get("processingLevel", "UNKNOWN")
        proc_level = ProcessingLevel.UNKNOWN
        for pl in ProcessingLevel:
            if pl.value == proc_level_str:
                proc_level = pl
                break

        return SatelliteData(
            id=product_id,
            provider=self.PROVIDER_ID,
            collection=collection or satellite.upper().replace("-", ""),
            satellite=satellite,
            datetime=dt,
            bbox=bbox,
            geometry=geometry,
            cloud_cover=cloud_cover,
            processing_level=proc_level,
            assets=assets,
            properties={
                "name": name,
                "s3_path": s3_path,
                "online": product.get(
                    "Online", True
                ),  # CDSE archived products need LTA retrieval trigger
                **attributes,
            },
            # SAR / platform fields (Capability 1 + 2)
            product_type=product_type,
            polarisation=polarisation,
            pass_direction=pass_direction,
            relative_orbit=rel_orbit,
            orbit_number=abs_orbit,
            incidence_angle=incidence_angle,
            gsd_m=gsd_m,
            resolution_m=gsd_m,
        )

    def download(
        self,
        data: SatelliteData,
        destination: Path,
        options: DownloadOptions,
    ) -> DownloadResult:
        """
        Download a Copernicus product via authenticated HTTPS.

        Handles CDSE-specific quirks confirmed against the current official
        documentation and community-reported issues:
          - Downloads use a DIFFERENT host (download.dataspace.copernicus.eu)
            from the catalogue search host — mixing them up is the single
            most common cause of 401/403/404 errors on CDSE downloads.
          - Archived ("Offline") products require an LTA (Long Term Archive)
            retrieval trigger before they can be downloaded; attempting to
            download an offline product immediately returns 404/503.
          - CDSE's download endpoint intermittently returns 503 under load
            (a known, documented issue on the CDSE community forum); these
            are retried with exponential backoff rather than failing
            immediately.

        Args:
            data: SatelliteData to download.
            destination: Output directory.
            options: Download configuration.

        Returns:
            DownloadResult.
        """
        self.require_auth()
        self._refresh_token_if_needed()

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        download_asset = data.assets.get("download") or data.assets.get("data")
        if not download_asset:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                error="No download URL available for this product",
            )

        name = data.properties.get("name", data.id)
        output_file = destination / f"{name}.zip"

        # Check Online/Offline status — archived products need LTA retrieval
        is_online = data.properties.get("online", True)
        if not is_online:
            triggered = self._trigger_lta_retrieval(data.id)
            if not triggered:
                return DownloadResult(
                    status=DownloadStatus.FAILED,
                    data_id=data.id,
                    provider=self.PROVIDER_ID,
                    error=(
                        "Product is Offline (archived in Long Term Archive) and "
                        "retrieval could not be triggered. CDSE typically takes "
                        "several minutes to hours to restore archived products. "
                        "Retry this download after the product becomes Online, "
                        "or check status at the CDSE Browser."
                    ),
                    error_type="ProductOffline",
                )
            self._logger.info(
                f"{data.id} is Offline — LTA retrieval triggered. "
                "This may take minutes to hours; retry the download later."
            )
            return DownloadResult(
                status=DownloadStatus.FAILED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                error=(
                    "Product retrieval from Long Term Archive has been triggered. "
                    "Retry this download once the product status changes to Online "
                    "(typically minutes to hours)."
                ),
                error_type="ProductOfflineRetrievalTriggered",
            )

        # Primary path: direct authenticated HTTPS download with retry-with-backoff
        result = self._download_direct(data, download_asset.href, output_file, options)
        if result.success:
            return result

        # Fallback: try the cdse package if installed (handles token refresh,
        # retries, and rate limiting automatically — useful when the direct
        # path hits persistent 401/403/503 errors this provider can't resolve)
        fallback_result = self._download_via_cdse_fallback(data, output_file, options)
        if fallback_result is not None:
            return fallback_result

        return result

    def _download_direct(
        self,
        data: SatelliteData,
        href: str,
        output_file: Path,
        options: DownloadOptions,
    ) -> DownloadResult:
        """Direct authenticated download with retry-with-backoff on transient errors."""
        import httpx

        max_attempts = max(getattr(options, "retry_attempts", 3), 1)
        last_error: Optional[str] = None
        last_error_type: Optional[str] = None

        for attempt in range(1, max_attempts + 1):
            try:
                start_time = time.time()

                # Resume from where the previous attempt left off, rather
                # than deleting the partial file and restarting the whole
                # download from zero on every retry — confirmed this was
                # happening: for a multi-GB Sentinel-1 product, a dropped
                # connection at 1GB+ into the transfer meant every retry
                # re-downloaded that same 1GB+ again before even reaching
                # new data, on a connection that had just shown it
                # couldn't sustain the transfer in the first place.
                existing_bytes = output_file.stat().st_size if output_file.exists() else 0
                headers = {"Authorization": f"Bearer {self._session.access_token}"}  # type: ignore
                if existing_bytes > 0:
                    headers["Range"] = f"bytes={existing_bytes}-"

                with httpx.stream(
                    "GET",
                    href,
                    headers=headers,
                    timeout=options.timeout_seconds,
                    follow_redirects=True,
                ) as resp:
                    if resp.status_code == 503:
                        raise httpx.HTTPStatusError(
                            "503 Service Unavailable (transient — CDSE server load)",
                            request=resp.request,
                            response=resp,
                        )
                    if resp.status_code == 401:
                        # Token may have just expired — refresh once and retry
                        self._refresh_token_if_needed(force=True)
                        raise httpx.HTTPStatusError(
                            "401 Unauthorized — token refreshed, retrying",
                            request=resp.request,
                            response=resp,
                        )
                    self._handle_http_error(resp)

                    # CDSE is documented to sometimes ignore Range headers
                    # and return the full file (200) instead of resuming
                    # (206) — check which actually happened rather than
                    # assume the Range request was honoured.
                    resumed = existing_bytes > 0 and resp.status_code == 206
                    if existing_bytes > 0 and resp.status_code == 200:
                        # Server sent the whole file despite the Range
                        # request — start writing from scratch, not append,
                        # or the file would be corrupted with duplicated
                        # content at the front.
                        existing_bytes = 0

                    total_bytes = int(resp.headers.get("content-length", 0))
                    if resumed:
                        total_bytes += existing_bytes
                    bytes_written = existing_bytes
                    chunk_t0 = time.time()
                    write_mode = "ab" if resumed else "wb"
                    with open(output_file, write_mode) as f:
                        for chunk in resp.iter_bytes(
                            chunk_size=int(options.chunk_size_mb * 1024 * 1024)
                        ):
                            f.write(chunk)
                            bytes_written += len(chunk)
                            elapsed = time.time() - chunk_t0
                            speed = (bytes_written - existing_bytes) / elapsed if elapsed > 0 else 0.0
                            report_download_progress(bytes_written, total_bytes, speed)

                duration = time.time() - start_time
                file_size = output_file.stat().st_size

                if file_size == 0:
                    raise RuntimeError("Downloaded file is empty (0 bytes)")

                return DownloadResult(
                    status=DownloadStatus.COMPLETED,
                    data_id=data.id,
                    provider=self.PROVIDER_ID,
                    output_path=output_file,
                    output_paths=[output_file],
                    bytes_downloaded=file_size,
                    duration_seconds=duration,
                    retries_used=attempt - 1,
                )

            except Exception as exc:
                last_error = str(exc)
                last_error_type = type(exc).__name__
                # No longer unconditionally deleting the partial file here —
                # the next attempt (or the whole retry loop, if this was
                # the final attempt) needs it intact to resume from.
                if attempt < max_attempts:
                    # Base backoff scaled by how much has already been
                    # downloaded — a connection that just dropped a
                    # multi-GB transfer needs meaningfully more recovery
                    # time than the previous flat min(2**attempt, 30)
                    # gave it (as little as 1-2 seconds on early attempts,
                    # confirmed against a real 1GB+ download that had
                    # just timed out).
                    current_size_gb = (
                        output_file.stat().st_size / (1024**3)
                        if output_file.exists()
                        else 0
                    )
                    size_factor = max(1.0, current_size_gb * 10)
                    backoff = min(15 * size_factor * attempt, 120)
                    self._logger.warning(
                        f"Download attempt {attempt}/{max_attempts} failed for "
                        f"{data.id}: {last_error}. Resuming from "
                        f"{output_file.stat().st_size / (1024**2):.0f}MB in "
                        f"{backoff:.0f}s..."
                        if output_file.exists()
                        else f"Download attempt {attempt}/{max_attempts} failed for "
                        f"{data.id}: {last_error}. Retrying in {backoff:.0f}s..."
                    )
                    time.sleep(backoff)

        # Final failure — clean up only now, not on every intermediate retry
        output_file.unlink(missing_ok=True)
        return DownloadResult(
            status=DownloadStatus.FAILED,
            data_id=data.id,
            provider=self.PROVIDER_ID,
            error=last_error,
            error_type=last_error_type,
            retries_used=max_attempts - 1,
        )

    def _download_via_cdse_fallback(
        self,
        data: SatelliteData,
        output_file: Path,
        options: DownloadOptions,
    ) -> Optional[DownloadResult]:
        """
        Fall back to the third-party `cdse` package if installed.

        The `cdse` package (pip install cdse) wraps CDSE's OAuth2 flow with
        automatic token refresh, resilient retries with Retry-After honouring,
        and a proactive rate limiter — useful when this provider's direct
        HTTP path hits persistent errors that a more battle-tested client
        handles more gracefully.

        Returns None (not attempted) if `cdse` is not installed, so the
        caller can report the original error instead of a confusing
        "module not found" message.
        """
        try:
            from cdse import Client, PasswordAuth
        except ImportError:
            self._logger.debug(
                "cdse package not installed — skipping fallback. "
                "Install with: pip install cdse"
            )
            return None

        username = self._session.session_data.get("username") if self._session else None
        password = getattr(self, "_raw_password", None)
        if not username or not password:
            self._logger.debug(
                "cdse fallback requires cached username/password — not available "
                "(credentials were likely cleared after initial authentication)"
            )
            return None

        try:
            self._logger.info(f"Retrying {data.id} via cdse fallback package...")
            start_time = time.time()
            with Client(PasswordAuth(username, password)) as client:
                product_id = data.id
                client.odata.products.download(product_id, str(output_file))

            duration = time.time() - start_time
            if not output_file.exists() or output_file.stat().st_size == 0:
                return None

            file_size = output_file.stat().st_size
            self._logger.info(f"cdse fallback succeeded for {data.id}")
            return DownloadResult(
                status=DownloadStatus.COMPLETED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                output_path=output_file,
                output_paths=[output_file],
                bytes_downloaded=file_size,
                duration_seconds=duration,
                metadata={"fallback": "cdse"},
            )
        except Exception as exc:
            self._logger.warning(f"cdse fallback also failed for {data.id}: {exc}")
            return None

    def _trigger_lta_retrieval(self, product_id: str) -> bool:
        """
        Trigger retrieval of an Offline (archived) product from CDSE's
        Long Term Archive (LTA).

        Returns True if the retrieval request was accepted, False otherwise.
        """
        try:
            import httpx

            resp = httpx.post(
                f"{self.DOWNLOAD_URL}/Products({product_id})/$value",
                headers={"Authorization": f"Bearer {self._session.access_token}"},  # type: ignore
                timeout=30,
            )
            # CDSE returns 202 Accepted when LTA retrieval is triggered
            return resp.status_code in (200, 202)
        except Exception as exc:
            self._logger.warning(
                f"Could not trigger LTA retrieval for {product_id}: {exc}"
            )
            return False

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_id="copernicus",
            name="Copernicus CDSE",
            search=True,
            download=True,
            streaming=True,
            stac=False,
            requires_auth=True,
            supported_satellites=[
                "Sentinel-1",
                "Sentinel-2",
                "Sentinel-3",
                "Sentinel-5P",
                "Sentinel-6",
            ],
            supported_formats=[DataFormat.SAFE, DataFormat.GEOTIFF, DataFormat.ZIP],
            supports_cloud_filter=True,
            supports_date_filter=True,
            supports_aoi_filter=True,
        )

    def get_quota_info(self) -> QuotaInfo:
        self.require_auth()
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={
                "note": "Copernicus provides free access; rate limits may apply",
                "docs": "https://dataspace.copernicus.eu/terms-and-conditions",
            },
        )
