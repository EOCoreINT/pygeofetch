"""
PyGeoFetch main engine.

The ``PyGeoFetch`` class is the single entry point for the Python API.
It wires together authentication, search, and download subsystems and
exposes a clean high-level interface.

Example::

    from pathlib import Path
    from pygeofetch import PyGeoFetch
    from pygeofetch.models import SearchQuery

    sb = PyGeoFetch()
    sb.auth.add("usgs", username="user", password="pass")

    results = sb.search(
        SearchQuery(
            bbox=(-74.1, 40.6, -73.7, 40.9),
            start_date="2024-01-01",
            end_date="2024-06-01",
            cloud_cover_max=20,
        ),
        providers=["usgs", "copernicus", "aws_earth"],
    )

    sb.download(results[:5], destination=Path("./data/"))
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pygeofetch.core.authenticator import AuthManager
from pygeofetch.core.downloader import AdaptiveDownloader
from pygeofetch.core.logging import configure_logging, get_logger
from pygeofetch.core.searcher import FederatedSearcher
from pygeofetch.models.satellite_data import SatelliteData
from pygeofetch.processing.batch import BatchProcessor
from pygeofetch.processing.indices import SpectralIndices
from pygeofetch.processing.postprocessor import PostProcessor
from pygeofetch.processing.preprocessor import Preprocessor
from pygeofetch.processing.sar import SARProcessor
from pygeofetch.utils.geo_utils import _normalise_satellite_name


# Backward-compatible alias
def setup_logging(
    level: str = "INFO", use_rich: bool = True, log_file=None, **kwargs
) -> None:
    """Thin wrapper kept for backward compatibility with engine.__init__."""
    configure_logging(level=level, log_file=str(log_file) if log_file else None)


if TYPE_CHECKING:
    from shapely.geometry import Polygon

    from pygeofetch.models.download_task import DownloadOptions, DownloadResult
    from pygeofetch.models.search_query import SearchQuery
    from pygeofetch.validation.optical_validator import OpticalValidationConfig

logger = get_logger(__name__)


def _dedup(lst: list) -> list:
    """Deduplicate a list preserving order."""
    # dict.fromkeys preserves insertion order (Python 3.7+)
    return list(dict.fromkeys(lst))


class PyGeoFetch:
    """
    Universal satellite data pipeline.

    ``PyGeoFetch`` is the top-level API object that provides access
    to all subsystems:

    - ``sb.auth``     – :class:`~pygeofetch.core.authenticator.AuthManager`
    - ``sb.searcher`` – :class:`~pygeofetch.core.searcher.FederatedSearcher`
    - ``sb.downloader`` – :class:`~pygeofetch.core.downloader.AdaptiveDownloader`

    Attributes:
        auth: Authentication manager for all providers.
        searcher: Federated search engine.
        downloader: Adaptive parallel downloader.
        config: Runtime configuration dict.

    Example::

        sb = PyGeoFetch(log_level="INFO")
        results = sb.search(SearchQuery(bbox=(-74, 40, -73, 41)))
        sb.download(results, Path("./data/"))
    """

    def __init__(
        self,
        config_path: Path | None = None,
        log_level: str = "INFO",
        log_json: bool = False,
        cache_ttl: int = 3600,
        max_search_workers: int = 8,
        progress_callback: Callable | None = None,
        auth_backend: str = "file",
        validate_optical: bool = False,
        optical_validation_config: "OpticalValidationConfig | None" = None,
    ) -> None:
        """
        Initialize PyGeoFetch.

        Args:
            config_path: Optional path to a custom config YAML file.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
            log_json: Emit JSON-formatted log lines if True.
            cache_ttl: Search result cache TTL in seconds.
            max_search_workers: Maximum parallel provider searches.
            progress_callback: Optional callable(DownloadProgress) for progress events.
            validate_optical: Default toggle for optical preflight
                validation (AOI coverage, cloud cover, required bands,
                processing level, temporal bounds -- see
                :class:`~pygeofetch.validation.optical_validator.OpticalValidationConfig`)
                applied automatically by :meth:`search` and
                :meth:`download`. Defaults to False -- fully opt-in,
                so existing behavior is unchanged unless explicitly
                enabled. Can still be overridden per-call via the
                ``validate_optical`` parameter on either method.
                Requires the ``geo`` or ``insar`` extra (for
                ``shapely``) -- only imported when actually enabled,
                so a base install is unaffected either way.
            optical_validation_config: Config to use when optical
                validation is enabled. Defaults to
                ``OpticalValidationConfig()``'s own defaults if not
                given.
        """
        setup_logging(level=log_level, use_rich=not log_json)
        logger.debug("Initializing PyGeoFetch")

        self.auth = AuthManager(storage_backend=auth_backend)
        self.searcher = FederatedSearcher(
            auth_manager=self.auth,
            cache_ttl=cache_ttl,
            max_workers=max_search_workers,
        )
        self.downloader = AdaptiveDownloader(
            auth_manager=self.auth,
            progress_callback=progress_callback,
        )
        self.config: dict[str, Any] = {}
        if config_path:
            self._load_config(config_path)

        # Processing subsystems (lazy-free — always available)
        self.preprocess = Preprocessor()
        self.indices = SpectralIndices()
        self.post = PostProcessor()
        self.sar = SARProcessor()
        self.batch = BatchProcessor(engine=self)

        # Optical preflight validation is opt-in and its dependency
        # (shapely) is only imported the first time it's actually
        # used — see _get_optical_validator().
        self.validate_optical = validate_optical
        self.optical_validation_config = optical_validation_config
        self._optical_validator = None  # lazily built by _get_optical_validator()

        logger.info("PyGeoFetch ready")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: SearchQuery,
        providers: list[str] | None = None,
        use_cache: bool = True,
        validate_optical: bool | None = None,
    ) -> list[SatelliteData]:
        """
        Search for satellite data across one or more providers.

        Args:
            query: Search parameters (bbox, dates, cloud cover, etc.).
            providers: Provider IDs to query. Queries all authenticated/free
                providers when omitted.
            use_cache: Use in-memory result cache (default True).
            validate_optical: Run optical preflight validation
                (:class:`~pygeofetch.validation.optical_validator.OpticalPreflightValidator`)
                on the results before returning them, filtering out any
                that fail a hard check (missing bands, 0% AOI coverage,
                wrong processing level, out-of-range date -- see
                ``OpticalValidationConfig`` for the full set and
                defaults). ``None`` (the default) falls back to the
                instance-level ``self.validate_optical`` set in
                ``__init__``; pass ``True``/``False`` explicitly to
                override just this call. The AOI is derived
                automatically from ``query.bbox``/``query.geometry`` --
                if the query has neither, validation is skipped with a
                logged warning rather than raising. Not appropriate for
                SAR/InSAR searches (the default ``required_bands`` are
                optical band names) -- leave this off for those, or
                pass a config with ``check_required_bands=False``.

        Returns:
            List of :class:`~pygeofetch.models.satellite_data.SatelliteData`
            sorted by relevance score descending.

        Example::

            results = sb.search(
                SearchQuery(
                    bbox=(-74.1, 40.6, -73.7, 40.9),
                    start_date="2024-01-01",
                    cloud_cover_max=10,
                ),
                providers=["copernicus", "aws_earth"],
            )

            # With optical preflight validation enabled for just this call:
            results = sb.search(query, validate_optical=True)
        """
        # Route SLC products away from GRD-only providers
        _product = getattr(query, "product_type", None) or ""
        _effective = list(providers) if providers else []
        if _product.upper() == "SLC" and _effective:
            _effective = self._route_slc_providers(_effective)
        results = self.searcher.search(query, providers=_effective, use_cache=use_cache)
        self._warn_if_outdated_constellation(results, query)

        effective_validate = (
            self.validate_optical if validate_optical is None else validate_optical
        )
        if effective_validate and results:
            results = self._run_optical_preflight(
                results,
                aoi=self._derive_aoi_polygon(query),
                start_date=getattr(query, "start_date", None),
                end_date=getattr(query, "end_date", None),
            )
        return results

    from pygeofetch.validation.optical_validator import (
        OpticalPreflightValidator,
        OpticalValidationConfig,
    )

    def _get_optical_validator(self) -> "OpticalPreflightValidator":
        """Lazily build (and cache) the OpticalPreflightValidator.
        Imported here, not at module level, so pygeofetch's core
        import never requires shapely -- only actually enabling
        optical validation does."""
        if self._optical_validator is None:
            from pygeofetch.validation.optical_validator import (
                OpticalPreflightValidator,
                OpticalValidationConfig,
            )

            self._optical_validator = OpticalPreflightValidator(
                self.optical_validation_config or OpticalValidationConfig()
            )
        return self._optical_validator

    @staticmethod
    def _derive_aoi_polygon(query: "SearchQuery") -> "Polygon | None":
        """Build a shapely Polygon from a SearchQuery's bbox or
        geometry, so callers don't have to specify the AOI twice (once
        in the query, once for validation). Returns None -- rather
        than raising -- when the query has neither, so validation
        degrades to "skip AOI-dependent checks" instead of failing the
        whole search."""
        bbox = getattr(query, "bbox", None)
        geometry = getattr(query, "geometry", None)
        try:
            from shapely.geometry import box, shape
        except ImportError:
            logger.warning(
                "Optical validation requires shapely (pip install "
                '"pygeofetch[geo]") -- AOI-dependent checks will be skipped.'
            )
            return None
        if geometry:
            try:
                return shape(geometry)
            except Exception as exc:
                logger.warning(
                    f"Could not build AOI polygon from query.geometry: {exc}"
                )
        if bbox is not None:
            try:
                return box(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)
            except AttributeError:
                # already a plain 4-tuple
                return box(*bbox)
        return None

    def _run_optical_preflight(
        self,
        results: list[SatelliteData],
        aoi: "Polygon | None",
        start_date=None,
        end_date=None,
    ) -> list[SatelliteData]:
        if aoi is None:
            logger.warning(
                "validate_optical is enabled but no AOI could be derived "
                "from the query (no bbox or geometry) -- AOI-dependent "
                "checks will be skipped; other checks still run."
            )
        validator = self._get_optical_validator()
        return validator.run_preflight(
            results, aoi, start_date=start_date, end_date=end_date
        )

    def search_and_save(
        self,
        query: SearchQuery,
        output: Path,
        providers: list[str] | None = None,
    ) -> list[SatelliteData]:
        """
        Search and save results to a GeoJSON file.

        Args:
            query: Search parameters.
            output: Output GeoJSON file path.
            providers: Provider IDs to query.

        Returns:
            List of results (also written to *output*).
        """
        results = self.search(query, providers=providers)
        self.searcher.save_results(results, output)
        return results

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    # ── SLC product routing ─────────────────────────────────────────────────

    _SLC_CAPABLE = {
        "copernicus",
        "alaska_satellite_facility",
        "asf_vertex",
        "asf",
        "eodag",
        "copernicus_dataspace",
        "nasa_earthdata",
    }
    _GRD_ONLY = {"planetary_computer", "aws_earth", "element84"}

    def _warn_if_outdated_constellation(self, results: list, query) -> None:
        """Log a warning if results contain only decommissioned satellites
        for a search whose OWN requested date range extends past S1A's
        real decommissioning date — not just because today happens to be
        past it. A search for genuine historical data (e.g. November
        2024, well before S1A's actual 1 July 2026 end of mission) must
        never trigger this, regardless of what today's date is; S1A
        results from before decommissioning are completely legitimate."""
        from datetime import date, datetime

        S1A_DECOM = date(2026, 7, 1)

        query_end = getattr(query, "end_date", None)
        if query_end is None:
            # No end date specified -- fall back to whether the search
            # was actually run after decommissioning (the old behaviour,
            # but only as a last resort when there's no query date to
            # check against).
            query_end_date = date.today()
        elif isinstance(query_end, str):
            query_end_date = datetime.fromisoformat(query_end.replace("Z", "")).date()
        elif isinstance(query_end, datetime):
            query_end_date = query_end.date()
        elif isinstance(query_end, date):
            query_end_date = query_end
        else:
            query_end_date = date.today()

        if query_end_date >= S1A_DECOM:
            platforms = {getattr(r, "satellite", "") or "" for r in results}
            normalised = {_normalise_satellite_name(p) for p in platforms}
            if any("S1" in p for p in normalised):  # only check if S1 results
                if not any(p in {"S1C", "S1D"} for p in normalised):
                    logger.warning(
                        "Search returned only S1A/S1B results for a date range "
                        "extending past Sentinel-1A's real decommissioning date "
                        "(1 July 2026). This may indicate a provider collection "
                        "routing issue. Check platform filter settings."
                    )

    def _route_slc_providers(self, providers: list) -> list:
        """
        SLC products are NOT available from Planetary Computer, AWS Earth, or Element84.
        Reroute automatically to capable providers.
        """
        routed = []
        for p in providers:
            if p in self._GRD_ONLY:
                fallback = "copernicus"
                routed.append(fallback)
                logger.info(
                    "Provider %r does not host SLC products. Automatically routing to %r instead.",
                    p,
                    fallback,
                )
            elif p in self._SLC_CAPABLE:
                routed.append(p)
            else:
                routed.append(
                    p
                )  # unknown provider — pass through, let it fail naturally
        # Deduplicate, preserve order
        return _dedup(routed)

    def download(
        self,
        data: SatelliteData | list[SatelliteData],
        destination: Path,
        options: DownloadOptions | None = None,
        item_done_callback=None,
        validate_optical: bool | None = None,
        aoi: "Polygon | None" = None,
    ) -> list[DownloadResult]:
        """
        Download one or more satellite data products.

        Args:
            data: Single item or list of SatelliteData to download.
            destination: Target directory (created if it does not exist).
            options: Download configuration (parallel, retry, checksum, etc.).
            validate_optical: Run optical preflight validation before
                downloading, same checks as :meth:`search`'s
                ``validate_optical``. ``None`` (the default) falls back
                to the instance-level ``self.validate_optical``; pass
                ``True``/``False`` to override just this call. Useful
                as a final gate right before spending bandwidth,
                independent of whether validation already ran at
                search time -- e.g. for items loaded via
                :meth:`download_from_file` that never went through
                :meth:`search` at all. Items that fail are **not**
                attempted; a synthesized ``DownloadResult`` with
                ``status=FAILED`` and an explanatory ``error`` is
                returned for each in the item's original position, so
                the returned list's length and order always match
                ``data`` regardless of how many items were rejected.
            aoi: AOI polygon for AOI-dependent checks
                (``check_aoi_coverage``, ``check_nodata_margins``).
                Optional -- when not given, those two checks are
                skipped (logged at debug level) while every other
                check still runs; there's no query to derive an AOI
                from at download time the way there is in
                :meth:`search`, so this has to be supplied explicitly
                if you want those checks here too.

        Returns:
            List of :class:`~pygeofetch.models.download_task.DownloadResult`.

        Example::

            results = sb.download(
                search_results[:10],
                destination=Path("./data/"),
                options=DownloadOptions(parallel=4, verify_checksum=True),
            )

            # With optical preflight validation as a final gate:
            results = sb.download(search_results, destination="./data", validate_optical=True)
        """
        if isinstance(data, SatelliteData):
            data = [data]

        effective_validate = (
            self.validate_optical if validate_optical is None else validate_optical
        )
        if effective_validate and data:
            return self._download_with_optical_preflight(
                data, destination, options, item_done_callback, aoi
            )

        return self.downloader.download_many(
            data, Path(destination), options, item_done_callback=item_done_callback
        )

    def _download_with_optical_preflight(
        self,
        data: list[SatelliteData],
        destination: Path,
        options: DownloadOptions | None,
        item_done_callback,
        aoi: "Polygon | None",
    ) -> list[DownloadResult]:
        """
        Run optical preflight before downloading, without breaking the
        guarantee that the returned list's length and order always
        match the input ``data`` list: rejected items are never passed
        to the real downloader at all -- a synthesized
        ``DownloadResult(status=FAILED, ...)`` stands in for each, at
        its original position, alongside the real results for the
        items that passed.
        """
        from pygeofetch.models.download_task import DownloadResult, DownloadStatus

        validator = self._get_optical_validator()
        if aoi is None:
            logger.debug(
                "download(validate_optical=True) called without an aoi -- "
                "AOI-dependent checks will be skipped for this batch."
            )

        safe_items: list[SatelliteData] = []
        rejected: dict[int, DownloadResult] = {}
        for idx, item in enumerate(data):
            report = validator.validate_scene(item, aoi)
            for issue in report.issues:
                log = logger.error if issue.severity == "ERROR" else logger.warning
                log(f"Scene {report.scene_id} [{issue.code}]: {issue.message}")
            if report.passed:
                safe_items.append(item)
            else:
                worst = report.errors[0] if report.errors else report.warnings[0]
                rejected[idx] = DownloadResult(
                    status=DownloadStatus.FAILED,
                    data_id=getattr(item, "id", str(idx)),
                    provider=getattr(item, "provider", "unknown"),
                    error=f"rejected by optical preflight [{worst.code}]: {worst.message}",
                )

        logger.info(
            f"Optical preflight before download: {len(safe_items)}/{len(data)} "
            "items passed; downloading those, skipping the rest."
        )

        real_results = (
            self.downloader.download_many(
                safe_items,
                Path(destination),
                options,
                item_done_callback=item_done_callback,
            )
            if safe_items
            else []
        )

        merged: list[DownloadResult] = []
        real_iter = iter(real_results)
        for idx in range(len(data)):
            merged.append(rejected[idx] if idx in rejected else next(real_iter))
        return merged

    def download_from_file(
        self,
        search_results_path: Path,
        destination: Path,
        options: DownloadOptions | None = None,
    ) -> list[DownloadResult]:
        """
        Load search results from a GeoJSON file and download them.

        Args:
            search_results_path: Path to GeoJSON file from :meth:`search_and_save`.
            destination: Target directory.
            options: Download configuration.

        Returns:
            List of DownloadResults.
        """
        data_list = FederatedSearcher.load_results(search_results_path)
        logger.info(f"Loaded {len(data_list)} items from {search_results_path}")
        return self.download(data_list, destination, options)

    # ------------------------------------------------------------------
    # Auth convenience methods
    # ------------------------------------------------------------------

    def add_credentials(
        self,
        provider: str,
        username: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        token: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Store credentials for a provider.

        Args:
            provider: Provider identifier (e.g. ``"usgs"``, ``"copernicus"``).
            username: Username or email.
            password: Password or secret.
            api_key: API key (alternative to username/password).
            **kwargs: Extra credentials passed through to the provider.

        Example::

            sb.add_credentials("usgs", username="user", password="s3cr3t")
            sb.add_credentials("planet", api_key="PL_KEY")
        """
        creds = {
            k: v
            for k, v in {
                "username": username,
                "password": password,
                "api_key": api_key,
                "client_id": client_id,
                "client_secret": client_secret,
                "token": token,
                "access_key": access_key,
                "secret_key": secret_key,
            }.items()
            if v is not None
        }
        creds.update({k: str(v) for k, v in kwargs.items() if v is not None})
        self.auth.add_credentials(provider, creds)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """
        Return a summary of the current system status.

        Returns:
            Dict with keys: ``providers_authenticated``, ``providers_free``,
            ``cache_entries``, ``version``.
        """
        from pygeofetch import __version__
        from pygeofetch.providers import get_free_providers

        try:
            authed = [item["provider"] for item in self.auth.list()]
        except Exception:
            authed = []
        free = list(get_free_providers())
        return {
            "version": __version__,
            "providers_authenticated": authed,
            "providers_free": free,
            "cache_entries": len(self.searcher.cache._cache),
        }

    def clear_cache(self) -> int:
        """
        Clear the in-memory search result cache.

        Returns:
            Number of cache entries removed.
        """
        count = self.searcher.cache.clear()
        logger.info(f"Cleared {count} cache entries")
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_config(self, path: Path) -> None:
        """Load additional configuration from a YAML file."""
        import yaml

        with open(path) as f:
            extra = yaml.safe_load(f) or {}
        self.config.update(extra)
        logger.debug(f"Loaded extra config from {path}")

    def pipeline(self, name: str):
        """Create a chainable processing pipeline."""
        from pygeofetch.processing.pipeline import ProcessingPipeline

        return ProcessingPipeline(name=name, engine=self)

    def batch_process(self, inputs, chain, output_dir=".", parallel=2):
        """Batch process multiple files through a processing chain."""
        return self.batch.process(
            inputs, chain, output_dir=output_dir, parallel=parallel
        )

    def fetch_orbit_file(
        self,
        product_name: str,
        output_dir: str = "./orbits/",
        orbit_type: str = "precise",
    ) -> str | None:
        """
        Download the precise orbit file for a Sentinel-1 SLC product.

        Precise orbit files (POEORB) are published 21 days after acquisition
        and are required for millimetre-precision InSAR processing with SNAP.
        Restituted orbits (RESORB) are available within ~3 hours for near-real-time.

        Args:
            product_name: Sentinel-1 product name or scene ID.
                          Must contain an 8-digit date and 6-digit time, e.g.
                          "S1C_IW_SLC__1SDV_20260601T053000_..."
            output_dir:   Cache directory for orbit files. Created if absent.
            orbit_type:   "precise" (recommended for InSAR, 21-day delay)
                          | "restituted" (~3-hour delay, near-real-time)

        Returns:
            Absolute path to orbit file as string, or None if not available.

        Example::

            path = engine.fetch_orbit_file(
                product_name="S1C_IW_SLC__1SDV_20260601T053000_...",
                output_dir="./orbits/",
                orbit_type="precise",
            )
        """
        from pygeofetch.core.orbits import fetch_orbit_file as _fetch_orbit

        return _fetch_orbit(product_name, output_dir, orbit_type)

    def __repr__(self) -> str:
        authed = [item["provider"] for item in self.auth.list()]
        return f"PyGeoFetch(authenticated={authed}, cache_entries={len(self.searcher.cache._cache)})"
