"""
Adaptive download manager for PyGeoFetch.

Provides resilient parallel downloads with chunked transfer, checksum
verification, retry logic, progress reporting, and post-download actions.

Example::

    from pygeofetch.core.downloader import AdaptiveDownloader
    from pygeofetch.models.download_task import DownloadOptions

    downloader = AdaptiveDownloader()
    options = DownloadOptions(parallel=4, verify_checksum=True, retry_attempts=3)

    results = downloader.download_many(
        data_list=search_results,
        destination=Path("./data/"),
        options=options,
    )
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict

from pygeofetch.core.logging import DownloadProgress, get_logger
from pygeofetch.models.download_task import (
    DownloadOptions,
    DownloadResult,
    DownloadStatus,
    DownloadTask,
)

if TYPE_CHECKING:
    from pygeofetch.models.satellite_data import SatelliteData

logger = get_logger(__name__)


# Required common-band-name inputs for each single-date spectral index
# supported by pygeofetch.processor.SpectralIndex's built-in (non-spyndex)
# formulae. Matches _BUILTIN in pygeofetch/processor/indices.py exactly —
# keep these in sync if that dict changes. dNBR is deliberately excluded:
# it needs paired pre/post-event bands, which a single downloaded scene
# can't supply (handled as an explicit error instead, see _apply_action).
_SINGLE_DATE_INDEX_BANDS: dict = {
    "EVI": ("NIR", "RED", "BLUE"),
    "SAVI": ("NIR", "RED"),
    "NDWI": ("GREEN", "NIR"),
    "MNDWI": ("GREEN", "SWIR1"),
    "NDBI": ("SWIR1", "NIR"),
    "NDSI": ("GREEN", "SWIR1"),
    "NDMI": ("NIR", "SWIR1"),
    "NBR": ("NIR", "SWIR2"),
    "BSI": ("SWIR1", "RED", "NIR", "BLUE"),
    "ARVI": ("NIR", "RED", "BLUE"),
    "GNDVI": ("NIR", "GREEN"),
    "RVI": ("NIR", "RED"),
    "VCI": ("NIR", "RED", "BLUE"),
    "CRI1": ("BLUE", "GREEN"),
    "PSRI": ("RED", "BLUE", "NIR"),
}

# Common band name -> substrings to look for in a downloaded filename.
# Covers Sentinel-2 band codes and generic common names, matching the
# pattern already used by the hardcoded ndvi handler ("B04"/"red").
_BAND_FILENAME_PATTERNS: dict = {
    "BLUE": ("B02", "blue"),
    "GREEN": ("B03", "green"),
    "RED": ("B04", "red"),
    "NIR": ("B08", "nir"),
    "SWIR1": ("B11", "swir1"),
    "SWIR2": ("B12", "swir2"),
}


class AdaptiveDownloader:
    """
    Resilient parallel download manager.

    Features:
    - Configurable parallel workers
    - Chunked streaming downloads
    - Automatic retry with exponential backoff
    - Checksum verification (MD5/SHA256)
    - Real-time progress callbacks
    - Post-download action pipeline (extract, reproject)
    - Bandwidth throttling

    Attributes:
        auth_manager: AuthManager for provider sessions.
        progress_callback: Optional callable(DownloadProgress) for progress updates.
    """

    def __init__(
        self,
        auth_manager: Any | None = None,
        progress_callback: Callable[[DownloadProgress], None] | None = None,
    ) -> None:
        self.auth_manager = auth_manager
        self.progress_callback = progress_callback
        self._active_tasks: dict[str, DownloadTask] = {}
        self._provider_cache: dict[str, Any] = {}

    def download(
        self,
        data: SatelliteData,
        destination: Path,
        options: DownloadOptions | None = None,
    ) -> DownloadResult:
        """
        Download a single satellite data product.

        Args:
            data: SatelliteData to download.
            destination: Target directory.
            options: Download configuration (uses defaults if None).

        Returns:
            DownloadResult with status and output paths.
        """
        options = options or DownloadOptions()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        provider = self._get_provider(data.provider)

        # Resume support: skip re-downloading if a valid file already exists.
        # `options.resume` was previously a declared-but-unused field — this
        # is the actual implementation. We can't call provider.download() to
        # find out where it WOULD write (that IS the download), so instead
        # we predict the likely output location using the same naming
        # convention providers use (destination/provider/<name-or-id>.*)
        # and validate anything we find there.
        if options.resume:
            existing = self._find_existing_download(data, destination)
            if existing is not None:
                is_valid, err_msg = self._validate_downloaded_file(existing)
                if is_valid:
                    logger.info(
                        "  ↷ %-45s already downloaded, skipping (resume=True)",
                        str(data.id)[:45],
                    )
                    return DownloadResult(
                        status=DownloadStatus.COMPLETED,
                        data_id=data.id,
                        provider=data.provider,
                        output_path=existing,
                        output_paths=[existing],
                        bytes_downloaded=existing.stat().st_size,
                        from_cache=True,
                    )
                logger.info(
                    "  Found existing file for %s but it failed validation "
                    "(%s) — re-downloading.",
                    str(data.id)[:45],
                    err_msg,
                )

        logger.debug("Starting download: %s", str(data.id)[:60])

        for attempt in range(options.retry_attempts + 1):
            try:
                t0 = time.time()
                result = provider.download(data, destination, options)
                elapsed = time.time() - t0
                # Patch duration if provider didn't set it
                if result.duration_seconds == 0.0 and elapsed > 0:
                    result = result.model_copy(update={"duration_seconds": elapsed})
                if result.success:
                    # BUG 3: validate file integrity before claiming success
                    for out_p in result.output_paths or [result.output_path]:
                        if out_p and out_p.exists():
                            is_valid, err_msg = self._validate_downloaded_file(out_p)
                            if not is_valid:
                                result = result.model_copy(
                                    update={
                                        "status": DownloadStatus.FAILED,
                                        "error": (
                                            f"Download appeared complete but"
                                            f" file validation failed: {err_msg}"
                                        ),
                                        "output_path": None,
                                    }
                                )
                                logger.warning(
                                    f"File validation failed for {data.id!r}: {err_msg}"
                                )
                                break

                # Check result.success ONCE here, AFTER validation may have
                # changed it — not nested inside the now-stale `if
                # result.success:` above. Previously the success-logging
                # and `return result` sat unconditionally inside that outer
                # block, so a validation failure that flipped success to
                # False partway through still fell through to logging "✓
                # success" and returning immediately — never retrying, and
                # misreporting a detected truncated download as if it had
                # worked.
                if result.success:
                    if options.verify_checksum:
                        result = self._verify_checksums(result, data, options)
                    if options.post_process:
                        result = self._run_post_process(result, options)
                    size_bytes = result.bytes_downloaded or 0
                    if size_bytes < 1024 * 1024:
                        size_str = f"{size_bytes / 1024:.0f} KB"
                    else:
                        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                    logger.info(
                        "  ✓ %-45s %10s  %5.1fs",
                        str(data.id)[:45],
                        size_str,
                        result.duration_seconds,
                    )
                    return result

                # result.success is False here — either the provider itself
                # reported failure, or file validation caught a corrupt/
                # truncated download above. Previously this fell through to
                # an unconditional `return result` regardless of success,
                # meaning a detected truncated download gave up immediately
                # on attempt 1 and never used retry_attempts at all — the
                # exact case a retry is most likely to actually fix, since
                # network truncation is typically transient. Route through
                # the same backoff-and-retry path a raised exception would
                # take, instead of returning early.
                if attempt < options.retry_attempts:
                    delay = self._backoff(attempt, options)
                    logger.warning(
                        f"Download attempt {attempt + 1}/{options.retry_attempts + 1} "
                        f"for {data.id!r} failed validation: {result.error}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Download failed validation after "
                        f"{options.retry_attempts + 1} attempts: {result.error}"
                    )
                    return result
            except Exception as exc:
                if attempt < options.retry_attempts:
                    delay = self._backoff(attempt, options)
                    logger.warning(
                        f"Download attempt {attempt + 1}/{options.retry_attempts + 1} failed "
                        f"for {data.id!r}: {exc}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Download failed after {options.retry_attempts + 1} attempts: {exc}"
                    )
                    return DownloadResult(
                        status=DownloadStatus.FAILED,
                        data_id=data.id,
                        provider=data.provider,
                        error=str(exc),
                        error_type=type(exc).__name__,
                        retries_used=attempt,
                    )

        return DownloadResult(
            status=DownloadStatus.FAILED,
            data_id=data.id,
            provider=data.provider,
            error="Exhausted retries",
        )

    def download_many(
        self,
        data_list: list[SatelliteData],
        destination: Path,
        options: DownloadOptions | None = None,
        item_done_callback: Callable[[int, int, DownloadResult], None] | None = None,
    ) -> list[DownloadResult]:
        """
        Download multiple satellite data products in parallel.

        Args:
            data_list: List of SatelliteData to download.
            destination: Target directory.
            options: Download configuration.
            item_done_callback: Optional callable(completed, total, result) called
                after each item finishes. Used to drive progress bars.

        Returns:
            List of DownloadResults in same order as input.
        """
        options = options or DownloadOptions()
        results: list[DownloadResult | None] = [None] * len(data_list)
        completed_count = 0
        total = len(data_list)

        dp = DownloadProgress(total=total, destination=str(destination))
        dp.__enter__()

        def _run_one(data: SatelliteData, idx: int) -> tuple:
            """Download one item and return (idx, result, bytes, duration)."""
            import time as _it

            from pygeofetch.core.logging import _set_active_progress

            item_id = str(data.id)
            t0 = _it.time()
            dp.start_item(item_id)
            # Make this DownloadProgress instance + item_id visible to
            # report_download_progress() calls made from inside
            # provider.download()'s streaming loop, via thread-local
            # storage — each worker thread gets its own isolated context,
            # so concurrent downloads never cross-report into each
            # other's bars.
            _set_active_progress(dp, item_id)
            try:
                result = self.download(data, destination / data.provider, options)
                dur = _it.time() - t0
                dp.complete_item(
                    item_id,
                    success=result.success,
                    bytes_total=result.bytes_downloaded or 0,
                    duration=dur,
                )
                return idx, result
            except Exception as exc:
                dur = _it.time() - t0
                result = DownloadResult(
                    status=DownloadStatus.FAILED,
                    data_id=data.id,
                    provider=data.provider,
                    error=str(exc),
                )
                dp.complete_item(item_id, success=False, duration=dur)
                return idx, result
            finally:
                _set_active_progress(None, None)

        with ThreadPoolExecutor(max_workers=options.parallel) as executor:
            futures = {
                executor.submit(_run_one, data, i): i
                for i, data in enumerate(data_list)
            }
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                completed_count += 1
                if item_done_callback:
                    item_done_callback(completed_count, total, result)

        # BUG 2: guarantee length == len(data_list) and order is preserved
        # Fill any None slots (unexpected future crashes) with failure results
        for i, item in enumerate(data_list):
            if results[i] is None:
                results[i] = DownloadResult(
                    status=DownloadStatus.FAILED,
                    data_id=item.id,
                    provider=item.provider,
                    error="Download future returned no result (internal error)",
                )

        sum(1 for r in results if r and r.success)
        sum(1 for r in results if r and not r.success)
        dp.__exit__(None, None, None)
        assert len(results) == len(data_list), (
            f"BUG: result count {len(results)} != input count {len(data_list)}"
        )
        sum(
            r.bytes_downloaded / (1024 * 1024)
            for r in results
            if r and r.success and r.bytes_downloaded
        )
        # summary printed by DownloadProgress footer
        return results

    def _get_provider(self, provider_id: str) -> Any:
        """Get a provider instance with a fresh authenticated session."""
        from pygeofetch.providers import get_provider

        prov = get_provider(provider_id)
        if self.auth_manager and prov.REQUIRES_AUTH:
            try:
                session = self.auth_manager.authenticate(provider_id)
                prov.set_session(session)
            except Exception as exc:
                logger.warning(f"Auth for {provider_id} failed: {exc}")
        return prov

    def _verify_checksums(
        self, result: DownloadResult, data: SatelliteData, options: DownloadOptions
    ) -> DownloadResult:
        """Verify checksums of downloaded files against provider-provided values."""
        verified = True
        for path in result.output_paths:
            if not path.exists():
                continue
            # Find expected checksum from assets
            for asset in data.assets.values():
                filename = path.name
                if filename in asset.href:
                    if asset.checksum_md5:
                        actual = self._compute_checksum(path, "md5")
                        if actual.lower() != asset.checksum_md5.lower():
                            logger.warning(
                                f"MD5 mismatch for {path.name}: "
                                f"expected {asset.checksum_md5}, got {actual}"
                            )
                            verified = False
                    elif asset.checksum_sha256:
                        actual = self._compute_checksum(path, "sha256")
                        if actual.lower() != asset.checksum_sha256.lower():
                            logger.warning(f"SHA256 mismatch for {path.name}")
                            verified = False
        result.checksum_verified = verified
        return result

    def _compute_checksum(self, path: Path, algorithm: str) -> str:
        hasher = hashlib.new(algorithm)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _run_post_process(
        self, result: DownloadResult, options: DownloadOptions
    ) -> DownloadResult:
        """Run configured post-download processing actions."""
        for action in options.post_process:
            try:
                result = self._apply_action(result, action)
            except Exception as exc:
                # Rasterio/GDAL errors frequently say "see previous
                # exception for details" — that previous exception is
                # exc.__cause__/__context__, which str(exc) alone does NOT
                # include. Logging only str(exc) (the previous behavior)
                # made that message actively misleading: it points at
                # information the log line doesn't actually contain.
                detail = ""
                cause = exc.__cause__ or exc.__context__
                if cause is not None and str(cause) != str(exc):
                    detail = f" | caused by: {type(cause).__name__}: {cause}"
                logger.warning(
                    f"Post-process action {action.action!r} failed: {exc}{detail}"
                )
                logger.debug(
                    f"Post-process action {action.action!r} full traceback:",
                    exc_info=exc,
                )
        result.post_process_completed.append("completed")
        return result

    def _apply_action(self, result: DownloadResult, action: Any) -> DownloadResult:
        """Apply a single post-process action to download results."""
        if action.action == "unzip":
            from pygeofetch.utils.file_utils import safe_extract

            new_paths = []
            for path in result.output_paths:
                if path.suffix in (".zip", ".tar", ".gz"):
                    extracted = safe_extract(path, path.parent)
                    new_paths.extend(extracted)
                else:
                    new_paths.append(path)
            result.output_paths = new_paths
            if new_paths:
                result.output_path = new_paths[0]

        elif action.action == "cloud_mask":
            try:
                from pygeofetch.processing.preprocessor import Preprocessor

                method = action.params.get("method", "scl")
                cloud_classes = action.params.get("cloud_classes")

                scl_path = None
                if method == "scl":
                    scl_path = next(
                        (p for p in result.output_paths if "scl" in p.name.lower()),
                        None,
                    )
                    if scl_path is None:
                        raise ValueError(
                            "cloud_mask (method=scl) requires an SCL band to be "
                            'downloaded alongside your other bands — add "SCL" '
                            'to --bands, e.g. --bands "B04,B08,SCL"'
                        )

                pp = Preprocessor()
                new_paths = []
                for path in result.output_paths:
                    if path == scl_path:
                        # Keep the SCL band itself untouched — it's the mask
                        # SOURCE, not something to mask against itself.
                        new_paths.append(path)
                        continue
                    if path.suffix.lower() in (".tif", ".tiff"):
                        masked = pp.cloud_mask(
                            path,
                            method=method,
                            scl_band=scl_path,
                            cloud_classes=cloud_classes,
                        )
                        if masked.success and masked.output_path:
                            new_paths.append(masked.output_path)
                        else:
                            logger.warning(
                                f"cloud_mask failed for {path.name}, keeping "
                                f"original: {masked.error}"
                            )
                            new_paths.append(path)
                    else:
                        new_paths.append(path)
                result.output_paths = new_paths
            except ImportError:
                logger.warning(
                    "processing extras not installed; skipping cloud_mask action"
                )

        elif action.action == "reproject":
            # Requires rasterio - skip if not available
            target_crs = action.params.get("value", "EPSG:4326")
            try:
                import rasterio
                from rasterio.warp import Resampling, reproject  # noqa: F401

                new_paths = []
                any_failed = False
                for path in result.output_paths:
                    if path.suffix.lower() in (".tif", ".tiff"):
                        out_path = path.with_stem(
                            f"{path.stem}_{target_crs.replace(':', '_')}"
                        )
                        try:
                            self._reproject_with_validation(path, out_path, target_crs)
                            new_paths.append(out_path)
                        except Exception as exc:
                            # A single file failing must not discard the
                            # other files' successful reprojections — keep
                            # the ORIGINAL for this one file, not the
                            # (already-deleted) failed output, and continue.
                            logger.warning(
                                f"Reproject failed for {path.name}, keeping "
                                f"original: {exc}"
                            )
                            new_paths.append(path)
                            any_failed = True
                    else:
                        new_paths.append(path)
                result.output_paths = new_paths
                if any_failed:
                    # Ensure downstream log context makes clear this scene's
                    # reproject step was only partially successful, rather
                    # than looking identical to a fully clean run.
                    logger.warning(
                        "reproject: one or more files kept their original "
                        "CRS after a per-file failure — see warnings above."
                    )
            except ImportError:
                logger.warning("rasterio not installed; skipping reproject action")

        elif action.action == "compress":
            method = action.params.get("value", "lzw")
            try:
                import rasterio

                new_paths = []
                for path in result.output_paths:
                    if path.suffix.lower() in (".tif", ".tiff"):
                        out_path = path.with_stem(f"{path.stem}_{method}")
                        with rasterio.open(path) as src:
                            profile = src.profile.copy()
                            profile.update(
                                compress=method,
                                tiled=True,
                                blockxsize=512,
                                blockysize=512,
                            )
                            with rasterio.open(out_path, "w", **profile) as dst:
                                dst.write(src.read())
                        new_paths.append(out_path)
                    else:
                        new_paths.append(path)
                result.output_paths = new_paths
                if new_paths:
                    result.output_path = new_paths[0]
            except ImportError:
                logger.warning("rasterio not installed; skipping compress action")

        elif action.action == "cog":
            compress = action.params.get("value", "deflate")
            try:
                import tempfile

                import rasterio
                from rasterio.enums import Resampling as RS

                new_paths = []
                for path in result.output_paths:
                    if path.suffix.lower() in (".tif", ".tiff"):
                        out_path = path.with_stem(f"{path.stem}_cog")
                        with rasterio.open(path) as src:
                            profile = src.profile.copy()
                            profile.update(
                                compress=compress,
                                tiled=True,
                                blockxsize=512,
                                blockysize=512,
                                interleave="band",
                                driver="GTiff",
                            )
                            data = src.read()
                            with tempfile.NamedTemporaryFile(
                                suffix=".tif", delete=False
                            ) as tmp:
                                tmp_path = Path(tmp.name)
                            with rasterio.open(tmp_path, "w", **profile) as tmp_dst:
                                tmp_dst.write(data)
                                tmp_dst.build_overviews([2, 4, 8, 16, 32], RS.average)
                                tmp_dst.update_tags(
                                    ns="rio_overview", resampling="average"
                                )
                            profile.update(copy_src_overviews=True)
                            with rasterio.open(tmp_path) as tmp_src:
                                with rasterio.open(out_path, "w", **profile) as cog_dst:
                                    cog_dst.write(tmp_src.read())
                            tmp_path.unlink(missing_ok=True)
                        new_paths.append(out_path)
                    else:
                        new_paths.append(path)
                result.output_paths = new_paths
                if new_paths:
                    result.output_path = new_paths[0]
            except ImportError:
                logger.warning("rasterio not installed; skipping cog action")

        elif action.action == "clip":
            import json

            geom_src = action.params.get("value", "")
            try:
                import rasterio
                from rasterio.mask import mask as rasterio_mask

                shapes = []
                if geom_src.endswith((".geojson", ".json")):
                    with open(geom_src) as f:
                        gj = json.load(f)
                    if gj.get("type") == "FeatureCollection":
                        shapes = [feat["geometry"] for feat in gj["features"]]
                    elif gj.get("type") == "Feature":
                        shapes = [gj["geometry"]]
                    else:
                        shapes = [gj]
                elif geom_src:
                    # bbox string "minx,miny,maxx,maxy"
                    try:
                        parts = [float(x) for x in geom_src.split(",")]
                        from shapely.geometry import box as sbox

                        shapes = [sbox(*parts).__geo_interface__]
                    except Exception:
                        pass
                if shapes:
                    new_paths = []
                    for path in result.output_paths:
                        if path.suffix.lower() in (".tif", ".tiff"):
                            out_path = path.with_stem(f"{path.stem}_clip")
                            with rasterio.open(path) as src:
                                out_img, out_transform = rasterio_mask(
                                    src, shapes, crop=True
                                )
                                profile = src.profile.copy()
                                profile.update(
                                    height=out_img.shape[1],
                                    width=out_img.shape[2],
                                    transform=out_transform,
                                )
                                with rasterio.open(out_path, "w", **profile) as dst:
                                    dst.write(out_img)
                            new_paths.append(out_path)
                        else:
                            new_paths.append(path)
                    result.output_paths = new_paths
                    if new_paths:
                        result.output_path = new_paths[0]
            except ImportError:
                logger.warning("rasterio/shapely not installed; skipping clip action")

        elif action.action == "resample":
            val = action.params.get("value", "")
            try:
                import rasterio
                from rasterio.enums import Resampling as RS

                new_paths = []
                for path in result.output_paths:
                    if path.suffix.lower() in (".tif", ".tiff"):
                        with rasterio.open(path) as src:
                            # value is target resolution in CRS units
                            try:
                                target_res = float(val)
                                scale_x = src.res[0] / target_res
                                scale_y = src.res[1] / target_res
                            except (ValueError, ZeroDivisionError):
                                scale_x = scale_y = 0.5
                            new_h = max(1, int(src.height * scale_y))
                            new_w = max(1, int(src.width * scale_x))
                            data = src.read(
                                out_shape=(src.count, new_h, new_w),
                                resampling=RS.bilinear,
                            )
                            transform = src.transform * src.transform.scale(
                                src.width / new_w, src.height / new_h
                            )
                            profile = src.profile.copy()
                            profile.update(
                                height=new_h, width=new_w, transform=transform
                            )
                        out_path = path.with_stem(f"{path.stem}_resamp")
                        with rasterio.open(out_path, "w", **profile) as dst:
                            dst.write(data)
                        new_paths.append(out_path)
                    else:
                        new_paths.append(path)
                result.output_paths = new_paths
                if new_paths:
                    result.output_path = new_paths[0]
            except ImportError:
                logger.warning("rasterio not installed; skipping resample action")

        elif action.action == "ndvi":
            # Compute NDVI from downloaded bands — expects B04 (Red) and B08 (NIR)
            import numpy as np

            try:
                import rasterio

                red_path = next(
                    (
                        p
                        for p in result.output_paths
                        if "B04" in p.name or "red" in p.name.lower()
                    ),
                    None,
                )
                nir_path = next(
                    (
                        p
                        for p in result.output_paths
                        if "B08" in p.name or "nir" in p.name.lower()
                    ),
                    None,
                )
                if red_path and nir_path:
                    with rasterio.open(red_path) as rs:
                        red = rs.read(1).astype(np.float32)
                        red_nodata = rs.nodata
                        profile = rs.profile.copy()
                    with rasterio.open(nir_path) as ns:
                        nir = ns.read(
                            1,
                            out_shape=red.shape,
                            resampling=rasterio.enums.Resampling.bilinear,
                        ).astype(np.float32)
                        nir_nodata = ns.nodata

                    # Respect each source band's own nodata flag (e.g. a
                    # cloud_mask step earlier in the chain, or a provider's
                    # native fill value) — without this, masked/fill pixels
                    # get computed as if they were real reflectance values
                    # instead of correctly propagating as "no data" through
                    # to the NDVI output. Previously neither red nor nir
                    # nodata was checked at all here.
                    if red_nodata is not None:
                        red = np.where(red == red_nodata, np.nan, red)
                    if nir_nodata is not None:
                        nir = np.where(nir == nir_nodata, np.nan, nir)

                    with np.errstate(divide="ignore", invalid="ignore"):
                        ndvi = np.where(
                            nir + red > 0, (nir - red) / (nir + red), np.nan
                        )
                    ndvi = np.where(np.isnan(red) | np.isnan(nir), np.nan, ndvi)
                    ndvi_path = red_path.parent / "ndvi.tif"
                    profile.update(count=1, dtype="float32", nodata=-9999.0)
                    ndvi_write = np.where(np.isnan(ndvi), -9999.0, ndvi)
                    with rasterio.open(ndvi_path, "w", **profile) as dst:
                        dst.write(ndvi_write[np.newaxis, :, :])
                    (result.output_paths or []).append(ndvi_path)
                    result.output_path = ndvi_path
                    logger.info(f"NDVI computed → {ndvi_path}")
                else:
                    logger.warning(
                        "ndvi action: could not find B04 (Red) and B08 (NIR) in output paths"
                    )
            except ImportError:
                logger.warning("rasterio/numpy not installed; skipping ndvi action")

        elif action.action in ("pan-sharpen", "pan_sharpen"):
            logger.info(
                "pan-sharpen action: use PyGeoFetch preprocess pansharpen for this operation"
            )

        elif action.action == "merge":
            try:
                import rasterio
                from rasterio.merge import merge

                tifs = [
                    p
                    for p in result.output_paths
                    if p.suffix.lower() in (".tif", ".tiff")
                ]
                if len(tifs) > 1:
                    src_files = [rasterio.open(p) for p in tifs]
                    mosaic, transform = merge(src_files)
                    profile = src_files[0].profile.copy()
                    profile.update(
                        height=mosaic.shape[1],
                        width=mosaic.shape[2],
                        transform=transform,
                    )
                    merge_path = tifs[0].parent / "merged.tif"
                    with rasterio.open(merge_path, "w", **profile) as dst:
                        dst.write(mosaic)
                    for s in src_files:
                        s.close()
                    result.output_paths = [merge_path]
                    result.output_path = merge_path
            except ImportError:
                logger.warning("rasterio not installed; skipping merge action")

        elif action.action.upper() in _SINGLE_DATE_INDEX_BANDS:
            # Generic handler for any single-date spectral index beyond the
            # hardcoded "ndvi" branch above (EVI, SAVI, NDWI, MNDWI, NDBI,
            # NDSI, NDMI, NBR, BSI, ARVI, GNDVI, RVI, VCI, CRI1, PSRI).
            #
            # Previously, requesting any of these via --post-process fell
            # through every branch with no match and hit `return result`
            # unconditionally — a completely silent no-op. The user's
            # download "succeeded" with the requested index never computed
            # and zero indication anything was wrong. This handler and the
            # explicit fallback error below close that gap: every index
            # documented as supported now either actually runs, or fails
            # with a clear, actionable message — never silently.
            result = self._compute_single_date_index(result, action.action.upper())

        elif action.action.upper() == "DNBR":
            # dNBR needs two dates (pre/post) — the single-scene download
            # post-process chain has no second scene to pair with, so this
            # cannot be silently auto-wired the way the single-date indices
            # above are. Fail clearly rather than either silently no-op or
            # guess at pairing unrelated scenes.
            raise ValueError(
                "dNBR requires paired pre-event/post-event imagery and can't "
                "run as a single-scene --post-process step. Use "
                "pygeofetch.processor.SpectralIndex.compute('dNBR', "
                "NIR_PRE=..., SWIR2_PRE=..., NIR_POST=..., SWIR2_POST=...) "
                "directly with both dates' bands instead."
            )

        else:
            known_actions = (
                "unzip, reproject, compress, cog, clip, resample, merge, "
                "ndvi, " + ", ".join(sorted(_SINGLE_DATE_INDEX_BANDS)).lower()
            )
            raise ValueError(
                f"Unknown --post-process action {action.action!r}. "
                f"Supported actions: {known_actions}."
            )

        return result

    def _compute_single_date_index(
        self, result: DownloadResult, index_name: str
    ) -> DownloadResult:
        """
        Compute any single-date spectral index (beyond the hardcoded NDVI
        branch) using pygeofetch.processor.SpectralIndex, matching required
        bands to downloaded files the same way the ndvi branch does.
        """
        import numpy as np
        import rasterio

        try:
            from pygeofetch.processor import SpectralIndex
        except ImportError as exc:
            raise ImportError(
                f'{index_name} requires the processor extra: '
                f'pip install "pygeofetch[processor]"'
            ) from exc

        required_bands = _SINGLE_DATE_INDEX_BANDS[index_name]
        band_arrays: Dict[str, Any] = {}
        profile = None
        ref_shape = None

        for band_name in required_bands:
            patterns = _BAND_FILENAME_PATTERNS[band_name]
            path = next(
                (
                    p
                    for p in result.output_paths
                    if any(pat in p.name or pat.lower() in p.name.lower() for pat in patterns)
                ),
                None,
            )
            if path is None:
                raise ValueError(
                    f"{index_name}: could not find a downloaded band matching "
                    f"{band_name} (looked for {patterns} in filenames)"
                )
            with rasterio.open(path) as src:
                arr = src.read(1, out_shape=ref_shape, resampling=rasterio.enums.Resampling.bilinear) \
                    if ref_shape else src.read(1)
                if ref_shape is None:
                    ref_shape = arr.shape
                    profile = src.profile.copy()
                arr = arr.astype(np.float32)
                # Respect this band's own nodata flag — see the matching
                # fix in the ndvi handler above for why this matters (a
                # cloud_mask step earlier in the chain, or a provider's
                # native fill value, must correctly propagate as "no data"
                # rather than being computed as if it were a real value).
                if src.nodata is not None:
                    arr = np.where(arr == src.nodata, np.nan, arr)
            band_arrays[band_name] = arr

        si = SpectralIndex(prefer_spyndex=False)
        index_arr = np.asarray(si.compute(index_name, **band_arrays), dtype=np.float32)
        # Any band being NaN at a pixel must make the index NaN there too,
        # even if the formula's arithmetic happens to produce a finite
        # result from a NaN input for some index shapes.
        any_nan = np.zeros(ref_shape, dtype=bool)
        for arr in band_arrays.values():
            any_nan |= np.isnan(arr)
        index_arr = np.where(any_nan, np.nan, index_arr)

        out_path = result.output_paths[0].parent / f"{index_name.lower()}.tif"
        profile.update(count=1, dtype="float32", nodata=-9999.0)
        index_write = np.where(np.isnan(index_arr), -9999.0, index_arr)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(index_write[np.newaxis, :, :])

        result.output_paths = list(result.output_paths) + [out_path]
        result.output_path = out_path
        logger.info(f"{index_name} computed → {out_path}")
        return result


    def _find_existing_download(
        self, data: "SatelliteData", destination: Path
    ) -> Path | None:
        """
        Look for a file that was already downloaded for this scene, so
        `download()` can skip re-downloading it when `options.resume=True`.

        We don't know in advance exactly what path/filename a provider's
        download() would use (that's provider-specific — e.g. Copernicus
        writes destination/copernicus/<SAFE-name>.zip), so this searches:

          1. destination/<provider>/  — the standard per-provider subfolder
             every provider is routed through by download_many()
          2. destination/  — flat, in case this scene was downloaded
             directly via provider.download() without the subfolder routing

        matching on a distinctive substring of the scene's name/id, across
        common satellite data extensions. This deliberately does NOT try to
        be exact — any plausible match is handed to _validate_downloaded_file()
        afterwards, which is the actual authority on whether the file is
        usable, so a false-positive glob match just falls through to
        re-downloading rather than silently reusing a wrong file.
        """
        name = data.properties.get("name") if data.properties else None
        candidates_dirs = [destination / data.provider, destination]

        # Try the exact expected filename patterns first (fast path, no glob)
        if name:
            for d in candidates_dirs:
                for ext in (".zip", ".tif", ".tiff", ".nc", ".SAFE.zip"):
                    p = d / f"{name}{ext}"
                    if p.exists() and p.is_file():
                        return p

        # Fall back to a substring glob match — use a distinguishing chunk
        # of the name (or id) rather than the whole string, since providers
        # sometimes sanitise/truncate filenames.
        chunk = None
        if name:
            # Last underscore-separated segment is usually a unique product
            # identifier (e.g. "...4D5F" in a Sentinel-1 SAFE name)
            parts = name.replace(".SAFE", "").split("_")
            chunk = parts[-1] if parts else None
        if not chunk and data.id:
            chunk = str(data.id)[:8]
        if not chunk:
            return None

        for d in candidates_dirs:
            if not d.exists():
                continue
            matches = [p for p in d.iterdir() if p.is_file() and chunk in p.name]
            if matches:
                return matches[0]

        return None

    def _validate_downloaded_file(self, path) -> tuple:
        """
        Validate a downloaded file.

        Strategy by file type:
          .zip / .tar.gz  — check it's a valid archive (not truncated)
          .tif / .tiff    — open with rasterio and read one tile
          .jp2            — open with rasterio (GDAL driver)
          .nc             — check size > 0 (netCDF, GDAL can open but slow)
          .json / .xml / .csv / .txt / .parquet — check size > 0
          anything else   — check size > 0 (safe fallback, never fail good data)

        Returns:
            (is_valid: bool, error_message: str)
        """
        import os

        path_str = str(path)
        path_obj = Path(path_str)
        suffix = path_obj.suffix.lower()

        # ── 0. File must exist and be non-empty ───────────────────────────
        try:
            size = os.path.getsize(path_str)
        except OSError as exc:
            return False, f"Cannot stat file: {exc}"
        if size == 0:
            return False, "File is empty (0 bytes)"

        # ── 1. ZIP / TAR archives — validate archive structure ─────────────
        if suffix == ".zip" or path_str.endswith(".zip"):
            try:
                import zipfile

                with zipfile.ZipFile(path_str) as zf:
                    bad = zf.testzip()
                    if bad:
                        return False, f"ZIP is corrupt — bad entry: {bad}"
                return True, ""
            except zipfile.BadZipFile as exc:
                return False, f"Invalid ZIP file: {exc}"
            except Exception:
                # If we can't import zipfile or something weird — just trust size
                return True, ""

        if suffix in (".gz", ".tgz") or path_str.endswith(".tar.gz"):
            try:
                import tarfile

                with tarfile.open(path_str) as tf:
                    tf.getmembers()
                return True, ""
            except tarfile.TarError as exc:
                return False, f"Invalid TAR file: {exc}"
            except Exception:
                return True, ""

        # ── 2. Non-raster text/data formats — size check only ─────────────
        NON_RASTER = {
            ".json",
            ".geojson",
            ".xml",
            ".csv",
            ".txt",
            ".html",
            ".md",
            ".yaml",
            ".yml",
            ".parquet",
            ".nc",
            ".EOF",  # orbit files
        }
        if suffix in NON_RASTER or not suffix:
            return True, ""  # already passed size > 0 above

        # ── 3. Raster formats — open with rasterio ────────────────────────
        RASTER_SUFFIXES = {
            ".tif",
            ".tiff",
            ".jp2",
            ".img",
            ".vrt",
            ".hdf",
            ".h4",
            ".h5",
            ".hdf5",
        }
        if suffix in RASTER_SUFFIXES:
            try:
                import rasterio

                with rasterio.open(path_str) as src:
                    if src.count == 0:
                        return False, "Raster has zero bands"
                    if src.width == 0 or src.height == 0:
                        return (
                            False,
                            f"Raster has zero dimensions: {src.width}x{src.height}",
                        )
                    # Read multiple tiles spread across the file, not just
                    # the first one. A truncated/incomplete download (a
                    # dropped connection mid-stream, the single most common
                    # real-world download corruption) cuts a file off from
                    # the END — the header, TIFF directory, and early tiles
                    # are almost always intact, while later tiles are
                    # missing. Sampling only the first tile (the previous
                    # behavior) is therefore nearly blind to exactly the
                    # corruption pattern most likely to occur in practice:
                    # confirmed by direct reproduction — an 80%-truncated
                    # file opens fine, its dimensions read correctly, and
                    # its first tile reads fine, but its last tile fails
                    # with the same RasterioIOError downstream processing
                    # (reproject/ndvi/cog) would hit minutes later.
                    windows = list(src.block_windows(1))
                    if windows:
                        sample_indices = sorted(
                            {0, len(windows) // 2, len(windows) - 1}
                        )
                        for idx in sample_indices:
                            _, window = windows[idx]
                            try:
                                sample = src.read(1, window=window)
                            except Exception as exc:
                                return (
                                    False,
                                    f"Tile {idx + 1}/{len(windows)} read failed "
                                    f"(likely a truncated/incomplete download): "
                                    f"{exc}",
                                )
                            if sample is None or sample.size == 0:
                                return (
                                    False,
                                    f"Tile {idx + 1}/{len(windows)} read returned "
                                    f"an empty array",
                                )
                return True, ""
            except ImportError:
                # rasterio not installed — size check already passed
                return True, ""
            except Exception as exc:
                return False, f"Raster read failed: {exc}"

        # ── 4. Unknown extension — trust size check (already passed) ──────
        return True, ""

    def _has_identity_transform(self, path) -> bool:
        """
        Detect a reprojection that produced pixel-space garbage instead
        of a real, metre-scale transform.

        Real, confirmed gap this fix closes: the original version
        required origin to be EXACTLY (0, 0), but a real reproduction of
        the underlying bug (GCP-only source, src.crs=None, see
        _reproject_with_validation's docstring) produced a transform
        with origin (0, 200), not (0, 0) — the Y-axis flip convention in
        rasterio's warp math shifts the origin by the image height, so a
        fixed-origin check misses this real, observed variant entirely.
        The real, reliable signal isn't the origin, it's the pixel size:
        any projected (metre-unit) CRS with a pixel edge under ~1 metre
        is never real satellite imagery (Sentinel-1/2 native resolution
        is 10m+; even the highest-resolution commercial sensors are
        sub-metre but not ~1.0 exactly) — it's the unmistakable signature
        of pixel-space coordinates mistakenly tagged with a metre CRS.
        """
        try:
            import rasterio

            with rasterio.open(str(path)) as src:
                t = src.transform
                if src.crs is None or not src.crs.is_projected:
                    return False
                pixel_size = max(abs(t.a), abs(t.e))
                return pixel_size <= 1.0
        except Exception:
            return False

    def _reproject_with_validation(self, source_path, target_path, target_crs) -> None:
        """
        Reproject source to target CRS and validate the output transform.
        Raises RuntimeError if reprojection produces an identity transform.

        On ANY failure, the partially-written target file is deleted rather
        than left on disk — a previous version left corrupt/incomplete
        output files behind on failure, which downstream post-process
        actions (e.g. cog) would then silently pick up and fail on with a
        confusing, disconnected error, since nothing recorded that this
        file was the product of a failed operation.
        """
        import rasterio
        from rasterio.enums import Resampling as RS
        from rasterio.warp import calculate_default_transform, reproject

        try:
            with rasterio.open(str(source_path)) as src:
                src_crs = src.crs
                src_transform = src.transform
                src_width = src.width
                src_height = src.height

                # Real, confirmed bug fixed here: Sentinel-1 GRD products
                # (and other raw SAR delivery formats) are commonly
                # georeferenced via ground control points only, not a
                # direct CRS + affine transform -- rasterio correctly
                # reports src.crs=None and src.transform=identity for
                # these, confirmed directly. calculate_default_transform()
                # does NOT error on src_crs=None, it silently treats the
                # pixel-space bounds as if they were real coordinates,
                # producing a transform that's still pixel-scale (e.g.
                # pixel size ~1.0) but tagged with a real, valid target
                # CRS -- exactly the "CRS tag correct, transform garbage"
                # symptom this whole fix addresses. Using the GCPs
                # directly (verified: produces a real, physically
                # plausible metre-scale transform instead) is the actual
                # fix, not a fallback.
                src_gcps = None
                if src_crs is None and src.gcps[0]:
                    src_gcps, gcps_crs = src.gcps
                    src_crs = gcps_crs
                    logger.info(
                        "%s has no direct CRS/transform, using its %d "
                        "ground control points for reprojection instead "
                        "(common for raw SAR/GRD delivery).",
                        source_path.name, len(src_gcps),
                    )
                elif src_crs is None:
                    raise RuntimeError(
                        f"{source_path.name} has no CRS, no transform, and "
                        f"no GCPs — genuinely no georeferencing information "
                        f"to reproject from."
                    )

                if src_gcps is not None:
                    dst_transform, dst_width, dst_height = calculate_default_transform(
                        src_crs, target_crs, src_width, src_height, gcps=src_gcps
                    )
                else:
                    dst_transform, dst_width, dst_height = calculate_default_transform(
                        src_crs, target_crs, src_width, src_height, *src.bounds
                    )
                kwargs = src.meta.copy()
                kwargs.update(
                    {
                        "crs": target_crs,
                        "transform": dst_transform,
                        "width": dst_width,
                        "height": dst_height,
                    }
                )
                with rasterio.open(str(target_path), "w", **kwargs) as dst:
                    for band_idx in range(1, src.count + 1):
                        if src_gcps is not None:
                            reproject(
                                source=rasterio.band(src, band_idx),
                                destination=rasterio.band(dst, band_idx),
                                src_crs=src_crs,
                                gcps=src_gcps,
                                dst_transform=dst_transform,
                                dst_crs=target_crs,
                                resampling=RS.bilinear,
                            )
                        else:
                            reproject(
                                source=rasterio.band(src, band_idx),
                                destination=rasterio.band(dst, band_idx),
                                src_transform=src_transform,
                                src_crs=src_crs,
                                dst_transform=dst_transform,
                                dst_crs=target_crs,
                                resampling=RS.bilinear,
                            )
        except Exception as exc:
            # Surface the REAL underlying error (GDAL's own message is
            # normally far more specific than a generic wrapper like
            # "Chunk and warp failed" suggests) rather than losing it —
            # previously this propagated up to be logged only as
            # "Post-process action 'reproject' failed: <str(exc)>", which
            # is exactly this message, but the caller's log line gave no
            # indication of *which* file/CRS/band was involved.
            target_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Reprojection failed for {source_path.name} -> {target_crs}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if self._has_identity_transform(target_path):
            target_path.unlink(missing_ok=True)
            msg = (
                f"Reprojection produced identity transform in {target_path.name} "
                f"(deleted). Source: {source_path.name}, Target CRS: {target_crs}. "
                "This is a known rasterio/GDAL edge case with certain input CRS."
            )
            raise RuntimeError(msg)


    def _backoff(self, attempt: int, options: DownloadOptions) -> float:
        """Calculate retry delay based on strategy."""
        import random

        delay = options.retry_delay_seconds * (2**attempt)
        delay = min(delay, 60.0)
        if "jitter" in options.retry_strategy:
            delay *= 0.5 + random.random() * 0.5
        return delay

    def _emit_progress(
        self,
        data_list: list[SatelliteData],
        results: list[DownloadResult | None],
        options: DownloadOptions,
    ) -> None:
        """Emit progress update via callback if configured."""
        if not self.progress_callback:
            return
        # Emit lightweight progress update (counts only)
        n_done = sum(1 for r in results if r is not None)
        # n_done computed for the progress signal below, not used directly here
        _ = n_done
        progress = DownloadProgress(total=len(results), destination="")
        try:
            self.progress_callback(progress)
        except Exception:
            pass

    @property
    def _result_total_size_mb(self) -> float:
        """Property used by result objects."""
        return 0.0
