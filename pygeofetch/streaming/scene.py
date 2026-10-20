"""
Scene-level AOI streaming: apply windowed reads to a ``SatelliteData`` record.

``stream_scene`` / ``stream_many`` are the AOI-only counterparts of
``client.download``. Output lands under ``<destination>/<provider>/`` exactly
like ``download_many()`` nests its output, so downstream code that already
globs that layout keeps working. ``read_scene_bands`` returns the AOI arrays
directly (no files) for in-memory analysis, bounded by ``max_pixels``.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from pygeofetch.models.download_task import DownloadResult, DownloadStatus
from pygeofetch.models.satellite_data import SatelliteData, resolve_band_keys
from pygeofetch.streaming.windowed import (
    StreamingUnsupportedError,
    WindowedRaster,
    assert_streamable,
    read_window,
    stream_window_to_geotiff,
)

logger = logging.getLogger("pygeofetch.streaming.scene")

_RASTER_EXT = (".tif", ".tiff", ".jp2")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def streamed_band_paths(
    data: SatelliteData, bands: Sequence[str], destination: str | Path
) -> dict[str, Path]:
    """Where ``stream_scene`` writes each *requested* band (alias-aware).

    Returns ``{requested_band: path}`` for bands the scene has; bands it does
    not have are omitted. Lets callers find files by the band name they asked
    for (``B04``) even when the provider calls the asset ``red``.
    """
    pool = {
        k: a
        for k, a in (data.data_assets or data.assets).items()
        if _is_raster_asset(a)
    }
    out: dict[str, Path] = {}
    for band in bands:
        hit = _strict_band_match(band, list(pool.keys()))
        if hit:
            out[band] = _band_path(data, hit[0], destination)
    return out


def _band_path(data: SatelliteData, key: str, destination: str | Path) -> Path:
    return Path(destination) / data.provider / f"{_safe(data.id)}_{_safe(key)}.tif"


def _is_raster_asset(asset: Any) -> bool:
    href = str(getattr(asset, "href", "") or "").split("?", 1)[0].lower()
    mt = (getattr(asset, "media_type", None) or "").lower()
    return href.endswith(_RASTER_EXT) or "tiff" in mt or "jp2" in mt or "jpeg2000" in mt


_NO_MATCH = "__pygeofetch_no_match__"


def _strict_band_match(band: str, keys: list[str]) -> list[str]:
    """Alias-aware match of one band with NO "fall back to everything".

    ``resolve_band_keys`` returns *all* keys when nothing matches. Adding a
    sentinel key makes that fallback detectable: it only appears in the
    result when the real keys did not match.
    """
    res = resolve_band_keys([band], keys + [_NO_MATCH])
    return [] if _NO_MATCH in res else res


def select_streamable_assets(
    data: SatelliteData, bands: Sequence[str] | None
) -> dict[str, Any]:
    """Pick raster assets to stream.

    Unlike the downloaders, a band request that matches nothing is an error
    here instead of silently falling back to "every asset": for streaming
    that fallback would quietly pull far more than asked for.
    """
    pool = data.data_assets or data.assets
    pool = {k: a for k, a in pool.items() if _is_raster_asset(a)}
    if not pool:
        raise StreamingUnsupportedError(
            f"{data.id}: no raster (.tif/.jp2) assets; this record looks archive-only "
            f"(provider={data.provider!r}) and cannot be streamed."
        )
    if bands:
        chosen: dict[str, Any] = {}
        missing: list[str] = []
        for band in bands:
            hit = _strict_band_match(band, list(pool.keys()))
            if not hit:
                missing.append(band)
            for k in hit:
                chosen[k] = pool[k]
        if missing:
            raise StreamingUnsupportedError(
                f"{data.id}: requested bands {missing} matched none of {sorted(pool)}."
            )
        return chosen
    return pool


def _coverage_of_existing(path: Path, bbox: Any, bbox_crs: str) -> float:
    """AOI coverage of an already-streamed file (used on resume)."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds

        from pygeofetch.streaming.windowed import _bbox_bounds

        with rasterio.open(path) as src:
            minx, miny, maxx, maxy = _bbox_bounds(bbox)
            if str(bbox_crs).upper() != str(src.crs).upper():
                minx, miny, maxx, maxy = transform_bounds(
                    bbox_crs, src.crs, minx, miny, maxx, maxy, densify_pts=21
                )
            b = src.bounds
            ix = max(0.0, min(maxx, b.right) - max(minx, b.left))
            iy = max(0.0, min(maxy, b.top) - max(miny, b.bottom))
            want = (maxx - minx) * (maxy - miny)
            # one-pixel tolerance: the streamed window is rounded outwards
            return min(1.0, (ix * iy) / want * 1.0) if want > 0 else 1.0
    except Exception:
        return 1.0


def _get_provider(data: SatelliteData, provider: Any | None) -> Any:
    if provider is not None:
        return provider
    from pygeofetch.providers import get_provider

    return get_provider(data.provider)


def _signed_href(provider: Any, data: SatelliteData, href: str) -> str:
    signer = getattr(provider, "sign_href", None)
    return signer(href, data) if callable(signer) else href


def read_scene_bands(
    data: SatelliteData,
    bbox: Any,
    bands: Sequence[str] | None = None,
    *,
    provider: Any | None = None,
    bbox_crs: str = "EPSG:4326",
    decimation: int = 1,
    max_pixels: int | None = 50_000_000,
    gdal_options: dict[str, str] | None = None,
) -> dict[str, WindowedRaster]:
    """Read AOI windows of the chosen bands into memory (no files written).

    Bands may come back with different shapes (e.g. Sentinel-2 10 m vs 20 m);
    each ``WindowedRaster`` carries its own transform.
    """
    prov = _get_provider(data, provider)
    out: dict[str, WindowedRaster] = {}
    for key, asset in select_streamable_assets(data, bands).items():
        href = _signed_href(prov, data, asset.href)
        out[key] = read_window(
            href,
            bbox,
            bbox_crs=bbox_crs,
            decimation=decimation,
            max_pixels=max_pixels,
            gdal_options=gdal_options,
        )
    return out


def stream_scene(
    data: SatelliteData,
    bbox: Any,
    destination: str | Path,
    bands: Sequence[str] | None = None,
    *,
    provider: Any | None = None,
    bbox_crs: str = "EPSG:4326",
    block_size: int = 1024,
    overwrite: bool = False,
    gdal_options: dict[str, str] | None = None,
) -> DownloadResult:
    """Stream the AOI window of each chosen band to small local GeoTIFFs.

    ``DownloadResult.bytes_downloaded`` is the size of the AOI GeoTIFFs
    written to disk, not network traffic.
    """
    t0 = time.time()
    try:
        assets = select_streamable_assets(data, bands)
        prov = _get_provider(data, provider)
    except (StreamingUnsupportedError, ValueError) as exc:
        return DownloadResult(
            status=DownloadStatus.FAILED,
            data_id=data.id,
            provider=data.provider,
            error=str(exc),
            error_type="streaming_unsupported",
        )

    (Path(destination) / data.provider).mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    coverages: list[float] = []
    total = 0
    errors: list[str] = []
    for key, asset in assets.items():
        out_file = _band_path(data, key, destination)
        if out_file.exists() and not overwrite:
            paths.append(out_file)
            total += out_file.stat().st_size
            coverages.append(_coverage_of_existing(out_file, bbox, bbox_crs))
            continue
        try:
            assert_streamable(asset.href)
            href = _signed_href(prov, data, asset.href)
            info = stream_window_to_geotiff(
                href,
                bbox,
                out_file,
                bbox_crs=bbox_crs,
                block_size=block_size,
                gdal_options=gdal_options,
            )
            paths.append(info["path"])
            total += info["bytes_written"]
            coverages.append(info["aoi_coverage"])
        except Exception as exc:  # one bad band must not abort the scene
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
            logger.warning("stream %s/%s failed: %s", data.id, key, exc)

    if not paths:
        return DownloadResult(
            status=DownloadStatus.FAILED,
            data_id=data.id,
            provider=data.provider,
            error="; ".join(errors) or "no assets streamed",
            error_type="streaming",
            duration_seconds=time.time() - t0,
        )
    res = DownloadResult(
        status=DownloadStatus.COMPLETED,
        data_id=data.id,
        provider=data.provider,
        output_path=paths[0],
        output_paths=paths,
        bytes_downloaded=total,
        duration_seconds=time.time() - t0,
        aoi_coverage=min(coverages) if coverages else None,
    )
    if res.aoi_coverage is not None and res.aoi_coverage < 0.99:
        logger.warning(
            "%s covers only %.0f%% of the AOI (it runs past the scene edge); "
            "stream an adjacent scene too or pick another.",
            data.id,
            100 * res.aoi_coverage,
        )
    if errors:
        res.error = "partial: " + "; ".join(errors)
    return res


def stream_many(
    items: Sequence[SatelliteData],
    bbox: Any,
    destination: str | Path,
    bands: Sequence[str] | None = None,
    *,
    parallel: int = 4,
    **kwargs: Any,
) -> list[DownloadResult]:
    """Stream several scenes in parallel; result order matches ``items``."""
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, int(parallel))) as pool:
        futs = [
            pool.submit(stream_scene, d, bbox, destination, bands, **kwargs)
            for d in items
        ]
        return [f.result() for f in futs]
