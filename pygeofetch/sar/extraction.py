"""
GRDExtractor — extract and correctly georeference a Sentinel-1 GRD
measurement band from a downloaded .SAFE.zip.

GRD products carry a single continuous measurement band per polarisation
(unlike SLC's 3 sub-swaths), so this is a simpler extraction than
SLCExtractor — but it needs the same real-world handling SLCExtractor
already applies: the raw measurement TIFF inside the .SAFE archive
commonly has NO standard CRS/transform at all (src.crs is None, and its
"bounds" are just raw pixel indices). Real georeferencing is instead
delivered as embedded Ground Control Points (GCPs). Skipping this step
doesn't raise an error — downstream operations like clip() can appear to
succeed while actually operating on a meaningless few pixels near the
raster's origin, unrelated to your real AOI. This was found and fixed
directly against a real Sentinel-1 GRD product during development, not
assumed.

Usage::

    from pygeofetch.sar import GRDExtractor

    extractor = GRDExtractor(polarisation="VV")
    vv_path = extractor.extract_band(download_result, output_dir="./data", label="pre_event")
    # vv_path is a real, properly-georeferenced GeoTIFF, ready for
    # Preprocessor.clip() / SARProcessor.calibrate()

Radiometrically calibrated backscatter (thermal-noise removal + sigma0)::

    cog = extractor.extract_sigma0(
        download_result, output_dir="./data", label="2026-10-17",
        streaming_cleanup=True,      # verify-then-delete the raw .SAFE.zip
    )
    # cog is a Cloud-Optimized GeoTIFF of sigma0 in dB (float32, NaN = no data)
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from pygeofetch.models.download_task import DownloadResult

logger = logging.getLogger("pygeofetch.sar.extraction")


def georeference_via_gcps_if_needed(path: Union[str, Path]) -> Path:
    """
    Georeference a raster via its embedded Ground Control Points, if it
    doesn't already have a real CRS.

    Raw Sentinel-1 measurement TIFFs (both GRD and SLC) commonly carry no
    standard CRS/transform — real georeferencing is delivered as embedded
    GCPs instead. Rasters that already have a real CRS are returned
    completely unchanged; this is safe to call on any raster
    unconditionally, not just ones known to need it.

    Args:
        path: Path to the raster to check/georeference.

    Returns:
        The original path if it already had a real CRS or had no GCPs to
        fall back on (in which case a warning is logged — downstream
        clipping will not work correctly against it); otherwise the path
        to a new, properly georeferenced GeoTIFF.
    """
    import rasterio
    import rasterio.shutil
    from rasterio.vrt import WarpedVRT

    path = Path(path)
    with rasterio.open(path) as src:
        if src.crs is not None:
            return path

        gcp_list, gcp_crs = src.gcps
        if not gcp_list:
            logger.warning(
                "%s has no CRS and no embedded GCPs — cannot georeference. "
                "Downstream clip()/calibrate() operations will not work "
                "correctly against this file.",
                path.name,
            )
            return path

        georef_path = path.with_stem(f"{path.stem}_georef")
        with WarpedVRT(src, src_crs=gcp_crs, crs=gcp_crs) as vrt:
            rasterio.shutil.copy(vrt, str(georef_path), driver="GTiff")

        logger.info(
            "Georeferenced %s via %d embedded GCPs → %s",
            path.name,
            len(gcp_list),
            georef_path.name,
        )
        return georef_path


# ──────────────────────────────────────────────────────────────────────────
# Sentinel-1 GRD radiometric calibration helpers
#
# Calibration equation (ESA S1 product specification, "Radiometric
# Calibration of S-1 Level-1 Products"):
#
#       value(i, j) = ( |DN(i, j)|^2  -  N(i, j) ) / A(i, j)^2
#
# where A is one of the calibration LUTs (sigmaNought, betaNought, gamma or
# dn) and N is the thermal-noise power:
#
#       N(i, j) = noiseRangeLut(i, j) * noiseAzimuthLut(i, j)
#
# (the azimuth factor only exists for IPF >= 2.9; older products carry the
# range LUT alone). Both LUTs are published on a coarse (line, pixel) grid
# and must be interpolated to every pixel — and this must happen in the
# ORIGINAL RADAR GEOMETRY, i.e. BEFORE the image is geocoded, because the
# LUT coordinates are measurement-TIFF line/pixel indices. That is why
# extract_sigma0() calibrates first and georeferences afterwards.
# ──────────────────────────────────────────────────────────────────────────

# Valid LUT element names inside <calibrationVector> and the S-1 convention
# for how each calibrated quantity is called.
_CAL_LUTS = ("sigmaNought", "betaNought", "gamma", "dn")

# Minimum share of finite pixels a product must contain to pass verification.
_MIN_VALID_FRACTION = 0.01


def _split_floats(text: Optional[str], dtype=np.float64) -> np.ndarray:
    """Parse a whitespace-separated annotation list (empty/None -> empty)."""
    if not text or not text.strip():
        return np.empty(0, dtype=dtype)
    return np.array(text.split(), dtype=dtype)


class _VectorLut:
    """
    A LUT published as vectors on sparse image lines, interpolated bilinearly
    to any block of full-resolution rows.

    Each vector gives one image line and a (usually irregular) list of
    (pixel, value) samples along range. Vectors are first resampled onto
    every range pixel (np.interp, edge values held), then blended linearly
    between the two neighbouring lines for each requested row. Rows outside
    the first/last vector line take the nearest vector (no extrapolation).
    """

    def __init__(self, entries: List[Tuple[int, np.ndarray, np.ndarray]],
                 width: int) -> None:
        if not entries:
            raise ValueError("no LUT vectors found")
        entries = sorted(entries, key=lambda e: e[0])
        self.lines = np.array([e[0] for e in entries], dtype=np.float64)
        cols = np.arange(width, dtype=np.float64)
        # (n_vectors, width) float32 — ~17 vectors x 17000 px = ~1 MB
        self.grid = np.stack(
            [np.interp(cols, pix, lut) for _, pix, lut in entries]
        ).astype(np.float32)

    def rows(self, r0: int, r1: int) -> np.ndarray:
        """Return LUT values for image rows r0..r1-1 as (r1-r0, width)."""
        n = len(self.lines)
        if n == 1:
            return np.broadcast_to(self.grid[0], (r1 - r0, self.grid.shape[1]))
        r = np.arange(r0, r1, dtype=np.float64)
        i1 = np.clip(np.searchsorted(self.lines, r, side="right"), 1, n - 1)
        i0 = i1 - 1
        span = self.lines[i1] - self.lines[i0]
        w = np.clip((r - self.lines[i0]) / np.where(span > 0, span, 1.0), 0, 1)
        w = w.astype(np.float32)[:, None]
        return self.grid[i0] * (1.0 - w) + self.grid[i1] * w


def _parse_vector_lut(
    xml_bytes: bytes, vector_tags: Tuple[str, ...], lut_tags: Tuple[str, ...],
    width: int,
) -> _VectorLut:
    """
    Build a _VectorLut from an annotation XML.

    ``vector_tags`` / ``lut_tags`` list the accepted element names so one
    parser serves calibration (``calibrationVector``/``sigmaNought``), new
    noise (``noiseRangeVector``/``noiseRangeLut``) and pre-IPF-2.9 noise
    (``noiseVector``/``noiseLut``). Vectors whose pixel and value lists
    disagree in length are skipped with a warning rather than guessed at.
    """
    root = ET.fromstring(xml_bytes)
    entries: List[Tuple[int, np.ndarray, np.ndarray]] = []
    skipped = 0
    for tag in vector_tags:
        for vec in root.iter(tag):
            line = vec.findtext("line")
            pix = _split_floats(vec.findtext("pixel"))
            lut = np.empty(0)
            for lt in lut_tags:
                lut = _split_floats(vec.findtext(lt))
                if lut.size:
                    break
            if line is None or pix.size == 0 or pix.size != lut.size:
                skipped += 1
                continue
            entries.append((int(float(line)), pix, lut))
        if entries:
            break  # first tag that yields vectors wins
    if skipped:
        logger.warning("Skipped %d malformed %s vector(s)", skipped, vector_tags[0])
    return _VectorLut(entries, width)


class _AzimuthNoise:
    """
    Azimuth noise factor (IPF >= 2.9 ``noiseAzimuthVector`` blocks).

    Each block covers a rectangle of the image (one swath) and carries a 1-D
    LUT along azimuth. Outside every block the factor is 1 (i.e. the range
    noise alone applies).
    """

    def __init__(self, xml_bytes: bytes) -> None:
        self.blocks: List[Tuple[int, int, int, int, np.ndarray, np.ndarray]] = []
        for v in ET.fromstring(xml_bytes).iter("noiseAzimuthVector"):
            try:
                first_az = int(v.findtext("firstAzimuthLine"))
                last_az = int(v.findtext("lastAzimuthLine"))
                first_rs = int(v.findtext("firstRangeSample"))
                last_rs = int(v.findtext("lastRangeSample"))
            except (TypeError, ValueError):
                continue
            lines = _split_floats(v.findtext("line"))
            lut = _split_floats(v.findtext("noiseAzimuthLut"))
            if lut.size == 0 or (lines.size and lines.size != lut.size):
                continue
            self.blocks.append((first_az, last_az, first_rs, last_rs, lines, lut))

    def rows(self, r0: int, r1: int, width: int) -> Optional[np.ndarray]:
        """Factor for rows r0..r1-1, or None when there are no blocks."""
        if not self.blocks:
            return None
        out = np.ones((r1 - r0, width), dtype=np.float32)
        for first_az, last_az, first_rs, last_rs, lines, lut in self.blocks:
            lo, hi = max(r0, first_az), min(r1 - 1, last_az)
            if lo > hi:
                continue
            ra = np.arange(lo, hi + 1, dtype=np.float64)
            f = np.interp(ra, lines, lut) if lines.size else np.full(ra.size, lut[0])
            c0, c1 = max(first_rs, 0), min(last_rs, width - 1)
            if c0 <= c1:
                out[lo - r0: hi - r0 + 1, c0: c1 + 1] = f.astype(np.float32)[:, None]
        return out


class _NoiseModel:
    """Thermal-noise power N(i, j) = range LUT x azimuth LUT, per row block."""

    def __init__(self, range_lut: _VectorLut,
                 azimuth: Optional[_AzimuthNoise]) -> None:
        self.range_lut = range_lut
        self.azimuth = azimuth

    def power(self, r0: int, r1: int, width: int) -> np.ndarray:
        noise = np.array(self.range_lut.rows(r0, r1), dtype=np.float32)
        if self.azimuth is not None:
            factor = self.azimuth.rows(r0, r1, width)
            if factor is not None:
                noise *= factor
        return noise


def _pick_member(names: List[str], folder: str, prefix: str,
                 pol: str) -> Optional[str]:
    """
    Find ``<SAFE>/annotation/<folder>/<prefix>-...-<pol>-...xml`` in a zip.

    The ``rfi-`` files that share the calibration folder are excluded by
    the prefix match.
    """
    hits = [
        n for n in names
        if f"/annotation/{folder}/" in n
        and os.path.basename(n).startswith(f"{prefix}-")
        and f"-{pol}-" in os.path.basename(n)
        and n.endswith(".xml")
    ]
    return sorted(hits)[0] if hits else None


# Pixels added around the AOI when cropping a GRD in radar geometry. The AOI
# corners are located from the product's sparse GCPs, so a generous margin
# absorbs the interpolation error of that lookup.
_AOI_MARGIN_PX = 300
# GCPs kept for georeferencing a cropped product (the nearest ones): a local
# polynomial fit is far more accurate over a small window than one fitted to
# GCPs spread over a whole 250 km scene.
_AOI_KEEP_GCPS = 16


def _aoi_window(gcps, bbox, width: int, height: int):
    """
    Rows/columns of a GRD that cover a lon/lat box, found from its GCPs.

    Returns ``(r0, r1, c0, c1)`` (half-open, clipped to the image, with
    ``_AOI_MARGIN_PX`` of margin) or None when the box lies outside the
    image. The (lon, lat) -> (row, col) lookup interpolates linearly between
    GCPs, falling back to the nearest GCP outside their hull.
    """
    from scipy.interpolate import griddata

    pts = np.array([[g.x, g.y] for g in gcps], dtype=np.float64)
    rc = np.array([[g.row, g.col] for g in gcps], dtype=np.float64)
    min_lon, min_lat, max_lon, max_lat = bbox
    lon, lat = np.meshgrid(np.linspace(min_lon, max_lon, 5),
                           np.linspace(min_lat, max_lat, 5))
    q = np.column_stack([lon.ravel(), lat.ravel()])
    out = griddata(pts, rc, q, method="linear")
    near = griddata(pts, rc, q, method="nearest")
    out = np.where(np.isnan(out), near, out)
    r0 = int(np.floor(out[:, 0].min())) - _AOI_MARGIN_PX
    r1 = int(np.ceil(out[:, 0].max())) + _AOI_MARGIN_PX + 1
    c0 = int(np.floor(out[:, 1].min())) - _AOI_MARGIN_PX
    c1 = int(np.ceil(out[:, 1].max())) + _AOI_MARGIN_PX + 1
    r0, r1, c0, c1 = max(r0, 0), min(r1, height), max(c0, 0), min(c1, width)
    if r1 <= r0 or c1 <= c0:
        return None
    return r0, r1, c0, c1


def _extract_member(zf: zipfile.ZipFile, member: str, dest: Path) -> None:
    """Stream one zip member to disk in chunks (no whole-file RAM copy)."""
    with zf.open(member) as src, open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)


def _verify_product(path: Path, require_crs: bool) -> Tuple[bool, str]:
    """
    Verify-step of verify-then-delete: is ``path`` a usable sigma0 raster?

    Checks (on a decimated read, so it is cheap even for 1.7 GB products):
    one band, non-empty shape, tiled layout (COG), a CRS when the source had
    GCPs, at least ``_MIN_VALID_FRACTION`` finite pixels, and not all zeros.
    Returns ``(ok, reason)``.
    """
    import rasterio

    try:
        with rasterio.open(path) as ds:
            if ds.count != 1 or ds.width < 1 or ds.height < 1:
                return False, (
                    f"unexpected shape count={ds.count} {ds.height}x{ds.width}"
                )
            if require_crs and ds.crs is None:
                return False, "output has no CRS"
            if not ds.profile.get("tiled", False):
                return False, "output is not tiled (not a COG)"
            scale = max(1, max(ds.height, ds.width) // 2048)
            arr = ds.read(
                1, out_shape=(max(1, ds.height // scale), max(1, ds.width // scale))
            )
    except Exception as exc:  # noqa: BLE001 — any read failure = not verified
        return False, f"could not open/read: {exc}"

    finite = np.isfinite(arr)
    if finite.mean() < _MIN_VALID_FRACTION:
        return False, f"only {finite.mean():.2%} finite pixels"
    if not np.any(arr[finite] != 0):
        return False, "all valid pixels are zero"
    return True, "ok"


class GRDExtractor:
    """
    Extract a correctly-georeferenced measurement band from a downloaded
    Sentinel-1 GRD .SAFE.zip.

    Args:
        polarisation: Which polarisation's measurement band to extract
                     ("VV", "VH", "HH", or "HV"). Default "VV".
    """

    def __init__(self, polarisation: str = "VV") -> None:
        self._pol = polarisation.lower()

    def extract_band(
        self,
        source: Union["DownloadResult", str, Path],
        output_dir: Union[str, Path],
        label: str = "",
    ) -> Optional[Path]:
        """
        Extract and georeference the configured polarisation's measurement
        band from a downloaded GRD .SAFE.zip.

        Args:
            source:     DownloadResult from client.download() (preferred —
                       uses its .output_path directly), or a direct path
                       to the downloaded .SAFE.zip.
            output_dir: Where to write the extracted GeoTIFF.
            label:      Optional label used in the output filename (e.g.
                       "pre_event", "post_day3") — useful when extracting
                       several dates into the same output_dir.

        Returns:
            Path to a real, properly-georeferenced GeoTIFF, ready for
            Preprocessor.clip() / SARProcessor.calibrate() — or None if
            extraction failed (see logged errors for why).
        """
        zip_path = self._resolve_path(source)
        if zip_path is None:
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path) as zf:
                pol_members = [
                    n
                    for n in zf.namelist()
                    if "/measurement/" in n
                    and f"-{self._pol}-" in n
                    and n.endswith(".tiff")
                ]
                if not pol_members:
                    logger.error(
                        "No %s measurement band found in %s",
                        self._pol.upper(),
                        zip_path.name,
                    )
                    return None

                member = pol_members[0]
                stem = f"{label}_{self._pol}_raw" if label else f"{self._pol}_raw"
                raw_path = output_dir / f"{stem}.tif"
                _extract_member(zf, member, raw_path)
        except (zipfile.BadZipFile, OSError) as exc:
            logger.error("Could not read %s: %s", zip_path.name, exc)
            return None

        return georeference_via_gcps_if_needed(raw_path)

    # ── Calibrated sigma0 (thermal-noise removal + radiometric calibration) ──

    def extract_sigma0(
        self,
        source: Union["DownloadResult", str, Path],
        output_dir: Union[str, Path],
        label: str = "",
        *,
        lut: str = "sigmaNought",
        remove_noise: bool = True,
        out_db: bool = True,
        floor_db: Optional[float] = None,
        block_rows: int = 512,
        aoi_bbox: Optional[Tuple[float, float, float, float]] = None,
        streaming_cleanup: bool = False,
    ) -> Optional[Path]:
        """
        Produce a calibrated, georeferenced backscatter COG from a GRD zip.

        Pipeline (all in the original radar geometry until the last step)::

            measurement DN  ->  DN^2 - N  ->  / A^2  ->  (10*log10)  ->
            GCP georeferencing  ->  Cloud-Optimized GeoTIFF

        with N the thermal-noise power (range LUT x azimuth LUT) and A the
        chosen calibration LUT, both interpolated to every pixel from the
        sparse annotation vectors. The product is processed in row blocks so
        peak RAM stays at a few hundred MB regardless of scene size.

        Args:
            source:  DownloadResult or path to the GRD .SAFE.zip.
            output_dir: Output folder.
            label:   Optional filename label (e.g. the acquisition date).
            lut:     Calibration LUT: ``"sigmaNought"`` (default, sigma0),
                     ``"betaNought"``, ``"gamma"`` or ``"dn"``. Note ``gamma``
                     is the ellipsoid-based gamma0 of the annotation, NOT
                     terrain-flattened gamma0.
            remove_noise: Subtract thermal noise before calibrating. If the
                     noise annotation is missing or unreadable the product is
                     still calibrated, a warning is logged and the output is
                     tagged ``NOISE_REMOVED=unavailable...``.
            out_db:  Write 10*log10(value) (default) or linear power.
            floor_db: Pixels whose noise-subtracted power is <= 0 (common in
                     very dark areas, especially VH) are NaN by default. Pass
                     e.g. ``-35.0`` to clamp every valid pixel to at least
                     that value instead (non-positive pixels become the
                     floor, so dark areas stay in the time series).
            block_rows: Rows per processing block.
            aoi_bbox: Optional ``(min_lon, min_lat, max_lon, max_lat)``. When
                     given, only the part of the scene around it (plus
                     ~3 km margin) is calibrated and written: roughly 10 MB
                     instead of the ~0.5-1 GB of a full scene, and much
                     faster. Needs the product's GCPs (GRD always has them);
                     without them the call fails rather than guessing.
            streaming_cleanup: Verify-then-delete. When True, the raw
                     .SAFE.zip is deleted ONLY after the output COG has been
                     written and verified. On any failure the zip is kept.
                     The temporary extracted/georeferenced intermediates are
                     always removed. WARNING: the zip is gone afterwards, so
                     extract every polarisation you need from it first (call
                     with ``streaming_cleanup=True`` on the last one).

        Returns:
            Path to the verified COG, or None on failure (see logged errors).
        """
        import rasterio
        import rasterio.shutil

        if lut not in _CAL_LUTS:
            raise ValueError(f"lut must be one of {_CAL_LUTS}, got {lut!r}")
        if block_rows < 1:
            raise ValueError("block_rows must be >= 1")

        zip_path = self._resolve_path(source)
        if zip_path is None:
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        quantity = {"sigmaNought": "sigma0", "betaNought": "beta0",
                    "gamma": "gamma0", "dn": "dn"}[lut]
        stem = f"{label}_{self._pol}" if label else self._pol
        final = output_dir / self.sigma0_filename(
            label, lut=lut, out_db=out_db, aoi_bbox=aoi_bbox)
        partial = final.with_name(final.name + ".partial")
        raw = output_dir / f"{stem}_raw.tif"          # extracted measurement
        radar = output_dir / f"{stem}_{quantity}_radar.tif"
        geo: Optional[Path] = None
        ok = False

        try:
            # RESUME: an already verified product is not recomputed.
            reusable = False
            if final.exists():
                reusable, why = _verify_product(final, require_crs=False)
                if reusable:
                    logger.info("Reusing verified %s", final.name)
                else:
                    logger.warning("Existing %s failed verification (%s); "
                                   "recomputing", final.name, why)

            if not reusable:
                # TRY: calibrate in radar geometry, georeference, write COG.
                with zipfile.ZipFile(zip_path) as zf:
                    tags, had_gcps = self._calibrate_to_radar_tif(
                        zf, raw, radar, lut, quantity, remove_noise,
                        out_db, floor_db, block_rows, aoi_bbox,
                    )
                if tags is None:
                    return None
                geo = georeference_via_gcps_if_needed(radar)
                with rasterio.open(geo, "r+") as ds:
                    ds.update_tags(**tags)
                rasterio.shutil.copy(
                    geo, partial, driver="COG", compress="DEFLATE",
                    predictor="YES", blocksize=512,
                    overview_resampling="AVERAGE",
                )

                # VERIFY before anything irreversible happens.
                good, why = _verify_product(partial, require_crs=had_gcps)
                if not good:
                    logger.error("Verification of %s failed (%s) — raw data "
                                 "kept", final.name, why)
                    return None
                os.replace(partial, final)

            ok = True

            # DELETE: only reachable after a verified product exists.
            if streaming_cleanup:
                try:
                    zip_path.unlink()
                    logger.info("streaming_cleanup: deleted %s", zip_path.name)
                except OSError as exc:
                    logger.warning("Could not delete %s: %s", zip_path, exc)
            return final

        except Exception as exc:  # noqa: BLE001 — keep raw data on ANY error
            logger.error("sigma0 extraction of %s failed: %s — raw data kept",
                         zip_path.name, exc)
            return None
        finally:
            # Scratch files are reproducible from the zip: always removed.
            for scratch in (raw, radar, partial, geo):
                if scratch is not None and scratch != final:
                    try:
                        Path(scratch).unlink()
                    except OSError:
                        pass
            if not ok:
                logger.debug("No verified output for %s", zip_path.name)
            gc.collect()

    def sigma0_filename(
        self, label: str = "", *, lut: str = "sigmaNought", out_db: bool = True,
        aoi_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> str:
        """Name extract_sigma0() gives its output, so a caller can check whether
        a verified product already exists BEFORE downloading the scene again.
        A cropped product carries a short hash of its box, so it is never
        mistaken for a full-scene one (or for another box)."""
        import hashlib

        quantity = {"sigmaNought": "sigma0", "betaNought": "beta0",
                    "gamma": "gamma0", "dn": "dn"}[lut]
        stem = f"{label}_{self._pol}" if label else self._pol
        crop = ""
        if aoi_bbox is not None:
            digest = hashlib.md5(
                ",".join(f"{v:.5f}" for v in aoi_bbox).encode()).hexdigest()[:6]
            crop = f"_aoi{digest}"
        return f"{stem}_{quantity}_{'db' if out_db else 'lin'}{crop}.tif"

    def _calibrate_to_radar_tif(
        self, zf: zipfile.ZipFile, raw: Path, radar: Path, lut: str,
        quantity: str, remove_noise: bool, out_db: bool,
        floor_db: Optional[float], block_rows: int,
        aoi_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[Optional[dict], bool]:
        """
        Extract the measurement TIFF and write the calibrated raster, still
        in radar geometry (GCPs copied over, no CRS), block by block.

        Returns ``(provenance_tags, source_had_gcps)``; tags is None when the
        product cannot be calibrated (missing calibration annotation etc.).
        """
        import warnings

        import rasterio
        import rasterio.control
        from rasterio.windows import Window

        names = zf.namelist()
        measurement = [
            n for n in names
            if "/measurement/" in n and f"-{self._pol}-" in n
            and n.endswith(".tiff")
        ]
        if not measurement:
            logger.error("No %s measurement band found", self._pol.upper())
            return None, False
        # Without a calibration LUT there is no honest sigma0: refuse
        # rather than silently writing uncalibrated numbers.
        cal_member = _pick_member(names, "calibration", "calibration", self._pol)
        if cal_member is None:
            logger.error("No %s calibration annotation in product — cannot "
                         "calibrate", self._pol.upper())
            return None, False
        noise_member = _pick_member(names, "calibration", "noise", self._pol)

        _extract_member(zf, measurement[0], raw)

        with rasterio.open(raw) as src:
            width, height = src.width, src.height
            gcps, gcp_crs = src.gcps
            # Crop window in the original radar geometry (the whole scene by
            # default). The LUTs are still indexed by full-scene line/pixel.
            r_lo, r_hi, c_lo, c_hi = 0, height, 0, width
            if aoi_bbox is not None:
                win = _aoi_window(gcps, aoi_bbox, width, height) if gcps else None
                if win is None:
                    logger.error(
                        "Cannot crop to the AOI (no GCPs, or the box lies "
                        "outside this scene)")
                    return None, False
                r_lo, r_hi, c_lo, c_hi = win
                # keep only the GCPs nearest the window, shifted to the crop
                cr, cc = (r_lo + r_hi) / 2, (c_lo + c_hi) / 2
                gcps = sorted(
                    gcps, key=lambda g: (g.row - cr) ** 2 + (g.col - cc) ** 2
                )[:_AOI_KEEP_GCPS]
                gcps = [
                    rasterio.control.GroundControlPoint(
                        row=g.row - r_lo, col=g.col - c_lo, x=g.x, y=g.y, z=g.z)
                    for g in gcps
                ]
            out_w, out_h = c_hi - c_lo, r_hi - r_lo
            cal = _parse_vector_lut(
                zf.read(cal_member), ("calibrationVector",), (lut,), width
            )
            noise, noise_note = None, "disabled"
            if remove_noise:
                noise, noise_note = self._load_noise(zf, noise_member, width)

            profile = {
                "driver": "GTiff", "dtype": "float32", "count": 1,
                "width": out_w, "height": out_h, "nodata": float("nan"),
                "tiled": True, "blockxsize": 512, "blockysize": 512,
                "compress": "deflate", "predictor": 3, "BIGTIFF": "IF_SAFER",
            }
            n_valid = n_neg = 0
            floor_lin = None if floor_db is None else 10.0 ** (floor_db / 10.0)

            with warnings.catch_warnings():
                # Radar-geometry rasters legitimately have no CRS yet.
                warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
                with rasterio.open(radar, "w", **profile) as dst:
                    if gcps:
                        dst.gcps = (gcps, gcp_crs)
                    for r0 in range(r_lo, r_hi, block_rows):
                        r1 = min(r_hi, r0 + block_rows)
                        win = Window(c_lo, r0, out_w, r1 - r0)
                        dn = src.read(1, window=win).astype(np.float32)

                        valid = dn > 0                      # DN 0 = no data
                        power = dn * dn
                        if noise is not None:
                            power -= noise.power(r0, r1, width)[:, c_lo:c_hi]
                        a = cal.rows(r0, r1)[:, c_lo:c_hi]
                        usable = valid & (a > 0)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            value = power / (a * a)

                        neg = usable & (value <= 0)
                        n_valid += int(usable.sum())
                        n_neg += int(neg.sum())
                        if floor_lin is None:
                            value[neg] = np.nan
                        else:
                            np.maximum(value, floor_lin, out=value, where=usable)
                        value[~usable] = np.nan
                        if out_db:
                            with np.errstate(divide="ignore", invalid="ignore"):
                                value = 10.0 * np.log10(value)
                        dst.write(value.astype(np.float32), 1,
                                  window=Window(0, r0 - r_lo, out_w, r1 - r0))

        neg_frac = n_neg / n_valid if n_valid else 0.0
        logger.info(
            "%s calibrated (%s, noise: %s); %.2f%% of valid pixels <= 0 "
            "after noise subtraction", quantity, self._pol.upper(),
            noise_note, 100 * neg_frac,
        )
        tags = {
            "CALIBRATION": quantity, "CALIBRATION_LUT": lut,
            "NOISE_REMOVED": noise_note,
            "UNITS": "dB" if out_db else "linear power",
            "NONPOSITIVE_AFTER_NOISE_FRACTION": f"{neg_frac:.6f}",
            "NONPOSITIVE_POLICY": "NaN" if floor_db is None else f"floor {floor_db} dB",
            "TERRAIN_CORRECTED": "no (ellipsoid, GRD ground range)",
        }
        if aoi_bbox is not None:
            tags["AOI_BBOX_LONLAT"] = ",".join(f"{v:.5f}" for v in aoi_bbox)
        return tags, bool(gcps)

    @staticmethod
    def _load_noise(
        zf: zipfile.ZipFile, member: Optional[str], width: int
    ) -> Tuple[Optional[_NoiseModel], str]:
        """Parse the noise annotation; degrade gracefully when unusable."""
        if member is None:
            logger.warning("No noise annotation in product — calibrating "
                           "WITHOUT thermal-noise removal")
            return None, "unavailable: no noise annotation"
        try:
            xml = zf.read(member)
            rng = _parse_vector_lut(
                xml, ("noiseRangeVector", "noiseVector"),
                ("noiseRangeLut", "noiseLut"), width,
            )
            azimuth = _AzimuthNoise(xml)
        except (ET.ParseError, ValueError, KeyError) as exc:
            logger.warning("Noise annotation unreadable (%s) — calibrating "
                           "WITHOUT thermal-noise removal", exc)
            return None, f"unavailable: unreadable ({exc})"
        if azimuth.blocks:
            return _NoiseModel(rng, azimuth), "range+azimuth"
        return _NoiseModel(rng, None), "range only (no azimuth noise, IPF<2.9)"

    def _resolve_path(
        self, source: Union["DownloadResult", str, Path]
    ) -> Optional[Path]:
        """Resolve a usable file path from a DownloadResult, string, or Path."""
        if hasattr(source, "output_path") or hasattr(source, "output_paths"):
            output_path = getattr(source, "output_path", None)
            if output_path is not None:
                p = Path(output_path)
                if p.exists():
                    return p
                logger.warning(
                    "DownloadResult.output_path does not exist on disk: %s", p
                )
            output_paths = getattr(source, "output_paths", None) or []
            for p in output_paths:
                p = Path(p)
                if p.exists():
                    return p
            success = getattr(source, "success", None)
            error = getattr(source, "error", None)
            if success is False:
                logger.error("Download did not succeed: %s", error)
            return None

        p = Path(source)
        if p.exists():
            return p
        logger.error("Path does not exist: %s", p)
        return None