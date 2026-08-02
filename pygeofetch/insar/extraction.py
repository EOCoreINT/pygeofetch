"""
SLCExtractor — extract usable VV/VH measurement GeoTIFFs from downloaded
Sentinel-1 SLC .SAFE archives.

Sentinel-1 SLC products are delivered as a .SAFE folder (inside a .zip)
containing 6 separate measurement TIFFs — one per sub-swath (IW1/IW2/IW3)
per polarisation (VV/VH). InterferogramGenerator needs a single flat
complex GeoTIFF per scene, so this module:

  1. Lists the measurement TIFFs inside the downloaded zip for a given
     polarisation.
  2. Reads each sub-swath's embedded Ground Control Points (GCPs) via
     rasterio — Sentinel-1 SLC TIFFs carry these directly, so no
     annotation XML parsing is needed for a coverage check.
  3. Picks the sub-swath whose GCP-derived footprint actually overlaps
     the requested AOI.
  4. Extracts just that one TIFF to disk as a flat, directly-usable file.

Takes DownloadResult objects directly (from client.download()) rather than
re-deriving file paths from scene metadata — DownloadResult.output_path
already holds the exact path the provider wrote to, which avoids an entire
class of filename/subfolder-mismatch bugs.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from pygeofetch.models.download_task import DownloadResult
    from pygeofetch.models.search_query import BoundingBox

logger = logging.getLogger("pygeofetch.insar.extraction")


class SLCExtractor:
    """
    Extract usable measurement GeoTIFFs from downloaded Sentinel-1 SLC archives.

    Example::

        from pygeofetch import PyGeoFetch
        from pygeofetch.insar import SLCExtractor

        client = PyGeoFetch()
        results = client.download([ref_scene, sec_scene], destination=out_dir)

        extractor = SLCExtractor(polarisation="VV")
        ref_tif, sec_tif = extractor.extract_pair(
            results[0], results[1], aoi=aoi_bbox, output_dir=out_dir,
        )

        # ref_tif / sec_tif are now flat GeoTIFFs ready for
        # InterferogramGenerator.process_pair(ref_tif, sec_tif, ...)
    """

    def __init__(self, polarisation: str = "VV") -> None:
        self._pol = polarisation.lower()

    # ── public API ────────────────────────────────────────────────────────────

    def extract_pair(
        self,
        reference: Union["DownloadResult", str, Path],
        secondary: Union["DownloadResult", str, Path],
        aoi: "BoundingBox",
        output_dir: Union[str, Path],
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Extract the AOI-matching sub-swath from both a reference and
        secondary SLC download in one call.

        Args:
            reference: DownloadResult from client.download() (preferred —
                       uses its .output_path directly, no guessing), or a
                       direct path to the downloaded .SAFE.zip.
            secondary: Same, for the secondary scene.
            aoi:       BoundingBox to match against sub-swath footprints.
            output_dir: Where to write the extracted flat GeoTIFFs.

        Returns:
            (reference_tif, secondary_tif) — either may be None if
            extraction failed for that scene (see logged warnings for why).
        """
        ref_zip = self._resolve_path(reference)
        sec_zip = self._resolve_path(secondary)

        if ref_zip is None or sec_zip is None:
            missing = "reference" if ref_zip is None else "secondary"
            logger.error(
                "Could not resolve a usable file path for the %s scene — "
                "check that its download completed successfully.",
                missing,
            )
            return None, None

        logger.info("Reference archive: %s", ref_zip.name)
        ref_tif = self.extract_scene(ref_zip, aoi, output_dir, label="reference")

        logger.info("Secondary archive: %s", sec_zip.name)
        sec_tif = self.extract_scene(sec_zip, aoi, output_dir, label="secondary")

        return ref_tif, sec_tif

    def extract_scene(
        self,
        zip_path: Union[str, Path],
        aoi: "BoundingBox",
        output_dir: Union[str, Path],
        label: str = "",
        resume: bool = False,
    ) -> Optional[Path]:
        """
        Find the sub-swath covering the AOI in one SLC archive and extract it.

        Args:
            zip_path:   Path to the downloaded .SAFE.zip.
            aoi:        BoundingBox to match against sub-swath footprints.
            output_dir: Where to write the extracted flat GeoTIFF.
            label:      Used to build the output filename
                       (f"{label}_{polarisation}.tif").
            resume:     If True and the expected output file already exists
                       AND opens as a genuine, valid raster, skip extraction
                       entirely and return that existing path. A corrupted
                       or partially-written file at that path always
                       triggers a fresh extraction regardless of this flag
                       — resume never means "tolerate a broken file", the
                       same contract already established and tested for
                       AdaptiveDownloader's own resume parameter. Default
                       False preserves the original always-extract
                       behaviour for existing callers.

        Returns:
            Path to the extracted GeoTIFF, or None if no sub-swath in this
            archive overlaps the AOI (check logs for per-sub-swath
            footprints when this happens).
        """
        zip_path = Path(zip_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if resume:
            expected_path = (
                output_dir / f"{label}_{self._pol}.tif"
                if label
                else output_dir / f"{zip_path.stem}_{self._pol}.tif"
            )
            if expected_path.exists():
                try:
                    import rasterio

                    with rasterio.open(expected_path) as src:
                        if src.width > 0 and src.height > 0:
                            logger.info(
                                "Resume: %s already extracted and valid (%dx%d) — skipping",
                                expected_path.name, src.width, src.height,
                            )
                            return expected_path
                except Exception as exc:
                    logger.warning(
                        "Resume: %s exists but failed to open (%s) — re-extracting",
                        expected_path.name, exc,
                    )

        if not zip_path.exists():
            logger.error("Archive not found: %s", zip_path)
            return None

        aoi_tuple = (aoi.min_lon, aoi.min_lat, aoi.max_lon, aoi.max_lat)
        members = self._find_subswaths(zip_path)

        if not members:
            logger.warning(
                "No %s measurement TIFFs found inside %s. This archive may "
                "not be a standard Sentinel-1 SLC .SAFE.zip, or the "
                "polarisation requested (%s) is not present.",
                self._pol.upper(),
                zip_path.name,
                self._pol.upper(),
            )
            return None

        logger.info(
            "Found %d %s sub-swath(s) in %s",
            len(members),
            self._pol.upper(),
            zip_path.name,
        )

        matched_member = None
        for member in members:
            footprint = self._gcp_footprint(zip_path, member)
            if footprint is None:
                continue
            swath = self._swath_label(member)
            overlaps = self._bbox_overlaps(footprint, aoi_tuple)
            logger.debug(
                "  %s: footprint=%s  overlaps AOI: %s",
                swath,
                tuple(round(v, 2) for v in footprint),
                overlaps,
            )
            if overlaps and matched_member is None:
                matched_member = member

        if matched_member is None:
            logger.warning(
                "No sub-swath in %s overlaps the requested AOI %s. "
                "Check your bbox or scene selection — the scene's overall "
                "footprint may cover the AOI while no single sub-swath does "
                "if the AOI straddles a sub-swath boundary.",
                zip_path.name,
                aoi_tuple,
            )
            return None

        out_path = (
            output_dir / f"{label}_{self._pol}.tif"
            if label
            else output_dir / f"{zip_path.stem}_{self._pol}.tif"
        )

        cropped = self._crop_to_aoi(zip_path, matched_member, aoi_tuple, out_path)
        if cropped is None:
            logger.warning(
                "Cropping %s to AOI failed — falling back to extracting the "
                "full, uncropped sub-swath (this will be much larger and "
                "slower for everything downstream).",
                self._swath_label(matched_member),
            )
            with zipfile.ZipFile(zip_path) as zf:
                with zf.open(matched_member) as src_f, open(out_path, "wb") as dst_f:
                    dst_f.write(src_f.read())
            self._tag_matched_swath(out_path, matched_member)

        logger.info(
            "Extracted %s -> %s", self._swath_label(matched_member), out_path.name
        )
        return out_path

    def show_on_map(
        self,
        extracted_path: Union[str, Path],
        colormap: str = "gray",
        opacity: float = 0.85,
    ) -> Any:
        """
        Display an extracted SLC/amplitude GeoTIFF on a real,
        georeferenced satellite basemap, using the same MapViewer
        infrastructure already proven earlier this session (real
        coherence and location maps).

        Real amplitude data has an extreme dynamic range (a handful of
        very bright, isolated returns next to a broad, dim background),
        so this reads real percentile bounds from the actual raster
        before display rather than using a fixed, likely-wrong vmin/vmax
        — a raw min/max would let a few bright outliers wash out
        everything else, an easy, real mistake to make with SAR
        amplitude specifically (unlike more evenly-distributed data).

        Args:
            extracted_path: Path returned by extract_scene().
            colormap:       Matplotlib colormap name. "gray" (default)
                            matches how SAR amplitude/intensity is
                            conventionally displayed (as in both
                            uploaded tutorials' own intensity figures).
            opacity:        Layer opacity, 0-1.

        Returns:
            The real MapViewer instance (call .show() in a notebook,
            or it can be displayed directly if that's the last
            expression in a cell).
        """
        import numpy as np
        import rasterio

        from pygeofetch.viz.map import MapViewer

        extracted_path = Path(extracted_path)
        with rasterio.open(extracted_path) as src:
            data = src.read(1)
            profile = src.profile.copy()

        # Real, necessary step, not just for the percentile math: the
        # underlying map rendering library (leafmap/localtileserver)
        # has no complex-dtype support at all -- confirmed directly,
        # no dtype/complex handling exists anywhere in that code path.
        # A real, complex-valued SLC file cannot be displayed directly
        # regardless of vmin/vmax; a real, temporary, real-valued
        # amplitude GeoTIFF must be written first.
        display_data = np.abs(data) if np.iscomplexobj(data) else data
        finite = display_data[np.isfinite(display_data) & (display_data != 0)]
        vmin, vmax = (
            (float(np.percentile(finite, 2)), float(np.percentile(finite, 98)))
            if finite.size > 0
            else (None, None)
        )

        import tempfile

        tmp_path = Path(tempfile.mkdtemp()) / f"{extracted_path.stem}_amplitude.tif"
        profile.update(dtype="float32", count=1, nodata=0.0)
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(display_data.astype(np.float32)[np.newaxis])

        with rasterio.open(tmp_path) as src:
            bounds = src.bounds
        center_lat = (bounds.bottom + bounds.top) / 2
        center_lon = (bounds.left + bounds.right) / 2

        mv = MapViewer(center=(center_lat, center_lon), zoom=12)
        mv.add_basemap("SATELLITE")
        mv.add_raster(
            str(tmp_path), colormap=colormap,
            layer_name=extracted_path.stem, opacity=opacity,
            vmin=vmin, vmax=vmax,
        )
        logger.info(
            "Real amplitude display range (2nd-98th percentile): [%.3f, %.3f]",
            vmin if vmin is not None else float("nan"),
            vmax if vmax is not None else float("nan"),
        )
        # Real fix: MapViewer needs .show() called explicitly to
        # trigger Jupyter's rich display -- confirmed directly earlier
        # this session (mv_social.show() case). Returning the raw
        # MapViewer without calling .show() meant the layer was really
        # added (confirmed by the log line) but nothing ever rendered.
        return mv.show()

    def _tag_matched_swath(self, out_path: Path, member_name: str) -> None:
        """Tag an already-written GeoTIFF with its matched sub-swath label,
        for the fallback (raw-copy) extraction path, which doesn't go
        through rasterio for its initial write the way _crop_to_aoi()
        does. Failure here is non-fatal -- the file itself is still
        correct, just missing this one piece of discoverable metadata,
        so downstream code falls back to arbitrary-first-match behaviour
        rather than losing the extraction entirely."""
        try:
            import rasterio

            with rasterio.open(out_path, "r+") as dst:
                dst.update_tags(matched_swath=self._swath_label(member_name).lower())
        except Exception as exc:
            logger.warning(
                "Could not tag %s with its matched sub-swath: %s", out_path.name, exc
            )

    def _crop_to_aoi(
        self,
        zip_path: Path,
        member_name: str,
        aoi_tuple: Tuple[float, float, float, float],
        out_path: Path,
        margin_frac: float = 0.15,
    ) -> Optional[Path]:
        """
        Crop a zipped Sentinel-1 measurement TIFF to the AOI, reading
        directly from inside the zip (no full-file extraction) and
        writing only the windowed AOI region.

        Sentinel-1 SLC TIFFs are GCP-georeferenced, not a simple affine
        transform, so an exact pixel window can't be computed directly.
        This fits an approximate affine transform from the real embedded
        GCPs (rasterio.transform.from_gcps — the standard technique for
        this), then adds a safety margin around the computed window to
        protect against that approximation's error rather than risk
        cutting off the real AOI.

        Args:
            margin_frac: Fractional padding added to the computed window
                       on each side (0.15 = 15%) — generous on purpose,
                       since padding too much only costs a little extra
                       size, while padding too little risks silently
                       losing part of the real area of interest.

        Returns:
            out_path on success, None if cropping wasn't possible for
            any reason (missing GCPs, unreadable TIFF, degenerate
            transform) — callers should fall back to full extraction
            rather than fail outright.
        """
        import rasterio
        from rasterio.transform import from_gcps
        from rasterio.windows import Window
        from rasterio.crs import CRS

        vsi_path = f"/vsizip/{zip_path}/{member_name}"
        try:
            with rasterio.open(vsi_path) as src:
                gcps, _gcp_crs = src.gcps
                if not gcps or len(gcps) < 4:
                    logger.warning(
                        "%s: too few GCPs (%d) to fit a reliable crop transform",
                        member_name, len(gcps) if gcps else 0,
                    )
                    return None

                approx_transform = from_gcps(gcps)

                min_lon, min_lat, max_lon, max_lat = aoi_tuple
                # Inverse-transform all 4 AOI corners and take the bounding
                # box of the results, rather than rasterio.windows.from_bounds()
                # -- that function assumes a standard north-up transform
                # (row increases as latitude decreases) and raises "Bounds
                # and transform are inconsistent" when it doesn't, which is
                # exactly the orientation real ascending-pass Sentinel-1
                # data has (confirmed directly against real product
                # footprints). This corner-based approach makes no
                # assumption about orientation at all.
                inv_transform = ~approx_transform
                corners = [
                    (min_lon, min_lat), (min_lon, max_lat),
                    (max_lon, min_lat), (max_lon, max_lat),
                ]
                cols, rows = [], []
                for lon, lat in corners:
                    col, row = inv_transform * (lon, lat)
                    cols.append(col)
                    rows.append(row)
                window = Window(
                    col_off=min(cols), row_off=min(rows),
                    width=max(cols) - min(cols), height=max(rows) - min(rows),
                )

                # Safety margin: the GCP-fitted transform is only an
                # approximation, so pad the window generously rather than
                # risk cropping out part of the real AOI.
                pad_w = max(window.width * margin_frac, 50)
                pad_h = max(window.height * margin_frac, 50)
                col_off = window.col_off - pad_w
                row_off = window.row_off - pad_h
                width = window.width + 2 * pad_w
                height = window.height + 2 * pad_h

                # Clamp to the actual raster extent -- the padded window
                # can legitimately extend past the real data bounds.
                col_off_clamped = max(0, col_off)
                row_off_clamped = max(0, row_off)
                width_clamped = min(width - (col_off_clamped - col_off), src.width - col_off_clamped)
                height_clamped = min(height - (row_off_clamped - row_off), src.height - row_off_clamped)

                if width_clamped <= 0 or height_clamped <= 0:
                    logger.warning(
                        "%s: computed crop window falls entirely outside the "
                        "raster bounds -- GCP transform may be unreliable",
                        member_name,
                    )
                    return None

                read_window = Window(
                    col_off_clamped, row_off_clamped, width_clamped, height_clamped
                )
                data = src.read(1, window=read_window)

                if data.size == 0:
                    return None

                # Real, locally-accurate transform for the cropped region,
                # derived from the same GCP-fit transform used to locate it.
                crop_transform = rasterio.windows.transform(read_window, approx_transform)

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(
                    out_path, "w", driver="GTiff",
                    dtype=data.dtype, count=1,
                    width=data.shape[1], height=data.shape[0],
                    crs=CRS.from_epsg(4326), transform=crop_transform,
                    compress="deflate",
                ) as dst:
                    dst.write(data, 1)
                    # Record the crop's origin in the FULL, UNCROPPED
                    # scene's pixel coordinates. Real orbit-based
                    # coregistration computes offset fields using
                    # annotation.py's real acquisition timing, which
                    # describes the full, uncropped sub-swath -- without
                    # this offset recorded, downstream code has no way to
                    # know that local row/col 0,0 in this cropped file is
                    # actually row/col crop_row_off,crop_col_off in the
                    # scene the orbit math is expressed in, and would
                    # silently evaluate the fitted offset polynomial at
                    # the wrong coordinates.
                    dst.update_tags(
                        crop_row_off=str(row_off_clamped),
                        crop_col_off=str(col_off_clamped),
                        matched_swath=self._swath_label(member_name).lower(),
                    )

                logger.info(
                    "Cropped %s: %dx%d -> %dx%d (%.1fx smaller)",
                    self._swath_label(member_name), src.width, src.height,
                    data.shape[1], data.shape[0],
                    (src.width * src.height) / max(data.size, 1),
                )
                return out_path
        except Exception as exc:
            logger.warning("Could not crop %s to AOI: %s", member_name, exc)
            return None

    def list_subswaths(self, zip_path: Union[str, Path]) -> List[str]:
        """
        List the measurement TIFF entries for the configured polarisation
        inside an SLC .SAFE zip, without extracting anything.
        """
        return self._find_subswaths(Path(zip_path))

    # ── internal helpers ──────────────────────────────────────────────────────

    def _resolve_path(
        self, source: Union["DownloadResult", str, Path]
    ) -> Optional[Path]:
        """
        Resolve a usable file path from a DownloadResult, string, or Path.

        Prefers DownloadResult.output_path (the exact path the provider
        actually wrote to) over any path re-derivation, which avoids
        filename/subfolder-mismatch bugs entirely.
        """
        # DownloadResult (duck-typed check to avoid a hard import dependency)
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

        # Plain path
        p = Path(source)
        if p.exists():
            return p
        logger.error("Path does not exist: %s", p)
        return None

    def _find_subswaths(self, zip_path: Path) -> List[str]:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
        except zipfile.BadZipFile as exc:
            logger.error("Cannot open %s as a zip archive: %s", zip_path.name, exc)
            return []

        marker = f"-{self._pol}-"
        matches = [
            n
            for n in names
            if "/measurement/" in n and marker in n and n.endswith(".tiff")
        ]
        return sorted(matches)

    def _gcp_footprint(
        self, zip_path: Path, member_name: str
    ) -> Optional[Tuple[float, float, float, float]]:
        """Read embedded GCPs from a zipped measurement TIFF and return its
        approximate (min_lon, min_lat, max_lon, max_lat) footprint."""
        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        vsi_path = f"/vsizip/{zip_path}/{member_name}"
        try:
            with rasterio.open(vsi_path) as src:
                gcps, _gcp_crs = src.gcps
                if not gcps:
                    return None
                lons = [g.x for g in gcps]
                lats = [g.y for g in gcps]
                return (min(lons), min(lats), max(lons), max(lats))
        except Exception as exc:
            logger.warning("Could not read GCPs from %s: %s", member_name, exc)
            return None

    def _bbox_overlaps(
        self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
    ) -> bool:
        """a, b = (min_lon, min_lat, max_lon, max_lat)."""
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    def _swath_label(self, member_name: str) -> str:
        lower = member_name.lower()
        for swath in ("iw1", "iw2", "iw3", "ew1", "ew2", "ew3", "ew4", "ew5"):
            if swath in lower:
                return swath.upper()
        return "?"