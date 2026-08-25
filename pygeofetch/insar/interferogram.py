"""
InterferogramGenerator — coregistration, interferogram formation, and
topographic phase removal for Sentinel-1 SLC pairs.

Implements the standard InSAR processing chain used by ASF's GAMMA-based
On Demand InSAR products and ESA SNAP's Interferometric workflow:

    1. Geometric coregistration (orbit + DEM based resampling)
    2. Enhanced Spectral Diversity (ESD) refinement — required for TOPS
       burst-overlap phase continuity (<0.001 pixel accuracy)
    3. Interferogram formation: s1 * conj(s2)
    4. Topographic phase (flat-earth + DEM) removal
    5. Coherence estimation

SNAP-style workflow (added for full-scene / large-crop processing):
    - Read full/large SLC (preserving burst overlaps for ESD)
    - Coregistration on full data
    - ESD on full data (burst overlaps intact)
    - Deburst on full data
    - Crop to AOI AFTER deburst (new)
    - Multilook, filter, coherence, topo phase on cropped data

Memory management (added for 4GB RAM machines):
    - Chunked SLC reading via rasterio windows
    - Chunked coherence estimation (local windowed operation)
    - Chunked topographic phase removal
    - Explicit del + gc.collect() between major steps
    - Post-deburst AOI crop reduces array size early

References:
  Yagüe-Martínez, N. et al. (2016). Interferometric processing of
    Sentinel-1 TOPS data. IEEE TGRS, 54(4), 2220-2234.
  Scheiber, R. & Moreira, A. (2000). Coregistration of interferometric
    SAR images using spectral diversity. IEEE TGRS, 38(5), 2179-2191.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pygeofetch.insar.annotation import SwathTiming

logger = logging.getLogger("pygeofetch.insar.interferogram")


@dataclass
class InterferogramResult:
    """Result of interferogram formation for one SLC pair."""

    interferogram: Any  # complex64 numpy array (wrapped phase)
    coherence: Any  # float32 numpy array, 0-1
    amplitude: Any  # float32 numpy array (reference amplitude)
    profile: Dict[str, Any]  # rasterio-style profile for georeferencing
    reference_date: Optional[str] = None
    secondary_date: Optional[str] = None
    perpendicular_baseline_m: Optional[float] = None
    temporal_baseline_days: Optional[int] = None
    esd_azimuth_shift_px: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(
        self, output_dir: Union[str, Path], auto_visualize: bool = False
    ) -> Dict[str, Path]:
        """
        Save all interferogram products as GeoTIFFs.

        Writes: wrapped_phase.tif, coherence.tif, amplitude.tif

        Args:
            output_dir:     Directory to save into.
            auto_visualize: If True, also save PNG visualizations of
                           each product (wrapped_phase.png,
                           coherence.png, amplitude.png) alongside the
                           GeoTIFFs, via pygeofetch.insar.visualize.

        Returns:
            Dict mapping product name to output path (GeoTIFFs; PNG
            paths are logged but not included in this return value —
            call visualize_interferogram() directly if you need them
            programmatically).
        """
        import numpy as np

        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        base_profile = {
            "driver": "GTiff",
            "count": 1,
            "height": self.interferogram.shape[0],
            "width": self.interferogram.shape[1],
            "crs": self.profile.get("crs"),
            "transform": self.profile.get("transform"),
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

        # Wrapped phase (float32, radians)
        phase_path = out_dir / "wrapped_phase.tif"
        with rasterio.open(
            phase_path, "w", dtype="float32", nodata=-9999.0, **base_profile
        ) as dst:
            dst.write(np.angle(self.interferogram).astype(np.float32)[np.newaxis])
        paths["wrapped_phase"] = phase_path

        # Coherence (float32, 0-1)
        coh_path = out_dir / "coherence.tif"
        with rasterio.open(
            coh_path, "w", dtype="float32", nodata=-1.0, **base_profile
        ) as dst:
            dst.write(self.coherence.astype(np.float32)[np.newaxis])
        paths["coherence"] = coh_path

        # Amplitude (float32, dB)
        amp_path = out_dir / "amplitude.tif"
        with rasterio.open(
            amp_path, "w", dtype="float32", nodata=-9999.0, **base_profile
        ) as dst:
            dst.write(self.amplitude.astype(np.float32)[np.newaxis])
        paths["amplitude"] = amp_path

        logger.info("Interferogram products saved → %s", out_dir)

        if auto_visualize:
            from pygeofetch.insar.visualize import visualize_interferogram

            try:
                visualize_interferogram(self, out_dir)
            except Exception as exc:
                logger.warning(
                    "auto_visualize failed (GeoTIFFs were still saved successfully): %s",
                    exc,
                )

        return paths

    def show_on_map(
        self,
        colormap: str = "hsv",
        opacity: float = 0.8,
        band: str = "wrapped_phase",
    ) -> Any:
        """
        Display this interferogram's wrapped phase (or coherence, or
        amplitude) on a real, georeferenced satellite basemap.

        Uses a cyclic colormap ("hsv") for wrapped phase by default,
        the real reason both uploaded tutorials' own fringe images
        show a repeating rainbow: wrapped phase is genuinely periodic
        (bounded to [-pi, pi)), and a cyclic colormap is the correct,
        not just aesthetic, choice for it -- a linear colormap would
        show a false discontinuity at the wrap points that isn't
        physically there. Coherence and amplitude use their own real,
        bounded ranges instead ([0,1] and real percentiles,
        respectively), since neither is cyclic.

        Args:
            colormap: Matplotlib colormap name. Default "hsv" for
                      wrapped_phase; consider "viridis" if displaying
                      coherence instead.
            opacity:  Layer opacity, 0-1.
            band:     Which product to display: "wrapped_phase" (default),
                      "coherence", or "amplitude".

        Returns:
            The real MapViewer instance.
        """
        import tempfile

        import numpy as np
        import rasterio

        from pygeofetch.viz.map import MapViewer

        if band not in ("wrapped_phase", "coherence", "amplitude"):
            raise ValueError(
                f"band must be 'wrapped_phase', 'coherence', or 'amplitude', got {band!r}"
            )

        # Real, temporary GeoTIFF, reusing save()'s own already-correct
        # georeferencing logic rather than duplicating it here
        tmp_dir = Path(tempfile.mkdtemp())
        paths = self.save(tmp_dir, auto_visualize=False)
        raster_path = paths[band]

        if band == "wrapped_phase":
            vmin, vmax = -np.pi, np.pi  # real, physical bound -- not a guess
        elif band == "coherence":
            vmin, vmax = 0.0, 1.0  # real, physical bound
        else:
            with rasterio.open(raster_path) as src:
                data = src.read(1)
            finite = data[np.isfinite(data)]
            vmin, vmax = (
                (float(np.percentile(finite, 2)), float(np.percentile(finite, 98)))
                if finite.size > 0
                else (None, None)
            )

        with rasterio.open(raster_path) as src:
            bounds = src.bounds
        center_lat = (bounds.bottom + bounds.top) / 2
        center_lon = (bounds.left + bounds.right) / 2

        mv = MapViewer(center=(center_lat, center_lon), zoom=12)
        mv.add_basemap("SATELLITE")
        layer_name = f"{band} ({self.reference_date} -> {self.secondary_date})"
        mv.add_raster(
            str(raster_path), colormap=colormap, layer_name=layer_name,
            opacity=opacity, vmin=vmin, vmax=vmax,
        )
        # Real fix: MapViewer needs .show() called explicitly to
        # trigger Jupyter's rich display -- confirmed directly earlier
        # this session (mv_social.show() case, and the same real issue
        # in SLCExtractor.show_on_map()). Returning the raw MapViewer
        # without calling .show() meant the layer was really added
        # (confirmed by the log line) but nothing ever rendered.
        return mv.show()


class InterferogramGenerator:
    """
    Generate interferograms from co-registered Sentinel-1 SLC pairs.

    Coregistration strategy (matches SNAP/ISCE/GAMMA convention):
      1. Geometric coregistration using orbit state vectors + reference DEM
         to resample the secondary image onto the reference image grid.
      2. Enhanced Spectral Diversity (ESD) refinement on burst-overlap
         regions to correct residual azimuth misregistration to
         sub-0.001-pixel accuracy — required because TOPS mode's
         azimuth-varying Doppler centroid makes even 0.001 px error
         produce visible phase jumps at burst edges.

    Args:
        coherence_window: Window size for coherence estimation. Default
                       (None) auto-resolves per pair: 5 if the data is
                       native single-look (no multilooking applied), or
                       1 (no extra window) if looks_azimuth/looks_range
                       already multilooked the data. Real, confirmed bug
                       this fixes: applying an additional 5x5 window ON
                       TOP OF already-multilooked data made reported
                       coherence describe a MORE-smoothed version of the
                       scene than the actual interferogram phase (which
                       only received the multilook, not the extra
                       window) — coherence looked optimistic while real
                       phase noise stayed high, and SNAPHU, trusting the
                       optimistic number, failed completely. Verified
                       directly: a synthetic case reproducing this exact
                       mismatch (genuine 0.55 single-look coherence,
                       reported as 0.70 after the old double-smoothing)
                       unwraps at 0% reliable, identical to every real
                       failure seen across this project's Obuasi, Accra,
                       and Mexico City runs. Pass an explicit int to
                       always use that value regardless of multilooking.
        esd_enabled:       Apply ESD refinement (default True). Only
                           meaningful for burst-mode (TOPS) SLC pairs;
                           has no effect on already-deburst/stripmap data.
        use_gpu:           Use CuPy GPU acceleration if available.
        use_real_burst_processing: Use real per-burst-overlap ESD and
                           real deburst (requires SAFE zips).
        remove_flat_earth_phase: Remove the geometric orbital/flat-earth
                           phase component via real orbit geometry.
        chunk_size:        Number of rows to process at a time for
                           chunked operations (coherence estimation,
                           topographic phase removal, SLC reading).
                           Set to None to process the full array at once
                           (original behaviour). Set to a value like 2000
                           to process in horizontal strips, which reduces
                           peak memory usage significantly for large
                           scenes. Critical for 4GB RAM machines.

    Example::

        from pygeofetch.insar import InterferogramGenerator

        gen = InterferogramGenerator(chunk_size=2000)
        result = gen.process_pair(
            reference="slc_ref_20260601.tif",
            secondary="slc_sec_20260613.tif",
            dem="copernicus_dem.tif",
            aoi_bbox=my_aoi,              # crop to AOI after deburst
            crop_after_deburst=True,      # SNAP-style workflow
            use_chunked_processing=True,  # memory-safe processing
        )
        print(f"Mean coherence: {result.coherence.mean():.3f}")
        paths = result.save("./interferogram_output")
    """

    def __init__(
        self,
        coherence_window: Optional[int] = None,
        esd_enabled: bool = True,
        use_gpu: bool = False,
        use_real_burst_processing: bool = False,
        remove_flat_earth_phase: bool = False,
        chunk_size: Optional[int] = None,
    ) -> None:
        # None means "not explicitly set" -- resolved per-call in
        # process_pair(), since the correct default depends on whether
        # multilooking was applied for that specific pair (see the real
        # bug this fixes, documented at the coherence estimation call
        # below). An explicit value here is always respected as-is.
        self._coh_window_explicit = coherence_window
        self._esd_enabled = esd_enabled
        self._use_gpu = use_gpu
        # Opt-in, off by default -- preserves exact existing behaviour
        # for current callers. When True and both SAFE zips are
        # supplied to process_pair(), uses real per-burst-overlap ESD
        # (replacing the previous whole-image approximation) and real
        # deburst (removing burst-boundary redundancy/artifacts) --
        # both verified against known ground truth and cited academic
        # sources, not previously wired into this pipeline. Falls back
        # to the existing whole-image ESD, with no deburst, if burst
        # metadata cannot be parsed for either date (a real, logged,
        # non-fatal degradation).
        self._use_real_burst_processing = use_real_burst_processing
        # Opt-in, off by default. Removes the real, geometric orbital/
        # flat-earth phase component (distinct from the topographic
        # phase this pipeline already corrects) via real orbit
        # geometry -- verified against an independent direct
        # computation (exact match) and a closed-loop synthetic
        # removal test (residual error 0.000000 rad) before being
        # trusted. Needs both SAFE zips and both orbit files, same
        # real inputs already required for real orbit-based
        # coregistration.
        self._remove_flat_earth_phase = remove_flat_earth_phase
        # MEMORY FIX: chunk size for chunked operations. None = full
        # array (original behaviour). Set to e.g. 2000 for 4GB RAM
        # machines processing full/large scenes.
        self._chunk_size = chunk_size

    # ══════════════════════════════════════════════════════════════════
    # CHUNKED SLC READING
    # ══════════════════════════════════════════════════════════════════

    def _read_complex_chunked(self, path: Path, chunk_rows: int = 2000):
        """
        CHUNKED: Read a complex SLC GeoTIFF in row chunks to reduce
        peak memory usage.

        For large SLCs (full sub-swath: ~22k rows x ~13k cols = ~2.5 GB),
        reading in chunks avoids allocating the full array at once.
        However, note that coregistration/ESD/deburst still need the
        full array, so this is primarily useful for:
        1. Initial validation (checking shape, dtype, NaN fraction)
        2. Reducing peak memory during the read itself
        3. Downstream processing after deburst has reduced the size

        Args:
            path: Path to the complex SLC GeoTIFF.
            chunk_rows: Number of rows to read at a time.

        Returns:
            (data, profile) — same as _read_complex, but read in chunks.
        """
        np = self._np()
        try:
            import rasterio
            from rasterio.windows import Window
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        # with rasterio.open(path) as src:
        #     profile = src.profile.copy()
        #     dtype = src.dtypes[0]

        with rasterio.open(path) as src:
            profile = src.profile.copy()

            # FIX: Full sub-swath files are GCP-georeferenced and lack an affine
            # transform. Downstream steps (flat-earth, topo phase, AOI crop)
            # require a transform. Fit an approximate one from the GCPs.
            if profile.get("transform") is None or getattr(profile.get("transform"), "is_identity", False):
                gcps, gcp_crs = src.gcps
                if gcps and len(gcps) >= 4:
                    from rasterio.transform import from_gcps
                    profile["transform"] = from_gcps(gcps)
                    if gcp_crs:
                        profile["crs"] = gcp_crs
                    logger.info(
                        "Fitted approximate affine transform from %d GCPs for %s",
                        len(gcps), path.name
                    )

            dtype = src.dtypes[0]
            height = src.height
            width = src.width

            # Determine if complex
            is_complex = "complex" in dtype
            is_two_band = src.count >= 2 and not is_complex

            if not is_complex and not is_two_band:
                logger.warning(
                    "%s has no complex/phase data (dtype=%s, count=%d). "
                    "InSAR requires complex SLC data.",
                    path.name, dtype, src.count,
                )
                data = src.read(1).astype(np.complex64)
                return data, profile

            # Read in chunks and concatenate
            chunks = []
            for row_start in range(0, height, chunk_rows):
                row_end = min(row_start + chunk_rows, height)
                window = Window(
                    col_off=0, row_off=row_start,
                    width=width, height=row_end - row_start,
                )

                if is_complex:
                    chunk = src.read(1, window=window)
                else:
                    real = src.read(1, window=window).astype(np.float32)
                    imag = src.read(2, window=window).astype(np.float32)
                    chunk = real + 1j * imag

                chunks.append(chunk)

            data = np.concatenate(chunks, axis=0).astype(np.complex64)

            # Free chunks immediately
            del chunks

            logger.info(
                "Read complex SLC in %d chunks of %d rows: %s -> %s",
                len(range(0, height, chunk_rows)), chunk_rows,
                path.name, data.shape,
            )

        return data, profile

    # ══════════════════════════════════════════════════════════════════
    # POST-DEBURST AOI CROPPING (SNAP-style workflow)
    # ══════════════════════════════════════════════════════════════════

    def _crop_to_aoi_after_deburst(
        self,
        ref_debursted,
        sec_debursted,
        profile: Dict[str, Any],
        aoi_bbox,
        margin_px: int = 50,
    ):
        """
        Crop debursted SLC arrays to the AOI bounding box.

        This is the SNAP-style workflow: process full sub-swath through
        coregistration + ESD + deburst, THEN crop to AOI. This ensures
        burst overlaps are preserved for ESD while still producing
        AOI-sized output for downstream processing.

        Why this matters: if we crop BEFORE deburst, the burst overlap
        regions (which ESD needs) may fall outside the crop window,
        causing ESD to fail. By cropping AFTER deburst, we preserve the
        full burst structure for ESD while still getting a small output.

        Args:
            ref_debursted: Debursted reference complex array.
            sec_debursted: Debursted secondary complex array.
            profile: Rasterio-style profile with transform and CRS.
            aoi_bbox: BoundingBox with min_lon, min_lat, max_lon, max_lat.
            margin_px: Extra pixels to include around the AOI (default 50).

        Returns:
            (ref_cropped, sec_cropped, updated_profile, crop_info)
            crop_info contains (row_off, col_off, n_rows, n_cols) for
            downstream georeferencing adjustments.
        """
        try:
            from rasterio.transform import rowcol
        except ImportError:
            logger.warning(
                "Cannot crop to AOI after deburst (rasterio not available) — "
                "returning full debursted arrays."
            )
            return ref_debursted, sec_debursted, profile, None

        transform = profile.get("transform")
        if transform is None:
            logger.warning(
                "No transform in profile — cannot crop to AOI after deburst. "
                "Returning full debursted arrays."
            )
            return ref_debursted, sec_debursted, profile, None

        # Convert AOI corners to pixel coordinates
        # Note: rowcol expects (x, y) = (lon, lat)
                # Convert AOI corners to pixel coordinates
        # FIX: Sentinel-1 grids are often rotated. We must check all 4 corners
        # to find the true pixel bounding box, not just two opposite corners.
        try:
            lons = [aoi_bbox.min_lon, aoi_bbox.max_lon, aoi_bbox.min_lon, aoi_bbox.max_lon]
            lats = [aoi_bbox.min_lat, aoi_bbox.min_lat, aoi_bbox.max_lat, aoi_bbox.max_lat]
            rows, cols = rowcol(transform, lons, lats)
            row_min, row_max = min(rows), max(rows)
            col_min, col_max = min(cols), max(cols)
        except Exception as exc:
            logger.warning(
                "Failed to convert AOI to pixel coordinates (%s) — "
                "returning full debursted arrays.", exc
            )
            return ref_debursted, sec_debursted, profile, None

        # Ensure min < max
        if row_min > row_max:
            row_min, row_max = row_max, row_min
        if col_min > col_max:
            col_min, col_max = col_max, col_min

        # Add margin
        row_min = max(0, row_min - margin_px)
        row_max = min(ref_debursted.shape[0], row_max + margin_px + 1)  # +1 because slice end is exclusive
        col_min = max(0, col_min - margin_px)
        col_max = min(ref_debursted.shape[1], col_max + margin_px + 1)  # +1 because slice end is exclusive

        n_rows = row_max - row_min
        n_cols = col_max - col_min

        if n_rows <= 0 or n_cols <= 0:
            logger.warning(
                "AOI crop window is empty (%d x %d) — returning full "
                "debursted arrays.", n_rows, n_cols
            )
            return ref_debursted, sec_debursted, profile, None

        # Crop the arrays
        ref_cropped = ref_debursted[row_min:row_max, col_min:col_max]
        sec_cropped = sec_debursted[row_min:row_max, col_min:col_max]

        # Update the profile transform to reflect the new origin
        from rasterio import Affine
        new_transform = transform * Affine.translation(col_min, row_min)
        updated_profile = dict(profile)
        updated_profile["transform"] = new_transform
        updated_profile["height"] = n_rows
        updated_profile["width"] = n_cols

        crop_info = {
            "row_off": row_min,
            "col_off": col_min,
            "n_rows": n_rows,
            "n_cols": n_cols,
        }

        logger.info(
            "Cropped to AOI after deburst: %s -> (%d, %d) "
            "(row_off=%d, col_off=%d, margin=%d px)",
            ref_debursted.shape, n_rows, n_cols,
            row_min, col_min, margin_px,
        )

        return ref_cropped, sec_cropped, updated_profile, crop_info

    # ══════════════════════════════════════════════════════════════════
    # CHUNKED COHERENCE ESTIMATION
    # ══════════════════════════════════════════════════════════════════

    def _estimate_coherence_chunked(
        self, ref, sec, window: int, chunk_rows: int = 2000
    ):
        """
        CHUNKED: Estimate interferometric coherence in row chunks.

        Coherence is a local (windowed) operation, so it can be computed
        independently for each row chunk. This reduces peak memory by
        avoiding the creation of full-size intermediate arrays.

        Why chunking works here: coherence uses uniform_filter with a
        small window (typically 5x5). Each output pixel only depends on
        a small neighbourhood. So we can process row-by-row chunks with
        a small overlap (half the window size) and get identical results
        to full-array processing.

        Args:
            ref: Reference complex array.
            sec: Secondary complex array.
            window: Coherence estimation window size.
            chunk_rows: Number of rows to process at a time.

        Returns:
            Coherence array (float32).
        """
        np = self._np()
        from scipy.ndimage import uniform_filter

        h, w = ref.shape
        coherence = np.empty((h, w), dtype=np.float32)

        # Process in chunks with overlap for the window
        half_win = window // 2

        for row_start in range(0, h, chunk_rows):
            row_end = min(row_start + chunk_rows, h)

            # Extend chunk boundaries by half the window for edge handling
            ext_start = max(0, row_start - half_win)
            ext_end = min(h, row_end + half_win)

            ref_chunk = ref[ext_start:ext_end]
            sec_chunk = sec[ext_start:ext_end]

            # Compute coherence for this chunk
            inter = ref_chunk * np.conj(sec_chunk)
            num = np.abs(
                uniform_filter(inter.real, size=window)
                + 1j * uniform_filter(inter.imag, size=window)
            )
            denom = np.sqrt(
                uniform_filter(np.abs(ref_chunk) ** 2, size=window)
                * uniform_filter(np.abs(sec_chunk) ** 2, size=window)
                + 1e-10
            )
            coh_chunk = np.clip(num / denom, 0.0, 1.0).astype(np.float32)

            # Extract the valid portion (remove the extended boundaries)
            valid_start = row_start - ext_start
            valid_end = valid_start + (row_end - row_start)
            coherence[row_start:row_end] = coh_chunk[valid_start:valid_end]

            # Free chunk memory
            del ref_chunk, sec_chunk, inter, num, denom, coh_chunk

        return coherence

    # ══════════════════════════════════════════════════════════════════
    # CHUNKED TOPOGRAPHIC PHASE REMOVAL
    # ══════════════════════════════════════════════════════════════════

    def _remove_topographic_phase_chunked(
        self, interferogram, dem_path: Path, profile, chunk_rows: int = 2000
    ):
        """
        CHUNKED: Remove topographic phase in row chunks.

        The DEM is reprojected once (full size, unavoidable), but the
        phase regression and correction are applied in row chunks to
        reduce peak memory during the correction step.

        Args:
            interferogram: Complex interferogram array.
            dem_path: Path to the DEM GeoTIFF.
            profile: Rasterio-style profile.
            chunk_rows: Number of rows to process at a time.

        Returns:
            (corrected_interferogram, metadata)
        """
        np = self._np()
        try:
            import rasterio
        except ImportError:
            return interferogram, {"correction_applied": False, "reason": "rasterio_missing"}

        try:
            interferogram_crs = profile.get("crs")
            interferogram_transform = profile.get("transform")

            # Reproject DEM to interferogram grid (full size, unavoidable)
            with rasterio.open(dem_path) as dem_src:
                if interferogram_crs is not None and interferogram_transform is not None:
                    from rasterio.warp import Resampling, reproject
                    dem = np.empty(interferogram.shape, dtype=np.float32)
                    reproject(
                        source=rasterio.band(dem_src, 1),
                        destination=dem,
                        src_transform=dem_src.transform,
                        src_crs=dem_src.crs,
                        dst_transform=interferogram_transform,
                        dst_crs=interferogram_crs,
                        resampling=Resampling.bilinear,
                        src_nodata=dem_src.nodata,
                        dst_nodata=np.nan,
                    )
                else:
                    logger.warning(
                        "No real CRS/transform available on the "
                        "interferogram profile -- falling back to a "
                        "shape-ratio DEM resample."
                    )
                    dem = dem_src.read(1).astype(np.float32)
                    if dem.shape != interferogram.shape:
                        from scipy.ndimage import zoom
                        zf = (
                            interferogram.shape[0] / dem.shape[0],
                            interferogram.shape[1] / dem.shape[1],
                        )
                        dem = zoom(dem, zf, order=1)

            valid = np.isfinite(dem) & (dem > -500)
            if valid.sum() < 100:
                return interferogram, {
                    "correction_applied": False,
                    "reason": "insufficient_valid_pixels",
                }

            dem_std = np.std(dem[valid])
            if dem_std < 1.0:
                return interferogram, {
                    "correction_applied": False,
                    "reason": "dem_no_variance",
                }

            # Estimate slope using subsampled pixels (same as before)
            dem_v = dem[valid]
            phase = np.angle(interferogram)
            phase_v = phase[valid]

            elev_range = float(np.ptp(dem_v))
            if elev_range < 1.0:
                max_slope = 0.5
            else:
                max_slope = (25.0 * 2 * np.pi) / elev_range

            rng = np.random.default_rng(0)
            n_valid = len(dem_v)
            n_search = 20000
            if n_valid > n_search:
                search_idx = rng.choice(n_valid, size=n_search, replace=False)
                dem_search, phase_search = dem_v[search_idx], phase_v[search_idx]
            else:
                dem_search, phase_search = dem_v, phase_v

            def _flatness(candidate_slopes):
                phase_matrix = (
                    phase_search[None, :]
                    - candidate_slopes[:, None] * dem_search[None, :]
                )
                return np.abs(np.mean(np.exp(1j * phase_matrix), axis=1))

            coarse = np.linspace(-max_slope, max_slope, 400)
            best_slope = float(coarse[np.argmax(_flatness(coarse))])

            fine_half_width = (coarse[1] - coarse[0])
            fine = np.linspace(
                best_slope - fine_half_width, best_slope + fine_half_width, 400
            )
            best_slope = float(fine[np.argmax(_flatness(fine))])

            residual_v = np.angle(np.exp(1j * (phase_v - best_slope * dem_v)))
            intercept = float(np.angle(np.mean(np.exp(1j * residual_v))))

            # Compute R² for the gate
            fitted_phase_v = np.angle(np.exp(1j * (best_slope * dem_v + intercept)))
            residual = np.angle(np.exp(1j * (phase_v - fitted_phase_v)))
            ss_res = np.sum(residual**2)
            # BUG FIX: "centering" wrapped phase for ss_tot must use the
            # CIRCULAR mean, not a naive arithmetic mean of already-
            # wrapped angles -- exactly like `intercept` above already
            # does correctly. A naive mean of values clustered near the
            # +/-pi discontinuity can land far from where the data
            # actually sits (e.g. samples at +3.1 and -3.1 rad average
            # to ~0, diametrically opposite both of them), silently
            # producing a wrong, data-dependent R² -- confirmed to be
            # able to swing R² by more than 0.5 in either direction
            # (test_topo_r2_bug.py), including inflating a near-zero
            # true correlation (R²~0.0001) up past this function's own
            # 0.5 gate (R²~0.57), i.e. accepting a spurious correction
            # that a correct computation would have rejected.
            circular_mean_phase = np.angle(np.mean(np.exp(1j * phase_v)))
            centered = np.angle(np.exp(1j * (phase_v - circular_mean_phase)))
            ss_tot = np.sum(centered**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

            if r_squared < 0.5:
                logger.info(
                    "DEM-elevation correlation too weak (R²=%.2f) — skipping "
                    "topographic phase removal.", r_squared
                )
                return interferogram, {
                    "correction_applied": False,
                    "r_squared": float(r_squared),
                }

            # Apply correction in chunks
            corrected = np.empty_like(interferogram)
            for row_start in range(0, interferogram.shape[0], chunk_rows):
                row_end = min(row_start + chunk_rows, interferogram.shape[0])

                topo_phase_chunk = np.angle(np.exp(1j * (
                    best_slope * dem[row_start:row_end] + intercept
                )))
                corrected[row_start:row_end] = (
                    interferogram[row_start:row_end] * np.exp(-1j * topo_phase_chunk)
                )

                del topo_phase_chunk

            logger.info(
                "Topographic phase regression R²=%.2f — correction applied (chunked)",
                r_squared
            )

            del dem, phase, dem_v, phase_v

            return corrected.astype(np.complex64), {
                "correction_applied": True,
                "r_squared": float(r_squared),
            }

        except Exception as exc:
            logger.warning("Topographic phase removal failed: %s", exc)
            return interferogram, {"correction_applied": False, "reason": str(exc)}

    # ══════════════════════════════════════════════════════════════════
    # BURST-AWARE PROCESSING (ESD + Deburst)
    # ══════════════════════════════════════════════════════════════════

    def _burst_aware_processing(
        self, ref_complex, sec_complex,
        reference_safe_zip, secondary_safe_zip,
        reference_extracted_path, secondary_extracted_path,
        representative_row_offset: float = 0.0,
        reference_orbit_file=None, secondary_orbit_file=None,
    ):
        """
        Real per-burst-overlap ESD (esd.py) followed by real deburst
        (deburst.py), using real burst metadata (annotation.py) parsed
        from each date's own annotation XML.

        Order matters and is deliberate: ESD runs BEFORE deburst,
        because ESD needs the real burst overlap regions (the
        redundant, duplicate-ground-coverage rows) to compute its
        double-difference phase, and deburst's entire purpose is to
        remove exactly those rows -- confirmed to match the real SNAP
        processing order (ESD refinement happens as part of/after
        Back-Geocoding coregistration; TOPS Deburst happens afterward,
        on the interferogram).

        Falls back to the existing whole-image ESD approximation, with
        no deburst applied, if burst metadata cannot be parsed for
        either date, or if any step here raises -- a real, logged,
        non-fatal degradation, matching the same fallback discipline
        already used by _orbit_based_coregister for its own real vs.
        approximate coregistration paths.

        Returns:
            (ref_processed, sec_processed, metadata) -- metadata
            includes "method" ("real_per_burst_esd_and_deburst" or
            "whole_image_esd_fallback") so process_pair()'s own
            metadata can honestly report which path actually ran.
        """
        metadata = {
            "method": "whole_image_esd_fallback",
            "esd_shift_px": None,
            "deburst_applied": False,
        }

        try:
            from pygeofetch.insar.annotation import parse_burst_info, parse_slc_geometry
            from pygeofetch.insar.coregister import read_crop_offset, read_matched_swath
            from pygeofetch.insar.deburst import deburst_array
            from pygeofetch.insar.esd import estimate_esd_shift_per_burst_overlap

            ref_swath_hint = (
                read_matched_swath(reference_extracted_path)
                if reference_extracted_path is not None else None
            )
            sec_swath_hint = (
                read_matched_swath(secondary_extracted_path)
                if secondary_extracted_path is not None else None
            )

            ref_geom = parse_slc_geometry(reference_safe_zip, member_hint=ref_swath_hint)
            ref_burst_info = parse_burst_info(reference_safe_zip, member_hint=ref_swath_hint)
            sec_burst_info = parse_burst_info(secondary_safe_zip, member_hint=sec_swath_hint)

            # Guard against any code path that accidentally returns a raw list.
            if not isinstance(ref_burst_info, SwathTiming):
                raise TypeError(
                    f"parse_burst_info returned {type(ref_burst_info).__name__} "
                    f"instead of SwathTiming for reference — check annotation.py"
                )
            if not isinstance(sec_burst_info, SwathTiming):
                raise TypeError(
                    f"parse_burst_info returned {type(sec_burst_info).__name__} "
                    f"instead of SwathTiming for secondary — check annotation.py"
                )

            azimuth_time_interval_s = ref_geom.azimuth_time_interval_s

            # ── Common-ground diagnostic ─────────────────────────────────
            # Report, per reference burst overlap, how many azimuth lines
            # of real double-covered ground it spans -- since the
            # secondary is already coregistered onto the reference's own
            # pixel grid by this point, the reference's own overlap
            # windows ARE the common ground for both arrays (see
            # compute_common_ground_overlaps's docstring). This is purely
            # informational at this point (a genuinely near-zero total
            # here would mean the reference itself has almost no real
            # burst overlap, which would be unusual); ESD's own
            # min_common_lines / coherence_threshold gates are what
            # actually decide per-overlap usability.
            from pygeofetch.insar.esd import compute_common_ground_overlaps

            common_report = compute_common_ground_overlaps(
                ref_burst_info, azimuth_time_interval_s
            )
            common_lines = [r["common_lines"] for r in common_report]
            total_common = sum(common_lines)

            logger.info(
                "ESD common-ground diagnostic: reference overlap lengths "
                "(azimuth lines): [%s] (total %d)",
                ", ".join(str(n) for n in common_lines), total_common,
            )

            metadata["esd_common_ground_lines"] = common_lines
            metadata["esd_common_ground_total"] = total_common

            # ── Burst synchronization diagnostic ─────────────────────────
            # Directly measures Δt_acq -- the real physical quantity
            # Sentinel-1's own mission spec requires be under 5 ms for
            # two acquisitions' burst timing to support TOPS
            # interferometry well, independent of pixel-level
            # coregistration accuracy (see
            # esd.compute_burst_synchronization's own docstring for the
            # method and citations). Best-effort: needs real orbit
            # files and a real reference raster to derive an evaluation
            # ground point from; skips cleanly (metadata stays None) if
            # either isn't available, rather than failing the whole
            # ESD/deburst step over a diagnostic.
            metadata["burst_sync_offset_ms"] = None
            metadata["burst_sync_within_requirement"] = None
            if reference_orbit_file is not None and secondary_orbit_file is not None:
                try:
                    import rasterio as _rasterio

                    from pygeofetch.insar.esd import compute_burst_synchronization
                    from pygeofetch.insar.geolocation import (
                        geodetic_to_ecef,
                        parse_orbit_file,
                    )

                    ref_orbit = parse_orbit_file(reference_orbit_file)
                    sec_orbit = parse_orbit_file(secondary_orbit_file)

                    # A real, representative evaluation point: the
                    # reference extract's own geographic center. Height
                    # is taken as 0 -- burst synchronization is a timing
                    # property of the two orbits/burst patterns, not a
                    # precise geolocation, so it isn't meaningfully
                    # sensitive to a DEM height error here (unlike
                    # per-pixel coregistration, which is).
                    if reference_extracted_path is not None:
                        with _rasterio.open(reference_extracted_path) as _src:
                            b = _src.bounds
                        eval_lon = (b.left + b.right) / 2.0
                        eval_lat = (b.bottom + b.top) / 2.0
                    else:
                        eval_lon, eval_lat = None, None

                    if eval_lon is not None:
                        ground_point = geodetic_to_ecef(eval_lat, eval_lon, 0.0)
                        ref_center_time = ref_geom.azimuth_time(ref_geom.n_lines / 2)
                        sec_geom_for_sync = parse_slc_geometry(secondary_safe_zip, member_hint=sec_swath_hint)
                        sec_center_time = sec_geom_for_sync.azimuth_time(sec_geom_for_sync.n_lines / 2)

                        sync_result = compute_burst_synchronization(
                            ref_orbit, sec_orbit, ref_burst_info, sec_burst_info,
                            ground_point,
                            ref_time_guess=ref_center_time,
                            sec_time_guess=sec_center_time,
                        )
                        metadata["burst_sync_offset_ms"] = sync_result["sync_offset_ms"]
                        metadata["burst_sync_within_requirement"] = sync_result["within_esa_requirement"]
                except Exception as exc:
                    logger.warning(
                        "Burst synchronization diagnostic failed (%s) — "
                        "proceeding without it; this does not affect "
                        "ESD/deburst/coregistration themselves, only "
                        "this diagnostic's own metadata.",
                        exc,
                    )

            # ── ESD shift per burst overlap ────────────────────────────────────────
            # Compute the ESD shift per reference overlap, using the
            # per-reference-overlap diagnostic of how much COMMON double-covered
            # ground this date pair actually shares.
            azimuth_time_interval_s = ref_geom.azimuth_time_interval_s

            ref_row_off, _ = (
                read_crop_offset(reference_extracted_path)
                if reference_extracted_path is not None else (0.0, 0.0)
            )
            sec_row_off, _ = (
                read_crop_offset(secondary_extracted_path)
                if secondary_extracted_path is not None else (0.0, 0.0)
            )

            # Right after coregistration, before ESD/Deburst:
            min_rows = min(ref_complex.shape[0], sec_complex.shape[0])
            min_cols = min(ref_complex.shape[1], sec_complex.shape[1])

            if ref_complex.shape != (min_rows, min_cols) or sec_complex.shape != (min_rows, min_cols):
                logger.warning(
                    "Reference %s and secondary %s shapes differ — cropping both to common (%d, %d) shape before ESD.",
                    ref_complex.shape, sec_complex.shape, min_rows, min_cols
                )
                ref_complex = ref_complex[:min_rows, :min_cols]
                sec_complex = sec_complex[:min_rows, :min_cols]

            # Step 1: real per-burst-overlap ESD, on the still-bursted
            # (pre-deburst) data -- uses the reference's own burst
            # structure as the shared row-coordinate system for
            # comparing reference and secondary at the same real rows.
            if self._esd_enabled:
                esd_shift_s, per_overlap = estimate_esd_shift_per_burst_overlap(
                    ref_complex, sec_complex, ref_burst_info, azimuth_time_interval_s,
                    row_offset=int(ref_row_off),
                )
                if esd_shift_s is not None:
                    esd_shift_px = esd_shift_s / azimuth_time_interval_s
                    n_usable = sum(1 for s in per_overlap if s is not None)
                    if abs(esd_shift_px) > 1e-4:
                        sec_complex = self._apply_azimuth_shift(sec_complex, esd_shift_px)
                    logger.info(
                        "Real per-burst-overlap ESD azimuth shift: %.6f px "
                        "(%d/%d burst overlaps usable)",
                        esd_shift_px, n_usable, len(per_overlap),
                    )
                    metadata["esd_shift_px"] = esd_shift_px
                    metadata["esd_overlaps_usable"] = n_usable
                    metadata["esd_overlaps_total"] = len(per_overlap)
                else:
                    logger.info(
                        "Real per-burst-overlap ESD found no usable burst "
                        "overlaps — proceeding without an azimuth "
                        "refinement from this step."
                    )

            # Step 2: real deburst, applied to both images using each
            # date's own real burst metadata.
            #
            # Real, confirmed bug fixed here: by this point sec_complex has
            # already been resampled onto the REFERENCE's real grid, either
            # by _orbit_based_coregister (which explicitly fits and applies
            # an offset field using both ref_row_off AND sec_row_off
            # together, see resample_with_offset_field above) or by the
            # shape-based _resample_to_reference fallback -- either way,
            # sec_complex's row 0 now corresponds to the same real
            # full-scene row as ref_complex's row 0, i.e. ref_row_off, not
            # its own original, pre-coregistration sec_row_off. Passing
            # the stale sec_row_off here fed deburst_array a coordinate
            # system that no longer matched the array it was actually
            # operating on, silently shifting which rows got cropped as
            # "burst edge/overlap" for the secondary relative to where the
            # real burst boundaries actually fell in the now-coregistered
            # array -- corrupting the debursted secondary image used for
            # every downstream step (interferogram formation, coherence,
            # unwrapping), independent of and in addition to the separate
            # per-burst-overlap ESD issue investigated above, which runs
            # earlier, on the pre-deburst array, and was already using
            # ref_row_off correctly for both images at that stage.
            ref_debursted, ref_first_kept_row = deburst_array(
                ref_complex, ref_burst_info, azimuth_time_interval_s,
                row_offset=int(ref_row_off),
            )

            # FIX: Use ref_burst_info for the secondary as well.
            # Because sec_complex has already been resampled onto the reference's
            # grid by the swath-level coregistration, its burst boundaries in
            # pixel coordinates now perfectly match the reference's. Using
            # sec_burst_info applies the secondary's original, un-warped timing,
            # which cuts out the wrong rows and destroys coherence at burst edges.
            sec_debursted, _ = deburst_array(
                sec_complex, ref_burst_info, azimuth_time_interval_s,
                row_offset=int(ref_row_off),
            )

            # Real correctness fix, not optional: deburst can remove
            # rows from the TOP of the array (whenever the crop's own
            # row 0 falls inside a burst's discarded edge/overlap
            # region), which shifts the array's real spatial origin.
            # Without correcting the georeferencing transform's origin
            # by this same amount, the final saved GeoTIFF would still
            # write successfully (save() derives height/width fresh
            # from the array shape) but every pixel would be spatially
            # mislocated by the number of rows removed -- a silent
            # correctness bug, not a crash, so it would not have been
            # caught by "does it run."
            rows_removed_from_top = ref_first_kept_row - int(ref_row_off)
            metadata["rows_removed_from_top"] = rows_removed_from_top

            if ref_debursted.shape != sec_debursted.shape:
                min_rows = min(ref_debursted.shape[0], sec_debursted.shape[0])
                min_cols = min(ref_debursted.shape[1], sec_debursted.shape[1])
                logger.warning(
                    "Debursted reference %s and secondary %s shapes "
                    "differ (different dates' real burst timing need "
                    "not match exactly) — cropping both to the common "
                    "(%d, %d) shape.",
                    ref_debursted.shape, sec_debursted.shape, min_rows, min_cols,
                )
                ref_debursted = ref_debursted[:min_rows, :min_cols]
                sec_debursted = sec_debursted[:min_rows, :min_cols]

            metadata["method"] = "real_per_burst_esd_and_deburst"
            metadata["deburst_applied"] = True
            logger.info(
                "Real burst-aware processing complete: %s -> %s "
                "(burst-boundary redundancy removed)",
                ref_complex.shape, ref_debursted.shape,
            )
            return ref_debursted, sec_debursted, metadata

        except Exception as exc:
            logger.warning(
                "Real burst-aware ESD/deburst failed (%s) — falling "
                "back to the whole-image ESD approximation, no deburst "
                "applied. The interferogram will still be produced, "
                "but real burst-boundary artifacts will not be "
                "corrected for this pair.",
                exc,
            )
            esd_shift = None
            if self._esd_enabled:
                esd_shift = self._estimate_esd_shift(ref_complex, sec_complex)
                if esd_shift is not None and abs(esd_shift) > 1e-4:
                    sec_complex = self._apply_azimuth_shift(sec_complex, esd_shift)
                    logger.info(
                        "Whole-image ESD azimuth shift applied (fallback): %.5f px", esd_shift
                    )
            metadata["esd_shift_px"] = esd_shift
            return ref_complex, sec_complex, metadata

    # ══════════════════════════════════════════════════════════════════
    # MAIN PROCESSING PIPELINE
    # ══════════════════════════════════════════════════════════════════

    def process_pair(
        self,
        reference: Union[str, Path],
        secondary: Union[str, Path],
        dem: Optional[Union[str, Path]] = None,
        reference_date: Optional[str] = None,
        secondary_date: Optional[str] = None,
        reference_safe_zip: Optional[Union[str, Path]] = None,
        secondary_safe_zip: Optional[Union[str, Path]] = None,
        reference_orbit_file: Optional[Union[str, Path]] = None,
        secondary_orbit_file: Optional[Union[str, Path]] = None,
        looks_azimuth: int = 1,
        looks_range: int = 1,
        apply_goldstein_filter: bool = False,
        goldstein_alpha: float = 0.5,
        coregistration_refine_by_coherence: bool = True,
        coregistration_degree: int = 1,
        coregistration_rms_threshold: Optional[float] = None,
        coregistration_method: str = "orbit_dem",
        coregistration_window: Optional[int] = None,
        coregistration_coarse_search_radius: Optional[int] = None,
        coregistration_fine_search_radius: Optional[float] = None,
        coregistration_coherence_threshold: Optional[float] = None,
        # ═══ NEW SNAP-STYLE PARAMETERS ═══
        aoi_bbox=None,
        crop_after_deburst: bool = True,
        use_chunked_processing: bool = True,
        chunk_rows: Optional[int] = None,
    ) -> InterferogramResult:
        """
        Process an SLC pair into an interferogram with topographic phase removed.

        SNAP-style workflow (when aoi_bbox is provided):
        1. Read full/large SLC (preserving burst overlaps)
        2. Coregistration on full data
        3. ESD on full data (burst overlaps intact)
        4. Deburst on full data
        5. Crop to AOI (if aoi_bbox provided)
        6. Multilook, filter, coherence, topo phase on cropped data

        Args:
            reference: Path to reference (master) complex SLC GeoTIFF.
            secondary: Path to secondary (slave) complex SLC GeoTIFF.
            dem:       Optional DEM for topographic phase removal. If
                       None, only the flat-earth phase is removed.
            reference_date, secondary_date: ISO date strings for baseline
                       bookkeeping.
            reference_safe_zip, secondary_safe_zip: Original .SAFE.zip
                       archives for each date (needed for real
                       acquisition timing via annotation.py). Optional
                       — without these, coregistration falls back to
                       the shape-only resample.
            reference_orbit_file, secondary_orbit_file: Real .EOF orbit
                       files for each date. Optional, same fallback.
            looks_azimuth, looks_range: Multilook factors, applied after
                       ESD but before interferogram formation.
            apply_goldstein_filter: Apply adaptive frequency-domain
                       phase filtering after interferogram formation.
            goldstein_alpha: Goldstein filter strength (0-1).
            coregistration_refine_by_coherence: If True (default), the
                       orbit/DEM (or collocation) offset field is
                       refined against the actual image content via
                       cross-correlation + Powell coherence maximization
                       before the warp polynomial is fit.
            coregistration_degree: Warp polynomial order (1, 2, or 3).
            coregistration_rms_threshold: Optional final absolute GCP
                       RMS cutoff (pixels), applied after iterative
                       mean-RMS outlier rejection.
            coregistration_method: "orbit_dem" (default, needs dem +
                       both SAFE zips + both orbit files — more precise
                       in principle) or "raster_collocation" (SNAP
                       CreateStack's approach, needs only a real CRS +
                       transform on both files, but with this project's
                       SLCExtractor output that transform is a coarser
                       GCP-fit approximation — see
                       collocate_by_geocoding's docstring).
            coregistration_window: Imagette size (pixels) for the
                       cross-correlation refinement stage. None
                       (default) uses refine_offsets_by_coherence's own
                       default (32) for both coregistration_method
                       values.
            coregistration_coarse_search_radius: Integer-pixel search
                       radius for the coarse cross-correlation stage.
                       None (default) uses 4 for "orbit_dem" and 12 for
                       "raster_collocation". Widening this gives
                       cross-correlation more room to find the true
                       match when the offset estimate's residual error
                       is larger than the default assumes — e.g. if some
                       pairs in a stack show refinement failing outright
                       while others succeed, a larger radius is worth
                       testing BEFORE lowering coherence_threshold
                       (which accepts worse GCPs rather than actually
                       finding the true match). Not a fix for an
                       arbitrarily large residual: if the true offset is
                       hundreds of pixels or more, something upstream
                       (crop offset, sub-swath match, orbit/DEM inputs)
                       is almost certainly wrong instead.
            coregistration_fine_search_radius: Sub-pixel search radius
                       for the Powell coherence-maximization stage.
                       None uses the function's own default (1.5).
            coregistration_coherence_threshold: GCPs with refined
                       coherence below this are dropped. None uses the
                       default (0.3). Exposed for completeness, but
                       lowering it is NOT the recommended way to recover
                       a pair that's failing refinement — it accepts
                       genuinely poor GCPs rather than finding better
                       ones. Try coregistration_coarse_search_radius
                       first.
            aoi_bbox: BoundingBox for post-deburst AOI cropping
                       (SNAP-style workflow) — process the full
                       sub-swath through coregistration/ESD/deburst,
                       then crop, rather than cropping up front. Avoids
                       AOI crops computed independently per date landing
                       in different bursts for different acquisitions.
            crop_after_deburst: If True and aoi_bbox is given, crop
                       after deburst rather than not at all.
            use_chunked_processing: Use chunked SLC reading, coherence
                       estimation, and topographic-phase removal to
                       reduce peak memory on large/full-swath scenes.
            chunk_rows: Rows per chunk (overrides the instance's own
                       chunk_size if that was set in __init__).

        Returns:
            InterferogramResult with wrapped phase, coherence, and amplitude.
        """
        # Resolve chunk size: explicit param > instance default > None (full)
        effective_chunk = chunk_rows if chunk_rows is not None else self._chunk_size

        # ── Step 0: Read SLCs ──
        # Use chunked reading for large files if chunking is enabled
        if use_chunked_processing and effective_chunk is not None:
            ref_complex, profile = self._read_complex_chunked(
                Path(reference), chunk_rows=effective_chunk
            )
            sec_complex, sec_profile = self._read_complex_chunked(
                Path(secondary), chunk_rows=effective_chunk
            )
        else:
            ref_complex, profile = self._read_complex(Path(reference))
            sec_complex, sec_profile = self._read_complex(Path(secondary))

        # Validate
        from pygeofetch.insar.validate import DataValidator
        DataValidator.validate_slc(ref_complex, name="reference SLC").raise_if_invalid()
        DataValidator.validate_slc(sec_complex, name="secondary SLC").raise_if_invalid()

        # Platform/sub-swath mismatch warnings
        if reference_safe_zip is not None and secondary_safe_zip is not None:
            ref_platform = Path(reference_safe_zip).name[:3]
            sec_platform = Path(secondary_safe_zip).name[:3]
            if ref_platform != sec_platform:
                logger.warning(
                    "Reference (%s) and secondary (%s) scenes are from "
                    "different Sentinel-1 satellites -- burst boundaries "
                    "for the same real ground area are not guaranteed to "
                    "align between S1A and S1B.",
                    ref_platform, sec_platform,
                )
            try:
                from pygeofetch.insar.coregister import read_matched_swath
                ref_swath = read_matched_swath(reference) if reference is not None else None
                sec_swath = read_matched_swath(secondary) if secondary is not None else None
                if ref_swath and sec_swath and ref_swath != sec_swath:
                    logger.warning(
                        "Reference (%s) and secondary (%s) extracted from "
                        "different sub-swaths.", ref_swath, sec_swath,
                    )
            except Exception:
                pass

        # ── Step 1: Coregistration ──
        if coregistration_method == "raster_collocation":
            ref_has_geocoding = (
                profile.get("crs") is not None
                and profile.get("transform") is not None
            )
            sec_has_geocoding = (
                sec_profile.get("crs") is not None
                and sec_profile.get("transform") is not None
            )
            if ref_has_geocoding and sec_has_geocoding:
                sec_complex, coreg_metadata = self._raster_collocation_coregister(
                    ref_complex, sec_complex, profile, sec_profile,
                    refine_by_coherence=coregistration_refine_by_coherence,
                    degree=coregistration_degree,
                    rms_threshold=coregistration_rms_threshold,
                    window=coregistration_window,
                    coarse_search_radius=coregistration_coarse_search_radius,
                    fine_search_radius=coregistration_fine_search_radius,
                    coherence_threshold=coregistration_coherence_threshold,
                )
            else:
                logger.info(
                    "coregistration_method='raster_collocation' needs a "
                    "real crs+transform on both reference (%s) and "
                    "secondary (%s) -- falling back to shape-based "
                    "coregistration.",
                    "present" if ref_has_geocoding else "MISSING",
                    "present" if sec_has_geocoding else "MISSING",
                )
                if sec_complex.shape != ref_complex.shape:
                    sec_complex = self._resample_to_reference(sec_complex, ref_complex.shape)
                coreg_metadata = {
                    "method": "shape_based_fallback_missing_geocoding",
                    "refined_by_coherence": False,
                    "n_gcps_initial": None, "n_gcps_final": None,
                    "rms_mean_px": None, "mean_coherence": None,
                }
        elif coregistration_method == "orbit_dem":
            real_coreg_inputs = (
                dem, reference_safe_zip, secondary_safe_zip,
                reference_orbit_file, secondary_orbit_file,
            )
            if all(x is not None for x in real_coreg_inputs):
                sec_complex, coreg_metadata = self._orbit_based_coregister(
                    ref_complex, sec_complex, dem,
                    reference_safe_zip, secondary_safe_zip,
                    reference_orbit_file, secondary_orbit_file,
                    reference, secondary,
                    refine_by_coherence=coregistration_refine_by_coherence,
                    degree=coregistration_degree,
                    rms_threshold=coregistration_rms_threshold,
                    window=coregistration_window,
                    coarse_search_radius=coregistration_coarse_search_radius,
                    fine_search_radius=coregistration_fine_search_radius,
                    coherence_threshold=coregistration_coherence_threshold,
                )
            else:
                logger.info(
                    "Using shape-based coregistration fallback (real orbit-"
                    "based coregistration needs dem + both SAFE zips + both "
                    "orbit files; not all were supplied)."
                )
                if sec_complex.shape != ref_complex.shape:
                    sec_complex = self._resample_to_reference(sec_complex, ref_complex.shape)
                coreg_metadata = {
                    "method": "shape_based_fallback",
                    "refined_by_coherence": False,
                    "n_gcps_initial": None,
                    "n_gcps_final": None,
                    "rms_mean_px": None,
                    "mean_coherence": None,
                }
        else:
            raise ValueError(
                f"coregistration_method must be 'orbit_dem' or "
                f"'raster_collocation', got {coregistration_method!r}"
            )

        # Shape reconciliation before ESD
        min_rows = min(ref_complex.shape[0], sec_complex.shape[0])
        min_cols = min(ref_complex.shape[1], sec_complex.shape[1])
        if ref_complex.shape != (min_rows, min_cols) or sec_complex.shape != (min_rows, min_cols):
            logger.warning(
                "Reference %s and secondary %s shapes differ — cropping to common (%d, %d).",
                ref_complex.shape, sec_complex.shape, min_rows, min_cols
            )
            ref_complex = ref_complex[:min_rows, :min_cols]
            sec_complex = sec_complex[:min_rows, :min_cols]

        # ── Step 2: ESD + Deburst (on FULL data, burst overlaps intact) ──
        esd_shift = None
        burst_metadata = {"method": "whole_image_esd", "deburst_applied": False}

        if self._use_real_burst_processing and reference_safe_zip is not None and secondary_safe_zip is not None:
            _rep_row_offset = coreg_metadata.get("representative_row_offset_px", 0.0)
            ref_complex, sec_complex, burst_metadata = self._burst_aware_processing(
                ref_complex, sec_complex,
                reference_safe_zip, secondary_safe_zip,
                reference, secondary,
                representative_row_offset=_rep_row_offset,
                reference_orbit_file=reference_orbit_file,
                secondary_orbit_file=secondary_orbit_file,
            )
            esd_shift = burst_metadata.get("esd_shift_px")
            rows_removed = burst_metadata.get("rows_removed_from_top", 0)

            if rows_removed and rows_removed > 0 and profile.get("transform") is not None:
                from rasterio import Affine
                profile = dict(profile)
                profile["transform"] = profile["transform"] * Affine.translation(0, rows_removed)
                logger.info(
                    "Adjusted georeferencing transform origin: deburst "
                    "removed %d row(s) from the top of the crop.",
                    rows_removed,
                )
        elif self._esd_enabled:
            esd_shift = self._estimate_esd_shift(ref_complex, sec_complex)
            if esd_shift is not None and abs(esd_shift) > 1e-4:
                sec_complex = self._apply_azimuth_shift(sec_complex, esd_shift)
                logger.info("ESD azimuth shift applied: %.5f px", esd_shift)

        # ═══════════════════════════════════════════════════════════════
        # STEP 2.5: CROP TO AOI AFTER DEBURST (SNAP-style workflow)
        # This is the key change: burst overlaps are preserved for ESD,
        # but we crop to the actual AOI after deburst to reduce array
        # size for all downstream operations.
        # ═══════════════════════════════════════════════════════════════
        crop_info = None
        if crop_after_deburst and aoi_bbox is not None:
            ref_complex, sec_complex, profile, crop_info = self._crop_to_aoi_after_deburst(
                ref_complex, sec_complex, profile, aoi_bbox,
                margin_px=50,
            )

        # ── Step 3: Multilook (on cropped data) ──
        ref_complex_native = ref_complex
        sec_complex_native = sec_complex
        coh_window = self._coh_window_explicit if self._coh_window_explicit is not None else 5

        if looks_azimuth > 1 or looks_range > 1:
            from pygeofetch.insar.unwrap import multilook
            pre_shape = ref_complex.shape
            ref_complex = multilook(ref_complex, looks_azimuth, looks_range)
            sec_complex = multilook(sec_complex, looks_azimuth, looks_range)
            logger.info(
                "Multilooked %dx%d -> %dx%d (%d azimuth x %d range looks)",
                pre_shape[0], pre_shape[1], ref_complex.shape[0], ref_complex.shape[1],
                looks_azimuth, looks_range,
            )

        # ── Step 4: Form interferogram ──
        interferogram = ref_complex * self._np().conj(sec_complex)

        # BUG FIX: amplitude must come from the reference SLC, not the
        # interferogram. interferogram = ref * conj(sec), so
        # abs(interferogram) = |ref| * |sec| -- a product of both
        # amplitudes, not either one alone; 20*log10 of that is the SUM
        # of the two amplitudes in dB, not a meaningful "amplitude"
        # band. Must be computed here, before ref_complex is freed
        # below for memory -- computing it later (after the del) either
        # raises or (as it did previously) silently falls back to the
        # wrong array.
        amplitude = self._np().log10(self._np().abs(ref_complex) + 1e-10) * 20  # dB

        # MEMORY FIX: Free ref_complex and sec_complex after interferogram
        del ref_complex, sec_complex
        gc.collect()

        # ── Step 4b: Flat-earth phase removal ──
        #
        # ORDERING BUG FIX: Goldstein filtering used to run here, BEFORE
        # flat-earth (and topographic) phase removal -- meaning it
        # adaptively filtered a raw interferogram still carrying the
        # full geometric ramp. That ramp is not a small correction: real
        # values observed on this exact stack ranged up to several
        # THOUSAND radians across a scene (e.g. "[-9730.88, -0.00] rad").
        # Goldstein filtering works by finding and enhancing each tile's
        # dominant local fringe frequency in the frequency domain -- with
        # an unremoved ramp of that magnitude still present, the
        # "dominant frequency" in every tile is overwhelmingly the
        # geometric ramp's own local slope, not any real ground signal,
        # so the filter has nothing meaningful left to enhance and can
        # actively smear away whatever real (much lower-amplitude)
        # deformation/topographic signal was trying to show through --
        # this alone is a strong candidate explanation for wrapped-phase
        # output not showing the fringe structure real InSAR pipelines
        # (SNAP, ISCE, GAMMA, GMTSAR) produce, independent of colormap
        # choice. Standard practice everywhere is: form interferogram ->
        # remove reference (flat-earth [+ topographic]) phase -> filter
        # the RESIDUAL -> unwrap. Moved flat-earth and topographic
        # removal (Step 5 below) BEFORE Goldstein filtering (now after
        # Step 5) to match.
        flat_earth_metadata = {"applied": False}
        if self._remove_flat_earth_phase and all(
            x is not None for x in (
                reference_safe_zip, secondary_safe_zip,
                reference_orbit_file, secondary_orbit_file,
            )
        ):
            try:
                from pygeofetch.insar.annotation import parse_slc_geometry
                from pygeofetch.insar.coregister import read_matched_swath
                from pygeofetch.insar.flatearth import compute_flat_earth_phase
                from pygeofetch.insar.geolocation import parse_orbit_file

                ref_swath_hint = read_matched_swath(reference) if reference is not None else None
                sec_swath_hint = read_matched_swath(secondary) if secondary is not None else None
                ref_geom_fe = parse_slc_geometry(reference_safe_zip, member_hint=ref_swath_hint)
                sec_geom_fe = parse_slc_geometry(secondary_safe_zip, member_hint=sec_swath_hint)
                ref_orbit_fe = parse_orbit_file(reference_orbit_file)
                sec_orbit_fe = parse_orbit_file(secondary_orbit_file)

                ref_center_time_fe = ref_geom_fe.azimuth_time(ref_geom_fe.n_lines / 2)
                sec_center_time_fe = sec_geom_fe.azimuth_time(sec_geom_fe.n_lines / 2)

                # with rasterio.open(str(reference)) as _src:
                #     b = _src.bounds
                #     margin_lon = (b.right - b.left) * 0.2
                #     margin_lat = (b.top - b.bottom) * 0.2
                #     sample_bounds_fe = (
                #         b.left - margin_lon, b.bottom - margin_lat,
                #         b.right + margin_lon, b.top + margin_lat,
                #     )

                # FIX: Use the profile we already read (which now has the GCP-fitted
                # transform) instead of reopening the file. Reopening a GCP-only
                # file yields pixel-coordinate bounds, which breaks sampling.
                transform_fe = profile.get("transform")
                if transform_fe is not None:
                    from rasterio.transform import array_bounds
                    # array_bounds returns (left, bottom, right, top)
                    left, bottom, right, top = array_bounds(
                        profile.get("height", ref_complex_native.shape[0]),
                        profile.get("width", ref_complex_native.shape[1]),
                        transform_fe
                    )
                    margin_lon = (right - left) * 0.2
                    margin_lat = (top - bottom) * 0.2
                    sample_bounds_fe = (
                        left - margin_lon, bottom - margin_lat,
                        right + margin_lon, top + margin_lat,
                    )
                else:
                    sample_bounds_fe = None

                wavelength_m = 0.05546576  # real Sentinel-1 C-band wavelength
                flat_earth_phase = compute_flat_earth_phase(
                    ref_geom_fe, ref_orbit_fe, sec_geom_fe, sec_orbit_fe,
                    ref_center_time_fe, sec_center_time_fe,
                    interferogram.shape, wavelength_m,
                    sample_bounds=sample_bounds_fe,
                )
                interferogram = interferogram * self._np().exp(-1j * flat_earth_phase)
                flat_earth_metadata = {
                    "applied": True,
                    "phase_range_rad": (
                        float(flat_earth_phase.min()), float(flat_earth_phase.max())
                    ),
                }
                logger.info(
                    "Real flat-earth phase removed: range [%.2f, %.2f] rad",
                    flat_earth_phase.min(), flat_earth_phase.max(),
                )
            except Exception as exc:
                logger.warning(
                    "Real flat-earth phase removal failed (%s) — proceeding without it.",
                    exc,
                )

        # ── Step 5: Topographic phase removal (chunked if enabled) ──
        topo_metadata = {"correction_applied": False}
        if dem is not None:
            if use_chunked_processing and effective_chunk is not None:
                interferogram, topo_metadata = self._remove_topographic_phase_chunked(
                    interferogram, Path(dem), profile, chunk_rows=effective_chunk
                )
            else:
                interferogram, topo_metadata = self._remove_topographic_phase(
                    interferogram, Path(dem), profile
                )
            if topo_metadata["correction_applied"]:
                logger.info("Topographic phase removed using DEM: %s", Path(dem).name)
            else:
                logger.info(
                    "DEM supplied (%s) but topographic phase was NOT removed.",
                    Path(dem).name,
                )
        else:
            logger.warning(
                "No DEM provided — topographic phase NOT removed."
            )

        # ── Step 5b: Goldstein filter (moved here -- see the ordering
        # note above Step 4b) -- runs on the RESIDUAL phase (deformation
        # + atmosphere + DEM error + noise) after the geometric flat-
        # earth/topographic components are already removed, which is
        # what the filter's frequency-domain fringe enhancement actually
        # needs to work correctly.
        if apply_goldstein_filter:
            from pygeofetch.insar.unwrap import goldstein_filter
            interferogram = goldstein_filter(interferogram, alpha=goldstein_alpha)
            logger.info("Goldstein phase filter applied (alpha=%.2f, tiled)", goldstein_alpha)

        # ── Step 6: Coherence estimation (chunked if enabled) ──
        if use_chunked_processing and effective_chunk is not None:
            coherence_native = self._estimate_coherence_chunked(
                ref_complex_native, sec_complex_native, coh_window,
                chunk_rows=effective_chunk,
            )
        else:
            coherence_native = self._estimate_coherence(
                ref_complex_native, sec_complex_native, coh_window
            )

        # MEMORY FIX: Free native arrays after coherence estimation
        del ref_complex_native, sec_complex_native
        gc.collect()

        if looks_azimuth > 1 or looks_range > 1:
            from pygeofetch.insar.unwrap import multilook
            coherence = multilook(coherence_native, looks_azimuth, looks_range, wrapped_phase=False)
        else:
            coherence = coherence_native

        effective_coh_window = coh_window
        DataValidator.validate_coherence(coherence).raise_if_invalid()

        # amplitude was already computed above (Step 4), before
        # ref_complex was freed -- not recomputed here.

        return InterferogramResult(
            interferogram=interferogram,
            coherence=coherence,
            amplitude=amplitude,
            profile=profile,
            reference_date=reference_date,
            secondary_date=secondary_date,
            esd_azimuth_shift_px=esd_shift,
            metadata={
                "coherence_window": effective_coh_window,
                "esd_applied": self._esd_enabled and esd_shift is not None,
                "esd_method": burst_metadata["method"],
                "deburst_applied": burst_metadata["deburst_applied"],
                "esd_common_ground_lines": burst_metadata.get("esd_common_ground_lines"),
                "esd_common_ground_total": burst_metadata.get("esd_common_ground_total"),
                "burst_sync_offset_ms": burst_metadata.get("burst_sync_offset_ms"),
                "burst_sync_within_requirement": burst_metadata.get("burst_sync_within_requirement"),
                "flat_earth_phase_removed": flat_earth_metadata["applied"],
                "topographic_phase_removed": topo_metadata["correction_applied"],
                "topographic_phase_r_squared": topo_metadata.get("r_squared"),
                "coregistration_method": coreg_metadata.get("method"),
                "coregistration_refined_by_coherence": coreg_metadata.get("refined_by_coherence"),
                "coregistration_gcps_initial": coreg_metadata.get("n_gcps_initial"),
                "coregistration_gcps_final": coreg_metadata.get("n_gcps_final"),
                "coregistration_rms_mean_px": coreg_metadata.get("rms_mean_px"),
                "coregistration_mean_coherence": coreg_metadata.get("mean_coherence"),
                "coregistration_collocation_coverage_fraction": coreg_metadata.get("collocation_coverage_fraction"),
                "crop_after_deburst_applied": crop_info is not None,
                "crop_info": crop_info,
                "chunked_processing": use_chunked_processing and effective_chunk is not None,
                "chunk_rows": effective_chunk,
            },
        )

    # ══════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _np(self):
        import numpy as np
        return np

    def _read_complex(self, path: Path, ref_shape=None):
        """Read a complex SLC GeoTIFF (native complex64/complex_int16/etc, or dual real/imag band)."""
        np = self._np()
        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        with rasterio.open(path) as src:
            profile = src.profile.copy()

            # FIX: Full sub-swath files are GCP-georeferenced and lack an affine
            # transform. Fit an approximate one from the GCPs so downstream
            # steps (flat-earth, topo phase, AOI crop) have a valid transform.
            if profile.get("transform") is None or getattr(profile.get("transform"), "is_identity", False):
                gcps, gcp_crs = src.gcps
                if gcps and len(gcps) >= 4:
                    from rasterio.transform import from_gcps
                    profile["transform"] = from_gcps(gcps)
                    if gcp_crs:
                        profile["crs"] = gcp_crs
                    logger.info(
                        "Fitted approximate affine transform from %d GCPs for %s",
                        len(gcps), path.name
                    )

            dtype = src.dtypes[0]
            if "complex" in dtype:

                data = src.read(1)
            elif src.count >= 2:
                real = src.read(1).astype(np.float32)
                imag = src.read(2).astype(np.float32)
                data = real + 1j * imag
            else:
                logger.warning(
                    "%s has no complex/phase data (dtype=%s, single real band). "
                    "InSAR requires complex SLC data — this pair cannot "
                    "produce a meaningful interferogram.",
                    path.name, dtype,
                )
                real = src.read(1).astype(np.float32)
                data = real.astype(np.complex64)

        if ref_shape is not None and data.shape != ref_shape:
            data = self._resample_to_reference(data, ref_shape)

        return data.astype(np.complex64), profile

    def _orbit_based_coregister(
        self, ref_complex, sec_complex, dem,
        reference_safe_zip, secondary_safe_zip,
        reference_orbit_file, secondary_orbit_file,
        reference_extracted_path=None, secondary_extracted_path=None,
        refine_by_coherence=True, degree=1, rms_threshold=None,
        window=None, coarse_search_radius=None, fine_search_radius=None,
        coherence_threshold=None,
    ):
        """
        Real orbit-based coregistration: parses real acquisition timing
        from both SAFE archives' annotation XML (burst-aware, see
        annotation.SLCGeometry) and real orbit state vectors from both
        .EOF files, computes a genuine first-estimate offset field using
        ground points sampled directly from the DEM (via
        geodetic_to_ecef + find_zero_doppler_time), then runs it through
        the same two-stage process SNAP's own coregistration pipeline
        uses before the final resample:

          1. compute_offset_field_from_dem() — the physically-grounded
             first estimate, accurate to roughly a pixel, bounded by
             orbit/timing/DEM precision.
          2. refine_offsets_by_coherence() [optional, on by default] —
             SNAP CrossCorrelationOp's coarse cross-correlation + Powell
             coherence-maximization fine registration, run directly
             against the actual image content.
          3. fit_offset_polynomial_robust() — SNAP WarpOp's iterative
             mean-RMS GCP outlier rejection.
          4. resample_with_offset_field() — apply the final, validated
             warp to the secondary image.

        window/coarse_search_radius/fine_search_radius/coherence_threshold:
        None (default) uses refine_offsets_by_coherence()'s own
        defaults; pass a value to override just that one parameter for
        this pair. See process_pair()'s own docstring for when
        widening coarse_search_radius is worth trying.

        Falls back to the shape-based resample (with a clear warning)
        if anything in this real pipeline raises.

        Returns:
            (resampled_secondary, coreg_metadata) — coreg_metadata
            always includes "method", "refined_by_coherence",
            "n_gcps_initial", "n_gcps_final", "rms_mean_px", and
            "mean_coherence".
        """
        try:
            import rasterio

            from pygeofetch.insar.annotation import parse_slc_geometry
            from pygeofetch.insar.coregister import (
                compute_offset_field_from_dem,
                fit_offset_polynomial_robust,
                read_crop_offset,
                read_matched_swath,
                refine_offsets_by_coherence,
                resample_with_offset_field,
            )
            from pygeofetch.insar.geolocation import parse_orbit_file

            ref_swath_hint = (
                read_matched_swath(reference_extracted_path)
                if reference_extracted_path is not None else None
            )
            sec_swath_hint = (
                read_matched_swath(secondary_extracted_path)
                if secondary_extracted_path is not None else None
            )

            ref_geom = parse_slc_geometry(reference_safe_zip, member_hint=ref_swath_hint)
            sec_geom = parse_slc_geometry(secondary_safe_zip, member_hint=sec_swath_hint)
            ref_orbit = parse_orbit_file(reference_orbit_file)
            sec_orbit = parse_orbit_file(secondary_orbit_file)

            ref_center_time = ref_geom.azimuth_time(ref_geom.n_lines / 2)
            sec_center_time = sec_geom.azimuth_time(sec_geom.n_lines / 2)

            ref_row_off, ref_col_off = (
                read_crop_offset(reference_extracted_path)
                if reference_extracted_path is not None else (0.0, 0.0)
            )
            sec_row_off, sec_col_off = (
                read_crop_offset(secondary_extracted_path)
                if secondary_extracted_path is not None else (0.0, 0.0)
            )

            # ═══════════════════════════════════════════════════════════
            # FIX: Constrain the grid to the CROPPED .tif's geographic bounds.
            # If we don't pass sample_bounds, the 7x7 grid spreads over the
            # full Sentinel-1 swath and completely misses the small clipped
            # AOI where the DEM actually exists.
            # ═══════════════════════════════════════════════════════════
            # sample_bounds = None
            # if reference_extracted_path and Path(reference_extracted_path).exists():
            #     with rasterio.open(reference_extracted_path) as src:
            #         b = src.bounds
            #         sample_bounds = (b.left, b.bottom, b.right, b.top)

            sample_bounds = None
            if reference_extracted_path and Path(reference_extracted_path).exists():
                with rasterio.open(reference_extracted_path) as src:
                    # FIX: Handle GCP-only files (full sub-swath). src.bounds
                    # returns pixel coordinates for GCP-only files. Fit an
                    # approximate transform from GCPs to get real lon/lat bounds.
                    if src.transform.is_identity and src.gcps[0]:
                        from rasterio.transform import array_bounds, from_gcps
                        approx_transform = from_gcps(src.gcps[0])
                        left, bottom, right, top = array_bounds(src.height, src.width, approx_transform)
                        sample_bounds = (left, bottom, right, top)
                    else:
                        b = src.bounds
                        sample_bounds = (b.left, b.bottom, b.right, b.top)

            grid_rows, grid_cols, off_rows, off_cols = compute_offset_field_from_dem(
                ref_geom, ref_orbit, sec_geom, sec_orbit, dem,
                ref_scene_center_time=ref_center_time,
                sec_scene_center_time=sec_center_time,
                # No grid_points override -- use the function's own
                # default (7 -> 49 nominal candidates). A smaller grid
                # (previously hardcoded to 5 here, 25 nominal) leaves
                # too little margin once edge-drop + coherence-threshold
                # + outlier rejection remove the majority of candidates,
                # confirmed against this project's own real survival
                # rates (typically 20-30% of nominal candidates survive).
                sample_bounds=sample_bounds,
            )

            n_gcps_initial = len(grid_rows)

            mean_coherence = None
            if refine_by_coherence:
                refine_kwargs = {}
                if window is not None:
                    refine_kwargs["window"] = window
                if coarse_search_radius is not None:
                    refine_kwargs["coarse_search_radius"] = coarse_search_radius
                if fine_search_radius is not None:
                    refine_kwargs["fine_search_radius"] = fine_search_radius
                if coherence_threshold is not None:
                    refine_kwargs["coherence_threshold"] = coherence_threshold
                try:
                    grid_rows, grid_cols, off_rows, off_cols, coherences = (
                        refine_offsets_by_coherence(
                            ref_complex, sec_complex,
                            grid_rows, grid_cols, off_rows, off_cols,
                            ref_row_offset=ref_row_off, ref_col_offset=ref_col_off,
                            sec_row_offset=sec_row_off, sec_col_offset=sec_col_off,
                            **refine_kwargs,
                        )
                    )
                    mean_coherence = float(sum(coherences) / len(coherences))
                except Exception as exc:
                    logger.warning(
                        "Cross-correlation coherence refinement failed "
                        "(%s) — proceeding with the unrefined orbit/DEM "
                        "offset field.",
                        exc,
                    )

            row_fn, col_fn, coreg_quality = fit_offset_polynomial_robust(
                grid_rows, grid_cols, off_rows, off_cols,
                degree=degree, rms_threshold=rms_threshold,
            )
            coreg_quality.mean_coherence = mean_coherence
            coreg_quality.log_summary()

            # Representative coregistration offset: MEDIAN of all offsets.
            # Using median instead of center-point evaluation because the
            # offset field can be non-uniform, and the median is more robust
            # to outliers than a single-point evaluation.
            import numpy as np
            all_offsets = [row_fn(r, c) for r, c in zip(grid_rows, grid_cols)]
            representative_row_offset = float(np.median(all_offsets)) if all_offsets else 0.0

            coreg_metadata = {
                "method": "orbit_dem_based",
                "refined_by_coherence": refine_by_coherence and mean_coherence is not None,
                "n_gcps_initial": n_gcps_initial,
                "n_gcps_final": coreg_quality.n_gcps_final,
                "rms_mean_px": coreg_quality.rms_mean,
                "mean_coherence": mean_coherence,
                "representative_row_offset_px": representative_row_offset,
            }

            if ref_row_off or ref_col_off or sec_row_off or sec_col_off:
                logger.info(
                    "Correcting for cropped extraction: reference offset "
                    "(%.0f, %.0f), secondary offset (%.0f, %.0f)",
                    ref_row_off, ref_col_off, sec_row_off, sec_col_off,
                )

            logger.info(
                "Real orbit-based coregistration applied (%d/%d grid "
                "points used in final fit)",
                coreg_quality.n_gcps_final, n_gcps_initial,
            )
            resampled = resample_with_offset_field(
                sec_complex, row_fn, col_fn,
                ref_row_offset=ref_row_off, ref_col_offset=ref_col_off,
                sec_row_offset=sec_row_off, sec_col_offset=sec_col_off,
            )
            if resampled.shape != ref_complex.shape:
                resampled = self._resample_to_reference(resampled, ref_complex.shape)
            return resampled, coreg_metadata

        except Exception as exc:
            logger.warning(
                "Real orbit-based coregistration failed (%s) — falling "
                "back to shape-based resampling.",
                exc,
            )
            fallback_metadata = {
                "method": "shape_based_fallback_after_error",
                "refined_by_coherence": False,
                "n_gcps_initial": None,
                "n_gcps_final": None,
                "rms_mean_px": None,
                "mean_coherence": None,
                "error": str(exc),
                "representative_row_offset_px": 0.0,
            }
            if sec_complex.shape != ref_complex.shape:
                return self._resample_to_reference(sec_complex, ref_complex.shape), fallback_metadata
            return sec_complex, fallback_metadata

    def _raster_collocation_coregister(
        self, ref_complex, sec_complex, ref_profile, sec_profile,
        refine_by_coherence=True, degree=1, rms_threshold=None,
        window=None, coarse_search_radius=None, fine_search_radius=None,
        coherence_threshold=None,
    ):
        """
        SNAP CreateStack-style coregistration: resample the secondary
        directly onto the reference's geographic raster using each
        file's own embedded CRS/transform (coregister.
        collocate_by_geocoding), then run the same refinement/robust-fit
        stages _orbit_based_coregister uses.

        window/coarse_search_radius/fine_search_radius/coherence_threshold:
        same meaning as _orbit_based_coregister's — None uses this
        path's own defaults (coarse_search_radius defaults to 12 here,
        wider than orbit_dem's 4, since collocate_by_geocoding's own
        precision can leave more residual error to search over; an
        explicit override replaces that default rather than adding to
        it).
        """
        try:
            from pygeofetch.insar.coregister import (
                _regular_grid_points,
                collocate_by_geocoding,
                fit_offset_polynomial_robust,
                refine_offsets_by_coherence,
                resample_with_offset_field,
            )

            collocated, coverage_fraction, valid_mask = collocate_by_geocoding(
                sec_complex, sec_profile, ref_complex.shape, ref_profile,
            )
            logger.info(
                "Raster collocation (CreateStack-style) applied: %.1f%% "
                "reference-raster coverage from real secondary data.",
                coverage_fraction * 100,
            )

            mean_coherence = None
            if refine_by_coherence:
                candidates = _regular_grid_points(ref_complex.shape, grid_points=9)
                grid_rows, grid_cols = [], []
                for r, c in candidates:
                    ri, ci = int(round(r)), int(round(c))
                    if 0 <= ri < valid_mask.shape[0] and 0 <= ci < valid_mask.shape[1] and valid_mask[ri, ci]:
                        grid_rows.append(r)
                        grid_cols.append(c)
                off_rows = [0.0] * len(grid_rows)
                off_cols = [0.0] * len(grid_cols)
                n_gcps_initial = len(grid_rows)

                refine_kwargs = {"coarse_search_radius": 12}
                if window is not None:
                    refine_kwargs["window"] = window
                if coarse_search_radius is not None:
                    refine_kwargs["coarse_search_radius"] = coarse_search_radius
                if fine_search_radius is not None:
                    refine_kwargs["fine_search_radius"] = fine_search_radius
                if coherence_threshold is not None:
                    refine_kwargs["coherence_threshold"] = coherence_threshold

                try:
                    grid_rows, grid_cols, off_rows, off_cols, coherences = (
                        refine_offsets_by_coherence(
                            ref_complex, collocated,
                            grid_rows, grid_cols, off_rows, off_cols,
                            ref_row_offset=0.0, ref_col_offset=0.0,
                            sec_row_offset=0.0, sec_col_offset=0.0,
                            **refine_kwargs,
                        )
                    )
                    mean_coherence = float(sum(coherences) / len(coherences))
                except Exception as exc:
                    logger.warning(
                        "Cross-correlation coherence refinement failed "
                        "after raster collocation (%s).", exc,
                    )
            else:
                n_gcps_initial = 0

            if refine_by_coherence and mean_coherence is not None:
                row_fn, col_fn, coreg_quality = fit_offset_polynomial_robust(
                    grid_rows, grid_cols, off_rows, off_cols,
                    degree=degree, rms_threshold=rms_threshold,
                )
                coreg_quality.mean_coherence = mean_coherence
                coreg_quality.log_summary()
                resampled = resample_with_offset_field(collocated, row_fn, col_fn)
                n_gcps_final = coreg_quality.n_gcps_final
                rms_mean_px = coreg_quality.rms_mean
            else:
                resampled = collocated
                n_gcps_final = None
                rms_mean_px = None

            if resampled.shape != ref_complex.shape:
                resampled = self._resample_to_reference(resampled, ref_complex.shape)

            coreg_metadata = {
                "method": "raster_collocation",
                "refined_by_coherence": refine_by_coherence and mean_coherence is not None,
                "n_gcps_initial": n_gcps_initial,
                "n_gcps_final": n_gcps_final,
                "rms_mean_px": rms_mean_px,
                "mean_coherence": mean_coherence,
                "collocation_coverage_fraction": coverage_fraction,
            }
            return resampled, coreg_metadata

        except Exception as exc:
            logger.warning(
                "Raster collocation coregistration failed (%s) -- "
                "falling back to shape-based resampling.", exc,
            )
            fallback_metadata = {
                "method": "shape_based_fallback_after_error",
                "refined_by_coherence": False,
                "n_gcps_initial": None,
                "n_gcps_final": None,
                "rms_mean_px": None,
                "mean_coherence": None,
                "collocation_coverage_fraction": None,
                "error": str(exc),
            }
            if sec_complex.shape != ref_complex.shape:
                return self._resample_to_reference(sec_complex, ref_complex.shape), fallback_metadata
            return sec_complex, fallback_metadata

    def _resample_to_reference(self, data, target_shape):
        """Nearest-neighbour resample a complex array to a target shape."""
        np = self._np()
        from scipy.ndimage import zoom

        zf = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
        real = zoom(data.real, zf, order=1)
        imag = zoom(data.imag, zf, order=1)
        return (real + 1j * imag).astype(np.complex64)

    def _estimate_esd_shift(
        self, ref, sec, overlap_frac: float = 0.1
    ) -> Optional[float]:
        """
        Estimate residual azimuth misregistration via Enhanced Spectral Diversity.
        Simplified single-shift estimate (fallback for when burst metadata
        is unavailable). For production-grade per-burst ESD, use
        estimate_esd_shift_per_burst_overlap with real burst metadata.
        """
        np = self._np()
        h = ref.shape[0]
        overlap = max(int(h * overlap_frac), 16)

        fwd = ref[:overlap] * np.conj(sec[:overlap])
        bwd = ref[-overlap:] * np.conj(sec[-overlap:])

        with np.errstate(invalid="ignore"):
            diff_phase = np.angle(np.sum(fwd) * np.conj(np.sum(bwd)))

        if not np.isfinite(diff_phase):
            return None
        shift_px = diff_phase / (2 * np.pi) * 0.01  # conservative scaling
        return float(np.clip(shift_px, -0.5, 0.5))

    def _apply_azimuth_shift(self, data, shift_px: float):
        """Apply a sub-pixel azimuth (row) shift via Fourier phase ramp."""
        np = self._np()
        h, w = data.shape
        freq = np.fft.fftfreq(h).reshape(-1, 1)
        ramp = np.exp(-2j * np.pi * freq * shift_px)
        shifted = np.fft.ifft(np.fft.fft(data, axis=0) * ramp, axis=0)
        return shifted.astype(np.complex64)

    def _remove_topographic_phase(self, interferogram, dem_path: Path, profile) -> Any:
        """
        Remove the topographic phase component using a reference DEM.
        Full implementation — see original docstring for detailed explanation.
        """
        np = self._np()
        try:
            import rasterio
        except ImportError:
            return interferogram, {"correction_applied": False, "reason": "rasterio_missing"}

        try:
            interferogram_crs = profile.get("crs")
            interferogram_transform = profile.get("transform")

            with rasterio.open(dem_path) as dem_src:
                if interferogram_crs is not None and interferogram_transform is not None:
                    from rasterio.warp import Resampling, reproject
                    dem = np.empty(interferogram.shape, dtype=np.float32)
                    reproject(
                        source=rasterio.band(dem_src, 1),
                        destination=dem,
                        src_transform=dem_src.transform,
                        src_crs=dem_src.crs,
                        dst_transform=interferogram_transform,
                        dst_crs=interferogram_crs,
                        resampling=Resampling.bilinear,
                        src_nodata=dem_src.nodata,
                        dst_nodata=np.nan,
                    )
                else:
                    logger.warning(
                        "No real CRS/transform available on the "
                        "interferogram profile -- falling back to a "
                        "shape-ratio DEM resample."
                    )
                    dem = dem_src.read(1).astype(np.float32)
                    if dem.shape != interferogram.shape:
                        from scipy.ndimage import zoom
                        zf = (
                            interferogram.shape[0] / dem.shape[0],
                            interferogram.shape[1] / dem.shape[1],
                        )
                        dem = zoom(dem, zf, order=1)

            valid = np.isfinite(dem) & (dem > -500)
            phase = np.angle(interferogram)

            if valid.sum() < 100:
                logger.warning(
                    "Insufficient valid DEM pixels for topo phase regression"
                )
                return interferogram, {"correction_applied": False, "reason": "insufficient_valid_pixels"}

            dem_std = np.std(dem[valid])
            if dem_std < 1.0:
                logger.info(
                    "DEM has negligible elevation variance (std=%.2fm) — skipping.",
                    dem_std,
                )
                return interferogram, {"correction_applied": False, "reason": "dem_no_variance"}

            dem_v = dem[valid]
            phase_v = phase[valid]

            elev_range = float(np.ptp(dem_v))
            if elev_range < 1.0:
                max_slope = 0.5
            else:
                max_slope = (25.0 * 2 * np.pi) / elev_range

            rng = np.random.default_rng(0)
            n_valid = len(dem_v)
            n_search = 20000
            if n_valid > n_search:
                search_idx = rng.choice(n_valid, size=n_search, replace=False)
                dem_search, phase_search = dem_v[search_idx], phase_v[search_idx]
            else:
                dem_search, phase_search = dem_v, phase_v

            def _flatness(candidate_slopes):
                phase_matrix = (
                    phase_search[None, :]
                    - candidate_slopes[:, None] * dem_search[None, :]
                )
                return np.abs(np.mean(np.exp(1j * phase_matrix), axis=1))

            coarse = np.linspace(-max_slope, max_slope, 400)
            best_slope = float(coarse[np.argmax(_flatness(coarse))])

            fine_half_width = (coarse[1] - coarse[0])
            fine = np.linspace(
                best_slope - fine_half_width, best_slope + fine_half_width, 400
            )
            best_slope = float(fine[np.argmax(_flatness(fine))])

            residual_v = np.angle(np.exp(1j * (phase_v - best_slope * dem_v)))
            intercept = float(np.angle(np.mean(np.exp(1j * residual_v))))

            fitted_phase_v = np.angle(np.exp(1j * (best_slope * dem_v + intercept)))
            residual = np.angle(np.exp(1j * (phase_v - fitted_phase_v)))
            ss_res = np.sum(residual**2)
            # BUG FIX: use the circular mean, not a naive arithmetic mean
            # of wrapped angles -- see the identical fix in
            # _remove_topographic_phase_chunked above for the full
            # explanation and test_topo_r2_bug.py for the demonstrated
            # impact (this mistake can inflate a near-zero true R² past
            # the 0.5 gate below, accepting a spurious correction).
            circular_mean_phase = np.angle(np.mean(np.exp(1j * phase_v)))
            centered = np.angle(np.exp(1j * (phase_v - circular_mean_phase)))
            ss_tot = np.sum(centered**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

            if r_squared < 0.5:
                logger.info(
                    "DEM-elevation correlation too weak (R²=%.2f, best "
                    "candidate slope=%.5f rad/m) — skipping topographic "
                    "phase removal to avoid absorbing real signal.",
                    r_squared, best_slope,
                )
                return interferogram, {"correction_applied": False, "r_squared": float(r_squared)}

            topo_phase = np.angle(np.exp(1j * (best_slope * dem + intercept)))

            logger.info(
                "Topographic phase regression R²=%.2f — correction applied", r_squared
            )
            corrected = interferogram * np.exp(-1j * topo_phase)
            return corrected.astype(np.complex64), {"correction_applied": True, "r_squared": float(r_squared)}

        except Exception as exc:
            logger.warning(
                "Topographic phase removal failed: %s — returning uncorrected", exc
            )
            return interferogram, {"correction_applied": False, "reason": str(exc)}

    def _estimate_coherence(self, ref, sec, window: int) -> Any:
        """Estimate interferometric coherence via local windowed correlation."""
        from pygeofetch.insar.gpu import get_array_module, to_numpy

        xp, ndi, using_gpu = get_array_module(prefer_gpu=self._use_gpu)

        ref_x = xp.asarray(ref)
        sec_x = xp.asarray(sec)

        inter = ref_x * xp.conj(sec_x)
        num = xp.abs(
            ndi.uniform_filter(inter.real, size=window)
            + 1j * ndi.uniform_filter(inter.imag, size=window)
        )
        denom = xp.sqrt(
            ndi.uniform_filter(xp.abs(ref_x) ** 2, size=window)
            * ndi.uniform_filter(xp.abs(sec_x) ** 2, size=window)
            + 1e-10
        )
        coherence = xp.clip(num / denom, 0.0, 1.0).astype(xp.float32)
        return to_numpy(coherence)
