"""
SpectralIndices — 18 spectral indices and band transformations.
All methods return a ProcessingResult with the computed raster path.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pygeofetch.processing.base import (
    ProcessingResult,
    _require_numpy,
    _require_scipy,
    _resolve_output,
    _safe_read_band,
    _safe_write_band,
    _timed,
    chunked_index_compute,
    detect_band_roles,
    resolve_shared_bands,
)

logger = logging.getLogger(__name__)


class SpectralIndices:
    """
    Spectral index computation engine.

    All methods accept per-band file paths (str or Path) and return a
    ProcessingResult with the index raster at output_path.
    Output rasters are float32, DEFLATE-compressed, COG-tiled GeoTIFFs.
    NaN marks nodata/invalid pixels; range is typically -1 to +1 for
    normalised indices.

    Example::

        from pygeofetch import PyGeoFetch
        client = PyGeoFetch()
        ndvi = client.indices.ndvi(red="B04.tif", nir="B08.tif")
        evi  = client.indices.evi(blue="B02.tif", red="B04.tif", nir="B08.tif")
    """

    # ── internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _read(path: str | Path, ref_shape=None, band: int = 1):
        """
        Read a specific band of a raster robustly (band=1 by default,
        preserving existing behavior for single-band inputs).
        Uses block-by-block fallback for tiled/COG/compressed GeoTIFFs.
        Optionally resamples to ref_shape (h, w).
        Returns (data_float32, profile, nodata).
        """
        return _safe_read_band(path, band=band, out_shape=ref_shape)

    @staticmethod
    def _norm_diff(a, b):
        """(a - b) / (a + b) with safe division → NaN where denominator=0."""
        np = _require_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(a + b != 0, (a - b) / (a + b), float("nan"))

    def _save(self, data, profile, out_path: Path, nodata=-9999.0) -> None:
        """Write result array to a clean, tiled, DEFLATE-compressed GeoTIFF."""
        _safe_write_band(data, profile, out_path, nodata=nodata)

    # ── E1: NDVI ─────────────────────────────────────────────────────────

    @_timed
    def ndvi(
        self,
        red: str | Path,
        nir: str | Path,
        output: str | None = None,
        chunked: bool = False,
        tile_size: int = 1024,
    ) -> ProcessingResult:
        """
        NDVI — Normalized Difference Vegetation Index.
        Formula: (NIR - Red) / (NIR + Red).  Range: -1 to +1.
        Values > 0.3 indicate healthy vegetation.

        Set chunked=True for large rasters (e.g. drone orthomosaics)
        that shouldn't be fully loaded into memory at once, and that
        may exceed the standard GeoTIFF 4GB size limit -- this writes
        a proper tiled, BigTIFF output, processed in real, fixed-size
        tiles rather than one full-array read/write. Default is False
        to preserve existing behavior and performance for the common,
        normal-sized-scene case.

        If red and nir point to the SAME merged multi-band image (e.g.
        a drone orthomosaic with all bands in one file), the correct
        bands are auto-detected from its real band descriptions -- no
        manual pre-splitting required.
        """
        red_ref, nir_ref = red, nir
        if str(red) == str(nir):
            resolved = resolve_shared_bands(red, ["red", "nir"])
            red_ref, nir_ref = resolved["red"], resolved["nir"]

        out_path = _resolve_output(Path(red), output, "ndvi")
        if chunked:
            chunked_index_compute(
                inputs=[red_ref, nir_ref],
                formula=lambda B: (B[1] - B[0]) / (B[1] + B[0]),
                output=out_path, tile_size=tile_size,
            )
        else:
            red_band = red_ref[1] if isinstance(red_ref, tuple) else 1
            nir_band = nir_ref[1] if isinstance(nir_ref, tuple) else 1
            red_d, profile, _ = self._read(red, band=red_band)
            nir_d, _, _ = self._read(nir, ref_shape=red_d.shape, band=nir_band)
            result = self._norm_diff(nir_d, red_d)
            self._save(result, profile, out_path)
        logger.info("NDVI → %s", out_path.name)
        return ProcessingResult(
            success=True, operation="ndvi", output_path=out_path, input_path=Path(red)
        )

    # ── E1b: NDRE ────────────────────────────────────────────────────────

    @_timed
    def ndre(
        self,
        rededge: str | Path,
        nir: str | Path,
        output: str | None = None,
        chunked: bool = False,
        tile_size: int = 1024,
    ) -> ProcessingResult:
        """
        NDRE — Normalized Difference Red Edge Index (Barnes et al. 2000).
        Formula: (NIR - RedEdge) / (NIR + RedEdge).  Range: -1 to +1.

        Uses the red edge band (~705-750nm) instead of red, which makes
        it less prone to saturation than NDVI in dense canopy -- useful
        for detecting crop stress and chlorophyll content in precision
        agriculture, commonly from drone multispectral sensors
        (MicaSense RedEdge/Altum, DJI P4 Multispectral, and similar)
        that carry a dedicated red edge band alongside NIR.

        Set chunked=True for large rasters (e.g. drone orthomosaics)
        that shouldn't be fully loaded into memory at once, and that
        may exceed the standard GeoTIFF 4GB size limit -- see ndvi's
        docstring for details.

        If rededge and nir point to the SAME multi-band file (e.g. a
        drone orthomosaic with all bands in one raster), the correct
        bands are auto-detected from the file's own real band
        descriptions -- no manual pre-splitting required. If the file
        has no real, matching band descriptions, a clear error is
        raised rather than silently reading band 1 for both roles.
        """
        rededge_ref, nir_ref = rededge, nir
        if str(rededge) == str(nir):
            resolved = resolve_shared_bands(rededge, ["rededge", "nir"])
            rededge_ref, nir_ref = resolved["rededge"], resolved["nir"]

        out_path = _resolve_output(Path(rededge), output, "ndre")
        if chunked:
            chunked_index_compute(
                inputs=[rededge_ref, nir_ref],
                formula=lambda B: (B[1] - B[0]) / (B[1] + B[0]),
                output=out_path, tile_size=tile_size,
            )
        else:
            re_band = rededge_ref[1] if isinstance(rededge_ref, tuple) else 1
            nir_band = nir_ref[1] if isinstance(nir_ref, tuple) else 1
            rededge_d, profile, _ = self._read(rededge, band=re_band)
            nir_d, _, _ = self._read(nir, ref_shape=rededge_d.shape, band=nir_band)
            result = self._norm_diff(nir_d, rededge_d)
            self._save(result, profile, out_path)
        logger.info("NDRE → %s", out_path.name)
        return ProcessingResult(
            success=True, operation="ndre", output_path=out_path, input_path=Path(rededge)
        )

    # ── E2: EVI ──────────────────────────────────────────────────────────

    @_timed
    def evi(
        self,
        blue: str | Path,
        red: str | Path,
        nir: str | Path,
        G: float = 2.5,
        C1: float = 6.0,
        C2: float = 7.5,
        L: float = 1.0,
        output: str | None = None,
        chunked: bool = False,
        tile_size: int = 1024,
    ) -> ProcessingResult:
        """
        EVI — Enhanced Vegetation Index.
        Formula: G * (NIR-Red) / (NIR + C1*Red - C2*Blue + L).
        Reduces atmospheric and canopy background effects vs NDVI.

        Set chunked=True for large rasters that shouldn't be fully
        loaded into memory at once (see ndvi's docstring for details).
        If blue, red, and nir point to the SAME merged multi-band
        image, the correct bands are auto-detected from its real band
        descriptions.
        """
        np = _require_numpy()
        blue_ref, red_ref, nir_ref = blue, red, nir
        if str(blue) == str(red) == str(nir):
            resolved = resolve_shared_bands(blue, ["blue", "red", "nir"])
            blue_ref, red_ref, nir_ref = resolved["blue"], resolved["red"], resolved["nir"]

        out_path = _resolve_output(Path(red), output, "evi")

        def _evi_formula(B):
            blue_d, red_d, nir_d = B
            denom = nir_d + C1 * red_d - C2 * blue_d + L
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(denom != 0, G * (nir_d - red_d) / denom, float("nan"))

        if chunked:
            chunked_index_compute(
                inputs=[blue_ref, red_ref, nir_ref],
                formula=_evi_formula,
                output=out_path, tile_size=tile_size,
            )
        else:
            blue_band = blue_ref[1] if isinstance(blue_ref, tuple) else 1
            red_band = red_ref[1] if isinstance(red_ref, tuple) else 1
            nir_band = nir_ref[1] if isinstance(nir_ref, tuple) else 1
            blue_d, profile, _ = self._read(blue, band=blue_band)
            red_d, _, _ = self._read(red, ref_shape=blue_d.shape, band=red_band)
            nir_d, _, _ = self._read(nir, ref_shape=blue_d.shape, band=nir_band)
            result = _evi_formula([blue_d, red_d, nir_d])
            self._save(result, profile, out_path)
        logger.info("EVI → %s", out_path.name)
        return ProcessingResult(
            success=True, operation="evi", output_path=out_path, input_path=Path(red)
        )

    # ── E3: SAVI ─────────────────────────────────────────────────────────

    @_timed
    def savi(
        self,
        red: str | Path,
        nir: str | Path,
        L: float = 0.5,
        output: str | None = None,
        chunked: bool = False,
        tile_size: int = 1024,
    ) -> ProcessingResult:
        """
        SAVI — Soil Adjusted Vegetation Index.
        Formula: (NIR-Red)/(NIR+Red+L) * (1+L).
        L=0.5 is standard; corrects for soil background reflectance.

        Set chunked=True for large rasters that shouldn't be fully
        loaded into memory at once (see ndvi's docstring for details).
        If red and nir point to the SAME merged multi-band image, the
        correct bands are auto-detected from its real band descriptions.
        """
        np = _require_numpy()
        red_ref, nir_ref = red, nir
        if str(red) == str(nir):
            resolved = resolve_shared_bands(red, ["red", "nir"])
            red_ref, nir_ref = resolved["red"], resolved["nir"]

        out_path = _resolve_output(Path(red), output, "savi")

        def _savi_formula(B):
            red_d, nir_d = B
            denom = nir_d + red_d + L
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(denom != 0, (nir_d - red_d) / denom * (1 + L), float("nan"))

        if chunked:
            chunked_index_compute(
                inputs=[red_ref, nir_ref],
                formula=_savi_formula,
                output=out_path, tile_size=tile_size,
            )
        else:
            red_band = red_ref[1] if isinstance(red_ref, tuple) else 1
            nir_band = nir_ref[1] if isinstance(nir_ref, tuple) else 1
            red_d, profile, _ = self._read(red, band=red_band)
            nir_d, _, _ = self._read(nir, ref_shape=red_d.shape, band=nir_band)
            result = _savi_formula([red_d, nir_d])
            self._save(result, profile, out_path)
        return ProcessingResult(
            success=True, operation="savi", output_path=out_path, input_path=Path(red)
        )

    # ── E4: NDWI ─────────────────────────────────────────────────────────

    @_timed
    def ndwi(
        self,
        green: str | Path,
        nir: str | Path,
        output: str | None = None,
        chunked: bool = False,
        tile_size: int = 1024,
    ) -> ProcessingResult:
        """
        NDWI — Normalized Difference Water Index (McFeeters 1996).
        Formula: (Green - NIR) / (Green + NIR).  Water > 0, land < 0.

        Set chunked=True for large rasters that shouldn't be fully
        loaded into memory at once (see ndvi's docstring for details).
        If green and nir point to the SAME merged multi-band image,
        the correct bands are auto-detected from its real band
        descriptions.
        """
        green_ref, nir_ref = green, nir
        if str(green) == str(nir):
            resolved = resolve_shared_bands(green, ["green", "nir"])
            green_ref, nir_ref = resolved["green"], resolved["nir"]

        out_path = _resolve_output(Path(green), output, "ndwi")
        if chunked:
            chunked_index_compute(
                inputs=[green_ref, nir_ref],
                formula=lambda B: (B[0] - B[1]) / (B[0] + B[1]),
                output=out_path, tile_size=tile_size,
            )
        else:
            green_band = green_ref[1] if isinstance(green_ref, tuple) else 1
            nir_band = nir_ref[1] if isinstance(nir_ref, tuple) else 1
            green_d, profile, _ = self._read(green, band=green_band)
            nir_d, _, _ = self._read(nir, ref_shape=green_d.shape, band=nir_band)
            result = self._norm_diff(green_d, nir_d)
            self._save(result, profile, out_path)
        return ProcessingResult(
            success=True, operation="ndwi", output_path=out_path, input_path=Path(green)
        )

    # ── E5: MNDWI ────────────────────────────────────────────────────────

    @_timed
    def mndwi(
        self,
        green: str | Path,
        swir1: str | Path,
        output: str | None = None,
        chunked: bool = False,
        tile_size: int = 1024,
    ) -> ProcessingResult:
        """
        MNDWI — Modified NDWI (Xu 2006).
        Formula: (Green - SWIR1) / (Green + SWIR1).
        Better separation of water from built-up areas than NDWI.

        Set chunked=True for large rasters that shouldn't be fully
        loaded into memory at once (see ndvi's docstring for details).
        If green and swir1 point to the SAME merged multi-band image,
        the correct bands are auto-detected from its real band
        descriptions.
        """
        green_ref, swir_ref = green, swir1
        if str(green) == str(swir1):
            resolved = resolve_shared_bands(green, ["green", "swir1"])
            green_ref, swir_ref = resolved["green"], resolved["swir1"]

        out_path = _resolve_output(Path(green), output, "mndwi")
        if chunked:
            chunked_index_compute(
                inputs=[green_ref, swir_ref],
                formula=lambda B: (B[0] - B[1]) / (B[0] + B[1]),
                output=out_path, tile_size=tile_size,
            )
        else:
            green_band = green_ref[1] if isinstance(green_ref, tuple) else 1
            swir_band = swir_ref[1] if isinstance(swir_ref, tuple) else 1
            green_d, profile, _ = self._read(green, band=green_band)
            swir_d, _, _ = self._read(swir1, ref_shape=green_d.shape, band=swir_band)
            result = self._norm_diff(green_d, swir_d)
            self._save(result, profile, out_path)
        return ProcessingResult(
            success=True,
            operation="mndwi",
            output_path=out_path,
            input_path=Path(green),
        )

    # ── E6: NDBI ─────────────────────────────────────────────────────────

    @_timed
    def ndbi(
        self,
        nir: str | Path,
        swir1: str | Path,
        output: str | None = None,
        chunked: bool = False,
        tile_size: int = 1024,
    ) -> ProcessingResult:
        """
        NDBI — Normalized Difference Built-up Index (Zha 2003).
        Formula: (SWIR1 - NIR) / (SWIR1 + NIR).  Urban > 0, vegetation < 0.

        Set chunked=True for large rasters that shouldn't be fully
        loaded into memory at once (see ndvi's docstring for details).
        If nir and swir1 point to the SAME merged multi-band image,
        the correct bands are auto-detected from its real band
        descriptions.
        """
        nir_ref, swir_ref = nir, swir1
        if str(nir) == str(swir1):
            resolved = resolve_shared_bands(nir, ["nir", "swir1"])
            nir_ref, swir_ref = resolved["nir"], resolved["swir1"]

        out_path = _resolve_output(Path(nir), output, "ndbi")
        if chunked:
            chunked_index_compute(
                inputs=[nir_ref, swir_ref],
                formula=lambda B: (B[1] - B[0]) / (B[1] + B[0]),
                output=out_path, tile_size=tile_size,
            )
        else:
            nir_band = nir_ref[1] if isinstance(nir_ref, tuple) else 1
            swir_band = swir_ref[1] if isinstance(swir_ref, tuple) else 1
            swir_d, profile, _ = self._read(swir1, band=swir_band)
            nir_d, _, _ = self._read(nir, ref_shape=swir_d.shape, band=nir_band)
            result = self._norm_diff(swir_d, nir_d)
            self._save(result, profile, out_path)
        return ProcessingResult(
            success=True, operation="ndbi", output_path=out_path, input_path=Path(nir)
        )

    # ── E7: NDSI ─────────────────────────────────────────────────────────

    @_timed
    def ndsi(
        self,
        green: str | Path,
        swir1: str | Path,
        output: str | None = None,
        chunked: bool = False,
        tile_size: int = 1024,
    ) -> ProcessingResult:
        """
        NDSI — Normalized Difference Snow Index (Hall 1995).
        Formula: (Green - SWIR1) / (Green + SWIR1).  Snow > 0.4.

        Set chunked=True for large rasters that shouldn't be fully
        loaded into memory at once (see ndvi's docstring for details).
        If green and swir1 point to the SAME merged multi-band image,
        the correct bands are auto-detected from its real band
        descriptions.
        """
        green_ref, swir_ref = green, swir1
        if str(green) == str(swir1):
            resolved = resolve_shared_bands(green, ["green", "swir1"])
            green_ref, swir_ref = resolved["green"], resolved["swir1"]

        out_path = _resolve_output(Path(green), output, "ndsi")
        if chunked:
            chunked_index_compute(
                inputs=[green_ref, swir_ref],
                formula=lambda B: (B[0] - B[1]) / (B[0] + B[1]),
                output=out_path, tile_size=tile_size,
            )
        else:
            green_band = green_ref[1] if isinstance(green_ref, tuple) else 1
            swir_band = swir_ref[1] if isinstance(swir_ref, tuple) else 1
            green_d, profile, _ = self._read(green, band=green_band)
            swir_d, _, _ = self._read(swir1, ref_shape=green_d.shape, band=swir_band)
            result = self._norm_diff(green_d, swir_d)
            self._save(result, profile, out_path)
        return ProcessingResult(
            success=True, operation="ndsi", output_path=out_path, input_path=Path(green)
        )

    # ── E8: NDMI ─────────────────────────────────────────────────────────

    @_timed
    def ndmi(
        self,
        nir: str | Path,
        swir1: str | Path,
        output: str | None = None,
        chunked: bool = False,
        tile_size: int = 1024,
    ) -> ProcessingResult:
        """
        NDMI — Normalized Difference Moisture Index (Wilson & Sader 2002).
        Formula: (NIR - SWIR1) / (NIR + SWIR1).
        Sensitive to canopy water content; positive = moist vegetation.

        Set chunked=True for large rasters that shouldn't be fully
        loaded into memory at once (see ndvi's docstring for details).
        If nir and swir1 point to the SAME merged multi-band image,
        the correct bands are auto-detected from its real band
        descriptions.
        """
        nir_ref, swir_ref = nir, swir1
        if str(nir) == str(swir1):
            resolved = resolve_shared_bands(nir, ["nir", "swir1"])
            nir_ref, swir_ref = resolved["nir"], resolved["swir1"]

        out_path = _resolve_output(Path(nir), output, "ndmi")
        if chunked:
            chunked_index_compute(
                inputs=[nir_ref, swir_ref],
                formula=lambda B: (B[0] - B[1]) / (B[0] + B[1]),
                output=out_path, tile_size=tile_size,
            )
        else:
            nir_band = nir_ref[1] if isinstance(nir_ref, tuple) else 1
            swir_band = swir_ref[1] if isinstance(swir_ref, tuple) else 1
            nir_d, profile, _ = self._read(nir, band=nir_band)
            swir_d, _, _ = self._read(swir1, ref_shape=nir_d.shape, band=swir_band)
            result = self._norm_diff(nir_d, swir_d)
            self._save(result, profile, out_path)
        return ProcessingResult(
            success=True, operation="ndmi", output_path=out_path, input_path=Path(nir)
        )

    # ── E9: NBR ──────────────────────────────────────────────────────────

    @_timed
    def nbr(
        self,
        nir: str | Path,
        swir2: str | Path,
        output: str | None = None,
        chunked: bool = False,
        tile_size: int = 1024,
    ) -> ProcessingResult:
        """
        NBR — Normalized Burn Ratio.
        Formula: (NIR - SWIR2) / (NIR + SWIR2).
        Use dNBR = pre_NBR - post_NBR for burn severity mapping.

        Set chunked=True for large rasters that shouldn't be fully
        loaded into memory at once (see ndvi's docstring for details).
        If nir and swir2 point to the SAME merged multi-band image,
        the correct bands are auto-detected from its real band
        descriptions.
        """
        nir_ref, swir_ref = nir, swir2
        if str(nir) == str(swir2):
            resolved = resolve_shared_bands(nir, ["nir", "swir2"])
            nir_ref, swir_ref = resolved["nir"], resolved["swir2"]

        out_path = _resolve_output(Path(nir), output, "nbr")
        if chunked:
            chunked_index_compute(
                inputs=[nir_ref, swir_ref],
                formula=lambda B: (B[0] - B[1]) / (B[0] + B[1]),
                output=out_path, tile_size=tile_size,
            )
        else:
            nir_band = nir_ref[1] if isinstance(nir_ref, tuple) else 1
            swir_band = swir_ref[1] if isinstance(swir_ref, tuple) else 1
            nir_d, profile, _ = self._read(nir, band=nir_band)
            swir2_d, _, _ = self._read(swir2, ref_shape=nir_d.shape, band=swir_band)
            result = self._norm_diff(nir_d, swir2_d)
            self._save(result, profile, out_path)
        return ProcessingResult(
            success=True, operation="nbr", output_path=out_path, input_path=Path(nir)
        )

    # ── E10: dNBR ────────────────────────────────────────────────────────

    @_timed
    def dnbr(
        self,
        pre_nir: str | Path,
        pre_swir2: str | Path,
        post_nir: str | Path,
        post_swir2: str | Path,
        output: str | None = None,
    ) -> ProcessingResult:
        """
        dNBR — differenced Normalized Burn Ratio (burn severity).
        Formula: NBR_pre - NBR_post.
        Range: < -0.25 = regrowth; > 0.66 = high severity fire.
        """
        pre_nir_d, profile, _ = self._read(pre_nir)
        pre_swir_d, _, _ = self._read(pre_swir2, ref_shape=pre_nir_d.shape)
        post_nir_d, _, _ = self._read(post_nir, ref_shape=pre_nir_d.shape)
        post_swir_d, _, _ = self._read(post_swir2, ref_shape=pre_nir_d.shape)
        nbr_pre = self._norm_diff(pre_nir_d, pre_swir_d)
        nbr_post = self._norm_diff(post_nir_d, post_swir_d)
        result = nbr_pre - nbr_post
        out_path = _resolve_output(Path(pre_nir), output, "dnbr")
        self._save(result, profile, out_path)
        return ProcessingResult(
            success=True,
            operation="dnbr",
            output_path=out_path,
            input_path=Path(pre_nir),
        )

    # ── E11: TCT — Tasseled Cap Transformation ────────────────────────────

    @_timed
    def tct(
        self,
        blue: str | Path,
        green: str | Path,
        red: str | Path,
        nir: str | Path,
        swir1: str | Path,
        swir2: str | Path,
        output: str | None = None,
        sensor: str = "sentinel2",
    ) -> ProcessingResult:
        """
        Tasseled Cap Transformation.
        Produces 3-band output: Brightness, Greenness, Wetness.

        Args:
            blue, green, red, nir, swir1, swir2: Input band paths.
            sensor: ``"sentinel2"`` (default) or ``"landsat8"``.
            output: Output path.
        """
        np = _require_numpy()

        ref_d, profile, _ = self._read(blue)
        ref_shape = ref_d.shape
        bands = [ref_d]
        for bp in [green, red, nir, swir1, swir2]:
            d, _, _ = self._read(bp, ref_shape=ref_shape)
            bands.append(d)

        B = np.stack(bands, axis=0)  # (6, H, W)

        if sensor == "sentinel2":
            # Nedkov (2017)
            coefs = np.array(
                [
                    [0.3510, 0.3813, 0.3437, 0.7196, 0.2396, 0.1949],
                    [-0.3599, -0.3533, -0.4734, 0.6633, 0.0087, -0.2856],
                    [0.2578, 0.2305, 0.0883, 0.1071, -0.7611, -0.5308],
                ]
            )
        else:  # landsat8 — Baig et al. (2014)
            coefs = np.array(
                [
                    [0.3029, 0.2786, 0.4733, 0.5599, 0.5080, 0.1872],
                    [-0.2941, -0.2430, -0.5424, 0.7276, 0.0713, -0.1608],
                    [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559],
                ]
            )

        tct_data = np.tensordot(coefs, B, axes=([1], [0]))  # (3, H, W)
        out_path = _resolve_output(Path(red), output, "tct")
        _safe_write_band(tct_data, profile, out_path)
        logger.info("TCT (Brightness, Greenness, Wetness) → %s", out_path.name)
        return ProcessingResult(
            success=True,
            operation="tct",
            output_path=out_path,
            metadata={
                "sensor": sensor,
                "bands": ["Brightness", "Greenness", "Wetness"],
            },
        )

    # ── E12: PCA ─────────────────────────────────────────────────────────

    @_timed
    def pca(
        self,
        inputs: list[str | Path],
        n_components: int = 3,
        output: str | None = None,
    ) -> ProcessingResult:
        """
        Principal Component Analysis — dimensionality reduction.

        Args:
            inputs:       List of single-band rasters.
            n_components: Number of components to retain.
            output:       Output (n_components-band) GeoTIFF.

        Returns:
            ProcessingResult with metadata['explained_variance_pct'].
        """
        np = _require_numpy()

        bands, profile = [], None
        ref_shape = None
        for p in inputs:
            d, prof, _ = self._read(p, ref_shape=ref_shape)
            if profile is None:
                profile = prof
                ref_shape = d.shape
            bands.append(d.ravel())

        X = np.stack(bands, axis=0).T  # (pixels, n_bands)
        valid = np.all(np.isfinite(X), axis=1)
        X_v = X[valid]

        mean = X_v.mean(axis=0)
        std = X_v.std(axis=0) + 1e-10
        X_std = (X_v - mean) / std

        cov = np.cov(X_std.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        n_components = min(n_components, len(inputs))
        W = eigenvectors[:, :n_components]
        explained = eigenvalues[idx[:n_components]] / (eigenvalues.sum() + 1e-10) * 100

        assert ref_shape is not None, (
            "ref_shape is set together with profile on the first "
            "iteration of the inputs loop above, so it is always set "
            "here for a non-empty inputs list."
        )
        h, w = ref_shape
        pcs = np.full((n_components, h * w), np.nan, dtype=np.float32)
        pcs[:, valid] = (X_std @ W).T
        pcs = pcs.reshape(n_components, h, w)

        if profile is None:
            msg = "No valid input bands provided to pca()"
            raise ValueError(msg)
        if profile is None:
            raise ValueError("No valid input bands provided")
        out_path = _resolve_output(Path(inputs[0]), output, f"pca_{n_components}comp")
        _safe_write_band(pcs, profile, out_path)
        logger.info("PCA %d components → %s", n_components, out_path.name)
        return ProcessingResult(
            success=True,
            operation="pca",
            output_path=out_path,
            metadata={
                "n_components": n_components,
                "explained_variance_pct": [round(float(e), 2) for e in explained],
            },
        )

    # ── E13: Texture (GLCM) ───────────────────────────────────────────────

    @_timed
    def texture(
        self,
        input: str | Path,
        window: int = 5,
        features: list[str] | None = None,
        output: str | None = None,
    ) -> ProcessingResult:
        """
        GLCM Texture Features — contrast, homogeneity, energy, correlation.

        Uses scipy.ndimage for efficient computation rather than pure-Python loops.

        Args:
            input:    Single-band raster (e.g. NIR).
            window:   Window size (must be odd, default 5).
            features: Subset of ``["contrast","dissimilarity","homogeneity",
                      "energy","correlation","ASM"]``. Default: all 6.
            output:   Output multi-band path (one band per feature).
        """
        np = _require_numpy()
        ndimage = _require_scipy()

        features = features or [
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
            "ASM",
        ]

        inp = Path(input)
        data, profile, _ = self._read(inp)

        # Quantise to 64 grey levels for tractable GLCM size
        valid = np.isfinite(data)
        q = np.zeros_like(data, dtype=np.int32)
        if valid.any():
            d_min, d_max = data[valid].min(), data[valid].max()
            q[valid] = ((data[valid] - d_min) / (d_max - d_min + 1e-10) * 63).astype(
                np.int32
            )

        n_feats = len(features)
        h, w = data.shape
        out_maps = np.zeros((n_feats, h, w), dtype=np.float32)

        # Compute GLCM-derived features efficiently with scipy uniform filters
        # This avoids the O(H*W*window²) pure-Python loop
        qf = q.astype(np.float32)

        for fi, feat in enumerate(features):
            if feat == "contrast":
                # Var of difference: E[(i-j)²] ≈ var of pixel - pixel+shift
                shifted = np.roll(qf, 1, axis=1)
                diff = (qf - shifted) ** 2
                out_maps[fi] = ndimage.uniform_filter(
                    diff.astype(np.float64), size=window
                ).astype(np.float32)

            elif feat == "dissimilarity":
                shifted = np.roll(qf, 1, axis=1)
                diff = np.abs(qf - shifted)
                out_maps[fi] = ndimage.uniform_filter(
                    diff.astype(np.float64), size=window
                ).astype(np.float32)

            elif feat == "homogeneity":
                shifted = np.roll(qf, 1, axis=1)
                diff_sq = (qf - shifted) ** 2
                homo = 1.0 / (1.0 + diff_sq)
                out_maps[fi] = ndimage.uniform_filter(
                    homo.astype(np.float64), size=window
                ).astype(np.float32)

            elif feat in ("energy", "ASM"):
                # Local ASM ≈ 1/std² of local patch
                local_mean = ndimage.uniform_filter(qf.astype(np.float64), size=window)
                local_mean2 = ndimage.uniform_filter(
                    (qf**2).astype(np.float64), size=window
                )
                local_var = np.maximum(local_mean2 - local_mean**2, 0)
                asm = 1.0 / (1.0 + local_var)
                out_maps[fi] = (np.sqrt(asm) if feat == "energy" else asm).astype(
                    np.float32
                )

            elif feat == "correlation":
                local_mean = ndimage.uniform_filter(qf.astype(np.float64), size=window)
                local_mean2 = ndimage.uniform_filter(
                    (qf**2).astype(np.float64), size=window
                )
                local_var = np.maximum(local_mean2 - local_mean**2, 1e-10)
                shifted = np.roll(qf, 1, axis=1)
                cross_mean = ndimage.uniform_filter(
                    (qf * shifted).astype(np.float64), size=window
                )
                cov_xy = cross_mean - local_mean * local_mean
                corr = np.clip(cov_xy / local_var, -1, 1)
                out_maps[fi] = corr.astype(np.float32)

        out_path = _resolve_output(inp, output, "texture")
        _safe_write_band(out_maps, profile, out_path)
        logger.info("GLCM texture (%s) → %s", features, out_path.name)
        return ProcessingResult(
            success=True,
            operation="texture",
            input_path=inp,
            output_path=out_path,
            metadata={"features": features, "window": window},
        )

    # ── E14: LST — Land Surface Temperature ──────────────────────────────

    @_timed
    def lst(
        self,
        thermal: str | Path,
        emissivity: float = 0.97,
        output: str | None = None,
        sensor: str = "landsat8",
    ) -> ProcessingResult:
        """
        Land Surface Temperature from Landsat/MODIS thermal band.

        Args:
            thermal:    Thermal band raster in DN (Landsat B10 or MODIS).
            emissivity: Surface emissivity (0.97=vegetation, 0.98=water).
            sensor:     ``"landsat8"``, ``"landsat9"``, or ``"modis"``.
            output:     Output path (2-band: Kelvin and Celsius).

        Returns:
            ProcessingResult; metadata includes emissivity and sensor.
        """
        np = _require_numpy()
        inp = Path(thermal)
        thermal_d, profile, _ = self._read(inp)

        if sensor in ("landsat8", "landsat9"):
            K1, K2 = 774.8853, 1321.0789  # Band 10 thermal constants
            ML, AL = 3.3420e-4, 0.1  # Radiance rescaling
        elif sensor == "modis":
            K1, K2 = 607.76, 1260.56
            ML, AL = 1.0, 0.0
        else:
            K1, K2 = 774.8853, 1321.0789
            ML, AL = 3.3420e-4, 0.1

        radiance = ML * thermal_d + AL
        with np.errstate(divide="ignore", invalid="ignore"):
            T_bright = K2 / np.log(K1 / (radiance + 1e-10) + 1)

        # Emissivity correction
        rho = 1.438e-2  # m·K
        lambda_t = 10.8e-6  # effective wavelength (m)
        with np.errstate(divide="ignore", invalid="ignore"):
            lst_k = T_bright / (1 + (lambda_t * T_bright / rho) * np.log(emissivity))
        lst_c = lst_k - 273.15

        result = np.stack([lst_k, lst_c], axis=0)
        out_path = _resolve_output(inp, output, "lst")
        _safe_write_band(result, profile, out_path)
        logger.info("LST (Band1=Kelvin, Band2=Celsius) → %s", out_path.name)
        return ProcessingResult(
            success=True,
            operation="lst",
            input_path=inp,
            output_path=out_path,
            metadata={
                "emissivity": emissivity,
                "sensor": sensor,
                "bands": ["LST_Kelvin", "LST_Celsius"],
            },
        )

    # ── E15: Albedo ───────────────────────────────────────────────────────

    @_timed
    def albedo(
        self,
        inputs: list[str | Path],
        output: str | None = None,
        sensor: str = "sentinel2",
    ) -> ProcessingResult:
        """
        Narrowband-to-broadband surface albedo (Liang 2001).

        Args:
            inputs: Band paths in sensor order.
                    Sentinel-2: [B02, B03, B04, B08, B11, B12]
                    Landsat-8:  [B2,  B3,  B4,  B5,  B6,  B7]
            sensor: ``"sentinel2"`` or ``"landsat8"``.
            output: Output path.
        """
        np = _require_numpy()

        if sensor == "sentinel2":
            coefs = [0.160, 0.291, 0.243, 0.116, 0.112, 0.081]
            intercept = -0.0015
        else:  # landsat8
            coefs = [0.356, 0.130, 0.373, 0.085, 0.072, -0.0018]
            intercept = -0.0018

        albedo_sum, profile, ref_shape = None, None, None
        for b_path, coef in zip(inputs, coefs):
            d, prof, _ = self._read(b_path, ref_shape=ref_shape)
            if albedo_sum is None:
                albedo_sum = np.zeros_like(d)
                profile = prof
                ref_shape = d.shape
            # Scale from reflectance (0-10000) to 0-1 if needed
            d_scaled = d / 10000.0 if np.nanmax(d) > 10 else d
            albedo_sum += coef * d_scaled

        if albedo_sum is None:
            raise ValueError("No bands in albedo()")
        albedo_sum = np.clip(albedo_sum + intercept, 0, 1)
        out_path = _resolve_output(Path(inputs[0]), output, "albedo")
        self._save(albedo_sum, profile, out_path)
        return ProcessingResult(
            success=True,
            operation="albedo",
            output_path=out_path,
            metadata={"sensor": sensor},
        )

    # ── E16: Band Math ────────────────────────────────────────────────────

    @_timed
    def band_math(
        self,
        inputs: list[str | Path],
        expression: str,
        output: str | None = None,
        chunked: bool = False,
        tile_size: int = 1024,
    ) -> ProcessingResult:
        """
        Arbitrary band arithmetic via a Python expression.

        Args:
            inputs:     Raster paths, OR (path, band_index) tuples to
                        reference a specific band within a shared,
                        merged multi-band image; accessible in
                        expression as B[0], B[1], …
            expression: E.g. ``"(B[1] - B[0]) / (B[1] + B[0] + 1e-6)"``
            output:     Output path.
            chunked:    Set True for large rasters (e.g. drone
                        orthomosaics) that shouldn't be fully loaded
                        into memory at once, and that may exceed the
                        standard GeoTIFF 4GB size limit. Processes in
                        real, fixed-size tiles and writes a proper
                        tiled, BigTIFF output. Default False preserves
                        existing behavior for normal-sized rasters.
        """
        np = _require_numpy()
        first_path = inputs[0][0] if isinstance(inputs[0], tuple) else inputs[0]
        out_path = _resolve_output(Path(first_path), output, "band_math")

        if chunked:
            chunked_index_compute(
                inputs=inputs,
                formula=lambda B: eval(expression, {"B": B, "np": np}),  # noqa: S307
                output=out_path, tile_size=tile_size,
            )
        else:
            B, profile, ref_shape = [], None, None
            for p in inputs:
                path, band = (p[0], p[1]) if isinstance(p, tuple) else (p, 1)
                d, prof, _ = self._read(path, ref_shape=ref_shape, band=band)
                if profile is None:
                    profile = prof
                    ref_shape = d.shape
                B.append(d)

            result = eval(expression, {"B": B, "np": np})  # noqa: S307
            self._save(result, profile, out_path)
        return ProcessingResult(
            success=True,
            operation="band_math",
            output_path=out_path,
            metadata={"expression": expression},
        )

    # ── E17: Stack ───────────────────────────────────────────────────────

    @_timed
    def stack(
        self,
        inputs: list[str | Path],
        output: str | None = None,
    ) -> ProcessingResult:
        """
        Stack multiple single-band rasters into a multi-band GeoTIFF.
        All inputs are resampled to the spatial resolution of the first band.

        Example::

            result = client.indices.stack(["B02.tif","B03.tif","B04.tif"])
        """
        np = _require_numpy()
        bands, profile, ref_shape = [], None, None
        for p in inputs:
            d, prof, _ = self._read(p, ref_shape=ref_shape)
            if profile is None:
                profile = prof
                ref_shape = d.shape
            bands.append(d)

        stacked = np.stack(bands, axis=0)
        if profile is None:
            raise ValueError("No input bands to stack()")
        out_path = _resolve_output(Path(inputs[0]), output, f"stack_{len(inputs)}bands")
        _safe_write_band(stacked, profile, out_path)
        return ProcessingResult(
            success=True,
            operation="stack",
            output_path=out_path,
            metadata={"n_bands": len(bands)},
        )