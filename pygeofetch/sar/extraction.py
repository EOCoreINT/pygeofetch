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
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

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
            path.name, len(gcp_list), georef_path.name,
        )
        return georef_path


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
                    n for n in zf.namelist()
                    if "/measurement/" in n
                    and f"-{self._pol}-" in n
                    and n.endswith(".tiff")
                ]
                if not pol_members:
                    logger.error(
                        "No %s measurement band found in %s",
                        self._pol.upper(), zip_path.name,
                    )
                    return None

                member = pol_members[0]
                stem = f"{label}_{self._pol}_raw" if label else f"{self._pol}_raw"
                raw_path = output_dir / f"{stem}.tif"
                with zf.open(member) as src, open(raw_path, "wb") as dst:
                    dst.write(src.read())
        except (zipfile.BadZipFile, OSError) as exc:
            logger.error("Could not read %s: %s", zip_path.name, exc)
            return None

        return georeference_via_gcps_if_needed(raw_path)

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
