"""
LandsatExtractor — extract and process Landsat Collection 2 Level-2 bundles.

Landsat C2L2 products are delivered as a .tar archive containing
individual band GeoTIFFs (scaled integer DN, not physical units) plus a
QA_PIXEL quality band encoding cloud/shadow/snow/water as bit flags.

This module handles the full chain other satellite-data libraries usually
leave to the user to reimplement per-project:

  1. Extract the .tar bundle
  2. Map band files to common names (correctly handling the band-number
     difference between OLI (Landsat 8/9) and TM/ETM+ (Landsat 4/5/7) —
     a common source of silent errors, since SR_B4 is Red on OLI but NIR
     on TM/ETM+)
  3. Apply the official USGS scale factor and offset to convert scaled
     integers to physical surface reflectance / temperature
  4. Decode QA_PIXEL bit flags into a cloud/shadow mask
  5. Return ready-to-use, cloud-masked reflectance arrays

Reference:
  USGS. Landsat 8-9 Collection 2 Level-2 Science Product Guide (LSDS-1619).
  USGS. "How do I use a scale factor with Landsat Level-2 science products?"
  https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products
"""

from __future__ import annotations

import logging
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from pygeofetch.models.download_task import DownloadResult

logger = logging.getLogger("pygeofetch.processor.landsat")


# Official USGS Collection 2 Level-2 scale factors and offsets.
# Source: USGS Landsat 8-9 C2 Level-2 Science Product Guide (LSDS-1619)
# and https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products
SR_SCALE, SR_OFFSET = 0.0000275, -0.2  # Surface reflectance (SR_B* bands)
ST_SCALE, ST_OFFSET = 0.00341802, 149.0  # Surface temperature (ST_B* bands)

# Band-number-to-common-name mapping. OLI (Landsat 8/9) and TM/ETM+
# (Landsat 4/5/7) use DIFFERENT band numbers for the same wavelengths —
# e.g. SR_B4 is Red on OLI but NIR on TM/ETM+. Silently assuming one
# mapping for both sensor families is a common, hard-to-notice bug.
_OLI_BAND_MAP = {
    "coastal": "SR_B1",
    "blue": "SR_B2",
    "green": "SR_B3",
    "red": "SR_B4",
    "nir": "SR_B5",
    "swir1": "SR_B6",
    "swir2": "SR_B7",
}
_TM_BAND_MAP = {
    "blue": "SR_B1",
    "green": "SR_B2",
    "red": "SR_B3",
    "nir": "SR_B4",
    "swir1": "SR_B5",
    "swir2": "SR_B7",
}

# Landsat product ID sensor prefixes → which band map applies
_OLI_PREFIXES = ("LC08", "LC09", "LO08", "LO09")
_TM_PREFIXES = ("LT04", "LT05", "LE07")


@dataclass
class LandsatScene:
    """
    A processed Landsat scene: scaled, optionally cloud-masked reflectance
    bands ready for spectral index computation or visualization.
    """

    bands: Dict[str, Any] = field(default_factory=dict)  # common_name -> float32 array
    profile: Optional[dict] = None
    cloud_mask: Optional[Any] = None  # True = masked out (cloud/shadow)
    cloud_pct: Optional[float] = None
    raw_paths: Dict[str, Path] = field(default_factory=dict)  # extracted file paths
    sensor: str = "unknown"

    def get(self, band_name: str) -> Optional[Any]:
        """Convenience accessor: scene.get('red') instead of scene.bands['red']."""
        return self.bands.get(band_name)

    @property
    def available_bands(self) -> List[str]:
        return sorted(self.bands.keys())


class LandsatExtractor:
    """
    Extract and process Landsat Collection 2 Level-2 .tar bundles.

    Example::

        from pygeofetch import PyGeoFetch
        from pygeofetch.processor import LandsatExtractor
        from pygeofetch.processor import SpectralIndex

        client = PyGeoFetch()
        results = client.download([scene], destination=out_dir)

        extractor = LandsatExtractor()
        landsat_scene = extractor.process_scene(results[0], output_dir=out_dir)

        si = SpectralIndex()
        ndvi = si.compute("NDVI", RED=landsat_scene.get("red"), NIR=landsat_scene.get("nir"))

    Two-date change detection is just two process_scene() calls::

        before = extractor.process_scene(before_result, output_dir=out_dir, label="before")
        after  = extractor.process_scene(after_result,  output_dir=out_dir, label="after")
        ndvi_change = (
            si.compute("NDVI", RED=after.get("red"),  NIR=after.get("nir"))
            - si.compute("NDVI", RED=before.get("red"), NIR=before.get("nir"))
        )
    """

    def __init__(self, mask_clouds: bool = True) -> None:
        self._mask_clouds = mask_clouds

    # ── public API ────────────────────────────────────────────────────────────

    def process_scene(
        self,
        source: Union["DownloadResult", str, Path],
        output_dir: Union[str, Path],
        bands: Optional[List[str]] = None,
        label: str = "",
        mask_clouds: Optional[bool] = None,
    ) -> LandsatScene:
        """
        Full chain: extract bundle -> scale bands -> cloud-mask -> return.

        Args:
            source:      DownloadResult from client.download() (preferred —
                        uses .output_path directly), or a direct path to
                        the downloaded .tar bundle.
            output_dir:  Where to extract band files to.
            bands:       Common band names to load. Defaults to
                        ["blue","green","red","nir","swir1","swir2"].
                        Use fewer for faster processing if you only need,
                        e.g., ["red","nir"] for NDVI.
            label:       Used to namespace the extraction subfolder when
                        processing multiple scenes into the same output_dir
                        (e.g. label="before" / label="after").
            mask_clouds: Override the instance-level mask_clouds setting.

        Returns:
            LandsatScene with scaled, optionally cloud-masked band arrays.

        Note on individual-file downloads:
            Not every provider delivers Landsat data as a single .tar
            bundle. Planetary Computer, for instance, downloads each band
            as its own separate asset file (SR_B4.TIF, SR_B5.TIF,
            QA_PIXEL.TIF, etc., each fetched individually) rather than one
            archive — DownloadResult.output_paths then holds many files,
            not one. process_scene() detects this automatically and uses
            those files directly, skipping the tar-extraction step
            entirely, so the same call works regardless of which delivery
            style the provider used.
        """
        individual_files = self._resolve_individual_files(source)

        if individual_files is not None:
            extracted = individual_files
            # Use the first file's name for sensor detection — any band
            # file's product-ID prefix works equally well for this.
            sensor_hint_name = next(iter(extracted.keys()), "")
            output_dir = Path(output_dir)
        else:
            bundle_path = self._resolve_path(source)
            if bundle_path is None:
                logger.error("Could not resolve a usable file path for this scene.")
                return LandsatScene()

            output_dir = Path(output_dir)
            extract_dir = output_dir / (f"extracted_{label}" if label else "extracted")

            extracted = self.extract_bundle(bundle_path, extract_dir)
            if not extracted:
                logger.error("No .TIF files found in %s", bundle_path.name)
                return LandsatScene()
            sensor_hint_name = bundle_path.name

        sensor = self._detect_sensor(sensor_hint_name)
        band_map = _OLI_BAND_MAP if sensor == "OLI" else _TM_BAND_MAP

        requested = bands or ["blue", "green", "red", "nir", "swir1", "swir2"]
        scene = LandsatScene(sensor=sensor, raw_paths=extracted)

        do_mask = self._mask_clouds if mask_clouds is None else mask_clouds
        cloud_mask_arr = None
        cloud_pct = None
        ref_shape = None

        if do_mask:
            qa_path = self._find_band(extracted, "QA_PIXEL")
            if qa_path:
                # Need a shape reference — peek at the first requested band
                first_band_suffix = band_map.get(requested[0])
                first_path = (
                    self._find_band(extracted, first_band_suffix)
                    if first_band_suffix
                    else None
                )
                if first_path:
                    _, ref_profile = self.load_scaled_band(first_path)
                    import rasterio

                    with rasterio.open(first_path) as src:
                        ref_shape = (src.height, src.width)
                    cloud_mask_arr = self.cloud_mask(qa_path, shape=ref_shape)
                    cloud_pct = float(100 * cloud_mask_arr.mean())
            else:
                logger.debug("No QA_PIXEL band found — skipping cloud masking.")

        for common_name in requested:
            band_suffix = band_map.get(common_name)
            if band_suffix is None:
                logger.warning(
                    "Unknown band name %r for sensor %s — skipping. Available: %s",
                    common_name,
                    sensor,
                    sorted(band_map.keys()),
                )
                continue
            band_path = self._find_band(extracted, band_suffix)
            if band_path is None:
                logger.warning(
                    "Band %s (%s) not found in extracted bundle",
                    common_name,
                    band_suffix,
                )
                continue

            data, profile = self.load_scaled_band(band_path)
            if scene.profile is None:
                scene.profile = profile

            if cloud_mask_arr is not None and data.shape == cloud_mask_arr.shape:
                import numpy as np

                data = np.where(cloud_mask_arr, np.nan, data)

            scene.bands[common_name] = data

        scene.cloud_mask = cloud_mask_arr
        scene.cloud_pct = cloud_pct

        logger.info(
            "Processed %s scene (%s): %d bands loaded%s",
            sensor,
            sensor_hint_name,
            len(scene.bands),
            f", {cloud_pct:.1f}% cloud-masked" if cloud_pct is not None else "",
        )
        return scene

    def extract_bundle(
        self, tar_path: Union[str, Path], output_dir: Union[str, Path]
    ) -> Dict[str, Path]:
        """
        Extract all band + QA GeoTIFFs from a Landsat C2L2 .tar bundle.

        Returns:
            Dict of {member_filename: extracted_path}.
        """
        tar_path = Path(tar_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted: Dict[str, Path] = {}

        if not tar_path.exists():
            logger.error("Bundle not found: %s", tar_path)
            return extracted

        try:
            with tarfile.open(tar_path) as tf:
                for member in tf.getmembers():
                    if member.name.endswith(".TIF"):
                        tf.extract(member, path=output_dir)
                        extracted[member.name] = output_dir / member.name
        except tarfile.TarError as exc:
            logger.error("Could not open %s as a tar archive: %s", tar_path.name, exc)

        return extracted

    def load_scaled_band(self, path: Union[str, Path]) -> tuple:
        """
        Load a Landsat band and apply the correct official USGS scale
        factor + offset, auto-selecting the right one based on the band
        type (surface reflectance SR_B* vs surface temperature ST_B*).

        Returns:
            (scaled_array, rasterio_profile). Fill (DN=0) pixels are NaN.
        """
        import numpy as np
        import rasterio

        path = Path(path)
        with rasterio.open(path) as src:
            dn = src.read(1).astype(np.float32)
            profile = src.profile.copy()

        if "ST_B" in path.name:
            scale, offset = ST_SCALE, ST_OFFSET
        else:
            scale, offset = SR_SCALE, SR_OFFSET

        scaled = dn * scale + offset
        scaled = np.where(dn == 0, np.nan, scaled)  # Collection 2 fill value is 0
        return scaled, profile

    def cloud_mask(
        self, qa_pixel_path: Union[str, Path], shape: Optional[tuple] = None
    ) -> Any:
        """
        Decode Landsat Collection 2 QA_PIXEL bit flags into a cloud/shadow
        mask. Returns True where the pixel should be masked OUT.

        Checks bits 1 (Dilated Cloud), 3 (Cloud), and 4 (Cloud Shadow) —
        per the official Landsat 8-9 C2 Level-2 Science Product Guide,
        Table 6-2/6-3. These bit positions are the same across the
        Landsat 4-9 Collection 2 QA_PIXEL specification.
        """
        import rasterio

        with rasterio.open(Path(qa_pixel_path)) as src:
            qa = src.read(1)
            if shape is not None and qa.shape != shape:
                qa = src.read(
                    1, out_shape=shape, resampling=rasterio.enums.Resampling.nearest
                )

        dilated_cloud = (qa & (1 << 1)) != 0
        cloud = (qa & (1 << 3)) != 0
        cloud_shadow = (qa & (1 << 4)) != 0

        return dilated_cloud | cloud | cloud_shadow

    def find_band(
        self, extracted_files: Dict[str, Path], band_suffix: str
    ) -> Optional[Path]:
        """Find the extracted path for a given band file suffix (e.g. 'SR_B4', 'QA_PIXEL')."""
        return self._find_band(extracted_files, band_suffix)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _find_band(
        self, extracted_files: Dict[str, Path], band_suffix: str
    ) -> Optional[Path]:
        for name, path in extracted_files.items():
            if band_suffix in name:
                return path
        return None

    def _detect_sensor(self, filename: str) -> str:
        """Detect OLI vs TM/ETM+ from the Landsat product ID prefix."""
        upper = filename.upper()
        if any(p in upper for p in _OLI_PREFIXES):
            return "OLI"
        if any(p in upper for p in _TM_PREFIXES):
            return "TM"
        logger.warning(
            "Could not detect sensor type from filename %r — defaulting to "
            "OLI band numbering (Landsat 8/9). If this is Landsat 4/5/7, "
            "band assignments will be wrong.",
            filename,
        )
        return "OLI"

    def _resolve_individual_files(
        self, source: Union["DownloadResult", str, Path]
    ) -> Optional[Dict[str, Path]]:
        """
        Detect whether `source` is already a set of individual band files
        (as delivered by providers like Planetary Computer, which download
        each asset — SR_B4.TIF, SR_B5.TIF, QA_PIXEL.TIF, etc. — separately
        rather than as one archive) rather than a single bundle to extract.

        Returns a dict shaped like extract_bundle()'s return value
        ({filename: path}) if this looks like the individual-files case,
        or None if `source` should instead go through the normal
        archive-extraction path.
        """
        output_paths = getattr(source, "output_paths", None)
        if not output_paths or len(output_paths) < 2:
            # A single file (or no output_paths attribute at all, e.g. a
            # plain path/string) — not the individual-files case.
            return None

        archive_exts = (".tar", ".tar.gz", ".tgz", ".zip")
        paths = [Path(p) for p in output_paths]
        if any(str(p).lower().endswith(archive_exts) for p in paths):
            # At least one entry is itself an archive — treat this as the
            # normal bundle case (resolved via _resolve_path instead).
            return None

        existing = {p.name: p for p in paths if p.exists()}
        if not existing:
            return None

        logger.info(
            "Detected %d individual band file(s) already on disk (no "
            "archive to extract) — using them directly.",
            len(existing),
        )
        return existing

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
