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
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pygeofetch.insar.annotation import SwathTiming
from pygeofetch.insar.errors import (
    AZIMUTH_COREG_REQUIREMENT_PX,
    BURST_SYNC_REQUIREMENT_MS,
    MAX_COREG_RMS_PX,
    MIN_COREG_GCPS,
    BurstProcessingError,
    CoregistrationError,
    CropError,
    FlatEarthError,
    InputError,
    ProcessingError,
    QualityRecord,
    TopographicPhaseError,
)

logger = logging.getLogger("pygeofetch.insar.interferogram")


def np_std(values):
    """Population standard deviation of a short list (no numpy import needed at module load)."""
    v = [float(x) for x in values]
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


class RegistrationFallbackWarning(UserWarning):
    """The secondary was NOT registered with orbit/DEM geometry (shape-based resampling was used).

    Phase from such a pair is usually not usable for persistent-scatterer work, which needs sub-pixel
    alignment. Raised as a warning (not only logged) so it is visible when logging is silenced.
    """


class FlatEarthWarning(UserWarning):
    """Flat-earth phase was requested or expected but could not be removed."""


class MixedSatelliteWarning(UserWarning):
    """Reference and secondary come from different Sentinel-1 satellites or relative orbits."""


def _days_between(d1, d2):
    """Whole days between two ISO date strings (absolute), or None if either is missing/unparseable."""
    try:
        from datetime import date as _date

        return abs((_date.fromisoformat(str(d2)[:10]) - _date.fromisoformat(str(d1)[:10])).days)
    except Exception:  # noqa: BLE001
        return None


def _s1_platform_and_track(path) -> tuple:
    """(platform, relative orbit) from a Sentinel-1 product name; (None, None) if not parseable."""
    name = Path(str(path)).name
    plat = name[:3] if name[:2] == "S1" else None
    try:
        absolute = int(name.split("_")[7])
    except (IndexError, ValueError):
        return plat, None
    if plat == "S1A":
        return plat, (absolute - 73) % 175 + 1
    if plat == "S1B":
        return plat, (absolute - 27) % 175 + 1
    return plat, None


# Bump whenever a change alters the numbers process_pair produces, so results saved by an
# older version are recomputed on resume instead of being silently reused.
PIPELINE_VERSION = "2026-10-29.geometry_first"
_SAVE_FORMAT = 1
_RESULT_JSON = "result.json"


def _jsonable(obj):
    """Metadata -> JSON, with numpy scalars/arrays, paths, Affine and CRS made explicit so load() can restore them."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    try:
        from affine import Affine

        if isinstance(obj, Affine):
            return {"__affine__": list(obj)[:6]}
    except ImportError:
        pass
    if hasattr(obj, "to_wkt"):
        return {"__crs__": obj.to_wkt()}
    return str(obj)


def _restore(obj):
    if isinstance(obj, dict):
        if set(obj) == {"__affine__"}:
            from affine import Affine

            return Affine(*obj["__affine__"])
        if set(obj) == {"__crs__"}:
            from rasterio.crs import CRS

            return CRS.from_wkt(obj["__crs__"])
        return {k: _restore(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore(v) for v in obj]
    return obj


def _fingerprint_value(v):
    """A stable stand-in for one input. Archives and orbit files by name (it carries the acquisition id);
    GeoTIFFs by name and raster shape (a re-extracted or re-clipped copy of the same data is the same input, whereas
    byte size and timestamps change when a file is merely rewritten); everything else by value."""
    if isinstance(v, (str, Path)):
        pth = Path(str(v))
        try:
            if pth.is_file():
                if pth.suffix.lower() in (".tif", ".tiff"):
                    try:
                        import rasterio

                        with rasterio.open(pth) as src:
                            return f"raster:{pth.name}:{src.count}x{src.height}x{src.width}:{src.dtypes[0]}"
                    except Exception:
                        return f"file:{pth.name}"
                return f"file:{pth.name}"
        except OSError:
            pass
        return str(v)
    if isinstance(v, (bool, int, float)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_fingerprint_value(x) for x in v]
    return repr(v)


def make_fingerprint(arguments: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    """What a saved result depends on: the pipeline version, the generator settings and the process_pair inputs."""
    import hashlib
    import json

    body = {
        "pipeline_version": PIPELINE_VERSION,
        "settings": {k: _fingerprint_value(v) for k, v in sorted(settings.items())},
        "arguments": {k: _fingerprint_value(v) for k, v in sorted(arguments.items())},
    }
    body["hash"] = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
    return body


# ──────────────────────────────────────────────────────────────────────────
# Streaming cleanup (verify-then-delete) support for process_pairs.
#
# These helpers never touch the InSAR maths: they only inspect what
# process_pair() has already saved and decide when an input scene is no
# longer needed.
# ──────────────────────────────────────────────────────────────────────────

# process_pair() arguments that name scene data which streaming_cleanup may
# delete once every pair using them has been saved AND verified. Orbit files
# and the DEM are small / shared and are never deleted.
_CLEANUP_INPUT_KEYS = ("reference", "secondary", "reference_safe_zip", "secondary_safe_zip")

# Written next to result.json after a verified save. It records a fingerprint of
# everything EXCEPT the scene paths, so a later resume can still recognise the pair
# as current after those scenes have been deleted (the regular fingerprint hashes
# the input files themselves, and cannot be rebuilt once they are gone).
_CLEANUP_MARKER = "streaming_cleanup.json"

# A saved pair needs at least this share of valid coherence pixels to verify.
_MIN_VALID_FRACTION = 0.01


def _verify_saved_pair(folder: Union[str, Path]) -> "tuple[bool, str]":
    """
    The *verify* step of verify-then-delete, run on what process_pair() saved.

    Checks result.json parses, wrapped_phase/coherence/amplitude GeoTIFFs open
    as one band with identical non-empty shape, interferogram_complex.npy has
    that same shape, geometry.pkl exists when result.json says it should, the
    coherence holds valid (>= 0) values that are not all zero, and the complex
    interferogram is not empty. Reads are decimated, so it is cheap on large
    pairs. Returns (ok, reason).
    """
    import json

    import numpy as np

    folder = Path(folder)
    try:
        side = json.loads((folder / _RESULT_JSON).read_text())
    except (OSError, ValueError) as exc:
        return False, f"{_RESULT_JSON} unreadable ({exc})"

    try:
        import rasterio

        shapes = {}
        coh = None
        for name in ("wrapped_phase", "coherence", "amplitude"):
            with rasterio.open(folder / f"{name}.tif") as ds:
                if ds.count != 1 or ds.height < 1 or ds.width < 1:
                    return False, f"{name}.tif has unexpected shape"
                shapes[name] = (ds.height, ds.width)
                if name == "coherence":
                    step = max(1, max(ds.height, ds.width) // 1024)
                    coh = ds.read(1, out_shape=(max(1, ds.height // step), max(1, ds.width // step)))
        if len(set(shapes.values())) != 1:
            return False, f"product shapes disagree: {shapes}"
        shape = shapes["coherence"]
        cplx = np.load(folder / "interferogram_complex.npy", mmap_mode="r")
        if tuple(cplx.shape) != tuple(shape):
            return False, f"interferogram_complex.npy shape {tuple(cplx.shape)} != {shape}"
        step = max(1, max(shape) // 1024)
        sample = np.asarray(cplx[::step, ::step])
        if side.get("has_geometry") and not (folder / "geometry.pkl").is_file():
            return False, "geometry.pkl is missing"
    except Exception as exc:  # noqa: BLE001 - any read failure means "not verified"
        return False, f"could not read saved products ({exc})"

    valid = np.isfinite(coh) & (coh >= 0)
    if valid.mean() < _MIN_VALID_FRACTION:
        return False, f"only {valid.mean():.2%} valid coherence pixels"
    if not np.any(coh[valid] > 0):
        return False, "coherence is all zero"
    if not np.any(np.isfinite(sample) & (np.abs(sample) > 0)):
        return False, "interferogram is empty"
    return True, "ok"


def _write_cleanup_marker(folder: Union[str, Path], params_hash: str) -> None:
    """Atomically record the scene-independent fingerprint of a verified pair."""
    import json

    folder = Path(folder)
    tmp = folder / (_CLEANUP_MARKER + ".tmp")
    tmp.write_text(json.dumps({"params_hash": params_hash, "pipeline_version": PIPELINE_VERSION}))
    tmp.replace(folder / _CLEANUP_MARKER)


def _read_cleanup_marker(folder: Union[str, Path]) -> Optional[Dict[str, Any]]:
    import json

    try:
        return json.loads((Path(folder) / _CLEANUP_MARKER).read_text())
    except (OSError, ValueError):
        return None


def _delete_input(path: str, protect: Path) -> bool:
    """
    Delete one input scene file, then its parent folder if that is now empty.

    Directories are never removed recursively (an input given as a folder is
    left alone with a warning), nothing inside ``protect`` (the results
    folder) is touched, and a failure only logs: the pair's results are
    already safe on disk.
    """
    import os

    p = Path(path)
    try:
        if not p.exists():
            return False
        if p.is_dir():
            logger.warning("streaming_cleanup: %s is a directory; not deleting it", p)
            return False
        if protect.resolve() in p.resolve().parents:
            logger.warning("streaming_cleanup: %s is inside the results folder; not deleting it", p)
            return False
        p.unlink()
        logger.info("streaming_cleanup: deleted %s", p.name)
        try:
            os.rmdir(p.parent)  # succeeds only when the folder is empty
        except OSError:
            pass
        return True
    except OSError as exc:
        logger.warning("streaming_cleanup: could not delete %s (%s)", p, exc)
        return False


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
    geometry: Any = None  # radar_geometry.GeometryNodes on the output grid (per-pixel incidence, lon/lat, baseline); None without orbits

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

        # Everything needed to rebuild this result exactly (the GeoTIFFs hold phase only, no complex values,
        # baselines, metadata or geometry). result.json is written LAST: its presence means the save finished.
        import json
        import pickle

        np.save(out_dir / "interferogram_complex.npy", np.asarray(self.interferogram, dtype=np.complex64))
        has_geometry = self.geometry is not None
        if has_geometry:
            with open(out_dir / "geometry.pkl", "wb") as fh:
                pickle.dump(self.geometry, fh, protocol=4)
        sidecar = {
            "format": _SAVE_FORMAT,
            "pipeline_version": PIPELINE_VERSION,
            "reference_date": self.reference_date,
            "secondary_date": self.secondary_date,
            "perpendicular_baseline_m": self.perpendicular_baseline_m,
            "temporal_baseline_days": self.temporal_baseline_days,
            "esd_azimuth_shift_px": self.esd_azimuth_shift_px,
            "has_geometry": has_geometry,
            "fingerprint": self.metadata.get("resume_fingerprint"),
            "profile": _jsonable(self.profile),
            "metadata": _jsonable(self.metadata),
        }
        tmp = out_dir / (_RESULT_JSON + ".tmp")
        tmp.write_text(json.dumps(sidecar))
        tmp.replace(out_dir / _RESULT_JSON)
        paths["result"] = out_dir / _RESULT_JSON

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

    @staticmethod
    def saved_fingerprint(directory: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """The fingerprint stored with a completed save, or None if the directory holds no complete save
        (no result.json: never finished, or written by a version that did not store everything)."""
        import json

        f = Path(directory) / _RESULT_JSON
        if not f.is_file():
            return None
        try:
            return json.loads(f.read_text()).get("fingerprint") or {}
        except (OSError, ValueError):
            return None

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "InterferogramResult":
        """Rebuild a result written by save(). Raises FileNotFoundError when the save is incomplete.

        geometry.pkl is read with pickle: only load directories that this pipeline wrote on your own machine."""
        import json
        import pickle

        import numpy as np

        d = Path(directory)
        f = d / _RESULT_JSON
        if not f.is_file():
            raise FileNotFoundError(f"{d} holds no complete saved interferogram ({_RESULT_JSON} is missing)")
        side = json.loads(f.read_text())
        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')
        with rasterio.open(d / "coherence.tif") as src:
            coherence = src.read(1)
        with rasterio.open(d / "amplitude.tif") as src:
            amplitude = src.read(1)
        interferogram = np.load(d / "interferogram_complex.npy")
        geometry = None
        if side.get("has_geometry"):
            with open(d / "geometry.pkl", "rb") as fh:
                geometry = pickle.load(fh)
        return cls(
            interferogram=interferogram, coherence=coherence, amplitude=amplitude,
            profile=_restore(side.get("profile") or {}),
            reference_date=side.get("reference_date"), secondary_date=side.get("secondary_date"),
            perpendicular_baseline_m=side.get("perpendicular_baseline_m"),
            temporal_baseline_days=side.get("temporal_baseline_days"),
            esd_azimuth_shift_px=side.get("esd_azimuth_shift_px"),
            metadata=_restore(side.get("metadata") or {}), geometry=geometry,
        )

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

        vmin: Optional[float]
        vmax: Optional[float]
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
            str(raster_path),
            colormap=colormap,
            layer_name=layer_name,
            opacity=opacity,
            vmin=vmin,
            vmax=vmax,
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
        esd_field:         After the constant ESD shift, also measure ESD per burst overlap and range block and
                           remove a smoothly varying residual misregistration (constant + range + azimuth terms).
                           TOPS pairs only (default True).
        esd_ambiguity_scan: How to pick the whole number of ESD ambiguities (~0.11 px each) that ESD cannot see:
                           "auto" (default) keeps k = 0 when cross-correlation pins the first-stage offset to
                           better than half an ambiguity and runs the coherence scan otherwise; "always" or
                           "never" force it.
        use_gpu:           Use CuPy GPU acceleration if available.
        memory_guard:      "warn" (default) | "raise" | "off". Compares the estimated peak memory of a pair with the
                           available memory before the SLCs are read; "raise" stops with MemoryError instead of
                           letting the system kill the process.
        verbosity:         None leaves logging alone. "quiet" | "normal" | "full" | "debug": how much the InSAR
                           modules log (see pygeofetch.insar.logs). "normal" drops the per-pair repeats of the
                           "Parsed real SLC geometry", "Parsed orbit state vectors" and "Plot saved" lines.
        use_real_burst_processing: Use real per-burst-overlap ESD and
                           real deburst (requires SAFE zips).
        remove_flat_earth_phase: Remove the geometric orbital/flat-earth
                           phase component via real orbit geometry.
                           ``None`` (default) = automatic: removed whenever
                           both SAFE zips and both orbit files are given,
                           which is the only way the phase is usable for
                           deformation or persistent-scatterer work. ``True``
                           = required (a warning is raised if the inputs are
                           missing or the removal fails). ``False`` = never
                           (raw reference-surface phase is kept).
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
        verbosity: Optional[str] = None,
        use_real_burst_processing: bool = False,
        remove_flat_earth_phase: Optional[bool] = None,
        chunk_size: Optional[int] = None,
        on_error: str = "raise",
        dem_geoid_offset_m: float = 0.0,
        memory_guard: str = "warn",
        esd_field: bool = True,
        esd_ambiguity_scan: str = "auto",
    ) -> None:
        # Added to every DEM height before use. Most DEMs are orthometric (Copernicus DEM: EGM2008) but the geometry
        # needs ellipsoidal heights; the difference (about +40 m in Silesia) shifts the geocoding by ~N/tan(incidence)
        # and the topographic phase by a nearly constant offset. 0.0 = use the DEM as is.
        self._dem_geoid_offset_m = float(dem_geoid_offset_m)
        # Before an SLC pair is read, compare the estimated peak memory with what the system has available.
        # "warn" logs it, "raise" stops with MemoryError before anything is allocated, "off" skips the check.
        if memory_guard not in ("off", "warn", "raise"):
            raise ValueError(f"memory_guard must be 'off', 'warn' or 'raise', got {memory_guard!r}")
        self._memory_guard = memory_guard
        # What to do when a step that was asked for (or whose inputs were supplied) fails:
        #   "raise"   (default) -- stop with a ProcessingError subclass naming the stage; the cause is chained.
        #                          This is what ISCE2, GAMMA, SNAP and HyP3 do. No weaker product is returned.
        #   "degrade"           -- keep the previous behaviour (fall back, warn, carry on), but the result is marked
        #                          quality grade "degraded" with the reason in metadata["quality"].
        # A step that cannot run because an input was never supplied (no orbit files, no DEM) is not an error in
        # either mode: it is recorded as skipped and the result is graded "quicklook".
        if on_error not in ("raise", "degrade"):
            raise ValueError(f"on_error must be 'raise' or 'degrade', got {on_error!r}")
        self._on_error = on_error
        # None means "not explicitly set" -- resolved per-call in
        # process_pair(), since the correct default depends on whether
        # multilooking was applied for that specific pair (see the real
        # bug this fixes, documented at the coherence estimation call
        # below). An explicit value here is always respected as-is.
        self._coh_window_explicit = coherence_window
        self._esd_enabled = esd_enabled
        self._esd_field_enabled = bool(esd_field)
        if esd_ambiguity_scan not in ("auto", "always", "never"):
            raise ValueError(f"esd_ambiguity_scan must be 'auto', 'always' or 'never', got {esd_ambiguity_scan!r}")
        self._esd_ambiguity_scan = esd_ambiguity_scan
        self._use_gpu = use_gpu
        if verbosity is not None:
            from pygeofetch.insar.logs import set_verbosity

            set_verbosity(verbosity)
        if use_gpu:
            # fail loudly at construction, with the reason, rather than quietly on CPU
            from pygeofetch.insar.gpu import get_array_module as _gam
            _gam(prefer_gpu=True)
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
        self._tops_ctx = None
        # Part of every pair fingerprint: results saved before TOPS deramping (0) or before resampling onto the
        # reference grid (1) are not reused. 7: SLC crops start on whole pixels (fractional crop origins left a
        # per-date misregistration of up to half a pixel, seen as along-track fringes after ESD), and orbit/DEM
        # coregistration is geometry-only by default.
        self._tops_deramp_version = 7
        # Automatic by default (None): on when the inputs exist, explicit
        # True/False always respected. It used to be opt-in and silent, which
        # left a full-width range ramp in every interferogram unless the caller
        # knew to ask for it. Removes the real, geometric orbital/
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
            if profile.get("transform") is None or getattr(
                profile.get("transform"), "is_identity", False
            ):
                gcps, gcp_crs = src.gcps
                if gcps and len(gcps) >= 4:
                    from rasterio.transform import from_gcps

                    profile["transform"] = from_gcps(gcps)
                    if gcp_crs:
                        profile["crs"] = gcp_crs
                    logger.info(
                        "Fitted approximate affine transform from %d GCPs for %s",
                        len(gcps),
                        path.name,
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
                    path.name,
                    dtype,
                    src.count,
                )
                data = src.read(1).astype(np.complex64)
                return data, profile

            # Read in chunks and concatenate
            chunks = []
            for row_start in range(0, height, chunk_rows):
                row_end = min(row_start + chunk_rows, height)
                window = Window(
                    col_off=0,
                    row_off=row_start,
                    width=width,
                    height=row_end - row_start,
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
                len(range(0, height, chunk_rows)),
                chunk_rows,
                path.name,
                data.shape,
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
            lons = [
                aoi_bbox.min_lon,
                aoi_bbox.max_lon,
                aoi_bbox.min_lon,
                aoi_bbox.max_lon,
            ]
            lats = [
                aoi_bbox.min_lat,
                aoi_bbox.min_lat,
                aoi_bbox.max_lat,
                aoi_bbox.max_lat,
            ]
            rows, cols = rowcol(transform, lons, lats)
            row_min, row_max = min(rows), max(rows)
            col_min, col_max = min(cols), max(cols)
        except Exception as exc:
            logger.warning(
                "Failed to convert AOI to pixel coordinates (%s) — "
                "returning full debursted arrays.",
                exc,
            )
            return ref_debursted, sec_debursted, profile, None

        # Ensure min < max
        if row_min > row_max:
            row_min, row_max = row_max, row_min
        if col_min > col_max:
            col_min, col_max = col_max, col_min

        # Add margin
        row_min = max(0, row_min - margin_px)
        row_max = min(
            ref_debursted.shape[0], row_max + margin_px + 1
        )  # +1 because slice end is exclusive
        col_min = max(0, col_min - margin_px)
        col_max = min(
            ref_debursted.shape[1], col_max + margin_px + 1
        )  # +1 because slice end is exclusive

        n_rows = row_max - row_min
        n_cols = col_max - col_min

        if n_rows <= 0 or n_cols <= 0:
            logger.warning(
                "AOI crop window is empty (%d x %d) — returning full "
                "debursted arrays.",
                n_rows,
                n_cols,
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
            ref_debursted.shape,
            n_rows,
            n_cols,
            row_min,
            col_min,
            margin_px,
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
        from pygeofetch.insar.gpu import get_array_module, to_numpy

        # numpy/scipy on CPU, or cupy/cupyx on the GPU when use_gpu=True and one works.
        xp, ndi, _on_gpu = get_array_module(prefer_gpu=self._use_gpu)
        self._coherence_backend = "gpu" if _on_gpu else "cpu"

        h, w = ref.shape
        coherence = np.empty((h, w), dtype=np.float32)

        # Process in chunks with overlap for the window
        half_win = window // 2

        for row_start in range(0, h, chunk_rows):
            row_end = min(row_start + chunk_rows, h)

            # Extend chunk boundaries by half the window for edge handling
            ext_start = max(0, row_start - half_win)
            ext_end = min(h, row_end + half_win)

            ref_chunk = xp.asarray(ref[ext_start:ext_end])
            sec_chunk = xp.asarray(sec[ext_start:ext_end])

            # Compute coherence for this chunk
            inter = ref_chunk * xp.conj(sec_chunk)
            num = xp.abs(
                ndi.uniform_filter(inter.real, size=window)
                + 1j * ndi.uniform_filter(inter.imag, size=window)
            )
            denom = xp.sqrt(
                ndi.uniform_filter(xp.abs(ref_chunk) ** 2, size=window)
                * ndi.uniform_filter(xp.abs(sec_chunk) ** 2, size=window)
                + 1e-10
            )
            coh_chunk = to_numpy(xp.clip(num / denom, 0.0, 1.0).astype(xp.float32))

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
            return interferogram, {
                "correction_applied": False,
                "reason": "rasterio_missing",
                "error": "rasterio_missing",
            }

        try:
            interferogram_crs = profile.get("crs")
            interferogram_transform = profile.get("transform")

            # Reproject DEM to interferogram grid (full size, unavoidable)
            with rasterio.open(dem_path) as dem_src:
                if (
                    interferogram_crs is not None
                    and interferogram_transform is not None
                ):
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

            fine_half_width = coarse[1] - coarse[0]
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
                    "topographic phase removal.",
                    r_squared,
                )
                return interferogram, {
                    "correction_applied": False,
                    "r_squared": float(r_squared),
                }

            # Apply correction in chunks
            corrected = np.empty_like(interferogram)
            for row_start in range(0, interferogram.shape[0], chunk_rows):
                row_end = min(row_start + chunk_rows, interferogram.shape[0])

                topo_phase_chunk = np.angle(
                    np.exp(1j * (best_slope * dem[row_start:row_end] + intercept))
                )
                corrected[row_start:row_end] = interferogram[
                    row_start:row_end
                ] * np.exp(-1j * topo_phase_chunk)

                del topo_phase_chunk

            logger.info(
                "Topographic phase regression R²=%.2f — correction applied (chunked)",
                r_squared,
            )

            del dem, phase, dem_v, phase_v

            return corrected.astype(np.complex64), {
                "correction_applied": True,
                "r_squared": float(r_squared),
            }

        except Exception as exc:
            logger.warning("Topographic phase removal failed: %s", exc)
            return interferogram, {"correction_applied": False, "reason": str(exc), "error": str(exc)}

    # ══════════════════════════════════════════════════════════════════
    # BURST-AWARE PROCESSING (ESD + Deburst)
    # ══════════════════════════════════════════════════════════════════

    def _burst_aware_processing(
        self,
        ref_complex,
        sec_complex,
        reference_safe_zip,
        secondary_safe_zip,
        reference_extracted_path,
        secondary_extracted_path,
        representative_row_offset: float = 0.0,
        reference_orbit_file=None,
        secondary_orbit_file=None,
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
        metadata: Dict[str, Any] = {
            "method": "whole_image_esd_fallback",
            "esd_shift_px": None,
            "deburst_applied": False,
        }

        try:
            from pygeofetch.insar.annotation import parse_burst_info, parse_slc_geometry
            from pygeofetch.insar.coregister import read_crop_offset, read_matched_swath
            from pygeofetch.insar.deburst import deburst_array, plan_deburst
            from pygeofetch.insar.esd import estimate_esd_shift_per_burst_overlap

            ref_swath_hint = (
                read_matched_swath(reference_extracted_path)
                if reference_extracted_path is not None
                else None
            )
            sec_swath_hint = (
                read_matched_swath(secondary_extracted_path)
                if secondary_extracted_path is not None
                else None
            )

            ref_geom = parse_slc_geometry(
                reference_safe_zip, member_hint=ref_swath_hint
            )
            ref_burst_info = parse_burst_info(
                reference_safe_zip, member_hint=ref_swath_hint
            )
            sec_burst_info = parse_burst_info(
                secondary_safe_zip, member_hint=sec_swath_hint
            )

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
                ", ".join(str(n) for n in common_lines),
                total_common,
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
                        sec_geom_for_sync = parse_slc_geometry(
                            secondary_safe_zip, member_hint=sec_swath_hint
                        )
                        sec_center_time = sec_geom_for_sync.azimuth_time(
                            sec_geom_for_sync.n_lines / 2
                        )

                        sync_result = compute_burst_synchronization(
                            ref_orbit,
                            sec_orbit,
                            ref_burst_info,
                            sec_burst_info,
                            ground_point,
                            ref_time_guess=ref_center_time,
                            sec_time_guess=sec_center_time,
                        )
                        metadata["burst_sync_offset_ms"] = sync_result["sync_offset_ms"]
                        metadata["burst_sync_within_requirement"] = sync_result[
                            "within_esa_requirement"
                        ]
                except Exception as exc:
                    metadata["burst_sync_error"] = str(exc)
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
                if reference_extracted_path is not None
                else (0.0, 0.0)
            )
            sec_row_off, _ = (
                read_crop_offset(secondary_extracted_path)
                if secondary_extracted_path is not None
                else (0.0, 0.0)
            )

            # Right after coregistration, before ESD/Deburst:
            min_rows = min(ref_complex.shape[0], sec_complex.shape[0])
            min_cols = min(ref_complex.shape[1], sec_complex.shape[1])

            if ref_complex.shape != (min_rows, min_cols) or sec_complex.shape != (
                min_rows,
                min_cols,
            ):
                logger.warning(
                    "Reference %s and secondary %s shapes differ — cropping both to common (%d, %d) shape before ESD.",
                    ref_complex.shape,
                    sec_complex.shape,
                    min_rows,
                    min_cols,
                )
                ref_complex = ref_complex[:min_rows, :min_cols]
                sec_complex = sec_complex[:min_rows, :min_cols]

            # Step 1: real per-burst-overlap ESD, on the still-bursted
            # (pre-deburst) data -- uses the reference's own burst
            # structure as the shared row-coordinate system for
            # comparing reference and secondary at the same real rows.
            if self._esd_enabled:
                # Doppler separation of the two looks in the overlaps, from this swath's own annotation when the
                # TOPS parameters were parsed (~4.8 kHz for IW2, not the 5.2 kHz maximum); the published
                # constant otherwise. Positive, as the ESD estimator's sign convention expects.
                _ctx = getattr(self, "_tops_ctx", None)
                _df = None
                if _ctx is not None:
                    _df = abs(_ctx["ref_ramp"].overlap_doppler_difference_hz())
                    metadata["esd_delta_f_ovl_hz"] = _df
                _esd_kwargs = {} if _df is None or not (_df > 0) else {"delta_f_ovl_hz": _df}
                esd_shift_s, per_overlap = estimate_esd_shift_per_burst_overlap(
                    ref_complex,
                    sec_complex,
                    ref_burst_info,
                    azimuth_time_interval_s,
                    row_offset=int(ref_row_off),
                    **_esd_kwargs,
                )
                if esd_shift_s is not None:
                    esd_shift_px = esd_shift_s / azimuth_time_interval_s
                    n_usable = sum(1 for s in per_overlap if s is not None)
                    if abs(esd_shift_px) > 1e-4:
                        sec_complex = self._apply_azimuth_shift(
                            sec_complex, esd_shift_px
                        )
                    logger.info(
                        "Real per-burst-overlap ESD azimuth shift: %.6f px "
                        "(%d/%d burst overlaps usable)",
                        esd_shift_px,
                        n_usable,
                        len(per_overlap),
                    )
                    metadata["esd_shift_px"] = esd_shift_px
                    metadata["esd_overlaps_usable"] = n_usable
                    metadata["esd_overlaps_total"] = len(per_overlap)
                    per_px = [s / azimuth_time_interval_s for s in per_overlap if s is not None]
                    metadata["esd_per_overlap_px"] = [round(float(v), 6) for v in per_px]
                    metadata["esd_overlap_std_px"] = float(np_std(per_px)) if len(per_px) > 1 else None
                    # Re-estimate on the corrected secondary: what is left after the shift was applied.
                    try:
                        resid_s, _ = estimate_esd_shift_per_burst_overlap(
                            ref_complex, sec_complex, ref_burst_info, azimuth_time_interval_s,
                            row_offset=int(ref_row_off), **_esd_kwargs,
                        )
                        metadata["esd_residual_after_correction_px"] = (
                            None if resid_s is None else float(resid_s / azimuth_time_interval_s)
                        )
                    except Exception as exc:  # noqa: BLE001
                        metadata["esd_residual_after_correction_px"] = None
                        logger.warning("ESD residual re-estimation failed: %s", exc)
                    # Resolve ESD's whole-ambiguity blind spot (TOPS pairs only).
                    if getattr(self, "_tops_ctx", None) is not None:
                        _amb = 1.0 / ((_df if _df else 5200.0) * azimuth_time_interval_s)
                        try:
                            _scan, _why = self._esd_scan_needed(_amb)
                            if _scan:
                                sec_complex, _amb_info = self._resolve_esd_ambiguity(ref_complex, sec_complex, _amb)
                            else:
                                _amb_info = {"ambiguity_px": float(_amb), "k": 0, "scanned": False, "reason": _why}
                                logger.info("ESD ambiguity: k = 0 kept without a coherence scan (%s)", _why)
                            if _amb_info is not None:
                                _amb_info.setdefault("scanned", True)
                                _amb_info.setdefault("reason", _why)
                                _total = esd_shift_px + _amb_info["k"] * _amb
                                if _amb_info["k"] != 0:
                                    # ESD again, now within its unambiguous range of the true alignment: the first
                                    # estimate was made far from it, where the overlaps decorrelate and bias ESD.
                                    _r2, _ = estimate_esd_shift_per_burst_overlap(
                                        ref_complex, sec_complex, ref_burst_info, azimuth_time_interval_s,
                                        row_offset=int(ref_row_off), **_esd_kwargs,
                                    )
                                    if _r2 is not None and abs(_r2 / azimuth_time_interval_s) > 1e-4:
                                        sec_complex = self._apply_azimuth_shift(sec_complex, _r2 / azimuth_time_interval_s)
                                        _total += _r2 / azimuth_time_interval_s
                                        _amb_info["esd_refinement_px"] = float(_r2 / azimuth_time_interval_s)
                                metadata["esd_ambiguity"] = _amb_info
                                metadata["esd_total_shift_px"] = _total
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("ESD ambiguity scan failed: %s", exc)
                        # What a single constant cannot remove: misregistration varying in range and between
                        # overlaps (steps at burst seams, ramps inside bursts).
                        if getattr(self, "_esd_field_enabled", True):
                            try:
                                sec_complex, metadata["esd_field"] = self._esd_field_refinement(
                                    ref_complex, sec_complex, ref_burst_info, azimuth_time_interval_s,
                                    int(ref_row_off), _amb, _esd_kwargs)
                            except Exception as exc:  # noqa: BLE001
                                logger.warning("ESD field refinement failed: %s", exc)
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
            # Plan both debursts first (cheap, raises before any data is touched), then compact each image in
            # place: kept rows move toward the top of the array that already exists, so no second full-size
            # copy of either image is made. Both inputs are replaced by the return values in process_pair.
            #
            # FIX: Use ref_burst_info for the secondary as well. sec_complex was already resampled onto the
            # reference's grid by the swath-level coregistration, so its burst boundaries in pixel
            # coordinates match the reference's; the secondary's own un-warped timing would cut the wrong
            # rows and destroy coherence at burst edges.
            _ref_spans, ref_first_kept_row = plan_deburst(
                ref_complex.shape[0], ref_burst_info, azimuth_time_interval_s, row_offset=int(ref_row_off))
            _sec_spans, _sec_first = plan_deburst(
                sec_complex.shape[0], ref_burst_info, azimuth_time_interval_s, row_offset=int(ref_row_off))
            ref_debursted, _ = deburst_array(
                ref_complex, ref_burst_info, azimuth_time_interval_s, row_offset=int(ref_row_off),
                inplace=True, spans=_ref_spans, first_kept_row=ref_first_kept_row)
            ref_complex = None
            sec_debursted, _ = deburst_array(
                sec_complex, ref_burst_info, azimuth_time_interval_s, row_offset=int(ref_row_off),
                inplace=True, spans=_sec_spans, first_kept_row=_sec_first)
            sec_complex = None

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
            metadata["deburst_first_kept_row"] = int(ref_first_kept_row)

            if ref_debursted.shape != sec_debursted.shape:
                min_rows = min(ref_debursted.shape[0], sec_debursted.shape[0])
                min_cols = min(ref_debursted.shape[1], sec_debursted.shape[1])
                logger.warning(
                    "Debursted reference %s and secondary %s shapes "
                    "differ (different dates' real burst timing need "
                    "not match exactly) — cropping both to the common "
                    "(%d, %d) shape.",
                    ref_debursted.shape,
                    sec_debursted.shape,
                    min_rows,
                    min_cols,
                )
                ref_debursted = ref_debursted[:min_rows, :min_cols]
                sec_debursted = sec_debursted[:min_rows, :min_cols]

            metadata["method"] = "real_per_burst_esd_and_deburst"
            metadata["deburst_applied"] = True
            logger.info(
                "Real burst-aware processing complete: %s -> %s "
                "(burst-boundary redundancy removed)",
                ref_debursted.base.shape if ref_debursted.base is not None else ref_debursted.shape,
                ref_debursted.shape,
            )
            return ref_debursted, sec_debursted, metadata

        except Exception as exc:
            if self._on_error == "raise":
                raise BurstProcessingError(
                    "burst_processing", f"burst-aware ESD/deburst failed: {exc}"
                ) from exc
            metadata["burst_processing_failed"] = str(exc)
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
                        "Whole-image ESD azimuth shift applied (fallback): %.5f px",
                        esd_shift,
                    )
            metadata["esd_shift_px"] = esd_shift
            return ref_complex, sec_complex, metadata

    # ══════════════════════════════════════════════════════════════════
    # MAIN PROCESSING PIPELINE
    # ══════════════════════════════════════════════════════════════════

    _FINGERPRINT_EXCLUDED_ARGS = frozenset({
        "self", "chunk_rows", "use_chunked_processing", "output_dir", "resume", "save_visualizations",
        "min_coreg_gcps", "max_coreg_rms_px", "max_burst_sync_ms"})
    _FINGERPRINT_EXCLUDED_SETTINGS = frozenset({"_use_gpu", "_chunk_size", "_on_error", "_coherence_backend", "_memory_guard",
                                                "_tops_ctx"})

    def _pair_fingerprint(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """What a saved result depends on, from the process_pair arguments (all of them, defaults included)."""
        args = {k: v for k, v in arguments.items()
                if k not in self._FINGERPRINT_EXCLUDED_ARGS and not k.startswith("_")}
        settings = {k: v for k, v in vars(self).items()
                    if isinstance(v, (bool, int, float, str, type(None)))
                    and k not in self._FINGERPRINT_EXCLUDED_SETTINGS}
        return make_fingerprint(args, settings)

    def pair_fingerprint(self, **process_pair_kwargs) -> Dict[str, Any]:
        """The fingerprint process_pair would store for these arguments, without running anything. Used to
        decide up front whether a pair on disk is current."""
        import inspect

        bound = inspect.signature(self.process_pair).bind(**process_pair_kwargs)
        bound.apply_defaults()
        return self._pair_fingerprint(dict(bound.arguments))

    def is_pair_current(self, output_dir, reference_date: str, secondary_date: str, **process_pair_kwargs) -> bool:
        """True when output_dir/<ref>_<sec> holds a complete save made with exactly these inputs and settings."""
        saved = InterferogramResult.saved_fingerprint(Path(output_dir) / f"{reference_date}_{secondary_date}")
        if not saved:
            return False
        return saved.get("hash") == self.pair_fingerprint(
            reference_date=reference_date, secondary_date=secondary_date, **process_pair_kwargs)["hash"]

    def _pair_params_hash(self, kw: Dict[str, Any]) -> str:
        """Fingerprint hash of a pair's dates, settings and non-scene inputs (orbit files, DEM, options),
        deliberately excluding the scene paths that streaming_cleanup deletes. See _CLEANUP_MARKER."""
        import inspect

        bound = inspect.signature(self.process_pair).bind(**kw)
        bound.apply_defaults()
        args = {k: v for k, v in bound.arguments.items() if k not in _CLEANUP_INPUT_KEYS}
        return self._pair_fingerprint(args)["hash"]

    def _resume_state(self, out: Path, kw: Dict[str, Any], verify: bool):
        """Decide whether one pair can be skipped. Returns (current, reason, via_marker).

        The regular fingerprint check is tried first. If it fails only because streaming_cleanup already
        deleted the scenes, the marker written at the time of cleanup is accepted instead, provided its
        scene-independent hash still matches. A pair is only called current when its saved products also
        verify (always when scenes are gone or verify=True; the plain resume path is unchanged)."""
        rd, sd = kw["reference_date"], kw["secondary_date"]
        folder = out / f"{rd}_{sd}"
        pair_kw = {k: v for k, v in kw.items() if k not in ("reference_date", "secondary_date")}
        current = self.is_pair_current(out, rd, sd, **pair_kw)
        via_marker = False
        if not current:
            scenes_gone = any(kw.get(k) is not None and not Path(str(kw[k])).exists()
                              for k in _CLEANUP_INPUT_KEYS)
            marker = _read_cleanup_marker(folder) if scenes_gone else None
            if marker and marker.get("params_hash") == self._pair_params_hash(kw):
                current = via_marker = True
        if current and (verify or via_marker):
            ok, why = _verify_saved_pair(folder)
            if not ok:
                return False, f"saved products failed verification ({why})", via_marker
        return current, "", via_marker

    def process_pairs(
        self,
        pairs,
        output_dir,
        resume: bool = True,
        keep_results: bool = False,
        stop_on_error: bool = False,
        streaming_cleanup: bool = False,
        keep_inputs=None,
        **common_kwargs,
    ):
        """Process many pairs one after another, saving each under output_dir/<ref>_<sec>.

        pairs: iterable of dicts of process_pair arguments for one pair (reference, secondary,
            reference_date, secondary_date, ... ). common_kwargs apply to every pair; a pair's own dict wins.
        resume: pairs whose folder already holds a complete save made with the same inputs and settings
            are skipped before any SLC is opened (checked with the same fingerprint process_pair stores).
            With streaming_cleanup the saved products are also verified first, and a pair whose scenes
            were deleted by an earlier cleanup is still recognised (see _CLEANUP_MARKER).
        keep_results: False (default) keeps only a small record per pair, so memory stays flat however many
            pairs run; True also returns each InterferogramResult (loaded from disk for skipped pairs).
        stop_on_error: False records the failure and carries on with the next pair.
        streaming_cleanup: False (default) changes nothing. True applies verify-then-delete to the
            INPUTS of each pair (reference / secondary extracted SLCs and reference_safe_zip /
            secondary_safe_zip):
              try     process_pair() as before, saving to output_dir/<ref>_<sec>;
              verify  re-open the saved products (shape, validity, not empty);
              delete  an input is deleted only once EVERY pair in this call that uses it has been saved
                      and verified, so a scene shared by several pairs survives until its last pair;
              except  the error is recorded and the inputs of that pair are KEPT (so is any scene the
                      failed pair shares with others);
              finally gc.collect().
            Deleted inputs cannot be recomputed from: to redo a pair, re-download its scenes. Pairs in other
            calls are not known to this one, so either pass all the pairs that need a scene in ONE call, or
            name the scenes a LATER call still needs in keep_inputs.
            Orbit files and the DEM are never deleted.
        keep_inputs: paths (extracted SLCs or zips) that streaming_cleanup must not delete in this call. For
            a rolling run, one pair per call, pass the scenes the next pair shares with this one; they are
            then freed by the call that finishes their last pair.

        Returns a list of records: reference_date, secondary_date, status ("skipped: already generated",
        "computed", "failed"), folder, metadata (the pair's summary, loaded from result.json when skipped),
        error, deleted (inputs removed after this pair, streaming_cleanup only) and result (only with
        keep_results). The decision for every pair is printed, so a run that did not skip anything says so.
        """
        import gc
        import json
        import os

        out = Path(output_dir)
        pairs = list(pairs)
        records = []
        n_skip = n_done = n_fail = 0
        print(f"process_pairs: {len(pairs)} pairs, output_dir={out}, resume={resume}"
              + (", streaming_cleanup=True" if streaming_cleanup else ""))

        specs = []
        for spec in pairs:
            kw = {**common_kwargs, **spec}
            if not (kw.get("reference_date") and kw.get("secondary_date")):
                raise InputError("input", "every pair needs reference_date and secondary_date")
            specs.append(kw)

        # Streaming bookkeeping. pending[input] = indices of pairs that still need that input; an input
        # becomes deletable when its set is empty. A pair leaves the sets only once its results are
        # saved and verified, so a failed pair pins every input it uses.
        pending: Dict[str, set] = {}
        states: Dict[int, Any] = {}
        if streaming_cleanup:
            for i, kw in enumerate(specs):
                for key in _CLEANUP_INPUT_KEYS:
                    if kw.get(key) is not None:
                        pending.setdefault(os.path.abspath(str(kw[key])), set()).add(i)

        for held in (keep_inputs or ()) if streaming_cleanup else ():
            held = os.path.abspath(str(held))
            if held in pending:
                pending[held].add(-1)      # -1 is never finished: this call leaves the file alone

        def _finish(i: int) -> list:
            """Pair i is saved and verified: release its inputs and delete those nobody else needs."""
            deleted = []
            for key in _CLEANUP_INPUT_KEYS:
                if specs[i].get(key) is None:
                    continue
                path = os.path.abspath(str(specs[i][key]))
                pending[path].discard(i)
                if not pending[path] and _delete_input(path, protect=out):
                    deleted.append(path)
            return deleted

        if streaming_cleanup and resume:
            # Decide every pair up front: pairs that are already done release their inputs straight away.
            for i, kw in enumerate(specs):
                try:
                    states[i] = self._resume_state(out, kw, verify=True)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("resume check failed for %s_%s (%s); computing it",
                                   kw["reference_date"], kw["secondary_date"], exc)
            for i, st in states.items():
                if st[0]:
                    _finish(i)

        for i, (kw, spec) in enumerate(zip(specs, pairs), 1):
            rd, sd = kw["reference_date"], kw["secondary_date"]
            folder = out / f"{rd}_{sd}"
            rec = {"reference_date": rd, "secondary_date": sd, "folder": folder, "status": None,
                   "metadata": None, "error": None, "result": None}
            if streaming_cleanup:
                rec["deleted"] = []
            tag = f"[{i}/{len(pairs)}] {rd} -> {sd}"
            current, why_not = False, ""
            try:
                if resume:
                    try:
                        # Plain resume keeps its original cost; verification is added only for streaming runs
                        # (or when the scenes are gone and the marker is all there is to go on).
                        current, why_not, _ = states.get(i - 1) or self._resume_state(
                            out, kw, verify=streaming_cleanup)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("resume check failed for %s_%s (%s); computing it", rd, sd, exc)
                if current:
                    rec["status"] = "skipped: already generated"
                    try:
                        rec["metadata"] = _restore(json.loads((folder / _RESULT_JSON).read_text()).get("metadata") or {})
                        if keep_results:
                            rec["result"] = InterferogramResult.load(folder)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("could not read saved result for %s_%s (%s)", rd, sd, exc)
                    n_skip += 1
                    print(f"{tag}: {rec['status']}")
                else:
                    why = why_not or ("resume off" if not resume else (
                        "no complete save" if InterferogramResult.saved_fingerprint(folder) is None
                        else "saved with different inputs/settings/version"))
                    print(f"{tag}: computing ({why})")
                    try:
                        res = self.process_pair(output_dir=out, resume=False, **kw)
                        rec["metadata"] = dict(res.metadata)
                        if keep_results:
                            rec["result"] = res
                        del res
                        if streaming_cleanup:
                            ok, bad = _verify_saved_pair(folder)
                            if not ok:
                                raise RuntimeError(f"saved products failed verification ({bad}); "
                                                   f"input scenes kept")
                            _write_cleanup_marker(folder, self._pair_params_hash(kw))
                            rec["deleted"] = _finish(i - 1)
                        rec["status"] = "computed"
                        n_done += 1
                    except Exception as exc:  # noqa: BLE001
                        rec["status"] = "failed"
                        rec["error"] = f"{type(exc).__name__}: {exc}"
                        if streaming_cleanup:
                            logger.error("%s: %s -- inputs of this pair kept", tag, rec["error"])
                        n_fail += 1
                        print(f"{tag}: FAILED {rec['error']}")
                        if stop_on_error:
                            records.append(rec)
                            raise
                records.append(rec)
            finally:
                gc.collect()
        print(f"process_pairs: {n_skip} skipped, {n_done} computed, {n_fail} failed")
        return records

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
        coregistration_refine_by_coherence: Optional[bool] = None,
        coregistration_degree: int = 1,
        coregistration_rms_threshold: Optional[float] = None,
        coregistration_method: str = "orbit_dem",
        coregistration_window: Optional[int] = None,
        coregistration_coarse_search_radius: Optional[int] = None,
        coregistration_fine_search_radius: Optional[float] = None,
        coregistration_coherence_threshold: Optional[float] = None,
        coregistration_fit: str = "geometry_first",
        # ═══ NEW SNAP-STYLE PARAMETERS ═══
        aoi_bbox=None,
        crop_after_deburst: bool = True,
        use_chunked_processing: bool = True,
        chunk_rows: Optional[int] = None,
        min_coreg_gcps: Optional[int] = None,
        max_coreg_rms_px: Optional[float] = None,
        max_burst_sync_ms: Optional[float] = None,
        output_dir: Optional[Union[str, Path]] = None,
        resume: bool = False,
        save_visualizations: bool = True,
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
            coregistration_refine_by_coherence: If True, the
                       orbit/DEM (or collocation) offset field is
                       refined against the actual image content via
                       cross-correlation + Powell coherence maximization
                       before the warp polynomial is fit. None (default)
                       chooses by method: False for "orbit_dem" (geometry
                       only, as ISCE2 topsApp and ASF HyP3 do: the orbit
                       and DEM give the offsets to a few hundredths of a
                       pixel, and on 36 Mexico City pairs geometry-only
                       registration agreed with HyP3 at median 0.90 with
                       triplet closure 0.07 rad against HyP3's 0.09),
                       True for "raster_collocation", whose geocoding
                       is too coarse to stand alone.
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
            coregistration_fit: "geometry_first" (default) builds the offset
                       field from the orbit/DEM geometry and corrects it
                       by only as much as the cross-correlation points can
                       support (linear with >= 8, constant with 2-7, none
                       otherwise; flagged coreg_geometry_only). "polynomial"
                       is the previous behaviour: a free polynomial through
                       the surviving cross-correlation points only.
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
            min_coreg_gcps, max_coreg_rms_px: Hard limits on the coregistration fit. When set and
                       violated the pair is rejected with CoregistrationError. When left None the
                       fit is only *reported* against the package's own reliability limits (6 GCPs,
                       1.0 px RMS) and the result is flagged, not rejected.
            max_burst_sync_ms: Hard limit on the burst-timing offset of the pair (ESA requires under
                       5 ms for TOPS interferometry). When set and exceeded the pair is rejected with
                       BurstProcessingError; when None it is reported and flagged only.
            output_dir: Parent folder. The pair is saved to ``output_dir/<reference_date>_<secondary_date>``
                       (GeoTIFFs, PNGs and everything needed to reload it) as soon as it is computed.
                       Needs reference_date and secondary_date.
            resume: With output_dir, load the pair from disk instead of recomputing it, but only when the saved
                       result carries the same fingerprint (pipeline version, generator settings, inputs and
                       process_pair arguments). A folder with no complete save, or from a different pipeline
                       version or settings, is recomputed and overwritten. Results saved before this feature
                       existed are never reused.
            save_visualizations: With output_dir, also write the PNG plots.

        Returns:
            InterferogramResult with wrapped phase, coherence, and amplitude. ``metadata["quality"]``
            holds what was done and not done, each check, and the grade (science / quicklook /
            degraded); see ``pygeofetch.insar.errors``.

        Raises:
            ProcessingError subclasses (all ValueError) when a step that was asked for fails, unless the
            generator was built with ``on_error="degrade"``.
        """
        # What this result depends on, taken before anything else is assigned. Used to decide on resume
        # whether a saved result was made by the same pipeline with the same inputs and settings.
        _fingerprint = self._pair_fingerprint(dict(locals()))
        _pair_dir = None
        _resume_note = "resume not requested" if not resume else "no output_dir"
        if output_dir is not None:
            if not (reference_date and secondary_date):
                raise InputError("input", "output_dir needs reference_date and secondary_date to name the pair folder")
            _pair_dir = Path(output_dir) / f"{reference_date}_{secondary_date}"
            if resume:
                _saved = InterferogramResult.saved_fingerprint(_pair_dir)
                if _saved is None:
                    _resume_note = "computed: no complete saved result in the folder"
                    logger.info("resume: %s_%s has no complete saved result; computing it", reference_date, secondary_date)
                elif _saved.get("hash") == _fingerprint["hash"]:
                    logger.info("resume: %s_%s already generated with the same inputs and settings; loaded from disk",
                                reference_date, secondary_date)
                    _loaded = InterferogramResult.load(_pair_dir)
                    _loaded.metadata["resumed_from_disk"] = True
                    _loaded.metadata["resume_status"] = "loaded from disk"
                    return _loaded
                else:
                    _changed = [f"{grp}.{k}" for grp in ("settings", "arguments")
                                for k in sorted(set(_saved.get(grp, {})) | set(_fingerprint[grp]))
                                if _saved.get(grp, {}).get(k) != _fingerprint[grp].get(k)]
                    if _saved.get("pipeline_version") != PIPELINE_VERSION:
                        _changed.insert(0, f"pipeline_version ({_saved.get('pipeline_version')} -> {PIPELINE_VERSION})")
                    _resume_note = "computed: saved result differs in " + (", ".join(_changed) or "unknown")
                    logger.warning("resume: %s_%s was saved with different inputs, settings or pipeline version "
                                   "(changed: %s); recomputing it", reference_date, secondary_date,
                                   ", ".join(_changed) or "unknown")

        quality = QualityRecord(on_error=self._on_error)
        # TOPS deramp context of this pair (set by _orbit_based_coregister, used by the ESD shift).
        self._tops_ctx = None

        # Resolve chunk size: explicit param > instance default > None (full)
        effective_chunk = chunk_rows if chunk_rows is not None else self._chunk_size

        from pygeofetch.insar.memory import check_memory

        _need_gb, _have_gb = check_memory(
            reference, secondary, chunked=bool(use_chunked_processing and effective_chunk is not None),
            mode=self._memory_guard)
        if _need_gb is not None and _have_gb is not None:
            logger.info("memory: estimated peak %.1f GB, %.1f GB available", _need_gb, _have_gb)

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

        for _arr, _nm in ((ref_complex, "reference SLC"), (sec_complex, "secondary SLC")):
            try:
                DataValidator.validate_slc(_arr, name=_nm).raise_if_invalid()
            except ProcessingError:
                raise
            except Exception as exc:
                raise InputError("input", f"{_nm} failed validation: {exc}") from exc
        quality.ok("input", f"reference {ref_complex.shape}, secondary {sec_complex.shape}")

        # Platform/sub-swath mismatch warnings
        pair_platforms: Dict[str, Any] = {
            "reference_platform": None, "secondary_platform": None,
            "reference_track": None, "secondary_track": None,
            "cross_satellite": None, "cross_track": None,
        }
        if reference_safe_zip is not None and secondary_safe_zip is not None:
            ref_platform, ref_track = _s1_platform_and_track(reference_safe_zip)
            sec_platform, sec_track = _s1_platform_and_track(secondary_safe_zip)
            pair_platforms.update(
                reference_platform=ref_platform, secondary_platform=sec_platform,
                reference_track=ref_track, secondary_track=sec_track,
                cross_satellite=(ref_platform != sec_platform
                                 if ref_platform and sec_platform else None),
                cross_track=(ref_track != sec_track
                             if ref_track is not None and sec_track is not None else None),
            )
            if pair_platforms["cross_satellite"]:
                msg = (
                    f"Reference ({ref_platform}) and secondary ({sec_platform}) scenes are "
                    "from different Sentinel-1 satellites -- burst boundaries for the same "
                    "ground area are not guaranteed to align between S1A and S1B. Check "
                    "burst_sync_offset_ms in the result."
                )
                logger.warning(msg)
                warnings.warn(msg, MixedSatelliteWarning, stacklevel=2)
            if pair_platforms["cross_track"]:
                msg = (
                    f"Reference (track {ref_track}) and secondary (track {sec_track}) are on "
                    "different relative orbits: they do not view the area from the same geometry."
                )
                logger.warning(msg)
                warnings.warn(msg, MixedSatelliteWarning, stacklevel=2)
            ref_swath = sec_swath = None
            try:
                from pygeofetch.insar.coregister import read_matched_swath

                ref_swath = read_matched_swath(reference) if reference is not None else None
                sec_swath = read_matched_swath(secondary) if secondary is not None else None
            except Exception as exc:  # noqa: BLE001
                quality.flag("swath_label_unreadable")
                logger.warning("Could not read the sub-swath labels of the SLCs (%s); the same-swath check was skipped.", exc)
            if ref_swath and sec_swath and ref_swath != sec_swath:
                # Different sub-swaths see different ground: there is no common area to interfere.
                if self._on_error == "raise":
                    raise InputError(
                        "input", f"reference was extracted from {ref_swath} and secondary from {sec_swath}",
                        {"reference_swath": ref_swath, "secondary_swath": sec_swath},
                    )
                quality.failed("input", f"different sub-swaths {ref_swath} / {sec_swath}", "different_subswaths")
            if pair_platforms["cross_satellite"] or pair_platforms["cross_track"]:
                quality.flag("cross_satellite" if pair_platforms["cross_satellite"] else "cross_track")

        # ── Step 1: Coregistration ──
        if coregistration_refine_by_coherence is None:
            coregistration_refine_by_coherence = coregistration_method != "orbit_dem"
        if coregistration_method == "raster_collocation":
            logger.warning(
                "coregistration_method='raster_collocation' resamples the secondary without TOPS deramping: "
                "for Sentinel-1 IW/EW bursts this corrupts the phase (fringes along azimuth, jumps at burst "
                "edges). Use coregistration_method='orbit_dem' for interferometry."
            )
            ref_has_geocoding = (
                profile.get("crs") is not None and profile.get("transform") is not None
            )
            sec_has_geocoding = (
                sec_profile.get("crs") is not None
                and sec_profile.get("transform") is not None
            )
            if ref_has_geocoding and sec_has_geocoding:
                sec_complex, coreg_metadata = self._raster_collocation_coregister(
                    ref_complex,
                    sec_complex,
                    profile,
                    sec_profile,
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
                    sec_complex = self._resample_to_reference(
                        sec_complex, ref_complex.shape
                    )
                coreg_metadata = {
                    "method": "shape_based_fallback_missing_geocoding",
                    "refined_by_coherence": False,
                    "n_gcps_initial": None,
                    "n_gcps_final": None,
                    "rms_mean_px": None,
                    "mean_coherence": None,
                }
                quality.skipped("coregistration", "raster collocation needs a CRS and transform on both SLCs; "
                                "shape-based resample only", flag="not_orbit_registered")
        elif coregistration_method == "orbit_dem":
            real_coreg_inputs = (
                dem,
                reference_safe_zip,
                secondary_safe_zip,
                reference_orbit_file,
                secondary_orbit_file,
            )
            if all(x is not None for x in real_coreg_inputs):
                sec_complex, coreg_metadata = self._orbit_based_coregister(
                    ref_complex,
                    sec_complex,
                    dem,
                    reference_safe_zip,
                    secondary_safe_zip,
                    reference_orbit_file,
                    secondary_orbit_file,
                    reference,
                    secondary,
                    refine_by_coherence=coregistration_refine_by_coherence,
                    degree=coregistration_degree,
                    rms_threshold=coregistration_rms_threshold,
                    window=coregistration_window,
                    coarse_search_radius=coregistration_coarse_search_radius,
                    fine_search_radius=coregistration_fine_search_radius,
                    coherence_threshold=coregistration_coherence_threshold,
                    fit=coregistration_fit,
                )
            else:
                logger.info(
                    "Using shape-based coregistration fallback (real orbit-"
                    "based coregistration needs dem + both SAFE zips + both "
                    "orbit files; not all were supplied)."
                )
                if sec_complex.shape != ref_complex.shape:
                    sec_complex = self._resample_to_reference(
                        sec_complex, ref_complex.shape
                    )
                coreg_metadata = {
                    "method": "shape_based_fallback",
                    "refined_by_coherence": False,
                    "n_gcps_initial": None,
                    "n_gcps_final": None,
                    "rms_mean_px": None,
                    "mean_coherence": None,
                }
                _missing = [n for n, x in zip(
                    ("dem", "reference_safe_zip", "secondary_safe_zip", "reference_orbit_file", "secondary_orbit_file"),
                    real_coreg_inputs) if x is None]
                quality.skipped("coregistration", "orbit/DEM inputs not supplied (" + ", ".join(_missing) +
                                "); shape-based resample only", flag="not_orbit_registered", missing=_missing)
        else:
            raise ValueError(
                f"coregistration_method must be 'orbit_dem' or "
                f"'raster_collocation', got {coregistration_method!r}"
            )

        # Shape reconciliation before ESD
        min_rows = min(ref_complex.shape[0], sec_complex.shape[0])
        min_cols = min(ref_complex.shape[1], sec_complex.shape[1])
        if ref_complex.shape != (min_rows, min_cols) or sec_complex.shape != (
            min_rows,
            min_cols,
        ):
            logger.warning(
                "Reference %s and secondary %s shapes differ — cropping to common (%d, %d).",
                ref_complex.shape,
                sec_complex.shape,
                min_rows,
                min_cols,
            )
            ref_complex = ref_complex[:min_rows, :min_cols]
            sec_complex = sec_complex[:min_rows, :min_cols]

        coreg_fallback = "fallback" in str(coreg_metadata.get("method") or "")
        if coreg_fallback and quality.status_of("coregistration") is None:
            # only reachable with on_error="degrade": a requested registration failed
            quality.failed("coregistration", str(coreg_metadata.get("error") or "registration fell back"),
                           "registration_fallback")
        if not coreg_fallback:
            _n = coreg_metadata.get("n_gcps_final")
            _r = coreg_metadata.get("rms_mean_px")
            _lim_n = min_coreg_gcps if min_coreg_gcps is not None else MIN_COREG_GCPS
            _lim_r = max_coreg_rms_px if max_coreg_rms_px is not None else MAX_COREG_RMS_PX
            _mode = coreg_metadata.get("coreg_mode")
            if _mode is not None:
                _bad = []
                if min_coreg_gcps is not None and _n < min_coreg_gcps:
                    _bad.append(f"{_n} supporting GCPs (limit {min_coreg_gcps})")
                if max_coreg_rms_px is not None and _n and _r > max_coreg_rms_px:
                    _bad.append(f"RMS {_r:.3f} px (limit {max_coreg_rms_px})")
                if _bad:
                    raise CoregistrationError(
                        "coregistration", "fit rejected: " + "; ".join(_bad),
                        {"n_gcps_final": _n, "rms_mean_px": _r, "coreg_mode": _mode})
                if _mode == "geometry_only":
                    quality.flag("coreg_geometry_only")
                elif _n < 3 or (_r is not None and _r > MAX_COREG_RMS_PX):
                    quality.flag("coreg_unreliable")
                quality.ok("coregistration", f"{_mode}: {_n} supporting cross-correlation GCPs"
                           + (f", RMS {_r:.3f} px" if _n else "; image content did not constrain it")
                           + f" [{coreg_metadata.get('n_geometry_points')} orbit/DEM points]",
                           n_gcps_final=_n, rms_mean_px=_r)
            elif _n is None or _r is None:
                quality.flag("coreg_quality_unknown")
                quality.ok("coregistration", "registered; fit quality not reported by this method")
            else:
                _bad = []
                if _n < _lim_n:
                    _bad.append(f"{_n} GCPs after rejection (limit {_lim_n})")
                if _r > _lim_r:
                    _bad.append(f"RMS {_r:.3f} px (limit {_lim_r})")
                _hard = (min_coreg_gcps is not None and _n < min_coreg_gcps) or \
                        (max_coreg_rms_px is not None and _r > max_coreg_rms_px)
                if _bad and _hard:
                    raise CoregistrationError(
                        "coregistration", "fit rejected: " + "; ".join(_bad),
                        {"n_gcps_final": _n, "rms_mean_px": _r})
                if _bad:
                    quality.flag("coreg_unreliable")
                quality.ok("coregistration", ("; ".join(_bad) if _bad else "fit within limits") +
                           f" [{_n} GCPs, RMS {_r:.3f} px]", n_gcps_final=_n, rms_mean_px=_r)
        if coreg_fallback:
            msg = (
                f"Registration fell back to '{coreg_metadata.get('method')}': the secondary was NOT "
                "registered with orbit/DEM geometry (it needs the DEM, both SAFE zips and both orbit "
                "files, and must succeed). Phase from this pair is not suitable for persistent-scatterer "
                "or deformation work."
            )
            logger.warning(msg)
            warnings.warn(msg, RegistrationFallbackWarning, stacklevel=2)

        # ── Step 2: ESD + Deburst (on FULL data, burst overlaps intact) ──
        esd_shift = None
        burst_metadata: Dict[str, Any] = {
            "method": "whole_image_esd",
            "deburst_applied": False,
        }

        if (
            self._use_real_burst_processing
            and reference_safe_zip is not None
            and secondary_safe_zip is not None
        ):
            _rep_row_offset = coreg_metadata.get("representative_row_offset_px", 0.0)
            ref_complex, sec_complex, burst_metadata = self._burst_aware_processing(
                ref_complex,
                sec_complex,
                reference_safe_zip,
                secondary_safe_zip,
                reference,
                secondary,
                representative_row_offset=_rep_row_offset,
                reference_orbit_file=reference_orbit_file,
                secondary_orbit_file=secondary_orbit_file,
            )
            esd_shift = burst_metadata.get("esd_shift_px")
            rows_removed = burst_metadata.get("rows_removed_from_top", 0)

            if burst_metadata.get("burst_processing_failed"):
                quality.failed("burst_processing", burst_metadata["burst_processing_failed"], "burst_processing_failed")
            else:
                quality.ok("burst_processing", f"{burst_metadata.get('method')}; deburst "
                           f"{'applied' if burst_metadata.get('deburst_applied') else 'not applied'}")
            _sync = burst_metadata.get("burst_sync_offset_ms")
            if _sync is None:
                quality.flag("burst_sync_unknown")
            else:
                _lim = max_burst_sync_ms if max_burst_sync_ms is not None else BURST_SYNC_REQUIREMENT_MS
                if _sync > _lim:
                    if max_burst_sync_ms is not None:
                        raise BurstProcessingError(
                            "burst_timing", f"burst synchronisation offset {_sync:.2f} ms exceeds {max_burst_sync_ms} ms",
                            {"burst_sync_offset_ms": _sync})
                    quality.flag("burst_sync_exceeds_requirement")
            _usable, _total = burst_metadata.get("esd_overlaps_usable"), burst_metadata.get("esd_overlaps_total")
            if self._esd_enabled and burst_metadata.get("esd_shift_px") is None and not burst_metadata.get("burst_processing_failed"):
                quality.flag("esd_not_estimated")
            _res = burst_metadata.get("esd_residual_after_correction_px")
            if _res is not None and abs(_res) > AZIMUTH_COREG_REQUIREMENT_PX:
                quality.flag("azimuth_misregistration_above_requirement")
            _sd = burst_metadata.get("esd_overlap_std_px")
            if _sd is not None and _sd > AZIMUTH_COREG_REQUIREMENT_PX:
                quality.flag("esd_overlaps_inconsistent")

            if (
                rows_removed
                and rows_removed > 0
                and profile.get("transform") is not None
            ):
                from rasterio import Affine

                profile = dict(profile)
                profile["transform"] = profile["transform"] * Affine.translation(
                    0, rows_removed
                )
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
            quality.skipped("burst_processing", "real burst processing not enabled (use_real_burst_processing=False): "
                            "no per-burst ESD, no deburst; whole-image ESD only", flag="no_deburst")
        else:
            quality.skipped("burst_processing", "ESD disabled and no deburst", flag="no_deburst")

        # ═══════════════════════════════════════════════════════════════
        # STEP 2.5: CROP TO AOI AFTER DEBURST (SNAP-style workflow)
        # This is the key change: burst overlaps are preserved for ESD,
        # but we crop to the actual AOI after deburst to reduce array
        # size for all downstream operations.
        # ═══════════════════════════════════════════════════════════════
        crop_info = None
        if crop_after_deburst and aoi_bbox is not None:
            ref_complex, sec_complex, profile, crop_info = (
                self._crop_to_aoi_after_deburst(
                    ref_complex,
                    sec_complex,
                    profile,
                    aoi_bbox,
                    margin_px=50,
                )
            )

        if crop_after_deburst and aoi_bbox is not None:
            if crop_info is None:
                if self._on_error == "raise":
                    raise CropError(
                        "crop", "the AOI crop was requested but could not be done (no georeferencing on the SLC, an AOI "
                        "outside the swath, or an empty window); the output would not cover the area asked for")
                quality.failed("crop", "AOI crop requested but not applied", "crop_not_applied")
            else:
                quality.ok("crop", f"cropped to {ref_complex.shape}")

        # ── Step 3/4: form the interferogram at FULL resolution ──
        #
        # ORDER (corrected): the interferogram is formed from the full-resolution SLCs, the geometric phase is removed at
        # full resolution (below), and only then is the complex interferogram multilooked. The previous code multilooked the
        # two SLCs and multiplied the averages. That leaves the phase noise at the single-look level whatever the number of
        # looks (measured on simulated speckle at coherence 0.8 and 32 looks: 0.92 rad that way, 0.095 rad when the
        # interferogram is averaged), and any geometric phase gradient inside a look window decorrelates the average.
        ref_complex_native = ref_complex
        sec_complex_native = sec_complex
        coh_window = (
            self._coh_window_explicit if self._coh_window_explicit is not None else 5
        )
        _ml = looks_azimuth > 1 or looks_range > 1
        _h_full = (ref_complex.shape[0] // looks_azimuth) * looks_azimuth
        _w_full = (ref_complex.shape[1] // looks_range) * looks_range
        if _h_full == 0 or _w_full == 0:
            raise InputError("input", f"the {ref_complex.shape} crop is smaller than one {looks_azimuth}x{looks_range} multilook window")
        out_shape = (_h_full // looks_azimuth, _w_full // looks_range)
        if _ml:
            logger.info(
                "Multilook %dx%d (azimuth x range) of the interferogram: %s -> %s",
                looks_azimuth, looks_range, ref_complex.shape, out_shape,
            )
        # Row blocks into one complex64 array. The one-line form built a conjugated copy, the product and a
        # third astype() copy of the whole scene; this holds only the result and one block.
        _np_ = self._np()
        interferogram = _np_.empty((_h_full, _w_full), dtype=_np_.complex64)
        _rb = max(16, 4_000_000 // _w_full)
        for _r0 in range(0, _h_full, _rb):
            _r1 = min(_h_full, _r0 + _rb)
            interferogram[_r0:_r1] = ref_complex[_r0:_r1, :_w_full] * _np_.conj(sec_complex[_r0:_r1, :_w_full])

        # amplitude of the reference (dB): the root-mean-square over the look window. For one look this is 20*log10|ref|.
        if _ml:
            from pygeofetch.insar.unwrap import multilook

            _power = (self._np().abs(ref_complex[:_h_full, :_w_full]) ** 2).astype(self._np().float32)
            amplitude = 10.0 * self._np().log10(multilook(_power, looks_azimuth, looks_range, wrapped_phase=False) + 1e-20)
            del _power
        else:
            amplitude = self._np().log10(self._np().abs(ref_complex) + 1e-10) * 20  # dB

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
        flat_earth_metadata: Dict[str, Any] = {"applied": False}
        _fe_inputs_ok = all(
            x is not None
            for x in (
                reference_safe_zip,
                secondary_safe_zip,
                reference_orbit_file,
                secondary_orbit_file,
            )
        )
        _fe_requested = (
            self._remove_flat_earth_phase
            if self._remove_flat_earth_phase is not None
            else _fe_inputs_ok
        )
        flat_earth_metadata["requested"] = bool(_fe_requested)
        if _fe_requested and not _fe_inputs_ok and self._remove_flat_earth_phase is True and self._on_error == "raise":
            raise FlatEarthError(
                "flat_earth", "flat-earth removal was requested explicitly but needs both SAFE zips and both orbit files")
        if not _fe_requested:
            quality.skipped("flat_earth", "not requested" if self._remove_flat_earth_phase is False
                            else "SAFE zips / orbit files not supplied", flag="no_flat_earth_removal")
        if _fe_requested and not _fe_inputs_ok:
            msg = (
                "Flat-earth phase removal was requested but needs both SAFE zips and both orbit "
                "files; not all were given. The interferogram keeps its flat-earth ramp."
            )
            logger.warning(msg)
            warnings.warn(msg, FlatEarthWarning, stacklevel=2)
        # ── Steps 4b/4c/5: flat-earth + topographic phase, simulated from the orbits and the DEM ──
        #
        # One geometry model for both (see radar_geometry.py and topo.py): exact range-Doppler geometry on a sparse node grid
        # in the coordinates of the OUTPUT interferogram (crop, deburst and multilook accounted for), the DEM giving the
        # terrain height of every pixel. This replaces (a) the old flat-earth polynomial, which was fitted in full-swath
        # pixel coordinates but evaluated at crop-relative, multilooked indices, and (b) the phase-vs-height regression,
        # which removes whatever correlates with height rather than the topographic phase.
        geometry_nodes = None
        acq_times: Dict[str, Any] = {}
        geo_info: Dict[str, Any] = {}
        topo_metadata = {"correction_applied": False}
        baseline_info: Dict[str, Any] = {"perpendicular_baseline_m": None, "perpendicular_baseline_source": None}
        _simulate = _fe_inputs_ok and (_fe_requested or dem is not None)
        if _simulate:
            try:
                from pygeofetch.insar.annotation import parse_slc_geometry
                from pygeofetch.insar.coregister import read_crop_offset, read_matched_swath
                from pygeofetch.insar.geolocation import parse_orbit_file
                from pygeofetch.insar.radar_geometry import PixelFrame, compute_geometry_nodes

                assert reference_safe_zip is not None and secondary_safe_zip is not None
                assert reference_orbit_file is not None and secondary_orbit_file is not None
                ref_geom_fe = parse_slc_geometry(
                    reference_safe_zip, member_hint=read_matched_swath(reference) if reference is not None else None)
                sec_geom_fe = parse_slc_geometry(
                    secondary_safe_zip, member_hint=read_matched_swath(secondary) if secondary is not None else None)
                _ci = crop_info or {}
                frame = PixelFrame(
                    shape=out_shape,
                    geometry=ref_geom_fe,
                    looks=(looks_azimuth, looks_range),
                    slc_offset=read_crop_offset(reference) if reference is not None else (0.0, 0.0),
                    crop_offset=(float(_ci.get("row_off", 0)), float(_ci.get("col_off", 0))),
                    deburst_first_kept_row=(burst_metadata.get("deburst_first_kept_row")
                                            if burst_metadata.get("deburst_applied") else None),
                )
                geometry_nodes = compute_geometry_nodes(
                    frame, parse_orbit_file(reference_orbit_file), parse_orbit_file(secondary_orbit_file), sec_geom_fe)
                acq_times = {
                    "reference_time_utc": frame.azimuth_time(out_shape[0] / 2).isoformat(),
                    "secondary_time_utc": sec_geom_fe.azimuth_time(sec_geom_fe.n_lines / 2).isoformat(),
                }
            except Exception as exc:  # noqa: BLE001
                if self._on_error == "raise":
                    raise FlatEarthError("geometry", f"radar geometry could not be solved: {exc}") from exc
                quality.failed("flat_earth", f"radar geometry: {exc}", "flat_earth_failed")
                warnings.warn(f"Radar geometry failed ({exc}); no flat-earth or topographic phase removed.",
                              FlatEarthWarning, stacklevel=2)
            if geometry_nodes is not None:
                try:
                    from pygeofetch.insar.topo import apply_geometric_phase

                    interferogram, geo_info = apply_geometric_phase(
                        interferogram, geometry_nodes, dem, remove_flat=bool(_fe_requested),
                        remove_topo=dem is not None, geoid_offset_m=self._dem_geoid_offset_m,
                        looks=(looks_azimuth, looks_range), inplace=True)
                except Exception as exc:  # noqa: BLE001
                    _cls = TopographicPhaseError if dem is not None else FlatEarthError
                    if self._on_error == "raise":
                        raise _cls("topographic_phase" if dem is not None else "flat_earth",
                                   f"geometric phase removal failed: {exc}") from exc
                    quality.failed("topographic_phase" if dem is not None else "flat_earth", str(exc), "geometry_phase_failed")
                    geo_info = {}
                else:
                    if _fe_requested:
                        flat_earth_metadata = {"applied": True, "requested": True,
                                               "phase_range_rad": None}
                        quality.ok("flat_earth", "removed from orbit geometry on the output grid")
                    if dem is not None:
                        topo_metadata = {"correction_applied": True, "method": "simulated",
                                         "r_squared": None}
                        _cov = geo_info.get("dem_coverage_fraction")
                        _lay = geo_info.get("layover_shadow_fraction")
                        quality.ok("topographic_phase",
                                   f"simulated from DEM and orbits (DEM coverage {100 * (_cov or 0):.0f}%, "
                                   f"layover/shadow {100 * (_lay or 0):.1f}%)", **{k: v for k, v in geo_info.items()
                                                                                    if k in ("dem_coverage_fraction",
                                                                                             "layover_shadow_fraction",
                                                                                             "height_range_m")})
                        if (_cov or 0) < 0.95:
                            quality.flag("dem_coverage_incomplete")
                    _bs = geometry_nodes.summary()
                    baseline_info = {"perpendicular_baseline_m": abs(_bs["bperp_mean_m"]),
                                     "perpendicular_baseline_source": "geometry_nodes_mean",
                                     "perpendicular_baseline_signed_mean_m": _bs["bperp_mean_m"],
                                     "perpendicular_baseline_range_m": (_bs["bperp_min_m"], _bs["bperp_max_m"]),
                                     "incidence_range_deg": (_bs["incidence_min_deg"], _bs["incidence_max_deg"])}
        if not _fe_inputs_ok:
            quality.flag("bperp_unknown")
        elif geometry_nodes is None and baseline_info["perpendicular_baseline_m"] is None:
            quality.flag("bperp_unknown")

        # Fallback for a DEM without orbit geometry (or a failed simulation in on_error="degrade"): the old empirical
        # phase-vs-height regression. It is not a topographic-phase model; the result is flagged so it is not mistaken for one.
        if dem is not None and not topo_metadata["correction_applied"] and quality.status_of("topographic_phase") != "failed":
            if use_chunked_processing and effective_chunk is not None:
                interferogram, topo_metadata = self._remove_topographic_phase_chunked(
                    interferogram, Path(dem), profile, chunk_rows=effective_chunk)
            else:
                interferogram, topo_metadata = self._remove_topographic_phase(interferogram, Path(dem), profile)
            if topo_metadata["correction_applied"]:
                topo_metadata["method"] = "regression"
                quality.skipped("topographic_phase", "orbit geometry not available: empirical phase-vs-height regression "
                                f"used (R2={topo_metadata.get('r_squared'):.2f}), not a topographic-phase model",
                                flag="topo_regression_fallback")
            elif topo_metadata.get("error"):
                if self._on_error == "raise":
                    raise TopographicPhaseError("topographic_phase", f"removal failed: {topo_metadata['error']}")
                quality.failed("topographic_phase", str(topo_metadata["error"]), "topo_failed")
            else:
                quality.skipped("topographic_phase", "orbit geometry not available and the regression did not reach R2 >= 0.5 "
                                f"({topo_metadata.get('reason') or topo_metadata.get('r_squared')}); nothing removed",
                                flag="topo_not_removed")
        elif dem is None:
            quality.skipped("topographic_phase", "no DEM supplied", flag="no_dem")
            logger.warning("No DEM provided — topographic phase NOT removed.")

        # ── Step 5a: multilook the corrected, full-resolution interferogram (complex mean) ──
        if _ml:
            from pygeofetch.insar.unwrap import multilook

            interferogram = multilook(interferogram, looks_azimuth, looks_range).astype(self._np().complex64)

        # ── Step 5b: Goldstein filter (moved here -- see the ordering
        # note above Step 4b) -- runs on the RESIDUAL phase (deformation
        # + atmosphere + DEM error + noise) after the geometric flat-
        # earth/topographic components are already removed, which is
        # what the filter's frequency-domain fringe enhancement actually
        # needs to work correctly.
        if apply_goldstein_filter:
            from pygeofetch.insar.unwrap import goldstein_filter

            interferogram = goldstein_filter(interferogram, alpha=goldstein_alpha)
            logger.info(
                "Goldstein phase filter applied (alpha=%.2f, tiled)", goldstein_alpha
            )

        # ── Step 6: Coherence estimation (chunked if enabled) ──
        if use_chunked_processing and effective_chunk is not None:
            coherence_native = self._estimate_coherence_chunked(
                ref_complex_native,
                sec_complex_native,
                coh_window,
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

            coherence = multilook(
                coherence_native, looks_azimuth, looks_range, wrapped_phase=False
            )
        else:
            coherence = coherence_native

        effective_coh_window = coh_window
        DataValidator.validate_coherence(coherence).raise_if_invalid()

        # amplitude was already computed above (Step 4), before
        # ref_complex was freed -- not recomputed here.

        result = InterferogramResult(
            interferogram=interferogram,
            coherence=coherence,
            amplitude=amplitude,
            profile=profile,
            reference_date=reference_date,
            secondary_date=secondary_date,
            esd_azimuth_shift_px=esd_shift,
            perpendicular_baseline_m=baseline_info["perpendicular_baseline_m"],
            temporal_baseline_days=_days_between(reference_date, secondary_date),
            geometry=geometry_nodes,
            metadata={
                "coherence_window": effective_coh_window,
                "esd_applied": self._esd_enabled and esd_shift is not None,
                "esd_method": burst_metadata["method"],
                "deburst_applied": burst_metadata["deburst_applied"],
                "esd_common_ground_lines": burst_metadata.get(
                    "esd_common_ground_lines"
                ),
                "esd_common_ground_total": burst_metadata.get(
                    "esd_common_ground_total"
                ),
                "burst_sync_offset_ms": burst_metadata.get("burst_sync_offset_ms"),
                "burst_sync_within_requirement": burst_metadata.get(
                    "burst_sync_within_requirement"
                ),
                "flat_earth_phase_removed": flat_earth_metadata["applied"],
                "flat_earth_phase_requested": flat_earth_metadata.get("requested"),
                "perpendicular_baseline_source": baseline_info.get("perpendicular_baseline_source"),
                "coregistration_fallback": coreg_fallback,
                "looks_azimuth": looks_azimuth,
                "looks_range": looks_range,
                "goldstein_applied": bool(apply_goldstein_filter),
                **pair_platforms,
                "topographic_phase_removed": topo_metadata["correction_applied"],
                "topographic_phase_r_squared": topo_metadata.get("r_squared"),
                "topographic_phase_method": topo_metadata.get("method"),
                **acq_times,
                "geometry": {k: v for k, v in geo_info.items() if k not in ("phase", "terrain_valid", "layover_shadow")},
                "perpendicular_baseline_signed_mean_m": baseline_info.get("perpendicular_baseline_signed_mean_m"),
                "perpendicular_baseline_range_m": baseline_info.get("perpendicular_baseline_range_m"),
                "incidence_range_deg": baseline_info.get("incidence_range_deg"),
                "coregistration_method": coreg_metadata.get("method"),
                "coregistration_refined_by_coherence": coreg_metadata.get(
                    "refined_by_coherence"
                ),
                "coregistration_gcps_initial": coreg_metadata.get("n_gcps_initial"),
                "coregistration_gcps_final": coreg_metadata.get("n_gcps_final"),
                "coregistration_rms_mean_px": coreg_metadata.get("rms_mean_px"),
                "coregistration_mean_coherence": coreg_metadata.get("mean_coherence"),
                "coregistration_collocation_coverage_fraction": coreg_metadata.get(
                    "collocation_coverage_fraction"
                ),
                "crop_after_deburst_applied": crop_info is not None,
                "crop_info": crop_info,
                "coherence_backend": getattr(self, "_coherence_backend", "cpu"),
                "chunked_processing": use_chunked_processing
                and effective_chunk is not None,
                "chunk_rows": effective_chunk,
                "esd_per_overlap_px": burst_metadata.get("esd_per_overlap_px"),
                "esd_overlap_std_px": burst_metadata.get("esd_overlap_std_px"),
                "tops_deramp": coreg_metadata.get("tops_deramp", False),
                "tops_deramp_reason": coreg_metadata.get("tops_deramp_reason"),
                "tops_sign": coreg_metadata.get("tops_sign"),
                "tops_sign_check_secondary": coreg_metadata.get("tops_sign_check_secondary"),
                "tops_doppler_rate_hz_s": coreg_metadata.get("tops_doppler_rate_hz_s"),
                "burst_exact_azimuth_mapping": coreg_metadata.get("burst_exact_azimuth_mapping", False),
                "burst_timing_steps_px": coreg_metadata.get("burst_timing_steps_px"),
                "coreg_rms_px": coreg_metadata.get("rms_mean_px"),
                "esd_delta_f_ovl_hz": burst_metadata.get("esd_delta_f_ovl_hz"),
                "esd_ambiguity": burst_metadata.get("esd_ambiguity"),
                "esd_total_shift_px": burst_metadata.get("esd_total_shift_px"),
                "esd_field": burst_metadata.get("esd_field"),
                "esd_residual_after_correction_px": burst_metadata.get("esd_residual_after_correction_px"),
                "quality": quality.to_dict(),
                "quality_grade": quality.grade,
                "resume_fingerprint": _fingerprint,
            },
        )
        result.metadata["resume_status"] = _resume_note
        if _pair_dir is not None:
            result.save(_pair_dir, auto_visualize=save_visualizations)
        return result

    # ══════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _pair_perpendicular_baseline(
        self, reference, reference_safe_zip, secondary_safe_zip,
        reference_orbit_file, secondary_orbit_file, secondary, profile, aoi_bbox, dem,
    ) -> Dict[str, Any]:
        """Perpendicular baseline (m, unsigned) at the zero-Doppler time of one ground point.

        The ground point is the AOI centre when ``aoi_bbox`` is given, else the centre of the
        geographic bounds of the cropped SLC, else the DEM centre. Satellite positions come from
        the precise orbit files at each scene's own zero-Doppler time for that point.
        """
        from pygeofetch.insar.annotation import parse_slc_geometry
        from pygeofetch.insar.coregister import read_matched_swath
        from pygeofetch.insar.geolocation import (
            find_zero_doppler_time, geodetic_to_ecef, interpolate_orbit_state,
            parse_orbit_file, perpendicular_baseline,
        )

        lat = lon = None
        source = None
        if aoi_bbox is not None:
            lat = (aoi_bbox.min_lat + aoi_bbox.max_lat) / 2
            lon = (aoi_bbox.min_lon + aoi_bbox.max_lon) / 2
            source = "aoi_centre"
        elif profile.get("transform") is not None and "4326" in str(profile.get("crs")):
            from rasterio.transform import array_bounds

            left, bottom, right, top = array_bounds(
                profile.get("height"), profile.get("width"), profile["transform"]
            )
            lat, lon, source = (bottom + top) / 2, (left + right) / 2, "slc_centre"
        elif dem is not None:
            import rasterio

            with rasterio.open(str(dem)) as d:
                b = d.bounds
            lat, lon, source = (b.bottom + b.top) / 2, (b.left + b.right) / 2, "dem_centre"
        if lat is None:
            raise ValueError("no ground point (give aoi_bbox, a georeferenced SLC or a DEM)")
        gp = geodetic_to_ecef(lat, lon, 0.0)
        pos = []
        for safe, orbit, slc in ((reference_safe_zip, reference_orbit_file, reference),
                                 (secondary_safe_zip, secondary_orbit_file, secondary)):
            hint = read_matched_swath(slc) if slc is not None else None
            geom = parse_slc_geometry(safe, member_hint=hint)
            times, positions, velocities = parse_orbit_file(orbit)
            tz = find_zero_doppler_time(
                times, positions, velocities, gp, initial_time_guess=geom.azimuth_time(geom.n_lines / 2)
            )
            p_sat, _ = interpolate_orbit_state(times, positions, velocities, target_time=tz)
            pos.append(p_sat)
        b = float(perpendicular_baseline(pos[0], pos[1], gp))
        return {"perpendicular_baseline_m": b, "perpendicular_baseline_source": source}

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
            if profile.get("transform") is None or getattr(
                profile.get("transform"), "is_identity", False
            ):
                gcps, gcp_crs = src.gcps
                if gcps and len(gcps) >= 4:
                    from rasterio.transform import from_gcps

                    profile["transform"] = from_gcps(gcps)
                    if gcp_crs:
                        profile["crs"] = gcp_crs
                    logger.info(
                        "Fitted approximate affine transform from %d GCPs for %s",
                        len(gcps),
                        path.name,
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
                    path.name,
                    dtype,
                )
                real = src.read(1).astype(np.float32)
                data = real.astype(np.complex64)

        if ref_shape is not None and data.shape != ref_shape:
            data = self._resample_to_reference(data, ref_shape)

        return data.astype(np.complex64), profile

    def _orbit_based_coregister(
        self,
        ref_complex,
        sec_complex,
        dem,
        reference_safe_zip,
        secondary_safe_zip,
        reference_orbit_file,
        secondary_orbit_file,
        reference_extracted_path=None,
        secondary_extracted_path=None,
        refine_by_coherence=True,
        degree=1,
        rms_threshold=None,
        window=None,
        coarse_search_radius=None,
        fine_search_radius=None,
        coherence_threshold=None,
        fit="geometry_first",
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
                fit_geometry_first_offsets,
                fit_offset_polynomial_robust,
                read_crop_offset,
                read_matched_swath,
                refine_offsets_by_coherence,
                resample_with_offset_field,
            )
            from pygeofetch.insar.geolocation import parse_orbit_file

            ref_swath_hint = (
                read_matched_swath(reference_extracted_path)
                if reference_extracted_path is not None
                else None
            )
            sec_swath_hint = (
                read_matched_swath(secondary_extracted_path)
                if secondary_extracted_path is not None
                else None
            )

            ref_geom = parse_slc_geometry(
                reference_safe_zip, member_hint=ref_swath_hint
            )
            sec_geom = parse_slc_geometry(
                secondary_safe_zip, member_hint=sec_swath_hint
            )
            ref_orbit = parse_orbit_file(reference_orbit_file)
            sec_orbit = parse_orbit_file(secondary_orbit_file)

            ref_center_time = ref_geom.azimuth_time(ref_geom.n_lines / 2)
            sec_center_time = sec_geom.azimuth_time(sec_geom.n_lines / 2)

            ref_row_off, ref_col_off = (
                read_crop_offset(reference_extracted_path)
                if reference_extracted_path is not None
                else (0.0, 0.0)
            )
            sec_row_off, sec_col_off = (
                read_crop_offset(secondary_extracted_path)
                if secondary_extracted_path is not None
                else (0.0, 0.0)
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
                        left, bottom, right, top = array_bounds(
                            src.height, src.width, approx_transform
                        )
                        sample_bounds = (left, bottom, right, top)
                    else:
                        b = src.bounds
                        sample_bounds = (b.left, b.bottom, b.right, b.top)

            # TOPS (IW/EW) bursts must be deramped before any sub-pixel interpolation or shift: their Doppler
            # centroid sweeps ~5 kHz through each burst, so interpolating them directly corrupts the phase
            # (fringes along azimuth, jumps at burst edges) and biases the cross-correlation offsets.
            ref_ramp = sec_ramp = None
            tops_meta: Dict[str, Any] = {"tops_deramp": False}
            try:
                from pygeofetch.insar.tops import parse_tops_ramp

                ref_ramp = parse_tops_ramp(reference_safe_zip, member_hint=ref_swath_hint)
                sec_ramp = parse_tops_ramp(secondary_safe_zip, member_hint=sec_swath_hint)
                sec_scores = sec_ramp.calibrate_sign(sec_complex, sec_row_off, sec_col_off)
                ref_scores = ref_ramp.calibrate_sign(ref_complex, ref_row_off, ref_col_off)
                tops_meta = {
                    "tops_deramp": True,
                    "tops_sign": sec_ramp.sign,
                    "tops_sign_check_secondary": {k: (float(v) if isinstance(v, (int, float)) else v)
                                                  for k, v in sec_scores.items()},
                    "tops_sign_check_reference": {k: (float(v) if isinstance(v, (int, float)) else v)
                                                  for k, v in ref_scores.items()},
                    "tops_doppler_rate_hz_s": ref_ramp.doppler_rate_hz_s(),
                    "tops_overlap_doppler_difference_hz": ref_ramp.overlap_doppler_difference_hz(),
                }
                if ref_ramp.sign != sec_ramp.sign:
                    logger.warning("TOPS deramp sign differs between reference (%+d) and secondary (%+d)",
                                   int(ref_ramp.sign), int(sec_ramp.sign))
            except Exception as exc:  # stripmap, or annotation without the TOPS fields
                ref_ramp = sec_ramp = None
                tops_meta = {"tops_deramp": False, "tops_deramp_reason": str(exc)}
                logger.info("No TOPS deramping for this pair (%s)", exc)

            grid_rows, grid_cols, off_rows, off_cols = compute_offset_field_from_dem(
                ref_geom,
                ref_orbit,
                sec_geom,
                sec_orbit,
                dem,
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
            geom_rows, geom_cols = list(grid_rows), list(grid_cols)
            geom_off_rows, geom_off_cols = list(off_rows), list(off_cols)
            xc_available = False

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
                            ref_complex,
                            sec_complex,
                            grid_rows,
                            grid_cols,
                            off_rows,
                            off_cols,
                            ref_row_offset=ref_row_off,
                            ref_col_offset=ref_col_off,
                            sec_row_offset=sec_row_off,
                            sec_col_offset=sec_col_off,
                            ref_phase_ramp=None if ref_ramp is None else ref_ramp.phase,
                            sec_phase_ramp=None if sec_ramp is None else sec_ramp.phase,
                            **refine_kwargs,
                        )
                    )
                    mean_coherence = float(sum(coherences) / len(coherences))
                    xc_available = True
                except Exception as exc:
                    logger.warning(
                        "Cross-correlation coherence refinement failed "
                        "(%s) — proceeding with the unrefined orbit/DEM "
                        "offset field.",
                        exc,
                    )

            # TOPS: fit the smooth time difference between the acquisitions, then map rows burst by burst
            # (BurstTimingMap). A polynomial fitted straight to row offsets cannot follow the steps that the two
            # acquisitions' burst timing puts between bursts, and leaves a misregistration that ramps through
            # each burst -- tenths of a pixel, far beyond what ESD (+-0.05 px) can see.
            timing_map = None
            if sec_ramp is not None:
                try:
                    import numpy as _np

                    from pygeofetch.insar.coregister import BurstTimingMap

                    timing_map = BurstTimingMap(ref_geom, sec_geom)
                    g_r = _np.asarray(geom_rows, dtype=float)
                    geom_off_rows = list(timing_map.to_u(g_r, g_r + _np.asarray(geom_off_rows, dtype=float)))
                    if xc_available:
                        x_r = _np.asarray(grid_rows, dtype=float)
                        off_rows = list(timing_map.to_u(x_r, x_r + _np.asarray(off_rows, dtype=float)))
                    timing_map.set_matching(float(_np.median(geom_off_rows)))
                    steps = timing_map.steps_px()
                    tops_meta["burst_timing_steps_px"] = steps
                    logger.info("Burst-exact azimuth mapping: relative burst timing steps (px) %s",
                                ", ".join(f"{s_:+.3f}" for s_ in steps))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Burst-exact azimuth mapping unavailable (%s); fitting row offsets directly", exc)
                    timing_map = None
            if fit == "geometry_first":
                if xc_available:
                    xc_args = (grid_rows, grid_cols, off_rows, off_cols)
                else:
                    xc_args = ([], [], [], [])
                row_fn, col_fn, coreg_quality = fit_geometry_first_offsets(
                    geom_rows, geom_cols, geom_off_rows, geom_off_cols,
                    *xc_args, degree=degree,
                )
            elif fit == "polynomial":
                row_fn, col_fn, coreg_quality = fit_offset_polynomial_robust(
                    grid_rows,
                    grid_cols,
                    off_rows,
                    off_cols,
                    degree=degree,
                    rms_threshold=rms_threshold,
                    outlier_rule="mean",
                )
            else:
                raise ValueError(
                    f"coregistration_fit must be 'geometry_first' or 'polynomial', got {fit!r}"
                )
            if timing_map is not None:
                row_fn = timing_map.wrap(row_fn)
                tops_meta["burst_exact_azimuth_mapping"] = True
            coreg_quality.mean_coherence = mean_coherence
            coreg_quality.log_summary()

            # Representative coregistration offset: MEDIAN of all offsets.
            # Using median instead of center-point evaluation because the
            # offset field can be non-uniform, and the median is more robust
            # to outliers than a single-point evaluation.
            import numpy as np

            all_offsets = [row_fn(r, c) for r, c in zip(grid_rows, grid_cols)]
            representative_row_offset = (
                float(np.median(all_offsets)) if all_offsets else 0.0
            )

            coreg_metadata = {
                "method": "orbit_dem_based",
                "refined_by_coherence": refine_by_coherence
                and mean_coherence is not None,
                "n_gcps_initial": n_gcps_initial,
                "n_gcps_final": coreg_quality.n_gcps_final,
                "rms_mean_px": coreg_quality.rms_mean,
                "mean_coherence": mean_coherence,
                "representative_row_offset_px": representative_row_offset,
                "coreg_mode": coreg_quality.mode,
                "n_geometry_points": coreg_quality.n_geometry_points,
                **tops_meta,
            }

            if ref_row_off or ref_col_off or sec_row_off or sec_col_off:
                logger.info(
                    "Correcting for cropped extraction: reference offset "
                    "(%.0f, %.0f), secondary offset (%.0f, %.0f)",
                    ref_row_off,
                    ref_col_off,
                    sec_row_off,
                    sec_col_off,
                )

            logger.info(
                "Real orbit-based coregistration applied (%d/%d grid "
                "points used in final fit)",
                coreg_quality.n_gcps_final,
                n_gcps_initial,
            )
            resampled = resample_with_offset_field(
                sec_complex,
                row_fn,
                col_fn,
                ref_row_offset=ref_row_off,
                ref_col_offset=ref_col_off,
                sec_row_offset=sec_row_off,
                sec_col_offset=sec_col_off,
                phase_ramp=None if sec_ramp is None else sec_ramp.phase,
                output_shape=ref_complex.shape,      # resample onto the reference grid, whatever the crop sizes
            )
            if resampled.shape != ref_complex.shape:
                resampled = self._resample_to_reference(resampled, ref_complex.shape)
            elif sec_ramp is not None:
                # Remember where the resampled secondary's samples came from, so a later azimuth shift (ESD)
                # can deramp and reramp it with the right phase.
                self._tops_ctx = {
                    "ref_ramp": ref_ramp, "sec_ramp": sec_ramp, "row_fn": row_fn, "col_fn": col_fn,
                    "ref_row_off": float(ref_row_off), "ref_col_off": float(ref_col_off),
                    # how well the first stage pins the constant azimuth offset (decides the ESD ambiguity scan)
                    "coreg_mode": coreg_quality.mode, "coreg_n": coreg_quality.n_gcps_final,
                    "coreg_rms_px": coreg_quality.rms_mean,
                }
            return resampled, coreg_metadata

        except Exception as exc:
            logger.warning(
                "Real orbit-based coregistration failed (%s) — falling "
                "back to shape-based resampling.",
                exc,
            )
            if self._on_error == "raise":
                raise CoregistrationError(
                    "coregistration", f"orbit/DEM coregistration failed: {exc}",
                    {"method": "orbit_dem"},
                ) from exc
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
                return (
                    self._resample_to_reference(sec_complex, ref_complex.shape),
                    fallback_metadata,
                )
            return sec_complex, fallback_metadata

    def _raster_collocation_coregister(
        self,
        ref_complex,
        sec_complex,
        ref_profile,
        sec_profile,
        refine_by_coherence=True,
        degree=1,
        rms_threshold=None,
        window=None,
        coarse_search_radius=None,
        fine_search_radius=None,
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
                sec_complex,
                sec_profile,
                ref_complex.shape,
                ref_profile,
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
                    if (
                        0 <= ri < valid_mask.shape[0]
                        and 0 <= ci < valid_mask.shape[1]
                        and valid_mask[ri, ci]
                    ):
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
                            ref_complex,
                            collocated,
                            grid_rows,
                            grid_cols,
                            off_rows,
                            off_cols,
                            ref_row_offset=0.0,
                            ref_col_offset=0.0,
                            sec_row_offset=0.0,
                            sec_col_offset=0.0,
                            **refine_kwargs,
                        )
                    )
                    mean_coherence = float(sum(coherences) / len(coherences))
                except Exception as exc:
                    logger.warning(
                        "Cross-correlation coherence refinement failed "
                        "after raster collocation (%s).",
                        exc,
                    )
            else:
                n_gcps_initial = 0

            if refine_by_coherence and mean_coherence is not None:
                row_fn, col_fn, coreg_quality = fit_offset_polynomial_robust(
                    grid_rows,
                    grid_cols,
                    off_rows,
                    off_cols,
                    degree=degree,
                    rms_threshold=rms_threshold,
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
                "refined_by_coherence": refine_by_coherence
                and mean_coherence is not None,
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
                "falling back to shape-based resampling.",
                exc,
            )
            if self._on_error == "raise":
                raise CoregistrationError(
                    "coregistration", f"raster-collocation coregistration failed: {exc}",
                    {"method": "raster_collocation"},
                ) from exc
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
                return (
                    self._resample_to_reference(sec_complex, ref_complex.shape),
                    fallback_metadata,
                )
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

    def _tops_content_phase(self, ctx, col_offset: float = 0.0):
        """Ramp phase carried by the resampled (and possibly already shifted) secondary at local (row, col)."""
        np = self._np()
        sec_ramp, row_fn, col_fn = ctx["sec_ramp"], ctx["row_fn"], ctx["col_fn"]
        r_off, c_off = ctx["ref_row_off"], ctx["ref_col_off"] + col_offset
        applied = ctx.get("applied_shift_px", 0.0)
        fields = list(ctx.get("applied_shift_fields", ()))      # spatially varying shifts, local coordinates

        def content_phase(rows, cols):
            rows = np.asarray(rows, dtype=np.float64)
            cols = np.asarray(cols, dtype=np.float64)
            rl = rows - applied
            for f in fields:
                rl = rl - np.asarray(f(rows, cols + col_offset), dtype=np.float64)
            rg = rl + r_off
            cg = cols + c_off
            return sec_ramp.phase(rg + np.asarray(row_fn(rg, cg), dtype=np.float64),
                                  cg + np.asarray(col_fn(rg, cg), dtype=np.float64))

        return content_phase

    def _apply_azimuth_shift_field(self, data, shift_fn):
        """TOPS-aware azimuth shift by a smoothly varying field ``shift_fn(rows, cols)`` (local coordinates, px)."""
        np = self._np()
        ctx = getattr(self, "_tops_ctx", None)
        if ctx is None:
            raise ValueError("a spatially varying azimuth shift needs the TOPS context")
        from pygeofetch.insar.tops import shift_azimuth_tops_field

        content_phase = self._tops_content_phase(ctx)
        out = shift_azimuth_tops_field(np.asarray(data), shift_fn, content_phase, ctx["ref_ramp"].lines_per_burst,
                                       ctx["ref_row_off"])
        ctx.setdefault("applied_shift_fields", []).append(shift_fn)
        return out

    def _esd_field_refinement(self, ref_complex, sec_complex, swath_timing, dt: float, row_offset: int,
                              ambiguity_px: float, esd_kwargs: dict, n_col_blocks: int = 10,
                              min_ptp_px: float = 0.002):
        """Measure ESD per overlap and range block, fit a smooth residual misregistration field and remove it.

        After the constant ESD shift (and its ambiguity), the misregistration can still vary across the scene: in
        range (the cross-correlation fit's range term) and from one burst overlap to the next. In TOPS data that
        leaves phase steps at the burst seams (about 55 rad per pixel of misregistration for IW) and ramps inside
        the bursts, which phase closure exposes but coherence and the constant ESD residual do not. Returns the
        (possibly shifted) secondary and a metadata dict.
        """
        from pygeofetch.insar.esd import estimate_esd_blocks, fit_esd_field

        blocks = estimate_esd_blocks(ref_complex, sec_complex, swath_timing, dt, row_offset=row_offset,
                                     n_col_blocks=n_col_blocks, **esd_kwargs)
        fit = fit_esd_field(blocks, ambiguity_px, ref_complex.shape)
        if fit is None:
            logger.info("ESD field: too few usable overlap blocks (%d) for a spatially varying correction", len(blocks))
            return sec_complex, {"applied": False, "n_blocks": len(blocks)}
        info = {k: v for k, v in fit.items() if k != "fn"}
        info["applied"] = False
        if fit["ptp_px"] >= min_ptp_px or abs(fit["coef_px"]["const"]) >= 1e-4:
            sec_complex = self._apply_azimuth_shift_field(sec_complex, fit["fn"])
            info["applied"] = True
            after = estimate_esd_blocks(ref_complex, sec_complex, swath_timing, dt, row_offset=row_offset,
                                        n_col_blocks=n_col_blocks, **esd_kwargs)
            refit = fit_esd_field(after, ambiguity_px, ref_complex.shape)
            if refit is not None:
                info["residual_rms_px"] = refit["rms_before_px"]
                info["residual_coef_px"] = refit["coef_px"]
        logger.info("ESD field (%d blocks in %d overlap(s), terms %s): %s px, spread %.4f px across the image; "
                    "block scatter %.4f px before, %.4f px about the field%s",
                    fit["n_used"], fit["n_overlaps"], "+".join(fit["terms"]),
                    ", ".join(f"{k} {v:+.4f}" for k, v in fit["coef_px"].items()), fit["ptp_px"],
                    fit["rms_before_px"], fit["rms_after_px"],
                    f"; after correction {info['residual_rms_px']:.4f} px" if "residual_rms_px" in info else "")
        return sec_complex, info

    def _esd_scan_needed(self, ambiguity_px: float):
        """Whether to choose the ESD ambiguity k by the coherence scan, and why.

        ``esd_ambiguity_scan``: "always", "never" or "auto" (default). The coherence scan is a weak criterion: the
        coherence changes by about 0.1 % per ambiguity near the optimum, and on Mexico City Sentinel-1 triplets it
        picked k = -2 and k = -1 where phase closure showed k = 0 was right (a wrong k leaves exactly k fringes per
        burst along track, invisible at the seams). "auto" therefore keeps k = 0 when the first-stage
        coregistration already pins the constant azimuth offset to well within half an ambiguity, i.e. when
        cross-correlation supports the fit and RMS / sqrt(n) < ambiguity / 2, and scans only otherwise
        (geometry-only first stage, or too few / too noisy cross-correlation points). On the full Mexico City
        stack the scan also chose k = +1 and +2 for pairs just above an ambiguity/4 limit.
        """
        policy = getattr(self, "_esd_ambiguity_scan", "auto")
        if policy == "always":
            return True, "esd_ambiguity_scan='always'"
        if policy == "never":
            return False, "esd_ambiguity_scan='never'"
        ctx = getattr(self, "_tops_ctx", None) or {}
        mode, n, rms = ctx.get("coreg_mode"), ctx.get("coreg_n"), ctx.get("coreg_rms_px")
        if mode in (None, "geometry_only") or not n or rms is None:
            return True, f"first stage without cross-correlation support (mode {mode})"
        sigma = float(rms) / max(float(n), 1.0) ** 0.5
        if sigma < ambiguity_px / 2:
            return False, (f"first stage {mode} from {n} cross-correlation points pins the offset to "
                           f"{sigma:.3f} px (< ambiguity/2 = {ambiguity_px / 2:.3f} px)")
        return True, f"first stage offset uncertain ({sigma:.3f} px >= ambiguity/2 = {ambiguity_px / 2:.3f} px)"

    def _resolve_esd_ambiguity(self, ref_complex, sec_complex, ambiguity_px: float, k_max: int = 4,
                               max_cols: int = 1536, k_limit: int = 15):
        """Pick the whole number of ESD ambiguities to add to the azimuth shift, by coherence.

        ESD measures the azimuth misregistration only modulo ``1 / (delta_f_ovl * dt_az)`` (about 0.1 px for IW):
        a first-stage coregistration error of a few tenths of a pixel leaves an error that ESD cannot see but that
        ramps the TOPS phase through every burst (about 3 rad/km per pixel) and steps it at burst edges. The
        interferometric coherence of a TOPS pair peaks at zero misregistration (the burst spectra overlap best),
        so the secondary is shifted by k ambiguities, k = -k_max..k_max, on a central block of columns, and the
        k with the highest coherence is kept; when the best k is at an end of the scan the scan is extended (up
        to +-k_limit, i.e. about 1.5 px). Returns the (possibly shifted) secondary and the scan.
        """
        np = self._np()
        ctx = getattr(self, "_tops_ctx", None)
        if ctx is None or not (ambiguity_px > 0) or not np.isfinite(ambiguity_px):
            return sec_complex, None
        from pygeofetch.insar.tops import shift_azimuth_tops

        h, w = sec_complex.shape
        c0 = max(0, (w - max_cols) // 2)
        c1 = min(w, c0 + max_cols)
        ref_s = np.asarray(ref_complex[:, c0:c1])
        sec_s = np.asarray(sec_complex[:, c0:c1], dtype=np.complex64)
        content_phase = self._tops_content_phase(ctx, col_offset=c0)
        lpb, r_off = ctx["ref_ramp"].lines_per_burst, ctx["ref_row_off"]

        def coherence(a, b, la=4, lr=16):
            n_r, n_c = (a.shape[0] // la) * la, (a.shape[1] // lr) * lr
            a, b = a[:n_r, :n_c].astype(np.complex128), b[:n_r, :n_c].astype(np.complex128)
            shp = (n_r // la, la, n_c // lr, lr)
            num = np.abs((a * np.conj(b)).reshape(shp).sum(axis=(1, 3)))
            den = np.sqrt((np.abs(a) ** 2).reshape(shp).sum(axis=(1, 3)) * (np.abs(b) ** 2).reshape(shp).sum(axis=(1, 3)))
            ok = den > 0
            return float(np.mean(num[ok] / den[ok])) if ok.any() else float("nan")

        cache = {}

        def score(k):
            if k not in cache:
                shifted = sec_s if k == 0 else shift_azimuth_tops(sec_s.copy(), k * ambiguity_px, content_phase, lpb, r_off)
                cache[k] = coherence(ref_s, shifted)
            return cache[k]

        for k in range(-k_max, k_max + 1):
            score(k)
        # If the peak sits at an end of the scan, keep extending in that direction (up to k_limit ambiguities).
        while True:
            ks_now = sorted(cache)
            k_best = max(ks_now, key=lambda k: cache[k])
            if k_best == ks_now[-1] and k_best < k_limit:
                score(k_best + 1)
            elif k_best == ks_now[0] and k_best > -k_limit:
                score(k_best - 1)
            else:
                break
        ks = sorted(cache)
        scores = [cache[k] for k in ks]
        info = {"ambiguity_px": float(ambiguity_px), "k": int(k_best),
                "coherence_scan": {int(k): round(float(cache[k]), 5) for k in ks},
                "at_scan_edge": abs(k_best) >= k_limit}
        logger.info("ESD ambiguity scan (coherence for k = %d..%d ambiguities of %.4f px): %s -> k = %+d",
                    ks[0], ks[-1], ambiguity_px, ", ".join(f"{v:.4f}" for v in scores), k_best)
        if info["at_scan_edge"]:
            logger.warning("ESD ambiguity scan peaks at its limit (k = %+d): the first-stage coregistration is off "
                           "by more than %d ambiguities", k_best, k_limit)
        if k_best != 0:
            sec_complex = self._apply_azimuth_shift(sec_complex, k_best * ambiguity_px)
        return sec_complex, info

    def _apply_azimuth_shift(self, data, shift_px: float):
        """Apply a sub-pixel azimuth (row) shift via Fourier phase ramp.

        The transform runs along the rows of each column independently, so it is done in column blocks: the
        double-precision FFT, ramp product and inverse FFT then exist for one block instead of the whole scene
        (three full-size complex128 arrays, about 4 GB on a 4500 x 18000 crop). The block results are identical to
        the whole-array ones. Written into ``data`` when that is a plain writable complex64 array of its own
        (every caller replaces its variable with the return value); otherwise into a new array."""
        np = self._np()
        ctx = getattr(self, "_tops_ctx", None)
        if ctx is not None:
            # TOPS: deramp with the phase the resampled secondary carries, shift at baseband, reramp at the
            # shifted positions (see pygeofetch.insar.tops.shift_azimuth_tops). A plain Fourier shift of a
            # TOPS burst moves its amplitude but leaves a 2 pi f_DC * shift phase error that ramps through
            # every burst.
            from pygeofetch.insar.tops import shift_azimuth_tops

            content_phase = self._tops_content_phase(ctx)
            out = shift_azimuth_tops(np.asarray(data), float(shift_px), content_phase,
                                     ctx["ref_ramp"].lines_per_burst, ctx["ref_row_off"])
            # The data now carry the ramp of the positions r - shift; a later shift must start from there.
            ctx["applied_shift_px"] = ctx.get("applied_shift_px", 0.0) + float(shift_px)
            return out
        h, w = data.shape
        freq = np.fft.fftfreq(h).reshape(-1, 1)
        ramp = np.exp(-2j * np.pi * freq * shift_px)
        in_place = data.dtype == np.complex64 and data.flags.writeable and data.flags.owndata
        out = data if in_place else np.empty((h, w), dtype=np.complex64)
        cb = max(16, 4_000_000 // max(h, 1))
        for c0 in range(0, w, cb):
            c1 = min(w, c0 + cb)
            out[:, c0:c1] = np.fft.ifft(np.fft.fft(data[:, c0:c1], axis=0) * ramp, axis=0)
        return out

    def _remove_topographic_phase(self, interferogram, dem_path: Path, profile) -> Any:
        """
        Remove the topographic phase component using a reference DEM.
        Full implementation — see original docstring for detailed explanation.
        """
        np = self._np()
        try:
            import rasterio
        except ImportError:
            return interferogram, {
                "correction_applied": False,
                "reason": "rasterio_missing",
                "error": "rasterio_missing",
            }

        try:
            interferogram_crs = profile.get("crs")
            interferogram_transform = profile.get("transform")

            with rasterio.open(dem_path) as dem_src:
                if (
                    interferogram_crs is not None
                    and interferogram_transform is not None
                ):
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
                return interferogram, {
                    "correction_applied": False,
                    "reason": "insufficient_valid_pixels",
                }

            dem_std = np.std(dem[valid])
            if dem_std < 1.0:
                logger.info(
                    "DEM has negligible elevation variance (std=%.2fm) — skipping.",
                    dem_std,
                )
                return interferogram, {
                    "correction_applied": False,
                    "reason": "dem_no_variance",
                }

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

            fine_half_width = coarse[1] - coarse[0]
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
                    r_squared,
                    best_slope,
                )
                return interferogram, {
                    "correction_applied": False,
                    "r_squared": float(r_squared),
                }

            topo_phase = np.angle(np.exp(1j * (best_slope * dem + intercept)))

            logger.info(
                "Topographic phase regression R²=%.2f — correction applied", r_squared
            )
            corrected = interferogram * np.exp(-1j * topo_phase)
            return corrected.astype(np.complex64), {
                "correction_applied": True,
                "r_squared": float(r_squared),
            }

        except Exception as exc:
            logger.warning(
                "Topographic phase removal failed: %s — returning uncorrected", exc
            )
            return interferogram, {"correction_applied": False, "reason": str(exc), "error": str(exc)}

    def _estimate_coherence(self, ref, sec, window: int) -> Any:
        """Estimate interferometric coherence via local windowed correlation."""
        from pygeofetch.insar.gpu import get_array_module, to_numpy

        xp, ndi, using_gpu = get_array_module(prefer_gpu=self._use_gpu)
        self._coherence_backend = "gpu" if using_gpu else "cpu"

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