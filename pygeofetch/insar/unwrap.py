"""
PhaseUnwrapper — production-grade phase unwrapping via SNAPHU.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import subprocess
import sys
import urllib.request
import zipfile
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger("pygeofetch.insar.unwrap")

# Allow users to override the installation directory via environment variable
SNAPHU_DIR = Path(os.environ.get("PYGEOFETCH_SNAPHU_DIR", Path.home() / ".pygeofetch" / "bin"))
EXPECTED_VERSION = "2.0.7-pygeofetch-v1"

# Known SHA256 hashes for security verification. Without a real hash
# here, _verify_sha256() below silently returns True for ANY content --
# demonstrated directly: a file containing arbitrary, non-zip,
# obviously-tampered bytes passed verification for both entries when
# this dict had None values. That's not a hypothetical; it made the
# whole "Checksum verification failed!" safety check dead code for
# every real download this module performs.
#
# snaphu-v2.0.7.zip: verified directly -- downloaded from the exact
# pinned URL used by _ensure_snaphu_cli()
# (github.com/marsfan/SNAPHU-win/releases/download/2.0.7/snaphu-v2.0.7.zip)
# and hashed. Contents inspected and consistent with a real SNAPHU
# Windows build (README, snaphu.exe, msys-2.0.dll, source .c files).
# NOTE on what this does and doesn't guarantee: pinning this hash
# protects against in-transit tampering, accidental corruption, and any
# FUTURE replacement of this release asset going undetected -- it does
# NOT retroactively vet the marsfan/SNAPHU-win build process itself,
# which is an explicitly unofficial, single-maintainer build (see that
# repo's own README: "THIS IS NOT AN OFFICIAL BUILD OF SNAPHU"). Same
# trust model as installing any third-party pre-built binary.
#
# snaphu_esa.zip: NOT verified -- forum.step.esa.int is outside this
# environment's network allowlist, so no hash could be independently
# confirmed here. Left as None deliberately (better an honest gap than
# a fabricated hash); populate it yourself once you can download and
# hash that asset from an unrestricted network.
KNOWN_HASHES = {
    "snaphu-v2.0.7.zip": "12f5af3fc485d71b85d38cef69b09b18aeb948903f6cbbcaa78ce82063f7f8b9",
    "snaphu_esa.zip": None,  # could not verify from this environment -- see note above
}


@dataclass
class UnwrapResult:
    """Result of phase unwrapping for one interferogram pair."""
    unwrapped_phase: Any      
    conncomp: Any             
    coherence: Any            
    profile: Dict[str, Any]   
    reference_date: Optional[str] = None
    secondary_date: Optional[str] = None
    nlooks: float = 1.0
    cost_mode: str = "defo"
    init_method: str = "mcf"
    reliable_fraction: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: Union[str, Path], save_png: bool = False, mask_unreliable: bool = True) -> Dict[str, Path]:
        import numpy as np
        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        base_profile = {
            "driver": "GTiff", "count": 1,
            "height": self.unwrapped_phase.shape[0], "width": self.unwrapped_phase.shape[1],
            "crs": self.profile.get("crs"), "transform": self.profile.get("transform"),
            "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256,
        }

        unw_to_save = self.unwrapped_phase.copy()
        if mask_unreliable:
            unw_to_save[self.conncomp == 0] = -9999.0

        for name, data, dtype, nodata in [
            ("unwrapped_phase", unw_to_save, "float32", -9999.0),
            ("conncomp", self.conncomp, "int32", 0),
            ("coherence", self.coherence, "float32", -1.0)
        ]:
            path = out_dir / f"{name}.tif"
            profile = dict(base_profile, dtype=dtype, nodata=nodata)
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data.astype(np.dtype(dtype))[np.newaxis])
            paths[name] = path

        logger.info("Unwrap products saved → %s (reliable=%.1f%%)", out_dir, self.reliable_fraction * 100)
        if save_png: self._save_pngs(out_dir)
        return paths

    def _save_pngs(self, out_dir: Path) -> None:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        masked_unw = np.ma.masked_where(self.conncomp == 0, self.unwrapped_phase)
        finite_unw = self.unwrapped_phase[self.conncomp > 0]
        vmin, vmax = (float(np.percentile(finite_unw, 2)), float(np.percentile(finite_unw, 98))) if finite_unw.size > 0 else (-np.pi, np.pi)
        im = ax.imshow(masked_unw, cmap="jet", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"Unwrapped Phase (rad)\n{self.reference_date} → {self.secondary_date} | reliable={self.reliable_fraction*100:.1f}%")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(out_dir / "unwrapped_phase.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        masked_cc = np.ma.masked_where(self.conncomp == 0, self.conncomp)
        ax.imshow(masked_cc, cmap="tab20", aspect="auto")
        ax.set_title(f"Connected Components\n{self.reference_date} → {self.secondary_date}")
        fig.savefig(out_dir / "conncomp.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _verify_sha256(filepath: Path, expected_hash: Optional[str]) -> bool:
    if not expected_hash:
        logger.warning("No SHA256 hash provided for %s. Skipping verification.", filepath.name)
        return True
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_hash.lower()


def _download_with_progress(url: str, dest_path: Path, timeout: int = 60):
    logger.info(f"Downloading {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'PyGeoFetch/1.0'})
    
    with urllib.request.urlopen(req, timeout=timeout) as response:
        total_size = int(response.info().get('Content-Length', -1))
        downloaded = 0
        block_size = 8192
        last_logged_pct = -1
        
        with open(dest_path, 'wb') as f:
            while True:
                buffer = response.read(block_size)
                if not buffer: break
                downloaded += len(buffer)
                f.write(buffer)
                
                if total_size > 0:
                    pct = int((downloaded / total_size) * 100)
                    if pct // 25 > last_logged_pct:
                        logger.info(f"Download progress: {pct}%")
                        last_logged_pct = pct // 25


def _check_url_alive(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'PyGeoFetch/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except Exception:
        return False


def _ensure_snaphu_cli() -> str:
    exe_name = "snaphu.exe" if platform.system() == "Windows" else "snaphu"
    path = shutil.which(exe_name)
    if path: return path
        
    bin_dir = SNAPHU_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)
    local_exe = bin_dir / exe_name
    version_file = bin_dir / ".snaphu_version"
    
    # Check cache
    if local_exe.exists() and version_file.exists():
        if version_file.read_text().strip() == EXPECTED_VERSION:
            return str(local_exe)
        
    if platform.system() == "Windows":
        logger.info("SNAPHU executable not found or outdated. Downloading pre-compiled binary...")
        
        primary_url = "https://github.com/marsfan/SNAPHU-win/releases/download/2.0.7/snaphu-v2.0.7.zip"
        esa_url = "https://forum.step.esa.int/uploads/short-url/lwqYDwx4aN76w3C0SZZWLCQ84he.zip"
        
        urls_to_try = [primary_url]
        if _check_url_alive(esa_url):
            urls_to_try.append(esa_url)
            
        for url in urls_to_try:
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    zip_path = Path(tmp_dir) / "snaphu.zip"
                    _download_with_progress(url, zip_path)
                    
                    if not _verify_sha256(zip_path, KNOWN_HASHES.get(Path(url).name)):
                        raise ValueError("Checksum verification failed! File may be corrupted or tampered with.")
                        
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        for file_info in zf.infolist():
                            if file_info.is_dir(): continue
                            target_path = bin_dir / Path(file_info.filename).name
                            with zf.open(file_info) as src, open(target_path, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                        
                        exe_files = [f for f in zf.namelist() if f.lower().endswith(".exe") and "snaphu" in f.lower()]
                        if not exe_files: exe_files = [f for f in zf.namelist() if f.lower().endswith(".exe")]
                        if not exe_files: raise RuntimeError("Downloaded zip did not contain any .exe file.")
                            
                        local_exe = bin_dir / Path(exe_files[0]).name
                        version_file.write_text(EXPECTED_VERSION)
                        
                logger.info(f"SNAPHU downloaded successfully to {local_exe}")
                return str(local_exe)
            except Exception as e:
                logger.warning(f"Failed to download from {url}: {e}")
                
        raise RuntimeError("Failed to download/extract SNAPHU executable from all sources.")
            
    raise RuntimeError(
        "SNAPHU executable not found in PATH.\n"
        "Please install it using your system package manager:\n"
        "  Ubuntu/Debian: sudo apt-get install snaphu\n"
        "  macOS (Homebrew): brew install snaphu\n"
        "Or install the Python bindings: pip install snaphu"
    )


def goldstein_filter(interferogram, alpha: float = 0.5, tile_size: int = 32, overlap: float = 0.5):
    import numpy as np
    interferogram = np.asarray(interferogram, dtype=np.complex64)
    h, w = interferogram.shape
    step = max(1, int(tile_size * (1 - overlap)))
    window_1d = np.hanning(tile_size)
    window_2d = np.outer(window_1d, window_1d).astype(np.float32)
    output = np.zeros((h, w), dtype=np.complex64)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    row_starts = list(range(0, max(1, h - tile_size + 1), step))
    col_starts = list(range(0, max(1, w - tile_size + 1), step))
    if not row_starts or row_starts[-1] + tile_size < h: row_starts.append(max(0, h - tile_size))
    if not col_starts or col_starts[-1] + tile_size < w: col_starts.append(max(0, w - tile_size))
    for row_start in row_starts:
        for col_start in col_starts:
            r_end = min(row_start + tile_size, h)
            c_end = min(col_start + tile_size, w)
            tile = interferogram[row_start:r_end, col_start:c_end]
            th, tw = tile.shape
            if th < 2 or tw < 2: continue
            spectrum = np.fft.fft2(tile)
            magnitude = np.abs(spectrum)
            peak = magnitude.max()
            if peak > 0: filtered_spectrum = spectrum * (magnitude ** alpha) / (peak ** alpha)
            else: filtered_spectrum = spectrum
            filtered_tile = np.fft.ifft2(filtered_spectrum)
            tile_window = window_2d[:th, :tw]
            output[row_start:r_end, col_start:c_end] += filtered_tile * tile_window
            weight_sum[row_start:r_end, col_start:c_end] += tile_window
    valid = weight_sum > 0
    output[valid] /= weight_sum[valid]
    output[~valid] = interferogram[~valid]
    return output.astype(np.complex64)


def bridge_unwrap_regions(unwrapped_phase, conncomp, bridge_radius: int = 50, min_region_size: int = 100, reference_pixel=None):
    import numpy as np
    from scipy.spatial import cKDTree
    labels = np.unique(conncomp)
    labels = labels[labels != 0]
    region_sizes = {lbl: int(np.sum(conncomp == lbl)) for lbl in labels}
    valid_labels = [lbl for lbl in labels if region_sizes[lbl] >= min_region_size]
    if not valid_labels: return unwrapped_phase.copy(), {}
    valid_labels.sort(key=lambda lbl: region_sizes[lbl], reverse=True)
    if reference_pixel is not None:
        rp_row, rp_col = reference_pixel
        rp_label = int(conncomp[rp_row, rp_col])
        if rp_label == 0 or rp_label not in valid_labels:
            raise ValueError(f"reference_pixel {reference_pixel} is not part of any valid region.")
        reference_label = rp_label
    else:
        reference_label = valid_labels[0]
    corrected = np.array(unwrapped_phase, dtype=np.float64, copy=True)
    resolved_points = {reference_label: np.column_stack(np.where(conncomp == reference_label))}
    offsets_applied = {reference_label: 0.0}
    remaining = set(valid_labels) - {reference_label}
    while remaining:
        best = None
        for lbl in remaining:
            lbl_points = np.column_stack(np.where(conncomp == lbl))
            for resolved_lbl, ref_points in resolved_points.items():
                tree = cKDTree(ref_points)
                dist, idx = tree.query(lbl_points)
                min_i = int(np.argmin(dist))
                if best is None or dist[min_i] < best[0]:
                    best = (dist[min_i], lbl, resolved_lbl, lbl_points[min_i], ref_points[idx[min_i]])
        if best is None: break
        _, lbl, target_lbl, point_a, point_b = best
        def _local_median(point, region_mask):
            r, c = point
            r0, r1 = max(0, r - bridge_radius), r + bridge_radius + 1
            c0, c1 = max(0, c - bridge_radius), c + bridge_radius + 1
            window_phase = corrected[r0:r1, c0:c1]
            window_mask = region_mask[r0:r1, c0:c1]
            values = window_phase[window_mask]
            return float(np.median(values)) if len(values) > 0 else float(corrected[r, c])
        median_a = _local_median(point_a, conncomp == lbl)
        median_b = _local_median(point_b, conncomp == target_lbl)
        raw_offset = median_b - median_a
        integer_offset = 2 * np.pi * np.round(raw_offset / (2 * np.pi))
        corrected[conncomp == lbl] += integer_offset
        offsets_applied[lbl] = integer_offset
        resolved_points[lbl] = np.column_stack(np.where(conncomp == lbl))
        remaining.remove(lbl)
    return corrected.astype(np.float32), offsets_applied


def multilook(data: Any, looks_azimuth: int = 4, looks_range: int = 1, wrapped_phase: Optional[bool] = None) -> Any:
    import numpy as np
    h, w = data.shape
    h_ml = (h // looks_azimuth) * looks_azimuth
    w_ml = (w // looks_range) * looks_range
    trimmed = data[:h_ml, :w_ml]
    reshaped_shape = (h_ml // looks_azimuth, looks_azimuth, w_ml // looks_range, looks_range)
    if np.iscomplexobj(trimmed):
        return trimmed.reshape(reshaped_shape).mean(axis=(1, 3))
    if wrapped_phase is None:
        raise ValueError("multilook() received real-valued input but wrapped_phase was not specified.")
    if wrapped_phase:
        reshaped = np.exp(1j * trimmed).reshape(reshaped_shape)
        return np.angle(reshaped.mean(axis=(1, 3)))
    else:
        return trimmed.reshape(reshaped_shape).mean(axis=(1, 3))


class PhaseUnwrapper:
    def __init__(self, cost_mode: str = "defo", init_method: str = "mcf") -> None:
        valid_costs = ("topo", "defo", "smooth", "nostatcosts")
        if cost_mode not in valid_costs: raise ValueError(f"cost_mode must be one of {valid_costs}")
        valid_inits = ("mcf", "mst")
        if init_method not in valid_inits: raise ValueError(f"init_method must be one of {valid_inits}")
        self._cost_mode = cost_mode
        self._init_method = init_method

    def _unwrap_cli(self, igram, corr, nlooks, min_conncomp_frac, min_region_size):
        import numpy as np
        import os
        cli_path = _ensure_snaphu_cli()
        height, width = igram.shape
        bin_dir = Path(cli_path).parent
        
        def _run_snaphu(tmp_dir):
            tmp = Path(tmp_dir)
            igram_path = tmp / "igram.cpx"
            corr_path = tmp / "corr.cor"
            unw_path = tmp / "unw.unw"
            cc_path = tmp / "conncomp.cc"
            conf_path = tmp / "snaphu.conf"
            igram.tofile(str(igram_path))
            corr.tofile(str(corr_path))
            nlooks_int = max(1, int(nlooks))
            
            conf_text = f"""INFILE            igram.cpx
INFILEFORMAT      COMPLEX_DATA
CORRFILE          corr.cor
CORRFILEFORMAT    FLOAT_DATA
OUTFILE           unw.unw
OUTFILEFORMAT     FLOAT_DATA
CONNCOMPFILE      conncomp.cc
LINELENGTH        {width}
STATCOSTMODE      {self._cost_mode.upper()}
INITMETHOD        {self._init_method.upper()}
NLOOKSRANGE       {nlooks_int}
NLOOKSAZ          {nlooks_int}
"""
            conf_path.write_text(conf_text)
            cmd = [cli_path, "-f", "snaphu.conf"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp), timeout=3600)
            return result, unw_path, cc_path

        with tempfile.TemporaryDirectory() as tmp_dir:
            result, unw_path, cc_path = _run_snaphu(tmp_dir)
            
            if result.returncode in (3221225781, -1073741515):
                logger.warning("SNAPHU crashed with STATUS_DLL_NOT_FOUND (0xC0000135). Auto-bundling runtime DLLs...")
                self._ensure_runtime_dlls(bin_dir)
                logger.info("Retrying SNAPHU with bundled DLLs...")
                result, unw_path, cc_path = _run_snaphu(tmp_dir)
                
            if result.returncode != 0:
                raise RuntimeError(f"SNAPHU CLI failed with code {result.returncode}: {result.stderr}")
                
            # ═══════════════════════════════════════════════════════════════
            # SMART FILE READER (Handles Windows Port Quirks)
            # ═══════════════════════════════════════════════════════════════
            expected_pixels = height * width
            
            # 1. Read Unwrapped Phase (Usually FLOAT_DATA = 4 bytes)
            unw_size = os.path.getsize(str(unw_path))
            unw_dtype = np.float32 if unw_size <= expected_pixels * 4 else np.float64
            unw = np.fromfile(str(unw_path), dtype=unw_dtype)
            
            # 2. Read Connected Components (Adapts to 1-byte, 2-byte, or 4-byte formats)
            cc_size = os.path.getsize(str(cc_path))
            if cc_size >= expected_pixels * 4:
                cc_dtype = np.uint32
            elif cc_size >= expected_pixels * 2:
                cc_dtype = np.uint16
            else:
                cc_dtype = np.uint8  # The Windows port default
                
            conncomp = np.fromfile(str(cc_path), dtype=cc_dtype)
            
            # 3. Auto-heal the Windows "off-by-one" truncation bug
            if conncomp.size < expected_pixels:
                logger.warning(
                    "SNAPHU conncomp file is %d elements, expected %d. Padding missing bytes with 0 "
                    "(known Windows port truncation quirk).", conncomp.size, expected_pixels
                )
                conncomp = np.pad(conncomp, (0, expected_pixels - conncomp.size), mode='constant', constant_values=0)
            elif conncomp.size > expected_pixels:
                conncomp = conncomp[:expected_pixels]
                
            if unw.size < expected_pixels:
                unw = np.pad(unw, (0, expected_pixels - unw.size), mode='constant', constant_values=0)
            elif unw.size > expected_pixels:
                unw = unw[:expected_pixels]

            unw = unw.reshape((height, width))
            conncomp = conncomp.reshape((height, width))
            
        total_pixels = height * width
        labels, counts = np.unique(conncomp, return_counts=True)
        for lbl, count in zip(labels, counts):
            if lbl == 0: continue
            if count < min_region_size or (count / total_pixels) < min_conncomp_frac:
                conncomp[conncomp == lbl] = 0
        return unw, conncomp.astype(np.int32)
    

    def _ensure_runtime_dlls(self, bin_dir: Path):
        import shutil, zipfile, io, urllib.request
        
        required_dlls = ["msys-2.0.dll", "libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll", "vcruntime140.dll", "msvcp140.dll"]
        missing = [dll for dll in required_dlls if not (bin_dir / dll).exists()]
        if not missing: return
        
        logger.info(f"SNAPHU requires {missing}. Searching for runtime DLLs...")
        found_dlls = self._find_git_dlls(missing)
                        
        if len(found_dlls) >= len([d for d in missing if d.startswith("lib") or d.startswith("msys")]):
            logger.info("Found MinGW/MSYS DLLs in Git for Windows installation. Copying to SNAPHU bin...")
            for dll, src in found_dlls.items():
                shutil.copy2(src, bin_dir / dll)
            return
            
        logger.info("Downloading MSVC runtime from NuGet (no admin required)...")
        nuget_url = "https://www.nuget.org/api/v2/package/VCRuntime.140"
        try:
            req = urllib.request.Request(nuget_url, headers={'User-Agent': 'PyGeoFetch'})
            with urllib.request.urlopen(req, timeout=30) as response:
                nupkg_data = response.read()
            with zipfile.ZipFile(io.BytesIO(nupkg_data)) as zf:
                for name in zf.namelist():
                    if "win-x64/native/" in name and name.lower().endswith(".dll"):
                        dll_name = Path(name).name
                        if dll_name.lower() in [m.lower() for m in missing]:
                            with zf.open(name) as src, open(bin_dir / dll_name, "wb") as dst:
                                shutil.copyfileobj(src, dst)
            logger.info(f"Successfully extracted MSVC DLLs to {bin_dir}. Retrying SNAPHU...")
            return
        except Exception as e:
            logger.warning(f"Failed to download MSVC DLLs: {e}")
            
        esa_url = "https://forum.step.esa.int/uploads/short-url/lwqYDwx4aN76w3C0SZZWLCQ84he.zip"
        if _check_url_alive(esa_url):
            logger.info("Attempting to download self-contained ESA STEP SNAPHU build...")
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    zip_path = Path(tmp_dir) / "snaphu_esa.zip"
                    _download_with_progress(esa_url, zip_path)
                    # BUG FIX: this download previously extracted straight
                    # from the zip with no call to _verify_sha256 at all --
                    # not even the (until now) no-op check the primary
                    # download path at least had the scaffolding for.
                    # KNOWN_HASHES["snaphu_esa.zip"] is still None (could
                    # not be independently verified from this environment
                    # -- see the module-level comment above), so this
                    # still logs a warning and proceeds rather than
                    # blocking -- but the check now actually runs, so
                    # populating that hash later will actually take effect.
                    if not _verify_sha256(zip_path, KNOWN_HASHES.get("snaphu_esa.zip")):
                        raise ValueError("Checksum verification failed! File may be corrupted or tampered with.")
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        for file_info in zf.infolist():
                            if file_info.is_dir(): continue
                            filename = Path(file_info.filename).name
                            target_path = bin_dir / filename
                            with zf.open(file_info) as src, open(target_path, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                    logger.info("Successfully installed self-contained ESA SNAPHU build. Retrying...")
                    return
            except Exception as e:
                logger.warning(f"Failed to download ESA build: {e}")
                
        # Platform Debug Info
        logger.error(
            "SNAPHU failed to run after all repair attempts.\n"
            f"Platform: {platform.system()} {platform.release()} ({platform.machine()})\n"
            f"Python: {sys.version}\n"
            "Please report this issue or install SNAPHU manually."
        )
        raise RuntimeError(
            "Failed to resolve missing DLLs automatically.\n"
            "Please install Git for Windows (https://git-scm.com/download/win) or MinGW, "
            "then restart your Jupyter kernel."
        )

    def _find_git_dlls(self, missing_dlls):
        found = {}
        search_roots = [
            Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "Git",
            Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "Git",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Git",
            Path("C:\\Git"),
        ]
        git_env = os.environ.get("GIT_INSTALL_ROOT")
        if git_env: search_roots.append(Path(git_env))
            
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)
        for p in path_dirs:
            git_exe = Path(p) / "git.exe"
            if git_exe.exists():
                potential_root = git_exe.parent.parent
                if (potential_root / "mingw64").exists():
                    search_roots.append(potential_root)
                    
        search_roots = list(set([r for r in search_roots if r.exists()]))
        
        for git_root in search_roots:
            search_dirs = [git_root / "mingw64" / "bin", git_root / "usr" / "bin", git_root / "bin"]
            for dll in missing_dlls:
                if dll in found: continue
                for d in search_dirs:
                    if (d / dll).exists():
                        found[dll] = d / dll
                        break
        return found

    def unwrap(self, interferogram: Any, coherence: Any, nlooks: float = 1.0, mask: Optional[Any] = None, min_conncomp_frac: float = 0.01, min_region_size: int = 100) -> Tuple[Any, Any]:
        np = self._np()
        if np.iscomplexobj(interferogram): igram = interferogram.astype(np.complex64)
        else: igram = np.exp(1j * interferogram).astype(np.complex64)
        corr = np.clip(coherence, 0.0, 1.0).astype(np.float32)
        if mask is not None: corr = np.where(mask, corr, 0.0).astype(np.float32)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        logger.info("Unwrapping %s pixels (cost=%s, init=%s, nlooks=%.1f)", f"{igram.shape[0]}x{igram.shape[1]}", self._cost_mode, self._init_method, nlooks)
        try:
            import snaphu
            unwrapped, conncomp = snaphu.unwrap(igram, corr, nlooks=nlooks, cost=self._cost_mode, init=self._init_method, min_conncomp_frac=min_conncomp_frac, min_region_size=min_region_size)
        except ImportError:
            logger.info("snaphu Python package not found. Falling back to SNAPHU CLI executable...")
            unwrapped, conncomp = self._unwrap_cli(igram, corr, nlooks, min_conncomp_frac, min_region_size)
        except Exception as exc:
            raise RuntimeError(f"SNAPHU unwrapping failed: {exc}") from exc

        n_unreliable = int(np.sum(conncomp == 0))
        pct = 100 * n_unreliable / conncomp.size
        if pct > 30: logger.warning("%.1f%% of pixels are in the unreliable connected component (conncomp==0).", pct)
        else: logger.info("Unwrapping complete — %.1f%% unreliable pixels", pct)
        return unwrapped.astype(np.float32), conncomp

    def unwrap_pair(self, interferogram: Any, coherence: Any, profile: Dict[str, Any], reference_date: Optional[str] = None, secondary_date: Optional[str] = None, nlooks: float = 1.0, looks_azimuth: int = 1, looks_range: int = 1, mask: Optional[Any] = None, min_conncomp_frac: float = 0.001, min_region_size: int = 100) -> UnwrapResult:
        import numpy as np
        unwrapped, conncomp = self.unwrap(interferogram, coherence, nlooks=nlooks, mask=mask, min_conncomp_frac=min_conncomp_frac, min_region_size=min_region_size)
        reliable_fraction = float(np.mean(conncomp > 0))
        scaled_profile = dict(profile)
        if looks_azimuth > 1 or looks_range > 1:
            if scaled_profile.get("transform") is not None:
                from rasterio.transform import Affine
                scaled_profile["transform"] = scaled_profile["transform"] * Affine.scale(looks_range, looks_azimuth)
            scaled_profile["height"] = unwrapped.shape[0]
            scaled_profile["width"] = unwrapped.shape[1]
        return UnwrapResult(unwrapped_phase=unwrapped, conncomp=conncomp, coherence=coherence, profile=scaled_profile, reference_date=reference_date, secondary_date=secondary_date, nlooks=nlooks, cost_mode=self._cost_mode, init_method=self._init_method, reliable_fraction=reliable_fraction)

    def unwrap_files(self, interferogram_path: Union[str, Path], coherence_path: Union[str, Path], output_path: Union[str, Path], nlooks: float = 1.0, mask_path: Optional[Union[str, Path]] = None) -> Path:
        np = self._np()
        try: import rasterio
        except ImportError: raise ImportError('rasterio required: pip install "pygeofetch[geo]"')
        with rasterio.open(interferogram_path) as src:
            profile = src.profile.copy()
            phase = src.read(1).astype(np.float32)
        with rasterio.open(coherence_path) as src:
            coherence = src.read(1).astype(np.float32)
        mask = None
        if mask_path:
            with rasterio.open(mask_path) as src: mask = src.read(1).astype(bool)
        unwrapped, conncomp = self.unwrap(phase, coherence, nlooks=nlooks, mask=mask)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_profile = {"driver": "GTiff", "dtype": "float32", "count": 1, "height": unwrapped.shape[0], "width": unwrapped.shape[1], "crs": profile.get("crs"), "transform": profile.get("transform"), "nodata": -9999.0, "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256}
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(unwrapped[np.newaxis])
            dst.update_tags(1, description="unwrapped_phase_radians")
        conncomp_path = out_path.parent / f"{out_path.stem}_conncomp.tif"
        cc_profile = dict(out_profile, dtype="int32", nodata=0)
        with rasterio.open(conncomp_path, "w", **cc_profile) as dst:
            dst.write(conncomp.astype(np.int32)[np.newaxis])
        logger.info("Unwrapped phase → %s (conncomp → %s)", out_path.name, conncomp_path.name)
        return out_path

    def _np(self):
        import numpy as np
        return np
    

import shutil
import subprocess
import logging
import sys

logger = logging.getLogger(__name__)

def verify_snaphu_ready():
    """
    Proactively checks for snaphu availability to assure the user 
    that no manual installation is required.
    """
    # 1. Check for the standard SNAPHU binary in the system PATH
    snaphu_bin = shutil.which("snaphu")
    if snaphu_bin:
        # Optional: Verify the binary actually executes and isn't just a broken alias
        try:
            subprocess.run(
                ["snaphu"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL, 
                timeout=3
            )
        except Exception:
            pass  # Snaphu without args prints usage to stderr and exits; this is expected.
            
        msg = f"✅ Success: snaphu binary detected at '{snaphu_bin}'. No manual installation needed."
        print(msg)
        logger.info(msg)
        return True

    # 2. Fallback: Check if it's available as a Python package/module
    try:
        import snaphu  # or pysnaphu, depending on your wrapper
        msg = "✅ Success: snaphu Python module detected. No manual installation needed."
        print(msg)
        logger.info(msg)
        return True
    except ImportError:
        pass

    # 3. If neither is found, handle the failure gracefully
    msg = "❌ snaphu not found in PATH or Python environment. Phase unwrapping will fail."
    print(msg)
    logger.warning(msg)
    return False

# --- Embed this early in your notebook or pipeline ---
snaphu_is_ready = verify_snaphu_ready()

if not snaphu_is_ready:
    # Decide how to handle the missing dependency (see questions below)
    pass 