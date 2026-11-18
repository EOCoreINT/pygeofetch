"""Learned phase unwrapping with a trained pyunwrap ``AmbiguityNet``.

What this module owns (and what it leaves to pyunwrap)
------------------------------------------------------
pyunwrap owns the network architecture and its training. pygeofetch owns everything between a pygeofetch
interferogram and the network: reading and checking the checkpoint, building the network input exactly as in
training, tiling, merging the tiles, deciding which pixels can be trusted, and reporting what was done.
The only thing taken from the ``pyunwrap`` package is the ``AmbiguityNet`` class, and only when a checkpoint is
loaded; the preprocessing, the tile grid and the blend window are mirrored here (amplitude normalisation and the
tile grid were compared numerically with pyunwrap's when this was written), so inference does not depend on how
pyunwrap's own inference class evolves.

The network predicts an integer cycle map ``k``; the unwrapped phase is rebuilt as ``phi = psi + 2*pi*k`` from the
*observed* wrapped phase, so re-wrapping the result always returns the input.

Why the tile merge here is not pyunwrap's ``PhaseUnwrapper._merge_tiles``
------------------------------------------------------------------------
Each tile's ``k`` is only defined up to a whole number of cycles. pyunwrap's merge averages ``k`` across tiles
without aligning them; the training run's own results table records the effect ("full-scene, no alignment":
44.8 % against 88.2 % aligned on the easy scenes). Here every tile is shifted by the median whole-cycle
difference to what is already merged, measured on valid, coherent pixels only.

Error policy (same as the rest of ``pygeofetch.insar``)
-------------------------------------------------------
* A requested step that fails raises a ``ProcessingError`` subclass (``ModelLoadError``,
  ``UnwrapDependencyError``, ``UnwrapError``, ``InputError``) with the cause chained.
* Nothing is guessed silently: a checkpoint does not record ``k_max``, the tile size or the multilook scales, so
  those come from the run's ``config.json`` or an explicit ``LearnedUnwrapConfig``; anything that had to be
  assumed is listed in ``model.info["assumed"]`` and raises the flag ``model_config_assumed`` on every result.
* Every result carries a ``QualityRecord`` (``result.metadata["quality"]``).

Limits, stated plainly
----------------------
* No ground truth exists for real scenes. The diagnostics here (``k_spread`` between overlapping tiles,
  ``gradient_violation_fraction``) are consistency checks, not accuracy measures.
* A model trained on synthetic scenes with square 20 m pixels has not been shown to work on strongly
  non-square, heavily multilooked pixels or at coherence below about 0.4. The checks flag these cases;
  they do not make the model work there.
* ``LearnedUnwrapConfig.real_data_validated`` is ``False`` until you set it, and every result carries the flag
  ``model_not_validated_on_real_data`` until then.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .errors import (
    InputError,
    ModelLoadError,
    QualityRecord,
    UnwrapDependencyError,
    UnwrapError,
)
from .unwrap import UnwrapResult, multilook

logger = logging.getLogger("pygeofetch.insar.learned_unwrap")

TWO_PI = 2.0 * np.pi
ENCODER_STRIDE = 32          # AmbiguityNet's ResNet-34 encoder downsamples by 32
_STEM_KEY = "encoder.stem.0.weight"
_MIN_ALIGN_PIXELS = 16       # fewest overlap pixels a tile alignment is allowed to rest on


# ---------------------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LearnedUnwrapConfig:
    """Everything the network input and the tiling depend on, plus the trust gates.

    ``tile``, ``overlap``, ``multilook_sigmas`` and ``k_max`` must be the values the model was trained / evaluated
    with. They are not stored in the checkpoint; ``from_run_config`` reads the first three from a training run's
    ``config.json``. ``k_max`` is not in that file either (the trainer's default is 10.0).
    """

    tile: int = 256
    overlap: int = 128
    multilook_sigmas: Tuple[float, ...] = ()
    k_max: float = 10.0
    batch_size: int = 8
    # trust gates
    min_coherence: float = 0.3            # pixels below this are never reported as reliable
    max_k_spread: float = 0.5             # cycles; overlapping tiles that disagree by more are not reliable
    max_residue_prob: Optional[float] = None   # model's own uncertainty; no default (its calibration is unknown)
    unreliable_warn_fraction: float = 0.30     # same level SNAPHU's wrapper warns at
    max_padding_share: float = 0.5        # share of the padded grid that is no-data padding before it is flagged
    # warnings about inputs the model was not trained for
    warn_median_coherence: float = 0.4
    trained_pixel_m: float = 20.0
    max_pixel_aspect: float = 2.0
    pixel_scale_tolerance: float = 2.0    # geometric-mean pixel size within [1/x, x] times trained_pixel_m
    # state of validation, set by the user, never by the code
    real_data_validated: bool = False

    def __post_init__(self) -> None:
        if self.tile <= 0 or self.tile % ENCODER_STRIDE:
            raise ValueError(f"tile={self.tile} must be a positive multiple of {ENCODER_STRIDE} (the encoder stride)")
        if not 0 <= self.overlap < self.tile:
            raise ValueError(f"overlap={self.overlap} must satisfy 0 <= overlap < tile ({self.tile})")
        if self.k_max <= 0:
            raise ValueError("k_max must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0.0 <= self.min_coherence <= 1.0:
            raise ValueError("min_coherence must be in [0, 1]")
        object.__setattr__(self, "multilook_sigmas", tuple(float(s) for s in self.multilook_sigmas))

    @property
    def in_channels(self) -> int:
        return 3 + 2 * len(self.multilook_sigmas)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["in_channels"] = self.in_channels
        return d


def _read_run_config(path: Path) -> Tuple[Dict[str, Any], List[str]]:
    """Fields of a training run's ``config.json`` that matter for inference, and which of them were found."""
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ModelLoadError("model_load", f"cannot read the run config {path}: {exc}") from exc
    fields: Dict[str, Any] = {}
    if "tile" in raw:
        fields["tile"] = int(raw["tile"])
    if "infer_overlap" in raw:
        fields["overlap"] = int(raw["infer_overlap"])
    if "multilook" in raw:
        fields["multilook_sigmas"] = tuple(float(s) for s in (raw["multilook"] or ()))
    for key in ("k_max",):
        if key in raw:
            fields[key] = float(raw[key])
    return fields, sorted(fields)


def _discover_run_config(checkpoint: Path) -> Optional[Path]:
    for cand in (checkpoint.parent / "config.json", checkpoint.parent.parent / "config.json"):
        if cand.is_file():
            return cand
    return None


# ---------------------------------------------------------------------------------------------------------
# network input (mirrors pyunwrap.data.preprocessing; parity-tested)
# ---------------------------------------------------------------------------------------------------------


def normalize_amplitude(amplitude: np.ndarray, valid: Optional[np.ndarray] = None,
                        percentile_clip: Tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    """Log-scale and percentile-clip amplitude to [0, 1]; identical to pyunwrap's when every pixel is valid.

    The percentiles are taken over valid pixels only, so a large no-data area does not set the scale.
    """
    amp = np.clip(np.asarray(amplitude, dtype=np.float64), 1e-6, None)
    log_amp = np.log1p(amp)
    ref = log_amp if valid is None or not np.any(valid) else log_amp[valid]
    lo, hi = np.percentile(ref, percentile_clip)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((log_amp - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _multilook_channels(psi: np.ndarray, coh: np.ndarray, sigmas: Sequence[float]) -> np.ndarray:
    """Coherence-weighted Gaussian average of exp(i*psi), real and imaginary part per scale (as in training)."""
    from scipy.ndimage import gaussian_filter

    z, w = np.exp(1j * psi) * coh, coh.astype(np.float64)
    out = []
    for sg in sigmas:
        den = gaussian_filter(w, sg, mode="reflect") + 1e-6
        out += [gaussian_filter(z.real, sg, mode="reflect") / den, gaussian_filter(z.imag, sg, mode="reflect") / den]
    return np.stack(out).astype(np.float32)


def build_channels(psi: np.ndarray, coh: np.ndarray, amp: np.ndarray, valid: np.ndarray,
                   sigmas: Sequence[float] = ()) -> np.ndarray:
    """Stack the network input ``[C, H, W]``: phase/pi, coherence, normalized amplitude (+ multilook channels).

    Pixels that are not valid get the no-data convention of the training data: phase 0, coherence 0, amplitude 0.
    """
    psi0 = np.where(valid, psi, 0.0)
    coh0 = np.where(valid, np.clip(coh, 0.0, 1.0), 0.0)
    amp_n = np.where(valid, normalize_amplitude(np.where(valid, amp, 0.0), valid), 0.0).astype(np.float32)
    x = np.stack([(psi0 / np.pi).astype(np.float32), coh0.astype(np.float32), amp_n])
    if sigmas:
        x = np.concatenate([x, _multilook_channels(psi0, coh0, sigmas)], axis=0)
    return x.astype(np.float32)


# ---------------------------------------------------------------------------------------------------------
# tiling (mirrors pyunwrap.data.preprocessing.compute_tile_grid and pyunwrap.inference.unwrapper windows)
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    height: int
    width: int


def tile_grid(shape: Tuple[int, int], tile: int, overlap: int) -> List[Tile]:
    """Row-major grid of full-size tiles covering ``shape`` (the last row/column is shifted inward)."""
    rows, cols = shape
    if rows < tile or cols < tile:
        raise ValueError(f"shape {shape} is smaller than the tile ({tile}); pad first")
    stride = tile - overlap

    def starts(extent: int) -> List[int]:
        s = list(range(0, extent - tile + 1, stride))
        if s[-1] != extent - tile:
            s.append(extent - tile)
        return s

    return [Tile(r, c, tile, tile) for r in starts(rows) for c in starts(cols)]


def _ramp(length: int, overlap: int, taper_start: bool, taper_end: bool) -> np.ndarray:
    win = np.ones(length, dtype=np.float64)
    ov = max(int(overlap), 1)
    ov = min(ov, length // 2) if length >= 2 else 0
    if ov == 0:
        return win
    up = 0.5 - 0.5 * np.cos(np.pi * (np.arange(ov) + 1) / (ov + 1))
    if taper_start:
        win[:ov] = up
    if taper_end:
        win[-ov:] = up[::-1]
    return win


def tile_window(t: Tile, overlap: int, shape: Tuple[int, int]) -> np.ndarray:
    """Blend weights for one tile; only edges that border another tile are tapered."""
    rows, cols = shape
    rw = _ramp(t.height, overlap, t.row > 0, t.row + t.height < rows)
    cw = _ramp(t.width, overlap, t.col > 0, t.col + t.width < cols)
    return np.outer(rw, cw)


# ---------------------------------------------------------------------------------------------------------
# predictors: anything that maps [N, C, H, W] float32 -> (k [N, H, W], residue_prob [N, H, W])
# ---------------------------------------------------------------------------------------------------------

Predictor = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


class TorchPredictor:
    """Runs a loaded ``AmbiguityNet`` in eval mode, without gradients."""

    def __init__(self, model: Any, device: str):
        import torch

        self._torch = torch
        self.model = model
        self.device = device

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        if self.model.training:
            self.model.eval()
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            out = self.model(t)
        return (out.k_hat[:, 0].detach().cpu().numpy().astype(np.float32),
                out.residue_prob[:, 0].detach().cpu().numpy().astype(np.float32))


@dataclass
class UnwrapModel:
    """A loaded model together with the configuration it must be run with and where it came from."""

    predictor: Predictor
    config: LearnedUnwrapConfig
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def unverified_assumptions(self) -> List[str]:
        """Settings taken from defaults that nothing has checked. ``k_max`` counts as checked once the
        self-test passed: a wrong value rescales every cycle count and fails it. Tile size, overlap and multilook
        scales cannot be checked that way."""
        passed = self.info.get("selftest", {}).get("passed") is True
        return [a for a in self.info.get("assumed", []) if not (a == "k_max" and passed)]

    @property
    def flags(self) -> List[str]:
        out = []
        if self.unverified_assumptions:
            out.append("model_config_assumed")
        if self.info.get("selftest", {}).get("passed") is False:
            out.append("model_selftest_failed")
        if not self.config.real_data_validated:
            out.append("model_not_validated_on_real_data")
        return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise UnwrapDependencyError(
            "model_load", "torch is not installed; it is needed to run a pyunwrap checkpoint "
                          "(install a build that matches your hardware from pytorch.org)") from exc
    return torch


def _default_model_factory(cfg: LearnedUnwrapConfig):
    try:
        from pyunwrap.models.ambiguity_net import AmbiguityNet
    except ImportError as exc:
        raise UnwrapDependencyError(
            "model_load", "the pyunwrap package is not importable; it provides the AmbiguityNet class the "
                          "checkpoint was trained with (install it, or pass model_factory=...)") from exc
    return AmbiguityNet(in_channels=cfg.in_channels, pretrained=False, k_max=cfg.k_max)


def _resolve_device(torch, device: Optional[str]) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ModelLoadError("model_load", f"device={device!r} was requested but CUDA is not available")
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ModelLoadError("model_load", "device='mps' was requested but MPS is not available")
    return device


def load_unwrap_model(
    checkpoint: Union[str, Path],
    *,
    config: Optional[LearnedUnwrapConfig] = None,
    run_config: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    allow_unsafe_pickle: bool = False,
    model_factory: Optional[Callable[[LearnedUnwrapConfig], Any]] = None,
    selftest: str = "warn",
) -> UnwrapModel:
    """Load a pyunwrap checkpoint for inference.

    Configuration is taken, in this order, from ``config`` (explicit), the training run's ``config.json``
    (``run_config``, or a ``config.json`` next to the checkpoint or one directory above), a ``config`` dict stored in
    the checkpoint, and finally the defaults. What came from the defaults is listed in ``info["assumed"]``.

    The checkpoint is read with ``weights_only=True`` (no arbitrary code runs). If that fails the error says so;
    pass ``allow_unsafe_pickle=True`` only for a file you wrote yourself.

    ``selftest``: ``"warn"`` (default) runs one noise-light synthetic tile with known cycles and records the
    result; ``"strict"`` raises if it fails; ``"off"`` skips it. The pass level is a heuristic, not a calibrated
    threshold (see ``_selftest``).
    """
    if selftest not in ("off", "warn", "strict"):
        raise ValueError("selftest must be 'off', 'warn' or 'strict'")
    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        raise ModelLoadError("model_load", f"checkpoint not found: {ckpt}")
    torch = _import_torch()
    dev = _resolve_device(torch, device)

    # -- read -------------------------------------------------------------------------------------------
    try:
        blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception as exc:
        if not allow_unsafe_pickle:
            raise ModelLoadError(
                "model_load",
                f"{ckpt.name} could not be read with weights_only=True ({type(exc).__name__}: {exc}). Loading it "
                f"would run arbitrary code stored in the file; pass allow_unsafe_pickle=True only if you created "
                f"this checkpoint yourself.") from exc
        try:
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception as exc2:
            raise ModelLoadError("model_load", f"cannot read {ckpt}: {exc2}") from exc2

    state = blob
    if isinstance(blob, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in blob and isinstance(blob[key], Mapping):
                state = blob[key]
                break
    if not isinstance(state, Mapping) or not state:
        raise ModelLoadError("model_load", f"{ckpt.name} does not contain a state dict")
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    if _STEM_KEY not in state:
        raise ModelLoadError("model_load", f"{ckpt.name} is not an AmbiguityNet checkpoint "
                                           f"(no {_STEM_KEY!r}; first keys: {list(state)[:3]})")
    ckpt_channels = int(state[_STEM_KEY].shape[1])

    # -- configuration ----------------------------------------------------------------------------------
    assumed: List[str] = []
    if config is not None:
        cfg, source = config, "explicit"
    else:
        found: Dict[str, Any] = {}
        rc = Path(run_config) if run_config else _discover_run_config(ckpt)
        if run_config and not rc.is_file():
            raise ModelLoadError("model_load", f"run config not found: {rc}")
        embedded = blob.get("config") if isinstance(blob, Mapping) else None
        if rc is not None:
            found, _ = _read_run_config(rc)
            source = f"run_config:{rc}"
        elif isinstance(embedded, Mapping):
            allowed = set(LearnedUnwrapConfig.__dataclass_fields__)
            found = {k: v for k, v in embedded.items() if k in allowed and k != "real_data_validated"}
            source = "checkpoint"
        else:
            source = "defaults"
        cfg = LearnedUnwrapConfig(**found)
        assumed = [f for f in ("tile", "overlap", "multilook_sigmas", "k_max") if f not in found]

    if cfg.in_channels != ckpt_channels:
        msg = f"the configuration implies {cfg.in_channels} input channels but the checkpoint has {ckpt_channels}"
        if ckpt_channels > 3 and not cfg.multilook_sigmas:
            msg += (f", i.e. {(ckpt_channels - 3) // 2} multilook scale(s) whose sigma values are not recorded in it; "
                    f"pass LearnedUnwrapConfig(multilook_sigmas=...) with the values used in training")
        raise ModelLoadError("model_load", msg)

    # -- build and load ---------------------------------------------------------------------------------
    factory = model_factory or _default_model_factory
    try:
        model = factory(cfg)
    except UnwrapError:
        raise
    except Exception as exc:
        raise ModelLoadError("model_load", f"cannot build the network: {type(exc).__name__}: {exc}") from exc
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        # torch puts the useful part (missing / unexpected / size-mismatch keys) after a one-line header
        detail = " ".join(ln.strip() for ln in str(exc).splitlines() if ln.strip())
        raise ModelLoadError(
            "model_load", f"the checkpoint does not match the network (architecture or pyunwrap version "
                          f"differs): {detail[:600]}") from exc
    model.to(dev).eval()

    info: Dict[str, Any] = {
        "checkpoint": str(ckpt), "sha256": _sha256(ckpt), "device": dev, "config_source": source,
        "assumed": assumed, "in_channels": ckpt_channels,
        "epoch": blob.get("epoch") if isinstance(blob, Mapping) else None,
        "val_rmse_rad": (float(blob["val_rmse_rad"]) if isinstance(blob, Mapping) and "val_rmse_rad" in blob
                         and blob["val_rmse_rad"] is not None else None),
    }
    um = UnwrapModel(TorchPredictor(model, dev), cfg, info)
    if selftest != "off":
        info["selftest"] = _selftest(um)
    info["assumed_unverified"] = um.unverified_assumptions
    if um.unverified_assumptions:
        logger.warning("%s: %s not found in %s - defaults used (tile=%d overlap=%d k_max=%g)", ckpt.name,
                       ", ".join(um.unverified_assumptions), source, cfg.tile, cfg.overlap, cfg.k_max)
    if selftest != "off":
        if not info["selftest"]["passed"]:
            msg = (f"self-test failed: {info['selftest']['exact_cycle_fraction']:.2f} of the cycles of a synthetic "
                   f"tile were recovered (need >= {info['selftest']['required']:.2f}); wrong k_max, wrong input "
                   f"channels or a mismatched checkpoint are the usual causes")
            if selftest == "strict":
                raise ModelLoadError("model_load", msg)
            logger.warning("%s: %s", ckpt.name, msg)
    return um


def _selftest(um: UnwrapModel, required: float = 0.8) -> Dict[str, Any]:
    """One synthetic tile with known cycles (a Gaussian bowl plus a ramp, small phase noise, speckled amplitude).

    A wrong ``k_max`` rescales every prediction, wrong channels or normalisation scramble them, and a mismatched
    checkpoint is noise; all of these fail here before a real scene is touched. ``required`` is a heuristic, chosen
    well below the 96-98 % exact cycles the training run reports on its easy scenes; it is not calibrated.
    """
    cfg = um.config
    n = cfg.tile
    rng = np.random.default_rng(12345)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    bowl = 4.0 * TWO_PI * np.exp(-(((yy - n / 2) ** 2 + (xx - n / 2) ** 2) / (2 * (n / 6.0) ** 2)))
    ramp = 1.5 * TWO_PI * (xx / n)
    truth = bowl + ramp
    psi = np.angle(np.exp(1j * (truth + rng.normal(0, 0.15, truth.shape))))
    k_true = np.round((truth - psi) / TWO_PI)
    coh = np.full((n, n), 0.95)
    amp = rng.rayleigh(100.0, (n, n))
    x = build_channels(psi, coh, amp, np.ones((n, n), bool), cfg.multilook_sigmas)
    try:
        k, _ = um.predictor(x[None])
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        return {"passed": False, "exact_cycle_fraction": 0.0, "required": required,
                "error": f"{type(exc).__name__}: {exc}"}
    k = np.asarray(k[0], dtype=np.float64)
    k = k - np.round(np.median(k - k_true))                     # a whole-cycle constant is not an error
    frac = float(np.mean(k == k_true))
    return {"passed": bool(frac >= required), "exact_cycle_fraction": frac, "required": required}


# ---------------------------------------------------------------------------------------------------------
# diagnostics that need no ground truth
# ---------------------------------------------------------------------------------------------------------


def unwrap_diagnostics(unwrapped: np.ndarray, wrapped: Optional[np.ndarray] = None,
                       mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Consistency checks that work for any unwrapper's output (SNAPHU or learned), so they can be compared.

    ``gradient_violation_fraction``: share of neighbouring pixel pairs, both inside ``mask``, whose unwrapped
    difference exceeds pi. A smooth, correctly unwrapped field has almost none; a wrong cycle jump creates a line of
    them. It is a symptom check, not an accuracy measure (a real fringe rate above pi per pixel also counts).
    ``wrap_error_max``: largest deviation of ``wrap(unwrapped - wrapped)`` from zero, radians.
    """
    u = np.asarray(unwrapped, dtype=np.float64)
    m = np.isfinite(u) if mask is None else (np.asarray(mask, bool) & np.isfinite(u))
    out: Dict[str, float] = {}
    n_pairs = 0
    n_bad = 0
    for ax in (0, 1):
        a = np.take(u, range(0, u.shape[ax] - 1), axis=ax)
        b = np.take(u, range(1, u.shape[ax]), axis=ax)
        ma = np.take(m, range(0, u.shape[ax] - 1), axis=ax) & np.take(m, range(1, u.shape[ax]), axis=ax)
        n_pairs += int(ma.sum())
        n_bad += int((np.abs(b - a)[ma] > np.pi).sum())
    out["gradient_pairs"] = float(n_pairs)
    out["gradient_violation_fraction"] = (n_bad / n_pairs) if n_pairs else float("nan")
    if wrapped is not None:
        w = np.asarray(wrapped, dtype=np.float64)
        d = np.angle(np.exp(1j * (u - w)))
        out["wrap_error_max"] = float(np.abs(d[m]).max()) if m.any() else float("nan")
    return out


# ---------------------------------------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------------------------------------


@dataclass
class LearnedUnwrapResult(UnwrapResult):
    """``UnwrapResult`` plus the model's own maps. ``conncomp`` is 0 where the result is not trusted."""

    ambiguity_map: Any = None      # integer cycle map k
    residue_prob: Any = None       # the model's uncertainty, blended over tiles
    k_spread: Any = None           # std of k between overlapping tiles after alignment, cycles

    def save(self, output_dir: Union[str, Path], save_png: bool = False, mask_unreliable: bool = True) -> Dict[str, Path]:
        paths = super().save(output_dir, save_png=save_png, mask_unreliable=mask_unreliable)
        import rasterio

        out = Path(output_dir)
        base = {
            "driver": "GTiff", "count": 1, "height": self.unwrapped_phase.shape[0],
            "width": self.unwrapped_phase.shape[1], "crs": self.profile.get("crs"),
            "transform": self.profile.get("transform"), "compress": "deflate", "dtype": "float32", "nodata": -9999.0,
        }
        for name in ("ambiguity_map", "residue_prob", "k_spread"):
            arr = getattr(self, name)
            if arr is None:
                continue
            path = out / f"{name}.tif"
            with rasterio.open(path, "w", **base) as dst:
                dst.write(np.nan_to_num(np.asarray(arr, np.float32), nan=-9999.0)[np.newaxis])
            paths[name] = path
        return paths


# ---------------------------------------------------------------------------------------------------------
# the unwrapper
# ---------------------------------------------------------------------------------------------------------


def _as_wrapped_phase(interferogram: Any) -> np.ndarray:
    arr = np.asarray(interferogram)
    if arr.ndim != 2:
        raise InputError("unwrap_inputs", f"the interferogram must be 2-D, got shape {arr.shape}")
    if np.iscomplexobj(arr):
        return np.angle(arr).astype(np.float64)
    arr = arr.astype(np.float64)
    finite = np.isfinite(arr)
    if finite.any() and float(np.abs(arr[finite]).max()) > np.pi + 1e-3:
        raise InputError(
            "unwrap_inputs", f"real-valued input must be wrapped phase in [-pi, pi]; its largest magnitude is "
                             f"{float(np.abs(arr[finite]).max()):.3f} rad (already unwrapped, or in degrees?)")
    return arr


class LearnedUnwrapper:
    """Unwrap a wrapped interferogram with a trained pyunwrap network.

    The call signatures follow ``PhaseUnwrapper`` so a loop written for SNAPHU can use this class by changing the
    constructor and the keyword that carries the amplitude.

        unwrapper = LearnedUnwrapper.from_checkpoint("model/best.pt", run_config="model/config.json")
        res = unwrapper.unwrap_pair(phase_ml, coh_ml, profile, amplitude=amp_ml, pixel_size_m=(451.0, 29.0))
    """

    def __init__(self, model: UnwrapModel):
        self.model = model
        self.config = model.config

    @classmethod
    def from_checkpoint(cls, checkpoint: Union[str, Path], **kwargs: Any) -> "LearnedUnwrapper":
        return cls(load_unwrap_model(checkpoint, **kwargs))

    # -- core -------------------------------------------------------------------------------------------
    def _predict_batches(self, x: np.ndarray, tiles: List[Tile]) -> Any:
        """Yield ``(tile, k, residue)`` for every tile, shrinking the batch if the device runs out of memory."""
        bs = self.config.batch_size
        t = self.config.tile
        i = 0
        while i < len(tiles):
            chunk = tiles[i:i + bs]
            batch = np.stack([x[:, s.row:s.row + t, s.col:s.col + t] for s in chunk])
            try:
                k, r = self.model.predictor(batch)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and bs > 1:
                    bs = max(1, bs // 2)
                    logger.warning("out of memory; batch size reduced to %d", bs)
                    continue
                raise UnwrapError("unwrap", f"the network failed on tiles {i}-{i + len(chunk) - 1}: "
                                            f"{type(exc).__name__}: {exc}") from exc
            except Exception as exc:  # noqa: BLE001
                raise UnwrapError("unwrap", f"the network failed on tiles {i}-{i + len(chunk) - 1}: "
                                            f"{type(exc).__name__}: {exc}") from exc
            k, r = np.asarray(k, dtype=np.float64), np.asarray(r, dtype=np.float64)
            if k.shape != (len(chunk), t, t) or r.shape != k.shape:
                raise UnwrapError("unwrap", f"the network returned shapes {k.shape}/{r.shape}, expected "
                                            f"{(len(chunk), t, t)}")
            if not (np.isfinite(k).all() and np.isfinite(r).all()):
                raise UnwrapError("unwrap", f"the network returned non-finite values for tiles "
                                            f"{i}-{i + len(chunk) - 1}")
            for s, kk, rr in zip(chunk, k, r):
                yield s, kk, rr
            i += len(chunk)

    def predict_k(self, psi: np.ndarray, coh: np.ndarray, amp: np.ndarray, valid: np.ndarray) -> Dict[str, Any]:
        """Integer cycle map for one scene, tiles aligned and blended. Returns arrays cropped to the scene."""
        cfg = self.config
        H, W = psi.shape
        x = build_channels(psi, coh, amp, valid, cfg.multilook_sigmas)
        Hp, Wp = max(H, cfg.tile), max(W, cfg.tile)
        if (Hp, Wp) != (H, W):                                     # scenes smaller than a tile: no-data padding
            x = np.pad(x, ((0, 0), (0, Hp - H), (0, Wp - W)))
        vpad = np.pad(valid, ((0, Hp - H), (0, Wp - W)))
        cpad = np.pad(np.where(valid, np.clip(coh, 0, 1), 0.0), ((0, Hp - H), (0, Wp - W)))

        tiles = [s for s in tile_grid((Hp, Wp), cfg.tile, cfg.overlap)
                 if vpad[s.row:s.row + s.height, s.col:s.col + s.width].any()]
        k_acc, k2_acc, r_acc, w_acc = (np.zeros((Hp, Wp)) for _ in range(4))
        n_unaligned = 0
        first = True
        for s, k, r in self._predict_batches(x, tiles):
            rs, cs = slice(s.row, s.row + s.height), slice(s.col, s.col + s.width)
            w_old = w_acc[rs, cs]
            if not first:
                trusted = (w_old > 0.3) & vpad[rs, cs] & (cpad[rs, cs] >= cfg.min_coherence)
                if trusted.sum() < _MIN_ALIGN_PIXELS:
                    trusted = (w_old > 0.3) & vpad[rs, cs]
                if trusted.sum() >= _MIN_ALIGN_PIXELS:
                    k = k - np.round(np.median(k[trusted] - k_acc[rs, cs][trusted] / w_old[trusted]))
                else:
                    n_unaligned += 1
            first = False
            wt = tile_window(s, cfg.overlap, (Hp, Wp)) * np.clip(1.0 - r, 0.05, 1.0)
            k_acc[rs, cs] += k * wt
            k2_acc[rs, cs] += k * k * wt
            r_acc[rs, cs] += r * wt
            w_acc[rs, cs] += wt
        w = np.clip(w_acc, 1e-8, None)
        k_mean = k_acc / w
        spread = np.sqrt(np.clip(k2_acc / w - k_mean ** 2, 0.0, None))
        k_int = np.round(k_mean)
        return {
            "k": k_int[:H, :W], "residue": (r_acc / w)[:H, :W], "spread": spread[:H, :W],
            "n_tiles": len(tiles), "n_unaligned": n_unaligned, "pad_share": 1.0 - (H * W) / float(Hp * Wp),
        }

    # -- public API -------------------------------------------------------------------------------------
    def unwrap_pair(
        self,
        interferogram: Any,
        coherence: Any,
        profile: Dict[str, Any],
        reference_date: Optional[str] = None,
        secondary_date: Optional[str] = None,
        nlooks: float = 1.0,
        looks_azimuth: int = 1,
        looks_range: int = 1,
        amplitude: Optional[Any] = None,
        mask: Optional[Any] = None,
        pixel_size_m: Optional[Tuple[float, float]] = None,
        min_region_size: int = 100,
    ) -> LearnedUnwrapResult:
        """Unwrap one pair.

        ``interferogram``: complex interferogram or wrapped phase in radians. ``coherence`` in [0, 1].
        ``amplitude``: linear amplitude of the same grid; taken from ``abs(interferogram)`` when that is complex,
        otherwise a constant (the amplitude channel is then uninformative and the result is flagged).
        ``pixel_size_m``: ``(azimuth, range)`` size of one pixel of THIS array in metres; used to warn when the
        pixels are far from what the model was trained on.
        """
        cfg = self.config
        q = QualityRecord(on_error="raise")
        for f in self.model.flags:
            q.flag(f)
        q.ok("model", f"{Path(self.model.info.get('checkpoint', 'model')).name} on {self.model.info.get('device', '?')}",
             **{k: self.model.info.get(k) for k in ("sha256", "epoch", "config_source") if self.model.info.get(k) is not None})

        # -- inputs -------------------------------------------------------------------------------------
        t0 = time.time()
        psi = _as_wrapped_phase(interferogram)
        coh = np.asarray(coherence, dtype=np.float64)
        if coh.shape != psi.shape:
            raise InputError("unwrap_inputs", f"coherence shape {coh.shape} differs from the interferogram {psi.shape}")
        if amplitude is not None:
            amp = np.asarray(amplitude, dtype=np.float64)
            if amp.shape != psi.shape:
                raise InputError("unwrap_inputs", f"amplitude shape {amp.shape} differs from the interferogram {psi.shape}")
        elif np.iscomplexobj(np.asarray(interferogram)):
            amp = np.abs(np.asarray(interferogram))
        else:
            amp = np.ones(psi.shape)
            q.flag("amplitude_not_supplied")
        valid = np.isfinite(psi) & np.isfinite(coh) & np.isfinite(amp)
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if m.shape != psi.shape:
                raise InputError("unwrap_inputs", f"mask shape {m.shape} differs from the interferogram {psi.shape}")
            valid &= m
        if not valid.any():
            raise InputError("unwrap_inputs", "no valid pixel (everything is NaN or masked)")
        q.ok("inputs", f"{psi.shape[0]}x{psi.shape[1]} px, {100 * valid.mean():.1f}% valid")

        med_coh = float(np.median(np.clip(coh[valid], 0, 1)))
        if med_coh < cfg.warn_median_coherence:
            q.flag("low_coherence_scene")
            q.ok("coherence", f"median coherence {med_coh:.2f} < {cfg.warn_median_coherence:.2f}: below the range "
                              f"this model was shown to work in")
        else:
            q.ok("coherence", f"median coherence {med_coh:.2f}")

        if pixel_size_m is None:
            q.skipped("pixel_geometry", "pixel size not supplied; the model was trained on square "
                                        f"{cfg.trained_pixel_m:g} m pixels and this was not checked")
        else:
            az, rg = float(pixel_size_m[0]), float(pixel_size_m[1])
            aspect = max(az, rg) / max(min(az, rg), 1e-9)
            gm = float(np.sqrt(az * rg))
            ratio = gm / cfg.trained_pixel_m
            if aspect > cfg.max_pixel_aspect or not (1.0 / cfg.pixel_scale_tolerance <= ratio <= cfg.pixel_scale_tolerance):
                q.flag("pixel_geometry_outside_training")
                q.ok("pixel_geometry", f"pixels {az:.0f} x {rg:.0f} m (aspect {aspect:.1f}:1, {gm:.0f} m geometric "
                                       f"mean) vs {cfg.trained_pixel_m:g} m square in training")
            else:
                q.ok("pixel_geometry", f"pixels {az:.0f} x {rg:.0f} m")

        # -- inference ----------------------------------------------------------------------------------
        out = self.predict_k(psi, coh, amp, valid)
        k = out["k"]
        if out["pad_share"] > cfg.max_padding_share:
            q.flag("heavy_padding")
            q.ok("padding", f"{100 * out['pad_share']:.0f}% of the padded grid is no-data padding (grid {psi.shape} on "
                            f"a {cfg.tile} px tile); the network sees mostly empty input")
        else:
            q.ok("padding", f"{100 * out['pad_share']:.0f}% padding")
        phi = np.where(valid, psi, 0.0) + TWO_PI * k
        offset = int(np.round(np.median(phi[valid]) / TWO_PI))   # a global whole-cycle constant is arbitrary; keep phi near 0
        k = k - offset
        phi = phi - TWO_PI * offset
        unwrapped = np.where(valid, phi, np.nan)
        if out["n_unaligned"]:
            q.flag("tiles_unaligned")
            q.ok("tile_alignment", f"{out['n_unaligned']} of {out['n_tiles']} tiles had no usable overlap with "
                                   f"already-merged tiles, so their absolute cycle count is arbitrary")
        else:
            q.ok("tile_alignment", f"{out['n_tiles']} tiles aligned", n_tiles=out["n_tiles"])

        # -- trust --------------------------------------------------------------------------------------
        good = valid & (coh >= cfg.min_coherence) & (out["spread"] <= cfg.max_k_spread)
        if cfg.max_residue_prob is not None:
            good &= out["residue"] <= cfg.max_residue_prob
        conncomp = self._components(good, min_region_size)
        reliable_fraction = float(np.mean(conncomp > 0))
        if (1.0 - reliable_fraction) > cfg.unreliable_warn_fraction:
            q.flag("high_unreliable_fraction")
            logger.warning("%.1f%% of pixels are not reliable", 100 * (1 - reliable_fraction))
        q.ok("reliability", f"{100 * reliable_fraction:.1f}% of pixels reliable "
                            f"(coherence >= {cfg.min_coherence:g}, tile spread <= {cfg.max_k_spread:g} cycles)")
        diag = unwrap_diagnostics(unwrapped, psi, conncomp > 0)
        if diag.get("wrap_error_max", 0.0) > 1e-6:
            raise UnwrapError("unwrap", f"internal check failed: wrap(unwrapped - wrapped) reaches "
                                        f"{diag['wrap_error_max']:.2e} rad")
        q.ok("consistency", f"{100 * diag['gradient_violation_fraction']:.2f}% of neighbouring pairs differ by > pi"
             if np.isfinite(diag["gradient_violation_fraction"]) else "no reliable neighbouring pairs")

        scaled = dict(profile)
        if looks_azimuth > 1 or looks_range > 1:
            if scaled.get("transform") is not None:
                from rasterio.transform import Affine

                scaled["transform"] = scaled["transform"] * Affine.scale(looks_range, looks_azimuth)
        scaled["height"], scaled["width"] = unwrapped.shape

        meta = {
            "method": "pyunwrap-ambiguitynet", "model": dict(self.model.info), "config": cfg.to_dict(),
            "diagnostics": diag, "quality": q.to_dict(), "seconds": time.time() - t0,
            "tiles": {"n": out["n_tiles"], "unaligned": out["n_unaligned"]}, "k_offset_removed": offset,
            "median_coherence": med_coh,
        }
        return LearnedUnwrapResult(
            unwrapped_phase=unwrapped.astype(np.float32), conncomp=conncomp, coherence=np.asarray(coherence),
            profile=scaled, reference_date=reference_date, secondary_date=secondary_date, nlooks=nlooks,
            cost_mode="learned", init_method="ambiguitynet", reliable_fraction=reliable_fraction, metadata=meta,
            ambiguity_map=k.astype(np.int16), residue_prob=out["residue"].astype(np.float32),
            k_spread=out["spread"].astype(np.float32),
        )

    def unwrap(self, interferogram: Any, coherence: Any, amplitude: Optional[Any] = None,
               mask: Optional[Any] = None, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """``(unwrapped, conncomp)`` like ``PhaseUnwrapper.unwrap``."""
        res = self.unwrap_pair(interferogram, coherence, {}, amplitude=amplitude, mask=mask, **kwargs)
        return res.unwrapped_phase, res.conncomp

    @staticmethod
    def _components(good: np.ndarray, min_region_size: int) -> np.ndarray:
        from scipy.ndimage import label

        lab, n = label(good)
        if n == 0:
            return np.zeros(good.shape, dtype=np.int32)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        keep = np.flatnonzero(sizes >= max(int(min_region_size), 1))
        keep = keep[keep > 0]
        remap = np.zeros(n + 1, dtype=np.int32)
        remap[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
        return remap[lab]


# ---------------------------------------------------------------------------------------------------------
# a whole stack, replacing the notebook loop
# ---------------------------------------------------------------------------------------------------------


def unwrap_interferograms(
    unwrapper: Any,
    interferograms: Mapping[Tuple[str, str], Any],
    *,
    phases: Optional[Mapping[Tuple[str, str], np.ndarray]] = None,
    looks: Tuple[int, int] = (1, 1),
    native_pixel_m: Optional[Tuple[float, float]] = None,
    nlooks: float = 1.0,
    on_error: str = "raise",
    min_region_size: int = 100,
) -> Tuple[Dict[Tuple[str, str], LearnedUnwrapResult], Dict[Tuple[str, str], Exception]]:
    """Unwrap every pair of a stack with a ``LearnedUnwrapper`` or a ``HybridUnwrapper``.

    ``nlooks``: equivalent number of looks of the arrays handed to the unwrapper (SNAPHU's statistical model uses it;
    the network ignores it). For a stack multilooked ``(a, r)`` by the generator and ``(ua, ur)`` before unwrapping it
    is ``a * r * ua * ur``.

    ``interferograms``: pair -> object with ``.interferogram`` (complex), ``.coherence`` and ``.profile``
    (an ``InterferogramResult``). ``phases``: optional pair -> wrapped phase (or complex) that replaces
    ``angle(interferogram)``, e.g. the atmosphere-corrected phase. ``looks``: ``(azimuth, range)`` extra looks taken
    before unwrapping, applied the same way as for SNAPHU. ``native_pixel_m``: ``(azimuth, range)`` pixel size of the
    interferograms; multiplied by ``looks`` to check the pixel geometry the network sees.

    ``on_error="raise"`` stops at the first failing pair; ``"skip"`` records it in the second return value and
    continues. Skipped pairs are never silently dropped: they are returned.
    """
    if on_error not in ("raise", "skip"):
        raise ValueError("on_error must be 'raise' or 'skip'")
    la, lr = int(looks[0]), int(looks[1])
    results: Dict[Tuple[str, str], LearnedUnwrapResult] = {}
    failures: Dict[Tuple[str, str], Exception] = {}
    for pair, ifg in interferograms.items():
        try:
            cplx = np.asarray(ifg.interferogram)
            src = cplx if phases is None or pair not in phases else np.asarray(phases[pair])
            phase = np.angle(src) if np.iscomplexobj(src) else src
            phase_ml = multilook(phase, la, lr, wrapped_phase=True) if (la > 1 or lr > 1) else phase
            coh_ml = multilook(np.asarray(ifg.coherence), la, lr, wrapped_phase=False) if (la > 1 or lr > 1) else ifg.coherence
            amp_ml = multilook(np.abs(cplx), la, lr, wrapped_phase=False) if (la > 1 or lr > 1) else np.abs(cplx)
            px = None if native_pixel_m is None else (native_pixel_m[0] * la, native_pixel_m[1] * lr)
            results[pair] = unwrapper.unwrap_pair(
                phase_ml, coh_ml, ifg.profile, reference_date=pair[0], secondary_date=pair[1], nlooks=nlooks,
                looks_azimuth=la, looks_range=lr, amplitude=amp_ml, pixel_size_m=px, min_region_size=min_region_size)
            r = results[pair]
            logger.info("%s -> %s: %s, reliable %.1f%%, %.1f s", pair[0], pair[1], r.unwrapped_phase.shape,
                        100 * r.reliable_fraction, r.metadata["seconds"])
        except Exception as exc:  # noqa: BLE001 - policy decides
            if on_error == "raise":
                raise
            failures[pair] = exc
            logger.error("%s -> %s skipped: %s", pair[0], pair[1], exc)
    return results, failures
