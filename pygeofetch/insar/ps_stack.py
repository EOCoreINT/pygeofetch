"""
Master-referenced amplitude/phase stacks for persistent-scatterer (PS) selection.

``ps_selection`` computes the amplitude dispersion index (ADI) from an amplitude stack but
does not say how to obtain one: every date must sit on the *same ground pixels*. If the
dates are shifted against each other by even one pixel, every pixel's time series mixes
different scatterers, ADI becomes large everywhere and the phase becomes noise. That
failure looks exactly like "PS selection does not work here". This module builds the stack
from ``InterferogramGenerator.process_pair`` results, which already register each
secondary to the master with orbit/DEM geometry and enhanced spectral diversity, and it
**rejects, with a stated reason, every date that cannot be trusted**.

Flow::

    results   = master-referenced interferograms (one process_pair per secondary)
    stack     = PSStack.from_interferograms(results, master)        # or build_ps_stack(...)
    selection = select_ps(stack, [ADISelector(), my_selector])      # any object with .select(stack)
    report    = compare_selections(stack, selection, reference="adi")

A model-based selector (for example PS-GNN) lives in its own package and only has to
provide ``name`` and ``select(stack) -> PSSelection``. This module never imports it.

Requirements on how the interferograms were made (checked, not assumed):

* one master as reference for every date, same crop (identical grid),
* full resolution (``looks_azimuth = looks_range = 1``) and no Goldstein filter, so the
  secondary amplitude is recovered exactly: the interferogram is ``ref * conj(sec)``, the
  phase corrections only change its phase, hence ``|sec| = |ifg| / |ref|``,
* flat-earth phase removed (done automatically when SAFE zips and orbit files are given),
* orbit/DEM registration, not the shape-based fallback.

Limits
------
* Topographic phase is removed by ``process_pair`` only when its phase-versus-elevation
  regression is convincing (R^2 gate); low-coherence pairs usually fail that gate. Where
  it was not removed the phase keeps a height term proportional to the perpendicular
  baseline. The report gives ``topo_removed`` and ``topo_rad_per_100m_relief`` per date.
* The recovered amplitude assumes the reference amplitude is the same crop for every date. Where the master
  amplitude is below about 3 % of its median (power 1e-3), the secondary amplitude cannot be recovered
  reliably; ``frac_master_amp_floored`` in the report says how many pixels that is.
* ``residual_phase_coherence`` is a reference-free stability measure (neighbourhood
  mean removed, so smooth atmosphere/deformation do not count against a pixel). It is a
  screening diagnostic, not ground truth.
"""

from __future__ import annotations

import datetime as _dt
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np

logger = logging.getLogger("pygeofetch.insar.ps_stack")

PathLike = Union[str, Path]

WAVELENGTH_M = 0.05546576  # Sentinel-1 C-band


# ---------------------------------------------------------------------------------------------------
# small numerical helpers
# ---------------------------------------------------------------------------------------------------
def boxcar(a: np.ndarray, looks: Tuple[int, int]) -> np.ndarray:
    """Block-average ``a`` by ``looks = (azimuth, range)``; the remainder rows/columns are dropped."""
    la, lr = looks
    h, w = (a.shape[0] // la) * la, (a.shape[1] // lr) * lr
    return a[:h, :w].reshape(h // la, la, w // lr, lr).mean(axis=(1, 3))


def topo_phase_per_100m_relief(
    bperp_m: float,
    wavelength_m: float = WAVELENGTH_M,
    slant_range_m: float = 850e3,
    incidence_deg: float = 39.0,
) -> float:
    """Phase (rad) that 100 m of unmodelled height produces: ``4*pi/lambda * B / (R*sin(theta)) * 100``.

    Sentinel-1 IW mid-swath geometry; use it to judge whether leaving topography in matters,
    not to correct it.
    """
    return float(
        4 * np.pi / wavelength_m * abs(bperp_m) / (slant_range_m * np.sin(np.deg2rad(incidence_deg))) * 100.0
    )


def stability_weights(amplitude: np.ndarray, adi_scale: float = 0.4) -> np.ndarray:
    """Per-pixel weights in [0, 1] that favour amplitude-stable pixels (low ADI), used to estimate the smooth phase.

    ``amplitude`` is (T, h, w) linear. This is the usual first step of PS processing: the spatially smooth
    phase (atmosphere, orbit ramp, deformation) is estimated from pixels that look stable in amplitude, because
    in a decorrelated scene the pixels around a point scatterer carry no usable phase.
    """
    from pygeofetch.insar.ps_selection import compute_amplitude_dispersion_index

    adi = compute_amplitude_dispersion_index(amplitude)
    return (1.0 - np.clip(adi / adi_scale, 0.0, 1.0)) ** 2


def residual_phase_coherence(
    phase: np.ndarray, weights: Optional[np.ndarray] = None, sigma: float = 6.0
) -> np.ndarray:
    """Temporal coherence of the phase left after removing the smooth phase around each pixel.

    ``phase`` is (T, h, w) wrapped, master-referenced. For each date the smooth signal at a pixel is the
    weighted mean unit phasor of its neighbours (Gaussian window ``sigma`` pixels, the pixel itself
    excluded); the pixel's coherence is ``|mean_t exp(j * (phase_t - smooth_t))|``. Close to 1 = the pixel
    follows its surroundings with little noise; close to 0 = its phase is erratic.

    ``weights`` (h, w) says which neighbours define the smooth signal; use :func:`stability_weights`.
    With ``weights=None`` all pixels count equally, which only works where most pixels are coherent: in a
    decorrelated scene the neighbours of a point scatterer are noise and the measure is then close to
    uninformative. It is a screening diagnostic, not ground truth.
    """
    from scipy import ndimage

    T, h, w = phase.shape
    wt = np.ones((h, w)) if weights is None else np.asarray(weights, np.float64)
    delta = np.zeros((h, w))
    delta[h // 2, w // 2] = 1.0
    k0 = float(ndimage.gaussian_filter(delta, sigma, mode="constant")[h // 2, w // 2])
    norm = ndimage.gaussian_filter(wt, sigma, mode="constant") - wt * k0
    acc = np.zeros((h, w), np.complex128)
    for t in range(T):
        z = np.exp(1j * phase[t].astype(np.float64))
        num = (ndimage.gaussian_filter((wt * z).real, sigma, mode="constant") - wt * z.real * k0) + 1j * (
            ndimage.gaussian_filter((wt * z).imag, sigma, mode="constant") - wt * z.imag * k0
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            nm = np.where(norm > 1e-6, num / norm, 0)
        mag = np.abs(nm)
        smooth = np.where(mag > 1e-9, nm / np.maximum(mag, 1e-12), 1.0)
        acc += z * np.conj(smooth)
    return (np.abs(acc) / T).astype(np.float32)


def choose_master(dates: Sequence[str], platform_by_date: Optional[Mapping[str, str]] = None) -> str:
    """The date that minimises the longest temporal baseline (the middle date).

    With ``platform_by_date`` only dates of the most frequent satellite are candidates, since a
    master from the other satellite would pair every date across the burst-timing boundary.
    """
    ds = sorted(dates)
    if platform_by_date:
        counts: Dict[str, int] = {}
        for d in ds:
            counts[platform_by_date.get(d, "?")] = counts.get(platform_by_date.get(d, "?"), 0) + 1
        main = max(counts, key=counts.get)
        ds = [d for d in ds if platform_by_date.get(d, "?") == main]
    days = [_dt.date.fromisoformat(d) for d in ds]
    best = min(range(len(ds)), key=lambda i: max(abs((days[i] - x).days) for x in days))
    return ds[best]


# ---------------------------------------------------------------------------------------------------
# the stack
# ---------------------------------------------------------------------------------------------------
@dataclass
class PSStack:
    """Common-grid, master-referenced amplitude and phase stack.

    ``amplitude`` is linear (as ``ps_selection`` expects) and radiometrically balanced so that
    every date has the same median; selectors that need another scale rescale it themselves.
    ``phase`` is wrapped interferometric phase against the master. The master itself is not in the
    arrays (its phase is zero by definition); its amplitude is ``master_amplitude``.
    """

    amplitude: np.ndarray  # (T, h, w) float32, linear, median-balanced per date
    phase: np.ndarray  # (T, h, w) float32, wrapped, master-referenced
    dates: List[str]  # dates in the arrays (master and rejected dates excluded)
    master: str
    looks: Tuple[int, int]  # (azimuth, range) boxcar applied to the full-resolution grid
    report: Any  # pandas DataFrame, one row per secondary date
    master_amplitude: Optional[np.ndarray] = None  # (h, w) float32, same balance
    rejected: List[str] = field(default_factory=list)
    transform: Any = None  # affine of the stack grid (None if the inputs carried no georeferencing)
    crs: Any = None
    wavelength_m: float = WAVELENGTH_M

    # ----------------------------------------------------------------- views
    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.amplitude.shape)  # type: ignore[return-value]

    @property
    def amplitude_with_master(self) -> np.ndarray:
        """Amplitude stack including the master, as ADI should see it (T+1, h, w)."""
        if self.master_amplitude is None:
            return self.amplitude
        return np.concatenate([self.master_amplitude[None], self.amplitude], axis=0)

    def subset(self, dates: Sequence[str]) -> "PSStack":
        """A stack with only ``dates`` (a subset of ``self.dates``), e.g. for hold-out tests."""
        idx = [self.dates.index(d) for d in dates]
        rep = self.report[self.report["date"].isin(dates)] if self.report is not None else self.report
        return PSStack(
            self.amplitude[idx], self.phase[idx], list(dates), self.master, self.looks, rep,
            self.master_amplitude, list(self.rejected), self.transform, self.crs, self.wavelength_m,
        )

    # --------------------------------------------------------------- geometry
    def pixel_to_map(self, rows, cols):
        """Stack pixel indices -> map coordinates of the pixel centres (needs ``transform``)."""
        if self.transform is None:
            raise ValueError("this stack has no georeferencing (the interferograms carried no transform)")
        x, y = self.transform * (np.asarray(cols) + 0.5, np.asarray(rows) + 0.5)
        return np.asarray(x), np.asarray(y)

    def pixel_to_full_resolution(self, rows, cols):
        """Stack pixel indices -> pixel indices of the full-resolution crop (centre of each box)."""
        return (np.asarray(rows) + 0.5) * self.looks[0], (np.asarray(cols) + 0.5) * self.looks[1]

    # --------------------------------------------------------------- build
    @classmethod
    def from_interferograms(
        cls,
        results: Mapping[str, Any],
        master: str,
        looks: Tuple[int, int] = (1, 3),
        **rules: Any,
    ) -> "PSStack":
        """Build from ``{secondary date: InterferogramResult}`` made against ``master``. See :func:`build_stack`."""
        return build_stack(results.items(), master, looks=looks, **rules)


@dataclass
class _Rules:
    require_flat_earth: bool = True
    require_topo: bool = False
    reject_registration_fallback: bool = True
    require_orbit_registration: bool = True
    reject_cross_satellite: bool = True
    require_burst_sync: bool = False
    min_gcps: Optional[int] = None
    max_coreg_rms_px: Optional[float] = None
    require_full_resolution: bool = True
    require_science_grade: bool = False
    min_valid_fraction: float = 0.5
    bperp: Optional[Mapping[str, float]] = None


def _ingest(date: str, master: str, r: Any, looks: Tuple[int, int], rules: _Rules):
    """One secondary: validate it, recover its amplitude and return ``(amp, phase, a_ref_int, row)``.

    Raises ``ValueError`` with the reason when the date must be rejected.
    """
    meta = dict(getattr(r, "metadata", None) or {})
    row: Dict[str, Any] = {"date": date}
    tb = getattr(r, "temporal_baseline_days", None)
    if tb is None:
        try:
            tb = abs((_dt.date.fromisoformat(date) - _dt.date.fromisoformat(master)).days)
        except ValueError:
            tb = None
    coh = np.asarray(r.coherence, dtype=float)
    coh = coh[np.isfinite(coh)]
    bp = getattr(r, "perpendicular_baseline_m", None)
    if bp is None and rules.bperp is not None:
        bp = rules.bperp.get(date)
    row.update(
        temporal_baseline_days=tb, bperp_m=bp,
        topo_rad_per_100m_relief=topo_phase_per_100m_relief(bp) if bp is not None else np.nan,
        mean_coherence=float(coh.mean()) if coh.size else np.nan,
        frac_coh_gt_0_5=float((coh > 0.5).mean()) if coh.size else np.nan,
        burst_sync_offset_ms=meta.get("burst_sync_offset_ms"),
        burst_sync_ok=meta.get("burst_sync_within_requirement"),
        esd_shift_px=getattr(r, "esd_azimuth_shift_px", None),
        coreg_method=meta.get("coregistration_method"), coreg_gcps=meta.get("coregistration_gcps_final"),
        coreg_rms_px=meta.get("coregistration_rms_mean_px"),
        flat_earth_removed=meta.get("flat_earth_phase_removed"), topo_removed=meta.get("topographic_phase_removed"),
        topo_r_squared=meta.get("topographic_phase_r_squared"),
        cross_satellite=meta.get("cross_satellite"), cross_track=meta.get("cross_track"),
        quality_grade=meta.get("quality_grade"), quality_flags=",".join((meta.get("quality") or {}).get("flags", [])),
        esd_residual_px=meta.get("esd_residual_after_correction_px"),
    )
    grade = meta.get("quality_grade")
    if grade == "degraded":
        raise ValueError("pair is graded 'degraded': " + row["quality_flags"])
    if rules.require_science_grade and grade is not None and grade != "science":
        raise ValueError(f"pair is graded '{grade}', not 'science': {row['quality_flags']}")
    if rules.require_flat_earth and not meta.get("flat_earth_phase_removed"):
        raise ValueError(
            "flat-earth phase was not removed: pass both SAFE zips and both orbit files to process_pair "
            "(removal is automatic when they are given)"
        )
    method = str(meta.get("coregistration_method") or "")
    if rules.reject_registration_fallback and ("fallback" in method or meta.get("coregistration_fallback")):
        raise ValueError(f"registration fell back to shape-based resampling ({method})")
    if rules.require_orbit_registration and method and "orbit" not in method and "fallback" not in method:
        raise ValueError(f"registration method '{method}' is not orbit/DEM based")
    if rules.reject_cross_satellite and meta.get("cross_satellite"):
        raise ValueError("secondary is from a different Sentinel-1 satellite than the master")
    if meta.get("cross_track"):
        raise ValueError("secondary is on a different relative orbit than the master")
    if rules.require_burst_sync and meta.get("burst_sync_within_requirement") is False:
        raise ValueError(f"burst timing offset {meta.get('burst_sync_offset_ms')} ms is outside the requirement")
    if rules.min_gcps is not None and (meta.get("coregistration_gcps_final") or 0) < rules.min_gcps:
        raise ValueError(f"only {meta.get('coregistration_gcps_final')} registration control points (< {rules.min_gcps})")
    if rules.max_coreg_rms_px is not None and (meta.get("coregistration_rms_mean_px") or 0) > rules.max_coreg_rms_px:
        raise ValueError(f"registration RMS {meta.get('coregistration_rms_mean_px')} px (> {rules.max_coreg_rms_px})")
    if rules.require_topo and not meta.get("topographic_phase_removed"):
        raise ValueError("topographic phase was not removed (no DEM, or no orbit geometry to simulate it from)")
    if rules.require_full_resolution:
        la, lr = meta.get("looks_azimuth"), meta.get("looks_range")
        if (la not in (None, 1)) or (lr not in (None, 1)):
            raise ValueError(f"interferogram was multilooked ({la}x{lr}); amplitude cannot be recovered. Use 1x1")
        if meta.get("goldstein_applied"):
            raise ValueError("a Goldstein filter was applied; amplitude cannot be recovered")
    z = np.asarray(r.interferogram)
    a_ref = 10.0 ** (np.asarray(r.amplitude, dtype=np.float64) / 20.0)  # process_pair stores 20*log10|ref|
    if z.shape != a_ref.shape:
        raise ValueError(f"interferogram {z.shape} and amplitude {a_ref.shape} differ in shape")
    valid = np.isfinite(z) & np.isfinite(a_ref) & (a_ref > 0)
    if valid.mean() < rules.min_valid_fraction:
        raise ValueError(f"only {100 * valid.mean():.0f}% valid pixels")
    i_ref = np.where(valid, a_ref, 0.0) ** 2
    floor = 1e-3 * float(np.median(i_ref[valid]))
    i_sec = np.abs(np.where(valid, z, 0)) ** 2 / np.maximum(i_ref, floor)
    row["frac_master_amp_floored"] = float((valid & (i_ref < floor)).sum() / max(valid.sum(), 1))
    amp = np.sqrt(boxcar(np.where(valid, i_sec, 0.0), looks))
    phase = np.angle(boxcar(np.where(valid, z, 0), looks)).astype(np.float32)
    a_ref_ml = np.sqrt(boxcar(i_ref, looks))
    return amp, phase, a_ref_ml, row, (r.profile if hasattr(r, "profile") else None)


class _Skipped:
    """Stand-in result for a date that was not processed; ``_ingest`` turns it into a rejection."""

    def __init__(self, reason: str):
        self.reason = reason

    @property
    def metadata(self):  # pragma: no cover - never read, _ingest raises first
        return {}


def build_stack(
    items: Union[Mapping[str, Any], Sequence[Tuple[str, Any]], Any],
    master: str,
    looks: Tuple[int, int] = (1, 3),
    **rule_kwargs: Any,
) -> PSStack:
    """Master-referenced stack from ``(date, InterferogramResult)`` pairs (a dict's ``.items()`` also works).

    ``items`` may be a generator, so results can be produced and released one at a time.
    Rules (keyword arguments, all checked per date; a date failing one is rejected and listed in
    ``report`` with the reason, never silently dropped):

    ``require_flat_earth=True``, ``reject_registration_fallback=True``, ``require_orbit_registration=True``,
    ``reject_cross_satellite=True``, ``require_burst_sync=False``, ``min_gcps=None``, ``max_coreg_rms_px=None``,
    ``require_topo=False``, ``require_full_resolution=True``, ``require_science_grade=False``, ``min_valid_fraction=0.5``,
    ``bperp`` (date -> metres, fills the baseline when a result carries none).
    """
    import pandas as pd

    rules = _Rules(**rule_kwargs)
    if hasattr(items, "items"):
        items = items.items()  # type: ignore[assignment]
    amps: List[np.ndarray] = []
    phases: List[np.ndarray] = []
    kept: List[str] = []
    rejected: List[str] = []
    rows: List[Dict[str, Any]] = []
    shape = None
    a_ref_ml = None
    transform = crs = None
    for date, r in items:  # type: ignore[misc]
        row: Dict[str, Any] = {"date": date}
        try:
            if isinstance(r, _Skipped):
                raise ValueError(r.reason)
            amp, phase, a_ref_here, row, profile = _ingest(date, master, r, looks, rules)
            if shape is None:
                shape = amp.shape
                a_ref_ml = a_ref_here
                if profile:
                    transform, crs = profile.get("transform"), profile.get("crs")
            elif amp.shape != shape:
                raise ValueError(f"grid {amp.shape} differs from the stack's {shape}: not cropped identically")
            amps.append(amp)
            phases.append(phase)
            kept.append(date)
            row["status"] = "ok"
        except ValueError as exc:
            row["status"] = f"REJECTED: {exc}"
            rejected.append(date)
            logger.warning("%s rejected: %s", date, exc)
        rows.append(row)
    report = pd.DataFrame(rows)
    if not kept:
        raise RuntimeError(
            "No date could be used for the stack; see the report (stack.report is not available because "
            "construction failed, so the reasons are:\n" + "\n".join(f"  {r['date']}: {r['status']}" for r in rows)
            + "\n). Do not continue to PS selection."
        )
    amp_arr = np.stack(amps).astype(np.float32)
    med = np.array([np.median(a) for a in amp_arr])
    target = float(np.median(med))
    amp_arr *= (target / np.maximum(med, 1e-12))[:, None, None]  # radiometric balance across dates
    master_amp = None
    if a_ref_ml is not None:
        master_amp = (a_ref_ml * target / max(float(np.median(a_ref_ml)), 1e-12)).astype(np.float32)
    out_transform = None
    if transform is not None and not getattr(transform, "is_identity", False):
        out_transform = transform * transform.__class__.scale(looks[1], looks[0])
    return PSStack(amp_arr, np.stack(phases), kept, master, looks, report, master_amp, rejected, out_transform, crs)


def build_ps_stack(
    generator: Any,
    slc_paths: Mapping[str, PathLike],
    master: str,
    *,
    dem: Optional[PathLike] = None,
    safe_zips: Optional[Mapping[str, PathLike]] = None,
    orbit_files: Optional[Mapping[str, PathLike]] = None,
    aoi_bbox: Any = None,
    looks: Tuple[int, int] = (1, 3),
    same_satellite_only: bool = True,
    process_kwargs: Optional[Mapping[str, Any]] = None,
    output_dir: Optional[PathLike] = None,
    resume: bool = True,
    **rule_kwargs: Any,
) -> PSStack:
    """Run ``generator.process_pair`` for every secondary against ``master`` and build the stack.

    Each result is converted and released before the next pair is processed, so memory holds one
    full-resolution pair at a time. ``generator`` is an ``InterferogramGenerator``. The pairs are made
    at full resolution without a Goldstein filter, as the amplitude recovery requires.

    With ``output_dir`` each date's pair is saved to ``output_dir/<master>_<date>`` as soon as it is processed, and
    with ``resume=True`` (default) a date already saved with the same inputs and settings is loaded from disk instead
    of being processed again, so a run that was interrupted continues from the last finished date. The saved pairs are
    full-resolution (about 16 bytes per pixel plus GeoTIFFs), so allow the disk space; plots are not written.

    With ``same_satellite_only`` (default) a secondary from the other Sentinel-1 satellite than the
    master is skipped *before* processing and listed as rejected (it would be a cross-satellite pair
    whose burst timing is not guaranteed to match).
    """
    from pygeofetch.insar.interferogram import _s1_platform_and_track

    safe_zips = safe_zips or {}
    orbit_files = orbit_files or {}
    kw = dict(process_kwargs or {})
    for forbidden in ("looks_azimuth", "looks_range", "apply_goldstein_filter"):
        kw.pop(forbidden, None)
    # Every pair here is single-look, the heaviest setting. Without chunking process_pair reads and processes each
    # full SLC in one piece, several times the memory. Chunking changes nothing in the result.
    if getattr(generator, "_chunk_size", None) is None:
        kw.setdefault("chunk_rows", 500)
    kw.setdefault("use_chunked_processing", True)
    save_kw = {} if output_dir is None else dict(output_dir=output_dir, resume=resume, save_visualizations=False)
    for k in save_kw:
        kw.pop(k, None)
    m_plat = _s1_platform_and_track(safe_zips[master])[0] if master in safe_zips else None

    def _gen():
        for d in sorted(slc_paths):
            if d == master:
                continue
            if same_satellite_only and m_plat and d in safe_zips:
                d_plat = _s1_platform_and_track(safe_zips[d])[0]
                if d_plat and d_plat != m_plat:
                    yield d, _Skipped(f"secondary is {d_plat}, master is {m_plat} (same_satellite_only)")
                    continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # reasons are recorded in the report, not repeated per pair
                    res = generator.process_pair(
                        reference=slc_paths[master], secondary=slc_paths[d], dem=dem,
                        reference_date=master, secondary_date=d,
                        reference_safe_zip=safe_zips.get(master), secondary_safe_zip=safe_zips.get(d),
                        reference_orbit_file=orbit_files.get(master), secondary_orbit_file=orbit_files.get(d),
                        looks_azimuth=1, looks_range=1, apply_goldstein_filter=False,
                        aoi_bbox=aoi_bbox, **save_kw, **kw,
                    )
            except MemoryError:
                raise  # not a property of this date: stop here so the message reaches the user
            except Exception as exc:  # noqa: BLE001
                # ProcessingError carries its stage; anything else is reported with its type so it is not mistaken for a rule
                yield d, _Skipped(f"process_pair failed: {exc}" if hasattr(exc, "stage")
                                  else f"process_pair failed with {type(exc).__name__}: {exc}")
                continue
            yield d, res
            del res

    return build_stack(_gen(), master, looks=looks, **rule_kwargs)


# ---------------------------------------------------------------------------------------------------
# selection: any selector (ADI, a model, ...) returns a PSSelection
# ---------------------------------------------------------------------------------------------------
@dataclass
class PSSelection:
    """Output of one selector on one stack."""

    name: str
    mask: np.ndarray  # (h, w) bool
    score: np.ndarray  # (h, w) float, higher = more PS-like (probability for a model, 1-ADI for ADI)
    params: Dict[str, Any] = field(default_factory=dict)
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)  # extra maps (for example the ADI map)

    @property
    def n_selected(self) -> int:
        return int(self.mask.sum())


@runtime_checkable
class PSSelector(Protocol):
    """What a selector must provide. No torch (or anything else) is required by this package."""

    name: str

    def select(self, stack: PSStack) -> PSSelection:  # pragma: no cover - interface
        ...


class ADISelector:
    """Classical amplitude-dispersion selection (Ferretti et al. 2001) via ``ps_selection``.

    ``include_master`` adds the master amplitude to the dates ADI is computed over (recommended: it is
    a real observation of the same pixels).
    """

    name = "ADI"

    def __init__(self, adi_threshold: float = 0.25, amplitude_percentile: float = 70.0, include_master: bool = True):
        self.adi_threshold = adi_threshold
        self.amplitude_percentile = amplitude_percentile
        self.include_master = include_master

    def select(self, stack: PSStack) -> PSSelection:
        from pygeofetch.insar.ps_selection import select_persistent_scatterers

        amp = stack.amplitude_with_master if self.include_master else stack.amplitude
        res = select_persistent_scatterers(
            amp, adi_threshold=self.adi_threshold, amplitude_percentile=self.amplitude_percentile
        )
        return PSSelection(
            self.name, np.asarray(res.ps_mask, bool), (1.0 - np.asarray(res.adi)).astype(np.float32),
            {"adi_threshold": self.adi_threshold, "amplitude_percentile": self.amplitude_percentile,
             "n_dates": int(amp.shape[0])},
            {"adi": np.asarray(res.adi), "mean_amplitude": np.asarray(res.mean_amplitude)},
        )


def select_ps(stack: PSStack, selectors: Sequence[PSSelector]) -> Dict[str, PSSelection]:
    """Run each selector on the stack; returns ``{selector.name: PSSelection}`` (names must be unique)."""
    out: Dict[str, PSSelection] = {}
    for s in selectors:
        sel = s.select(stack)
        if sel.mask.shape != stack.shape[1:]:
            raise ValueError(f"selector {s.name!r} returned a mask of shape {sel.mask.shape}, stack is {stack.shape[1:]}")
        if sel.name in out:
            raise ValueError(f"two selectors are called {sel.name!r}")
        out[sel.name] = sel
    return out


def compare_selections(stack: PSStack, selections: Mapping[str, PSSelection], reference: str = "ADI",
                       sigma: float = 6.0):
    """Per selection: how many pixels and how phase-stable they are, and how each differs from ``reference``.

    Stability is :func:`residual_phase_coherence` over the whole stack, with the smooth phase estimated from
    amplitude-stable pixels (so it is not independent of ADI: use :func:`holdout_stability` for a fair test).
    For every other selection
    the pixels *added* (selected, not in the reference) and *dropped* (in the reference, not selected)
    are scored separately, so you can see whether the extra pixels are stable or noise. Returns a
    DataFrame. This uses the same dates the selection used; for an unbiased check use
    :func:`holdout_stability`.
    """
    import pandas as pd

    coh = residual_phase_coherence(stack.phase, stability_weights(stack.amplitude_with_master), sigma)
    ref = selections[reference].mask if reference in selections else None
    rows = []
    for name, s in selections.items():
        m = s.mask
        rows.append({"selection": name, "n": int(m.sum()), "share_pct": round(100 * float(m.mean()), 3),
                     "mean_residual_coherence": float(coh[m].mean()) if m.any() else np.nan,
                     "median_residual_coherence": float(np.median(coh[m])) if m.any() else np.nan})
        if ref is not None and name != reference:
            for label, g in ((f"{name} added to {reference}", m & ~ref), (f"{name} dropped from {reference}", ref & ~m)):
                rows.append({"selection": label, "n": int(g.sum()), "share_pct": round(100 * float(g.mean()), 3),
                             "mean_residual_coherence": float(coh[g].mean()) if g.any() else np.nan,
                             "median_residual_coherence": float(np.median(coh[g])) if g.any() else np.nan})
    rows.append({"selection": "all pixels", "n": int(coh.size), "share_pct": 100.0,
                 "mean_residual_coherence": float(coh.mean()), "median_residual_coherence": float(np.median(coh))})
    return pd.DataFrame(rows).set_index("selection").round(3)


def holdout_stability(stack: PSStack, selectors: Sequence[PSSelector], n_eval: int = 3, fraction: float = 0.06,
                      sigma: float = 6.0, seed: int = 0):
    """Do selected pixels stay stable on dates the selector never saw?

    The last ``n_eval`` dates are held out. Each selector picks the top ``fraction`` of pixels by
    ``score`` from the remaining dates only (a random pick is the baseline). The residual phase
    coherence of those pixels is then measured on the selection dates ("in-sample") and on the held-out
    dates. A selection that follows real stable scatterers keeps its advantage over random on the
    held-out dates; one that picked pixels where noise happened to line up falls back to the random
    level. ``neighbour_ratio`` = selected neighbours of a selected pixel relative to random placement
    (about 1 = scattered, clearly above 1 = clustered on structures).
    """
    import pandas as pd
    from scipy import ndimage

    T = len(stack.dates)
    if n_eval < 3 or T - n_eval < 3:
        raise ValueError(f"need at least 3 dates for selection and 3 held out; the stack has {T}")
    sel_stack = stack.subset(stack.dates[: T - n_eval])
    ev_phase = stack.phase[T - n_eval:]
    h, w = stack.shape[1:]
    k = max(1, int(fraction * h * w))
    wts = stability_weights(sel_stack.amplitude_with_master)  # from the selection dates only
    c_in = residual_phase_coherence(sel_stack.phase, wts, sigma)
    c_out = residual_phase_coherence(ev_phase, wts, sigma)
    rng = np.random.default_rng(seed)
    rnd = np.zeros(h * w, bool)
    rnd[rng.choice(h * w, k, replace=False)] = True
    picks = {"random": rnd.reshape(h, w)}
    for s in selectors:
        score = np.nan_to_num(s.select(sel_stack).score.ravel().astype(np.float64), nan=-np.inf)
        m = np.zeros(h * w, bool)
        m[np.argsort(score)[-k:]] = True
        picks[s.name] = m.reshape(h, w)
    rows = {}
    for name, m in picks.items():
        nb = (ndimage.convolve(m.astype(int), np.ones((3, 3), int), mode="constant") - m)[m].mean()
        rows[name] = {"n": int(m.sum()), "coherence_in_sample": c_in[m].mean(), "coherence_held_out": c_out[m].mean(),
                      "neighbour_ratio": nb / max(8 * m.mean(), 1e-9)}
    df = pd.DataFrame(rows).T
    df["held_out_minus_random"] = df["coherence_held_out"] - df.loc["random", "coherence_held_out"]
    return df.round(3)


def to_selection_result(selection: PSSelection, threshold: Optional[float] = None):
    """Convert a selection to ``ps_selection.PSSelectionResult`` so existing downstream code can use it."""
    from pygeofetch.insar.ps_selection import PSSelectionResult

    adi = selection.arrays.get("adi")
    mean_amp = selection.arrays.get("mean_amplitude")
    shape = selection.mask.shape
    return PSSelectionResult(
        adi=adi if adi is not None else np.full(shape, np.nan, np.float32),
        ps_mask=selection.mask,
        mean_amplitude=mean_amp if mean_amp is not None else np.full(shape, np.nan, np.float32),
        n_candidates=selection.n_selected,
        threshold_used=float(threshold if threshold is not None else selection.params.get("adi_threshold", np.nan)),
    )


def export_points(stack: PSStack, selection: PSSelection, path: PathLike):
    """CSV of the selected pixels: stack and full-resolution crop indices, score, map coordinates if georeferenced."""
    import pandas as pd

    r, c = np.nonzero(selection.mask)
    fr, fc = stack.pixel_to_full_resolution(r, c)
    df = pd.DataFrame({"crop_row": fr, "crop_col": fc, "stack_row": r, "stack_col": c, "score": selection.score[r, c]})
    if stack.transform is not None:
        x, y = stack.pixel_to_map(r, c)
        geographic = "4326" in str(stack.crs)
        df["lon" if geographic else "x"], df["lat" if geographic else "y"] = x, y
    df.to_csv(path, index=False)
    return df


__all__ = [
    "PSStack", "PSSelection", "PSSelector", "ADISelector", "build_stack", "build_ps_stack", "select_ps",
    "compare_selections", "holdout_stability", "residual_phase_coherence", "stability_weights", "choose_master",
    "topo_phase_per_100m_relief", "to_selection_result", "export_points", "boxcar",
]