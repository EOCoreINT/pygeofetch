"""Hybrid phase unwrapping: SNAPHU where it is reliable, the pyunwrap network where it is not.

What is different from pyunwrap's own ``HybridUnwrapper``
---------------------------------------------------------
Both unwrappers return an integer cycle map ``k`` that is only defined up to a whole number of cycles, and SNAPHU
defines it separately for every connected component (a real run on a scene split by a dead band gave two components at
offsets of 0 and -1 cycles from the truth). pyunwrap's hybrid picks ``k`` per pixel from one method or the other
without removing those offsets, so every boundary between the two sources is a cycle jump, and its own "SNAPHU +
do-nothing" control loses 15 points to SNAPHU alone on the hard scenes (47.4 % against 63.2 %).

Here every SNAPHU component is tied to the network's global cycle frame before any pixel is exchanged: the offset is
the most common whole-cycle difference ``k_network - k_snaphu`` over pixels where both are trusted and coherence is
at least ``offset_min_coherence``. A component whose pixels do not agree on one offset (``min_offset_agreement``) or
that is too small is *not* tied; it keeps SNAPHU's own offset, is labelled as such (provenance 3, its own
``conncomp`` label) and raises the flag ``component_offset_uncertain``.

Which method supplies which pixel
---------------------------------
* coherence at or below ``low_coherence``: the network (SNAPHU integrates through noise there);
* pixels SNAPHU left out of every component (``conncomp == 0``): the network, if it trusts them;
* everything else: SNAPHU, shifted into the network's frame.

A pixel is reported as reliable only if the method that supplied it reports it reliable. The network's own gate
(coherence >= 0.3 by default) therefore still applies to the pixels it supplies, so the hybrid completes the field at
very low coherence but does not claim those pixels.

Defaults and what they rest on
------------------------------
``low_coherence = 0.15`` is the threshold chosen on the validation scenes in the training run's hybrid summary
(``low_threshold_chosen_on_val``). Those scenes are synthetic. Nothing here was tuned on real data, and the same
summary shows the tuned hybrid only equal to SNAPHU on its hard test scenes (63.3 % against 63.2 %). Expect the hybrid to
differ from SNAPHU mainly by tying components together and by filling regions SNAPHU dropped.

Error policy: ``on_error="raise"`` (default) raises ``UnwrapError`` if either method fails; ``"degrade"`` returns the
surviving method's result, records the failure and grades the result ``degraded``. If both fail it always raises.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .errors import InputError, QualityRecord, UnwrapError
from .learned_unwrap import (
    TWO_PI,
    LearnedUnwrapper,
    LearnedUnwrapResult,
    _as_wrapped_phase,
    unwrap_diagnostics,
)
from .unwrap import PhaseUnwrapper

logger = logging.getLogger("pygeofetch.insar.hybrid_unwrap")

PROV_NONE, PROV_CLASSICAL, PROV_LEARNED, PROV_CLASSICAL_UNTIED = 0, 1, 2, 3


@dataclass(frozen=True)
class HybridConfig:
    low_coherence: float = 0.15               # at or below: the network supplies k (chosen on synthetic validation scenes)
    learned_fills_classical_gaps: bool = True  # pixels outside every SNAPHU component go to the network
    offset_min_coherence: float = 0.4         # pixels used to tie a component to the network's frame
    min_offset_pixels: int = 200              # fewer trusted pixels than this: the component is not tied
    min_offset_agreement: float = 0.5         # share of those pixels that must sit at the modal offset
    on_error: str = "raise"

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_coherence <= 1.0:
            raise ValueError("low_coherence must be in [0, 1]")
        if not 0.0 < self.min_offset_agreement <= 1.0:
            raise ValueError("min_offset_agreement must be in (0, 1]")
        if self.min_offset_pixels < 1:
            raise ValueError("min_offset_pixels must be >= 1")
        if self.on_error not in ("raise", "degrade"):
            raise ValueError("on_error must be 'raise' or 'degrade'")


@dataclass
class HybridUnwrapResult(LearnedUnwrapResult):
    """``LearnedUnwrapResult`` plus ``provenance``: 0 none, 1 SNAPHU (tied), 2 network, 3 SNAPHU (not tied).

    ``ambiguity_map`` is the merged cycle map. ``conncomp`` is 0 where the supplying method does not trust the pixel.
    """

    provenance: Any = None

    def save(self, output_dir: Union[str, Path], save_png: bool = False, mask_unreliable: bool = True) -> Dict[str, Path]:
        paths = super().save(output_dir, save_png=save_png, mask_unreliable=mask_unreliable)
        if self.provenance is not None:
            import rasterio

            path = Path(output_dir) / "provenance.tif"
            prof = {"driver": "GTiff", "count": 1, "height": self.provenance.shape[0], "width": self.provenance.shape[1],
                    "crs": self.profile.get("crs"), "transform": self.profile.get("transform"), "compress": "deflate",
                    "dtype": "uint8", "nodata": 255}
            with rasterio.open(path, "w", **prof) as dst:
                dst.write(np.asarray(self.provenance, dtype=np.uint8)[np.newaxis])
            paths["provenance"] = path
        return paths


def tie_components(
    k_classical: np.ndarray,
    components: np.ndarray,
    k_learned: np.ndarray,
    trusted: np.ndarray,
    min_pixels: int = 200,
    min_agreement: float = 0.5,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Whole-cycle offset that brings each SNAPHU component into the network's cycle frame.

    ``trusted``: pixels where both methods may be compared. Returns ``(offsets_by_pixel, table)`` where
    ``offsets_by_pixel`` holds the offset to add to ``k_classical`` (0 for components that could not be tied) and
    ``table`` has one row per component: label, pixels, offset, agreement, tied.
    """
    offsets = np.zeros(k_classical.shape, dtype=np.float64)
    table: List[Dict[str, Any]] = []
    for c in np.unique(components[components > 0]):
        in_c = components == c
        m = in_c & trusted
        n = int(m.sum())
        row = {"label": int(c), "pixels": int(in_c.sum()), "trusted_pixels": n, "offset": 0, "agreement": float("nan"),
               "tied": False}
        if n >= min_pixels:
            d = np.rint(k_learned[m] - k_classical[m]).astype(np.int64)
            vals, counts = np.unique(d, return_counts=True)
            j = int(np.argmax(counts))
            agreement = float(counts[j] / n)
            row["agreement"] = agreement
            if agreement >= min_agreement:
                row.update(offset=int(vals[j]), tied=True)
                offsets[in_c] = float(vals[j])
        table.append(row)
    return offsets, table


class HybridUnwrapper:
    """SNAPHU and the network, tied into one cycle frame.

        hybrid = HybridUnwrapper(LearnedUnwrapper.from_checkpoint("model/best.pt", run_config="model/config.json"),
                                 PhaseUnwrapper(cost_mode="defo", init_method="mcf"))
        res = hybrid.unwrap_pair(phase_ml, coh_ml, profile, amplitude=amp_ml, nlooks=256.0)

    The call signature matches ``LearnedUnwrapper.unwrap_pair`` plus the SNAPHU arguments, so ``unwrap_interferograms``
    accepts it in place of the learned unwrapper.
    """

    def __init__(self, learned: LearnedUnwrapper, classical: Optional[PhaseUnwrapper] = None,
                 config: Optional[HybridConfig] = None):
        self.learned = learned
        self.classical = classical if classical is not None else PhaseUnwrapper()
        self.config = config or HybridConfig()

    # -- helpers ----------------------------------------------------------------------------------------
    def _run_classical(self, psi0: np.ndarray, coh: np.ndarray, valid: np.ndarray, profile: Dict[str, Any],
                       nlooks: float, min_conncomp_frac: float, min_region_size: int):
        return self.classical.unwrap_pair(
            psi0, np.where(valid, coh, 0.0), profile, nlooks=nlooks, mask=valid,
            min_conncomp_frac=min_conncomp_frac, min_region_size=min_region_size)

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
        min_conncomp_frac: float = 0.001,
    ) -> HybridUnwrapResult:
        cfg = self.config
        t0 = time.time()
        psi = _as_wrapped_phase(interferogram)
        coh = np.asarray(coherence, dtype=np.float64)
        if coh.shape != psi.shape:
            raise InputError("unwrap_inputs", f"coherence shape {coh.shape} differs from the interferogram {psi.shape}")
        valid = np.isfinite(psi) & np.isfinite(coh)
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if m.shape != psi.shape:
                raise InputError("unwrap_inputs", f"mask shape {m.shape} differs from the interferogram {psi.shape}")
            valid &= m
        if not valid.any():
            raise InputError("unwrap_inputs", "no valid pixel (everything is NaN or masked)")
        psi0 = np.where(valid, psi, 0.0)

        q = QualityRecord(on_error=cfg.on_error)
        cl = le = None
        errors: Dict[str, Exception] = {}

        # -- the two unwrappers ---------------------------------------------------------------------
        try:
            cl = self._run_classical(psi0, coh, valid, profile, nlooks, min_conncomp_frac, min_region_size)
            q.ok("classical", f"SNAPHU: {100 * cl.reliable_fraction:.1f}% of pixels in a component")
        except Exception as exc:  # noqa: BLE001 - policy below decides
            errors["classical"] = exc
        try:
            le = self.learned.unwrap_pair(
                psi, coh, profile, reference_date, secondary_date, nlooks, looks_azimuth, looks_range,
                amplitude=amplitude, mask=mask, pixel_size_m=pixel_size_m, min_region_size=min_region_size)
        except Exception as exc:  # noqa: BLE001
            errors["learned"] = exc

        if len(errors) == 2:
            raise UnwrapError("hybrid", "both unwrappers failed: SNAPHU: "
                                        f"{type(errors['classical']).__name__}: {errors['classical']}; network: "
                                        f"{type(errors['learned']).__name__}: {errors['learned']}") from errors["learned"]
        if errors and cfg.on_error == "raise":
            which, exc = next(iter(errors.items()))
            raise UnwrapError("hybrid", f"the {'SNAPHU' if which == 'classical' else 'network'} unwrapper failed and "
                                        f"on_error='raise': {type(exc).__name__}: {exc}") from exc

        scaled_profile = (le.profile if le is not None else cl.profile)
        if le is None:                                   # network failed, degrade to SNAPHU alone
            q.failed("learned", f"{type(errors['learned']).__name__}: {errors['learned']}", "hybrid_learned_failed")
            return self._single(cl, psi, valid, coh, q, "classical", scaled_profile, reference_date, secondary_date,
                                nlooks, errors, t0, min_region_size)
        for f in le.metadata["quality"]["flags"]:
            q.flag(f)
        q.ok("learned", f"network: {100 * le.reliable_fraction:.1f}% of pixels reliable, grade "
                        f"{le.metadata['quality']['grade']}")
        if cl is None:                                   # SNAPHU failed, degrade to the network alone
            q.failed("classical", f"{type(errors['classical']).__name__}: {errors['classical']}", "hybrid_classical_failed")
            return self._single(le, psi, valid, coh, q, "learned", scaled_profile, reference_date, secondary_date,
                                nlooks, errors, t0, min_region_size)

        # -- tie SNAPHU's components into the network's cycle frame ---------------------------------
        phi_c = np.asarray(cl.unwrapped_phase, dtype=np.float64)
        cc = np.asarray(cl.conncomp)
        k_c = np.where(np.isfinite(phi_c) & valid, np.rint((phi_c - psi0) / TWO_PI), 0.0)
        k_l = np.asarray(le.ambiguity_map, dtype=np.float64)
        trusted = valid & (cc > 0) & (le.conncomp > 0) & (coh >= cfg.offset_min_coherence)
        offsets, table = tie_components(k_c, cc, k_l, trusted, cfg.min_offset_pixels, cfg.min_offset_agreement)
        n_tied = sum(r["tied"] for r in table)
        if table and n_tied < len(table):
            q.flag("component_offset_uncertain")
        if table:
            q.ok("offset_alignment", f"{n_tied} of {len(table)} SNAPHU components tied to the network's cycle frame "
                                     f"(offsets {[r['offset'] for r in table if r['tied']]})")
        else:
            q.flag("no_classical_component")
            q.ok("offset_alignment", "SNAPHU returned no component; the network supplies every pixel")
        tied_px = np.zeros(psi.shape, bool)
        for r in table:
            if r["tied"]:
                tied_px |= cc == r["label"]

        # -- choose a source per pixel ----------------------------------------------------------------
        use_learned = valid & (coh <= cfg.low_coherence)
        if cfg.learned_fills_classical_gaps:
            use_learned |= valid & (cc == 0)
        prov = np.zeros(psi.shape, dtype=np.uint8)
        prov[valid & ~use_learned & tied_px] = PROV_CLASSICAL
        prov[valid & ~use_learned & (cc > 0) & ~tied_px] = PROV_CLASSICAL_UNTIED
        prov[use_learned] = PROV_LEARNED
        k = np.where(prov == PROV_LEARNED, k_l, k_c + offsets)
        unwrapped = np.where(valid, psi0 + TWO_PI * k, np.nan)

        # -- reliability: only where the supplying method trusts the pixel ---------------------------
        rel_aligned = ((prov == PROV_CLASSICAL) & (cc > 0)) | ((prov == PROV_LEARNED) & (le.conncomp > 0))
        conncomp = LearnedUnwrapper._components(rel_aligned, min_region_size)
        nxt = int(conncomp.max()) + 1
        for r in table:
            if not r["tied"]:
                sel = (prov == PROV_CLASSICAL_UNTIED) & (cc == r["label"])
                if sel.any():
                    conncomp[sel] = nxt
                    nxt += 1
        reliable_fraction = float(np.mean(conncomp > 0))
        share = {n: float(np.mean(prov == v)) for n, v in
                 (("classical", PROV_CLASSICAL), ("learned", PROV_LEARNED), ("classical_untied", PROV_CLASSICAL_UNTIED))}
        q.ok("merge", f"{100 * share['classical']:.1f}% SNAPHU, {100 * share['learned']:.1f}% network, "
                      f"{100 * share['classical_untied']:.1f}% SNAPHU not tied; {100 * reliable_fraction:.1f}% reliable")
        both = valid & (cc > 0) & (le.conncomp > 0) & tied_px
        agree = float(np.mean((k_c + offsets)[both] == k_l[both])) if both.any() else float("nan")
        diag = unwrap_diagnostics(unwrapped, psi, conncomp > 0)
        if diag.get("wrap_error_max", 0.0) > 1e-4:
            raise UnwrapError("hybrid", f"internal check failed: wrap(unwrapped - wrapped) reaches {diag['wrap_error_max']:.2e} rad")
        meta = {
            "method": "hybrid-snaphu-pyunwrap", "hybrid_config": asdict(cfg), "components": table,
            "source_share": share, "agreement_where_both_trusted": agree, "diagnostics": diag, "quality": q.to_dict(),
            "model": le.metadata["model"], "learned_config": le.metadata["config"], "seconds": time.time() - t0,
            "classical_reliable_fraction": cl.reliable_fraction, "learned_reliable_fraction": le.reliable_fraction,
            "median_coherence": le.metadata["median_coherence"],
        }
        return HybridUnwrapResult(
            unwrapped_phase=unwrapped.astype(np.float32), conncomp=conncomp, coherence=np.asarray(coherence),
            profile=scaled_profile, reference_date=reference_date, secondary_date=secondary_date, nlooks=nlooks,
            cost_mode="hybrid", init_method="snaphu+ambiguitynet", reliable_fraction=reliable_fraction, metadata=meta,
            ambiguity_map=np.where(valid, k, 0).astype(np.int16), residue_prob=le.residue_prob, k_spread=le.k_spread,
            provenance=prov)

    def _single(self, src, psi, valid, coh, q, which, profile, d1, d2, nlooks, errors, t0, min_region_size):
        """Result from one surviving method (``on_error='degrade'``), in the hybrid result type."""
        psi0 = np.where(valid, psi, 0.0)
        phi = np.asarray(src.unwrapped_phase, dtype=np.float64)
        k = np.where(np.isfinite(phi) & valid, np.rint((phi - psi0) / TWO_PI), 0.0)
        prov_val = PROV_CLASSICAL_UNTIED if which == "classical" else PROV_LEARNED
        prov = np.where(valid, prov_val, PROV_NONE).astype(np.uint8)
        conncomp = np.asarray(src.conncomp)
        unwrapped = np.where(valid, psi0 + TWO_PI * k, np.nan)
        diag = unwrap_diagnostics(unwrapped, psi, conncomp > 0)
        meta = {"method": f"hybrid-degraded-{which}-only", "components": [], "diagnostics": diag, "quality": q.to_dict(),
                "failures": {k_: f"{type(v).__name__}: {v}" for k_, v in errors.items()}, "seconds": time.time() - t0,
                "hybrid_config": asdict(self.config)}
        if which == "learned":
            meta.update(model=src.metadata["model"], learned_config=src.metadata["config"])
        return HybridUnwrapResult(
            unwrapped_phase=unwrapped.astype(np.float32), conncomp=conncomp, coherence=np.asarray(coh), profile=profile,
            reference_date=d1, secondary_date=d2, nlooks=nlooks, cost_mode="hybrid", init_method="degraded",
            reliable_fraction=float(np.mean(conncomp > 0)), metadata=meta, ambiguity_map=np.where(valid, k, 0).astype(np.int16),
            residue_prob=getattr(src, "residue_prob", None), k_spread=getattr(src, "k_spread", None), provenance=prov)
