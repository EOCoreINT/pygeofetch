"""
ps_gnn.integration.pygeofetch_selector -- PS-GNN as a pygeofetch persistent-scatterer selector.

pygeofetch builds the common-grid, master-referenced stack (orbit/DEM registration and ESD, rejection of
dates that cannot be trusted) and defines the selector protocol; this file is the only place that knows
how the model was trained (amplitude scale, land-cover input, MC-dropout, seeding).

Usage::

    from pygeofetch.insar import build_ps_stack, ADISelector, select_ps, compare_selections
    from ps_gnn.models.bundle import PSGNNBundle
    from ps_gnn.integration.pygeofetch_selector import PSGNNSelector

    stack = build_ps_stack(generator, slc_paths, master, dem=dem_path, safe_zips=safe, orbit_files=orbits,
                           aoi_bbox=aoi_bbox, looks=(1, 3))
    gnn = PSGNNSelector(PSGNNBundle.load(bundle_dir))
    selections = select_ps(stack, [ADISelector(), gnn])
    print(compare_selections(stack, selections, reference="ADI"))

Nothing here changes pygeofetch: it only needs ``name`` and ``select(stack) -> PSSelection``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

# Median amplitude of the synthetic training scenes (5-20 across scenes); same value real_stack used.
SYNTHETIC_MEDIAN_AMPLITUDE = 11.0


def rescale_amplitude(stack, amplitude_scale: float = SYNTHETIC_MEDIAN_AMPLITUDE) -> np.ndarray:
    """Amplitude (T, h, w) rescaled so its median is the training scale. Dates are already balanced by pygeofetch."""
    amp = np.asarray(stack.amplitude, np.float32)
    return amp * (amplitude_scale / max(float(np.median(amp)), 1e-12))


class PSGNNSelector:
    """Runs a PS-GNN bundle on a pygeofetch ``PSStack``."""

    name = "PS-GNN"

    def __init__(self, bundle: Any, incidence_deg: float = 39.0, worldcover: Optional[np.ndarray] = None,
                 tile_size: int = 128, overlap: int = 16, mc_dropout_passes: int = 10, seed: int = 0,
                 amplitude_scale: float = SYNTHETIC_MEDIAN_AMPLITUDE):
        self.bundle, self.incidence_deg, self.worldcover = bundle, incidence_deg, worldcover
        self.tile_size, self.overlap, self.mc_dropout_passes = tile_size, overlap, mc_dropout_passes
        self.seed, self.amplitude_scale = seed, amplitude_scale

    def select(self, stack):
        import torch
        from pygeofetch.insar.ps_stack import PSSelection

        h, w = stack.shape[1:]
        wc = self.worldcover
        if wc is None:  # no land-cover map: the land-cover features are uninformative (constant class)
            wc = np.full((h, w), 50, dtype=np.int32)
        torch.manual_seed(self.seed)  # MC-dropout is stochastic; seeding makes repeated runs agree
        det = self.bundle.detector(tile_size=self.tile_size, overlap=self.overlap,
                                   mc_dropout_passes=self.mc_dropout_passes)
        out = det.detect(amplitude=rescale_amplitude(stack, self.amplitude_scale), phase=stack.phase,
                         incidence_angle=self.incidence_deg, worldcover=wc)
        prob = np.asarray(out["probability"], np.float32)
        return PSSelection(
            self.name, np.asarray(out["ps_mask"], bool), prob,
            {"tile_size": self.tile_size, "overlap": self.overlap, "mc_dropout_passes": self.mc_dropout_passes,
             "seed": self.seed, "threshold": getattr(self.bundle, "threshold", None),
             "worldcover": "constant (none supplied)" if self.worldcover is None else "supplied"},
            {"probability": prob},
        )


def to_prepared_stack(stack, amplitude_scale: float = SYNTHETIC_MEDIAN_AMPLITUDE):
    """A ``ps_gnn.validation.real_stack.PreparedStack`` view of a pygeofetch stack.

    Lets ps-gnn's model-specific diagnostics (``rescue_report``, ``holdout_test``, ``feature_scan``,
    ``rescue_breakdown`` and the plots) keep working unchanged after the stack building moved to pygeofetch.
    """
    from ps_gnn.validation.real_stack import PreparedStack

    h, w = stack.shape[1:]
    la, lr = stack.looks
    return PreparedStack(
        amplitude=rescale_amplitude(stack, amplitude_scale), phase=np.asarray(stack.phase, np.float32),
        dates=list(stack.dates), master=stack.master, roi=(0, h * la, 0, w * lr), looks=tuple(stack.looks),
        report=stack.report, rejected=list(stack.rejected), transform=stack.transform, crs=stack.crs,
    )
