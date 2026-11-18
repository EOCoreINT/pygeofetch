"""Errors and the per-pair quality record for interferogram processing.

Policy, taken from how the established engines behave:

* ISCE2 (topsApp), GAMMA (S1_coreg_TOPS) and HyP3 stop with an error when a step they were asked to do
  fails (no usable coregistration, ESD that cannot be estimated). They do not hand back a weaker product
  that looks normal. SNAP's Back-Geocoding and Enhanced-Spectral-Diversity operators likewise throw an
  operator exception. dolphin validates its inputs up front and raises.
* A step the caller did not ask for, or could not ask for because an input was not supplied (no orbit files,
  no DEM), is not an error. It is recorded, and the product is labelled so it cannot be mistaken for a
  full-quality one.

So here:

* A step that was **requested (or inputs for it were supplied) and failed** raises a ``ProcessingError``
  subclass, with the stage, the reason and the original exception chained (``raise ... from exc``).
* A step that was **not possible because an input is missing** is recorded as ``skipped`` with the reason.
* Everything ends up in ``QualityRecord``, stored in ``result.metadata["quality"]``, with a grade:
  ``science`` (every correction a deformation or PS analysis needs was applied and passed its check),
  ``quicklook`` (inputs for some correction were not supplied, so the product is for looking only) or
  ``degraded`` (only possible when the generator was told ``on_error="degrade"``: a requested step failed).

All errors subclass ``ValueError`` so existing ``except ValueError`` loops around ``process_pair`` keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ProcessingError(ValueError):
    """A processing step that was asked for failed. ``stage`` names the step; the cause is chained."""

    def __init__(self, stage: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.reason = message
        self.details = details or {}


class InputError(ProcessingError):
    """An input is unusable (wrong shape, no valid data, not complex, mismatched geometry)."""


class CoregistrationError(ProcessingError):
    """The secondary could not be registered to the reference with the requested method, or the result is unreliable."""


class BurstProcessingError(ProcessingError):
    """Burst-aware ESD / deburst failed, or the burst timing of the pair does not meet the stated requirement."""


class FlatEarthError(ProcessingError):
    """Flat-earth phase removal was requested (or its inputs were supplied) and failed."""


class TopographicPhaseError(ProcessingError):
    """A DEM was supplied but topographic phase could not be removed."""


class AtmosphereError(ProcessingError):
    """An atmospheric correction was requested and could not be computed (missing or unreadable model data, no geocoding)."""


class GeometryProcessingError(ProcessingError):
    """The radar geometry of the pair (per-pixel position, incidence, baseline) could not be solved."""


class CropError(ProcessingError):
    """The AOI crop was requested and could not be done (the caller would otherwise get a different extent than asked)."""


class UnwrapError(ProcessingError):
    """Phase unwrapping was requested and failed (a tile could not be predicted, the output is not finite, no valid pixels)."""


class ModelLoadError(UnwrapError):
    """A learned-unwrapping checkpoint could not be loaded or does not match the model it is meant for."""


class UnwrapDependencyError(UnwrapError):
    """A package a learned unwrapper needs (torch, pyunwrap) is not installed."""


# ---------------------------------------------------------------------------------------------------------


@dataclass
class StageRecord:
    stage: str
    status: str                      # "ok" | "skipped" | "failed"
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


# thresholds taken from the ESA TOPSAR guide and the engines named above; each is a parameter, not a constant
AZIMUTH_COREG_REQUIREMENT_PX = 0.001      # ESA: azimuth coregistration better than 0.001 samples (< 3 deg phase error)
BURST_SYNC_REQUIREMENT_MS = 5.0           # ESA: burst synchronisation better than 5 ms
MIN_COREG_GCPS = 6                        # CoregistrationQuality.is_reliable default
MAX_COREG_RMS_PX = 1.0                    # CoregistrationQuality.is_reliable default
ESD_COHERENCE_NOTE = 0.2                  # gate used by the ESD overlap estimator here (ISCE2 uses 0.85)


@dataclass
class QualityRecord:
    """What was done to a pair, what was not, and whether each check passed. One per ``process_pair`` call."""

    stages: List[StageRecord] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    on_error: str = "raise"

    # -- recording ---------------------------------------------------------------------------------
    def ok(self, stage: str, detail: str = "", **data) -> None:
        self.stages.append(StageRecord(stage, "ok", detail, data))

    def skipped(self, stage: str, detail: str, flag: Optional[str] = None, **data) -> None:
        self.stages.append(StageRecord(stage, "skipped", detail, data))
        if flag:
            self.flags.append(flag)

    def failed(self, stage: str, detail: str, flag: str, **data) -> None:
        self.stages.append(StageRecord(stage, "failed", detail, data))
        self.flags.append(flag)

    def flag(self, name: str) -> None:
        if name not in self.flags:
            self.flags.append(name)

    # -- reading -----------------------------------------------------------------------------------
    def status_of(self, stage: str) -> Optional[str]:
        for s in reversed(self.stages):
            if s.stage == stage:
                return s.status
        return None

    @property
    def grade(self) -> str:
        if any(s.status == "failed" for s in self.stages):
            return "degraded"
        # a correction that science use needs was not applied, or a check did not pass
        if any(s.status == "skipped" for s in self.stages) or self.flags:
            return "quicklook"
        return "science"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grade": self.grade,
            "flags": list(self.flags),
            "on_error": self.on_error,
            "stages": [{"stage": s.stage, "status": s.status, "detail": s.detail, **({"data": s.data} if s.data else {})}
                       for s in self.stages],
        }