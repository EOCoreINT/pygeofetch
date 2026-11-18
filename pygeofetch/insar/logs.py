"""How much the InSAR modules say while they run.

    from pygeofetch.insar import set_verbosity
    set_verbosity("normal")        # or InterferogramGenerator(..., verbosity="normal")

Levels:
  quiet   warnings and errors only (a notebook that prints its own one-line summary per pair needs nothing else)
  normal  the lines that carry numbers: coregistration, ESD, crop, multilook, filter, save; not the per-pair
          repeats of "Parsed real SLC geometry" / "Parsed orbit state vectors" / "Plot saved"
  full    everything the modules log at INFO
  debug   everything, including per-step detail

This sets logger levels only; it does not add handlers. Warnings are never hidden by quiet/normal/full.
"""

from __future__ import annotations

import logging
from typing import Optional

_ROOTS = ("pygeofetch.insar", "pygeofetch.viz")
# The loggers that repeat for every pair without carrying a result.
_REPEATING = (
    "pygeofetch.insar.annotation",
    "pygeofetch.insar.geolocation",
    "pygeofetch.insar.visualize",
    "pygeofetch.viz",
)
LEVELS = ("quiet", "normal", "full", "debug")


def _children(prefix: str):
    for name in list(logging.root.manager.loggerDict):
        if name.startswith(prefix + "."):
            yield name


def set_verbosity(level: str = "normal") -> str:
    """Apply a verbosity level to the pygeofetch.insar and pygeofetch.viz loggers. Returns the level applied."""
    if level not in LEVELS:
        raise ValueError(f"verbosity must be one of {LEVELS}, got {level!r}")
    base = {"quiet": logging.WARNING, "normal": logging.INFO, "full": logging.INFO, "debug": logging.DEBUG}[level]
    for root in _ROOTS:
        logging.getLogger(root).setLevel(base)
        for name in _children(root):
            logging.getLogger(name).setLevel(logging.NOTSET)       # inherit: undo any earlier setting
    if level == "normal":
        for name in _REPEATING:
            logging.getLogger(name).setLevel(logging.WARNING)
    # the repeating loggers exist before their modules are imported; make sure the level sticks for late imports
    return level


def current_verbosity() -> Optional[str]:
    lv = logging.getLogger("pygeofetch.insar").level
    if lv == logging.WARNING:
        return "quiet"
    if lv == logging.DEBUG:
        return "debug"
    if lv == logging.INFO:
        return "normal" if logging.getLogger("pygeofetch.insar.annotation").level == logging.WARNING else "full"
    return None
