# pygeofetch/insar/provenance.py
"""
Provenance manifest writer — makes every InSAR run reproducible and
defensible by recording, alongside the outputs, exactly which scenes
were used, which corrections were applied at which versions, and every
processing parameter.
"""

from __future__ import annotations

import datetime
import platform
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def write_provenance_manifest(
    output_dir: Path,
    preflight_manifest: Dict[str, Any],
    processing: Dict[str, Any],
    corrections: Dict[str, Any],
    quality: Optional[Dict[str, Any]] = None,
    filename: str = "provenance.yaml",
) -> Path:
    """
    Write the full provenance record for one InSAR run.

    Args:
        output_dir:         Where the interferogram/velocity outputs live.
        preflight_manifest: The `report.manifest` from PreflightGate.
        processing:         All processing parameters used.
        corrections:        Correction methods + software/data versions.
        quality:            Optional post-hoc quality metrics.

    Returns:
        Path to the written provenance.yaml.
    """
    import pygeofetch
    record = {
        "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "environment": {
            "pygeofetch_version": getattr(pygeofetch, "__version__", "unknown"),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "preflight": preflight_manifest,
        "processing": processing,
        "corrections": corrections,
    }
    if quality:
        record["quality"] = quality

    out_path = Path(output_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(record, f, sort_keys=False, default_flow_style=False)
    return out_path
