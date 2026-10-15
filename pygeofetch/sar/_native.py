"""Native SAR backend — wraps pygeofetch.processing.sar (always available)."""

from __future__ import annotations

from pathlib import Path


class NativeSARBackend:
    """Thin wrapper around the built-in SARProcessor in processing.sar."""

    def __init__(self):
        from pygeofetch.processing.sar import SARProcessor

        self._proc = SARProcessor()

    def despeckle(self, path: Path, filter="lee", window=5, output=None, **kw):
        return self._proc.despeckle(
            str(path), filter=filter, window=window, output=output, **kw
        )

    def calibrate(
        self, path: Path, output_type="sigma0", in_db=True, output=None, **kw
    ):
        return self._proc.calibrate(
            str(path), output_type=output_type, in_db=in_db, output=output, **kw
        )

    def terrain_correct(self, path: Path, dem="srtm", output=None, **kw):
        raise NotImplementedError(
            "Terrain correction requires the OST backend: "
            "SARProcessor(backend='ost')  (requires SNAP)"
        )

    def flood_map(self, path: Path, threshold=-15.0, reference=None, output=None, **kw):
        return self._proc.flood_map(
            str(path),
            threshold=threshold,
            reference=str(reference) if reference else None,
            output=output,
            **kw,
        )

    def coherence(self, image1: Path, image2: Path, window=7, output=None, **kw):
        return self._proc.coherence(
            str(image1), str(image2), window=window, output=output, **kw
        )
