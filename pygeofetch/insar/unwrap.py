"""
PhaseUnwrapper — production-grade phase unwrapping via SNAPHU.

Uses snaphu-py (https://github.com/isce-framework/snaphu-py), the official
Python bindings for SNAPHU (Statistical-cost, Network-flow Algorithm for
Phase Unwrapping), maintained by the same JPL/Caltech team behind ISCE2/3.

SNAPHU is the algorithm used in production by:
  - ASF HyP3's On Demand InSAR products (via GAMMA's MCF variant)
  - ESA SNAP (bundled as an external unwrapping step)
  - ISCE2/ISCE3 (native binding, same as snaphu-py)
  - GMTSAR

Reference:
  Chen, C.W. & Zebker, H.A. (2001). Two-dimensional phase unwrapping with
  use of statistical models for cost functions in a network programming
  framework. Journal of the Optical Society of America A, 18(2), 338-351.

Install: pip install "pygeofetch[insar]"   (installs snaphu, scipy)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Union

logger = logging.getLogger("pygeofetch.insar.unwrap")


def _require_snaphu():
    try:
        import snaphu

        return snaphu
    except ImportError:
        raise ImportError(
            "snaphu-py is not installed.\n"
            'Install with: pip install "pygeofetch[insar]"\n'
            "Or directly:  pip install snaphu\n\n"
            "snaphu-py provides Python bindings for SNAPHU, the same "
            "phase-unwrapping algorithm (Chen & Zebker 2001) used by "
            "ASF, ISCE2/3, GAMMA, and SNAP."
        )


def multilook(
    data: Any,
    looks_azimuth: int = 4,
    looks_range: int = 1,
    wrapped_phase: Optional[bool] = None,
) -> Any:
    """
    Average an array down by the given factor in each dimension before
    unwrapping — reduces both pixel count and phase noise, the standard
    technique for improving unwrapping reliability on low-coherence data
    (Eineder 1999; cited directly in ESA's InSAR processing tutorial,
    TM-19: "In cases of low coherence (say 0.1), the number of looks to
    be averaged should increase up to 100").

    Args:
        data:            2D array to multilook. Complex (interferogram),
                        or real-valued (coherence, wrapped phase,
                        unwrapped phase, or any other raster).
        looks_azimuth:   Averaging factor along axis 0.
        looks_range:     Averaging factor along axis 1.
        wrapped_phase:   Required for real-valued input, not optional or
                        defaulted — dtype alone cannot tell wrapped phase
                        apart from coherence or unwrapped phase, and
                        guessing wrong is not a small error:

                        - wrapped_phase=True: circular (complex-exponential)
                          averaging. Correct for phase bounded to
                          [-pi, pi) — plain arithmetic averaging would be
                          biased wherever a look-window straddles the
                          wrap boundary (e.g. averaging -3.1 and +3.1 rad
                          arithmetically gives ~0, when the true circular
                          average is near +-pi).
                        - wrapped_phase=False: plain arithmetic averaging.
                          Correct for coherence (bounded [0,1], not an
                          angle at all) and for unwrapped phase (an
                          unbounded real value). Verified directly:
                          circular-averaging real unwrapped phase of
                          ~45 rad silently collapses it to ~1.3 rad,
                          destroying the very thing unwrapping produced.

                        Complex input (interferograms) ignores this
                        parameter entirely and always averages directly
                        as complex numbers — unambiguous, no real/wrapped
                        distinction applies.

    Returns:
        The multilooked array, same dtype family as the input.

    Raises:
        ValueError: if `data` is real-valued and `wrapped_phase` was not
            explicitly specified — a real, deliberate refusal to guess,
            not an oversight. Silent dtype-based guessing was the
            original design of this function before this exact failure
            mode was found and fixed.

    Example::

        # A real interferogram (complex) -- unambiguous
        igram_ml = multilook(interferogram, looks_azimuth=4, looks_range=1)

        # Wrapped phase -- must say so explicitly
        phase_ml = multilook(wrapped_phase_rad, 4, 1, wrapped_phase=True)

        # Coherence -- must say so explicitly (plain averaging)
        coherence_ml = multilook(coherence, 4, 1, wrapped_phase=False)
    """
    import numpy as np

    h, w = data.shape
    h_ml = (h // looks_azimuth) * looks_azimuth
    w_ml = (w // looks_range) * looks_range
    trimmed = data[:h_ml, :w_ml]
    reshaped_shape = (h_ml // looks_azimuth, looks_azimuth, w_ml // looks_range, looks_range)

    if np.iscomplexobj(trimmed):
        return trimmed.reshape(reshaped_shape).mean(axis=(1, 3))

    if wrapped_phase is None:
        raise ValueError(
            "multilook() received real-valued input but wrapped_phase was "
            "not specified. This is required, not optional: dtype alone "
            "cannot distinguish wrapped phase (needs circular averaging) "
            "from coherence or unwrapped phase (needs plain arithmetic "
            "averaging), and guessing wrong silently corrupts the data "
            "rather than raising an error. Pass wrapped_phase=True for "
            "wrapped phase in radians, or wrapped_phase=False for "
            "coherence, unwrapped phase, or any other real-valued raster."
        )

    if wrapped_phase:
        reshaped = np.exp(1j * trimmed).reshape(reshaped_shape)
        return np.angle(reshaped.mean(axis=(1, 3)))
    else:
        return trimmed.reshape(reshaped_shape).mean(axis=(1, 3))


class PhaseUnwrapper:
    """
    Phase unwrapping via SNAPHU (Statistical-cost, Network-flow Algorithm).

    Args:
        cost_mode: SNAPHU cost function — ``"topo"`` (default, for terrain),
                   ``"defo"`` (for deformation — less smoothing bias),
                   ``"smooth"`` (generic smoothness prior),
                   ``"nostatcosts"`` (uniform cost, fastest, least accurate).
        init_method: ``"mcf"`` (default — Minimum Cost Flow, matches ASF/GAMMA)
                     or ``"mst"`` (Minimum Spanning Tree — faster, less optimal).

    Example::

        from pygeofetch.insar import InterferogramGenerator, PhaseUnwrapper

        gen    = InterferogramGenerator()
        result = gen.process_pair("ref.tif", "sec.tif", dem="dem.tif")

        unwrapper = PhaseUnwrapper(cost_mode="defo")
        unwrapped, conncomp = unwrapper.unwrap(
            result.interferogram, result.coherence
        )

    Choosing cost_mode:
        - Use ``"topo"`` for DEM generation / topographic mapping tasks
          where phase gradients follow terrain.
        - Use ``"defo"`` for deformation monitoring (subsidence, volcanic,
          earthquake) where phase gradients follow displacement, not
          terrain — this is the standard choice for MintPy/SBAS workflows.
    """

    def __init__(self, cost_mode: str = "defo", init_method: str = "mcf") -> None:
        valid_costs = ("topo", "defo", "smooth", "nostatcosts")
        if cost_mode not in valid_costs:
            raise ValueError(
                f"cost_mode must be one of {valid_costs}, got {cost_mode!r}"
            )
        valid_inits = ("mcf", "mst")
        if init_method not in valid_inits:
            raise ValueError(
                f"init_method must be one of {valid_inits}, got {init_method!r}"
            )
        self._cost_mode = cost_mode
        self._init_method = init_method

    def unwrap(
        self,
        interferogram: Any,
        coherence: Any,
        nlooks: float = 1.0,
        mask: Optional[Any] = None,
        min_conncomp_frac: float = 0.01,
        min_region_size: int = 100,
    ) -> Tuple[Any, Any]:
        """
        Unwrap a wrapped interferogram phase.

        Args:
            interferogram: Complex64 array (wrapped interferogram) OR
                           float32 array of wrapped phase in radians.
            coherence:     Float32 coherence array, 0-1, same shape.
            nlooks:        Effective number of looks (affects statistical
                           cost weighting). Use the multilook factor applied
                           during interferogram formation.
            mask:          Optional boolean array — True = valid, False = masked out
                           (e.g. water bodies, layover/shadow regions).
            min_conncomp_frac: Minimum size of a connected component, as a
                           fraction of total pixels, for SNAPHU to mark it
                           reliable (conncomp != 0). snaphu-py's own
                           default (0.01 = 1% of the whole scene) can
                           discard genuinely valid, internally-consistent
                           regions when the real coherent area is small
                           relative to the full scene — confirmed
                           directly: a real low-coherence scene with a
                           small but real coherent patch reported 100%
                           unreliable at the default, purely because that
                           real patch fell under the 1%-of-scene pixel
                           threshold, not because nothing was actually
                           unwrappable. Lower this (e.g. 0.001-0.005) for
                           scenes with small AOIs or patchy real coherence.
            min_region_size: Minimum absolute pixel count for the same
                           reliability decision — snaphu-py's own default
                           is 100. Works alongside min_conncomp_frac (the
                           less restrictive of the two typically governs
                           on a small scene).

        Returns:
            (unwrapped_phase, conncomp) — both same shape as input.
            unwrapped_phase: float32 radians, continuous (not wrapped to [-pi, pi))
            conncomp:        int32 connected-component labels (0 = unreliable/masked)

        Example::

            unwrapped, conncomp = unwrapper.unwrap(igram, coherence, nlooks=4.0)
            # conncomp == 0 marks pixels SNAPHU could not confidently unwrap
            reliable = conncomp > 0
        """
        np = self._np()
        sx = _require_snaphu()

        if np.iscomplexobj(interferogram):
            igram = interferogram.astype(np.complex64)
        else:
            # Treat as wrapped phase in radians — convert to unit-magnitude complex
            igram = np.exp(1j * interferogram).astype(np.complex64)

        corr = np.clip(coherence, 0.0, 1.0).astype(np.float32)

        if mask is not None:
            corr = np.where(mask, corr, 0.0).astype(np.float32)

        logger.info(
            "Unwrapping %s pixels (cost=%s, init=%s, nlooks=%.1f)",
            f"{igram.shape[0]}x{igram.shape[1]}",
            self._cost_mode,
            self._init_method,
            nlooks,
        )

        try:
            unwrapped, conncomp = sx.unwrap(
                igram,
                corr,
                nlooks=nlooks,
                cost=self._cost_mode,
                init=self._init_method,
                min_conncomp_frac=min_conncomp_frac,
                min_region_size=min_region_size,
            )
        except Exception as exc:
            raise RuntimeError(
                f"SNAPHU unwrapping failed: {exc}\n"
                "Common causes: incompatible array shapes, all-zero coherence, "
                "or insufficient memory for large scenes. Consider multilooking "
                "the interferogram first to reduce pixel count."
            ) from exc

        n_unreliable = int(np.sum(conncomp == 0))
        pct = 100 * n_unreliable / conncomp.size
        if pct > 30:
            logger.warning(
                "%.1f%% of pixels are in the unreliable connected component "
                "(conncomp==0). Consider improving coherence via multilooking "
                "or filtering, or check for large decorrelated areas.",
                pct,
            )
        else:
            logger.info("Unwrapping complete — %.1f%% unreliable pixels", pct)

        return unwrapped.astype(np.float32), conncomp

    def unwrap_files(
        self,
        interferogram_path: Union[str, Path],
        coherence_path: Union[str, Path],
        output_path: Union[str, Path],
        nlooks: float = 1.0,
        mask_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Unwrap directly from/to GeoTIFF files, preserving georeferencing.

        Args:
            interferogram_path: Wrapped phase or complex interferogram GeoTIFF.
            coherence_path:     Coherence GeoTIFF (0-1).
            output_path:        Output path for the unwrapped phase GeoTIFF.
            nlooks:              Effective number of looks.
            mask_path:           Optional binary mask GeoTIFF (1=valid, 0=masked).

        Returns:
            Path to the unwrapped phase GeoTIFF.

        Example::

            unwrapper.unwrap_files(
                "wrapped_phase.tif", "coherence.tif",
                output_path="unwrapped.tif", nlooks=4.0,
            )
        """
        np = self._np()
        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        with rasterio.open(interferogram_path) as src:
            profile = src.profile.copy()
            phase = src.read(1).astype(np.float32)

        with rasterio.open(coherence_path) as src:
            coherence = src.read(1).astype(np.float32)

        mask = None
        if mask_path:
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(bool)

        unwrapped, conncomp = self.unwrap(phase, coherence, nlooks=nlooks, mask=mask)

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out_profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "height": unwrapped.shape[0],
            "width": unwrapped.shape[1],
            "crs": profile.get("crs"),
            "transform": profile.get("transform"),
            "nodata": -9999.0,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(unwrapped[np.newaxis])
            dst.update_tags(1, description="unwrapped_phase_radians")

        conncomp_path = out_path.parent / f"{out_path.stem}_conncomp.tif"
        cc_profile = dict(out_profile, dtype="int32", nodata=0)
        with rasterio.open(conncomp_path, "w", **cc_profile) as dst:
            dst.write(conncomp.astype(np.int32)[np.newaxis])

        logger.info(
            "Unwrapped phase → %s (conncomp → %s)", out_path.name, conncomp_path.name
        )
        return out_path

    def _np(self):
        import numpy as np

        return np