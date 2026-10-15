"""
Automatic visualization for each step of the InSAR pipeline.

Every function here reuses PyGeoFetch's existing Plotter — no separate
plotting logic duplicated for InSAR specifically. Designed to be called
automatically from InterferogramResult.save(), TimeSeriesResult.save(),
and after PhaseUnwrapper.unwrap() via auto_visualize=True, so a real
processing run produces both GIS-ready GeoTIFFs and immediately-viewable
PNGs without a separate manual plotting step a user has to remember.

Usage::

    from pygeofetch.insar.visualize import visualize_interferogram

    visualize_interferogram(result, output_dir="./results")
    # or, more commonly, via the result object directly:
    result.save("./results", auto_visualize=True)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger("pygeofetch.insar.visualize")


def visualize_interferogram(
    result: Any, output_dir: Union[str, Path]
) -> Dict[str, Path]:
    """
    Plot wrapped phase, coherence, and amplitude for an InterferogramResult.

    Args:
        result:     An InterferogramResult (or any object with
                   .interferogram, .coherence, .amplitude attributes).
        output_dir: Directory to save the PNGs into.

    Returns:
        Dict mapping product name to saved PNG path.
    """
    import numpy as np

    from pygeofetch.viz import Plotter

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pl = Plotter()
    paths = {}

    label = ""
    if getattr(result, "reference_date", None) and getattr(
        result, "secondary_date", None
    ):
        label = f" ({result.reference_date} → {result.secondary_date})"

    # Colormap: "hsv" -- a real, full-hue-cycling colormap, chosen to
    # match the conventional InSAR "rainbow fringe" appearance produced
    # by SNAP, ISCE, GAMMA, and GMTSAR (and virtually every InSAR
    # tutorial/paper figure). A prior version used matplotlib's
    # "twilight" here -- also a genuinely CYCLIC colormap (value at -pi
    # matches value at +pi, so no false discontinuity), but visually a
    # muted, low-saturation dark-blue-purple/white/orange gradient, NOT
    # a saturated hue cycle. Measured directly: twilight's saturation
    # stays under 0.4 throughout the cycle; hsv is fully saturated
    # (1.0) at every point. Both are mathematically valid cyclic
    # choices, but only one matches the field's actual visual
    # convention.
    wrapped_phase = np.angle(result.interferogram)
    paths["wrapped_phase"] = pl.plot_raster(
        wrapped_phase,
        title=f"Wrapped Interferometric Phase{label}",
        colormap="hsv",
        vmin=-np.pi,
        vmax=np.pi,
        colorbar_label="Phase (radians)",
        output=str(out_dir / "wrapped_phase.png"),
    )

    paths["coherence"] = pl.plot_raster(
        result.coherence,
        title=f"Coherence{label}",
        colormap="gray",
        vmin=0,
        vmax=1,
        colorbar_label="Coherence (0-1)",
        output=str(out_dir / "coherence.png"),
    )

    paths["amplitude"] = pl.plot_raster(
        result.amplitude,
        title=f"Amplitude (dB){label}",
        colormap="gray",
        colorbar_label="Amplitude (dB)",
        output=str(out_dir / "amplitude.png"),
    )

    logger.info("Interferogram visualizations saved → %s", out_dir)
    return paths


def visualize_unwrapped(
    unwrapped: Any,
    conncomp: Optional[Any],
    output_dir: Union[str, Path],
    coherence: Optional[Any] = None,
) -> Dict[str, Path]:
    """
    Plot unwrapped phase (and connected-component mask, if available)
    after PhaseUnwrapper.unwrap(). unwrap() returns a plain
    (unwrapped, conncomp) tuple rather than a result object with its
    own .save(), so this is called directly with its outputs::

        unwrapped, conncomp = unwrapper.unwrap(interferogram, coherence)
        visualize_unwrapped(unwrapped, conncomp, "./results", coherence=coherence)

    Args:
        unwrapped:  Unwrapped phase array (radians), from unwrap().
        conncomp:   Connected-component mask from unwrap(), or None.
        output_dir: Directory to save the PNGs into.
        coherence:  Optional coherence array, plotted alongside for
                   direct comparison against the unwrapping result.

    Returns:
        Dict mapping product name to saved PNG path.
    """
    from pygeofetch.viz import Plotter

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pl = Plotter()
    paths = {}

    paths["unwrapped_phase"] = pl.plot_raster(
        unwrapped,
        title="Unwrapped Phase",
        colormap="RdBu",
        colorbar_label="Phase (radians)",
        output=str(out_dir / "unwrapped_phase.png"),
    )

    if conncomp is not None:
        paths["connected_components"] = pl.plot_raster(
            conncomp,
            title="Connected Components (unwrapping regions)",
            colormap="tab20",
            colorbar_label="Component ID",
            output=str(out_dir / "connected_components.png"),
        )

    if coherence is not None:
        paths["coherence"] = pl.plot_raster(
            coherence,
            title="Coherence (for reference)",
            colormap="gray",
            vmin=0,
            vmax=1,
            colorbar_label="Coherence (0-1)",
            output=str(out_dir / "coherence_reference.png"),
        )

    logger.info("Unwrapping visualizations saved → %s", out_dir)
    return paths


def visualize_timeseries(
    result: Any, output_dir: Union[str, Path], max_date_panels: int = 6
) -> Dict[str, Path]:
    """
    Plot velocity map and displacement time series for a TimeSeriesResult.

    Args:
        result:          A TimeSeriesResult (from SBASTimeSeries.invert()).
        output_dir:      Directory to save the PNGs into.
        max_date_panels: Cap on how many individual per-date displacement
                        maps to render as a composite grid — real SBAS
                        results can have dozens of dates, and rendering
                        every single one is rarely useful; shows an
                        evenly-spaced sample instead if there are more.

    Returns:
        Dict mapping product name to saved PNG path.
    """
    import numpy as np

    from pygeofetch.viz import Plotter

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pl = Plotter()
    paths = {}

    vmax = float(np.nanpercentile(np.abs(result.velocity), 98))
    paths["velocity"] = pl.plot_raster(
        result.velocity,
        title=f"Mean Displacement Velocity (reference: {result.reference_date})",
        colormap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        colorbar_label="Velocity (m/year)",
        output=str(out_dir / "velocity.png"),
    )

    paths["residual_rms"] = pl.plot_raster(
        result.residual_rms,
        title="Inversion Residual RMS (fit quality)",
        colormap="inferno",
        colorbar_label="RMS residual (m)",
        output=str(out_dir / "residual_rms.png"),
    )

    n_dates = len(result.dates)
    if n_dates > max_date_panels:
        indices = np.linspace(0, n_dates - 1, max_date_panels).astype(int)
    else:
        indices = np.arange(n_dates)

    import matplotlib.pyplot as plt

    ncols = min(3, len(indices))
    nrows = int(np.ceil(len(indices) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), facecolor="white"
    )
    axes = np.atleast_1d(axes).ravel()
    disp_vmax = float(np.nanpercentile(np.abs(result.displacement), 98))
    for ax, idx in zip(axes, indices):
        im = ax.imshow(
            result.displacement[idx],
            cmap="RdBu_r",
            vmin=-disp_vmax,
            vmax=disp_vmax,
            aspect="auto",
        )
        ax.set_title(result.dates[idx], fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes[len(indices) :]:
        ax.axis("off")
    fig.suptitle("Cumulative Displacement by Date (m)", fontsize=14)
    plt.tight_layout()
    disp_path = out_dir / "displacement_by_date.png"
    plt.savefig(disp_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths["displacement_by_date"] = disp_path

    logger.info("Time series visualizations saved → %s", out_dir)
    return paths
