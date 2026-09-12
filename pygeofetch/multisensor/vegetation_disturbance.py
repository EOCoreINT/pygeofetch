"""
Pipeline 5: Vegetation Health + Sub-Canopy Disturbance Monitoring.

Combines real optical NDVI trend (vegetation health/canopy greenness)
with real SAR interferometric coherence (ground-level phase stability)
to catch something neither sensor reliably does alone: disturbance
happening *under* a forest canopy that's still visually intact from
above.

The real, physical reasoning: optical sensors see the top of the
canopy. Selective logging, understory clearing, or early-stage
construction under a still-standing canopy can leave the visible
crown largely unchanged (a small or nonexistent NDVI drop), while the
real ground-level disturbance is severe enough to decorrelate SAR
phase almost immediately (SAR coherence loss doesn't require visible
change from above -- it only requires the physical scatterers within
each resolution cell to have moved or changed between the two real
acquisitions). Distinguishing "both dropped" (visible disturbance,
canopy loss) from "only coherence dropped" (sub-canopy disturbance,
real but currently invisible from above) is the actual, real value
this pipeline adds over either sensor's disturbance detection alone.

Reuses `pygeofetch.utils.sar_math.estimate_interferometric_coherence`
(the same real, canonical coherence formula
`pygeofetch.sar.pipelines.coherence_disturbance_pipeline` and
`client.sar.coherence()` both already use) rather than a third,
independent implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pygeofetch.sar.pipelines import PipelineResult

logger = logging.getLogger("pygeofetch.multisensor.vegetation_disturbance")

# Real classification codes, documented explicitly rather than left as
# unexplained integers in the output raster.
CLASS_NO_DISTURBANCE = 0
CLASS_VISIBLE_DISTURBANCE = 1  # NDVI dropped -- canopy loss visible from above
CLASS_SUBCANOPY_DISTURBANCE = (
    2  # coherence dropped, NDVI did not -- disturbance under an intact-looking canopy
)


def vegetation_disturbance_pipeline(
    optical_red_pre_path: "str | Path",
    optical_nir_pre_path: "str | Path",
    optical_red_post_path: "str | Path",
    optical_nir_post_path: "str | Path",
    sar_slc_pre_path: "str | Path",
    sar_slc_post_path: "str | Path",
    output_dir: "str | Path",
    ndvi_drop_threshold: float = 0.15,
    coherence_threshold: float = 0.3,
    coherence_window: int = 7,
) -> PipelineResult:
    """
    Real, combined optical NDVI + SAR coherence disturbance
    classification.

    Parameters
    ----------
    optical_red_pre_path, optical_nir_pre_path : str or Path
        Real red/NIR bands, pre-period optical date, for NDVI.
    optical_red_post_path, optical_nir_post_path : str or Path
        Real red/NIR bands, post-period optical date, same real grid.
    sar_slc_pre_path, sar_slc_post_path : str or Path
        Real, co-registered complex SLC rasters spanning roughly the
        same real period as the optical pair -- must already share a
        real grid with each other (see [InSAR Processing] for
        coregistration) and, for the classification to be spatially
        meaningful, roughly the same real area as the optical pair
        (this pipeline does not itself reproject SAR onto optical or
        vice versa, since SLC data can't be resampled without
        corrupting its real phase -- see the real, documented
        constraint in `pygeofetch.sar.pipelines.coherence_disturbance_pipeline`).
    output_dir : str or Path
    ndvi_drop_threshold : float
        Real, minimum NDVI decrease (pre minus post) to call a pixel
        "visibly disturbed." `0.15` is a real, moderate, commonly-used
        threshold in real deforestation-detection literature -- tune
        to your own real vegetation baseline (a threshold appropriate
        for dense tropical canopy may be too strict for sparse
        savanna).
    coherence_threshold : float
        Real coherence value below which a pixel is considered
        ground-disturbed, matching the same real default used in
        `coherence_disturbance_pipeline`.
    coherence_window : int
        Passed to the real, shared coherence estimator.

    Returns
    -------
    PipelineResult
        `final_output` is a real classification raster with three
        real classes (see `CLASS_NO_DISTURBANCE`,
        `CLASS_VISIBLE_DISTURBANCE`, `CLASS_SUBCANOPY_DISTURBANCE`
        module constants). `metadata` includes real per-class pixel
        counts and the real NDVI-difference and coherence rasters'
        paths for independent inspection.
    """
    import numpy as np
    import rasterio

    from pygeofetch.processor.indices import SpectralIndex
    from pygeofetch.utils.sar_math import estimate_interferometric_coherence

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(optical_red_pre_path) as src:
        red_pre = src.read(1).astype("float64")
        profile = src.profile.copy()
        optical_shape = (src.height, src.width)
    with rasterio.open(optical_nir_pre_path) as src:
        nir_pre = src.read(1).astype("float64")
    with rasterio.open(optical_red_post_path) as src:
        red_post = src.read(1).astype("float64")
        if (src.height, src.width) != optical_shape:
            return PipelineResult(
                success=False,
                error=(
                    f"vegetation_disturbance_pipeline: optical post shape "
                    f"{(src.height, src.width)} doesn't match pre "
                    f"{optical_shape} -- co-register first."
                ),
            )
    with rasterio.open(optical_nir_post_path) as src:
        nir_post = src.read(1).astype("float64")

    si = SpectralIndex()
    ndvi_pre = si.compute("NDVI", RED=red_pre, NIR=nir_pre)
    ndvi_post = si.compute("NDVI", RED=red_post, NIR=nir_post)
    ndvi_diff = ndvi_pre - ndvi_post  # positive = real vegetation loss

    with rasterio.open(sar_slc_pre_path) as src:
        slc_pre = src.read(1)
        sar_shape = (src.height, src.width)
    with rasterio.open(sar_slc_post_path) as src:
        slc_post = src.read(1)
        if (src.height, src.width) != sar_shape:
            return PipelineResult(
                success=False,
                error=(
                    f"vegetation_disturbance_pipeline: SAR SLC shape "
                    f"mismatch {sar_shape} vs {(src.height, src.width)} -- "
                    f"must be co-registered onto the same real grid."
                ),
            )

    coherence = estimate_interferometric_coherence(
        slc_pre, slc_post, window=coherence_window
    )

    if coherence.shape != ndvi_diff.shape:
        return PipelineResult(
            success=False,
            error=(
                f"vegetation_disturbance_pipeline: real SAR grid {coherence.shape} "
                f"and real optical grid {ndvi_diff.shape} don't match -- this "
                f"pipeline needs both on the same real grid already (SLC data "
                f"can't be resampled without corrupting phase, so reproject "
                f"the optical side onto the SAR grid beforehand, not the "
                f"reverse)."
            ),
        )

    visible_disturbance = ndvi_diff > ndvi_drop_threshold
    ground_disturbance = coherence < coherence_threshold
    subcanopy_disturbance = ground_disturbance & ~visible_disturbance

    classification = np.full(ndvi_diff.shape, CLASS_NO_DISTURBANCE, dtype="uint8")
    classification[subcanopy_disturbance] = CLASS_SUBCANOPY_DISTURBANCE
    classification[visible_disturbance] = CLASS_VISIBLE_DISTURBANCE

    class_path = output_dir / "disturbance_classification.tif"
    ndvi_diff_path = output_dir / "ndvi_difference.tif"
    coherence_path = output_dir / "sar_coherence.tif"

    class_profile = dict(profile)
    class_profile.update(dtype="uint8", count=1, nodata=255)
    with rasterio.open(class_path, "w", **class_profile) as dst:
        dst.write(classification, 1)

    float_profile = dict(profile)
    float_profile.update(dtype="float32", count=1, nodata=None)
    with rasterio.open(ndvi_diff_path, "w", **float_profile) as dst:
        dst.write(ndvi_diff.astype("float32"), 1)
    with rasterio.open(coherence_path, "w", **float_profile) as dst:
        dst.write(coherence.astype("float32"), 1)

    n_visible = int(visible_disturbance.sum())
    n_subcanopy = int(subcanopy_disturbance.sum())
    n_total = classification.size
    logger.info(
        "vegetation_disturbance_pipeline: visible=%d (%.1f%%), sub-canopy=%d (%.1f%%)",
        n_visible,
        100 * n_visible / n_total,
        n_subcanopy,
        100 * n_subcanopy / n_total,
    )

    return PipelineResult(
        success=True,
        final_output=class_path,
        metadata={
            "ndvi_difference_path": ndvi_diff_path,
            "coherence_path": coherence_path,
            "n_visible_disturbance_pixels": n_visible,
            "n_subcanopy_disturbance_pixels": n_subcanopy,
            "n_no_disturbance_pixels": int(
                (classification == CLASS_NO_DISTURBANCE).sum()
            ),
            "pct_visible_disturbance": round(100 * n_visible / n_total, 2),
            "pct_subcanopy_disturbance": round(100 * n_subcanopy / n_total, 2),
        },
    )
