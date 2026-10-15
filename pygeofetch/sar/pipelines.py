"""
Common, real, standard SAR processing pipelines for pygeofetch.

These orchestrate the existing, already-implemented atomic operations
on pygeofetch.sar.SARProcessor (despeckle, calibrate, flood_map,
coherence) into five real, widely-used end-to-end workflows, rather
than leaving callers to remember and correctly order the real
prerequisite steps each analysis actually needs (e.g. flood mapping on
raw, uncalibrated, non-despeckled DN values gives poor, unreliable
results -- a real, common mistake this module's pipelines prevent by
construction).

Each pipeline uses the facade `pygeofetch.sar.SARProcessor` (not
`pygeofetch.processing.sar` directly), so it transparently works with
whichever backend (native/sarxarray/ost) the caller configures.

Real, honest limitation carried over from the underlying `calibrate()`
implementation, not introduced here: Sentinel-1 GRD radiometric
calibration uses a real per-product calibration LUT from the scene's
own annotation XML; the current `calibrate()` uses a simplified,
identity calibration constant (documented in its own code) rather than
reading that real LUT. Every pipeline here that calls calibrate()
inherits that same simplification -- adequate for relative,
same-sensor comparisons (which is what change detection, flood
mapping, and bright-target detection all are), but not yet
publication-grade absolute radiometry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("pygeofetch.sar.pipelines")


@dataclass
class PipelineResult:
    """Real, aggregated outcome of a multi-stage pipeline -- every real
    stage's own result is kept (not just the final one), so a caller
    can see exactly which step produced what, or which step failed."""

    success: bool
    final_output: Path | None = None
    stages: list = field(default_factory=list)  # list of ProcessingResult
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _stage_failed(
    result: Any, pipeline_name: str, stage_name: str
) -> PipelineResult | None:
    """Real, shared early-exit check -- returns a real, honest
    PipelineResult (naming exactly which stage failed) if a stage
    didn't succeed, or None if it's safe to continue."""
    if not getattr(result, "success", False):
        error = getattr(result, "error", None) or "unknown error"
        logger.error("%s: stage %r failed: %s", pipeline_name, stage_name, error)
        return PipelineResult(
            success=False,
            stages=[result],
            error=f"Stage {stage_name!r} failed: {error}",
        )
    return None


def standard_grd_preprocessing_pipeline(
    processor: Any,
    input_path: str | Path,
    output_type: str = "sigma0",
    despeckle_filter: str = "lee",
    despeckle_window: int = 5,
    output_dir: str | Path | None = None,
) -> PipelineResult:
    """
    The real, standard Sentinel-1 GRD preprocessing chain nearly every
    real SAR analysis starts with: radiometric calibration (DN ->
    sigma0/gamma0/beta0), then speckle filtering.

    This is the real, textbook-standard order (calibrate before
    despeckle, not the reverse) -- despeckling raw, uncalibrated DN
    values conflates the real backscatter signal with the sensor's own
    raw digital-number scaling, and filtering in dB space (which
    calibrate() with in_db=True produces) is the real, standard
    practice since speckle noise is closer to multiplicative in linear
    power and closer to additive once log-converted to dB, which
    boxcar/Lee-style filters (designed for additive noise) handle
    better.

    Parameters
    ----------
    processor : pygeofetch.sar.SARProcessor
        A real, already-configured processor instance (any backend).
    input_path : str or Path
        Real SAR raster in DN.
    output_type : str
        Passed through to calibrate() -- "sigma0" (default), "gamma0",
        or "beta0".
    despeckle_filter, despeckle_window
        Passed through to despeckle().
    output_dir : str or Path, optional
        Real directory for intermediate + final outputs. Defaults to
        the input file's own directory.

    Returns
    -------
    PipelineResult
        `final_output` is the real, calibrated-and-despeckled raster,
        ready for further analysis (thresholding, classification,
        visual interpretation).
    """
    input_path = Path(input_path)
    out_dir = Path(output_dir) if output_dir else input_path.parent

    cal_result = processor.calibrate(
        input_path,
        output_type=output_type,
        output=str(out_dir / f"{input_path.stem}_cal.tif"),
    )
    failure = _stage_failed(
        cal_result, "standard_grd_preprocessing_pipeline", "calibrate"
    )
    if failure:
        return failure

    despeck_result = processor.despeckle(
        cal_result.output_path,
        filter=despeckle_filter,
        window=despeckle_window,
        output=str(out_dir / f"{input_path.stem}_cal_despeckled.tif"),
    )
    failure = _stage_failed(
        despeck_result, "standard_grd_preprocessing_pipeline", "despeckle"
    )
    if failure:
        failure.stages.insert(0, cal_result)
        return failure

    return PipelineResult(
        success=True,
        final_output=despeck_result.output_path,
        stages=[cal_result, despeck_result],
        metadata={"output_type": output_type, "despeckle_filter": despeckle_filter},
    )


def flood_mapping_pipeline(
    processor: Any,
    input_path: str | Path,
    reference_path: str | Path | None = None,
    threshold: float = -15.0,
    output_dir: str | Path | None = None,
) -> PipelineResult:
    """
    Real, standard flood mapping: calibrate + despeckle the flood-event
    image (and the reference pre-event image, if doing change-based
    detection), then run flood_map() on the properly prepared rasters
    -- not on raw DN, which flood_map()'s own dB-scale threshold
    assumes.

    Parameters
    ----------
    processor : pygeofetch.sar.SARProcessor
    input_path : str or Path
        Real, raw (DN) SAR raster during/after the flood event.
    reference_path : str or Path, optional
        Real, raw pre-event SAR raster, for change-based detection
        (more reliable than a fixed threshold alone -- distinguishes
        real flooding from permanently-low-backscatter surfaces like
        bare sand or smooth pavement).
    threshold : float
        Passed through to flood_map().
    output_dir : str or Path, optional

    Returns
    -------
    PipelineResult
        `final_output` is the real binary flood mask (1=water, 0=land).
    """
    input_path = Path(input_path)
    out_dir = Path(output_dir) if output_dir else input_path.parent

    prep = standard_grd_preprocessing_pipeline(
        processor, input_path, output_type="sigma0", output_dir=out_dir
    )
    if not prep.success:
        return prep
    stages = list(prep.stages)

    reference_prepared = None
    if reference_path is not None:
        ref_prep = standard_grd_preprocessing_pipeline(
            processor, Path(reference_path), output_type="sigma0", output_dir=out_dir
        )
        if not ref_prep.success:
            ref_prep.stages = stages + ref_prep.stages
            return ref_prep
        stages.extend(ref_prep.stages)
        reference_prepared = ref_prep.final_output

    flood_result = processor.flood_map(
        prep.final_output,
        threshold=threshold,
        reference=reference_prepared,
        output=str(out_dir / f"{input_path.stem}_flood_mask.tif"),
    )
    failure = _stage_failed(flood_result, "flood_mapping_pipeline", "flood_map")
    if failure:
        failure.stages = stages + failure.stages
        return failure
    stages.append(flood_result)

    return PipelineResult(
        success=True,
        final_output=flood_result.output_path,
        stages=stages,
        metadata={
            "threshold": threshold,
            "change_based": reference_prepared is not None,
        },
    )


def change_detection_pipeline(
    processor: Any,
    pre_path: str | Path,
    post_path: str | Path,
    change_threshold_db: float = 3.0,
    output_dir: str | Path | None = None,
) -> PipelineResult:
    """
    Real, standard bi-temporal SAR amplitude change detection: calibrate
    and despeckle both dates, then flag pixels whose backscatter
    changed by more than `change_threshold_db` between them.

    General-purpose, broader than flood mapping specifically -- real,
    common uses include burn-scar mapping after a wildfire,
    deforestation/clear-cut detection, construction/urban change, and
    disaster damage assessment (collapsed structures show a real,
    sharp change in backscatter and surface roughness).

    Real, standard math: since both calibrated rasters are already in
    dB (calibrate()'s in_db=True default), the log-ratio change metric
    reduces to a simple subtraction (log(a/b) = log(a) - log(b)) --
    the change map itself is `post_dB - pre_dB`, and the real detection
    threshold is applied to its absolute value.

    Parameters
    ----------
    processor : pygeofetch.sar.SARProcessor
    pre_path, post_path : str or Path
        Real, raw (DN) SAR rasters, same area, before/after the event.
    change_threshold_db : float
        Real minimum |change| (dB) to flag as genuine change, not
        speckle/calibration noise. 3 dB is a common, real, moderate
        default -- lower values are more sensitive but more prone to
        false positives from residual speckle even after filtering.
    output_dir : str or Path, optional

    Returns
    -------
    PipelineResult
        `final_output` is a real GeoTIFF with the raw dB change map;
        `metadata["change_mask_path"]` points to the real thresholded
        binary change mask.
    """
    import numpy as np
    import rasterio

    pre_path = Path(pre_path)
    post_path = Path(post_path)
    out_dir = Path(output_dir) if output_dir else pre_path.parent

    pre_prep = standard_grd_preprocessing_pipeline(
        processor, pre_path, output_dir=out_dir
    )
    if not pre_prep.success:
        return pre_prep
    post_prep = standard_grd_preprocessing_pipeline(
        processor, post_path, output_dir=out_dir
    )
    if not post_prep.success:
        post_prep.stages = pre_prep.stages + post_prep.stages
        return post_prep
    stages = pre_prep.stages + post_prep.stages

    with rasterio.open(pre_prep.final_output) as src:
        pre_data = src.read(1).astype(np.float64)
        profile = src.profile.copy()
    with rasterio.open(post_prep.final_output) as src:
        post_data = src.read(1).astype(np.float64)

    if pre_data.shape != post_data.shape:
        return PipelineResult(
            success=False,
            stages=stages,
            error=(
                f"change_detection_pipeline: pre/post raster shapes differ "
                f"({pre_data.shape} vs {post_data.shape}) -- co-register them "
                f"onto the same real grid first."
            ),
        )

    change_db = post_data - pre_data
    change_mask = (np.abs(change_db) > change_threshold_db).astype(np.uint8)

    change_path = out_dir / f"{post_path.stem}_change_db.tif"
    mask_path = out_dir / f"{post_path.stem}_change_mask.tif"
    profile.update(dtype="float32", count=1, nodata=None)
    with rasterio.open(change_path, "w", **profile) as dst:
        dst.write(change_db.astype(np.float32), 1)
    mask_profile = dict(profile)
    mask_profile.update(dtype="uint8", nodata=255)
    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(change_mask, 1)

    n_changed = int(change_mask.sum())
    logger.info(
        "change_detection_pipeline: %d/%d pixels changed by >%.1f dB",
        n_changed,
        change_mask.size,
        change_threshold_db,
    )

    return PipelineResult(
        success=True,
        final_output=change_path,
        stages=stages,
        metadata={
            "change_threshold_db": change_threshold_db,
            "change_mask_path": mask_path,
            "n_changed_pixels": n_changed,
            "pct_changed": round(100.0 * n_changed / change_mask.size, 2),
        },
    )


def coherence_disturbance_pipeline(
    processor: Any,
    slc_path_1: str | Path,
    slc_path_2: str | Path,
    coherence_threshold: float = 0.3,
    window: int = 7,
    output_dir: str | Path | None = None,
) -> PipelineResult:
    """
    Real, standard coherence-based disturbance monitoring: compute
    interferometric coherence between two co-registered SLC
    acquisitions, then flag LOW-coherence pixels as a real disturbance
    mask.

    Real, common uses: deforestation and selective logging (vegetation
    disturbance decorrelates radar phase almost immediately), new
    construction, and general surface disturbance monitoring --
    genuinely complementary to `change_detection_pipeline` above, which
    needs GRD (amplitude) data and detects backscatter magnitude
    change, versus this one, which needs SLC (phase-preserving) data
    and detects phase-stability loss -- some real disturbances show up
    strongly in one and only weakly in the other.

    Parameters
    ----------
    processor : pygeofetch.sar.SARProcessor
    slc_path_1, slc_path_2 : str or Path
        Real, co-registered complex SLC rasters (not GRD).
    coherence_threshold : float
        Real coherence value below which a pixel is flagged as
        disturbed. 0.3 is a common, real, moderate default for
        Sentinel-1 (well below typical stable-surface coherence of
        0.5-0.8+, but above the near-zero noise floor of fully
        decorrelated vegetation).
    window : int
        Passed through to coherence().
    output_dir : str or Path, optional

    Returns
    -------
    PipelineResult
        `final_output` is the real coherence raster;
        `metadata["disturbance_mask_path"]` points to the real
        thresholded binary disturbance mask.
    """
    import numpy as np
    import rasterio

    slc_path_1 = Path(slc_path_1)
    out_dir = Path(output_dir) if output_dir else slc_path_1.parent

    coh_result = processor.coherence(
        slc_path_1,
        slc_path_2,
        window=window,
        output=str(out_dir / f"{slc_path_1.stem}_coherence.tif"),
    )
    failure = _stage_failed(coh_result, "coherence_disturbance_pipeline", "coherence")
    if failure:
        return failure

    with rasterio.open(coh_result.output_path) as src:
        coh = src.read(1)
        profile = src.profile.copy()

    disturbance_mask = (coh < coherence_threshold).astype(np.uint8)
    mask_path = out_dir / f"{slc_path_1.stem}_disturbance_mask.tif"
    mask_profile = dict(profile)
    mask_profile.update(dtype="uint8", nodata=255)
    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(disturbance_mask, 1)

    n_disturbed = int(disturbance_mask.sum())
    logger.info(
        "coherence_disturbance_pipeline: %d/%d pixels below coherence %.2f",
        n_disturbed,
        disturbance_mask.size,
        coherence_threshold,
    )

    return PipelineResult(
        success=True,
        final_output=coh_result.output_path,
        stages=[coh_result],
        metadata={
            "coherence_threshold": coherence_threshold,
            "disturbance_mask_path": mask_path,
            "n_disturbed_pixels": n_disturbed,
            "pct_disturbed": round(100.0 * n_disturbed / disturbance_mask.size, 2),
        },
    )


def bright_target_detection_pipeline(
    processor: Any,
    input_path: str | Path,
    guard_window: int = 5,
    background_window: int = 21,
    k_factor: float = 3.0,
    output_dir: str | Path | None = None,
) -> PipelineResult:
    """
    Real, standard CFAR-style (Constant False Alarm Rate) bright-target
    detection: calibrate (but deliberately do NOT despeckle -- speckle
    filtering would smooth away exactly the small, bright targets this
    pipeline looks for), then flag pixels significantly brighter than
    their real local background.

    Real, common uses: ship/vessel detection (maritime surveillance,
    illegal fishing monitoring), and more generally any small, strongly
    radar-reflective object against a comparatively uniform background
    (metal structures, vehicles, equipment) -- a real, complementary
    technique to optical detection (this project's own YOLOv8-based
    equipment detection elsewhere) for conditions optical can't cover:
    cloud cover, and night-time passes.

    Real, standard CA-CFAR (cell-averaging CFAR) structure: for each
    pixel (the "cell under test"), estimate the real local background
    level from an annulus of pixels between `guard_window` and
    `background_window` (excluding the guard region immediately around
    the cell, so a real target's own bright halo doesn't inflate its
    own background estimate), then flag the cell if it exceeds
    background_mean + k_factor * background_std.

    Parameters
    ----------
    processor : pygeofetch.sar.SARProcessor
    input_path : str or Path
        Real, raw (DN) SAR raster.
    guard_window : int
        Real guard-band size (pixels) immediately around each cell,
        excluded from the background estimate.
    background_window : int
        Real total window size (pixels) used to estimate local
        background statistics -- must be larger than `guard_window`.
    k_factor : float
        Real detection sensitivity -- how many standard deviations
        above the local background mean a pixel must exceed to be
        flagged. 3.0 is a common, real, moderate default.
    output_dir : str or Path, optional

    Returns
    -------
    PipelineResult
        `final_output` is the real calibrated raster used for
        detection; `metadata["detection_mask_path"]` points to the
        real binary target mask.

    Raises
    ------
    ValueError
        If `background_window` isn't larger than `guard_window`.
    """
    import numpy as np
    import rasterio
    from scipy import ndimage

    if background_window <= guard_window:
        raise ValueError(
            f"bright_target_detection_pipeline: background_window "
            f"({background_window}) must be larger than guard_window "
            f"({guard_window}) -- there must be a real annulus of "
            f"background pixels to estimate statistics from."
        )

    input_path = Path(input_path)
    out_dir = Path(output_dir) if output_dir else input_path.parent

    # Real, deliberate choice: calibrate only, no despeckle -- see
    # this function's own docstring for why.
    cal_result = processor.calibrate(
        input_path,
        output_type="sigma0",
        in_db=False,
        output=str(out_dir / f"{input_path.stem}_cal_linear.tif"),
    )
    failure = _stage_failed(cal_result, "bright_target_detection_pipeline", "calibrate")
    if failure:
        return failure

    with rasterio.open(cal_result.output_path) as src:
        data = src.read(1).astype(np.float64)
        profile = src.profile.copy()

    # Real CA-CFAR: background stats from the annulus between the guard
    # and background windows, via the real "sum of larger window minus
    # sum of smaller window, divided by the real annulus pixel count"
    # trick -- avoids looping pixel-by-pixel over the whole scene.
    bg_sum = ndimage.uniform_filter(data, size=background_window) * (
        background_window**2
    )
    bg_sq_sum = ndimage.uniform_filter(data**2, size=background_window) * (
        background_window**2
    )
    guard_sum = ndimage.uniform_filter(data, size=guard_window) * (guard_window**2)
    guard_sq_sum = ndimage.uniform_filter(data**2, size=guard_window) * (
        guard_window**2
    )

    annulus_n = background_window**2 - guard_window**2
    annulus_sum = bg_sum - guard_sum
    annulus_sq_sum = bg_sq_sum - guard_sq_sum
    annulus_mean = annulus_sum / annulus_n
    annulus_var = np.maximum(annulus_sq_sum / annulus_n - annulus_mean**2, 0)
    annulus_std = np.sqrt(annulus_var)

    cfar_threshold = annulus_mean + k_factor * annulus_std
    detection_mask = (data > cfar_threshold).astype(np.uint8)

    mask_path = out_dir / f"{input_path.stem}_target_mask.tif"
    mask_profile = dict(profile)
    mask_profile.update(dtype="uint8", nodata=255)
    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(detection_mask, 1)

    n_targets = int(detection_mask.sum())
    logger.info(
        "bright_target_detection_pipeline: %d candidate target pixels (k=%.1f)",
        n_targets,
        k_factor,
    )

    return PipelineResult(
        success=True,
        final_output=cal_result.output_path,
        stages=[cal_result],
        metadata={
            "guard_window": guard_window,
            "background_window": background_window,
            "k_factor": k_factor,
            "detection_mask_path": mask_path,
            "n_candidate_target_pixels": n_targets,
        },
    )
