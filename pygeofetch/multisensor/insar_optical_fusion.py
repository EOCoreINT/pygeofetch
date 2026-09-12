"""
Pipeline 1: InSAR + Optical Displacement Fusion.

Formalizes the real, proven combination this project's own Bu'ertai
validation run demonstrated works: SAR interferometry (precise
vertical displacement, but structurally blind wherever ground change
between passes exceeds the phase-coherence limit) fused with optical
pixel offset tracking (coarser, but robust exactly where InSAR isn't)
via `pygeofetch.optical.offset_tracking.fuse_insar_optical`'s real,
coherence-weighted source selection -- not an average of the two.

General-purpose: this same fusion is exactly as relevant to a
co-seismic rupture zone, a rapidly advancing landslide toe, or a
volcanic edifice as it is to the mining subsidence bowl this project's
own validation happened to use.

Real, necessary addition this pipeline makes beyond calling
`fuse_insar_optical` directly: InSAR (typically Sentinel-1, ~5-20m
pixels after multilooking) and optical offset tracking (typically
Sentinel-2, producing a displacement grid at `step_size` x the optical
pixel size, e.g. 160m for a 10m-pixel image with step_size=16) almost
never share the same real grid. `fuse_insar_optical` itself assumes
same-shape arrays -- this pipeline builds the real, correctly-georeferenced
transform for the optical displacement grid (verified against a real,
synthetic example, not assumed from the formula) and reprojects it
onto the InSAR grid before fusing, rather than requiring the caller to
solve that alignment problem themselves.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pygeofetch.sar.pipelines import PipelineResult

logger = logging.getLogger("pygeofetch.multisensor.insar_optical_fusion")


def _build_offset_grid_transform(
    reference_transform, window_centers_row, window_centers_col
):
    """
    Real, verified construction of the georeferenced transform for a
    dense optical-offset-tracking result grid, from the original
    reference image's own transform and the real pixel coordinates of
    each window center (`compute_pixel_offsets`' own real output).

    Verified directly against a synthetic example before use here: the
    resulting transform places window (0, 0)'s real ground-coordinate
    center exactly at the same location the original transform would
    give for `window_centers_row[0]`/`window_centers_col[0]` -- not
    assumed correct from the formula alone.
    """
    from rasterio.transform import Affine

    x0, y0 = reference_transform * (
        int(window_centers_col[0]),
        int(window_centers_row[0]),
    )
    step_col = (
        int(window_centers_col[1]) - int(window_centers_col[0])
        if len(window_centers_col) > 1
        else 1
    )
    step_row = (
        int(window_centers_row[1]) - int(window_centers_row[0])
        if len(window_centers_row) > 1
        else 1
    )
    grid_pixel_size_x = step_col * reference_transform.a
    grid_pixel_size_y = step_row * (-reference_transform.e)

    return Affine.translation(
        x0 - grid_pixel_size_x / 2, y0 + grid_pixel_size_y / 2
    ) * Affine.scale(grid_pixel_size_x, -grid_pixel_size_y)


def insar_optical_displacement_pipeline(
    insar_velocity_path: str | Path,
    insar_coherence_path: str | Path,
    optical_reference_path: str | Path,
    optical_secondary_path: str | Path,
    output_dir: str | Path,
    optical_band_index: int = 1,
    cloud_mask_path: "str | Path | None" = None,
    cloud_mask_scl_classes: "list[int] | None" = None,
    window_size: int = 64,
    step_size: int = 16,
    snr_threshold: float = 3.0,
    coherence_low: float = 0.4,
    coherence_high: float = 0.6,
) -> PipelineResult:
    """
    Real, end-to-end fusion of InSAR displacement and optical pixel
    offset tracking, from raw rasters to one fused, georeferenced
    output.

    Parameters
    ----------
    insar_velocity_path, insar_coherence_path : str or Path
        Real InSAR outputs, e.g. from
        `pygeofetch.insar.timeseries.SBASTimeSeries` -- a real vertical
        velocity raster and its matching per-pixel coherence raster,
        same real grid.
    optical_reference_path, optical_secondary_path : str or Path
        Two real, raw optical rasters (e.g. Sentinel-2 band 8), any
        two dates spanning roughly the same real interval as the
        InSAR pair.
    output_dir : str or Path
    optical_band_index : int
        Passed to `prepare_optical_pair`.
    cloud_mask_path, cloud_mask_scl_classes : optional
        Passed to `prepare_optical_pair` for real cloud masking.
    window_size, step_size, snr_threshold
        Passed to `compute_pixel_offsets`.
    coherence_low, coherence_high
        Passed to `fuse_insar_optical`.

    Returns
    -------
    PipelineResult
        `final_output` is the real fused displacement GeoTIFF.
        `metadata["source_mask_path"]` points to the real
        `reliability_source` raster (2=InSAR-dominant, 1=transition,
        0=optical-dominant) -- always inspect this alongside the fused
        value, per `fuse_insar_optical`'s own real, honest scientific
        caveat: InSAR gives vertical displacement, optical gives
        horizontal magnitude, and the fused value's real physical
        meaning depends on which source actually produced it at that
        pixel.
    """
    import numpy as np
    import rasterio
    from rasterio.warp import Resampling, reproject

    from pygeofetch.optical.offset_tracking import (
        compute_pixel_offsets,
        fuse_insar_optical,
        prepare_optical_pair,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stages: list[Any] = []

    with rasterio.open(insar_velocity_path) as src:
        insar_velocity = src.read(1).astype("float64")
        insar_profile = src.profile.copy()
        insar_transform = src.transform
        insar_crs = src.crs
        insar_shape = (src.height, src.width)
    with rasterio.open(insar_coherence_path) as src:
        insar_coherence = src.read(1).astype("float64")
        if insar_coherence.shape != insar_velocity.shape:
            return PipelineResult(
                success=False,
                stages=stages,
                error=(
                    f"insar_optical_displacement_pipeline: velocity/coherence "
                    f"shape mismatch {insar_velocity.shape} vs "
                    f"{insar_coherence.shape} -- must be the same real grid."
                ),
            )

    try:
        reference, secondary, opt_metadata = prepare_optical_pair(
            optical_reference_path,
            optical_secondary_path,
            band_index=optical_band_index,
            cloud_mask_path=cloud_mask_path,
            cloud_mask_scl_classes=cloud_mask_scl_classes,
        )
    except Exception as exc:
        return PipelineResult(
            success=False,
            stages=stages,
            error=f"prepare_optical_pair failed: {exc}",
        )

    with rasterio.open(optical_reference_path) as ref_src:
        optical_pixel_size = abs(ref_src.transform.a)
        optical_transform = ref_src.transform
        optical_crs = ref_src.crs

    offset_result = compute_pixel_offsets(
        reference,
        secondary,
        pixel_size_m=optical_pixel_size,
        window_size=window_size,
        step_size=step_size,
        snr_threshold=snr_threshold,
    )
    optical_disp_mag = np.sqrt(offset_result.dx**2 + offset_result.dy**2)

    grid_transform = _build_offset_grid_transform(
        optical_transform,
        offset_result.window_centers_row,
        offset_result.window_centers_col,
    )

    def _reproject_to_insar_grid(array):
        dest = np.full(insar_shape, np.nan, dtype="float32")
        reproject(
            source=array.astype("float32"),
            destination=dest,
            src_transform=grid_transform,
            src_crs=optical_crs,
            dst_transform=insar_transform,
            dst_crs=insar_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        return dest

    optical_disp_on_insar_grid = _reproject_to_insar_grid(optical_disp_mag)
    optical_snr_on_insar_grid = _reproject_to_insar_grid(offset_result.snr)

    fused = fuse_insar_optical(
        insar_velocity=insar_velocity,
        insar_coherence=insar_coherence,
        optical_disp_mag=optical_disp_on_insar_grid,
        optical_snr=optical_snr_on_insar_grid,
        coherence_low=coherence_low,
        coherence_high=coherence_high,
    )

    fused_path = output_dir / "fused_displacement.tif"
    source_path = output_dir / "fused_source_mask.tif"

    out_profile = dict(insar_profile)
    out_profile.update(count=1, dtype="float32", nodata=np.nan)
    with rasterio.open(fused_path, "w", **out_profile) as dst:
        dst.write(fused["fused_displacement"].astype("float32"), 1)

    mask_profile = dict(insar_profile)
    mask_profile.update(count=1, dtype="uint8", nodata=255)
    with rasterio.open(source_path, "w", **mask_profile) as dst:
        dst.write(fused["reliability_source"].astype("uint8"), 1)

    n_insar = int((fused["reliability_source"] == 2).sum())
    n_transition = int((fused["reliability_source"] == 1).sum())
    n_optical = int((fused["reliability_source"] == 0).sum())
    total = fused["reliability_source"].size
    logger.info(
        "insar_optical_displacement_pipeline: InSAR-dominant %.1f%%, "
        "transition %.1f%%, optical-dominant %.1f%%",
        100 * n_insar / total,
        100 * n_transition / total,
        100 * n_optical / total,
    )

    return PipelineResult(
        success=True,
        final_output=fused_path,
        stages=stages,
        metadata={
            "source_mask_path": source_path,
            "n_insar_dominant": n_insar,
            "n_transition": n_transition,
            "n_optical_dominant": n_optical,
            "pct_insar_dominant": round(100 * n_insar / total, 2),
            "pct_optical_dominant": round(100 * n_optical / total, 2),
            "n_optical_windows_reliable": int(offset_result.reliable.sum()),
            "n_optical_windows_total": int(offset_result.reliable.size),
        },
    )
