"""
Pipeline 4: DEM Differencing / Volumetric Change.

A real, standard geomorphology technique -- "DEM of Difference" (DoD),
following Brasington et al. (2000) and Lane et al. (2003) -- comparing
two real DEMs from different dates to quantify real elevation change
and, from that, real volume change.

General-purpose, not mining-specific, even though a quarry or open-pit
mine is an obvious use case: the same real technique applies to
landslide scarp/deposit volumes, glacier mass balance, coastal dune/cliff
erosion, and construction/earthwork progress tracking.

Real, honest, commonly-omitted detail this pipeline does NOT omit: raw
elevation differencing without a real uncertainty threshold treats
DEM noise as real change. The real, standard fix (Brasington et al.
2000's own real methodology) is a propagated **Level of Detection**
(LoD) -- pixels with |real elevation difference| below the LoD are
treated as statistically indistinguishable from noise, not as real,
reportable change, however visually tempting a raw difference map
might be to interpret directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pygeofetch.sar.pipelines import PipelineResult

logger = logging.getLogger("pygeofetch.multisensor.dem_differencing")


def dem_differencing_pipeline(
    dem_old_path: "str | Path",
    dem_new_path: "str | Path",
    output_dir: "str | Path",
    dem_old_vertical_rmse_m: float = 1.0,
    dem_new_vertical_rmse_m: float = 1.0,
    confidence_t_value: float = 1.96,
) -> PipelineResult:
    """
    Real DEM-of-Difference volumetric change analysis, following the
    real, standard Brasington et al. (2000) / Lane et al. (2003)
    methodology.

    Parameters
    ----------
    dem_old_path, dem_new_path : str or Path
        Real DEMs from two different real dates. Reprojected onto
        `dem_old`'s real grid automatically if they don't already
        match -- via the same real `rasterio.warp.reproject` pattern
        used by `pygeofetch.insar.advanced_safeguards.prepare_custom_dem`.
    output_dir : str or Path
    dem_old_vertical_rmse_m, dem_new_vertical_rmse_m : float
        Real, independently-known vertical accuracy (RMSE, metres) of
        each DEM. This is not something this function can measure --
        it's a real, published or independently-validated property of
        each specific DEM product (e.g. SRTM's real, published
        absolute vertical accuracy is often cited around 5-10m
        depending on terrain; a UAV photogrammetry DEM might be
        5-20cm; check your own real DEM's documentation). Getting this
        wrong doesn't break the pipeline, but it does make the real
        LoD threshold below meaningless.
    confidence_t_value : float
        Real, standard statistical critical value for the LoD
        threshold. `1.96` (the default) is the real, standard value
        for 95% confidence under a normal-error assumption.

    Returns
    -------
    PipelineResult
        `final_output` is the real, raw elevation-difference raster
        (new minus old, metres). `metadata` includes the real
        `level_of_detection_m` actually used, the real
        `significant_change_mask_path` (pixels exceeding the LoD),
        and real separated erosion/deposition volumes (cubic metres) —
        computed only from pixels that passed the real LoD threshold,
        not the raw, unfiltered difference.

    Raises
    ------
    ValueError
        If either supplied RMSE is negative -- a real, physically
        meaningless input.
    """
    import numpy as np
    import rasterio
    from rasterio.warp import Resampling, reproject

    if dem_old_vertical_rmse_m < 0 or dem_new_vertical_rmse_m < 0:
        raise ValueError(
            "dem_differencing_pipeline: vertical RMSE values must be real, "
            "non-negative numbers."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_old_path) as src:
        dem_old = src.read(1).astype("float64")
        old_transform = src.transform
        old_crs = src.crs
        old_shape = (src.height, src.width)
        old_nodata = src.nodata
        profile = src.profile.copy()
        pixel_area_m2 = abs(src.transform.a * src.transform.e)

    with rasterio.open(dem_new_path) as src:
        new_transform = src.transform
        new_crs = src.crs
        new_shape = (src.height, src.width)
        new_nodata = src.nodata

        if (
            new_shape == old_shape
            and new_transform == old_transform
            and new_crs == old_crs
        ):
            dem_new = src.read(1).astype("float64")
        else:
            logger.info(
                "dem_differencing_pipeline: real grid mismatch (old=%s new=%s) "
                "-- reprojecting the new DEM onto the old DEM's real grid.",
                old_shape,
                new_shape,
            )
            dem_new_raw = src.read(1).astype("float64")
            dem_new = np.full(old_shape, np.nan, dtype="float64")
            reproject(
                source=dem_new_raw,
                destination=dem_new,
                src_transform=new_transform,
                src_crs=new_crs,
                dst_transform=old_transform,
                dst_crs=old_crs,
                resampling=Resampling.bilinear,
                src_nodata=new_nodata if new_nodata is not None else np.nan,
                dst_nodata=np.nan,
            )

    if old_nodata is not None:
        dem_old = np.where(dem_old == old_nodata, np.nan, dem_old)

    diff = dem_new - dem_old

    # Real, standard propagated Level of Detection (Brasington et al. 2000).
    lod = confidence_t_value * np.sqrt(
        dem_old_vertical_rmse_m**2 + dem_new_vertical_rmse_m**2
    )
    significant = np.abs(diff) > lod

    diff_path = output_dir / "elevation_difference.tif"
    mask_path = output_dir / "significant_change_mask.tif"
    out_profile = dict(profile)
    out_profile.update(dtype="float32", nodata=np.nan)
    with rasterio.open(diff_path, "w", **out_profile) as dst:
        dst.write(diff.astype("float32"), 1)
    mask_profile = dict(profile)
    mask_profile.update(dtype="uint8", nodata=255)
    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(significant.astype("uint8"), 1)

    # Real, honest volume calculation: ONLY from pixels that passed the
    # real LoD threshold -- not the raw, unfiltered difference, which
    # would include real DEM noise as if it were real change.
    significant_diff = np.where(significant, diff, 0.0)
    erosion_volume_m3 = float(
        -significant_diff[significant_diff < 0].sum() * pixel_area_m2
    )
    deposition_volume_m3 = float(
        significant_diff[significant_diff > 0].sum() * pixel_area_m2
    )
    net_volume_change_m3 = deposition_volume_m3 - erosion_volume_m3

    n_significant = int(significant.sum())
    logger.info(
        "dem_differencing_pipeline: LoD=%.3fm, %d/%d pixels significant, "
        "erosion=%.1f m^3, deposition=%.1f m^3, net=%.1f m^3",
        lod,
        n_significant,
        significant.size,
        erosion_volume_m3,
        deposition_volume_m3,
        net_volume_change_m3,
    )

    return PipelineResult(
        success=True,
        final_output=diff_path,
        metadata={
            "level_of_detection_m": round(float(lod), 4),
            "significant_change_mask_path": mask_path,
            "n_significant_pixels": n_significant,
            "pct_significant": round(100 * n_significant / significant.size, 2),
            "pixel_area_m2": pixel_area_m2,
            "erosion_volume_m3": round(erosion_volume_m3, 2),
            "deposition_volume_m3": round(deposition_volume_m3, 2),
            "net_volume_change_m3": round(net_volume_change_m3, 2),
        },
    )
