"""
Pipeline 2: SAR + Optical Flood/Water Mapping.

Real, well-established technique matching the same logic Copernicus's
Emergency Management Service uses for its own flood extent products:
SAR sees through the cloud cover that almost always accompanies flood
events (a real, physical property of radar, not marketing); optical
gives cleaner, more precise spectral water discrimination (NDWI)
wherever clouds do clear. Neither alone is as complete as the two
combined.

Reuses `pygeofetch.sar.pipelines.flood_mapping_pipeline` for the real
SAR side (which itself already enforces the real calibrate+despeckle
prerequisite chain) rather than reimplementing SAR flood detection --
this pipeline's own, new contribution is the real optical NDWI side
and the real, physically-motivated fusion logic between the two.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pygeofetch.sar.pipelines import PipelineResult, flood_mapping_pipeline

logger = logging.getLogger("pygeofetch.multisensor.flood_mapping")

VALID_FUSION_MODES = ("cloud_aware", "union", "intersection")


def multi_sensor_flood_pipeline(
    sar_processor: Any,
    sar_input_path: "str | Path",
    optical_green_path: "str | Path",
    optical_nir_path: "str | Path",
    output_dir: "str | Path",
    sar_reference_path: "str | Path | None" = None,
    sar_threshold: float = -15.0,
    optical_ndwi_threshold: float = 0.0,
    cloud_mask_path: "str | Path | None" = None,
    cloud_mask_scl_classes: "list[int] | None" = None,
    fusion_mode: str = "cloud_aware",
    optical_offset: float = 0.0,
    optical_scale: float = 1.0,
) -> PipelineResult:
    """
    Real, end-to-end SAR + optical flood/water extent mapping.

    Parameters
    ----------
    sar_processor : pygeofetch.sar.SARProcessor
    sar_input_path : str or Path
        Real, raw (DN) SAR GRD raster during/after the flood event.
    optical_green_path, optical_nir_path : str or Path
        Real optical green and near-infrared bands (e.g. Sentinel-2
        B03/B08), same real date range as the SAR scene, for NDWI.
        Must already share the SAR scene's real grid -- reproject
        first if not (see `pygeofetch.insar.advanced_safeguards.prepare_custom_dem`
        for the same real reproject pattern applied to a raster instead
        of a DEM).
    output_dir : str or Path
    sar_reference_path : str or Path, optional
        Passed to the real SAR `flood_mapping_pipeline` for
        change-based (rather than fixed-threshold) SAR water detection.
    sar_threshold : float
        Passed to the real SAR `flood_mapping_pipeline`.
    optical_ndwi_threshold : float
        Real NDWI threshold above which a pixel is optical-water.
        `0.0` is McFeeters (1996)'s own original, real, standard
        threshold.
    cloud_mask_path, cloud_mask_scl_classes : optional
        A real Sentinel-2 SCL raster (or similar) and the real class
        values that mean "cloud" -- pixels under cloud have no real
        optical water signal and are excluded from the optical side
        before fusion, not silently treated as "not water."
    fusion_mode : str
        - `"cloud_aware"` (default): trust optical wherever it has a
          real, cloud-free reading; fall back to SAR wherever clouded.
          The real, physically-motivated default -- optical is more
          precise where it has genuine signal, SAR is the only real
          signal where it doesn't.
        - `"union"`: water if *either* real source says water,
          wherever both have real coverage. More sensitive -- the real,
          standard choice in disaster response where missing flooded
          area is a worse error than a false positive.
        - `"intersection"`: water only where *both* real sources agree.
          Higher-confidence, fewer false positives, less complete
          coverage.

    optical_offset, optical_scale : float
        Applied as ``(DN + optical_offset) * optical_scale`` before NDWI.
        Defaults leave the DNs unchanged. NDWI is only scale-invariant, not
        offset-invariant: Sentinel-2 L2A at processing baseline >= 04.00
        stores ``DN = reflectance * 10000 + 1000``, and computing
        ``(G - N) / (G + N)`` on those raw DNs biases NDWI toward 0 and moves
        pixels across the threshold. For such data pass
        ``optical_offset=-1000, optical_scale=1e-4``.

    Optical pixels that are nodata (the raster's nodata value, or both bands
    exactly 0, or a non-finite NDWI) are treated like cloud: excluded from
    the optical side and filled from SAR, never counted as "land".

    Returns
    -------
    PipelineResult
        `final_output` is the real fused binary water mask (1=water,
        0=land). `metadata` includes the real SAR-only and optical-only
        masks' paths for comparison, and real per-source pixel counts.

    Raises
    ------
    ValueError
        If `fusion_mode` isn't one of the three real, documented modes.
    """
    import numpy as np
    import rasterio

    from pygeofetch.processor.indices import SpectralIndex

    if fusion_mode not in VALID_FUSION_MODES:
        raise ValueError(
            f"multi_sensor_flood_pipeline: fusion_mode must be one of "
            f"{VALID_FUSION_MODES}, got {fusion_mode!r}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stages: list[Any] = []

    sar_result = flood_mapping_pipeline(
        sar_processor,
        sar_input_path,
        reference_path=sar_reference_path,
        threshold=sar_threshold,
        output_dir=output_dir,
    )
    if not sar_result.success:
        return sar_result
    stages.extend(sar_result.stages)

    with rasterio.open(sar_result.final_output) as src:
        sar_water = src.read(1)
        profile = src.profile.copy()
        sar_shape = (src.height, src.width)

    def _read_on_sar_grid(path, label):
        with rasterio.open(path) as src:
            if (src.height, src.width) != sar_shape:
                return None, None, (
                    f"multi_sensor_flood_pipeline: {label} shape "
                    f"{(src.height, src.width)} doesn't match the real SAR "
                    f"grid {sar_shape} -- reproject it onto the SAR "
                    f"raster's real grid first."
                )
            return src.read(1), src.nodata, None

    green_raw, green_nodata, err = _read_on_sar_grid(optical_green_path, "optical green")
    if err is None:
        nir_raw, nir_nodata, err = _read_on_sar_grid(optical_nir_path, "optical NIR")
    if err is None and cloud_mask_path is not None:
        scl_raw, _, err = _read_on_sar_grid(cloud_mask_path, "cloud mask")
    if err is not None:
        return PipelineResult(success=False, stages=stages, error=err)

    # Optical nodata: declared nodata, or both bands exactly 0 (outside the
    # swath / tile). Must be excluded like cloud: NDWI there is 0/0 = NaN,
    # and `NaN > threshold` is False, which would silently read as "land".
    optical_nodata = np.zeros(sar_shape, dtype=bool)
    if green_nodata is not None:
        optical_nodata |= green_raw == green_nodata
    if nir_nodata is not None:
        optical_nodata |= nir_raw == nir_nodata
    optical_nodata |= (green_raw == 0) & (nir_raw == 0)

    green = (green_raw.astype("float64") + optical_offset) * optical_scale
    nir = (nir_raw.astype("float64") + optical_offset) * optical_scale

    si = SpectralIndex()
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = si.compute("NDWI", GREEN=green, NIR=nir)
    optical_nodata |= ~np.isfinite(ndwi)
    optical_water = (ndwi > optical_ndwi_threshold).astype("uint8")

    cloud_mask = np.zeros(sar_shape, dtype=bool)
    if cloud_mask_path is not None:
        scl = scl_raw
        classes = cloud_mask_scl_classes or [
            3,
            8,
            9,
            10,
        ]  # real Sentinel-2 SCL cloud/shadow classes
        cloud_mask = np.isin(scl, classes)

    optical_valid = ~(cloud_mask | optical_nodata)
    optical_water = np.where(optical_valid, optical_water, 0).astype("uint8")

    if fusion_mode == "cloud_aware":
        fused = np.where(optical_valid, optical_water, sar_water)
    elif fusion_mode == "union":
        fused = np.where(
            optical_valid, (optical_water | sar_water).astype("uint8"), sar_water
        )
    else:  # intersection
        fused = np.where(
            optical_valid, (optical_water & sar_water).astype("uint8"), sar_water
        )

    fused_path = output_dir / "fused_water_mask.tif"
    optical_path = output_dir / "optical_water_mask.tif"
    mask_profile = dict(profile)
    mask_profile.update(dtype="uint8", nodata=255, count=1)
    with rasterio.open(fused_path, "w", **mask_profile) as dst:
        dst.write(fused.astype("uint8"), 1)
    with rasterio.open(optical_path, "w", **mask_profile) as dst:
        dst.write(optical_water, 1)

    n_sar = int(sar_water.sum())
    n_optical = int(optical_water[optical_valid].sum()) if optical_valid.any() else 0
    n_fused = int(fused.sum())
    n_cloud = int(cloud_mask.sum())
    logger.info(
        "multi_sensor_flood_pipeline (%s): SAR water=%d, optical water=%d "
        "(over %d cloud-free px), fused water=%d, cloud-masked=%d",
        fusion_mode,
        n_sar,
        n_optical,
        int(optical_valid.sum()),
        n_fused,
        n_cloud,
    )

    return PipelineResult(
        success=True,
        final_output=fused_path,
        stages=stages,
        metadata={
            "fusion_mode": fusion_mode,
            "sar_water_mask_path": sar_result.final_output,
            "optical_water_mask_path": optical_path,
            "n_sar_water_pixels": n_sar,
            "n_optical_water_pixels": n_optical,
            "n_fused_water_pixels": n_fused,
            "n_cloud_masked_pixels": n_cloud,
            "n_optical_nodata_pixels": int(optical_nodata.sum()),
            "pct_cloud_masked": round(100 * n_cloud / sar_water.size, 2),
        },
    )
