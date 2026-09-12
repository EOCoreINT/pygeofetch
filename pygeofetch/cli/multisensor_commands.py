"""CLI access to pygeofetch's 5 real multi-sensor pipelines."""

from __future__ import annotations

import json

import click


@click.group()
def multisensor() -> None:
    """Multi-sensor fusion pipelines: InSAR+optical, SAR+optical flood
    mapping, terrain-corrected change, DEM differencing, vegetation
    disturbance. See docs/processing/multi-sensor-pipelines.md."""


def _print_result(result) -> None:
    if result.success:
        click.echo(f"✓ Success -> {result.final_output}")
        if result.metadata:
            click.echo(
                json.dumps(
                    {k: str(v) for k, v in result.metadata.items()},
                    indent=2,
                )
            )
    else:
        click.echo(f"✗ Failed: {result.error}", err=True)
        raise SystemExit(1)


@multisensor.command("insar-optical-fusion")
@click.option("--insar-velocity", required=True, type=click.Path(exists=True))
@click.option("--insar-coherence", required=True, type=click.Path(exists=True))
@click.option("--optical-ref", required=True, type=click.Path(exists=True))
@click.option("--optical-sec", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--window-size", default=64, show_default=True, type=int)
@click.option("--step-size", default=16, show_default=True, type=int)
@click.option("--snr-threshold", default=3.0, show_default=True, type=float)
@click.option("--coherence-low", default=0.4, show_default=True, type=float)
@click.option("--coherence-high", default=0.6, show_default=True, type=float)
@click.option("--cloud-mask", default=None, type=click.Path(exists=True))
def insar_optical_fusion_cmd(
    insar_velocity,
    insar_coherence,
    optical_ref,
    optical_sec,
    output_dir,
    window_size,
    step_size,
    snr_threshold,
    coherence_low,
    coherence_high,
    cloud_mask,
):
    """Fuse InSAR displacement with optical pixel offset tracking."""
    from pygeofetch.multisensor import insar_optical_displacement_pipeline

    result = insar_optical_displacement_pipeline(
        insar_velocity,
        insar_coherence,
        optical_ref,
        optical_sec,
        output_dir,
        cloud_mask_path=cloud_mask,
        window_size=window_size,
        step_size=step_size,
        snr_threshold=snr_threshold,
        coherence_low=coherence_low,
        coherence_high=coherence_high,
    )
    _print_result(result)


@multisensor.command("flood-map")
@click.option("--sar", "sar_input", required=True, type=click.Path(exists=True))
@click.option("--optical-green", required=True, type=click.Path(exists=True))
@click.option("--optical-nir", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--sar-reference", default=None, type=click.Path(exists=True))
@click.option("--sar-threshold", default=-15.0, show_default=True, type=float)
@click.option("--ndwi-threshold", default=0.0, show_default=True, type=float)
@click.option("--cloud-mask", default=None, type=click.Path(exists=True))
@click.option(
    "--fusion-mode",
    default="cloud_aware",
    show_default=True,
    type=click.Choice(["cloud_aware", "union", "intersection"]),
)
def flood_map_cmd(
    sar_input,
    optical_green,
    optical_nir,
    output_dir,
    sar_reference,
    sar_threshold,
    ndwi_threshold,
    cloud_mask,
    fusion_mode,
):
    """Fuse SAR and optical NDWI for flood/water extent mapping."""
    from pygeofetch.multisensor import multi_sensor_flood_pipeline
    from pygeofetch.sar import SARProcessor

    result = multi_sensor_flood_pipeline(
        SARProcessor(),
        sar_input,
        optical_green,
        optical_nir,
        output_dir,
        sar_reference_path=sar_reference,
        sar_threshold=sar_threshold,
        optical_ndwi_threshold=ndwi_threshold,
        cloud_mask_path=cloud_mask,
        fusion_mode=fusion_mode,
    )
    _print_result(result)


@multisensor.command("terrain-change")
@click.option("--pre", "pre_path", required=True, type=click.Path(exists=True))
@click.option("--post", "post_path", required=True, type=click.Path(exists=True))
@click.option("--dem", "dem_path", required=True, type=click.Path(exists=True))
@click.option(
    "--pre-datetime",
    required=True,
    help="Real UTC ISO datetime, e.g. 2024-01-15T10:30:00",
)
@click.option("--post-datetime", required=True, help="Real UTC ISO datetime")
@click.option("--lat", "latitude", required=True, type=float)
@click.option("--lon", "longitude", required=True, type=float)
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--change-threshold", default=0.1, show_default=True, type=float)
def terrain_change_cmd(
    pre_path,
    post_path,
    dem_path,
    pre_datetime,
    post_datetime,
    latitude,
    longitude,
    output_dir,
    change_threshold,
):
    """Terrain-corrected optical change detection (removes real
    illumination artifacts using each date's own solar position)."""
    from datetime import datetime, timezone

    from pygeofetch.multisensor import terrain_corrected_change_pipeline

    def _parse(dt_str):
        dt = datetime.fromisoformat(dt_str)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    result = terrain_corrected_change_pipeline(
        pre_path,
        post_path,
        dem_path,
        _parse(pre_datetime),
        _parse(post_datetime),
        latitude,
        longitude,
        output_dir,
        change_threshold=change_threshold,
    )
    _print_result(result)


@multisensor.command("dem-diff")
@click.option("--dem-old", required=True, type=click.Path(exists=True))
@click.option("--dem-new", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option(
    "--old-rmse",
    default=1.0,
    show_default=True,
    type=float,
    help="Real vertical RMSE of the old DEM, metres.",
)
@click.option(
    "--new-rmse",
    default=1.0,
    show_default=True,
    type=float,
    help="Real vertical RMSE of the new DEM, metres.",
)
@click.option("--confidence-t", default=1.96, show_default=True, type=float)
def dem_diff_cmd(dem_old, dem_new, output_dir, old_rmse, new_rmse, confidence_t):
    """DEM-of-Difference volumetric change analysis."""
    from pygeofetch.multisensor import dem_differencing_pipeline

    result = dem_differencing_pipeline(
        dem_old,
        dem_new,
        output_dir,
        dem_old_vertical_rmse_m=old_rmse,
        dem_new_vertical_rmse_m=new_rmse,
        confidence_t_value=confidence_t,
    )
    _print_result(result)


@multisensor.command("vegetation-disturbance")
@click.option("--red-pre", required=True, type=click.Path(exists=True))
@click.option("--nir-pre", required=True, type=click.Path(exists=True))
@click.option("--red-post", required=True, type=click.Path(exists=True))
@click.option("--nir-post", required=True, type=click.Path(exists=True))
@click.option("--slc-pre", required=True, type=click.Path(exists=True))
@click.option("--slc-post", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--ndvi-drop-threshold", default=0.15, show_default=True, type=float)
@click.option("--coherence-threshold", default=0.3, show_default=True, type=float)
@click.option("--coherence-window", default=7, show_default=True, type=int)
def vegetation_disturbance_cmd(
    red_pre,
    nir_pre,
    red_post,
    nir_post,
    slc_pre,
    slc_post,
    output_dir,
    ndvi_drop_threshold,
    coherence_threshold,
    coherence_window,
):
    """Classify visible canopy loss vs. real sub-canopy ground
    disturbance, from optical NDVI + SAR coherence."""
    from pygeofetch.multisensor import vegetation_disturbance_pipeline

    result = vegetation_disturbance_pipeline(
        red_pre,
        nir_pre,
        red_post,
        nir_post,
        slc_pre,
        slc_post,
        output_dir,
        ndvi_drop_threshold=ndvi_drop_threshold,
        coherence_threshold=coherence_threshold,
        coherence_window=coherence_window,
    )
    _print_result(result)
