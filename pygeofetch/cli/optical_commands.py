"""CLI commands for optical pixel offset tracking (POT)."""

from __future__ import annotations

import sys

import click
from rich.console import Console

console = Console()


@click.group()
def optical() -> None:
    """Optical pixel offset tracking — measures large 2D ground
    displacement via image correlation, complementing InSAR for
    motion that exceeds phase InSAR's coherence limit."""


@optical.command("offset-track")
@click.option("--ref", "reference_path", required=True, type=click.Path(exists=True), help="Reference-date raster.")
@click.option("--sec", "secondary_path", required=True, type=click.Path(exists=True), help="Secondary-date raster.")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output GeoTIFF (dx/dy/snr, 3 bands).")
@click.option("--band", default=1, show_default=True, type=int, help="1-based band index to correlate.")
@click.option("--window-size", default=64, show_default=True, type=int, help="Search window size, pixels.")
@click.option("--step-size", default=16, show_default=True, type=int, help="Spacing between window centers, pixels.")
@click.option("--chip-size", default=None, type=int, help="Reference template size, pixels. Default: window-size // 2.")
@click.option("--snr-threshold", default=3.0, show_default=True, type=float, help="Minimum SNR to mark a window reliable.")
@click.option("--cloud-mask", default=None, type=click.Path(exists=True), help="Real Sentinel-2 SCL raster, already aligned to --ref.")
@click.option("--water-mask-threshold", default=None, type=float, help="Mask pixels where NDWI exceeds this value (needs --green-band/--nir-band).")
@click.option("--green-band", default=None, type=int, help="1-based green band index, for --water-mask-threshold.")
@click.option("--nir-band", default=None, type=int, help="1-based NIR band index, for --water-mask-threshold.")
def offset_track_cmd(
    reference_path,
    secondary_path,
    output,
    band,
    window_size,
    step_size,
    chip_size,
    snr_threshold,
    cloud_mask,
    water_mask_threshold,
    green_band,
    nir_band,
):
    """
    Measure 2D ground displacement between two optical scenes via
    image correlation.

    \b
    Examples:
      # Basic: two co-registered scenes, real cloud masking via SCL
      pygeofetch optical offset-track \\
          --ref before.tif --sec after.tif \\
          --cloud-mask before_SCL.tif \\
          --output displacement.tif

      # Mining-scale displacement, larger search window
      pygeofetch optical offset-track \\
          --ref jan.tif --sec jun.tif \\
          --window-size 96 --step-size 24 \\
          --output mine_displacement.tif
    """
    from pygeofetch.optical.offset_tracking import compute_pixel_offsets, prepare_optical_pair

    try:
        ref_arr, sec_arr, ref_profile = prepare_optical_pair(
            reference_path,
            secondary_path,
            band_index=band,
            cloud_mask_path=cloud_mask,
            water_mask_threshold=water_mask_threshold,
            green_band_index=green_band,
            nir_band_index=nir_band,
        )
    except ValueError as exc:
        console.print(f"[red]✗[/] {exc}")
        sys.exit(1)

    pixel_size_m = abs(ref_profile["transform"].a)
    result = compute_pixel_offsets(
        ref_arr,
        sec_arr,
        pixel_size_m=pixel_size_m,
        window_size=window_size,
        step_size=step_size,
        chip_size=chip_size,
        snr_threshold=snr_threshold,
    )

    out_transform = ref_profile["transform"] * ref_profile["transform"].scale(step_size, step_size)
    out_profile = dict(ref_profile)
    out_profile.update(
        height=result.dx.shape[0],
        width=result.dx.shape[1],
        transform=out_transform,
    )
    written = result.export_geotiff(output, out_profile)

    n_reliable = int(result.reliable.sum())
    n_total = int(result.reliable.size)
    console.print(f"[green]✓[/] offset-track → {written}")
    console.print(f"  {n_reliable}/{n_total} windows reliable ({100 * n_reliable / n_total:.1f}%)")
    console.print(f"  window={window_size}px  step={step_size}px  pixel_size={pixel_size_m:.2f}m")