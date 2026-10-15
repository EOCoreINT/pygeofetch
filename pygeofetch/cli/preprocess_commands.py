"""CLI commands for preprocessing: clip, reproject, resample, cloud-mask, etc."""

from __future__ import annotations

import sys

import click
from rich.console import Console

console = Console()


@click.group()
def preprocess() -> None:
    """Preprocess satellite imagery — clip, reproject, cloud mask, resample, etc."""


def _engine():
    from pygeofetch.core.engine import PyGeoFetch

    return PyGeoFetch(log_level="WARNING")


def _print_result(result, operation: str) -> None:
    if result.success:
        size_mb = (
            result.output_path.stat().st_size / (1024 * 1024)
            if result.output_path and result.output_path.exists()
            else 0
        )
        console.print(
            f"[green]✓[/] {operation} → {result.output_path}"
            f" ({size_mb:.1f} MB, {result.duration_seconds:.2f}s)"
        )
    else:
        console.print(f"[red]✗[/] {operation} failed: {result.error}")
        sys.exit(1)


@preprocess.command("clip")
@click.argument("input", type=click.Path(exists=True))
@click.option("--bbox", "-b", default=None, help='"minx,miny,maxx,maxy"')
@click.option("--geometry", "-g", default=None, type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
@click.option("--all-touched", is_flag=True)
def clip_cmd(input, bbox, geometry, output, all_touched):
    """Clip raster to bounding box or polygon geometry."""
    e = _engine()
    bbox_tuple = None
    if bbox:
        parts = [float(x) for x in bbox.split(",")]
        bbox_tuple = tuple(parts)
    r = e.preprocess.clip(
        input,
        bbox=bbox_tuple,
        geometry=geometry,
        output=output,
        all_touched=all_touched,
    )
    _print_result(r, "clip")


@preprocess.command("reproject")
@click.argument("input", type=click.Path(exists=True))
@click.option("--crs", "-c", default="EPSG:4326", show_default=True)
@click.option(
    "--resampling",
    default="bilinear",
    show_default=True,
    type=click.Choice(["nearest", "bilinear", "cubic", "lanczos", "average"]),
)
@click.option("--resolution", "-r", default=None, type=float)
@click.option("--output", "-o", default=None)
def reproject_cmd(input, crs, resampling, resolution, output):
    """Reproject raster to a new CRS (e.g. EPSG:4326, EPSG:32618)."""
    e = _engine()
    r = e.preprocess.reproject(
        input, crs=crs, resampling=resampling, resolution=resolution, output=output
    )
    _print_result(r, f"reproject → {crs}")


@preprocess.command("resample")
@click.argument("input", type=click.Path(exists=True))
@click.option(
    "--resolution",
    "-r",
    default=None,
    type=float,
    help="Target resolution in CRS units.",
)
@click.option(
    "--scale",
    "-s",
    default=None,
    type=float,
    help="Scale factor (0.5=half, 2.0=double).",
)
@click.option(
    "--method",
    "-m",
    default="bilinear",
    show_default=True,
    type=click.Choice(["nearest", "bilinear", "cubic", "lanczos", "average"]),
)
@click.option("--output", "-o", default=None)
def resample_cmd(input, resolution, scale, method, output):
    """Resample raster to different spatial resolution."""
    if not resolution and not scale:
        console.print("[red]Provide --resolution or --scale[/]")
        sys.exit(1)
    e = _engine()
    r = e.preprocess.resample(
        input, resolution=resolution, scale_factor=scale, method=method, output=output
    )
    _print_result(r, "resample")


@preprocess.command("cloud-mask")
@click.argument("input", type=click.Path(exists=True))
@click.option(
    "--method",
    "-m",
    default="scl",
    show_default=True,
    type=click.Choice(["scl", "fmask", "threshold", "ndsi"]),
)
@click.option("--scl-band", default=None, type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
def cloud_mask_cmd(input, method, scl_band, output):
    """Apply cloud masking to set cloud pixels to NoData."""
    e = _engine()
    r = e.preprocess.cloud_mask(input, method=method, scl_band=scl_band, output=output)
    _print_result(r, f"cloud-mask ({method})")
    if r.success:
        console.print(f"  Masked: {r.metadata.get('masked_pct', '?')}% pixels")


@preprocess.command("cloud-fill")
@click.argument("input", type=click.Path(exists=True))
@click.argument("time_series", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--method",
    "-m",
    default="interpolate",
    type=click.Choice(["interpolate", "nearest"]),
)
@click.option("--output", "-o", default=None)
def cloud_fill_cmd(input, time_series, method, output):
    """Fill cloud gaps using a time series of auxiliary scenes."""
    if not time_series:
        console.print("[red]Provide at least one time series scene[/]")
        sys.exit(1)
    e = _engine()
    r = e.preprocess.cloud_fill(input, list(time_series), method=method, output=output)
    _print_result(r, "cloud-fill")


@preprocess.command("atmos")
@click.argument("input", type=click.Path(exists=True))
@click.option(
    "--method",
    "-m",
    default="dos1",
    show_default=True,
    type=click.Choice(["dos1", "dos2", "sen2cor", "flaash", "6s", "icor"]),
)
@click.option("--output", "-o", default=None)
def atmos_cmd(input, method, output):
    """Atmospheric correction (DOS1, DOS2, Sen2Cor, FLAASH, 6S, iCOR)."""
    e = _engine()
    r = e.preprocess.atmos(input, method=method, output=output)
    _print_result(r, f"atmos ({method})")


@preprocess.command("topo-correct")
@click.argument("input", type=click.Path(exists=True))
@click.argument("dem", type=click.Path(exists=True))
@click.option(
    "--method",
    "-m",
    default="cosine",
    type=click.Choice(["cosine", "minnaert", "c_correction"]),
)
@click.option("--output", "-o", default=None)
def topo_cmd(input, dem, method, output):
    """Topographic correction using a DEM (cosine, Minnaert, C-correction)."""
    e = _engine()
    r = e.preprocess.topo_correct(input, dem, method=method, output=output)
    _print_result(r, f"topo-correct ({method})")


@preprocess.command("pansharpen")
@click.argument("pan", type=click.Path(exists=True))
@click.argument("ms", type=click.Path(exists=True))
@click.option(
    "--method",
    "-m",
    default="brovey",
    type=click.Choice(["brovey", "ihs", "gram_schmidt", "simple_mean"]),
)
@click.option("--output", "-o", default=None)
def pansharpen_cmd(pan, ms, method, output):
    """Pan-sharpen multispectral using panchromatic band."""
    e = _engine()
    r = e.preprocess.pansharpen(pan, ms, method=method, output=output)
    _print_result(r, f"pansharpen ({method})")


@preprocess.command("mosaic")
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--method",
    "-m",
    default="first",
    type=click.Choice(["first", "last", "min", "max"]),
)
@click.option("--output", "-o", default=None)
def mosaic_cmd(inputs, method, output):
    """Merge multiple rasters into a seamless mosaic."""
    if not inputs:
        console.print("[red]Provide at least 2 input rasters[/]")
        sys.exit(1)
    e = _engine()
    r = e.preprocess.mosaic(list(inputs), method=method, output=output)
    _print_result(r, f"mosaic ({len(inputs)} inputs, {method})")


@preprocess.command("composite")
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--method",
    "-m",
    default="median",
    type=click.Choice(["median", "mean", "max", "min", "best_pixel"]),
)
@click.option("--output", "-o", default=None)
def composite_cmd(inputs, method, output):
    """Create multi-temporal composite (median, mean, max, min, best-pixel)."""
    if len(inputs) < 2:
        console.print("[red]Provide at least 2 input rasters[/]")
        sys.exit(1)
    e = _engine()
    r = e.preprocess.composite(list(inputs), method=method, output=output)
    _print_result(r, f"composite ({method}, {len(inputs)} inputs)")


@preprocess.command("tile")
@click.argument("input", type=click.Path(exists=True))
@click.option("--tile-size", default=512, show_default=True, type=int)
@click.option("--overlap", default=64, show_default=True, type=int)
@click.option("--output-dir", "-o", default=None)
@click.option("--min-coverage", default=0.1, show_default=True, type=float)
def tile_cmd(input, tile_size, overlap, output_dir, min_coverage):
    """Split large raster into overlapping tiles for AI inference."""
    e = _engine()
    r = e.preprocess.tile(
        input,
        tile_size=tile_size,
        overlap=overlap,
        output_dir=output_dir,
        min_coverage=min_coverage,
    )
    _print_result(r, "tile")
    if r.success:
        console.print(
            f"  Created {r.metadata.get('tile_count', '?')} tiles in {r.output_path}"
        )
