"""CLI commands for SAR processing."""

from __future__ import annotations

import sys

import click
from rich.console import Console

console = Console()


def _engine():
    from pygeofetch.core.engine import PyGeoFetch

    return PyGeoFetch(log_level="WARNING")


def _pr(result, name):
    if result.success:
        size = (
            result.output_path.stat().st_size / (1024 * 1024)
            if result.output_path and result.output_path.exists()
            else 0
        )
        console.print(
            f"[green]✓[/] {name} → {result.output_path}"
            f" ({size:.1f} MB, {result.duration_seconds:.2f}s)"
        )
        for k, v in (result.metadata or {}).items():
            console.print(f"  {k}: {v}")
    else:
        console.print(f"[red]✗[/] {name} failed: {result.error}")
        sys.exit(1)


@click.group()
def sar() -> None:
    """SAR processing — despeckle, calibrate, flood map, coherence."""


@sar.command("despeckle")
@click.argument("input", type=click.Path(exists=True))
@click.option(
    "--filter",
    "-f",
    "filter_name",
    default="lee",
    show_default=True,
    type=click.Choice(["lee", "enhanced_lee", "frost", "gamma", "boxcar"]),
)
@click.option("--window", "-w", default=5, show_default=True, type=int)
@click.option("--looks", default=1, show_default=True, type=int)
@click.option("--output", "-o", default=None)
def despeckle_cmd(input, filter_name, window, looks, output):
    """SAR speckle filtering (Lee, Enhanced Lee, Frost, Gamma MAP, Boxcar)."""
    e = _engine()
    r = e.sar.despeckle(
        input, filter=filter_name, window=window, num_looks=looks, output=output
    )
    _pr(r, f"despeckle ({filter_name}, {window}×{window})")


@sar.command("calibrate")
@click.argument("input", type=click.Path(exists=True))
@click.option(
    "--output-type", default="sigma0", type=click.Choice(["sigma0", "gamma0", "beta0"])
)
@click.option("--db/--linear", "in_db", default=True, show_default=True)
@click.option("--output", "-o", default=None)
def calibrate_cmd(input, output_type, in_db, output):
    """Radiometric calibration: DN → sigma0/gamma0/beta0 backscatter coefficient."""
    e = _engine()
    r = e.sar.calibrate(input, output_type=output_type, in_db=in_db, output=output)
    _pr(r, f"calibrate ({output_type}, {'dB' if in_db else 'linear'})")


@sar.command("flood-map")
@click.argument("input", type=click.Path(exists=True))
@click.option(
    "--threshold",
    "-t",
    default=-15.0,
    show_default=True,
    type=float,
    help="Backscatter threshold below which pixels = water (dB).",
)
@click.option(
    "--reference",
    "-r",
    default=None,
    type=click.Path(exists=True),
    help="Pre-event reference image for change-based detection.",
)
@click.option("--output", "-o", default=None)
def flood_map_cmd(input, threshold, reference, output):
    """Flood mapping from SAR backscatter (simple threshold or change-based)."""
    e = _engine()
    r = e.sar.flood_map(input, threshold=threshold, reference=reference, output=output)
    _pr(r, "flood-map")


@sar.command("coherence")
@click.argument("image1", type=click.Path(exists=True))
@click.argument("image2", type=click.Path(exists=True))
@click.option("--window", "-w", default=7, show_default=True, type=int)
@click.option("--output", "-o", default=None)
def coherence_cmd(image1, image2, window, output):
    """Interferometric coherence between two co-registered SAR images."""
    e = _engine()
    r = e.sar.coherence(image1, image2, window=window, output=output)
    _pr(r, "coherence")
