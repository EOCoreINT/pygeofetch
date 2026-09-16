"""CLI commands for post-processing: vectorize, smooth, zonal-stats, COG, etc."""

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
def post() -> None:
    """Post-process outputs — vectorize, smooth, zonal stats, buffer, COG, compress."""


@post.command("vectorize")
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
@click.option("--band", "-b", default=1, show_default=True, type=int)
@click.option(
    "--threshold",
    "-t",
    default=None,
    type=float,
    help="Binary threshold before vectorizing.",
)
@click.option(
    "--format", "-f", default="geojson", type=click.Choice(["geojson", "gpkg", "shp"])
)
@click.option(
    "--min-area",
    default=None,
    type=float,
    help="Minimum polygon area to keep (CRS units²).",
)
def vectorize_cmd(input, output, band, threshold, format, min_area):
    """Convert raster to vector polygons (polygonize)."""
    e = _engine()
    r = e.post.vectorize(
        input,
        output=output,
        band=band,
        threshold=threshold,
        format=format,
        min_area=min_area,
    )
    _pr(r, "vectorize")


@post.command("smooth")
@click.argument("input", type=click.Path(exists=True))
@click.option("--tolerance", "-t", default=1.0, show_default=True, type=float)
@click.option(
    "--method", "-m", default="simplify", type=click.Choice(["simplify", "buffer"])
)
@click.option("--output", "-o", default=None)
def smooth_cmd(input, tolerance, method, output):
    """Smooth/simplify vector geometries (Douglas-Peucker or buffer-unbuffer)."""
    e = _engine()
    r = e.post.smooth(input, tolerance=tolerance, method=method, output=output)
    _pr(r, f"smooth ({method}, tol={tolerance})")


@post.command("regularize")
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
def regularize_cmd(input, output):
    """Regularize (orthogonalize) building footprints or polygons."""
    e = _engine()
    r = e.post.regularize(input, output=output)
    _pr(r, "regularize")


@post.command("zonal-stats")
@click.argument("raster", type=click.Path(exists=True))
@click.argument("zones", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output CSV path.")
@click.option(
    "--stats",
    default="count,mean,median,min,max,std",
    help="Comma-separated statistics to compute.",
)
@click.option("--band", "-b", default=1, show_default=True, type=int)
def zonal_stats_cmd(raster, zones, output, stats, band):
    """Compute zonal statistics for each polygon zone."""
    e = _engine()
    stat_list = [s.strip() for s in stats.split(",")]
    r = e.post.zonal_stats(
        raster=raster, zones=zones, output=output, stats=stat_list, band=band
    )
    _pr(r, "zonal-stats")


@post.command("buffer")
@click.argument("input", type=click.Path(exists=True))
@click.option("--distance", "-d", required=True, type=float)
@click.option(
    "--cap-style", default="round", type=click.Choice(["round", "flat", "square"])
)
@click.option("--output", "-o", default=None)
def buffer_cmd(input, distance, cap_style, output):
    """Add buffer around vector geometries."""
    e = _engine()
    r = e.post.buffer(input, distance=distance, cap_style=cap_style, output=output)
    _pr(r, f"buffer ({distance} units)")


@post.command("centroids")
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
def centroids_cmd(input, output):
    """Extract centroid points from polygons."""
    e = _engine()
    r = e.post.centroids(input, output=output)
    _pr(r, "centroids")


@post.command("geometry-metrics")
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
def metrics_cmd(input, output):
    """Add area, perimeter, and compactness columns to vector file."""
    e = _engine()
    r = e.post.add_geometry_metrics(input, output=output)
    _pr(r, "geometry-metrics")


@post.command("compress")
@click.argument("input", type=click.Path(exists=True))
@click.option(
    "--method",
    "-m",
    default="lzw",
    show_default=True,
    type=click.Choice(["lzw", "deflate", "zstd", "packbits"]),
)
@click.option("--output", "-o", default=None)
def compress_cmd(input, method, output):
    """Apply lossless compression to a GeoTIFF."""
    e = _engine()
    r = e.post.compress(input, method=method, output=output)
    _pr(r, f"compress ({method})")


@post.command("cog")
@click.argument("input", type=click.Path(exists=True))
@click.option(
    "--compress",
    "-c",
    default="deflate",
    type=click.Choice(["deflate", "lzw", "zstd", "none"]),
)
@click.option("--blocksize", "-b", default=512, show_default=True, type=int)
@click.option("--output", "-o", default=None)
def cog_cmd(input, compress, blocksize, output):
    """Convert to Cloud Optimized GeoTIFF (COG) with internal tiling and overviews."""
    e = _engine()
    r = e.post.cog(input, compress=compress, blocksize=blocksize, output=output)
    _pr(r, f"COG ({compress}, {blocksize}px)")


@post.command("overlay")
@click.argument("left", type=click.Path(exists=True))
@click.argument("right", type=click.Path(exists=True))
@click.option("--output", "-o", required=True)
@click.option("--how", default="intersection", show_default=True,
              type=click.Choice(["intersection", "union", "difference", "symmetric_difference", "identity"]))
def overlay_cmd(left, right, output, how):
    """Real polygon overlay between two vector layers.

    Example: pygeofetch post overlay flood.geojson parcels.geojson -o flooded_parcels.geojson --how intersection
    """
    e = _engine()
    r = e.post.overlay(left, right, output=output, how=how)
    _pr(r, f"overlay ({how})")


@post.command("dissolve")
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", "-o", required=True)
@click.option("--by", default=None, help="Attribute column to group by before dissolving.")
@click.option("--aggfunc", default="first", show_default=True, help="How to aggregate other columns within each group.")
def dissolve_cmd(input, output, by, aggfunc):
    """Merge polygons sharing the same attribute value into single geometries.

    Example: pygeofetch post dissolve parcels.geojson -o zones.geojson --by land_use
    """
    e = _engine()
    r = e.post.dissolve(input, output=output, by=by, aggfunc=aggfunc)
    _pr(r, f"dissolve (by={by})")


@post.command("spatial-join")
@click.argument("left", type=click.Path(exists=True))
@click.argument("right", type=click.Path(exists=True))
@click.option("--output", "-o", required=True)
@click.option("--how", default="inner", show_default=True, type=click.Choice(["inner", "left", "right"]))
@click.option("--predicate", default="intersects", show_default=True,
              type=click.Choice(["intersects", "contains", "within", "touches", "crosses", "overlaps"]))
def spatial_join_cmd(left, right, output, how, predicate):
    """Attach attributes from one vector layer to another by real spatial relationship.

    Example: pygeofetch post spatial-join points.geojson zones.geojson -o joined.geojson --predicate within
    """
    e = _engine()
    r = e.post.spatial_join(left, right, output=output, how=how, predicate=predicate)
    _pr(r, f"spatial-join ({predicate})")
