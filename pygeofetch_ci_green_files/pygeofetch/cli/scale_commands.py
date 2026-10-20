"""CLI commands for remote streaming, GeoParquet/Zarr export, and
overlap-aware chunked processing."""

from __future__ import annotations

import sys

import click
from rich.console import Console

console = Console()


def _engine():
    from pygeofetch.core.engine import PyGeoFetch

    return PyGeoFetch(log_level="WARNING")


def _pr_success(message: str, path=None) -> None:
    console.print(f"[green]✓[/] {message}" + (f" → {path}" if path else ""))


def _pr_fail(message: str) -> None:
    console.print(f"[red]✗[/] {message}")
    sys.exit(1)


@click.group()
def stream() -> None:
    """Remote COG streaming, GeoParquet/Zarr export, and chunked processing."""


@stream.command("cog-window")
@click.argument("url")
@click.option("--bbox", required=True, help="min_lon,min_lat,max_lon,max_lat")
@click.option("--bbox-crs", default="EPSG:4326", show_default=True)
@click.option(
    "--bands",
    default=None,
    help="Comma-separated 1-based band indices. Default: all real bands.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    help="Output GeoTIFF path for the real, windowed result.",
)
@click.option(
    "--no-range-check",
    is_flag=True,
    help="Skip the real, honest pre-flight range-request support check.",
)
def cog_window_cmd(url, bbox, bbox_crs, bands, output, no_range_check):
    """Read a real, spatial window from a remote COG without downloading the whole file.

    Example: pygeofetch stream cog-window https://example.com/scene.tif \\
             --bbox 36.7,-1.4,36.9,-1.2 -o window.tif
    """
    from pygeofetch.stac_compute.cog_streamer import read_remote_window

    try:
        parts = [float(x) for x in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError
        bbox_tuple = tuple(parts)
    except ValueError:
        _pr_fail(
            "--bbox must be exactly 4 comma-separated numbers: min_lon,min_lat,max_lon,max_lat"
        )
        return
    band_list = [int(b) for b in bands.split(",")] if bands else None

    try:
        result = read_remote_window(
            url,
            bbox=bbox_tuple,
            bbox_crs=bbox_crs,
            bands=band_list,
            check_range_support=not no_range_check,
        )
    except Exception as exc:
        _pr_fail(f"cog-window failed: {exc}")
        return

    import rasterio

    data = result.data
    count = 1 if data.ndim == 2 else data.shape[0]
    profile = {
        "driver": "GTiff",
        "height": result.window.height,
        "width": result.window.width,
        "count": count,
        "dtype": str(data.dtype),
        "transform": result.transform,
        "crs": result.crs,
        "BIGTIFF": "YES",
    }
    with rasterio.open(output, "w", **profile) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data)

    real_saved_pct = 100 * (
        1
        - result.real_window_pixel_count
        * 4
        / max(result.bytes_would_have_downloaded_full_file, 1)
    )
    _pr_success(
        f"cog-window ({result.real_window_pixel_count} real pixels read, "
        f"~{max(real_saved_pct, 0):.0f}% less than a full download)",
        output,
    )


@stream.command("to-geoparquet")
@click.argument("input_vector", type=click.Path(exists=True))
@click.option("--output", "-o", required=True)
def to_geoparquet_cmd(input_vector, output):
    """Convert any real vector file to GeoParquet.

    Example: pygeofetch stream to-geoparquet parcels.geojson -o parcels.parquet
    """
    from pygeofetch.core.data_formats import MissingDependencyError
    from pygeofetch.core.data_formats.geoparquet_io import vector_to_geoparquet

    try:
        out = vector_to_geoparquet(input_vector, output)
    except MissingDependencyError as exc:
        _pr_fail(str(exc))
        return
    _pr_success("to-geoparquet", out)


@stream.command("to-zarr")
@click.argument("input_raster", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output .zarr store directory.")
@click.option(
    "--chunks",
    default=None,
    help="Comma-separated chunk shape, e.g. 512,512. Default: real, automatic.",
)
@click.option(
    "--compressor",
    default="zstd",
    show_default=True,
    type=click.Choice(["zstd", "lz4", "gzip", "zlib", "none"]),
)
def to_zarr_cmd(input_raster, output, chunks, compressor):
    """Convert a real GeoTIFF to a chunked Zarr store.

    Example: pygeofetch stream to-zarr velocity.tif -o velocity.zarr --chunks 512,512
    """
    import rasterio

    from pygeofetch.core.data_formats import MissingDependencyError
    from pygeofetch.core.data_formats.zarr_io import write_raster_to_zarr

    with rasterio.open(input_raster) as src:
        data = src.read(1)
        transform, crs, nodata = src.transform, src.crs, src.nodata

    chunk_tuple = tuple(int(c) for c in chunks.split(",")) if chunks else None
    try:
        out = write_raster_to_zarr(
            data,
            output,
            transform=transform,
            crs=crs,
            nodata=nodata,
            chunks=chunk_tuple,
            compressor=compressor,
        )
    except MissingDependencyError as exc:
        _pr_fail(str(exc))
        return
    _pr_success(
        f"to-zarr ({data.shape[0]}x{data.shape[1]}, compressor={compressor})", out
    )


@stream.command("zarr-to-tif")
@click.argument("input_zarr", type=click.Path(exists=True))
@click.option("--output", "-o", required=True)
def zarr_to_tif_cmd(input_zarr, output):
    """Convert a real Zarr store (written with real geo metadata) back to GeoTIFF.

    Example: pygeofetch stream zarr-to-tif velocity.zarr -o velocity.tif
    """
    from pygeofetch.core.data_formats.zarr_io import zarr_to_geotiff

    try:
        out = zarr_to_geotiff(input_zarr, output)
    except ValueError as exc:
        _pr_fail(str(exc))
        return
    _pr_success("zarr-to-tif", out)


@stream.command("chunked-process")
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--expression",
    "-e",
    required=True,
    help='Python expression over tile arrays, e.g. "(B[1]-B[0])/(B[1]+B[0]+1e-6)" '
    'for pixel-independent math, or "uniform_filter(B[0], size=3)" for a real neighborhood filter.',
)
@click.option("--tile-size", default=1024, show_default=True, type=int)
@click.option(
    "--overlap",
    default=0,
    show_default=True,
    type=int,
    help="Real pixel margin for neighborhood operations -- 0 for pixel-independent math.",
)
@click.option(
    "--output-format",
    default="cog",
    show_default=True,
    type=click.Choice(["cog", "zarr"]),
)
@click.option("--output", "-o", required=True)
def chunked_process_cmd(inputs, expression, tile_size, overlap, output_format, output):
    """Process a large raster in real, memory-safe, overlap-aware tiles.

    B[0], B[1], ... refer to each input in order. numpy is available as
    np, and scipy.ndimage's uniform_filter/median_filter/generic_filter
    are available directly by name for real neighborhood operations.

    Examples:

    \b
      # Pixel-independent (overlap=0 is fine)
      pygeofetch stream chunked-process red.tif nir.tif \\
          -e "(B[1]-B[0])/(B[1]+B[0]+1e-6)" -o ndvi.tif

    \b
      # Real neighborhood operation (needs overlap >= window_size // 2)
      pygeofetch stream chunked-process large_raster.tif \\
          -e "uniform_filter(B[0], size=3, mode='nearest')" \\
          --overlap 1 -o smoothed.tif
    """
    from pydantic import ValidationError

    from pygeofetch.core.chunked_processor import ChunkedProcessor
    from pygeofetch.models.stream_options import ChunkConfig

    try:
        config = ChunkConfig(
            tile_size=tile_size, overlap=overlap, output_format=output_format
        )
    except ValidationError as exc:
        _pr_fail(f"Invalid configuration: {exc}")
        return

    def _formula(tiles):
        import numpy as np
        from scipy.ndimage import generic_filter, median_filter, uniform_filter

        return eval(
            expression,
            {  # noqa: S307
                "B": tiles,
                "np": np,
                "uniform_filter": uniform_filter,
                "median_filter": median_filter,
                "generic_filter": generic_filter,
            },
        )

    processor = ChunkedProcessor(
        tile_size=config.tile_size,
        overlap=config.overlap,
        output_format=config.output_format,
    )
    try:
        out = processor.process(inputs=list(inputs), formula=_formula, output=output)
    except Exception as exc:
        _pr_fail(f"chunked-process failed: {exc}")
        return
    _pr_success(
        f"chunked-process (tile_size={tile_size}, overlap={overlap}, format={output_format})",
        out,
    )
