"""CLI commands for spectral indices."""

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
        size_mb = (
            result.output_path.stat().st_size / (1024 * 1024)
            if result.output_path and result.output_path.exists()
            else 0
        )
        meta = result.metadata or {}
        console.print(
            f"[green]✓[/] {name} → {result.output_path}"
            f" ({size_mb:.1f} MB, {result.duration_seconds:.2f}s)"
        )
        if meta:
            for k, v in meta.items():
                if k != "expression":
                    console.print(f"  {k}: {v}")
    else:
        console.print(f"[red]✗[/] {name} failed: {result.error}")
        sys.exit(1)


@click.group()
def index() -> None:
    """Compute spectral indices — NDVI, EVI, NDWI, NDBI, TCT, PCA, LST, and more."""


@index.command("ndvi")
@click.option(
    "--red", required=True, type=click.Path(exists=True), help="Red band (e.g. B04.tif)"
)
@click.option(
    "--nir", required=True, type=click.Path(exists=True), help="NIR band (e.g. B08.tif)"
)
@click.option("--output", "-o", default=None)
@click.option(
    "--chunked", is_flag=True,
    help="Process in memory-safe tiles for large rasters (e.g. drone orthomosaics) that exceed available RAM or the 4GB GeoTIFF limit.",
)
@click.option("--tile-size", default=1024, show_default=True, type=int, help="Tile edge length in pixels when --chunked is set.")
def ndvi_cmd(red, nir, output, chunked, tile_size):
    """NDVI — Normalized Difference Vegetation Index. Range: -1 to +1. Vegetation > 0.3.

    Pass the SAME path to --red and --nir to auto-detect both bands from
    one merged multi-band image (e.g. a drone orthomosaic), using its
    real, embedded band descriptions.
    """
    e = _engine()
    r = e.indices.ndvi(red=red, nir=nir, output=output, chunked=chunked, tile_size=tile_size)
    _pr(r, "NDVI")


@index.command("ndre")
@click.option(
    "--rededge", required=True, type=click.Path(exists=True), help="Red edge band"
)
@click.option(
    "--nir", required=True, type=click.Path(exists=True), help="NIR band"
)
@click.option("--output", "-o", default=None)
@click.option(
    "--chunked", is_flag=True,
    help="Process in memory-safe tiles for large rasters that exceed available RAM or the 4GB GeoTIFF limit.",
)
@click.option("--tile-size", default=1024, show_default=True, type=int, help="Tile edge length in pixels when --chunked is set.")
def ndre_cmd(rededge, nir, output, chunked, tile_size):
    """NDRE — Normalized Difference Red Edge Index. Range: -1 to +1.

    Less prone to saturation than NDVI in dense canopy -- useful for
    crop stress and chlorophyll monitoring from drone multispectral
    sensors. Pass the SAME path to --rededge and --nir to auto-detect
    both bands from one merged multi-band image.
    """
    e = _engine()
    r = e.indices.ndre(rededge=rededge, nir=nir, output=output, chunked=chunked, tile_size=tile_size)
    _pr(r, "NDRE")


@index.command("evi")
@click.option("--blue", required=True, type=click.Path(exists=True))
@click.option("--red", required=True, type=click.Path(exists=True))
@click.option("--nir", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
@click.option("--chunked", is_flag=True, help="Process in memory-safe tiles for large rasters.")
@click.option("--tile-size", default=1024, show_default=True, type=int)
def evi_cmd(blue, red, nir, output, chunked, tile_size):
    """EVI — Enhanced Vegetation Index. Better than NDVI in high-biomass areas.

    Pass the SAME path to --blue, --red, and --nir to auto-detect all
    three bands from one merged multi-band image.
    """
    e = _engine()
    r = e.indices.evi(blue=blue, red=red, nir=nir, output=output, chunked=chunked, tile_size=tile_size)
    _pr(r, "EVI")


@index.command("savi")
@click.option("--red", required=True, type=click.Path(exists=True))
@click.option("--nir", required=True, type=click.Path(exists=True))
@click.option("--soil-l", default=0.5, show_default=True, type=float)
@click.option("--output", "-o", default=None)
@click.option("--chunked", is_flag=True, help="Process in memory-safe tiles for large rasters.")
@click.option("--tile-size", default=1024, show_default=True, type=int)
def savi_cmd(red, nir, soil_l, output, chunked, tile_size):
    """SAVI — Soil Adjusted Vegetation Index. Reduces soil brightness influence.

    Pass the SAME path to --red and --nir to auto-detect both bands
    from one merged multi-band image.
    """
    e = _engine()
    r = e.indices.savi(red=red, nir=nir, L=soil_l, output=output, chunked=chunked, tile_size=tile_size)
    _pr(r, "SAVI")


@index.command("ndwi")
@click.option("--green", required=True, type=click.Path(exists=True))
@click.option("--nir", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
@click.option("--chunked", is_flag=True, help="Process in memory-safe tiles for large rasters.")
@click.option("--tile-size", default=1024, show_default=True, type=int)
def ndwi_cmd(green, nir, output, chunked, tile_size):
    """NDWI — Normalized Difference Water Index. Water > 0.

    Pass the SAME path to --green and --nir to auto-detect both bands
    from one merged multi-band image.
    """
    e = _engine()
    r = e.indices.ndwi(green=green, nir=nir, output=output, chunked=chunked, tile_size=tile_size)
    _pr(r, "NDWI")


@index.command("mndwi")
@click.option("--green", required=True, type=click.Path(exists=True))
@click.option("--swir1", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
@click.option("--chunked", is_flag=True, help="Process in memory-safe tiles for large rasters.")
@click.option("--tile-size", default=1024, show_default=True, type=int)
def mndwi_cmd(green, swir1, output, chunked, tile_size):
    """MNDWI — Modified NDWI. Better separation of water from built-up areas.

    Pass the SAME path to --green and --swir1 to auto-detect both
    bands from one merged multi-band image.
    """
    e = _engine()
    r = e.indices.mndwi(green=green, swir1=swir1, output=output, chunked=chunked, tile_size=tile_size)
    _pr(r, "MNDWI")


@index.command("ndbi")
@click.option("--nir", required=True, type=click.Path(exists=True))
@click.option("--swir1", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
@click.option("--chunked", is_flag=True, help="Process in memory-safe tiles for large rasters.")
@click.option("--tile-size", default=1024, show_default=True, type=int)
def ndbi_cmd(nir, swir1, output, chunked, tile_size):
    """NDBI — Normalized Difference Built-up Index. Urban > 0.

    Pass the SAME path to --nir and --swir1 to auto-detect both bands
    from one merged multi-band image.
    """
    e = _engine()
    r = e.indices.ndbi(nir=nir, swir1=swir1, output=output, chunked=chunked, tile_size=tile_size)
    _pr(r, "NDBI")


@index.command("ndsi")
@click.option("--green", required=True, type=click.Path(exists=True))
@click.option("--swir1", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
@click.option("--chunked", is_flag=True, help="Process in memory-safe tiles for large rasters.")
@click.option("--tile-size", default=1024, show_default=True, type=int)
def ndsi_cmd(green, swir1, output, chunked, tile_size):
    """NDSI — Snow Index. Snow > 0.4.

    Pass the SAME path to --green and --swir1 to auto-detect both
    bands from one merged multi-band image.
    """
    e = _engine()
    r = e.indices.ndsi(green=green, swir1=swir1, output=output, chunked=chunked, tile_size=tile_size)
    _pr(r, "NDSI")


@index.command("ndmi")
@click.option("--nir", required=True, type=click.Path(exists=True))
@click.option("--swir1", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
@click.option("--chunked", is_flag=True, help="Process in memory-safe tiles for large rasters.")
@click.option("--tile-size", default=1024, show_default=True, type=int)
def ndmi_cmd(nir, swir1, output, chunked, tile_size):
    """NDMI — Moisture Index. Sensitive to canopy water content.

    Pass the SAME path to --nir and --swir1 to auto-detect both bands
    from one merged multi-band image.
    """
    e = _engine()
    r = e.indices.ndmi(nir=nir, swir1=swir1, output=output, chunked=chunked, tile_size=tile_size)
    _pr(r, "NDMI")


@index.command("nbr")
@click.option("--nir", required=True, type=click.Path(exists=True))
@click.option("--swir2", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
@click.option("--chunked", is_flag=True, help="Process in memory-safe tiles for large rasters.")
@click.option("--tile-size", default=1024, show_default=True, type=int)
def nbr_cmd(nir, swir2, output, chunked, tile_size):
    """NBR — Normalized Burn Ratio. Use dNBR for burn severity.

    Pass the SAME path to --nir and --swir2 to auto-detect both bands
    from one merged multi-band image.
    """
    e = _engine()
    r = e.indices.nbr(nir=nir, swir2=swir2, output=output, chunked=chunked, tile_size=tile_size)
    _pr(r, "NBR")


@index.command("dnbr")
@click.option("--pre-nir", required=True, type=click.Path(exists=True))
@click.option("--pre-swir2", required=True, type=click.Path(exists=True))
@click.option("--post-nir", required=True, type=click.Path(exists=True))
@click.option("--post-swir2", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
def dnbr_cmd(pre_nir, pre_swir2, post_nir, post_swir2, output):
    """dNBR — Differenced Burn Ratio. Range: >0.66 high severity, <-0.25 regrowth."""
    e = _engine()
    r = e.indices.dnbr(
        pre_nir=pre_nir,
        pre_swir2=pre_swir2,
        post_nir=post_nir,
        post_swir2=post_swir2,
        output=output,
    )
    _pr(r, "dNBR")


@index.command("tct")
@click.option("--blue", required=True, type=click.Path(exists=True))
@click.option("--green", required=True, type=click.Path(exists=True))
@click.option("--red", required=True, type=click.Path(exists=True))
@click.option("--nir", required=True, type=click.Path(exists=True))
@click.option("--swir1", required=True, type=click.Path(exists=True))
@click.option("--swir2", required=True, type=click.Path(exists=True))
@click.option(
    "--sensor", default="sentinel2", type=click.Choice(["sentinel2", "landsat8"])
)
@click.option("--output", "-o", default=None)
def tct_cmd(blue, green, red, nir, swir1, swir2, sensor, output):
    """Tasseled Cap Transformation — Brightness, Greenness, Wetness (3-band output)."""
    e = _engine()
    r = e.indices.tct(
        blue=blue,
        green=green,
        red=red,
        nir=nir,
        swir1=swir1,
        swir2=swir2,
        sensor=sensor,
        output=output,
    )
    _pr(r, "TCT")


@index.command("pca")
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@click.option("--components", "-n", default=3, show_default=True, type=int)
@click.option("--output", "-o", default=None)
def pca_cmd(inputs, components, output):
    """PCA — Principal Component Analysis. Reduces dimensionality of multi-band data."""
    if not inputs:
        console.print("[red]Provide at least 2 input band rasters[/]")
        sys.exit(1)
    e = _engine()
    r = e.indices.pca(inputs=list(inputs), n_components=components, output=output)
    _pr(r, f"PCA ({components} components)")


@index.command("texture")
@click.argument("input", type=click.Path(exists=True))
@click.option("--window", "-w", default=5, show_default=True, type=int)
@click.option(
    "--features",
    default="contrast,homogeneity,energy,correlation",
    help="Comma-separated GLCM features.",
)
@click.option("--output", "-o", default=None)
def texture_cmd(input, window, features, output):
    """GLCM Texture — contrast, homogeneity, energy, correlation, dissimilarity, ASM."""
    e = _engine()
    feat_list = [f.strip() for f in features.split(",")]
    r = e.indices.texture(input, window=window, features=feat_list, output=output)
    _pr(r, f"Texture ({feat_list})")


@index.command("lst")
@click.argument("thermal", type=click.Path(exists=True))
@click.option("--emissivity", "-e", default=0.97, show_default=True, type=float)
@click.option(
    "--sensor", default="landsat8", type=click.Choice(["landsat8", "landsat9", "modis"])
)
@click.option("--output", "-o", default=None)
def lst_cmd(thermal, emissivity, sensor, output):
    """Land Surface Temperature from thermal band. Output: Band1=Kelvin, Band2=Celsius."""
    e = _engine()
    r = e.indices.lst(
        thermal=thermal, emissivity=emissivity, sensor=sensor, output=output
    )
    _pr(r, "LST")


@index.command("albedo")
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--sensor", default="sentinel2", type=click.Choice(["sentinel2", "landsat8"])
)
@click.option("--output", "-o", default=None)
def albedo_cmd(inputs, sensor, output):
    """Surface albedo from narrowband to broadband conversion (Liang 2001)."""
    if len(inputs) < 4:
        console.print("[red]Provide at least 4 band rasters[/]")
        sys.exit(1)
    e = _engine()
    r = e.indices.albedo(inputs=list(inputs), sensor=sensor, output=output)
    _pr(r, f"Albedo ({sensor})")


@index.command("band-math")
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--expr", "-e", required=True, help="Python expression. Bands as B[0], B[1], ..."
)
@click.option("--output", "-o", default=None)
@click.option(
    "--chunked", is_flag=True,
    help="Process in memory-safe tiles for large rasters (e.g. drone orthomosaics) that exceed available RAM or the 4GB GeoTIFF limit.",
)
@click.option("--tile-size", default=1024, show_default=True, type=int, help="Tile edge length in pixels when --chunked is set.")
def band_math_cmd(inputs, expr, output, chunked, tile_size):
    """Arbitrary band arithmetic. Example: --expr '(B[1]-B[0])/(B[1]+B[0])'."""
    e = _engine()
    r = e.indices.band_math(inputs=list(inputs), expression=expr, output=output, chunked=chunked, tile_size=tile_size)
    _pr(r, f"band-math ({expr[:40]})")


@index.command("stack")
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
def stack_cmd(inputs, output):
    """Stack single-band rasters into a multi-band GeoTIFF."""
    e = _engine()
    r = e.indices.stack(inputs=list(inputs), output=output)
    _pr(r, f"stack ({len(inputs)} bands)")