"""Search CLI commands for PyGeoFetch — full flag set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def search() -> None:
    """Search satellite data across one or more providers."""


@search.command(name="run")
@click.option("--bbox", "-b", default=None, help='Bounding box "minx,miny,maxx,maxy".')
@click.option(
    "--geometry-file",
    "-g",
    default=None,
    type=click.Path(exists=True),
    help="GeoJSON file with search AOI (alternative to --bbox).",
)
@click.option("--start-date", "-s", default=None, help="Start date YYYY-MM-DD.")
@click.option("--end-date", "-e", default=None, help="End date YYYY-MM-DD.")
@click.option(
    "--cloud-cover",
    "-c",
    default="0-100",
    show_default=True,
    help='Cloud cover range "min-max" e.g. 0-20.',
)
@click.option(
    "--resolution",
    default=None,
    help='Resolution range in metres "min-max" e.g. 10-30.',
)
@click.option("--processing-level", default=None, help="Processing level e.g. L2A.")
@click.option(
    "--providers",
    "-p",
    default=None,
    help="Comma-separated provider IDs e.g. usgs,copernicus,planetary_computer.",
)
@click.option(
    "--satellites",
    default=None,
    help="Comma-separated satellite names e.g. Sentinel-2,Landsat-8.",
)
@click.option(
    "--product-type",
    default=None,
    help=(
        "SAR product type e.g. GRD, SLC, GRD-COG. Currently only "
        "meaningfully filters Copernicus (OData-based, not STAC) — "
        "STAC providers like planetary_computer/element84 offer only "
        "pre-processed Sentinel-1 RTC via their collection mapping, not "
        "raw GRD/SLC, so this has no effect there regardless of value."
    ),
)
@click.option(
    "--polarisation",
    default=None,
    help='SAR polarisation e.g. VV, VH, HH, HV. Same Copernicus-only caveat as --product-type.',
)
@click.option("--max-results", "-n", default=100, show_default=True)
@click.option(
    "--sort-by",
    default="datetime",
    type=click.Choice(["datetime", "cloud_cover", "score", "satellite"]),
    show_default=True,
    help="Sort field.",
)
@click.option(
    "--sort-order",
    default="desc",
    type=click.Choice(["asc", "desc"]),
    show_default=True,
)
@click.option("--cql2", default=None, help="CQL2 filter expression.")
@click.option("--output", "-o", default=None, help="Save results to this file.")
@click.option(
    "--format",
    "fmt",
    default="table",
    type=click.Choice(["table", "json", "stac", "geojson", "geoparquet", "csv", "ids"]),
    show_default=True,
    help="Output format.",
)
@click.option(
    "--on-provider-failure",
    default="skip",
    type=click.Choice(["skip", "abort", "retry"]),
    show_default=True,
    help="How to handle provider failures.",
)
@click.option("--timeout", default=60, show_default=True, help="Per-provider timeout in seconds.")
@click.option("--no-cache", is_flag=True, default=False, help="Bypass result cache.")
@click.option(
    "--validate-optical",
    is_flag=True,
    default=False,
    help=(
        "Run optical preflight validation (AOI coverage, cloud cover, "
        "required bands, processing level, temporal bounds) on results "
        "before returning them. Not appropriate for SAR/InSAR searches "
        "-- the default required bands are optical band names."
    ),
)
@click.option(
    "--optical-max-cloud-cover",
    default=None,
    type=float,
    help="Override the max cloud cover threshold used by --validate-optical (default 20.0).",
)
@click.option(
    "--optical-min-coverage",
    default=None,
    type=float,
    help="Override the min AOI coverage ratio used by --validate-optical, 0-1 (default 0.8).",
)
@click.option(
    "--optical-required-bands",
    default=None,
    help="Comma-separated required bands for --validate-optical (default B02,B03,B04,B08,SCL).",
)
def search_run(
    bbox,
    geometry_file,
    start_date,
    end_date,
    cloud_cover,
    resolution,
    processing_level,
    providers,
    satellites,
    product_type,
    polarisation,
    max_results,
    sort_by,
    sort_order,
    cql2,
    output,
    fmt,
    on_provider_failure,
    timeout,
    no_cache,
    validate_optical,
    optical_max_cloud_cover,
    optical_min_coverage,
    optical_required_bands,
) -> None:
    """
    Search for satellite imagery across one or more providers.

    \b
    Examples:
      # Basic search with bbox and cloud cover
      pygeofetch search run --bbox "-74,40,-73,41" --cloud-cover 0-20

      # Multiple providers, specific satellite, save GeoJSON
      pygeofetch search run \\
          --bbox "-74,40,-73,41" --start-date 2024-01-01 \\
          --providers copernicus,aws_earth,planetary_computer \\
          --satellites Sentinel-2 --cloud-cover 0-10 \\
          --output results.geojson --format geojson

      # AOI from file, CQL2 filter, GeoParquet output
      pygeofetch search run \\
          --geometry-file area.geojson \\
          --cql2 "eo:cloud_cover < 5 AND platform = 'sentinel-2b'" \\
          --format geoparquet --output results.parquet

      # With optical preflight validation (skip cloudy/off-target/wrong-level scenes)
      pygeofetch search run \\
          --bbox "-74,40,-73,41" --cloud-cover 0-100 \\
          --validate-optical --optical-max-cloud-cover 10 --optical-min-coverage 0.9
    """
    from pygeofetch.core.engine import PyGeoFetch
    from pygeofetch.models.search_query import BoundingBox, SearchQuery

    # Parse cloud cover
    try:
        parts = cloud_cover.split("-")
        cloud_min = float(parts[0])
        cloud_max = float(parts[1]) if len(parts) > 1 else 100.0
    except Exception:
        console.print(
            f"[red]Invalid --cloud-cover: {cloud_cover!r} (expected min-max)[/]"
        )
        sys.exit(1)

    # Parse resolution
    res_min, res_max = None, None
    if resolution:
        try:
            parts = resolution.split("-")
            res_min = float(parts[0])
            res_max = float(parts[1]) if len(parts) > 1 else None
        except Exception:
            console.print(f"[red]Invalid --resolution: {resolution!r}[/]")
            sys.exit(1)

    # Build bbox
    bbox_obj = None
    geometry_geojson = None
    if geometry_file:
        with open(geometry_file) as f:
            gj = json.load(f)
        geom = (
            gj
            if gj.get("type") in ("Polygon", "MultiPolygon")
            else gj.get("geometry") or (gj.get("features") or [{}])[0].get("geometry")
        )
        geometry_geojson = geom
        # Derive bbox from geometry
        coords = []
        if geom and geom.get("type") == "Polygon":
            coords = [pt for ring in geom["coordinates"] for pt in ring]
        elif geom and geom.get("type") == "MultiPolygon":
            coords = [
                pt for poly in geom["coordinates"] for ring in poly for pt in ring
            ]
        if coords:
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            bbox_obj = BoundingBox(
                min_lon=min(lons),
                min_lat=min(lats),
                max_lon=max(lons),
                max_lat=max(lats),
            )
    elif bbox:
        try:
            bbox_obj = BoundingBox.from_string(bbox)
        except Exception as exc:
            console.print(f"[red]Invalid --bbox: {exc}[/]")
            sys.exit(1)

    provider_list = [p.strip() for p in providers.split(",")] if providers else None
    sat_list = [s.strip() for s in satellites.split(",")] if satellites else []
    pl_list = [processing_level] if processing_level else []

    query = SearchQuery(
        bbox=bbox_obj,
        # Was "geometry_geojson=geometry_geojson" — a SEPARATE, unused
        # field that nothing in the codebase actually reads. Every
        # provider (and the model's own to_stac_params()/
        # has_spatial_filter() methods) reads query.geometry, not
        # query.geometry_geojson. This meant --geometry-file silently
        # never applied real polygon-based spatial filtering at all —
        # it fell back to the bbox approximation computed just above
        # (still functional, but loses the actual polygon precision
        # that geometry-based filtering exists to provide).
        geometry=geometry_geojson,
        start_date=start_date,
        end_date=end_date,
        cloud_cover_min=cloud_min,
        cloud_cover_max=cloud_max,
        resolution_min_m=res_min,
        resolution_max_m=res_max,
        processing_levels=pl_list,
        satellites=sat_list,
        product_type=product_type,
        polarisation=polarisation,
        max_results=max_results,
        providers=provider_list or [],
        sort_by=sort_by,
        sort_ascending=(sort_order == "asc"),
        cql2_filter=cql2,
        on_provider_failure=on_provider_failure,
        timeout_seconds=timeout,
    )

    sb = PyGeoFetch()

    if validate_optical and (optical_max_cloud_cover, optical_min_coverage, optical_required_bands) != (None, None, None):
        from pygeofetch.validation import OpticalValidationConfig

        overrides = {}
        if optical_max_cloud_cover is not None:
            overrides["max_cloud_cover_pct"] = optical_max_cloud_cover
        if optical_min_coverage is not None:
            overrides["min_coverage_ratio"] = optical_min_coverage
        if optical_required_bands is not None:
            overrides["required_bands"] = [b.strip() for b in optical_required_bands.split(",")]
        sb.optical_validation_config = OpticalValidationConfig(**overrides)

    with console.status(
        f"[cyan]Searching {len(provider_list or [])} provider(s)...[/]"
    ):
        results = sb.search(
            query, providers=provider_list, use_cache=not no_cache,
            validate_optical=validate_optical,
        )

    if not results:
        console.print("[yellow]No results found.[/]")
        return

    # Save if requested
    if output:
        out_path = Path(output)
        if fmt == "geoparquet":
            _save_geoparquet(results, out_path)
        elif fmt == "csv":
            _save_csv(results, out_path)
        else:
            sb.searcher.save_results(results, out_path)
        console.print(f"[green]Saved {len(results)} results → {output}[/]\n")

    # Display
    if fmt == "table":
        _display_table(results)
    elif fmt == "json":
        click.echo(
            json.dumps([_item_to_dict(r) for r in results], indent=2, default=str)
        )
    elif fmt in ("stac", "geojson"):
        fc = sb.searcher.to_geojson(results)
        click.echo(json.dumps(fc, indent=2, default=str))
    elif fmt == "geoparquet":
        if not output:
            console.print("[yellow]--output required for geoparquet format[/]")
    elif fmt == "csv":
        if not output:
            import csv as csv_mod
            import io

            buf = io.StringIO()
            writer = csv_mod.DictWriter(
                buf,
                fieldnames=[
                    "id",
                    "provider",
                    "satellite",
                    "datetime",
                    "cloud_cover",
                    "score",
                    "bbox",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow(_item_to_dict(r))
            click.echo(buf.getvalue())
    elif fmt == "ids":
        for r in results:
            click.echo(r.id)


def _display_table(results) -> None:
    table = Table(title=f"{len(results)} Results", header_style="bold blue")
    table.add_column("ID", style="cyan", max_width=30)
    table.add_column("Provider", style="dim")
    table.add_column("Date")
    table.add_column("Cloud%", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Satellite")
    for r in results[:50]:
        date_str = str(r.properties.get("datetime", ""))[:10] if r.properties else "—"
        cloud = f"{r.cloud_cover:.0f}" if r.cloud_cover is not None else "—"
        score = f"{r.score:.2f}" if r.score else "—"
        table.add_row(r.id[:30], r.provider, date_str, cloud, score, r.satellite or "—")
    if len(results) > 50:
        table.add_row("...", "...", "...", "...", "...", f"(+{len(results) - 50} more)")
    console.print(table)


def _item_to_dict(r) -> dict:
    return {
        "id": r.id,
        "provider": r.provider,
        "satellite": r.satellite,
        "datetime": str(r.properties.get("datetime", "") if r.properties else ""),
        "cloud_cover": r.cloud_cover,
        "score": r.score,
        "bbox": list(r.bbox) if r.bbox else None,
    }


def _save_geoparquet(results, path: Path) -> None:
    try:
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import box as shapely_box

        records = [_item_to_dict(r) for r in results]
        df = pd.DataFrame(records)
        geoms = [shapely_box(*r["bbox"]) if r["bbox"] else None for r in records]
        gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
        gdf.to_parquet(path)
        console.print(f"[green]Saved GeoParquet: {path}[/]")
    except ImportError:
        console.print("[yellow]geopandas not installed — saving as GeoJSON instead.[/]")
        import json as _json

        path.with_suffix(".geojson").write_text(
            _json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [_item_to_dict(r) for r in results],
                },
                indent=2,
                default=str,
            )
        )


def _save_csv(results, path: Path) -> None:
    import csv as csv_mod

    with open(path, "w", newline="") as f:
        writer = csv_mod.DictWriter(
            f,
            fieldnames=[
                "id",
                "provider",
                "satellite",
                "datetime",
                "cloud_cover",
                "score",
                "bbox",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(_item_to_dict(r))