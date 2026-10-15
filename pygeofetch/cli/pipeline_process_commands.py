"""CLI for processing pipelines — run YAML or chain steps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group(name="proc-pipeline")
def proc_pipeline() -> None:
    """Processing pipeline — run chained preprocess/index/post steps from YAML."""


@proc_pipeline.command("run")
@click.argument("yaml_file", type=click.Path(exists=True))
@click.option(
    "--input",
    "-i",
    default=None,
    type=click.Path(exists=True),
    help="Starting input file.",
)
@click.option("--output-dir", "-o", default="./pipeline_out", show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Output summary as JSON.")
def proc_run_cmd(yaml_file, input, output_dir, as_json):
    """Run a processing pipeline defined in YAML."""
    from pygeofetch.core.engine import PyGeoFetch
    from pygeofetch.processing.pipeline import ProcessingPipeline

    client = PyGeoFetch(log_level="WARNING")
    pl = ProcessingPipeline.from_yaml(yaml_file, engine=client)
    console.print(f"[cyan]Running pipeline:[/] {pl.name!r} ({len(pl._steps)} steps)")
    result = pl.run(input=input, output_dir=output_dir)
    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2, default=str))
        return
    status = "[green]SUCCESS[/]" if result.success else "[red]FAILED[/]"
    console.print(f"\n{status} in {result.duration_seconds:.1f}s")
    table = Table(header_style="bold blue")
    table.add_column("Step")
    table.add_column("Status")
    table.add_column("Output")
    for s in result.steps:
        status_str = "[green]✓[/]" if s["status"] == "ok" else "[red]✗[/]"
        out = str(s.get("output", ""))[-50:] if s.get("output") else "—"
        table.add_row(s["step"], status_str, out)
    console.print(table)


@proc_pipeline.command("validate")
@click.argument("yaml_file", type=click.Path(exists=True))
def proc_validate_cmd(yaml_file):
    """Validate a processing pipeline YAML without running it."""
    from pygeofetch.processing.pipeline import ProcessingPipeline

    try:
        pl = ProcessingPipeline.from_yaml(yaml_file)
        console.print(f"[green]✓[/] Pipeline {pl.name!r} is valid")
        console.print(f"  Steps: {len(pl._steps)}")
        for i, s in enumerate(pl._steps, 1):
            console.print(f"  {i}. {s.step_type}  {s.config}")
    except Exception as exc:
        console.print(f"[red]✗[/] Invalid pipeline: {exc}")
        sys.exit(1)


@proc_pipeline.command("template")
@click.argument(
    "template_name",
    type=click.Choice(
        [
            "ndvi",
            "change_detection",
            "flood_map",
            "urban_mapping",
            "sar_analysis",
            "land_cover",
        ]
    ),
)
@click.option("--output", "-o", default=None)
def proc_template_cmd(template_name, output):
    """Generate a starter YAML pipeline template."""
    import yaml as _yaml

    templates = {
        "ndvi": {
            "name": "ndvi-monitoring",
            "description": "Atmospheric correction → cloud mask → NDVI → COG",
            "steps": [
                {"atmos": {"method": "dos1"}},
                {"cloud_mask": {"method": "scl", "scl_band": "SCL.tif"}},
                {"clip": {"bbox": "-74.1,40.6,-73.7,40.9"}},
                {"ndvi": {"red": "B04.tif", "nir": "B08.tif"}},
                {"cog": {"compress": "deflate"}},
            ],
        },
        "change_detection": {
            "name": "change-detection",
            "description": "Compare pre/post scenes for land cover change",
            "steps": [
                {"atmos": {"method": "dos1"}},
                {"clip": {"bbox": "minx,miny,maxx,maxy"}},
                {"reproject": {"crs": "EPSG:4326"}},
                {
                    "dnbr": {
                        "pre_nir": "pre_B08.tif",
                        "pre_swir2": "pre_B12.tif",
                        "post_nir": "post_B08.tif",
                        "post_swir2": "post_B12.tif",
                    }
                },
                {"vectorize": {"threshold": 0.27, "format": "geojson"}},
                {"smooth": {"tolerance": 0.5}},
            ],
        },
        "flood_map": {
            "name": "sar-flood-mapping",
            "description": "Sentinel-1 SAR → despeckle → calibrate → flood map",
            "steps": [
                {"despeckle": {"filter": "enhanced_lee", "window": 7}},
                {"calibrate": {"output_type": "sigma0", "in_db": True}},
                {"flood_map": {"threshold": -15.0}},
                {"vectorize": {"threshold": 0.5, "format": "geojson"}},
                {"smooth": {"tolerance": 0.5}},
            ],
        },
        "urban_mapping": {
            "name": "urban-built-up-mapping",
            "description": "NDVI + NDBI → built-up area delineation",
            "steps": [
                {"atmos": {"method": "dos1"}},
                {"clip": {"bbox": "minx,miny,maxx,maxy"}},
                {"ndvi": {"red": "B04.tif", "nir": "B08.tif"}},
                {"ndbi": {"nir": "B08.tif", "swir1": "B11.tif"}},
                {"band_math": {"expression": "B[0] - B[1]"}},  # NDBI - NDVI
                {"vectorize": {"threshold": 0.0, "format": "geojson"}},
                {"regularize": {}},
            ],
        },
        "sar_analysis": {
            "name": "sar-coherence-change",
            "description": "SAR coherence-based change detection",
            "steps": [
                {"despeckle": {"filter": "lee"}},
                {"calibrate": {"output_type": "gamma0"}},
                {"coherence": {"window": 7}},
                {"vectorize": {"threshold": 0.3}},
            ],
        },
        "land_cover": {
            "name": "land-cover-preprocessing",
            "description": "Full preprocessing chain for land cover classification",
            "steps": [
                {"atmos": {"method": "sen2cor"}},
                {"cloud_mask": {"method": "fmask"}},
                {"clip": {"bbox": "minx,miny,maxx,maxy"}},
                {"reproject": {"crs": "EPSG:4326"}},
                {"resample": {"resolution": 10}},
                {"composite": {"method": "median"}},
                {"cog": {"compress": "deflate"}},
            ],
        },
    }

    template = templates[template_name]
    yaml_str = _yaml.dump(template, default_flow_style=False, sort_keys=False)
    out_path = Path(output) if output else Path(f"{template_name}_pipeline.yaml")
    out_path.write_text(yaml_str)
    console.print(f"[green]✓[/] Template written → {out_path}")
    console.print("\n[dim]Edit the YAML then run:[/]")
    console.print(f"  PyGeoFetch proc-pipeline validate {out_path}")
    console.print(f"  PyGeoFetch proc-pipeline run {out_path} --input scene.tif")
