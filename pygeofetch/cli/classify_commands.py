"""CLI commands for pixel classification."""

from __future__ import annotations

import sys

import click
from rich.console import Console

from pygeofetch.cli.index_commands import clip_options, _parse_bbox

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
        console.print(
            f"[green]✓[/] {name} → {result.output_path}"
            f" ({size_mb:.1f} MB, {result.duration_seconds:.2f}s)"
        )
        meta = result.metadata or {}
        for k, v in meta.items():
            if k != "classification_report":
                console.print(f"  {k}: {v}")
    else:
        console.print(f"[red]✗[/] {name} failed: {result.error}")
        sys.exit(1)


@click.group()
def classify() -> None:
    """Pixel classification — unsupervised K-means and supervised Random Forest/SVM."""


@classify.command("kmeans")
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@click.option("--clusters", "-k", default=5, show_default=True, type=int, help="Number of clusters.")
@click.option("--output", "-o", default=None)
@click.option("--random-state", default=42, show_default=True, type=int)
@clip_options
def kmeans_cmd(inputs, clusters, output, random_state, bbox, geometry, geometry_crs):
    """Unsupervised K-means clustering. No training data needed.

    Example: pygeofetch classify kmeans B02.tif B03.tif B04.tif B08.tif -k 5
    """
    if len(inputs) < 1:
        console.print("[red]Provide at least 1 input band raster[/]")
        sys.exit(1)
    e = _engine()
    r = e.classify.kmeans(
        inputs=list(inputs), n_clusters=clusters, output=output, random_state=random_state,
        bbox=bbox, geometry=geometry, geometry_crs=geometry_crs,
    )
    _pr(r, f"K-means ({clusters} clusters)")


@classify.command("train")
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@click.option("--training-data", required=True, type=click.Path(exists=True), help="Vector file (GeoJSON/Shapefile) with labeled training geometries.")
@click.option("--class-field", required=True, help="Attribute column holding the real class label.")
@click.option("--output-model", "-o", required=True, help="Where to save the trained model.")
@click.option("--model-type", default="random_forest", type=click.Choice(["random_forest", "svm"]), show_default=True)
@click.option("--n-estimators", default=100, show_default=True, type=int, help="Random Forest tree count.")
@click.option("--test-size", default=0.2, show_default=True, type=float, help="Fraction of samples held out for real accuracy reporting.")
def train_cmd(inputs, training_data, class_field, output_model, model_type, n_estimators, test_size):
    """Train a supervised classifier from labeled vector training samples.

    Example: pygeofetch classify train B02.tif B03.tif B04.tif B08.tif
              --training-data samples.geojson --class-field land_cover -o model.joblib
    """
    if len(inputs) < 1:
        console.print("[red]Provide at least 1 input band raster[/]")
        sys.exit(1)
    e = _engine()
    r = e.classify.train(
        inputs=list(inputs), training_data=training_data, class_field=class_field,
        output_model=output_model, model_type=model_type, n_estimators=n_estimators,
        test_size=test_size,
    )
    _pr(r, f"train ({model_type})")


@classify.command("predict")
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="A model trained via 'classify train'.")
@click.option("--output", "-o", default=None)
@clip_options
def predict_cmd(inputs, model, output, bbox, geometry, geometry_crs):
    """Apply a trained model to a real raster stack, producing a classification map.

    Example: pygeofetch classify predict B02.tif B03.tif B04.tif B08.tif -m model.joblib
    """
    if len(inputs) < 1:
        console.print("[red]Provide at least 1 input band raster[/]")
        sys.exit(1)
    e = _engine()
    r = e.classify.predict(
        inputs=list(inputs), model=model, output=output,
        bbox=bbox, geometry=geometry, geometry_crs=geometry_crs,
    )
    _pr(r, "predict")
