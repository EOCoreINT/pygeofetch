"""Download CLI commands for PyGeoFetch — full flag set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

console = Console()


@click.group()
def download() -> None:
    """Download satellite data products."""


@download.command(name="run")
@click.option(
    "--from-search",
    "-f",
    default=None,
    type=click.Path(exists=True),
    help="GeoJSON results file from `search run --output`.",
)
@click.option(
    "--scene-ids", default=None, help="Comma-separated scene IDs to download."
)
@click.option(
    "--output",
    "-o",
    default="./pygeofetch_data",
    show_default=True,
    help="Output directory.",
)
@click.option(
    "--parallel", "-p", default=2, show_default=True, help="Parallel download workers."
)
@click.option(
    "--retry", "-r", default=3, show_default=True, help="Retry attempts per file."
)
@click.option(
    "--retry-delay",
    default=5.0,
    show_default=True,
    help="Base retry delay seconds (exponential backoff).",
)
@click.option(
    "--verify-checksum",
    is_flag=True,
    default=False,
    help="SHA256 checksum verification after download.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=True,
    help="Resume interrupted downloads (default on).",
)
@click.option(
    "--bandwidth-limit",
    "bandwidth_limit",
    default=None,
    help="Max bandwidth e.g. 10MB, 5MB (0=unlimited).",
)
@click.option(
    "--priority",
    default="normal",
    type=click.Choice(["high", "normal", "low"]),
    show_default=True,
    help="Download priority.",
)
@click.option(
    "--notify",
    default=None,
    multiple=True,
    help="Notifications: webhook:URL or email:ADDRESS (repeatable).",
)
@click.option(
    "--post-process",
    default=None,
    help='Post-process chain e.g. "unzip,reproject:EPSG:4326,compress:lzw".',
)
@click.option(
    "--on-failure",
    default="skip",
    type=click.Choice(["skip", "abort", "retry"]),
    show_default=True,
    help="How to handle individual file failures.",
)
@click.option(
    "--max-items",
    "-n",
    default=None,
    type=int,
    help="Limit number of items to download.",
)
@click.option(
    "--overwrite", is_flag=True, default=False, help="Overwrite existing files."
)
@click.option(
    "--bands",
    default=None,
    help='Comma-separated band names to download e.g. "B02,B03,B04" (default: all data assets).',
)
@click.option("--json", "as_json", is_flag=True, help="Output results summary as JSON.")
@click.option(
    "--validate-optical",
    is_flag=True,
    default=False,
    help=(
        "Run optical preflight validation as a final gate before "
        "downloading (AOI coverage, cloud cover, required bands, "
        "processing level, temporal bounds). Items that fail are "
        "never attempted -- a failed result explaining why is "
        "returned in their place, same position, so output length "
        "always matches input."
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
def download_run(
    from_search,
    scene_ids,
    output,
    parallel,
    retry,
    retry_delay,
    verify_checksum,
    resume,
    bandwidth_limit,
    priority,
    notify,
    post_process,
    on_failure,
    max_items,
    overwrite,
    bands,
    as_json,
    validate_optical,
    optical_max_cloud_cover,
    optical_min_coverage,
    optical_required_bands,
) -> None:
    """
    Download satellite data products from a search results file or scene IDs.

    \b
    Examples:
      # Basic download
      pygeofetch download run \\
          --from-search results.geojson --output ./data/

      # Full-featured download
      pygeofetch download run \\
          --from-search results.geojson --output ./data/ \\
          --parallel 4 --verify-checksum --resume \\
          --bandwidth-limit 10MB --priority high \\
          --post-process "unzip,reproject:EPSG:4326,compress:lzw" \\
          --notify webhook:https://hooks.slack.com/T01/B01/XYZ \\
          --notify email:user@example.com \\
          --on-failure skip

      # With optical preflight validation as a final gate before downloading
      pygeofetch download run \\
          --from-search results.geojson --output ./data/ \\
          --validate-optical --optical-max-cloud-cover 10
    """
    from pygeofetch.core.engine import PyGeoFetch
    from pygeofetch.core.searcher import FederatedSearcher
    from pygeofetch.models.download_task import DownloadOptions, PostProcessAction

    if not from_search and not scene_ids:
        console.print("[red]Provide --from-search FILE or --scene-ids IDs[/]")
        sys.exit(1)

    # Parse post-process actions
    pp_actions = []
    if post_process:
        for token in post_process.split(","):
            token = token.strip()
            if ":" in token:
                action_name, _, param_val = token.partition(":")
                pp_actions.append(
                    PostProcessAction(
                        action=action_name.strip(), params={"value": param_val.strip()}
                    )
                )
            else:
                pp_actions.append(PostProcessAction(action=token))

    # Parse bandwidth limit
    bw_mbps = 0.0
    if bandwidth_limit:
        multipliers = {"MB": 1.0, "GB": 1024.0, "KB": 1 / 1024}
        for unit, mult in multipliers.items():
            if bandwidth_limit.upper().endswith(unit):
                try:
                    bw_mbps = float(bandwidth_limit[: -len(unit)]) * mult
                    break
                except ValueError:
                    pass

    # Parse notify
    notify_webhook = None
    notify_email = None
    for n in notify:
        if n.startswith("webhook:"):
            notify_webhook = n[8:]
        elif n.startswith("email:"):
            notify_email = n[6:]

    priority_map = {"high": 9, "normal": 5, "low": 1}

    parsed_bands = [b.strip() for b in bands.split(",")] if bands else []

    options = DownloadOptions(
        parallel=parallel,
        retry_attempts=retry,
        retry_delay_seconds=retry_delay,
        verify_checksum=verify_checksum,
        resume=resume,
        bandwidth_limit_mbps=bw_mbps,
        priority=priority_map.get(priority, 5),
        notify_webhook=notify_webhook,
        notify_email=notify_email,
        post_process=pp_actions,
        on_failure=on_failure,
        overwrite=overwrite,
        bands=parsed_bands,
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

    # Load data
    if from_search:
        data_list = FederatedSearcher.load_results(Path(from_search))
    else:
        console.print("[yellow]--scene-ids not yet supported; use --from-search[/]")
        sys.exit(1)

    if not data_list:
        console.print("[yellow]No items to download.[/]")
        return

    if max_items:
        data_list = data_list[:max_items]

    console.print(
        f"[cyan]Downloading {len(data_list)} item(s)[/] → [bold]{output}[/]\n"
        f"  parallel={parallel}, retry={retry}, checksum={verify_checksum}, "
        f"resume={resume}, on_failure={on_failure}"
    )
    if pp_actions:
        console.print(f"  post-process: {' → '.join(a.action for a in pp_actions)}")

    results = []
    total_items = len(data_list)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(
            f"Downloading 0/{total_items}...",
            total=total_items,
        )

        def on_item_done(completed: int, total: int, result) -> None:
            status = "✓" if result.success else "✗"
            mb = (
                result.bytes_downloaded / (1024 * 1024)
                if result.bytes_downloaded
                else 0
            )
            label = (
                f"{status} {result.data_id[:28]}  {mb:.0f} MB  ({completed}/{total})"
                if result.success
                else f"{status} {result.data_id[:28]}  failed  ({completed}/{total})"
            )
            progress.update(task, advance=1, description=label)

        # Use sb.download() with the already-sliced data_list (respects --max-items)
        results = sb.download(
            data_list, Path(output), options, item_done_callback=on_item_done,
            validate_optical=validate_optical,
        )

    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    total_mb = sum(
        r.bytes_downloaded / (1024 * 1024)
        for r in results
        if r.success and r.bytes_downloaded
    )

    # Fire notifications
    if notify_webhook and succeeded:
        _notify_webhook(notify_webhook, len(succeeded), len(failed), total_mb)

    if as_json:
        click.echo(
            json.dumps(
                {
                    "total": len(results),
                    "succeeded": len(succeeded),
                    "failed": len(failed),
                    "total_mb": round(total_mb, 2),
                    "failures": [{"id": r.data_id, "error": r.error} for r in failed],
                },
                indent=2,
            )
        )
        return

    console.print("\n[bold]Download Summary[/]")
    console.print(f"  [green]Succeeded:[/] {len(succeeded)}")
    console.print(f"  [red]Failed:[/]    {len(failed)}")
    console.print(f"  [cyan]Total size:[/] {total_mb:.1f} MB")

    if failed:
        console.print("\n[red]Failures:[/]")
        for r in failed:
            console.print(f"  - {r.data_id}: {r.error}")
        if on_failure == "abort":
            sys.exit(1)


@download.command(name="status")
@click.argument("output_dir", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True)
def download_status_cmd(output_dir: str, as_json: bool) -> None:
    """Show downloaded files in OUTPUT_DIR."""
    dest = Path(output_dir)
    files = sorted(f for f in dest.rglob("*") if f.is_file())
    total = sum(f.stat().st_size for f in files)

    if as_json:
        data = [
            {"file": str(f.relative_to(dest)), "size_bytes": f.stat().st_size}
            for f in files
        ]
        click.echo(
            json.dumps(
                {"files": data, "total_bytes": total, "count": len(files)}, indent=2
            )
        )
        return

    from rich.table import Table

    table = Table(title=f"Files in {output_dir}", header_style="bold blue")
    table.add_column("File", style="cyan")
    table.add_column("Size", justify="right")
    for f in files:
        mb = f.stat().st_size / (1024 * 1024)
        table.add_row(str(f.relative_to(dest)), f"{mb:.2f} MB")
    console.print(table)
    console.print(f"\n  Total: {len(files)} files, {total / (1024 * 1024):.1f} MB")


@download.command(name="history")
@click.option("--limit", default=20, show_default=True)
@click.option(
    "--status",
    "filter_status",
    default=None,
    type=click.Choice(["success", "failed", "active"]),
)
@click.option("--json", "as_json", is_flag=True)
def download_history(limit: int, filter_status: str, as_json: bool) -> None:
    """Show download history."""
    import json as _json

    from pygeofetch.config.settings import get_config_dir

    history_file = get_config_dir() / "download_history.jsonl"
    if not history_file.exists():
        console.print("[dim]No download history recorded.[/]")
        return
    lines = history_file.read_text().strip().splitlines()[-limit:]
    runs = [_json.loads(ln) for ln in lines if ln.strip()]
    if filter_status:
        runs = [r for r in runs if r.get("status") == filter_status]
    if as_json:
        click.echo(_json.dumps(runs, indent=2, default=str))
        return
    for r in runs:
        console.print(
            f"  {r.get('id', '?')} | {r.get('status', '?')} | {r.get('bytes_downloaded', 0) // 1024} KB"  # noqa: E501
        )


def _notify_webhook(url: str, succeeded: int, failed: int, total_mb: float) -> None:
    """Send a completion notification to a webhook URL."""
    try:
        import httpx

        payload = {
            "text": f"PyGeoFetch download complete: {succeeded} succeeded, {failed} failed, {total_mb:.1f} MB"  # noqa: E501
        }
        httpx.post(url, json=payload, timeout=10)
    except Exception as exc:
        console.print(f"[yellow]Webhook notification failed: {exc}[/]")