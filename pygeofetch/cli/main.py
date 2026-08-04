"""
PyGeoFetch CLI v1.9.7 main entry point.

Full command reference:
  pygeofetch auth       — credential management
  pygeofetch search     — federated satellite search
  pygeofetch download   — parallel download engine
  pygeofetch providers  — provider listing and info
  pygeofetch pipeline   — YAML pipeline orchestration
  pygeofetch config     — configuration management
  pygeofetch cache      — cache management
  pygeofetch status     — system status dashboard
  pygeofetch doctor     — installation diagnostics
  pygeofetch version    — version information
"""

from __future__ import annotations

import sys

import click
from rich.console import Console

from pygeofetch import __version__
from pygeofetch.cli.auth_commands import auth
from pygeofetch.cli.config_commands import config
from pygeofetch.cli.download_commands import download
from pygeofetch.cli.index_commands import index
from pygeofetch.cli.pipeline_process_commands import proc_pipeline
from pygeofetch.cli.postprocess_commands import post
from pygeofetch.cli.preprocess_commands import preprocess
from pygeofetch.cli.sar_commands import sar
from pygeofetch.cli.search_commands import search
from pygeofetch.utils.logging_setup import setup_logging

console = Console()


@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100}
)
@click.version_option(__version__, prog_name="pygeofetch")
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    envvar="PYGEOFETCH_LOG_LEVEL",
    show_default=True,
    help="Logging verbosity.",
)
@click.option(
    "--log-file", default=None, type=click.Path(), help="Write logs to this file."
)
@click.option(
    "--log-format",
    default="console",
    type=click.Choice(["console", "json"]),
    show_default=True,
    help="Log output format.",
)
@click.option(
    "--config",
    "config_file",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom config YAML file.",
)
@click.pass_context
def cli(
    ctx: click.Context, log_level: str, log_file: str, log_format: str, config_file: str
) -> None:
    """
    \b
    PyGeoFetch v1.9.7 — Universal Satellite Data Pipeline
    ===========================================================
    Unified access to 22+ satellite data providers.

    \b
    Quick start:
      pygeofetch auth add usgs --username USER --password PASS
      pygeofetch search run --bbox "-74,40,-73,41" --providers usgs,aws_earth
      pygeofetch download run --from-search results.geojson --output ./data/

    Run any command with --help for details.
    """
    from pathlib import Path

    setup_logging(
        level=log_level,
        log_file=str(Path(log_file)) if log_file else None,
    )
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level
    ctx.obj["config_file"] = config_file


cli.add_command(auth)
cli.add_command(search)
cli.add_command(download)
cli.add_command(config)
cli.add_command(preprocess)
cli.add_command(index)
cli.add_command(post)
cli.add_command(sar)
cli.add_command(proc_pipeline)


# ---------------------------------------------------------------------------
# STATUS
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def status(as_json: bool) -> None:
    """
    Show full system status: providers, cache, downloads, pipelines, version.

    Displays authenticated providers with quota, cache statistics, active
    downloads, scheduled pipelines, and system information.
    """
    import json as _json
    import platform

    from rich.table import Table

    from pygeofetch.core.cache_manager import CacheManager
    from pygeofetch.core.engine import PyGeoFetch

    sb = PyGeoFetch()
    info = sb.status()
    cache_stats = CacheManager().stats()

    if as_json:
        out = {
            "version": info["version"],
            "python": platform.python_version(),
            "platform": platform.system(),
            "providers_authenticated": info["providers_authenticated"],
            "providers_free": info["providers_free"],
            "cache": cache_stats,
        }
        click.echo(_json.dumps(out, indent=2, default=str))
        return

    console.print(f"\n[bold cyan]PyGeoFetch[/] v{info['version']}")
    console.print(f"Python {platform.python_version()} on {platform.system()}\n")

    # Provider table
    from pygeofetch.providers import list_provider_info

    all_info = {p["id"]: p for p in list_provider_info()}
    authed = set(info["providers_authenticated"])
    free = set(info["providers_free"])

    table = Table(title="Providers", header_style="bold blue", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Status", justify="center")
    table.add_column("SAR", justify="center")
    table.add_column("<1m", justify="center")
    table.add_column("STAC", justify="center")

    for pid, pinfo in sorted(all_info.items()):
        if pid in authed:
            status_str = "[green]✓ authenticated[/]"
        elif pid in free:
            status_str = "[blue]🌐 open[/]"
        else:
            status_str = "[dim]✗ not configured[/]"
        table.add_row(
            pid,
            pinfo.get("display_name", pid),
            status_str,
            "[green]✓[/]" if pinfo.get("supports_sar") else "—",
            "[green]✓[/]" if pinfo.get("supports_sub_meter") else "—",
            "[green]✓[/]" if pinfo.get("stac") else "—",
        )
    console.print(table)

    # Cache stats
    console.print(
        f"\n[bold]Cache[/]  {cache_stats['valid']} valid entries "
        f"/ {cache_stats['expired']} expired "
        f"/ {cache_stats['size_bytes'] // 1024} KB "
        f"at {cache_stats['cache_dir']}"
    )


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def version(as_json: bool) -> None:
    """Show version information."""
    import json as _json
    import platform

    data = {
        "version": __version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
    }
    if as_json:
        click.echo(_json.dumps(data))
    else:
        console.print(
            f"[bold cyan]PyGeoFetch[/] v{data['version']} | "
            f"Python {data['python']} | {data['platform']}"
        )


# ---------------------------------------------------------------------------
# DOCTOR
# ---------------------------------------------------------------------------


@cli.command()
def doctor() -> None:
    """
    Diagnose the PyGeoFetch installation and connectivity.

    Checks: Python version, required packages, config directory, keyring,
    network connectivity to provider endpoints.
    """
    import importlib
    import sys

    ok = "[green]✓[/]"
    fail = "[red]✗[/]"
    warn = "[yellow]⚠[/]"

    console.print(f"\n[bold]PyGeoFetch Doctor[/] v{__version__}\n")

    # Python version
    vi = sys.version_info
    py_ok = vi >= (3, 9)
    console.print(
        f"  {ok if py_ok else fail} Python {vi.major}.{vi.minor}.{vi.micro}"
        + ("" if py_ok else " — requires 3.9+")
    )

    # Required packages
    required = ["httpx", "pydantic", "click", "rich", "yaml", "cryptography", "keyring"]
    for pkg in required:
        mod = "yaml" if pkg == "yaml" else pkg
        try:
            importlib.import_module(mod)
            console.print(f"  {ok} {pkg}")
        except ImportError:
            console.print(f"  {fail} {pkg} — not installed")

    # Optional packages
    optional = [
        ("boto3", "AWS S3 direct access"),
        ("rasterio", "raster post-processing"),
        ("geopandas", "GeoParquet output"),
        ("croniter", "cron scheduling"),
    ]
    for pkg, purpose in optional:
        try:
            importlib.import_module(pkg)
            console.print(f"  {ok} {pkg} (optional: {purpose})")
        except ImportError:
            console.print(f"  {warn} {pkg} not installed — {purpose} unavailable")

    # Config dir
    from pygeofetch.config.settings import get_config_dir

    cfg_dir = get_config_dir()
    if cfg_dir.exists():
        console.print(f"  {ok} Config directory: {cfg_dir}")
    else:
        console.print(
            f"  {warn} Config dir missing (will be created on first use): {cfg_dir}"
        )

    # Keyring
    try:
        import keyring

        kr = keyring.get_keyring()
        console.print(f"  {ok} Keyring backend: {type(kr).__name__}")
    except Exception as exc:
        console.print(f"  {warn} Keyring: {exc}")

    # Network connectivity
    console.print("\n  [dim]Checking provider connectivity...[/]")
    import httpx

    endpoints = [
        ("AWS Earth Search", "https://earth-search.aws.element84.com/v1/collections"),
        ("Planetary Computer", "https://planetarycomputer.microsoft.com/api/stac/v1/"),
        ("Element 84", "https://earth-search.aws.element84.com/v1"),
    ]
    for name, url in endpoints:
        try:
            resp = httpx.get(url, timeout=8)
            if resp.status_code < 400:
                console.print(f"  {ok} {name}: HTTP {resp.status_code}")
            else:
                console.print(f"  {warn} {name}: HTTP {resp.status_code}")
        except Exception as exc:
            console.print(f"  {fail} {name}: {exc}")

    console.print("\n[green]Doctor complete.[/]")


# ---------------------------------------------------------------------------
# PROVIDERS
# ---------------------------------------------------------------------------


@cli.group()
def providers() -> None:
    """List, filter, and inspect available satellite data providers."""


@providers.command(name="list")
@click.option("--auth/--no-auth", default=None, help="Filter by auth requirement.")
@click.option(
    "--capabilities",
    default=None,
    help="Comma-separated capabilities: sar,optical,sub-meter,stac,direct-s3",
)
@click.option(
    "--region", default=None, help="Filter by region (e.g. 'global', 'europe')."
)
@click.option("--satellite", default=None, help="Filter by satellite name substring.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def providers_list(auth, capabilities, region, satellite, as_json) -> None:
    """
    List available providers with optional filtering.

    \b
    Examples:
      pygeofetch providers list
      pygeofetch providers list --no-auth
      pygeofetch providers list --capabilities sar,stac
      pygeofetch providers list --satellite Sentinel
    """
    import json as _json

    from rich.table import Table

    from pygeofetch.providers import list_provider_info

    items = list_provider_info()

    # Apply filters
    if auth is not None:
        items = [i for i in items if i["requires_auth"] == auth]
    if capabilities:
        cap_set = {c.strip().lower() for c in capabilities.split(",")}

        def matches_caps(item):
            item_caps = set()
            if item.get("supports_sar"):
                item_caps.add("sar")
            if not item.get("supports_sar"):
                item_caps.add("optical")
            if item.get("supports_sub_meter"):
                item_caps.add("sub-meter")
            if item.get("stac"):
                item_caps.add("stac")
            if item.get("supports_direct_s3"):
                item_caps.add("direct-s3")
            return cap_set.issubset(item_caps)

        items = [i for i in items if matches_caps(i)]
    if region:
        items = [i for i in items if region in (i.get("regions") or [])]
    if satellite:
        sat_lower = satellite.lower()
        items = [
            i
            for i in items
            if any(sat_lower in s.lower() for s in (i.get("satellites") or []))
        ]

    if as_json:
        click.echo(_json.dumps(items, indent=2, default=str))
        return

    table = Table(
        title=f"Providers ({len(items)})", header_style="bold blue", show_lines=False
    )
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Auth", justify="center")
    table.add_column("SAR", justify="center")
    table.add_column("<1m", justify="center")
    table.add_column("STAC", justify="center")
    table.add_column("Satellites")

    for item in items:
        auth_str = (
            ("🔐 " + item.get("auth_type", "?")) if item["requires_auth"] else "🌐 open"
        )
        sats = ", ".join((item.get("satellites") or [])[:3])
        if len(item.get("satellites") or []) > 3:
            sats += f" +{len(item['satellites']) - 3}"
        table.add_row(
            item["id"],
            item.get("display_name", item["id"]),
            auth_str,
            "[green]✓[/]" if item.get("supports_sar") else "—",
            "[green]✓[/]" if item.get("supports_sub_meter") else "—",
            "[green]✓[/]" if item.get("stac") else "—",
            sats,
        )
    console.print(table)


@providers.command(name="info")
@click.argument("provider_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def providers_info(provider_id: str, as_json: bool) -> None:
    """
    Show detailed information about a specific provider.

    \b
    Example:
      pygeofetch providers info copernicus
      pygeofetch providers info planetary_computer --json
    """
    import json as _json

    from rich.table import Table

    from pygeofetch.core.authenticator import AuthManager
    from pygeofetch.providers import get_provider

    try:
        p = get_provider(provider_id)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        sys.exit(1)

    caps = p.get_capabilities()
    quota = p.get_quota_info()

    # Check auth status
    auth_mgr = AuthManager()
    authed_providers = [e["provider"] for e in auth_mgr.list()]
    if provider_id in authed_providers:
        auth_status = "✓ authenticated"
    elif not caps.requires_auth:
        auth_status = "🌐 no auth needed"
    else:
        auth_status = "✗ not configured"

    if as_json:
        data = {
            "id": provider_id,
            "name": caps.name,
            "description": caps.description,
            "auth_type": caps.auth_type,
            "auth_status": auth_status,
            "satellites": caps.satellites,
            "supports_sar": caps.supports_sar,
            "supports_sub_meter": caps.supports_sub_meter,
            "stac": caps.stac,
            "supports_cql2": caps.supports_cql2,
            "resolution_min_m": caps.resolution_min_m,
            "resolution_max_m": caps.resolution_max_m,
            "regions": caps.regions,
            "endpoint_url": caps.endpoint_url,
            "docs_url": caps.docs_url,
            "quota": {
                "total_bytes": quota.total_bytes,
                "used_bytes": quota.used_bytes,
                "requests_per_minute": quota.requests_per_minute,
            },
        }
        click.echo(_json.dumps(data, indent=2, default=str))
        return

    console.print(f"\n[bold cyan]{caps.name}[/] ({provider_id})")
    console.print(f"{caps.description}\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")

    table.add_row("Auth type", caps.auth_type)
    table.add_row(
        "Auth status",
        f"[green]{auth_status}[/]"
        if "✓" in auth_status or "🌐" in auth_status
        else f"[red]{auth_status}[/]",
    )
    table.add_row("Satellites", ", ".join(caps.satellites) if caps.satellites else "—")
    table.add_row("SAR support", "[green]Yes[/]" if caps.supports_sar else "No")
    table.add_row("Sub-meter", "[green]Yes[/]" if caps.supports_sub_meter else "No")
    table.add_row("STAC API", "[green]Yes[/]" if caps.stac else "No")
    table.add_row("CQL2 filter", "[green]Yes[/]" if caps.supports_cql2 else "No")
    table.add_row(
        "Resolution",
        f"{caps.resolution_min_m}m – {caps.resolution_max_m}m"
        if caps.resolution_min_m
        else "Varies",
    )
    table.add_row("Regions", ", ".join(caps.regions) if caps.regions else "global")
    table.add_row("Endpoint", caps.endpoint_url or "—")
    table.add_row("Docs", caps.docs_url or "—")

    if quota.total_bytes:
        used_pct = quota.usage_percent or 0
        table.add_row(
            "Quota used",
            f"{quota.used_bytes or 0:,} / {quota.total_bytes:,} bytes ({used_pct:.1f}%)",
        )
    if quota.requests_per_minute:
        table.add_row("Rate limit", f"{quota.requests_per_minute} req/min")

    console.print(table)


@providers.command(name="search")
@click.argument("term")
@click.option("--json", "as_json", is_flag=True)
def providers_search(term: str, as_json: bool) -> None:
    """Search providers by name, satellite, or description."""
    import json as _json

    from pygeofetch.providers import list_provider_info

    term_lower = term.lower()
    items = [
        i
        for i in list_provider_info()
        if (
            term_lower in i.get("display_name", "").lower()
            or term_lower in i.get("description", "").lower()
            or any(term_lower in s.lower() for s in (i.get("satellites") or []))
        )
    ]
    if as_json:
        click.echo(_json.dumps(items, indent=2, default=str))
        return
    for item in items:
        console.print(
            f"  [cyan]{item['id']}[/]: {item['display_name']} — {item['description'][:80]}"
        )


# ---------------------------------------------------------------------------
# CACHE
# ---------------------------------------------------------------------------


@cli.group()
def cache() -> None:
    """Manage the satellite data search result cache."""


@cache.command(name="stats")
@click.option("--json", "as_json", is_flag=True)
def cache_stats(as_json: bool) -> None:
    """Show cache statistics: size, entry count, location, TTL."""
    import json as _json

    from pygeofetch.core.cache_manager import CacheManager

    mgr = CacheManager()
    stats = mgr.stats()
    if as_json:
        click.echo(_json.dumps(stats, indent=2))
        return
    console.print("[bold]Cache Statistics[/]")
    for k, v in stats.items():
        console.print(f"  {k}: {v}")


@cache.command(name="clear")
@click.option("--provider", default=None, help="Clear only entries for this provider.")
@click.option(
    "--older-than",
    "older_than",
    default=None,
    help="Clear entries older than (e.g. 7d, 24h).",
)
@click.option(
    "--dry-run", is_flag=True, help="Show what would be removed without deleting."
)
@click.confirmation_option(prompt="Clear cache entries?")
def cache_clear(provider: str, older_than: str, dry_run: bool) -> None:
    """
    Delete cache entries with optional filters.

    \b
    Examples:
      pygeofetch cache clear
      pygeofetch cache clear --provider usgs
      pygeofetch cache clear --older-than 7d --dry-run
    """
    from pygeofetch.core.cache_manager import CacheManager

    mgr = CacheManager()

    max_age_seconds = None
    if older_than:
        unit = older_than[-1]
        val = int(older_than[:-1])
        max_age_seconds = val * {"d": 86400, "h": 3600, "m": 60}.get(unit, 1)

    if dry_run:
        console.print("[yellow]Dry run — no entries deleted.[/]")
        stats = mgr.stats()
        console.print(f"Would delete up to {stats['total']} entries.")
        return

    count = mgr.clear_filtered(
        provider_filter=provider, max_age_seconds=max_age_seconds
    )
    console.print(f"[green]Cleared {count} cache entries.[/]")


@cache.command(name="ttl")
@click.argument("action", type=click.Choice(["show", "set"]))
@click.argument("seconds", required=False, type=int)
def cache_ttl(action: str, seconds: int) -> None:
    """Show or set cache TTL. Usage: ttl show | ttl set SECONDS"""
    from pygeofetch.config.settings import get_settings, save_user_config

    if action == "show":
        settings = get_settings()
        ttl = getattr(settings, "cache_ttl_seconds", 3600)
        console.print(f"Cache TTL: {ttl} seconds ({ttl // 60} minutes)")
    else:
        if not seconds:
            console.print("[red]Provide SECONDS argument.[/]")
            sys.exit(1)
        save_user_config({"cache": {"ttl_seconds": seconds}})
        console.print(f"[green]Cache TTL set to {seconds}s ({seconds // 60} min).[/]")


@cache.command(name="location")
def cache_location() -> None:
    """Show the cache directory path."""
    from pygeofetch.core.cache_manager import CacheManager

    mgr = CacheManager()
    console.print(f"Cache directory: [cyan]{mgr.cache_dir}[/]")


@cache.command(name="prune")
@click.option(
    "--max-size",
    "max_size",
    default="1GB",
    show_default=True,
    help="Prune oldest entries until cache is below this size (e.g. 500MB, 2GB).",
)
def cache_prune(max_size: str) -> None:
    """Remove oldest cache entries to stay under max-size."""
    from pygeofetch.core.cache_manager import CacheManager

    # Parse max_size string
    multipliers = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    size_bytes = 1024**3  # default 1GB
    for unit, mult in multipliers.items():
        if max_size.upper().endswith(unit):
            try:
                size_bytes = int(float(max_size[: -len(unit)]) * mult)
                break
            except ValueError:
                pass
    mgr = CacheManager()
    n = mgr.purge_expired()
    stats = mgr.stats()
    if stats["size_bytes"] > size_bytes:
        extra = mgr.prune_to_size(size_bytes)
        n += extra
    console.print(
        f"[green]Pruned {n} entries. Cache now: {mgr.stats()['size_bytes'] // 1024} KB[/]"
    )


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------


@cli.group()
def pipeline() -> None:
    """Manage and run YAML-defined data pipelines."""


@pipeline.command(name="run")
@click.argument("pipeline_file", type=click.Path(exists=True))
@click.option("--step", default=None, help="Run only this step name.")
def pipeline_run(pipeline_file: str, step: str) -> None:
    """Run a pipeline from a YAML definition file."""
    from pygeofetch.core.engine import PyGeoFetch
    from pygeofetch.core.scheduler import PipelineScheduler

    sb = PyGeoFetch()
    scheduler = PipelineScheduler(engine=sb)
    pipeline_obj = scheduler.load_pipeline(pipeline_file)
    console.print(
        f"Running pipeline [bold]{pipeline_obj.name!r}[/] ({len(pipeline_obj.steps)} steps)..."
    )
    result = scheduler.run_once(pipeline_obj.name)
    if result["success"]:
        console.print(f"[green]✓ Completed in {result['duration_seconds']:.1f}s[/]")
    else:
        console.print(f"[red]✗ Failed after {result['duration_seconds']:.1f}s[/]")
        sys.exit(1)


@pipeline.command(name="validate")
@click.argument("pipeline_file", type=click.Path(exists=True))
def pipeline_validate(pipeline_file: str) -> None:
    """Validate a pipeline YAML file without executing it."""
    import yaml

    from pygeofetch.core.scheduler import Pipeline

    with open(pipeline_file) as f:
        data = yaml.safe_load(f)
    try:
        p = Pipeline.from_dict(data)
        console.print(
            f"[green]Valid[/]: {p.name!r} — {len(p.steps)} steps"
            + (f", cron={p.schedule!r}" if p.schedule else "")
        )
    except Exception as exc:
        console.print(f"[red]Invalid: {exc}[/]")
        sys.exit(1)


@pipeline.command(name="schedule")
@click.argument("pipeline_file", type=click.Path(exists=True))
@click.option("--name", default=None, help="Override pipeline name.")
@click.option("--cron", default=None, help="Override cron schedule expression.")
def pipeline_schedule(pipeline_file: str, name: str, cron: str) -> None:
    """Schedule a pipeline for recurring execution (saves to config)."""
    import json

    import yaml

    from pygeofetch.config.settings import get_config_dir
    from pygeofetch.core.scheduler import Pipeline

    with open(pipeline_file) as f:
        data = yaml.safe_load(f)
    if name:
        data["name"] = name
    if cron:
        data["schedule"] = cron
    p = Pipeline.from_dict(data)
    scheduled_file = get_config_dir() / "scheduled_pipelines.json"
    existing = {}
    if scheduled_file.exists():
        existing = json.loads(scheduled_file.read_text())
    existing[p.name] = {
        "file": str(pipeline_file),
        "schedule": p.schedule,
        "name": p.name,
    }
    scheduled_file.write_text(json.dumps(existing, indent=2))
    console.print(
        f"[green]Scheduled {p.name!r}[/]"
        + (f" at cron={p.schedule!r}" if p.schedule else " (no cron — run manually)")
    )


@pipeline.command(name="list-scheduled")
@click.option("--json", "as_json", is_flag=True)
def pipeline_list_scheduled(as_json: bool) -> None:
    """List all scheduled pipelines."""
    import json as _json

    from pygeofetch.config.settings import get_config_dir

    scheduled_file = get_config_dir() / "scheduled_pipelines.json"
    if not scheduled_file.exists():
        console.print("[dim]No scheduled pipelines.[/]")
        return
    data = _json.loads(scheduled_file.read_text())
    if as_json:
        click.echo(_json.dumps(data, indent=2))
        return
    for name, info in data.items():
        console.print(
            f"  [cyan]{name}[/]: {info.get('schedule', '(manual)')} — {info.get('file', '')}"
        )


@pipeline.command(name="unschedule")
@click.argument("name")
def pipeline_unschedule(name: str) -> None:
    """Remove a pipeline from the schedule."""
    import json as _json

    from pygeofetch.config.settings import get_config_dir

    scheduled_file = get_config_dir() / "scheduled_pipelines.json"
    if not scheduled_file.exists():
        console.print("[yellow]No scheduled pipelines found.[/]")
        return
    data = _json.loads(scheduled_file.read_text())
    if name not in data:
        console.print(f"[red]Pipeline {name!r} not found.[/]")
        sys.exit(1)
    del data[name]
    scheduled_file.write_text(_json.dumps(data, indent=2))
    console.print(f"[green]Unscheduled {name!r}.[/]")


@pipeline.command(name="logs")
@click.argument("name")
@click.option("--tail", default=50, show_default=True, help="Show last N lines.")
@click.option("--follow", is_flag=True, help="Follow log output (streaming).")
def pipeline_logs(name: str, tail: int, follow: bool) -> None:
    """Show execution logs for a pipeline."""
    import json as _json

    from pygeofetch.config.settings import get_config_dir

    log_file = get_config_dir() / "pipeline_logs" / f"{name}.jsonl"
    if not log_file.exists():
        console.print(f"[dim]No logs found for pipeline {name!r}.[/]")
        return
    lines = log_file.read_text().strip().splitlines()
    for line in lines[-tail:]:
        try:
            entry = _json.loads(line)
            ts = entry.get("timestamp", "")
            msg = entry.get("message", line)
            level = entry.get("level", "INFO")
            color = {"ERROR": "red", "WARNING": "yellow", "INFO": "white"}.get(
                level, "white"
            )
            console.print(f"[dim]{ts}[/] [{color}]{msg}[/]")
        except Exception:
            console.print(line)


@pipeline.command(name="history")
@click.option("--limit", default=20, show_default=True)
@click.option("--json", "as_json", is_flag=True)
def pipeline_history(limit: int, as_json: bool) -> None:
    """Show pipeline execution history."""
    import json as _json

    from pygeofetch.config.settings import get_config_dir

    history_file = get_config_dir() / "pipeline_history.jsonl"
    if not history_file.exists():
        console.print("[dim]No pipeline history recorded.[/]")
        return
    lines = history_file.read_text().strip().splitlines()[-limit:]
    runs = [_json.loads(ln) for ln in lines if ln.strip()]
    if as_json:
        click.echo(_json.dumps(runs, indent=2, default=str))
        return
    from rich.table import Table

    table = Table(header_style="bold blue")
    table.add_column("Pipeline")
    table.add_column("Started")
    table.add_column("Duration")
    table.add_column("Status")
    for run in reversed(runs):
        status_str = "[green]✓ OK[/]" if run.get("success") else "[red]✗ FAILED[/]"
        table.add_row(
            run.get("pipeline", "?"),
            run.get("started_at", "?"),
            f"{run.get('duration_seconds', 0):.1f}s",
            status_str,
        )
    console.print(table)


@pipeline.command(name="retry")
@click.argument("run_id")
def pipeline_retry(run_id: str) -> None:
    """Retry a failed pipeline run by its run ID."""
    console.print(
        f"[yellow]Retry for run {run_id!r}: re-run the pipeline file directly.[/]"
    )
    console.print("  pygeofetch pipeline run PIPELINE_FILE")


# ---------------------------------------------------------------------------
# SHELL COMPLETION
# ---------------------------------------------------------------------------


@cli.command(name="--install-completion", hidden=False)
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def install_completion(shell: str) -> None:
    """Install shell tab completion. Usage: --install-completion bash|zsh|fish"""
    import os

    prog = "pygeofetch"
    if shell == "bash":
        line = f'eval "$(_PYGEOFETCH_COMPLETE=bash_source {prog})"'
        rc = os.path.expanduser("~/.bashrc")
        console.print(f"Add to {rc}:\n  {line}")
    elif shell == "zsh":
        line = f'eval "$(_PYGEOFETCH_COMPLETE=zsh_source {prog})"'
        rc = os.path.expanduser("~/.zshrc")
        console.print(f"Add to {rc}:\n  {line}")
    elif shell == "fish":
        line = f"eval (env _PYGEOFETCH_COMPLETE=fish_source {prog})"
        cfg = os.path.expanduser("~/.config/fish/completions/pygeofetch.fish")
        console.print(f"Add to {cfg}:\n  {line}")


def main() -> None:
    """Package entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
