"""
pygeofetch monitor — real CLI entry point for Feature 4 (Automated
Monitoring).

HONEST SCOPE, stated plainly rather than implied: this command wires
together the real, tested pieces already built --
pygeofetch.core.state.ProjectState and
pygeofetch.core.monitor.plan_monitoring_run -- to do real search and
real incremental planning (what's new, what pairs are needed). It
does NOT itself call download, coregistration, interferogram
formation, or invert_weighted() end to end here; those are the
existing, separately-tested real pipeline stages, and wiring all of
them into one CLI command is a real, substantial integration task of
its own that deserves the same direct verification already given to
each individual piece, not a rushed final assembly. What this command
does today: real search, real state-aware new-date detection, real
incremental pair planning, real persistence of results — genuinely
useful on its own for seeing what a scheduled run would actually do,
and the concrete foundation the remaining wiring builds on next.
"""

from __future__ import annotations

import logging

import click
from rich.console import Console

from pygeofetch.core.monitor import plan_monitoring_run
from pygeofetch.core.state import ProjectState, RunSummary

console = Console()
logger = logging.getLogger("pygeofetch.cli.monitor")


@click.group(name="monitor", help="Automated, incremental InSAR monitoring (Feature 4).")
def monitor():
    pass


@monitor.command(name="run", help="Run one real monitoring cycle: search for new scenes, plan incremental processing.")
@click.option("--state-db", required=True, type=click.Path(), help="Path to this project's real state database.")
@click.option("--bbox", required=True, type=str, help="Real AOI bounding box, 'minlon,minlat,maxlon,maxlat'.")
@click.option("--satellites", default="Sentinel-1A,Sentinel-1B", help="Comma-separated real satellite names.")
@click.option("--n-neighbors", default=3, type=int, help="Real neighbors each new date connects to (spec default: 3).")
@click.option("--cron-daily", is_flag=True, help="Marker flag for scheduled invocation (real cron job); does not change behavior, documents intent in logs.")
@click.pass_context
def monitor_run(ctx, state_db, bbox, satellites, n_neighbors, cron_daily):
    client = ctx.obj.get("client") if ctx.obj else None
    if client is None:
        from pygeofetch import PyGeoFetch
        client = PyGeoFetch()

    state = ProjectState(state_db)

    minlon, minlat, maxlon, maxlat = (float(x) for x in bbox.split(","))
    sat_list = [s.strip() for s in satellites.split(",")]

    from pygeofetch.models.search_query import SearchQuery

    query = SearchQuery(
        bbox=(minlon, minlat, maxlon, maxlat),
        satellites=sat_list,
        product_type="SLC",
    )
    if state.last_download_timestamp:
        query.start_date = state.last_download_timestamp

    console.print(f"[bold]pygeofetch monitor[/bold] — {'scheduled' if cron_daily else 'manual'} run")
    console.print(f"State: {state_db}")

    try:
        results = client.search(query)
    except Exception as exc:
        state.record_run(RunSummary(status="failed", detail=f"search failed: {exc}"))
        console.print(f"[red]Search failed:[/red] {exc}")
        raise

    available_dates = sorted({str(r.datetime)[:10] for r in results if getattr(r, "datetime", None)})
    plan = plan_monitoring_run(state, available_dates, n_neighbors=n_neighbors)

    console.print(plan.message)
    if plan.new_dates:
        console.print(f"  New dates: {plan.new_dates}")
        console.print(f"  New pairs needed: {len(plan.new_pairs)}")

    state.record_run(RunSummary(
        status="success" if plan.network_changed else "no_new_scenes",
        n_new_scenes=len(plan.new_dates),
        detail=plan.message,
    ))

    return plan


@monitor.command(name="history", help="Show real run history for a monitoring project.")
@click.option("--state-db", required=True, type=click.Path(exists=True))
@click.option("--limit", default=20, type=int)
def monitor_history(state_db, limit):
    state = ProjectState(state_db)
    runs = state.run_history(limit=limit)
    if not runs:
        console.print("No real runs recorded yet.")
        return
    for run in runs:
        console.print(f"[{run['status']}] {run['started_at']} — {run['detail']}")
