"""
Configuration CLI commands for PyGeoFetch.

Usage::

    pygeofetch config show
    pygeofetch config get download.parallel
    pygeofetch config set download.parallel 4
    pygeofetch config path
"""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.syntax import Syntax

console = Console()


@click.group()
def config() -> None:
    """Inspect and modify PyGeoFetch configuration."""


@config.command(name="show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def config_show(as_json: bool) -> None:
    """Show the effective configuration (all layers merged)."""
    import json

    from pygeofetch.config.settings import get_settings

    settings = get_settings()
    data = settings.model_dump()

    if as_json:
        click.echo(json.dumps(data, indent=2, default=str))
        return

    import yaml

    rendered = yaml.dump(data, default_flow_style=False, sort_keys=True)
    console.print(Syntax(rendered, "yaml", theme="monokai"))


@config.command(name="get")
@click.argument("key")
def config_get(key: str) -> None:
    """
    Get a single configuration value by dotted KEY path.

    \b
    Example:
      pygeofetch config get download.parallel
    """
    from pygeofetch.config.settings import get_settings

    settings = get_settings()
    data = settings.model_dump()

    parts = key.split(".")
    value = data
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            console.print(f"[red]Key not found: {key!r}[/]")
            sys.exit(1)

    console.print(f"{key} = {value!r}")


@config.command(name="set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """
    Set a configuration value in the user config file.

    The value is cast to int/float/bool if it looks like one; otherwise stored as string.

    \b
    Example:
      pygeofetch config set download.parallel 4
      pygeofetch config set cache.ttl_seconds 7200
    """
    from pygeofetch.config.settings import save_user_config

    # Try to coerce to typed value
    typed_value: object = value
    if value.lower() in ("true", "yes"):
        typed_value = True
    elif value.lower() in ("false", "no"):
        typed_value = False
    else:
        try:
            typed_value = int(value)
        except ValueError:
            try:
                typed_value = float(value)
            except ValueError:
                pass  # keep as string

    # Build nested dict from dotted key
    parts = key.split(".")
    nested: dict = {}
    current = nested
    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            current[part] = typed_value
        else:
            current[part] = {}
            current = current[part]

    save_user_config(nested)
    console.print(f"[green]Set[/] {key} = {typed_value!r}")


@config.command(name="path")
def config_path() -> None:
    """Show the location of configuration files."""
    from pygeofetch.config.settings import get_config_dir

    cfg_dir = get_config_dir()
    console.print(f"Config directory: [cyan]{cfg_dir}[/]")
    console.print(f"User config:      [cyan]{cfg_dir / 'config.yaml'}[/]")
    console.print(f"Credentials:      [cyan]{cfg_dir / 'credentials.json'}[/]")
    console.print(f"Cache:            [cyan]{cfg_dir / 'cache'}[/]")


@config.command(name="reset")
@click.confirmation_option(prompt="Reset user config to defaults?")
def config_reset() -> None:
    """Remove the user configuration file, restoring defaults."""
    from pygeofetch.config.settings import get_config_dir

    cfg_path = get_config_dir() / "config.yaml"
    if cfg_path.exists():
        cfg_path.unlink()
        console.print(f"[green]Removed {cfg_path}[/]")
    else:
        console.print("[dim]No user config file found.[/]")
