"""
Authentication CLI commands for PyGeoFetch.

Provides the ``pygeofetch auth`` command group for managing provider
credentials: adding, listing, removing, testing, and exporting them.

Usage::

    pygeofetch auth add usgs --username user --password pass
    pygeofetch auth add planet --api-key PL_KEY
    pygeofetch auth login copernicus        # interactive prompt
    pygeofetch auth list
    pygeofetch auth test usgs
    pygeofetch auth remove planet
    pygeofetch auth export --output creds_backup.json
"""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def auth() -> None:
    """Manage provider authentication credentials."""


@auth.command(name="add")
@click.argument("provider")
@click.option("--username", "-u", default=None, help="Username or email.")
@click.option("--password", "-p", default=None, help="Password (prompted if omitted).")
@click.option(
    "--api-key", "-k", default=None, help="API key (alternative to username/password)."
)
@click.option("--client-id", default=None, help="OAuth2 client ID.")
@click.option("--client-secret", default=None, help="OAuth2 client secret.")
@click.option("--token", default=None, help="Bearer token.")
@click.option(
    "--store",
    default="keyring",
    type=click.Choice(["keyring", "file"]),
    show_default=True,
    help="Credential storage backend.",
)
def auth_add(
    provider: str,
    username: str | None,
    password: str | None,
    api_key: str | None,
    client_id: str | None,
    client_secret: str | None,
    token: str | None,
    store: str,
) -> None:
    """
    Add credentials for PROVIDER.

    \b
    Examples:
      pygeofetch auth add usgs --username user --password pass
      pygeofetch auth add planet --api-key PL_KEY
      pygeofetch auth add copernicus --username me@example.com
    """
    from pygeofetch.core.authenticator import AuthManager

    creds: dict = {}

    if api_key:
        creds["api_key"] = api_key
    if client_id:
        creds["client_id"] = client_id
    if client_secret:
        creds["client_secret"] = client_secret
    if token:
        creds["token"] = token

    if not creds and not username:
        username = click.prompt(f"Username for {provider}")

    if username:
        creds["username"] = username
        if not password and not api_key:
            password = click.prompt(f"Password for {provider}", hide_input=True)
        if password:
            creds["password"] = password

    if not creds:
        console.print("[red]No credentials provided.[/]")
        sys.exit(1)

    mgr = AuthManager()
    mgr.add_credentials(provider, creds)
    console.print(f"[green]Credentials for [bold]{provider}[/] saved.[/]")


@auth.command(name="login")
@click.argument("provider")
def auth_login(provider: str) -> None:
    """
    Interactive login for PROVIDER (prompts for all required fields).

    \b
    Example:
      pygeofetch auth login copernicus
    """
    from pygeofetch.core.authenticator import AuthManager
    from pygeofetch.providers import get_provider

    prov = get_provider(provider)
    caps = prov.get_capabilities()

    console.print(f"\n[bold]Logging in to {caps.name}[/]\n")

    creds: dict = {}

    # Determine what to prompt based on auth type
    auth_type = caps.auth_type if hasattr(caps, "auth_type") else "username_password"

    if auth_type == "api_key":
        creds["api_key"] = click.prompt("API Key", hide_input=True)
    elif auth_type == "oauth2_client":
        creds["client_id"] = click.prompt("Client ID")
        creds["client_secret"] = click.prompt("Client Secret", hide_input=True)
    else:
        creds["username"] = click.prompt("Username")
        creds["password"] = click.prompt("Password", hide_input=True)

    mgr = AuthManager()
    mgr.add_credentials(provider, creds)

    console.print(f"\n[green]Credentials for [bold]{provider}[/] saved.[/]")

    # Optionally verify
    if click.confirm("Verify credentials now?", default=True):
        try:
            session = mgr.authenticate(provider)
            console.print(
                f"[green]Authentication successful![/] Session expires: {session.expires_at}"
            )
        except Exception as exc:
            console.print(f"[red]Authentication failed: {exc}[/]")


@auth.command(name="list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def auth_list(as_json: bool) -> None:
    """List all stored provider credentials."""
    from pygeofetch.core.authenticator import AuthManager

    mgr = AuthManager()
    entries = mgr.list()

    if as_json:
        click.echo(json.dumps(entries, indent=2, default=str))
        return

    if not entries:
        console.print("[dim]No credentials stored.[/]")
        return

    table = Table(title="Stored Credentials", header_style="bold blue")
    table.add_column("Provider", style="cyan")
    table.add_column("Has Username", justify="center")
    table.add_column("Has API Key", justify="center")
    table.add_column("Has Token", justify="center")

    for entry in entries:
        table.add_row(
            entry["provider"],
            "[green]✓[/]" if entry.get("has_username") else "—",
            "[green]✓[/]" if entry.get("has_api_key") else "—",
            "[green]✓[/]" if entry.get("has_token") else "—",
        )

    console.print(table)


@auth.command(name="test")
@click.argument("provider")
def auth_test(provider: str) -> None:
    """
    Test stored credentials for PROVIDER by attempting authentication.

    \b
    Example:
      pygeofetch auth test usgs
    """
    from pygeofetch.core.authenticator import AuthManager

    console.print(f"Testing credentials for [bold]{provider}[/]...")
    mgr = AuthManager()
    try:
        session = mgr.authenticate(provider)
        console.print(f"[green]Success![/] Session valid until: {session.expires_at}")
    except Exception as exc:
        console.print(f"[red]Failed: {exc}[/]")
        sys.exit(1)


@auth.command(name="remove")
@click.argument("provider")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def auth_remove(provider: str, yes: bool) -> None:
    """
    Remove stored credentials for PROVIDER.

    \b
    Example:
      pygeofetch auth remove planet
    """
    from pygeofetch.core.authenticator import AuthManager

    if not yes:
        click.confirm(f"Remove credentials for {provider!r}?", abort=True)

    mgr = AuthManager()
    removed = mgr.remove(provider)
    if removed:
        console.print(f"[green]Credentials for [bold]{provider}[/] removed.[/]")
    else:
        console.print(f"[yellow]No stored credentials found for {provider!r}.[/]")


@auth.command(name="export")
@click.option(
    "--provider", default=None, help="Export only this provider (default: all)."
)
@click.option(
    "--output",
    "-o",
    default="credentials_backup.json",
    show_default=True,
    help="Output file path.",
)
def auth_export(provider: str | None, output: str) -> None:
    """
    Export stored credentials to a JSON file for backup.

    WARNING: Exported file contains sensitive credentials — store securely.

    \b
    Example:
      pygeofetch auth export --output backup.json
      pygeofetch auth export --provider usgs --output usgs_creds.json
    """
    from pygeofetch.core.authenticator import AuthManager

    mgr = AuthManager()
    data = mgr.export_credentials(providers=[provider] if provider else None)

    with open(output, "w") as f:
        json.dump(data, f, indent=2, default=str)

    console.print(
        f"[yellow]WARNING: {output} contains sensitive credentials. "
        "Store it securely and delete after use.[/]\n"
        f"[green]Exported {len(data)} provider(s) to {output}[/]"
    )
