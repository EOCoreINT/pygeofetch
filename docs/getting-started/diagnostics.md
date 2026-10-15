# Doctor / Diagnostics

```bash
pygeofetch doctor
```

This is exactly what `doctor` checks, in order.

## Overview

`pygeofetch doctor` is the first thing to run after installing and the first thing to run when something does not work. It checks the Python version, that the required packages import, which optional feature groups are available, whether a keyring backend is usable, and whether three open providers can be reached. Each line of the output says what was checked and whether it passed, so most installation problems can be located without reading a traceback.

If a check fails, the message usually names the missing extra or the network problem. Install the extra it points to, or check your proxy and firewall settings, and run the command again. Include the output of `doctor` when you report a problem, since it records the environment the failure happened in.

## What it actually checks, in order

**1. Python version** — must be 3.9+ (pygeofetch's minimum).

**2. Required packages** — imports each of `httpx`, `pydantic`,
`click`, `rich`, `yaml`, `cryptography`, `keyring` directly and reports
✓/✗ per package. These are pygeofetch's always-installed core
dependencies (see [Installation](installation.md)) — a ✗ here means
your install is broken, not just missing an optional extra.

**3. Optional packages**, each reported with what it unlocks if
missing:

| Package | Unlocks |
|---|---|
| `boto3` | AWS S3 direct access |
| `rasterio` | Raster post-processing |
| `geopandas` | GeoParquet output |
| `croniter` | Cron scheduling |

A ⚠ (not a ✗) here is expected and fine if you haven't installed the
relevant extra yet — see [Installation](installation.md) for which
`pygeofetch[...]` extra provides each one. This list is deliberately
short; it does **not** check every optional dependency group (`insar`,
`sar`, `viz`, etc.) individually — only these four, specifically
chosen because they gate entire features (S3 export, any raster
post-processing at all, GeoParquet, scheduling) rather than one
specific provider or index.

**4. Config directory** — confirms `pygeofetch`'s config directory
exists (or notes it'll be created on first use).

**5. Keyring backend** — reports the actual keyring backend
class in use (e.g. `SecretService Keyring` on Linux desktop,
`macOS Keyring` on macOS). A ⚠ here (not a hard failure) is the real,
practical signal you're in a headless environment (Docker, CI, SSH
without a desktop session) where credential storage needs the
environment-variable fallback — see
[Authentication](../data-access/authentication.md) for that fallback.

**6. Live connectivity** — HTTP GET requests (8-second timeout,
no auth) against three currently-open provider endpoints:

| Check | Real URL |
|---|---|
| AWS Earth Search | `https://earth-search.aws.element84.com/v1/collections` |
| Planetary Computer | `https://planetarycomputer.microsoft.com/api/stac/v1/` |
| Element 84 | `https://earth-search.aws.element84.com/v1` |

:::{note}
"AWS Earth Search" and "Element 84" check the same host
(`earth-search.aws.element84.com`) at two different paths — the
collections-listing endpoint and the root API endpoint respectively —
this is intentional, not a duplicate check; a network/firewall issue
that blocks one path but not the other is useful information.
:::

A ✗ on all three usually means a outright network problem
(no internet, or a corporate proxy blocking outbound HTTPS); a ✗ on
just one usually means that specific service is down or blocked
specifically, not your whole connection.

## Status dashboard

```bash
pygeofetch status
pygeofetch status --json
```

Shows the current `PyGeoFetch` instance's version, Python/platform
info, every registered provider with its authentication status
(`✓ authenticated`, `🌐 open` for no-auth providers, or
`✗ not configured`), and per-provider capability flags (SAR
support, sub-metre resolution, STAC compliance) alongside cache
statistics.

The `--json` output schema:

```json
{
  "version": "2.6.3.0",
  "python": "3.12.1",
  "platform": "Linux",
  "providers_authenticated": ["copernicus", "usgs"],
  "providers_free": ["aws_earth", "planetary_computer", "element84", "..."],
  "cache": {
    "...": "CacheManager().stats() output"
  }
}
```

:::{note}
The JSON output is intentionally a smaller subset of what the
human-readable table shows — it does not include the per-provider
SAR/`<1m`/STAC capability columns the table view does. If you need
those programmatically, use
`pygeofetch.providers.list_provider_info()` directly in Python rather
than parsing `status --json`.
:::

Useful for confirming, before kicking off a pipeline run, exactly
which providers a given environment can actually reach and which
credentials are actually loaded — rather than discovering a missing
`auth add` step partway through a long-running batch job.
