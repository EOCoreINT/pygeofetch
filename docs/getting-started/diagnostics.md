# Doctor / Diagnostics

```bash
pygeofetch doctor
```

Checked directly against the real, current implementation — every
check below is what `doctor` actually does, not a general description
of what a diagnostic tool might do.

## What it actually checks, in order

**1. Python version** — must be 3.9+ (pygeofetch's real minimum).

**2. Required packages** — imports each of `httpx`, `pydantic`,
`click`, `rich`, `yaml`, `cryptography`, `keyring` directly and reports
✓/✗ per package. These are pygeofetch's real, always-installed core
dependencies (see [Installation](installation.md)) — a ✗ here means
your install is genuinely broken, not just missing an optional extra.

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

**4. Config directory** — confirms `pygeofetch`'s real config directory
exists (or notes it'll be created on first use).

**5. Keyring backend** — reports the real, actual keyring backend
class in use (e.g. `SecretService Keyring` on Linux desktop,
`macOS Keyring` on macOS). A ⚠ here (not a hard failure) is the real,
practical signal you're in a headless environment (Docker, CI, SSH
without a desktop session) where credential storage needs the
environment-variable fallback — see
[Authentication](../core-features/authentication.md) for that fallback.

**6. Live connectivity** — real HTTP GET requests (8-second timeout,
no auth) against three real, currently-open provider endpoints:

| Check | Real URL |
|---|---|
| AWS Earth Search | `https://earth-search.aws.element84.com/v1/collections` |
| Planetary Computer | `https://planetarycomputer.microsoft.com/api/stac/v1/` |
| Element 84 | `https://earth-search.aws.element84.com/v1` |

:::{note}
"AWS Earth Search" and "Element 84" check the same real host
(`earth-search.aws.element84.com`) at two different real paths — the
collections-listing endpoint and the root API endpoint respectively —
this is intentional, not a duplicate check; a network/firewall issue
that blocks one path but not the other is real, useful information.
:::

A ✗ on all three usually means a real, outright network problem
(no internet, or a corporate proxy blocking outbound HTTPS); a ✗ on
just one usually means that specific service is down or blocked
specifically, not your whole connection.

## Status dashboard

```bash
pygeofetch status
pygeofetch status --json
```

Shows the real, current `PyGeoFetch` instance's version, Python/platform
info, every registered provider with its real authentication status
(`✓ authenticated`, `🌐 open` for no-auth providers, or
`✗ not configured`), and real per-provider capability flags (SAR
support, sub-metre resolution, STAC compliance) alongside real cache
statistics.

The real `--json` output schema (confirmed directly against the
implementation, not assumed):

```json
{
  "version": "2.6.2.9",
  "python": "3.12.1",
  "platform": "Linux",
  "providers_authenticated": ["copernicus", "usgs"],
  "providers_free": ["aws_earth", "planetary_computer", "element84", "..."],
  "cache": {
    "...": "real, current CacheManager().stats() output"
  }
}
```

:::{note}
The JSON output is intentionally a smaller subset of what the
human-readable table shows — it does not include the real per-provider
SAR/`<1m`/STAC capability columns the table view does. If you need
those programmatically, use
`pygeofetch.providers.list_provider_info()` directly in Python rather
than parsing `status --json`.
:::

Useful for confirming, before kicking off a real pipeline run, exactly
which providers a given environment can actually reach and which
credentials are actually loaded — rather than discovering a missing
`auth add` step partway through a long-running batch job.
