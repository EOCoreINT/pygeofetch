# Configuration Reference

pygeofetch uses a layered config system. Settings are merged from
lowest to highest precedence:

1. **Bundled defaults** (`pygeofetch/config/defaults.yaml`)
2. **`~/.pygeofetch/config.yaml`** — user-level config
3. **Environment variables** — `PYGEOFETCH_*`, case-insensitive, `__` for nesting
4. **CLI arguments** — highest priority

:::{note}

This page is generated directly from `pygeofetch/config/settings.py`
and its bundled `defaults.yaml`, not from a hand-maintained example —
so the field names and defaults below are exact.
:::
## Real bundled defaults

```yaml
general:
  output_dir: "./satellite_data"
  temp_dir: null            # null = system temp
  log_level: "INFO"
  log_file: null            # null = stderr only

download:
  parallel: 2
  chunk_size_mb: 8.0
  retry_attempts: 3
  retry_strategy: "exponential_jitter"
  retry_delay_seconds: 1.0
  verify_checksum: true
  checksum_algorithm: "md5"     # declared here, but see the warning below
  bandwidth_limit_mbps: 0       # 0 = unlimited; a float, not "10MB"
  timeout_seconds: 300
  overwrite: false
  keep_original: false
```

:::{danger}
**Real, confirmed bug, the same pattern as `security.credential_storage`
below**: `download.checksum_algorithm` above is declared in
`defaults.yaml`, but `client.download()` never actually reads it.
Confirmed directly: `Downloader.download_many()`'s real implementation
is `options = options or DownloadOptions()` — a bare pydantic model
construction that uses `DownloadOptions`' own independent default
(`ChecksumAlgorithm.SHA256`), completely bypassing the `Settings`/YAML
config layer shown here.

**The real, effective default when you call `client.download(...)`
without an explicit `options=` is `SHA256`, not `"md5"`.** If you want
`md5` (or any other algorithm), you must pass it explicitly:
`DownloadOptions(checksum_algorithm=ChecksumAlgorithm.MD5)` — setting
`download.checksum_algorithm` in config, via `pygeofetch config set`,
or via `PYGEOFETCH_DOWNLOAD__CHECKSUM_ALGORITHM` currently has no
effect on this. See [Python API Reference](python-api.md) for
`DownloadOptions`' real, effective defaults.
:::

```yaml
search:
  max_results: 100
  page_size: 100
  cache_ttl_seconds: 3600
  deduplicate: true
  sort_by: "datetime"
  sort_ascending: false         # bool, not sort_order: "asc"/"desc"

cache:
  enabled: true
  directory: "~/.pygeofetch/cache"
  max_size_gb: 5.0
  ttl_seconds: 86400
  provider_ttl:
    usgs: 3600
    copernicus: 3600
    nasa_earthdata: 3600

providers:
  usgs:
    base_url: "https://m2m.cr.usgs.gov/api/api/json/stable"
    timeout: 60
  copernicus:
    base_url: "https://catalogue.dataspace.copernicus.eu/odata/v1"
    auth_url: "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    timeout: 60
  # ... one block per provider with a non-default base_url/timeout

notifications:
  webhook: null
  email: null
  slack: null

proxy:
  http: null
  https: null
  no_proxy: []

security:
  verify_ssl: true
  credential_storage: "keyring"   # keyring | encrypted_file | plain_file — see the warning below
  credential_file: "~/.pygeofetch/credentials.enc"
```

:::{danger}
**Real, confirmed bug**: `security.credential_storage` (shown above,
settable via config file, `pygeofetch config set`, or
`PYGEOFETCH_SECURITY__CREDENTIAL_STORAGE`) is declared with a real
default in both `defaults.yaml` and the `Settings` pydantic model —
but is **never read anywhere else in the codebase**. Confirmed by
searching the entire real source tree for any other reference to it:
there is none. Changing this setting currently has **zero effect** on
which credential backend is actually used.

The setting that actually controls this is `PyGeoFetch()`'s own
`auth_backend` constructor parameter, which has its own real,
independent, hardcoded default of `"file"` — completely disconnected
from the config value above. See
[Security Model](../reference/security.md) for what `auth_backend`
actually does and its own real default.

If you need a specific credential backend, pass it explicitly:
`PyGeoFetch(auth_backend="keyring")` — do not rely on setting
`security.credential_storage` in config, which this documentation
would otherwise (incorrectly) suggest works.
:::

## Environment variables

```bash
# Format: PYGEOFETCH_{SECTION}__{FIELD} — double underscore for nesting.
# The prefix is case-insensitive, so lowercase also works.
export PYGEOFETCH_DOWNLOAD__PARALLEL=4
export PYGEOFETCH_GENERAL__LOG_LEVEL=DEBUG
export PYGEOFETCH_GENERAL__OUTPUT_DIR=/data/satellite
export PYGEOFETCH_CACHE__TTL_SECONDS=7200
```

## Config CLI commands

```bash
pygeofetch config show                                  # merged effective config
pygeofetch config get download.parallel
pygeofetch config set download.parallel 8
pygeofetch config set search.default_providers "aws_earth,planetary_computer"
pygeofetch config path                                    # show config file path
pygeofetch config reset                                    # reset to defaults
```

## In Python

```python
from pygeofetch.config.settings import Settings

settings = Settings()                       # defaults + env vars
settings = Settings.from_yaml("my_config.yaml")
print(settings.download.parallel)
```
