# Configuration

:::{important}
**Scope.** pygeofetch has a layered settings model (`pygeofetch.config.settings.Settings`),
but in the current release only the `pygeofetch config` commands and `pygeofetch cache ttl`
read it. The search, download and processing code does **not** consult it. Changing
`download.parallel` in `config.yaml` or through `PYGEOFETCH_DOWNLOAD__PARALLEL` does not
change how many downloads run.

Behaviour is controlled by the [options that actually take effect](#what-controls-behaviour)
below. See [Known issues](known-issues.md#settings-layer-is-not-applied-at-runtime).
:::

## What controls behaviour

| You want to change | Set it with |
|---|---|
| Parallelism, retries, checksum, bandwidth, bands, post-processing | CLI flags on `download run`, or [`DownloadOptions`](python-api.md#downloadoptions) in Python |
| Search limits, sorting, cloud cover | CLI flags on `search run`, or [`SearchQuery`](python-api.md#searchquery) in Python |
| Search-cache lifetime | `PyGeoFetch(cache_ttl=...)`, or `CacheManager(ttl_seconds=...)` |
| Search workers, credential backend, optical validation | `PyGeoFetch(max_search_workers=..., auth_backend=..., validate_optical=...)` |
| Log level, file and format | Global CLI options `--log-level`, `--log-file`, `--log-format`; or `PyGeoFetch(log_level=..., log_json=...)` |
| Where `snaphu` is found | Environment variable `PYGEOFETCH_SNAPHU_DIR` |
| Path of an external atmospheric-correction binary | Environment variable `PYGEOFETCH_<METHOD>_BIN` (for example `PYGEOFETCH_FLAASH_BIN`) |
| Log level for the CLI | Environment variable `PYGEOFETCH_LOG_LEVEL` |

`PyGeoFetch(config_path="my.yaml")` loads the file into `client.config`, a plain dictionary.
Nothing in the engine currently reads it.

## Credentials

Credentials are **not** read from environment variables. Pass them explicitly with
`add_credentials()` or `pygeofetch auth add`. See [Authentication](../data-access/authentication.md)
and [Security](security.md).

## The settings model

The `Settings` model and `pygeofetch config` commands provide a structured place to store
preferences. The schema and bundled defaults (`pygeofetch/config/defaults.yaml`) are:

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
  checksum_algorithm: "md5"
  bandwidth_limit_mbps: 0       # 0 = unlimited
  timeout_seconds: 300
  overwrite: false
  keep_original: false

search:
  max_results: 100
  page_size: 100
  cache_ttl_seconds: 3600
  deduplicate: true
  sort_by: "datetime"
  sort_ascending: false

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
  # ... one block per provider with a non-default base_url or timeout

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
  credential_storage: "keyring"
  credential_file: "~/.pygeofetch/credentials.enc"
```

Note the differences from the values that apply at runtime. `DownloadOptions()` defaults to
`parallel=4`, `verify_checksum=False` and `SHA256`, not the values above.

### Precedence

When `Settings` is loaded, later sources override earlier ones:

1. Bundled defaults (`pygeofetch/config/defaults.yaml`)
2. `~/.pygeofetch/config.yaml`
3. Environment variables `PYGEOFETCH_<SECTION>__<FIELD>` (case-insensitive, double underscore for nesting)

```bash
export PYGEOFETCH_DOWNLOAD__PARALLEL=4
export PYGEOFETCH_GENERAL__LOG_LEVEL=DEBUG
export PYGEOFETCH_CACHE__TTL_SECONDS=7200
```

## Configuration commands

```bash
pygeofetch config show                    # merged settings
pygeofetch config get download.parallel
pygeofetch config set download.parallel 8
pygeofetch config path                    # location of config.yaml
pygeofetch config reset                   # remove user overrides
```

`config set` writes to `~/.pygeofetch/config.yaml`. As described above, the value is stored
and shown by `config show`, but search and download do not read it.

:::{note}
`pygeofetch cache ttl show` reads a field that does not exist on `Settings` and always
prints `3600`. `pygeofetch cache ttl set N` does store `cache.ttl_seconds`. See
[Known issues](known-issues.md).
:::

## In Python

```python
from pygeofetch.config.settings import Settings

settings = Settings()                         # defaults, user file and environment
print(settings.download.parallel)
```
