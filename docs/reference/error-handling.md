# Error handling and resilience

This page describes how pygeofetch reports failures and which parts of the download path
retry automatically.

## Exceptions

Provider code raises subclasses of `ProviderError`, defined in `pygeofetch.providers.base`:

| Exception | Raised when |
|---|---|
| `ProviderError` | Base class for everything below. |
| `AuthenticationError` | Credentials are missing, expired or rejected. |
| `SearchError` | A search request fails. |
| `DownloadError` | A download request fails. |
| `QuotaExceededError` | The provider's quota or rate limit is exceeded. |
| `ProviderUnavailableError` | The provider is temporarily unavailable. |

Other exceptions you may meet:

| Exception | Module | Meaning |
|---|---|---|
| `CircuitBreakerOpenError` | `pygeofetch.utils.retry_handler` | A provider failed repeatedly and its breaker is open (see below). |
| `OpticalValidationError` | `pygeofetch.validation.optical_validator` | A validation setting is invalid. |
| `StreamingUnsupportedError`, `AOIOutsideRasterError` | `pygeofetch.streaming` | The asset is an archive, or the AOI does not intersect the raster. |
| `MissingDependencyError` | `pygeofetch.core.data_formats` | An optional dependency (for example Zarr or GeoParquet support) is missing. |

```python
from pygeofetch.providers.base import ProviderError, AuthenticationError

try:
    results = client.search(query, providers=["usgs"])
except AuthenticationError:
    ...   # re-run `pygeofetch auth test usgs`
except ProviderError as exc:
    ...   # any other provider failure
```

### Search does not raise for a failed provider

`client.search()` runs providers in parallel. A provider that times out or raises is logged and
**excluded**, and the results from the other providers are returned. You get an empty list,
not an exception, if every provider fails. Check the log, or run
`pygeofetch auth test PROVIDER`, when a search unexpectedly returns nothing. The
`--on-provider-failure abort` and `retry` values are accepted but have no effect; see
[Known issues](known-issues.md).

### Downloads return results, not exceptions

`client.download()` returns one `DownloadResult` per input item. Failures are reported as
`status="failed"` with `error` and `error_type` set (the exception class name, or a
provider-specific value such as `ProductOffline`). The list length and order always match the
input. Inspect results rather than wrapping the call in `try`:

```python
for r in client.download(results, destination="data"):
    if r.status != "completed":
        print(r.data_id, r.error_type, r.error)
```

## Common problems

| Symptom | Likely cause and fix |
|---|---|
| Authentication fails | Run `pygeofetch auth test PROVIDER`. Re-add credentials with `auth add`. Environment variables are **not** read for credentials. |
| A provider times out | Raise `search run --timeout 120` (per-provider seconds, default 60). |
| Zero results | Widen `--cloud-cover 0-30`, check the bbox is `min_lon,min_lat,max_lon,max_lat` (longitude first), and check the date range. For Copernicus Sentinel-2 L2A the processing level value is `S2MSI2A`, not `L2A`. |
| Keyring errors in Docker or over SSH | There is no keyring daemon. Use `PyGeoFetch(auth_backend="file")`, or set `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`. |
| Rate limiting | Use `download run --parallel 1 --bandwidth-limit 2MB`. Commercial providers are the strictest. |
| Post-processing fails | Usually a missing `rasterio` or GDAL: `pip install "pygeofetch[geo]"`. |
| Downloaded data "not found" by your own code | Files are under `<destination>/<provider>/`, and archives are not extracted. See [Downloading](../data-access/download.md#where-files-are-written). |

## Download resilience

**Retries.** A download is attempted `retry_attempts + 1` times (default 3 retries). A retry
follows a provider failure, an exception, or a file that fails validation after download (for
example a truncated file). The delay before retry *n* is `retry_delay_seconds x 2^n`, capped at
60 seconds. When `retry_strategy` contains `"jitter"` (the default is `exponential_jitter`) the
delay is multiplied by a random factor between 0.5 and 1.0.

**Resume.** `resume=True` skips scenes whose file already exists and passes validation. A
missing or corrupt file is downloaded again from the start. The Copernicus provider additionally
continues a partial file across retries with an HTTP `Range` request. See
[Downloading](../data-access/download.md#resuming).

**Integrity.** After each download the file is checked for truncation or corruption. With
`verify_checksum=True`, the file is also compared with the checksum **the provider supplied** in
the asset metadata (MD5 if present, otherwise SHA256). A mismatch is logged and sets
`checksum_verified=False` on the result. It does not retry or fail the download. If the provider
supplied no checksum, nothing is compared. The `checksum_algorithm` option does not choose the
algorithm.

**Search cache.** Results are cached per query. `PyGeoFetch(cache_ttl=3600)` sets the lifetime,
and `search run --no-cache` bypasses it.

## Circuit breaker

`FederatedSearcher` keeps one provider instance per provider ID for its lifetime, and wraps
each provider search in that provider's circuit breaker. After `failure_threshold` consecutive
failures (default 5), the breaker opens and further calls fail immediately with
`CircuitBreakerOpenError`, which appears as a normal per-provider search error. After
`recovery_timeout` (default 60 seconds) one call is allowed through, and a success resets the
count. The breaker lives for the life of the `PyGeoFetch` instance, so a new process starts
with all breakers closed.
