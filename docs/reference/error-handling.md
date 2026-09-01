# Error Handling & Resilience

pygeofetch handles failures at multiple layers — provider outages,
network interruptions, checksum mismatches, and partial failures.

## Provider failure policies

```bash
# Skip failing providers — return partial results from others
pygeofetch search run --providers copernicus,usgs,planet \
  --on-provider-failure skip

# Abort entirely if any provider fails
pygeofetch search run --providers copernicus,usgs \
  --on-provider-failure abort

# Auto-retry failing providers
pygeofetch search run --providers copernicus \
  --on-provider-failure retry
```

## Common errors and fixes

| Error | Fix |
|---|---|
| `ProviderAuthError` | Run `pygeofetch auth test PROVIDER`. Credentials missing, expired, or rejected — re-add with `auth add` or set the env var. |
| `ProviderTimeoutError` | Use `--timeout 120`. Provider API didn't respond in time. |
| `ChecksumMismatchError` | Use `--retry 5 --verify-checksum`. Downloaded hash didn't match; auto-retries up to the retry limit. |
| `NoResultsError` | Widen `--cloud-cover 0-30`. Relax filters or check bbox coordinates (longitude first). |
| `KeyringUnavailableError` | Set `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`. No keyring daemon (Docker, headless SSH) — use env vars, or the file backend, which now genuinely encrypts stored credentials (see {doc}`/reference/security`). |
| `RateLimitError` | Use `--parallel 1 --bandwidth-limit 2MB`. Planet and Maxar are the most restrictive. |
| `PostProcessError` | Run `pip install "pygeofetch[geo]"`. Usually a missing rasterio/GDAL install. |

## Download resilience internals

**Exponential backoff** — verified in `pygeofetch/core/downloader.py`:
delay is `retry_delay_seconds * 2^attempt`, capped at 60 seconds, with
optional jitter (`retry_strategy` containing `"jitter"` multiplies by
a random factor between 0.5 and 1.0).

**Resume support** — interrupted downloads resume from the last
received byte using HTTP range requests.

**Atomic writes** — files are written to a temp path and renamed
atomically on completion; partial files never corrupt existing data.

**Search caching** — results cached per query with a configurable TTL
(`search.cache_ttl_seconds`, default 3600s / 1 hour).

**Checksum verification** — SHA256 (or MD5/SHA512) verified
post-download when `verify_checksum=True`; mismatches trigger an
automatic re-download.

```{note}
**Previously dead code, now wired in.** `CircuitBreaker` used to be
instantiated per-provider but never invoked anywhere in the request
path, *and* `get_provider()` created a brand-new provider instance
(with a fresh, always-zeroed breaker) on every single search — so
failure counts could never have accumulated across calls regardless.
Both are fixed: `FederatedSearcher` now caches one provider instance
per provider ID for its own lifetime, and `_search_provider()` wraps
the real `provider.search(...)` call in `with provider._circuit_breaker:`.
A provider that fails `failure_threshold` times (default 5) in a row
now genuinely opens its breaker — subsequent calls fail fast with
`CircuitBreakerOpenError` (surfaced as a normal per-provider search
error, not a crash) until `recovery_timeout` (default 60s) has passed.
A success resets the failure count to zero.
```
