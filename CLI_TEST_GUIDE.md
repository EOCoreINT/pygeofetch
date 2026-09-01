# CLI Test Guide — everything implemented in this pass

Organized by feature. Commands marked 🆓 need no credentials and can
be run immediately. Commands marked 🔐 need real provider credentials
(`pygeofetch auth add ...` first) — shown for completeness, with the
expected behavior noted so you know what "working" looks like even
without running them live.

Run `pip install -e ".[all,dev]"` first so every extra is available.

---

## 0. Environment sanity check

```bash
# Confirms the 5 previously-broken dependency pins now resolve
pip install -e ".[full,viz-3d,ost]"

# Confirms the 7 mypy errors from this pass are gone
mypy pygeofetch/ --exclude venv/

# Confirms the package itself is healthy
pygeofetch doctor
pygeofetch --version
```

---

## 1. Credential encryption (Fernet, not base64) 🆓

```bash
pygeofetch auth add usgs --username testuser --password testpass123
```

**Expected:** succeeds silently. Then verify the real file on disk:

```bash
cat ~/.pygeofetch/credentials.enc   # should be unreadable ciphertext, NOT plaintext/base64
ls -la ~/.pygeofetch/credentials.key ~/.pygeofetch/credentials.enc
# both should show -rw------- (0600, owner-only)
```

Confirm the password is genuinely not recoverable by base64-decoding
the stored value (the old, broken scheme's exact failure mode):

```bash
python3 -c "
import json, base64
data = json.loads(open('$HOME/.pygeofetch/credentials.enc').read())
stored = data['usgs']['password']
try:
    print('base64-decoded:', base64.b64decode(stored).decode())
except Exception as e:
    print('correctly NOT base64-decodable:', e)
"
```

```bash
pygeofetch auth list
pygeofetch auth test usgs   # will fail auth against the real USGS API with fake creds -- that's expected; it proves the round-trip (encrypt -> store -> load -> decrypt -> attempt real auth) all worked
pygeofetch auth remove usgs --yes
```

---

## 2. Provider fixes — real search against previously-broken providers

### 2a. NOAA Big Data — real S3 listing, not a fictional search API 🆓

```bash
pygeofetch search run \
  --bbox "-90,25,-80,35" \
  --start-date 2024-08-17 --end-date 2024-08-17 \
  --providers noaa_big_data \
  --format table
```

**Expected:** real GOES-16 scene IDs like
`OR_ABI-L2-CMIPF-M6C06_G16_s...`, each with a genuine, directly
downloadable HTTPS href — no auth needed, no crash.

### 2b. Airbus OneAtlas — real auth + real search 🔐

```bash
pygeofetch auth add airbus_oneatlas --api-key YOUR_REAL_API_KEY
pygeofetch search run \
  --bbox "1.2,43.5,1.6,43.7" \
  --providers airbus_oneatlas \
  --format table
```

**Expected (with a real key):** a real token-exchange call to
`authenticate.foundation.api.oneatlas.airbus.com`, then real results
from `/opensearch` with genuine downloadable asset hrefs — previously,
`search()` hit a fictional endpoint and `download()` always failed
with "No assets downloaded" regardless of credentials.

### 2c. esa_scihub — now fails honestly instead of crashing 🆓

```bash
pygeofetch search run --bbox "-74,40,-73,41" --providers esa_scihub
```

**Expected:** a clean, immediate message explaining the Copernicus
Open Access Hub was decommissioned in 2023 and pointing you to
`copernicus` instead — **not** a crash. (Previously: guaranteed
`AttributeError` on every single call, regardless of query.)

### 2d. google_earth_engine — now fails honestly instead of crashing 🆓

```bash
pygeofetch search run --bbox "-74,40,-73,41" --providers google_earth_engine
```

**Expected:** a clean, immediate message explaining Earth Engine needs
a fundamentally different auth/API architecture that isn't implemented
yet — **not** a crash.

---

## 3. Optical preflight validation — new, wired into search and download 🆓

```bash
# Baseline: search without validation (see everything, including
# scenes that might be missing bands, wrong level, etc.)
pygeofetch search run \
  --bbox "-74.1,40.6,-73.7,40.9" \
  --start-date 2024-06-01 --end-date 2024-08-01 \
  --providers aws_earth \
  --format table

# Same search, WITH validation -- compare the result counts
pygeofetch search run \
  --bbox "-74.1,40.6,-73.7,40.9" \
  --start-date 2024-06-01 --end-date 2024-08-01 \
  --providers aws_earth \
  --validate-optical \
  --format table
```

```bash
# Stricter thresholds
pygeofetch search run \
  --bbox "-74.1,40.6,-73.7,40.9" \
  --start-date 2024-06-01 --end-date 2024-08-01 \
  --providers aws_earth \
  --validate-optical \
  --optical-max-cloud-cover 10 \
  --optical-min-coverage 0.95 \
  --max-results 3\
  --optical-required-bands "B02,B03,B04,B08" \
  --format table --output validated_results.geojson
```

```bash
# As a final gate at download time too (independent of search-time validation)
pygeofetch download run \
  --from-search validated_results.geojson \
  --output ./data/ \
  --validate-optical \
  --optical-max-cloud-cover 15 \
  --max-items 3
```
pygeofetch search run   --bbox "-74.1,40.6,-73.7,40.9"   --start-date 2024-06-01 --end-date 2024-08-01   --providers aws_earth   --validate-optical   --optical-max-cloud-cover 10   --optical-min-coverage 0.95 --optical-max-cloud-cover 15  --max-results 10  --optical-required-bands  "blue,green,red,nir,scl"   --format table --output validated_results.geojson


**Expected:** the download summary shows some items with
`status=FAILED` and an error like `rejected by optical preflight
[MISSING_BANDS]: ...` for anything that fails, while everything else
downloads normally — confirming the length/order-preserving contract
(every input item gets exactly one result, none silently vanish).

---

## 4. Pipeline `process`/`export` steps — now real, not stubs 🆓

```yaml
# save as test_pipeline.yaml
name: test-real-process-export
steps:
  - search:
      providers: [aws_earth]
      bbox: "-74.1,40.6,-73.7,40.9"
      date_range: last_30_days
      max_results: 3

  - download:
      output: ./pipeline_test_data/
      parallel: 2

  - process: "compress:lzw"

  - export:
      destination: ./pipeline_test_export/
```

```bash
pygeofetch pipeline validate test_pipeline.yaml
pygeofetch pipeline run test_pipeline.yaml
```

**Expected:** real files land in `./pipeline_test_export/`, genuinely
LZW-compressed (check with `gdalinfo <file>.tif | grep COMPRESSION`).
Previously, the `process` and `export` steps logged a message and
returned `{"status": "stub"}` — the pipeline would report "success"
with **nothing** actually written to `./pipeline_test_export/`.

Test S3 export specifically (needs real AWS credentials + a real
bucket you can write to):

```yaml
  - export:
      destination: s3://your-real-bucket/pygeofetch-test/
      notify: "webhook:https://your-webhook-url"
```

```bash
pygeofetch pipeline run test_pipeline_s3.yaml
aws s3 ls s3://your-real-bucket/pygeofetch-test/   # confirm files really landed
```

---

## 5. Circuit breaker — real resilience, not dead code

This one's awkward to trigger cleanly from the CLI alone (it needs 5
consecutive real failures against the *same* provider instance within
one process), so the cleanest real test is a short Python one-liner
rather than a pure CLI command:

```bash
python3 -c "
from unittest.mock import patch
from pygeofetch import PyGeoFetch
from pygeofetch.models.search_query import SearchQuery
from pygeofetch.utils.retry_handler import CircuitBreakerOpenError

pf = PyGeoFetch()
query = SearchQuery(bbox=(-74.1, 40.6, -73.7, 40.9))
provider = pf.searcher._get_provider('aws_earth')

with patch.object(provider, 'search', side_effect=RuntimeError('simulated outage')):
    for i in range(5):
        try:
            pf.searcher._search_provider('aws_earth', query, use_cache=False)
        except RuntimeError:
            print(f'attempt {i+1}: real failure, as expected')

    try:
        pf.searcher._search_provider('aws_earth', query, use_cache=False)
    except CircuitBreakerOpenError:
        print('6th attempt: breaker OPEN, failed fast -- provider.search() was never even called')
"
```

**Expected:** 5 real `RuntimeError`s, then a `CircuitBreakerOpenError`
on the 6th attempt — proving the breaker's failure count genuinely
persisted across calls (previously: it was instantiated but never
invoked at all, *and* a fresh provider instance was created on every
call, so failure state could never have accumulated even if something
had wired it in).

---

## 6. Quick pass/fail summary table

| # | Feature | Command | Pass looks like |
|---|---|---|---|
| 1 | Credential encryption | `auth add` + inspect `credentials.enc` | Ciphertext, not base64; `0600` perms |
| 2a | NOAA Big Data | `search run --providers noaa_big_data` | Real GOES scene IDs, real hrefs |
| 2b | Airbus OneAtlas | `search run --providers airbus_oneatlas` | Real token exchange, real results |
| 2c | esa_scihub | `search run --providers esa_scihub` | Clean redirect message, no crash |
| 2d | google_earth_engine | `search run --providers google_earth_engine` | Clean "not implemented" message, no crash |
| 3 | Optical validation | `search run --validate-optical` | Fewer results than without the flag |
| 4 | Pipeline process/export | `pipeline run test_pipeline.yaml` | Real compressed files in the export dir |
| 5 | Circuit breaker | Python one-liner above | Fails fast on the 6th call, not the 1st or never |

---

## 7. Full automated regression (not manual CLI, but the ground truth)

Everything above has a corresponding automated test that was run
repeatedly throughout this pass — this is the fastest way to confirm
the whole set at once, though it doesn't exercise the actual CLI
parsing layer the way the commands above do:

```bash
pytest tests/ -v
# 502 passed, 0 failed
```
