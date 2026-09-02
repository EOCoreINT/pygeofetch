# Roadmap

```{note}
Roadmap items are maintained by the project, not derivable from static
source inspection — this page intentionally doesn't repeat specific
version/feature claims from prior documentation that couldn't be
verified in this pass. For the current roadmap and to vote on
priorities, see GitHub Discussions on the
[EOCoreINT/pygeofetch](https://github.com/EOCoreINT/pygeofetch) repo.
```

## Fixed during this documentation audit

These concrete, source-verified gaps were found during this audit and
have since been fixed, with real tests added for each:

- **Pipeline `process` and `export` steps** were stub implementations
  that silently did nothing — now delegate to the real post-processing
  action executor and genuinely transfer files (local disk, S3, GCS)
  with optional webhook notification. See {doc}`/reference/pipelines`.
- **`CircuitBreaker` was instantiated but never invoked**, and provider
  instances were recreated fresh on every call so failure state could
  never have accumulated even if it had been wired in — both fixed
  together. See {doc}`/reference/error-handling`.
- **Default credential file storage used base64, not encryption** —
  now uses real Fernet symmetric encryption with transparent migration
  for existing users. See {doc}`/reference/security`.
- **`airbus_oneatlas` provider was entirely non-functional** — fake
  auth, a fictional search endpoint, and a parser that never populated
  downloadable assets, so `download()` always failed regardless of
  credentials. Fully rewritten against Airbus's real, documented
  OneAtlas Data Living Library API. See {doc}`/core-features/providers`.
- **`noaa_big_data` provider hit a fictional REST endpoint** — GOES
  imagery is served from real, public S3 buckets, not a search API.
  Rewritten to list real objects with real, directly-downloadable
  hrefs. See {doc}`/core-features/providers`.
- **`esa_scihub` and `google_earth_engine` crashed on every call** —
  both called a method, `_check_integration_verified()`, that does not
  exist anywhere in the codebase. Both now fail immediately and
  honestly instead of crashing. See {doc}`/core-features/providers`.
- **5 broken dependency pins in `pyproject.toml`** (`whitebox`,
  `whiteboxgui`, `sidecar`, `contextily`, `opensartoolkit`) were
  pinned to minimum versions that have never existed on PyPI, breaking
  `pip install` for the affected extras outright.
- **7 real `mypy` errors** across `scheduler.py`, `authenticator.py`,
  and `viz/plot.py` — including a genuinely deprecated
  `matplotlib.cm.get_cmap()` call (already removed from newer
  matplotlib type stubs) and a latent type mismatch it had been
  masking.

## Found during this documentation pass, not yet fixed

- **Real test-suite maintenance debt, found from a fresh source
  upload**: 2 stale-duplicate test files and 2 hardcoded-absolute-path
  import hacks had crept back in since the project-wide fix earlier
  in this documentation effort — fixed. After that, **15 real,
  reproducible test failures remain** (an `InterferogramPair`
  constructor signature the tests weren't updated for, a stale
  synthetic-fixture XML missing a now-required field, and a few not
  yet individually triaged). See {doc}`/reference/testing` for the
  full, honest breakdown.

- **`--on-provider-failure` (search) is only partially real** —
  `abort`/`retry` are accepted and stored on `SearchQuery` but
  `FederatedSearcher.search()` never reads that field; every provider
  always runs and any failure is always just logged and skipped. See
  {doc}`/core-features/search`.
- **A real, significant documentation error corrected**: an earlier
  version of {doc}`/core-features/authentication` documented a
  `PYGEOFETCH_{PROVIDER}_{FIELD}` environment-variable credential
  auto-loading mechanism. It does not exist anywhere in the codebase
  — verified by searching the entire source tree for any
  environment-variable reading related to credentials. Corrected to
  document the real, working approach (read your own env vars, pass
  them to `add_credentials()` explicitly).
- **A real, verified correction to `--resume`'s documented
  behavior**: it does not perform byte-range/partial-file resume —
  it skips re-downloading files that already exist and pass
  validation, restarting from scratch otherwise. See
  {doc}`/core-features/download`.

- **`monitor` CLI command group is real but completely unreachable** —
  `pygeofetch/cli/monitor_commands.py` defines a working `monitor run`
  / `monitor history` group, but it's never registered in
  `pygeofetch/cli/main.py` (`cli.add_command(monitor)` is simply
  missing). Confirmed by running `pygeofetch --help` directly — no
  `monitor` entry appears. See {doc}`/reference/cli`.
- **Two more real architectural duplications**, following the same
  pattern as the `SARProcessor` one found earlier: `client.indices`
  (`pygeofetch.processing.indices.SpectralIndices`, wired in) vs. the
  standalone `pygeofetch.processor.indices.SpectralIndex` (spyndex-
  backed, not wired in) have different call shapes entirely. See
  {doc}`/processing/spectral-indices`.
- **A real bug in `ProcessingPipeline`'s own class docstring**: its
  usage example (`client.pipeline.from_yaml(...)`) doesn't work —
  confirmed by running it (`AttributeError`). The real, working form
  is `ProcessingPipeline.from_yaml(path, engine=client)`. See
  {doc}`/reference/pipelines`.

## Added since this documentation audit

- **Optical data validation and preflight system**
  (`pygeofetch.validation`) — a configurable pre-download quality gate
  for optical imagery (AOI coverage, cloud cover, required bands,
  processing level, temporal bounds, and two opt-in heuristic checks),
  wired directly into `PyGeoFetch.search()` and `.download()` via a
  `validate_optical` toggle. See
  {doc}`/core-features/optical-validation`.
