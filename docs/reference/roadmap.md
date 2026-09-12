# Roadmap

:::{note}

Roadmap items are maintained by the project, not derivable from static
source inspection — this page intentionally doesn't repeat specific
version/feature claims from prior documentation that couldn't be
verified in this pass. For the current roadmap and to vote on
priorities, see GitHub Discussions on the
[EOCoreINT/pygeofetch](https://github.com/EOCoreINT/pygeofetch) repo.
:::
## Fixed during this documentation audit

These concrete, source-verified gaps were found during this audit and
have since been fixed, with real tests added for each:

- **Pipeline `process` and `export` steps** were stub implementations
  that silently did nothing — now delegate to the real post-processing
  action executor and genuinely transfer files (local disk, S3, GCS)
  with optional webhook notification. See [Pipelines & Batch Processing](pipelines.md).
- **`CircuitBreaker` was instantiated but never invoked**, and provider
  instances were recreated fresh on every call so failure state could
  never have accumulated even if it had been wired in — both fixed
  together. See [Error Handling & Resilience](error-handling.md).
- **Default credential file storage used base64, not encryption** —
  now uses real Fernet symmetric encryption with transparent migration
  for existing users. See [Security Model](security.md).
- **`airbus_oneatlas` provider was entirely non-functional** — fake
  auth, a fictional search endpoint, and a parser that never populated
  downloadable assets, so `download()` always failed regardless of
  credentials. Fully rewritten against Airbus's real, documented
  OneAtlas Data Living Library API. See [Providers](../core-features/providers.md).
- **`noaa_big_data` provider hit a fictional REST endpoint** — GOES
  imagery is served from real, public S3 buckets, not a search API.
  Rewritten to list real objects with real, directly-downloadable
  hrefs. See [Providers](../core-features/providers.md).
- **`esa_scihub` and `google_earth_engine` crashed on every call** —
  both called a method, `_check_integration_verified()`, that does not
  exist anywhere in the codebase. Both now fail immediately and
  honestly instead of crashing. See [Providers](../core-features/providers.md).
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
  yet individually triaged). See [Testing](testing.md) for the
  full, honest breakdown.

- **`--on-provider-failure` (search) is only partially real** —
  `abort`/`retry` are accepted and stored on `SearchQuery` but
  `FederatedSearcher.search()` never reads that field; every provider
  always runs and any failure is always just logged and skipped. See
  [Searching Satellite Data](../core-features/search.md).
- **A real, significant documentation error corrected**: an earlier
  version of [Authentication](../core-features/authentication.md) documented a
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
  [Downloading Satellite Data](../core-features/download.md).

- **`monitor` CLI command group is real but completely unreachable** —
  `pygeofetch/cli/monitor_commands.py` defines a working `monitor run`
  / `monitor history` group, but it's never registered in
  `pygeofetch/cli/main.py` (`cli.add_command(monitor)` is simply
  missing). Confirmed by running `pygeofetch --help` directly — no
  `monitor` entry appears. See [Full CLI Reference](cli.md).
- **Two more real architectural duplications**, following the same
  pattern as the `SARProcessor` one found earlier: `client.indices`
  (`pygeofetch.processing.indices.SpectralIndices`, wired in) vs. the
  standalone `pygeofetch.processor.indices.SpectralIndex` (spyndex-
  backed, not wired in) have different call shapes entirely. See
  [Spectral Indices](../processing/spectral-indices.md).
- **A real bug in `ProcessingPipeline`'s own class docstring**: its
  usage example (`client.pipeline.from_yaml(...)`) doesn't work —
  confirmed by running it (`AttributeError`). The real, working form
  is `ProcessingPipeline.from_yaml(path, engine=client)`. See
  [Pipelines & Batch Processing](pipelines.md).

## Added since this documentation audit

- **Optical data validation and preflight system**
  (`pygeofetch.validation`) — a configurable pre-download quality gate
  for optical imagery (AOI coverage, cloud cover, required bands,
  processing level, temporal bounds, and two opt-in heuristic checks),
  wired directly into `PyGeoFetch.search()` and `.download()` via a
  `validate_optical` toggle. See
  [Optical Data Validation & Preflight](../core-features/optical-validation.md).
- **`DataOrganizer`** (`pygeofetch.core.data_organizer`) — structured
  post-download hierarchies (by date/orbit/satellite/processing level),
  a real manifest mapping structure back to scene metadata, and
  `prepare_insar_stack()` real burst-sync-aware SLC grouping. See
  [Organizing Downloaded Data](../core-features/data-organizer.md).
- **Optical pixel offset tracking, strain, and InSAR fusion**
  (`pygeofetch.optical.offset_tracking`) — `compute_pixel_offsets`,
  `compute_horizontal_strain` (with real, tested outlier protection
  found necessary after a real 617%-strain artifact surfaced in this
  project's own Bu'ertai validation run), and `fuse_insar_optical`
  (real, coherence-weighted source selection, not an average). See
  [Optical Pixel Offset Tracking](../processing/optical-offset-tracking.md).
- **InSAR advanced safeguards** (`pygeofetch.insar.advanced_safeguards`)
  — real, geometry-based layover/shadow masking (verified against a
  synthetic cone DEM), custom high-resolution DEM alignment, and SBAS
  topographic-residual (DEM-error) co-estimation. See
  [InSAR Processing](../processing/insar.md#advanced-safeguards-custom-dems-layovershadow-and-topographic-residuals).
- **Five real, standard SAR processing pipelines**
  (`pygeofetch.sar.pipelines`) — standard GRD preprocessing, flood
  mapping, change detection, coherence-based disturbance monitoring,
  and CFAR bright-target detection, each orchestrating `SARProcessor`'s
  atomic operations. Also fixed a real, serious bug found while
  building these: `SARProcessor.coherence()` was completely
  non-functional for genuine complex SLC input (a shared read helper
  silently discarded the imaginary/phase part before a `.view()` call
  tried to reinterpret it). See [SAR Processing](../processing/sar.md).
- **Five real multi-sensor pipelines** (`pygeofetch.multisensor`) —
  combining two genuinely different sensor types each: InSAR+optical
  displacement fusion, SAR+optical flood mapping, terrain-corrected
  optical change detection (with a real, independently-verified solar
  position calculation), DEM-of-Difference volumetric change analysis,
  and optical NDVI + SAR coherence sub-canopy disturbance
  classification. All five are also exposed as real CLI commands
  (`pygeofetch multisensor ...`). See
  [Multi-Sensor Pipelines](../processing/multi-sensor-pipelines.md).
- **6 real geology/mineral-exploration spectral indices** and a real,
  confirmed bug fix in `RVI` (a genuine ratio index was being silently
  clamped to a meaningless constant `1.0` by a blanket `[-1, 1]` clip
  meant for normalized-difference indices only). See
  [Spectral Indices](../processing/spectral-indices.md).
- **`eodag_provider` and `terrabotics` providers removed** — the
  former only ever delegated to a separate, third-party multi-provider
  aggregator (and was never actually wired into the real provider
  registry in the first place); the latter targeted a real company
  with no public, verifiable API documentation to check its real
  endpoints against. See [Providers](../core-features/providers.md).
- **8 more providers individually rewritten** against their real,
  current APIs after the initial `airbus_oneatlas`/`noaa_big_data`
  fixes: `digitalglobe`, `maxar_gbdx` (GBDX itself confirmed shut down
  in 2022), `alaska_satellite_facility`, `inpe_cbers`, `jaxa_earth`,
  `isro_bhuvan`, `earth_explorer_additional`, `geoserver_generic` —
  plus targeted regression fixes in `nasa_earthdata`,
  `nasa_earthdata_cloud`, `planet`, and `sentinel_hub`. See
  [Providers](../core-features/providers.md) for what was real and
  specifically wrong in each.
