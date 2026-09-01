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

## Added since this documentation audit

- **Optical data validation and preflight system**
  (`pygeofetch.validation`) — a configurable pre-download quality gate
  for optical imagery (AOI coverage, cloud cover, required bands,
  processing level, temporal bounds, and two opt-in heuristic checks),
  wired directly into `PyGeoFetch.search()` and `.download()` via a
  `validate_optical` toggle. See
  {doc}`/core-features/optical-validation`.
