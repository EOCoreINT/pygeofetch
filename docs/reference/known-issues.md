# Known issues

Behaviours that differ from what you might expect, with workarounds. Each entry was checked
against the source or reproduced by running the code. Entries describe the current code; when
one is fixed, remove it here and note it in the [changelog](../project/changelog.md).

| Area | Issue | Impact |
|---|---|---|
| SAR | [Coherence fails on complex input](#sar-coherence-fails-on-complex-input) | `client.sar.coherence()` and `pygeofetch sar coherence` do not work |
| SAR | [Backends are not wired in](#sar-backends-are-not-wired-in) | `SARProcessor(backend=...)` raises `TypeError` |
| Configuration | [Settings layer is not applied](#settings-layer-is-not-applied-at-runtime) | Config files and `PYGEOFETCH_*` variables do not change behaviour |
| Search | [`--on-provider-failure`](#on-provider-failure-has-no-effect) | `abort` and `retry` do nothing |
| Download | [`extract_archives` and archives](#archives-are-not-extracted-by-extract-archives) | Option has no effect |
| Download | [Checksum verification](#checksum-verification-is-advisory) | Mismatch is logged, not retried |
| Install | [Undeclared dependencies](#undeclared-dependencies) | `pygeofetch --help` can fail on a clean install |
| Providers | [Registered but unusable](#registered-but-unusable-providers) | Two providers cannot return data |
| CLI | [`monitor` is not registered](#monitor-command-group-is-not-registered) | `pygeofetch monitor` does not exist |

## SAR coherence fails on complex input

**Symptom.** `client.sar.coherence(a, b)` and `pygeofetch sar coherence a.tif b.tif` raise
`ValueError: cannot reshape array of size 2048 into shape (64,64)` for `complex64` SLC rasters.

**Cause.** `pygeofetch.processing.sar.SARProcessor.coherence` reads complex bands through
`_safe_read_band`, which casts to real `float32`, and then reinterprets that memory as complex.
The standalone `pygeofetch.sar.SARProcessor` reads complex bands directly with `rasterio` and
does not have this problem.

**Workaround.**

```python
from pygeofetch.sar import SARProcessor
SARProcessor().coherence("slc_a.tif", "slc_b.tif", window=7, output="coherence.tif")
```

Reproduced with a 64 x 64 `complex64` pair: the `client.sar` call fails and the standalone call
returns a mean coherence of 1.0 for two identical-up-to-phase images.

## SAR backends are not wired in

**Symptom.** `SARProcessor(backend="sarxarray")` raises
`TypeError: SARProcessor() takes no arguments`.

**Cause.** `pygeofetch.sar.processor.SARProcessor` has no `__init__`. The modules `_native.py`,
`_sarxarray.py` and `_ost.py` exist in `pygeofetch/sar/` but nothing imports them, so the
`sarxarray` and `ost` backends, and `terrain_correct()`, are unreachable. The `pygeofetch[sar]`
and `pygeofetch[ost]` extras install packages that are not currently used.

**Workaround.** Use `SARProcessor()` with no arguments for despeckle, calibrate, flood mapping
and coherence.

## Settings layer is not applied at runtime

**Symptom.** Setting `download.parallel` in `~/.pygeofetch/config.yaml`, with
`pygeofetch config set`, or through `PYGEOFETCH_DOWNLOAD__PARALLEL` changes what
`pygeofetch config get` prints but not how downloads run.

**Cause.** `pygeofetch.config.settings.Settings` is read only by the `config` commands and
`cache ttl`. Search, download and the engine take their values from CLI flags and from
`SearchQuery`, `DownloadOptions` and the `PyGeoFetch()` constructor. Verified: with
`PYGEOFETCH_DOWNLOAD__PARALLEL=7`, `Settings().download.parallel` is `7` while
`DownloadOptions().parallel` is `4`. The same applies to `security.credential_storage`,
`security.verify_ssl`, `proxy.*`, `notifications.*` and `download.checksum_algorithm`. The
engine's `config_path` argument loads a dictionary that nothing reads.

**Related.** `pygeofetch cache ttl show` reads an attribute that does not exist on `Settings`
and always prints `3600`.

**Workaround.** Pass values explicitly. See [Configuration](configuration.md#what-controls-behaviour).

(on-provider-failure-has-no-effect)=
## `--on-provider-failure` has no effect

`search run --on-provider-failure abort` and `retry` are accepted and stored on `SearchQuery`,
but `FederatedSearcher` never reads the field. A provider that fails is logged and skipped
regardless of the value. Implement abort or retry in your own code.

(archives-are-not-extracted-by-extract-archives)=
## Archives are not extracted by `extract_archives`

`DownloadOptions.extract_archives` is declared (default `True`) but no code reads it. Copernicus
delivers `.SAFE.zip` and USGS delivers `.tar`; neither is extracted unless you add the `unzip`
post-processing action. See [Downloading](../data-access/download.md#archives).

## Checksum verification is advisory

With `verify_checksum=True`, a file is compared with the checksum the provider included in the
asset metadata (MD5 if present, otherwise SHA256). A mismatch is logged and
`checksum_verified` is set to `False`; the download is not retried or failed. If the provider
supplies no checksum, nothing is compared. `checksum_algorithm` does not select the algorithm.
Check `result.checksum_verified` and the logs yourself if integrity matters.

## Undeclared dependencies

The command-line interface imports `rich`, and optical validation imports `shapely`. In `pyproject.toml`, neither is a core dependency (`shapely` is
only in the `geo` and `insar` extras; the conda recipe does list `rich`). On a clean install, `pygeofetch --help` can fail with
`ModuleNotFoundError: No module named 'rich'`.

**Workaround.** `pip install rich shapely`, or install the `geo` extra for `shapely`.

## Registered but unusable providers

`esa_scihub` targets a service ESA decommissioned on 2 November 2023, and `google_earth_engine`
is not implemented. Both are registered, so they appear in `providers list` and count towards
"22 providers", but every call fails with an explanatory message. Use `copernicus` instead of
`esa_scihub`. The source files `providers/eodag_provider.py` and `providers/terrabotics.py`
remain in the package but are not registered. See [Providers](../data-access/providers.md).

## `monitor` command group is not registered

`pygeofetch/cli/monitor_commands.py` defines `monitor run` and `monitor history`, but
`cli/main.py` never calls `cli.add_command(monitor)`. `pygeofetch monitor` reports "No such
command".

## Smaller items

**Search and scoring**

- Deduplication uses provider and ID only. The same scene returned by two providers appears twice. De-duplicate on your own key.
- The relevance score uses cloud cover, recency and processing level. The docstring's fourth factor (spatial coverage) is not applied.
- STAC providers ignore `sort_by`. Sort results yourself, for example `sorted(results, key=lambda s: s.cloud_cover or 100)`.
- Credentials are not read from environment variables.

**Download behaviour**

- `download()` falls back to **all** assets when a requested band matches nothing (`resolve_band_keys()`), and logs a warning. [Streaming](../data-access/streaming.md) treats it as an error.
- Files are written under `<destination>/<provider>/`, which surprises code that expects a flat folder.

**Packaging and code**

- `pygeofetch/multisensor/_init.py` is a stale copy of `__init__.py` left beside it. It is unused and can be deleted.
- Version strings disagree: `pygeofetch.__version__` and the conda recipe say `2.6.3.0.4`, `pyproject.toml` says `2.6.3.0`, and the package docstring and CLI banner refer to `v2.4.0`. The documentation and conda metadata also point to different documentation URLs (GitHub Pages versus Read the Docs).
- `Affine * tuple` arithmetic (deprecated in the `affine` library) remains in `insar/flatearth.py`, `insar/coregister.py`, `insar/extraction.py` and `processing/classification.py`. The multi-sensor modules no longer use it.
- Two overlapping streaming implementations exist: `pygeofetch.stac_compute` (single-URL windowed reads and Sentinel-1 burst reads) and `pygeofetch.streaming` (scene-level). See [Streaming](../data-access/streaming.md#other-streaming-tools).
- Importing `pygeofetch.insar` prints a "snaphu not found" message when `snaphu` is not installed, even if you do not unwrap.

**Processing limitations**

- SAR radiometric calibration uses a fixed constant and nominal incidence angle: suitable for relative comparison within a scene, not for absolute accuracy. See [SAR](../processing/sar-insar/sar.md#radiometric-calibration).
- Atmospheric correction: only `dos1` and `dos2` are full implementations. See [Preprocessing](../processing/optical/preprocessing.md).
- Sentinel-2 L2A at baseline 04.00 or later carries a +1000 DN offset that spectral indices do not remove automatically.
