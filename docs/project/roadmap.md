# Roadmap

:::{note}
This page lists open work that follows from the [known issues](../reference/known-issues.md).
It is a proposal compiled during the documentation audit, **not a commitment by the
maintainers**. For priorities and voting, use GitHub Discussions on
[EOCoreINT/pygeofetch](https://github.com/EOCoreINT/pygeofetch).
:::

## Correctness

| Item | Why |
|---|---|
| Fix `client.sar.coherence` for complex input | Currently raises `ValueError`; the standalone processor already works. |
| Wire in or remove the SAR backends (`_native`, `_sarxarray`, `_ost`) | `SARProcessor(backend=...)` is documented in code but unreachable. |
| Implement `--on-provider-failure abort` and `retry` | The flag is accepted and ignored. |
| Honour `extract_archives` | The option is declared and unused. |
| Make checksum mismatches retry or fail | Verification is advisory today. |

## Configuration and packaging

| Item | Why |
|---|---|
| Apply the Settings layer at runtime | Files and `PYGEOFETCH_*` variables currently change nothing but `config get`. |
| Read credentials from environment variables | Needed for CI and containers. |
| Declare `rich` and `shapely` | A clean install can fail on `--help`. |
| Unify version strings | Three different values are in the source. |
| Register `monitor`, remove `multisensor/_init.py` | Dead or unreachable code. |

## Providers and data access

| Item | Why |
|---|---|
| Remove or unregister `esa_scihub` and `google_earth_engine` | They count as providers but cannot return data. |
| Apply `sort_by` for STAC providers | Currently ignored. |
| Cross-provider de-duplication | Dedup uses provider plus ID only. |
| Merge `stac_compute` and `streaming` | Two overlapping implementations. |
| Live integration tests for streaming | Verified only against a local range server. |

## Processing

| Item | Why |
|---|---|
| Absolute SAR radiometric calibration from product metadata | The current constant is suitable for relative comparison only. |
| Remaining `Affine * tuple` call sites | Deprecated arithmetic in four modules. |
| Advanced topographic correction | Cosine correction under-corrects on steep terrain. |

## Documentation

- Keep the generated CLI reference in CI with `tools/check_docs.py`.
- Add provider-specific tutorials once live access can be tested.

Completed work is in the [changelog](changelog.md).
