# Testing

:::{admonition} Updated with a fresh, direct verification — real, current numbers
:class: tip

**Directly verified against the real, current codebase in this
pass**, after the fixes below — not the same snapshot the earlier
"359 passed, 15 failed" figure documented. The real, current test
suite is **73 files**. Real, fresh count: **684 passed, 0 failed.**

What changed between that figure and this one, concretely:

- **The stale-duplicate-test-file issue, found and fixed a third
  time**: `test_provider_geometry_audit.py` existed as two versions
  (one plain, one suffixed `(1).py`) — not identical copies, but
  genuinely different points in time, tracking which providers had
  moved off a shared generic template. Both were missing updates for
  5 providers rewritten in a later pass (`digitalglobe`,
  `jaxa_earth`, `earth_explorer_additional`,
  `alaska_satellite_facility`, `geoserver_generic`), each of which
  now has its own real, bespoke `_parse_item` shape matching its
  real API — causing 16 real `TypeError`/`AttributeError`/assertion
  failures that traced to a stale test expectation, not a bug in any
  provider. Fixed by extending this file's own already-established
  precedent (removing providers from the shared list once they get
  a bespoke rewrite, pointing to their own dedicated test file
  instead) and consolidating the two files into one.
- **A real, confirmed gap in the `test` extra**: `test_noaa_big_data.py`
  and one `test_pipeline_process_export.py` test hard-import `boto3`
  directly. The `test` extra didn't include it, breaking the extra's
  own documented promise ("everything required to run `pytest tests/`
  without ImportErrors"). Fixed by adding `cloud` to the `test`
  extra in `pyproject.toml` — confirmed by installing fresh and
  re-running: 4 real `ModuleNotFoundError`s became 0.

The 15-failures-with-named-root-causes list from the previous pass
(`InterferogramPair` signature drift, an annotation fixture gap, a
few untriaged failures) is **not re-confirmed here** — those were a
real, separate maintenance debt issue, independent of the two fixes
above, and weren't touched in this pass. If your checkout still
shows failures beyond the two categories just described, that debt
likely hasn't been addressed yet — run `pytest tests/ -v` yourself
against your current checkout for the live, true count; this page
can only describe the count at the moment it was last verified.
:::
## Running tests

```bash
# Install everything needed to run the real, complete suite
pip install -e ".[test]"

# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=pygeofetch --cov-report=html
open htmlcov/index.html

# Run a specific test file
pytest tests/test_insar.py -v
```

:::{admonition} Use `[test]`, not `[dev,all]`
:class: note

`pygeofetch[dev]` alone is enough for linting/formatting/type
checking, but not for running the full suite — several real tests
(S3 export, the `noaa_big_data` provider) hard-import `boto3`,
which isn't in `dev` or `all`. `pygeofetch[test]` is the real,
complete extra for this — see [Installation](../getting-started/installation.md).
:::
## What the suite actually covers (from the audited files)

The test files span the full processing stack — burst synchronization,
coregistration (`test_coregister_integration.py`,
`test_coregister_upgrades.py`), ESD/flat-earth phase
(`test_flat_earth_phase.py`, `test_sign_convention.py`), atmospheric
correction (`test_era5_atmospheric_correction.py`,
`test_atmospheric_circular_regression.py`), ionosphere
(`test_ionosphere.py`), SBAS network construction
(`test_build_sbas_network.py`, `test_wls_inversion.py`), stack
selection (`test_stack_selection_consolidation.py`,
`test_preflight_gate.py`), provider geometry
(`test_provider_geometry_audit.py`, `test_provider_scene_disambiguation.py`,
`test_providers.py`), visualization (`test_viz_plot.py`), and general
models/utilities (`test_models.py`, `test_utils.py`, `test_state.py`).

Since the pass that produced the counts above, further audit work
added real end-to-end tests for: pipeline `process`/`export` steps
(`test_pipeline_process_export.py`), the circuit breaker's real wiring
(`test_circuit_breaker_wiring.py`), Fernet credential encryption
(`test_credential_encryption.py`), the Airbus OneAtlas and NOAA Big
Data provider rewrites (`test_airbus_oneatlas.py`,
`test_noaa_big_data.py`), the `esa_scihub`/`google_earth_engine`
crash-to-honest-failure fixes (`test_esa_scihub.py`,
`test_google_earth_engine.py`), and the new optical validation module
(`test_optical_validator.py`, `test_optical_validation_wiring.py`).

A further, later pass added real tests for: eight more individually
researched-and-rewritten providers (`test_digitalglobe.py`,
`test_maxar_gbdx.py`, `test_alaska_satellite_facility.py`,
`test_inpe_cbers.py`, `test_jaxa_earth.py`, `test_isro_bhuvan.py`,
`test_earth_explorer_additional.py`, `test_geoserver_generic.py`),
targeted regression tests for real bugs found in already-existing
providers (`test_nasa_earthdata_public_search.py`,
`test_nasa_earthdata_cloud_s3_fix.py`, `test_planet_asset_type_fix.py`,
`test_sentinel_hub_catalog_fix.py`), the new geology spectral indices
and a real RVI clipping-bug fix (`test_geology_indices_and_rvi_fix.py`),
`DataOrganizer` (`test_data_organizer.py`), the optical
strain/fusion functions and the real outlier-protection story behind
them (`test_strain_and_fusion.py`), the InSAR layover/shadow/SBAS-topo
safeguards (`test_advanced_safeguards.py`), the real spatial-baseline
addition to `PreflightGate` (`test_preflight_spatial_baseline.py`),
expanded OpenTopography coverage — USGS 3DEP rasters and real point
cloud dataset discovery (`test_opentopography_pointcloud_usgs.py`) —
and the five new SAR processing pipelines plus the real coherence
bug fix they surfaced (`test_sar_pipelines.py`).

:::{note}

Claims about VCR cassette recording, hypothesis property-based
testing, a specific 80% coverage gate, and Codecov integration from
prior documentation were not independently re-confirmed against this
snapshot — none of those directories or tools were present in the
audited test tree. If your checkout has them, this page is describing
an older or different snapshot than yours.
:::