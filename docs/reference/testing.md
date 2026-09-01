# Testing

```{note}
**Directly verified in this documentation pass, against the specific
package snapshot audited here** (results may differ from whatever is
currently on the `main` branch of the GitHub repo, which this session
could not fetch directly): the test suite in this snapshot is **50
files, flat under `tests/`** — not organized into
`tests/unit/`/`tests/integration/`/`tests/property/`/`tests/cli/`
subdirectories. After a full audit and fix pass, all tests passed:
**497 passed, 0 failed**, stable across repeated and reordered runs
(up from 377 at the start of this pass — 9 new provider/core-fix test
files and the new optical validation module's own test files account
for the difference). Run `pytest tests/ -v` yourself against your
current checkout to see the live count and pass rate.
```

## Running tests

```bash
# Install dev dependencies
pip install -e ".[dev,all]"

# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=pygeofetch --cov-report=html
open htmlcov/index.html

# Run a specific test file
pytest tests/test_insar.py -v
```

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

```{note}
Claims about VCR cassette recording, hypothesis property-based
testing, a specific 80% coverage gate, and Codecov integration from
prior documentation were not independently re-confirmed against this
snapshot — none of those directories or tools were present in the
audited test tree. If your checkout has them, this page is describing
an older or different snapshot than yours.
```
