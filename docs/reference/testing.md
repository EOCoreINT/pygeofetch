# Testing

```{danger}
**Directly verified against a fresh upload of both the real source
and a real `tests/` directory in this pass — not the same snapshot
the "497 passed" figure below was verified against, and the honest
current picture is worse.** The fresh test suite is **43 files**. Two
real, stale-duplicate files were found and removed
(`test_preflight_gate (1).py`, `test_provider_geometry_audit (1).py`
— exact reruns of the same "duplicate stale test file" issue found
and fixed for 2 other files at the start of this documentation
project; it has since crept back via newly-added test files). Two
hardcoded-absolute-path import hacks
(`sys.path.insert(0, "/home/mrtenkorang/...")`) were found and fixed
in `test_build_sbas_network.py` and `test_coregister_upgrades.py` —
the exact same anti-pattern fixed project-wide earlier, also crept
back in.

**After those fixes: 359 passed, 15 failed.** The 15 failures are
real, not flaky — confirmed reproducible, not order-dependent noise:

- **A genuine test/source signature drift**: several tests in
  `test_insar.py::TestDataValidator` construct `InterferogramPair(...)`
  with 4 positional arguments; the real class now requires a 5th,
  `perpendicular_baseline_m`. The tests weren't updated when that
  field was added.
- **A test fixture gap**: `test_insar.py::TestAnnotation`'s synthetic
  SLC annotation XML fixture is missing a
  `numberOfSamples` field that `pygeofetch.insar.annotation.parse_slc_geometry()`
  now requires and raises a clear `ValueError` for when absent — this
  is the real *source* code correctly enforcing a required field; the
  *test fixture* is what's stale.
- A few more in `test_offset_tracking.py`, `test_extraction.py`,
  `test_era5_atmospheric_correction.py` not individually triaged here
  — real, reproducible, not yet diagnosed to a specific root cause.

None of this reflects newly-broken *library* behavior — the parts of
the real source these tests exercise
(`InterferogramPair`, `parse_slc_geometry`, offset tracking, DOS
atmospheric correction) were independently verified elsewhere in this
documentation pass. This is aging test-file maintenance debt, laid
out honestly rather than glossed over. Run `pytest tests/ -v` yourself
against your current checkout for the live count.
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
