# Testing

The test suite lives in a flat `tests/` directory and runs with `pytest`. Test counts change
with every commit, so this page does not quote one. Run the suite against your own checkout
for the current result.

## Run the tests

```bash
pip install -e ".[test]"        # dev tools + geo + insar + cloud

pytest tests/ -v                # everything
pytest tests/test_insar.py -v   # one file
pytest tests/ -k streaming      # by keyword

pytest tests/ --cov=pygeofetch --cov-report=html   # coverage report in htmlcov/
```

:::{important}
Install the `test` extra, not `dev,all`. Several tests (S3 export, the `noaa_big_data`
provider) import `boto3` directly, which is part of `cloud`. Without it they fail with
`ModuleNotFoundError`, not with a real test failure. See
[Installation](../getting-started/installation.md).
:::

## What is covered

| Area | Representative test files |
|---|---|
| **Models and utilities** | `test_models.py`, `test_utils.py`, `test_state.py` |
| **Providers** | `test_providers.py`, `test_provider_geometry_audit.py`, `test_provider_scene_disambiguation.py`, and one file per rewritten provider: `test_airbus_oneatlas.py`, `test_digitalglobe.py`, `test_maxar_gbdx.py`, `test_alaska_satellite_facility.py`, `test_inpe_cbers.py`, `test_jaxa_earth.py`, `test_isro_bhuvan.py`, `test_noaa_big_data.py`, `test_opentopography_pointcloud_usgs.py` |
| **Provider regressions** | `test_nasa_earthdata_public_search.py`, `test_nasa_earthdata_cloud_s3_fix.py`, `test_planet_asset_type_fix.py`, `test_sentinel_hub_catalog_fix.py` |
| **Authentication and security** | `test_credential_encryption.py` |
| **Resilience** | `test_circuit_breaker_wiring.py`, `test_pipeline_process_export.py` |
| **Optical validation** | `test_optical_validator.py`, `test_optical_validator_stac.py`, `test_optical_validation_wiring.py`, `test_optical_validation_real_provider_bugs.py` |
| **Spectral indices** | `test_geology_indices_and_rvi_fix.py` |
| **InSAR** | `test_coregister_integration.py`, `test_coregister_upgrades.py`, `test_flat_earth_phase.py`, `test_sign_convention.py`, `test_ionosphere.py`, `test_build_sbas_network.py`, `test_wls_inversion.py`, `test_stack_selection_consolidation.py`, `test_preflight_gate.py`, `test_preflight_spatial_baseline.py`, `test_advanced_safeguards.py` |
| **Atmospheric correction** | `test_era5_atmospheric_correction.py`, `test_atmospheric_circular_regression.py` |
| **SAR pipelines** | `test_sar_pipelines.py` |
| **Optical offset tracking, strain and fusion** | `test_optical_offset_tracking.py`, `test_strain_and_fusion.py` |
| **Multi-sensor pipelines** | `tests/multisensor/` (five pipeline suites, plus the audit regressions and terrain/packaging pins) |
| **Streaming** | `test_streaming.py` |
| **Data organisation** | `test_data_organizer.py` |
| **Visualization** | `test_viz_plot.py` |

## Streaming tests

`tests/test_streaming.py` runs a local HTTP server that implements range requests and counts
the bytes it serves. The tests assert that an AOI read moves a small fraction of the file,
that pixel values equal a local read of the same window, and that archives are refused. No
test contacts a live provider.

## Multi-sensor regression tests

The multi-sensor audit added tests that fail on the earlier code and pass on the fix, for
example a flood scene with optical nodata, a DEM with a `-9999` void, volume on a geographic
grid compared with a geodesic-area reference, and illumination derived independently from
`unit_normal . unit_sun`. Deriving the expectation independently matters: an earlier terrain
test inverted the pipeline's own correction with the pipeline's own aspect function, so a
sign error cancelled out and passed.

## Writing tests

- Put new tests in `tests/`, named `test_<area>.py`.
- Test against a real behaviour, not only that a call returns. For numerical code, derive the expected value independently of the code under test.
- Do not depend on network access. Use a local server or mock for HTTP.
- Skip, rather than fail, when an optional dependency is missing: `pytest.importorskip("rasterio")`.
