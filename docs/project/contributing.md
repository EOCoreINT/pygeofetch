# Contributing

Contributions of all kinds are welcome: bug reports, fixes, new providers, tests and
documentation. The repository's `CONTRIBUTING.md` holds the authoritative guidelines; this
page is a short orientation.

## Development setup

```bash
git clone git@github.com:EOCoreINT/pygeofetch.git
cd pygeofetch
pip install -e ".[test]"
pytest tests/ -v
```

See [Testing](../reference/testing.md) for how the suite is organised and why the `test`
extra is the right one.

## Where to start

Each item below links to the page that describes the current behaviour.

**Close a documented gap** (all listed in [Known issues](../reference/known-issues.md))

- Make `FederatedSearcher` honour `--on-provider-failure abort` and `retry`.
- Register the `monitor` command group in `cli/main.py`.
- Connect the [Settings layer](../reference/configuration.md) to search, download and the engine.
- Fix `client.sar.coherence` for complex input by reusing the approach in `pygeofetch.sar.processor`.
- Declare `rich` and `shapely` as dependencies.
- Delete the stale `multisensor/_init.py`.

**Improve quality**

- Add a GCS export test with a mocked bucket, to match the S3 coverage in `tests/test_pipeline_process_export.py`.
- Add post-processing actions (see [Pipelines](../workflows/pipelines.md)).
- Extend terrain-corrected change detection beyond the cosine correction to C-correction, Minnaert or SCS+C for steep terrain ([Multi-sensor pipelines](../processing/multisensor.md)).
- Re-verify providers whose upstream APIs may have changed. A failing provider is a live bug report, not a stub to reimplement.

## Conventions

- Add a test for every fix, and derive expected values independently of the code under test.
- Do not make tests depend on network access.
- When you change a CLI command, regenerate its reference page: `python tools/gen_cli_reference.py`.
- Keep documentation claims verifiable: run the example before you publish it.

## Licence

MIT. Copyright Samuel Appiah Kubi. pygeofetch is part of the PyGeoVision platform:
pygeofetch handles data acquisition and processing, and PyGeoVision builds the Earth
observation machine-learning layer on top.
