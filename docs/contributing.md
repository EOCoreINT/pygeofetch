# Contributing

Contributions of all kinds are welcome. See the repository's
`CONTRIBUTING.md` for full guidelines.

## Development setup

```bash
git clone git@github.com:EOCoreINT/pygeofetch.git
cd pygeofetch
pip install -e ".[test]"
pytest tests/ -v
```

## Good first issues

- 22 providers are now individually verified against their real,
  current APIs (see [Providers](core-features/providers.md) for the
  full, honest per-provider status) — if you find one whose real API
  has since changed, that's a real, live issue to open, not a stub to
  implement from scratch
- Closing the remaining, real, documented gaps flagged throughout
  these docs: `--on-provider-failure abort/retry` accepted but not
  actually read by `FederatedSearcher` (see
  [Searching Satellite Data](core-features/search.md)); the `monitor`
  CLI group defined but never registered (see
  [Full CLI Reference](reference/cli.md)); `security.credential_storage`
  declared but never read (see
  [Configuration Reference](reference/configuration.md))
- Improving test coverage — see [Testing](reference/testing.md) for
  the real, current pass/fail count and what's still untriaged
- Adding new post-processing actions (see [Pipelines & Batch Processing](reference/pipelines.md)
  for the action executor `process` pipeline steps now delegate to)
- Adding a GCS-export test path with a real (or mocked) bucket, to
  complement the S3 path already covered in
  `tests/test_pipeline_process_export.py`
- Extending [Multi-Sensor Pipelines](processing/multi-sensor-pipelines.md)'s
  terrain-corrected change detection beyond the current cosine
  correction to a more sophisticated real model (C-correction,
  Minnaert, SCS+C) for very steep terrain or dense canopy, where cosine
  correction is honestly documented as under-correcting

## License

MIT License. © Samuel Appiah Kubi.

`pygeofetch` is part of the PyGeoVision platform — pygeofetch handles
data acquisition and processing; PyGeoVision builds the Earth
observation AI/ML layer on top of it.
