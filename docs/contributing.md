# Contributing

Contributions of all kinds are welcome. See the repository's
`CONTRIBUTING.md` for full guidelines.

## Development setup

```bash
git clone git@github.com:EOCoreINT/pygeofetch.git
cd pygeofetch
pip install -e ".[dev,all]"
pytest tests/unit/ -v
```

## Good first issues

- Implementing stub providers into full API integrations (this refers
  to incomplete *provider* integrations — a different, still-open
  category from the pipeline stub steps fixed during this pass)
- Extending real footprint geometry support to the remaining
  bbox-only providers (see {doc}`/core-features/providers`)
- Improving test coverage
- Adding new post-processing actions (see {doc}`/reference/pipelines`
  for the action executor `process` pipeline steps now delegate to)
- Adding a GCS-export test path with a real (or mocked) bucket, to
  complement the S3 path already covered in
  `tests/test_pipeline_process_export.py`

## License

MIT License. © Samuel Appiah Kubi.

`pygeofetch` is part of the PyGeoVision platform — pygeofetch handles
data acquisition and processing; PyGeoVision builds the Earth
observation AI/ML layer on top of it.
