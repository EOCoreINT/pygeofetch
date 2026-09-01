# PyGeoFetch Official Documentation

<div align="center">
<br>
<img src="https://raw.githubusercontent.com/EOCoreINT/pygeofetch/refs/heads/main/icon/concept_a.png" alt="PyGeoFetch Logo" width="350">
<br>

**A universal satellite data pipeline.** One CLI, one Python API, 24
provider integrations — federated search, authenticated downloads,
InSAR/SAR processing, and pipeline orchestration, in pure Python.

```{note}
This documentation is being rebuilt from source. A handful of examples
on the previous docs site didn't match the actual installed package
(e.g. the Python API examples referenced a lowercase `pygeofetch()`
constructor and plain-tuple `bbox` values that don't match the real
`PyGeoFetch` class and `SearchQuery`/`BoundingBox` models). Pages here
are checked against `pygeofetch`'s source directly; see each page's
examples for the corrected, working forms.
```

## Who this is for

- Geospatial researchers who need data from multiple providers without
  learning 24 different APIs
- Engineers automating satellite data pipelines that need to run
  unattended, on a schedule
- Teams that need open-source Earth observation tooling without a
  commercial platform lock-in

## Quick links

- {doc}`getting-started/quickstart` — install to first download in five minutes
- {doc}`reference/python-api` — the `PyGeoFetch` class, `SearchQuery`, `DownloadOptions`
- {doc}`core-features/providers` — all 24 providers, honest per-provider status
- {doc}`core-features/optical-validation` — pre-download quality gates for optical imagery
- {doc}`processing/insar` — the pure-Python InSAR chain (no SNAP/ISCE required)
- {doc}`reference/cli` — full command reference

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting-started/installation
getting-started/quickstart
getting-started/diagnostics
```

```{toctree}
:maxdepth: 2
:caption: Core Features

core-features/authentication
core-features/search
core-features/download
core-features/providers
core-features/optical-validation
```

```{toctree}
:maxdepth: 2
:caption: Processing & Analysis

processing/insar
processing/sar
processing/spectral-indices
processing/landsat
processing/timeseries
processing/preprocessing
processing/terrain
processing/postprocessing
```

```{toctree}
:maxdepth: 2
:caption: Visualization

visualization/plotter
visualization/mapviewer
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/python-api
reference/cli
reference/pipelines
reference/configuration
reference/security
reference/error-handling
reference/docker
reference/testing
reference/roadmap
```

```{toctree}
:maxdepth: 1
:caption: Project

contributing
```
