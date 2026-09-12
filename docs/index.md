# pygeofetch documentation

**A universal satellite data pipeline.** One CLI, one Python API, 22
provider integrations — federated search, authenticated downloads,
InSAR/SAR processing, and pipeline orchestration, in pure Python.

:::{note}

This documentation is being rebuilt from source. A handful of examples
on the previous docs site didn't match the actual installed package
(e.g. the Python API examples referenced a lowercase `pygeofetch()`
constructor and plain-tuple `bbox` values that don't match the real
`PyGeoFetch` class and `SearchQuery`/`BoundingBox` models). Pages here
are checked against `pygeofetch`'s source directly; see each page's
examples for the corrected, working forms.
:::
## Who this is for

- Geospatial researchers who need data from multiple providers without
  learning 22 different APIs
- Engineers automating satellite data pipelines that need to run
  unattended, on a schedule
- Teams that need open-source Earth observation tooling without a
  commercial platform lock-in

## Quick links

- [Quick Start (5 Minutes)](getting-started/quickstart.md) — install to first download in five minutes
- [Python API Reference](reference/python-api.md) — the `PyGeoFetch` class, `SearchQuery`, `DownloadOptions`
- [Providers](core-features/providers.md) — all 22 providers, honest per-provider status
- [Organizing Downloaded Data](core-features/data-organizer.md) — structured hierarchies and InSAR stack prep from a flat download directory
- [Optical Data Validation & Preflight](core-features/optical-validation.md) — pre-download quality gates for optical imagery
- [InSAR Processing](processing/insar.md) — the pure-Python InSAR chain (no SNAP/ISCE required)
- [Optical Pixel Offset Tracking](processing/optical-offset-tracking.md) — measuring large ground displacement where InSAR structurally cannot
- [Multi-Sensor Pipelines](processing/multi-sensor-pipelines.md) — 5 real pipelines combining InSAR, optical, SAR, and DEM data
- [Complete Worked Example: Mexico City Subsidence](processing/insar-mexico-city-tutorial.md) — a complete, real, cell-by-cell InSAR run (search to validated subsidence map)
- [Full CLI Reference](reference/cli.md) — full command reference


:::{toctree}
:hidden:
:caption: Getting Started

getting-started/installation.md
getting-started/quickstart.md
getting-started/diagnostics.md
:::

:::{toctree}
:hidden:
:caption: Core Features

core-features/authentication.md
core-features/search.md
core-features/download.md
core-features/data-organizer.md
core-features/providers.md
core-features/optical-validation.md
:::

:::{toctree}
:hidden:
:caption: Processing & Analysis

processing/insar.md
processing/insar-mexico-city-tutorial.md
processing/optical-offset-tracking.md
processing/multi-sensor-pipelines.md
processing/sar.md
processing/spectral-indices.md
processing/landsat.md
processing/timeseries.md
processing/preprocessing.md
processing/terrain.md
processing/postprocessing.md
:::

:::{toctree}
:hidden:
:caption: Visualization

visualization/plotter.md
visualization/mapviewer.md
:::

:::{toctree}
:hidden:
:caption: Reference

reference/python-api.md
reference/cli.md
reference/pipelines.md
reference/configuration.md
reference/security.md
reference/error-handling.md
reference/docker.md
reference/testing.md
reference/roadmap.md
:::

:::{toctree}
:hidden:
:caption: Project

contributing.md
:::
