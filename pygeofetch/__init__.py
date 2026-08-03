"""
PyGeoFetch v1.9.5 — Universal Satellite Data Pipeline.

Unified access to 22+ satellite data providers with a consistent
Python API and CLI. Supports authentication management, federated search,
parallel downloads, pipeline orchestration, and post-processing.

Quick Start::

    from pygeofetch import PyGeoFetch

    sb = PyGeoFetch()
    sb.add_credentials("usgs", username="user", password="pass")

    results = sb.search(
        providers=["usgs", "copernicus", "planetary_computer"],
        bbox=(-74.1, 40.6, -73.7, 40.9),
        start_date="2024-01-01",
        cloud_cover_max=20,
    )

    sb.download(results[:5], output="./data/", parallel=4)

CLI::

    pygeofetch auth add usgs --username user --password pass
    pygeofetch search run --bbox "-74,40,-73,41" --providers copernicus
    pygeofetch download run --from-search results.geojson --output ./data/
    pygeofetch status
    pygeofetch doctor
"""

__version__ = "1.9.5"
__author__ = "PyGeoFetch Contributors"
__license__ = "MIT"

from pygeofetch.core.engine import PyGeoFetch  # noqa: F401

__all__ = ["PyGeoFetch", "__version__"]


# Optional modules — lazy imported to avoid hard dependency errors
def _lazy(module_path: str, name: str):
    """Return a lazy accessor that imports on first use."""

    def _get():
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, name)

    return _get


# Convenience accessors (fail gracefully if optional deps not installed)
try:
    from pygeofetch.processor import BandStacker, DataLoader, SpectralIndex
except ImportError:
    DataLoader = SpectralIndex = BandStacker = None  # type: ignore[assignment,misc]

try:
    from pygeofetch.sar import SARProcessor as AdvancedSARProcessor
except ImportError:
    AdvancedSARProcessor = None  # type: ignore[assignment,misc]

try:
    from pygeofetch.viz import MapViewer, Plotter
except ImportError:
    MapViewer = Plotter = None  # type: ignore[assignment,misc]
