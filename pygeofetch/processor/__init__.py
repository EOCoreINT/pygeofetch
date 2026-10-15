"""
PyGeoFetch Processor — optional data loading and processing layer.

Install with: pip install "pygeofetch[processor]"

Usage::

    from pygeofetch.processor import DataLoader, SpectralIndex, BandStacker

    loader = DataLoader()
    data   = loader.load("B04.tif")

    si     = SpectralIndex()
    ndvi   = si.compute("NDVI", red="B04.tif", nir="B08.tif")
"""

from __future__ import annotations

from pygeofetch.processor.indices import SpectralIndex
from pygeofetch.processor.landsat import LandsatExtractor, LandsatScene
from pygeofetch.processor.loader import DataLoader
from pygeofetch.processor.stacker import BandStacker
from pygeofetch.processor.timeseries import IndexTimeStack, TimeSeriesAnalyzer

__all__ = [
    "DataLoader",
    "SpectralIndex",
    "BandStacker",
    "LandsatExtractor",
    "LandsatScene",
    "TimeSeriesAnalyzer",
    "IndexTimeStack",
]
