"""
PyGeoFetch Processing Engine — complete geospatial preprocessing,
spectral index computation, and post-processing.
"""

from pygeofetch.processing.base import ProcessingResult
from pygeofetch.processing.indices import SpectralIndices
from pygeofetch.processing.pipeline import ProcessingPipeline
from pygeofetch.processing.postprocessor import PostProcessor
from pygeofetch.processing.preprocessor import Preprocessor
from pygeofetch.processing.sar import SARProcessor

__all__ = [
    "ProcessingResult",
    "Preprocessor",
    "SpectralIndices",
    "PostProcessor",
    "SARProcessor",
    "ProcessingPipeline",
]
