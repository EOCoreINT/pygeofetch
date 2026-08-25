"""
PyGeoFetch SAR — optional SAR processing layer.

Lightweight backend (sarxarray): pip install "pygeofetch[sar]"
Heavy backend (SNAP via OST):    pip install "pygeofetch[ost]"

Usage::

    from pygeofetch.sar import SARProcessor, GRDExtractor

    extractor = GRDExtractor(polarisation="VV")
    vv_path = extractor.extract_band(download_result, output_dir="./data")

    proc   = SARProcessor(backend="sarxarray")  # lightweight
    result = proc.calibrate(str(vv_path), output_type="sigma0")
    result = proc.despeckle("calibrated.tif", filter="lee")
"""

from pygeofetch.sar.extraction import GRDExtractor, georeference_via_gcps_if_needed
from pygeofetch.sar.processor import SARProcessor

__all__ = ["SARProcessor", "GRDExtractor", "georeference_via_gcps_if_needed"]
