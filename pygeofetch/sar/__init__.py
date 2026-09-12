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

Five real, standard end-to-end pipelines are also available, each
orchestrating the atomic operations above into a real, common workflow
(see pygeofetch.sar.pipelines for full docs on each)::

    from pygeofetch.sar.pipelines import standard_grd_preprocessing_pipeline

    proc = SARProcessor()  # native backend, no extra deps
    result = standard_grd_preprocessing_pipeline(proc, "s1_grd_dn.tif")
"""

from pygeofetch.sar.extraction import GRDExtractor, georeference_via_gcps_if_needed
from pygeofetch.sar.pipelines import (
    PipelineResult,
    bright_target_detection_pipeline,
    change_detection_pipeline,
    coherence_disturbance_pipeline,
    flood_mapping_pipeline,
    standard_grd_preprocessing_pipeline,
)
from pygeofetch.sar.processor import SARProcessor

__all__ = [
    "SARProcessor",
    "GRDExtractor",
    "georeference_via_gcps_if_needed",
    "PipelineResult",
    "standard_grd_preprocessing_pipeline",
    "flood_mapping_pipeline",
    "change_detection_pipeline",
    "coherence_disturbance_pipeline",
    "bright_target_detection_pipeline",
]