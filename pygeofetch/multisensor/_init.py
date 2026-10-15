"""
Five real, standard multi-sensor task pipelines for pygeofetch.

Each combines two genuinely different sensor types to do something
neither one reliably does alone -- not multi-date processing of a
single sensor (see `pygeofetch.sar.pipelines` for that).

1. `insar_optical_displacement_pipeline` -- SAR interferometry
   (precise, but blind past the coherence limit) + optical pixel
   offset tracking (coarser, but robust exactly there). The real,
   proven combination from this project's own Bu'ertai validation run.
2. `multi_sensor_flood_pipeline` -- SAR (sees through cloud) + optical
   NDWI (cleaner water discrimination where cloud-free). The same real
   idea Copernicus's Emergency Management Service uses.
3. `terrain_corrected_change_pipeline` -- DEM-derived slope/aspect +
   real solar position per acquisition date, removing illumination
   artifacts from optical change detection in high-relief terrain.
4. `dem_differencing_pipeline` -- two DEMs, real "DEM of Difference"
   volumetric change analysis (Brasington et al. 2000).
5. `vegetation_disturbance_pipeline` -- optical NDVI trend + SAR
   coherence, distinguishing visible canopy loss from real sub-canopy
   ground disturbance optical can't see through the canopy.

Each returns a `pygeofetch.sar.pipelines.PipelineResult` -- the same
real result type the SAR pipelines use, for a consistent interface
across all of pygeofetch's orchestrated pipelines.
"""

from pygeofetch.multisensor.dem_differencing import dem_differencing_pipeline
from pygeofetch.multisensor.flood_mapping import multi_sensor_flood_pipeline
from pygeofetch.multisensor.insar_optical_fusion import (
    insar_optical_displacement_pipeline,
)
from pygeofetch.multisensor.terrain_corrected_change import (
    solar_position,
    terrain_corrected_change_pipeline,
)
from pygeofetch.multisensor.vegetation_disturbance import (
    CLASS_NO_DISTURBANCE,
    CLASS_SUBCANOPY_DISTURBANCE,
    CLASS_VISIBLE_DISTURBANCE,
    vegetation_disturbance_pipeline,
)

__all__ = [
    "insar_optical_displacement_pipeline",
    "multi_sensor_flood_pipeline",
    "terrain_corrected_change_pipeline",
    "solar_position",
    "dem_differencing_pipeline",
    "vegetation_disturbance_pipeline",
    "CLASS_NO_DISTURBANCE",
    "CLASS_VISIBLE_DISTURBANCE",
    "CLASS_SUBCANOPY_DISTURBANCE",
]
