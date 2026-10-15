"""
PyGeoFetch Viz — optional visualization layer.

Install with: pip install "pygeofetch[viz]"

Usage::

    from pygeofetch.viz import MapViewer, Plotter

    mv = MapViewer()
    mv.add_raster("ndvi.tif", colormap="RdYlGn")
    mv.show()

    pl = Plotter()
    pl.plot_raster("ndvi.tif", title="NDVI", output="ndvi.png")
"""

from pygeofetch.viz.map import MapViewer
from pygeofetch.viz.plot import Plotter

__all__ = ["MapViewer", "Plotter"]
