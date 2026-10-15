"""
pygeofetch.core.data_formats — state-of-the-art analytical geospatial
formats: GeoParquet for vector/tabular data, Zarr for chunked,
out-of-core raster arrays.

Both `pyarrow` and `zarr` are real, heavy, optional dependencies --
neither is imported at package load time, so `import pygeofetch` stays
fast and lightweight for users who only need search/download.
"""

from __future__ import annotations


class MissingDependencyError(ImportError):
    """
    Real, specific exception for a missing optional dependency in this
    subpackage -- deliberately a *subclass* of ImportError, not a
    wholly separate exception type, so existing code that catches
    plain ImportError (the convention used everywhere else in
    pygeofetch) still catches this correctly, while callers who want
    to distinguish "a real dependency is missing" from "some other
    import genuinely failed" can catch this specific type instead.
    """
