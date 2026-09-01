"""
Task 3 integration example: fetch_optical_stack().

Illustrates how OpticalPreflightValidator and OpticalValidationConfig
slot between PyGeoFetch.search() and PyGeoFetch.download() in a real
workflow, and how OpticalValidationError surfaces as a clear,
actionable message.

This is a documentation/example script, not a pygeofetch module --
the real integration point is simply calling
`validator.run_preflight(results, aoi)` between your own search and
download calls (see the docstring at the top of
pygeofetch/validation/optical_validator.py for the minimal form).
"""

from __future__ import annotations

from pathlib import Path

from shapely.geometry import Polygon

from pygeofetch import PyGeoFetch
from pygeofetch.models import SearchQuery
from pygeofetch.validation import (
    OpticalPreflightValidator,
    OpticalValidationConfig,
    OpticalValidationError,
)


def fetch_optical_stack(
    aoi: Polygon,
    start_date: str,
    end_date: str,
    destination: str | Path = "./data",
    providers: list[str] | None = None,
    validation_config: OpticalValidationConfig | None = None,
) -> list:
    """
    Search, validate, and download an optical scene stack in one call.

    Parameters
    ----------
    aoi : shapely.geometry.Polygon
        Area of interest.
    start_date, end_date : str
        ISO date strings bounding the search.
    destination : str or Path
        Where downloaded scenes are written.
    providers : list[str] or None
        Providers to search. Defaults to a few reliable, no-auth
        optical providers if not given.
    validation_config : OpticalValidationConfig or None
        Defaults to :class:`OpticalValidationConfig`'s own defaults
        (AOI coverage / cloud cover / required bands / processing
        level / temporal bounds all enabled; snow-ice and nodata-margin
        checks off).

    Returns
    -------
    list[DownloadResult]
        Results for every scene that passed validation and was
        downloaded.

    Raises
    ------
    OpticalValidationError
        Only if a genuinely unrecoverable, catalog-wide problem occurs
        (e.g. every candidate scene is missing required bands) --
        per-scene hard failures are filtered out silently by
        ``run_preflight`` and logged, not raised, so a few bad scenes
        in a big search never abort the whole run.
    """
    pf = PyGeoFetch()
    validator = OpticalPreflightValidator(validation_config or OpticalValidationConfig())

    minx, miny, maxx, maxy = aoi.bounds
    results = pf.search(
        SearchQuery(
            bbox=(minx, miny, maxx, maxy),
            start_date=start_date,
            end_date=end_date,
            max_results=100,
        ),
        providers=providers or ["copernicus", "aws_earth", "planetary_computer"],
    )

    if not results:
        print("No scenes found for this AOI/date range.")
        return []

    try:
        safe_results = validator.run_preflight(
            results, aoi, start_date=start_date, end_date=end_date
        )
    except OpticalValidationError as exc:
        # run_preflight itself doesn't raise per-scene -- this branch
        # exists for the case where you call validate_bands() or
        # another check directly (e.g. pre-checking one specific
        # scene before committing to a full search), where a hard
        # failure IS raised rather than filtered.
        print(f"Scene {exc.scene_id} rejected: {exc.reason}")
        return []

    if not safe_results:
        print(
            f"All {len(results)} candidate scene(s) failed validation -- "
            "see logged warnings/errors above for why."
        )
        return []

    print(f"{len(safe_results)}/{len(results)} scenes passed preflight; downloading...")
    return pf.download(safe_results, destination=destination)


if __name__ == "__main__":
    from shapely.geometry import box

    nyc_aoi = box(-74.1, 40.6, -73.7, 40.9)

    # Default config: standard checks, cloud cover as a warning.
    results = fetch_optical_stack(nyc_aoi, "2024-06-01", "2024-08-01")

    # Stricter config: reject cloudy scenes outright, require exact
    # snow-free scenes, and demand very high AOI coverage.
    strict_config = OpticalValidationConfig(
        max_cloud_cover_pct=10.0,
        cloud_cover_is_hard_failure=True,
        check_snow_ice_cover=True,
        max_snow_ice_pct=5.0,
        min_coverage_ratio=0.95,
    )
    strict_results = fetch_optical_stack(
        nyc_aoi, "2024-06-01", "2024-08-01", validation_config=strict_config
    )
