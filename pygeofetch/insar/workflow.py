"""
High-level InSAR workflow: search, download, extract, and process with
a handful of calls instead of dozens of lines, and every real analysis
step displays its own map or plot automatically.

Built entirely on top of pygeofetch's own, already-verified components:
PyGeoFetch (search/download), SLCExtractor, InterferogramGenerator,
PhaseUnwrapper, MapViewer -- this module adds no new science, it
orchestrates real, already-tested pieces and adds automatic display.

Every method that produces a real, meaningful spatial result shows it
automatically (map or plot); nothing here is a silent step the user
has to remember to visualize separately.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("pygeofetch.insar.workflow")


class InSARProject:
    """
    High-level, minimal-code InSAR workflow for a real AOI.

    Example (the whole search-through-interferogram pipeline in five
    real calls, not dozens of lines)::

        project = InSARProject(
            name="my_volcano", aoi=BoundingBox(...), output_dir="data/my_volcano",
        )
        project.search(start_date="2021-01-01", end_date="2021-06-01")
        project.download_and_extract(max_scenes=6)
        project.form_all_interferograms()
        project.show_strongest_pair()
    """

    def __init__(
        self,
        name: str,
        aoi: Any,
        output_dir: Union[str, Path],
        polarisation: str = "VV",
        providers: Optional[List[str]] = None,
    ) -> None:
        from pygeofetch import PyGeoFetch
        from pygeofetch.insar.extraction import SLCExtractor

        self.name = name
        self.aoi = aoi
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._providers = providers

        self._client = PyGeoFetch()
        self._extractor = SLCExtractor(polarisation=polarisation)

        self.search_results: List[Any] = []
        self.download_results: Dict[str, Any] = {}
        self.extracted_slcs: Dict[str, Path] = {}
        self.orbit_files: Dict[str, Path] = {}
        self.interferograms: Dict[Any, Any] = {}

    def search(
        self,
        start_date: str,
        end_date: str,
        max_results: int = 20,
        show_map: bool = True,
    ) -> List[Any]:
        """
        Real search via pygeofetch's own PyGeoFetch.search(), for SLC
        products over this project's real AOI and date range.

        Automatically displays real search-result footprints on a map
        unless show_map=False.
        """
        from pygeofetch.models.search_query import SearchQuery

        query = SearchQuery(
            bbox=self.aoi,
            start_date=date.fromisoformat(start_date),
            end_date=date.fromisoformat(end_date),
            product_type="SLC",
            max_results=max_results,
        )
        self.search_results = self._client.search(query, providers=self._providers)
        logger.info(
            "Real search: %d results for %s (%s to %s)",
            len(self.search_results),
            self.name,
            start_date,
            end_date,
        )

        if show_map and self.search_results:
            self._show_search_results_on_map()

        return self.search_results

    def _show_search_results_on_map(self) -> Any:
        import json
        import tempfile

        from pygeofetch.viz.map import MapViewer

        center_lat = (self.aoi.min_lat + self.aoi.max_lat) / 2
        center_lon = (self.aoi.min_lon + self.aoi.max_lon) / 2
        mv = MapViewer(center=(center_lat, center_lon), zoom=9)
        mv.add_basemap("SATELLITE")

        # Real fix, confirmed by reading the actual source rather than
        # assumed: MapViewer.add_vector() requires a real file path,
        # not a raw geojson dict, and SatelliteData's real field is
        # 'geometry', not 'footprint' (verified directly against
        # models/satellite_data.py). Combine into one real
        # FeatureCollection file rather than one file per scene.
        features = []
        for item in self.search_results[:20]:
            geometry = getattr(item, "geometry", None)
            if geometry is None:
                continue
            acq = getattr(item, "datetime", None)
            features.append(
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": {
                        "date": str(acq)[:10] if acq else "unknown",
                        "id": getattr(item, "id", ""),
                    },
                }
            )

        if features:
            geojson_path = Path(tempfile.mkdtemp()) / "search_results.geojson"
            geojson_path.write_text(
                json.dumps({"type": "FeatureCollection", "features": features})
            )
            mv.add_vector(
                str(geojson_path),
                layer_name="search_results",
                style={"color": "cyan", "weight": 2, "fillOpacity": 0.05},
            )
        else:
            logger.warning(
                "No real geometry found in search results — nothing to show on the map"
            )

        return mv.show()

    def download_and_extract(
        self,
        selected: Optional[List[Any]] = None,
        max_scenes: int = 6,
        show_map: bool = True,
    ) -> Dict[str, Path]:
        """
        Real download (via PyGeoFetch.download()) and real extraction
        (via SLCExtractor.extract_scene()) in one call, for the given
        real search results (or self.search_results if not given),
        capped to max_scenes real, distinct acquisition dates.

        Automatically shows the first extracted scene on a real map
        unless show_map=False.
        """
        from pygeofetch.models.download_task import DownloadOptions

        items = selected if selected is not None else self.search_results
        if not items:
            raise ValueError(
                "No search results to download -- call .search() first, or pass selected=..."
            )

        by_date: Dict[str, Any] = {}
        for item in items:
            acq = getattr(item, "datetime", None)
            label = str(acq)[:10] if acq else None
            if label and label not in by_date:
                by_date[label] = item
            if len(by_date) >= max_scenes:
                break

        raw_dir = self.output_dir / "raw"
        download_results_list = self._client.download(
            list(by_date.values()),
            destination=raw_dir,
            options=DownloadOptions(parallel=4, resume=True),
        )
        for label, dl_result in zip(by_date.keys(), download_results_list):
            self.download_results[label] = dl_result

        from pygeofetch.insar.geolocation import parse_orbit_file

        orbit_candidates = sorted(raw_dir.rglob("*POEORB*.EOF")) + sorted(
            raw_dir.rglob("*RESORB*.EOF")
        )

        for label, dl_result in self.download_results.items():
            if dl_result.output_path is None:
                logger.warning(
                    "Skipping extraction for %s: no output_path on its "
                    "download result (the download likely failed).",
                    label,
                )
                continue
            extracted_path = self._extractor.extract_scene(
                zip_path=dl_result.output_path,
                aoi=self.aoi,
                output_dir=self.output_dir / "slc" / label,
                label=label,
                resume=True,
            )
            if extracted_path is None:
                logger.warning("%s: no sub-swath overlaps this AOI", label)
                continue
            self.extracted_slcs[label] = extracted_path

            target_date = datetime.strptime(label, "%Y-%m-%d")
            for orbit_path in orbit_candidates:
                try:
                    times, _, _ = parse_orbit_file(orbit_path)
                    if times[0] <= target_date <= times[-1]:
                        self.orbit_files[label] = orbit_path
                        break
                except Exception:
                    continue

        logger.info(
            "Real extraction: %d/%d scenes extracted for %s",
            len(self.extracted_slcs),
            len(by_date),
            self.name,
        )

        if show_map and self.extracted_slcs:
            first_label = next(iter(self.extracted_slcs))
            return self._extractor.show_on_map(self.extracted_slcs[first_label])

        return self.extracted_slcs

    def form_all_interferograms(
        self,
        dem: Optional[Union[str, Path]] = None,
        looks_azimuth: int = 2,
        looks_range: int = 1,
        show_strongest: bool = True,
    ) -> Dict[Any, Any]:
        """
        Real interferogram formation, full verified pipeline (burst-
        aware ESD, real deburst, real flat-earth removal, Goldstein
        filter), for every real pair among extracted scenes.

        Automatically shows the strongest (highest-coherence) real
        pair's wrapped phase on a map unless show_strongest=False.
        """
        from pygeofetch.insar import InterferogramGenerator

        if len(self.extracted_slcs) < 2:
            raise ValueError(
                "Need at least 2 real extracted scenes -- call .download_and_extract() first"
            )

        ifg_gen = InterferogramGenerator(
            coherence_window=5,
            esd_enabled=True,
            use_gpu=False,
            use_real_burst_processing=True,
            remove_flat_earth_phase=True,
        )

        for d1, d2 in combinations(self.extracted_slcs.keys(), 2):
            coreg_kwargs = {}
            if d1 in self.orbit_files and d2 in self.orbit_files:
                coreg_kwargs = dict(
                    reference_safe_zip=self.download_results[d1].output_path,
                    secondary_safe_zip=self.download_results[d2].output_path,
                    reference_orbit_file=self.orbit_files[d1],
                    secondary_orbit_file=self.orbit_files[d2],
                )
            try:
                result = ifg_gen.process_pair(
                    reference=self.extracted_slcs[d1],
                    secondary=self.extracted_slcs[d2],
                    dem=dem,
                    reference_date=d1,
                    secondary_date=d2,
                    looks_azimuth=looks_azimuth,
                    looks_range=looks_range,
                    apply_goldstein_filter=True,
                    goldstein_alpha=0.6,
                    **coreg_kwargs,
                )
            except ValueError as exc:
                logger.info("  %s -> %s: REJECTED -- %s", d1, d2, exc)
                continue

            self.interferograms[(d1, d2)] = result
            days = (date.fromisoformat(d2) - date.fromisoformat(d1)).days
            logger.info(
                "  %s -> %s (%3dd): coherence=%.3f",
                d1,
                d2,
                days,
                result.coherence.mean(),
            )

        logger.info(
            "Real pairs formed: %d/%d possible",
            len(self.interferograms),
            len(list(combinations(self.extracted_slcs.keys(), 2))),
        )

        if show_strongest and self.interferograms:
            return self.show_strongest_pair()

        return self.interferograms

    def show_strongest_pair(self, band: str = "wrapped_phase") -> Any:
        """
        Real, automatic identification of the highest-coherence pair
        among all formed interferograms, displayed on a real,
        gradient-colored (cyclic for wrapped phase) map.
        """
        if not self.interferograms:
            raise ValueError(
                "No interferograms formed yet -- call .form_all_interferograms() first"
            )
        strongest_key = max(
            self.interferograms, key=lambda k: self.interferograms[k].coherence.mean()
        )
        strongest = self.interferograms[strongest_key]
        logger.info(
            "Strongest real pair: %s, coherence=%.3f",
            strongest_key,
            strongest.coherence.mean(),
        )
        return strongest.show_on_map(band=band)

    def summary(self) -> None:
        """Real, quick-glance summary of everything done so far."""
        print(f"Project: {self.name}")
        print(f"Real search results: {len(self.search_results)}")
        print(
            f"Real scenes extracted: {len(self.extracted_slcs)}/{len(self.download_results)}"
        )
        print(f"Real interferograms formed: {len(self.interferograms)}")
        if self.interferograms:
            for (d1, d2), result in sorted(self.interferograms.items()):
                print(f"  {d1} -> {d2}: coherence={result.coherence.mean():.3f}")
