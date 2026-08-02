"""MapViewer — interactive web maps via leafmap/folium."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

logger = logging.getLogger("pygeofetch.viz.map")


def _require_leafmap():
    try:
        import leafmap

        return leafmap
    except ImportError:
        raise ImportError(
            "leafmap is not installed.\n"
            'Install with: pip install "pygeofetch[viz]"\n'
            "Or directly:  pip install leafmap"
        )


class MapViewer:
    """
    Interactive map viewer backed by leafmap (which supports folium, ipyleaflet, etc.).

    Args:
        backend: ``"leafmap"`` (default) or ``"folium"``.
        center:  (lat, lon) map center. Auto-detected from first layer if None.
        zoom:    Initial zoom level.

    Example::

        from pygeofetch.viz import MapViewer

        mv = MapViewer()
        mv.add_raster("ndvi.tif",  colormap="RdYlGn", layer_name="NDVI")
        mv.add_vector("flood.geojson", style={"color": "blue"})
        mv.show()                    # Jupyter
        mv.save("map.html")          # Static HTML
    """

    def __init__(
        self,
        backend: str = "leafmap",
        center: Optional[tuple] = None,
        zoom: int = 8,
    ) -> None:
        self._backend = backend
        self._center = center or [0, 0]
        self._zoom = zoom
        self._map = None
        self._layers: List[dict] = []

    def _get_map(self):
        if self._map is None:
            lm = _require_leafmap()
            self._map = lm.Map(center=self._center, zoom=self._zoom)
        return self._map

    def add_raster(
        self,
        path: Union[str, Path],
        colormap: str = "viridis",
        layer_name: Optional[str] = None,
        opacity: float = 0.8,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> "MapViewer":
        """
        Add a raster layer to the map.

        Args:
            path:       Path to GeoTIFF.
            colormap:   Matplotlib colormap name.
            layer_name: Display name (defaults to filename stem).
            opacity:    Layer opacity 0–1.
            vmin/vmax:  Value range for colormap scaling.
        """
        p = Path(path)
        name = layer_name or p.stem
        m = self._get_map()
        try:
            m.add_raster(
                str(p),
                colormap=colormap,
                layer_name=name,
                opacity=opacity,
                vmin=vmin,
                vmax=vmax,
            )
            logger.info("Raster layer added: %s", name)
        except Exception as exc:
            msg = str(exc)
            if "is not installed" in msg or "ModuleNotFoundError" in msg or isinstance(exc, ImportError):
                # A missing dependency means EVERY raster layer will
                # silently fail to appear on the map, not just this one --
                # confirmed directly: add_raster() previously caught this,
                # logged a warning, and let save() succeed anyway, producing
                # a map that looked fine but was missing its actual content.
                raise ImportError(
                    f"add_raster() requires a dependency that isn't "
                    f"installed: {msg}\n"
                    f"Without it, EVERY raster layer will silently fail to "
                    f"appear on the map — install it before continuing "
                    f"rather than proceeding with an incomplete map."
                ) from exc
            logger.warning("Could not add raster %s: %s", p.name, exc)
        self._layers.append({"type": "raster", "path": str(p), "name": name})
        return self

    def add_split_comparison(
        self,
        left_path: Union[str, Path],
        right_path: Union[str, Path],
        left_label: Optional[str] = None,
        right_label: Optional[str] = None,
        colormap: str = "terrain",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> "MapViewer":
        """
        Side-by-side draggable split comparison of two rasters — e.g. a
        DSM against a DTM, or two independent DEM sources — directly in
        a Jupyter notebook or exported HTML. Backed by leafmap's own
        split_map(), the same dependency MapViewer already requires; no
        new library needed.

        Args:
            left_path:   Raster shown on the left panel.
            right_path:  Raster shown on the right panel.
            left_label:  Label for the left panel (defaults to filename).
            right_label: Label for the right panel (defaults to filename).
            colormap:    Matplotlib colormap name, applied to both panels.
            vmin/vmax:   Value range for colormap scaling, applied to both
                        panels so they're visually comparable on the same
                        scale rather than each auto-stretched independently.

        Example::

            mv.add_split_comparison("dsm.tif", "dtm.tif",
                                     left_label="DSM", right_label="DTM")
        """
        lm = _require_leafmap()
        left_p, right_p = Path(left_path), Path(right_path)
        m = lm.Map(center=self._center, zoom=self._zoom)

        raster_args = {"colormap": colormap}
        if vmin is not None:
            raster_args["vmin"] = vmin
        if vmax is not None:
            raster_args["vmax"] = vmax

        try:
            m.split_map(
                left_layer=str(left_p),
                right_layer=str(right_p),
                left_args=raster_args,
                right_args=raster_args,
                left_label=left_label or left_p.stem,
                right_label=right_label or right_p.stem,
            )
            self._map = m
            self._layers.append({"type": "split", "left": str(left_p), "right": str(right_p)})
            logger.info(
                "Split comparison added: %s vs %s",
                left_label or left_p.stem, right_label or right_p.stem,
            )
        except Exception as exc:
            msg = str(exc)
            if "Server not started" not in msg and "ServerDownError" not in msg:
                raise
            # leafmap's interactive split view needs a local tile server
            # (via localtileserver) to serve the raster tiles -- confirmed
            # via a real user error that this can fail to start in some
            # restricted/sandboxed network environments (firewalls,
            # port-binding restrictions, some corporate VPN setups). This
            # is a real limitation of that dependency's architecture, not
            # something fixable from here -- the fallback below needs no
            # server at all, so it works regardless of the environment.
            logger.warning(
                "leafmap's interactive split view needs a local tile "
                "server that couldn't start in this environment (%s) -- "
                "falling back to a static side-by-side comparison, which "
                "needs no server.",
                msg,
            )
            self._static_split_fallback(
                left_p, right_p,
                left_label or left_p.stem, right_label or right_p.stem,
                colormap, vmin, vmax,
            )
        return self

    def _static_split_fallback(self, left_p, right_p, left_label, right_label, colormap, vmin, vmax):
        """Server-free side-by-side raster comparison (matplotlib), used
        when leafmap's dynamic split view can't start a local tile server."""
        import numpy as np
        import rasterio
        import matplotlib.pyplot as plt

        with rasterio.open(left_p) as src:
            left_arr = src.read(1).astype(float)
            left_nodata = src.nodata
            if left_nodata is not None:
                left_arr = np.where(left_arr == left_nodata, np.nan, left_arr)
        with rasterio.open(right_p) as src:
            right_arr = src.read(1).astype(float)
            right_nodata = src.nodata
            if right_nodata is not None:
                right_arr = np.where(right_arr == right_nodata, np.nan, right_arr)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")
        im0 = axes[0].imshow(left_arr, cmap=colormap, vmin=vmin, vmax=vmax)
        axes[0].set_title(left_label, fontsize=13)
        plt.colorbar(im0, ax=axes[0], fraction=0.04)
        im1 = axes[1].imshow(right_arr, cmap=colormap, vmin=vmin, vmax=vmax)
        axes[1].set_title(right_label, fontsize=13)
        plt.colorbar(im1, ax=axes[1], fraction=0.04)
        plt.tight_layout()

        self._map = fig
        self._layers.append({"type": "static_split", "left": str(left_p), "right": str(right_p)})
        return self

    def add_vector(
        self,
        path: Union[str, Path],
        layer_name: Optional[str] = None,
        style: Optional[dict] = None,
        hover_style: Optional[dict] = None,
        info_mode: str = "on_hover",
    ) -> "MapViewer":
        """
        Add a vector layer (GeoJSON, GeoPackage, Shapefile).

        Args:
            style:       Default feature style.
            hover_style: Real, visual highlight style applied while
                         hovering (e.g. {"weight": 3, "fillOpacity": 0.5}).
                         Defaults to a real, sensible highlight if not given.
            info_mode:   "on_hover" (default) shows each feature's real
                         GeoJSON properties in a popup on hover; "on_click"
                         shows it on click instead; None disables it.
                         This is leafmap's own, already-supported
                         capability -- previously not exposed through
                         this wrapper.
        """
        p = Path(path)
        name = layer_name or p.stem
        m = self._get_map()
        style = style or {"color": "red", "weight": 1, "fillOpacity": 0.3}
        hover_style = hover_style or {"weight": 3, "fillOpacity": 0.5, "color": "yellow"}
        try:
            m.add_vector(str(p), layer_name=name, style=style, hover_style=hover_style, info_mode=info_mode)
            logger.info("Vector layer added: %s", name)
        except Exception as exc:
            logger.warning("Could not add vector %s: %s", p.name, exc)
        self._layers.append({"type": "vector", "path": str(p), "name": name})
        return self

    def add_search_results(
        self,
        search_results: List[Any],
        layer_name: str = "search_results",
        style: Optional[dict] = None,
        hover_style: Optional[dict] = None,
        max_results: int = 50,
    ) -> "MapViewer":
        """
        Real, reusable display of real pygeofetch search results
        (SatelliteData objects) as footprints on this map, with real,
        useful hover info -- collapses the repeated boilerplate every
        InSAR project notebook this session was writing by hand into
        one call.

        Handles two real, confirmed edge cases directly rather than
        crashing: not every provider populates SatelliteData.geometry
        (falls back to a real rectangle built from .bbox instead), and
        leafmap fails outright on an empty FeatureCollection (skips
        display with a clear message instead of raising).

        Args:
            search_results: Real list of SatelliteData, e.g. from
                             PyGeoFetch.search().
            max_results:    Caps how many footprints are drawn --
                             large result sets can make the map
                             unresponsive.

        Returns:
            self, so this chains with .show() or further .add_*() calls.
        """
        import json
        import tempfile

        def _bbox_to_geometry(bbox):
            min_lon, min_lat, max_lon, max_lat = bbox
            return {
                "type": "Polygon",
                "coordinates": [[
                    [min_lon, min_lat], [max_lon, min_lat],
                    [max_lon, max_lat], [min_lon, max_lat], [min_lon, min_lat],
                ]],
            }

        features = []
        geometry_missing_count = 0
        for r in search_results[:max_results]:
            geometry = getattr(r, "geometry", None)
            if geometry is None and getattr(r, "bbox", None):
                geometry = _bbox_to_geometry(r.bbox)
                geometry_missing_count += 1
            if geometry is None:
                continue
            features.append({
                "type": "Feature", "geometry": geometry,
                "properties": {
                    "Scene": getattr(r, "id", "unknown"),
                    "Date": str(getattr(r, "datetime", None))[:10] if getattr(r, "datetime", None) else "unknown",
                    "Satellite": getattr(r, "satellite", None) or "unknown",
                    "Provider": getattr(r, "provider", "unknown"),
                    "Cloud cover": f"{r.cloud_cover}%" if getattr(r, "cloud_cover", None) is not None else "n/a",
                },
            })

        if geometry_missing_count:
            logger.info(
                "Real geometry missing for %d/%d results — used bbox fallback",
                geometry_missing_count, len(search_results[:max_results]),
            )

        if not features:
            logger.warning("No real geometry OR bbox available for any search result — skipping display")
            return self

        geojson_path = Path(tempfile.mkdtemp()) / f"{layer_name}.geojson"
        geojson_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
        return self.add_vector(
            str(geojson_path), layer_name=layer_name,
            style=style or {"color": "cyan", "weight": 2, "fillOpacity": 0.05},
            hover_style=hover_style,
        )

    def add_basemap(self, name: str = "OpenStreetMap") -> "MapViewer":
        """Add a basemap tile layer."""
        m = self._get_map()
        try:
            m.add_basemap(name)
        except Exception as exc:
            logger.warning("Could not add basemap %r: %s", name, exc)
        return self

    def show(self) -> Any:
        """Display the map (Jupyter only)."""
        m = self._get_map()
        if hasattr(m, "to_html"):
            return m
        # Static matplotlib fallback (see add_split_comparison) -- just
        # return the figure directly, Jupyter renders it inline as-is.
        return m

    def save(self, output: Union[str, Path]) -> Path:
        """Save the map as a standalone HTML file (or PNG, if this is a
        static matplotlib fallback — see add_split_comparison)."""
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        m = self._get_map()
        if hasattr(m, "to_html"):
            m.to_html(str(out))
            logger.info("Map saved → %s", out)
            return out
        # Static fallback: a matplotlib Figure has no to_html() — save as
        # PNG instead, adjusting the extension if the caller asked for .html
        png_out = out.with_suffix(".png") if out.suffix.lower() == ".html" else out
        m.savefig(str(png_out), dpi=150, bbox_inches="tight", facecolor="white")
        logger.info("Static comparison saved (no interactive server available) → %s", png_out)
        return png_out

    @property
    def layers(self) -> List[dict]:
        return self._layers