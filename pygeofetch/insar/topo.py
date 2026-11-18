"""Topographic phase from the DEM and the two orbits (simulation), applied together with the flat-earth phase.

What was wrong before
---------------------
``interferogram._remove_topographic_phase*`` regressed the interferogram phase against DEM height and removed the fit
only when R2 >= 0.5. That does not remove topographic phase: it removes whatever phase correlates with height, which
includes atmosphere (stratified delay) and deformation on a slope, and it removes *nothing* where the correlation is
weak, which is where the topographic phase is small relative to noise but still not zero. SNAP, ISCE and GAMMA instead
*simulate* the phase from the DEM and the orbits and subtract it regardless of how well it fits.

What this does
--------------
For every output pixel, with reference (time, range) from ``radar_geometry.PixelFrame``:

1. ground position of the pixel on the terrain: start on the ellipsoid, then iterate "height at this lon/lat from the DEM ->
   where the same (time, range) lands at that height" (positions move by ``dlon_dh``, ``dlat_dh`` from the nodes);
2. geometric phase = ``phase_flat + k1*h + 0.5*k2*h**2`` with ``phase_flat``, ``k1``, ``k2`` from the exact orbit geometry on
   the node grid (interpolated);
3. subtract it: the interferogram is multiplied by ``exp(-1j*phase)``.

Accuracy (synthetic geometry, 27-30 degrees incidence, 53 m baseline, tested in this repository's scratch tests): the
quadratic in height reproduces the exact phase to 3e-5 rad at 100 m, below 1e-5 rad at 500-1000 m, and 2e-3 rad at
2000 m; the linear term alone would be off by 0.14 rad at 1000 m. Flat-earth interpolation error is 2e-4 rad on a 9 x 9 grid.

Limits, stated
--------------
* The DEM height is used as the ellipsoidal height. Most DEMs are orthometric (the Copernicus DEM uses EGM2008); the geoid
  undulation (about +40 m in Silesia) then shifts the phase by ``k1 * N``, a nearly constant offset that does not matter
  after referencing to a pixel, and shifts the geocoding by about ``N / tan(incidence)``. Pass ``geoid_offset_m`` to correct
  both.
* Layover and shadow have no unique height; those pixels are flagged and get the flat-earth phase only.
* The result is only as good as the DEM and the geocoding. A DEM error ``dh`` leaves ``k1 * dh`` of residual phase
  (about 0.04 rad per metre at 50 m baseline), proportional to the baseline: that is what the stack-level DEM-error
  estimate in ``atmosphere.estimate_height_dependent_per_date`` tests for.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from pygeofetch.insar.radar_geometry import GeometryNodes

logger = logging.getLogger("pygeofetch.insar.topo")


class DemSampler:
    """Bilinear DEM heights at arbitrary lon/lat, on any DEM CRS (reprojected on the fly to WGS84)."""

    def __init__(self, path: Union[str, Path], geoid_offset_m: float = 0.0):
        import rasterio
        from rasterio.vrt import WarpedVRT
        from rasterio.enums import Resampling

        self._src = rasterio.open(str(path))
        ds = self._src
        if ds.crs is not None and not ds.crs.is_geographic:
            ds = WarpedVRT(ds, crs="EPSG:4326", resampling=Resampling.bilinear)
            self._vrt = ds
        self._ds = ds
        self.transform = ds.transform
        self.height, self.width = ds.height, ds.width
        self.nodata = ds.nodata
        self.geoid_offset_m = float(geoid_offset_m)
        if abs(self.transform.b) > 1e-12 or abs(self.transform.d) > 1e-12:
            raise ValueError("a rotated DEM grid is not supported; reproject the DEM to a north-up grid")

    def close(self):
        for d in (getattr(self, "_vrt", None), self._src):
            try:
                if d is not None:
                    d.close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        t = self.transform
        return (t.c, t.f + t.e * self.height, t.c + t.a * self.width, t.f)   # left, bottom, right, top (north-up)

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Heights (metres, float64) at the points; NaN outside the DEM or where it has no data."""
        from scipy.ndimage import map_coordinates
        from rasterio.windows import Window

        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        t = self.transform
        col = (lon - t.c) / t.a - 0.5
        row = (lat - t.f) / t.e - 0.5
        inside = (col >= 0) & (col <= self.width - 1) & (row >= 0) & (row <= self.height - 1) & np.isfinite(col) & np.isfinite(row)
        out = np.full(lon.shape, np.nan)
        if not inside.any():
            return out
        r0 = int(np.floor(row[inside].min()))
        r1 = int(np.ceil(row[inside].max())) + 2
        c0 = int(np.floor(col[inside].min()))
        c1 = int(np.ceil(col[inside].max())) + 2
        r1, c1 = min(r1, self.height), min(c1, self.width)
        data = self._ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float64)
        if self.nodata is not None:
            data[data == self.nodata] = np.nan
        data[data < -500.0] = np.nan
        vals = map_coordinates(data, [row[inside] - r0, col[inside] - c0], order=1, mode="nearest")
        out[inside] = vals + self.geoid_offset_m
        return out


def terrain_heights(
    lon0: np.ndarray,
    lat0: np.ndarray,
    dlon: np.ndarray,
    dlat: np.ndarray,
    dem: DemSampler,
    n_iter: int = 10,
    tolerance_m: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Terrain height of every pixel at the position where its (time, range) meets the terrain.

    ``lon0``/``lat0`` are the ellipsoid ground positions and ``dlon``/``dlat`` how they move per metre of height
    (``GeometryNodes`` fields, evaluated on the pixels of interest). Returns ``(height_m, converged)``; height is NaN
    outside the DEM. ``converged`` is False where the iteration had not settled to ``tolerance_m``.
    """
    h = dem.sample(lon0, lat0)
    h = np.where(np.isfinite(h), h, np.nan)
    converged = np.zeros(h.shape, dtype=bool)
    for _ in range(n_iter):
        hh = np.where(np.isfinite(h), h, 0.0)
        h_new = dem.sample(lon0 + dlon * hh, lat0 + dlat * hh)
        converged = np.isfinite(h) & np.isfinite(h_new) & (np.abs(h_new - h) < tolerance_m)
        h = h_new
        # Stop when every pixel that has a height has settled to the tolerance. The loop used to run all
        # n_iter passes even when the first two had already converged; each pass samples the DEM at every pixel.
        n_fin = int(np.isfinite(h).sum())
        if n_fin and int(converged.sum()) >= n_fin:
            break
    return h, converged


def _slope_mask(h: np.ndarray, theta_deg: np.ndarray, spacing_m: float) -> np.ndarray:
    """True where the terrain is in layover or shadow for this look direction.

    Slope is taken along the columns (range direction) and converted to ground range with the incidence angle. A slope
    rising toward the sensor steeper than the incidence is layover; falling away steeper than 90 - incidence is shadow.
    """
    g = spacing_m / np.maximum(np.sin(np.radians(theta_deg)), 0.2)
    with np.errstate(invalid="ignore"):
        dh = np.gradient(h, axis=1) / g
        toward = -np.degrees(np.arctan(dh))        # positive: terrain tilted toward the sensor
        local = theta_deg - toward
    return ~np.isfinite(local) | (local < 1.0) | (local > 89.0)


def apply_geometric_phase(
    interferogram: np.ndarray,
    nodes: GeometryNodes,
    dem: Optional[Union[str, Path, DemSampler]] = None,
    *,
    remove_flat: bool = True,
    remove_topo: bool = True,
    geoid_offset_m: float = 0.0,
    looks: Tuple[int, int] = (1, 1),
    row_block: Optional[int] = None,
    return_phase: bool = False,
    inplace: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Remove the flat-earth and the simulated topographic phase from ``interferogram`` (a complex array, new array returned).

    ``interferogram`` is at *full resolution*: ``nodes.shape * looks`` pixels. The geometric phase varies by radians between
    neighbouring pixels on steep terrain, so it has to come off before the interferogram is multilooked (averaging complex
    values that still carry it would decorrelate them). ``nodes`` are defined on the multilooked output grid; ``looks`` is
    the multilook factor that will be applied afterwards.

    ``row_block`` is the number of full-resolution rows handled at a time (None: sized so that one float64 field holds about
    2.5 million pixels, whatever the scene width, since a dozen such fields are alive per block). ``inplace=True`` writes
    the corrected values into ``interferogram`` itself and returns it, saving one full-size complex array; the caller must
    not need the uncorrected values afterwards.

    Returns ``(corrected, info)``. ``info`` holds the fraction of pixels with a DEM height, the layover/shadow fraction, the
    height and topographic-phase ranges and, with ``return_phase=True``, the removed ``phase`` array and the ``terrain_valid`` mask.
    """
    la, lr = looks
    h, w = interferogram.shape
    if (h, w) != (nodes.shape[0] * la, nodes.shape[1] * lr):
        raise ValueError(
            f"interferogram {interferogram.shape} does not match the geometry grid {nodes.shape} x looks {looks}; "
            "trim the interferogram to whole multilook windows first")
    own = False
    sampler: Optional[DemSampler]
    if dem is None or not remove_topo:
        sampler = None
    elif isinstance(dem, DemSampler):
        sampler = dem
    else:
        sampler = DemSampler(dem, geoid_offset_m)
        own = True
    if row_block is None:
        row_block = max(16, int(2_500_000 // max(w, 1)))
    if inplace and interferogram.dtype == np.complex64 and interferogram.flags.writeable:
        out = interferogram
    else:
        out = np.empty_like(interferogram, dtype=np.complex64)
    phase_all = np.empty((h, w), dtype=np.float32) if return_phase else None
    valid_all = np.zeros((h, w), dtype=bool)
    layover_all = np.zeros((h, w), dtype=bool)
    hmin, hmax, tmin, tmax = np.inf, -np.inf, np.inf, -np.inf
    n_dem = n_conv = n_lay = 0
    try:
        for r0 in range(0, h, row_block):
            rs = slice(r0, min(h, r0 + row_block))
            def fld(name):
                return nodes.dense_native(name, rs, looks, np.float64)

            phase = fld("phase_flat") if remove_flat else np.zeros(interferogram[rs].shape)
            if sampler is not None:
                ht, conv = terrain_heights(fld("lon"), fld("lat"), fld("dlon_dh"), fld("dlat_dh"), sampler)
                theta = fld("theta_deg")
                lay = _slope_mask(ht, theta, nodes.frame.pixel_spacing_range_m() / lr)
                ok = np.isfinite(ht) & ~lay
                k1 = fld("k1")
                k2 = fld("k2")
                hz = np.where(ok, ht, 0.0)
                topo = k1 * hz + 0.5 * k2 * hz * hz
                phase = phase + topo
                valid_all[rs] = ok
                layover_all[rs] = lay & np.isfinite(ht)
                n_dem += int(np.isfinite(ht).sum())
                n_conv += int((conv & ok).sum())
                n_lay += int((lay & np.isfinite(ht)).sum())
                if ok.any():
                    hmin, hmax = min(hmin, float(ht[ok].min())), max(hmax, float(ht[ok].max()))
                    tmin, tmax = min(tmin, float(topo[ok].min())), max(tmax, float(topo[ok].max()))
            out[rs] = (interferogram[rs] * np.exp(-1j * phase)).astype(np.complex64)
            if phase_all is not None:
                phase_all[rs] = phase.astype(np.float32)
    finally:
        if own and sampler is not None:
            sampler.close()
    n = h * w
    info: Dict[str, Any] = {
        "flat_earth_removed": bool(remove_flat),
        "topographic_phase_simulated": sampler is not None,
        "dem_coverage_fraction": float(n_dem / n) if sampler is not None else None,
        "terrain_converged_fraction": float(n_conv / n) if sampler is not None else None,
        "layover_shadow_fraction": float(n_lay / n) if sampler is not None else None,
        "height_range_m": (hmin, hmax) if sampler is not None and np.isfinite(hmin) else None,
        "topographic_phase_range_rad": (tmin, tmax) if sampler is not None and np.isfinite(tmin) else None,
        "bperp_mean_m": nodes.summary()["bperp_mean_m"],
    }
    if sampler is not None and n_dem == 0:
        raise ValueError("the DEM does not cover any pixel of the interferogram; check the DEM extent and CRS")
    if return_phase:
        info["phase"] = phase_all
        info["terrain_valid"] = valid_all
        info["layover_shadow"] = layover_all
    return out, info


def heights_on_output_grid(
    nodes: GeometryNodes,
    dem: Union[str, Path, DemSampler],
    geoid_offset_m: float = 0.0,
    row_block: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Terrain height and a validity mask (DEM present, not layover/shadow) on the multilooked output grid of ``nodes``.

    This is the height every correction that depends on terrain should use (atmospheric height dependence, masks), so that
    all of them agree with the topographic phase that was removed.
    """
    own = False
    sampler = dem if isinstance(dem, DemSampler) else DemSampler(dem, geoid_offset_m)
    own = not isinstance(dem, DemSampler)
    h, w = nodes.shape
    out = np.full((h, w), np.nan, dtype=np.float32)
    ok = np.zeros((h, w), dtype=bool)
    try:
        for r0 in range(0, h, row_block):
            rs = slice(r0, min(h, r0 + row_block))
            ht, _ = terrain_heights(nodes.dense("lon", rs, np.float64), nodes.dense("lat", rs, np.float64),
                                    nodes.dense("dlon_dh", rs, np.float64), nodes.dense("dlat_dh", rs, np.float64), sampler)
            lay = _slope_mask(ht, nodes.dense("theta_deg", rs, np.float64), nodes.frame.pixel_spacing_range_m())
            out[rs] = ht
            ok[rs] = np.isfinite(ht) & ~lay
    finally:
        if own:
            sampler.close()
    return out, ok