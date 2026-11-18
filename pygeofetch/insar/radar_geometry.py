"""Per-pixel radar geometry for an interferogram grid.

Why this module exists
----------------------
Flat-earth phase, topographic phase and atmospheric corrections all need the same thing: for every pixel of the
*output* interferogram, where it is on the ground, how it is viewed, and how the two orbits differ there. The output
grid is not the SLC grid. It has been debursted, cropped to the AOI and multilooked, so a pixel index in the
interferogram is not a pixel index in the SLC. Computing a correction from the SLC geometry and evaluating it at
interferogram indices (what ``flatearth.compute_flat_earth_phase`` does) puts the correction in the wrong place by
the crop offset and, for multilooked products, at the wrong scale.

What it does (the approach of the range-Doppler simulators in ISCE, GAMMA and SNAP, done on a sparse grid)
-----------------------------------------------------------------------------------------------------------
1. ``PixelFrame`` maps an output pixel (row, column) to the reference SLC's azimuth time and slant range, applying the
   SLC extraction offset, the deburst offset, the AOI crop offset and the multilook window centre.
2. On a sparse grid of nodes (default 9 x 9) the exact range-Doppler geometry is solved from the two orbits:
   the ground point on the ellipsoid, the reference and secondary slant ranges, the geometric phase, the incidence angle,
   and how all of these change with terrain height at fixed reference (time, range).
3. Those node values are smooth, so they are interpolated with a bicubic spline to every pixel. Terrain height is *not*
   smooth, and is handled separately in ``topo.py`` with the DEM.

Sign convention (derived, not assumed)
--------------------------------------
Interferogram = reference * conj(secondary). The propagation phase of a return from range R is -4*pi*R/lambda, so
the interferogram phase is 4*pi/lambda * (R_secondary - R_reference). A longer secondary path (for example more
atmospheric delay on the secondary date) gives a *positive* interferogram phase. Everything here uses that.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from pygeofetch.insar.geolocation import (
    SPEED_OF_LIGHT,
    WGS84_A,
    WGS84_B,
    find_zero_doppler_time,
    interpolate_orbit_state,
    solve_ground_point,
)

logger = logging.getLogger("pygeofetch.insar.radar_geometry")

SENTINEL1_WAVELENGTH_M = 0.05546576


class GeometryError(RuntimeError):
    """The radar geometry of a pair could not be solved (orbit does not cover the scene, solver did not converge)."""


# ---------------------------------------------------------------------------------------------------
# coordinates
# ---------------------------------------------------------------------------------------------------
def ecef_to_geodetic(x, y, z):
    """WGS84 ECEF -> (lon_deg, lat_deg, height_m). Vectorised, Bowring-style fixed-point iteration (sub-mm in 6 passes)."""
    x, y, z = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
    a, b = WGS84_A, WGS84_B
    e2 = 1.0 - (b * b) / (a * a)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - e2))
    h = np.zeros_like(lat)
    for _ in range(6):
        n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - e2 * n / (n + h)))
    return np.degrees(lon), np.degrees(lat), h


def _geodetic_up(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon, lat = math.radians(lon_deg), math.radians(lat_deg)
    return np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])


# ---------------------------------------------------------------------------------------------------
# output pixel -> SLC time / range
# ---------------------------------------------------------------------------------------------------
@dataclass
class PixelFrame:
    """Where the pixels of an interferogram sit in the reference SLC.

    Args:
        shape: (rows, cols) of the interferogram the correction is for (after crop and multilook).
        geometry: the reference ``SLCGeometry`` (from ``annotation.parse_slc_geometry``).
        looks: (azimuth, range) multilook factors that were applied after the crop.
        slc_offset: (row, col) of the extracted SLC's pixel (0, 0) in the full burst-stack scene
            (``coregister.read_crop_offset``). (0, 0) for a whole sub-swath.
        crop_offset: (row, col) of the interferogram's full-resolution pixel (0, 0) in the SLC *after deburst*
            (``crop_info['row_off']``, ``crop_info['col_off']``); (0, 0) if there was no crop.
        deburst_first_kept_row: full-scene row of the first line kept by deburst, or None when the SLC was not
            debursted. After deburst the rows are contiguous in azimuth time, so row ``k`` of the debursted array is at
            ``time(first_kept_row) + k * dt``. Without deburst the burst-stack row is used (``SLCGeometry.azimuth_time``
            is burst-aware).
    """

    shape: Tuple[int, int]
    geometry: Any
    looks: Tuple[int, int] = (1, 1)
    slc_offset: Tuple[float, float] = (0.0, 0.0)
    crop_offset: Tuple[float, float] = (0.0, 0.0)
    deburst_first_kept_row: Optional[float] = None

    def _full_row(self, i: float) -> float:
        # centre of the i-th multilook window, in full-resolution pixel units, counted from the crop origin
        return self.crop_offset[0] + (i + 0.5) * self.looks[0] - 0.5

    def _full_col(self, j: float) -> float:
        return self.slc_offset[1] + self.crop_offset[1] + (j + 0.5) * self.looks[1] - 0.5

    def azimuth_time(self, i: float) -> datetime:
        g = self.geometry
        k = self._full_row(i)
        if self.deburst_first_kept_row is not None:
            t0 = g.azimuth_time(self.deburst_first_kept_row)
            return t0 + timedelta(seconds=k * g.azimuth_time_interval_s)
        return g.azimuth_time(self.slc_offset[0] + k)

    def slant_range_m(self, j: float) -> float:
        return float(self.geometry.slant_range_m(self._full_col(j)))

    def pixel_spacing_range_m(self) -> float:
        return SPEED_OF_LIGHT / (2.0 * self.geometry.range_sampling_rate_hz) * self.looks[1]


# ---------------------------------------------------------------------------------------------------
# node geometry
# ---------------------------------------------------------------------------------------------------
_FIELDS = ("lon", "lat", "dlon_dh", "dlat_dh", "phase_flat", "k1", "k2", "theta_deg", "range_m")


@dataclass
class GeometryNodes:
    """Exact range-Doppler geometry on a sparse grid of the output interferogram, plus dense interpolation.

    Fields (per node): ``lon``/``lat`` of the ellipsoid ground point; ``dlon_dh``/``dlat_dh`` how the ground position of the same
    (time, range) moves with terrain height; ``phase_flat`` = 4pi/lambda (R_sec - R_ref) on the ellipsoid; ``k1``, ``k2``: the
    first and second derivative of that phase with terrain height at fixed reference (time, range), so the topographic phase of
    a pixel at height h is ``k1*h + 0.5*k2*h**2``; ``theta_deg`` incidence on the ellipsoid; ``range_m`` reference slant range.
    """

    shape: Tuple[int, int]
    rows: np.ndarray
    cols: np.ndarray
    values: Dict[str, np.ndarray]
    wavelength_m: float
    frame: PixelFrame
    h_probe_m: float = 500.0
    _splines: Dict[str, Any] = field(default_factory=dict, repr=False)

    def _spline(self, name: str):
        if name not in self._splines:
            from scipy.interpolate import RectBivariateSpline

            kx = min(3, len(self.rows) - 1)
            ky = min(3, len(self.cols) - 1)
            self._splines[name] = RectBivariateSpline(self.rows, self.cols, self.values[name], kx=kx, ky=ky)
        return self._splines[name]

    def dense(self, name: str, row_slice: Optional[slice] = None, dtype=np.float32) -> np.ndarray:
        """The field at every output pixel (or the rows of ``row_slice``). Use ``dtype=np.float64`` for phase fields
        (thousands of radians do not survive float32 to better than ~1e-3 rad)."""
        h, w = self.shape
        rs = row_slice or slice(0, h)
        rows = np.arange(h, dtype=np.float64)[rs]
        cols = np.arange(w, dtype=np.float64)
        return self._spline(name)(rows, cols).astype(dtype)

    def dense_native(self, name: str, native_rows: slice, looks: Tuple[int, int] = (1, 1), dtype=np.float64) -> np.ndarray:
        """The field at the *full-resolution* pixels (before multilooking) of ``native_rows``.

        Output pixel ``i`` is the mean of native pixels ``i*looks .. (i+1)*looks - 1``, so native pixel ``k`` sits at output
        coordinate ``(k + 0.5)/looks - 0.5``. Columns cover ``shape[1] * looks[1]`` native pixels.
        """
        la, lr = looks
        _, w = self.shape
        rows = (np.arange(native_rows.start, native_rows.stop, dtype=np.float64) + 0.5) / la - 0.5
        cols = (np.arange(w * lr, dtype=np.float64) + 0.5) / lr - 0.5
        return self._spline(name)(rows, cols).astype(dtype)

    def bperp_m(self) -> np.ndarray:
        """Signed perpendicular baseline at each node (metres): k1 * R * sin(theta) * lambda / (4 pi)."""
        v = self.values
        return v["k1"] * v["range_m"] * np.sin(np.radians(v["theta_deg"])) * self.wavelength_m / (4.0 * np.pi)

    def summary(self) -> Dict[str, Any]:
        b = self.bperp_m()
        t = self.values["theta_deg"]
        return {
            "bperp_mean_m": float(b.mean()), "bperp_min_m": float(b.min()), "bperp_max_m": float(b.max()),
            "incidence_min_deg": float(t.min()), "incidence_max_deg": float(t.max()),
            "n_nodes": int(b.size),
        }


def _unit(v):
    return v / np.linalg.norm(v)


def _node(frame, ref_orbit, sec_orbit, ref_guess, sec_shift, i, j, wavelength_m, h_probe):
    t_ref = frame.azimuth_time(i)
    r_ref = frame.slant_range_m(j)
    s_ref, v_ref = interpolate_orbit_state(ref_orbit[0], ref_orbit[1], ref_orbit[2], t_ref)
    rt = 2.0 * r_ref / SPEED_OF_LIGHT

    out = {}
    pts = []
    for h in (0.0, h_probe, 2.0 * h_probe):
        p = solve_ground_point(tuple(s_ref), tuple(v_ref), rt, dem_height_m=h)
        pts.append(np.asarray(p, dtype=np.float64))

    phis = []
    for p in pts:
        t_sec = find_zero_doppler_time(sec_orbit[0], sec_orbit[1], sec_orbit[2], tuple(p), t_ref + sec_shift)
        s_sec, _ = interpolate_orbit_state(sec_orbit[0], sec_orbit[1], sec_orbit[2], t_sec)
        r_sec = float(np.linalg.norm(np.asarray(s_sec) - p))
        r_ref_p = float(np.linalg.norm(np.asarray(s_ref) - p))      # equals r_ref to solver tolerance; used so the difference is exact
        phis.append(4.0 * np.pi / wavelength_m * (r_sec - r_ref_p))

    lon0, lat0, _ = ecef_to_geodetic(*pts[0])
    lon1, lat1, _ = ecef_to_geodetic(*pts[1])
    k1_fd = (phis[1] - phis[0]) / h_probe
    k2 = (phis[2] - 2.0 * phis[1] + phis[0]) / (h_probe ** 2)
    # one-sided differences at h=0 give k1 + k2*h/2 -> subtract the half-step curvature term to get the derivative at h=0
    k1 = k1_fd - 0.5 * k2 * h_probe
    los = _unit(np.asarray(s_ref) - pts[0])
    theta = math.degrees(math.acos(float(np.clip(np.dot(los, _geodetic_up(float(lon0), float(lat0))), -1.0, 1.0))))
    out.update(
        lon=float(lon0), lat=float(lat0),
        dlon_dh=float((lon1 - lon0) / h_probe), dlat_dh=float((lat1 - lat0) / h_probe),
        phase_flat=float(phis[0]), k1=float(k1), k2=float(k2), theta_deg=theta, range_m=r_ref,
    )
    return out


def compute_geometry_nodes(
    frame: PixelFrame,
    ref_orbit,
    sec_orbit,
    sec_geometry,
    wavelength_m: float = SENTINEL1_WAVELENGTH_M,
    n_nodes: Tuple[int, int] = (9, 9),
    h_probe_m: float = 500.0,
) -> GeometryNodes:
    """Solve the exact range-Doppler geometry of a pair on a sparse node grid.

    Args:
        frame: output pixel -> reference SLC time / range (see ``PixelFrame``).
        ref_orbit, sec_orbit: ``geolocation.parse_orbit_file`` outputs.
        sec_geometry: the secondary's ``SLCGeometry`` (only its first line time is used, to start the secondary's
            zero-Doppler search near the right pass).
        n_nodes: nodes per axis (rows, cols). More nodes follow a longer, more curved scene better; 9 x 9 is ample for a
            sub-swath (the fields are smooth).
        h_probe_m: height step for the terrain derivatives.

    Raises:
        GeometryError: when nodes fail.
    """
    h, w = frame.shape
    if h < 2 or w < 2:
        raise GeometryError(f"interferogram grid {frame.shape} is too small for geometry nodes")
    nr, nc = max(2, min(n_nodes[0], h)), max(2, min(n_nodes[1], w))
    # Nodes run one output pixel beyond the grid on every side, so the spline can be evaluated at the full-resolution
    # pixel centres of the first and last multilook windows without extrapolating.
    rows = np.linspace(-1.0, float(h), nr)
    cols = np.linspace(-1.0, float(w), nc)
    ref_guess = frame.azimuth_time(h / 2)
    sec_shift = sec_geometry.first_line_time - frame.geometry.first_line_time
    vals = {k: np.full((len(rows), len(cols)), np.nan) for k in _FIELDS}
    failed = []
    for a, i in enumerate(rows):
        for b, j in enumerate(cols):
            try:
                node = _node(frame, ref_orbit, sec_orbit, ref_guess, sec_shift, i, j, wavelength_m, h_probe_m)
            except Exception as exc:  # noqa: BLE001 - recorded, then judged below
                failed.append((float(i), float(j), str(exc)))
                continue
            for k in _FIELDS:
                vals[k][a, b] = node[k]
    if failed:
        # A grid with holes cannot be interpolated reliably, so any failed node is an error.
        raise GeometryError(
            f"radar geometry failed at {len(failed)} of {len(rows) * len(cols)} nodes (first: {failed[0][2]}); check that "
            "the orbit files cover both acquisitions and that the SLC geometry matches the SLC"
        )
    return GeometryNodes((h, w), rows, cols, vals, wavelength_m, frame, h_probe_m)
