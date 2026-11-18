"""
Native InSAR figures for pygeofetch.

Every figure the Silesia case study shows (footprints, baseline network, DEM,
SLC amplitude and tiles, interferogram sheets, tile quality, pyunwrap tile
windows, stack diagnostics, fringe profiles, unwrapping panels, velocity) is a
method of :class:`InSARFigures`. Each method takes the data it draws as
arguments (no notebook globals), saves a ``.png`` and returns its path.

Usage::

    from pygeofetch.viz.insar import InSARFigures

    figs = InSARFigures("results/figures", looks=(8, 4), unwrap_looks=(4, 2))
    figs.acquisitions(scenes, aoi_bbox)
    pos, gp = figs.compute_baselines(orbit_files, scenes, aoi_bbox)
    figs.baseline_network(pos, gp, scenes, interferograms=interferograms)
    figs.pair_sheet(pair, interferograms)

Conventions: axes in kilometres, cyclic ``hsv`` (rainbow) for wrapped phase only, the same map the interferogram PNGs use,
sequential single-hue for magnitudes, two-hue diverging for signed values.
One fringe is half a wavelength of line-of-sight change.

Nothing here generates data. Plots that cannot be drawn from what was passed
in say so on the figure instead of inventing content.
"""

from __future__ import annotations

import functools
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("pygeofetch.viz.insar")

PathLike = Union[str, Path]
Pair = Tuple[str, str]

# ---------------------------------------------------------------------------------------------------
# palette (validated categorical slots for identity; neutral ink; one-hue sequential; two-hue diverging)
# ---------------------------------------------------------------------------------------------------
INK, INK2, GRID_C, SURF = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"
C_S1A, C_S1B = "#2a78d6", "#eb6834"  # satellite identity, same in every figure
C_IN, C_OUT, C_UNK = "#1baf7a", "#8a8984", "#52514e"  # burst timing: in spec / out of spec / unknown
PHASE_CMAP = "hsv"  # cyclic rainbow: -pi and +pi are both red; same map as insar.visualize

_RC = {
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
    "axes.labelcolor": INK2, "axes.edgecolor": GRID_C, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 100,
}


def _mpl():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, LightSource
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch, Rectangle
    except ImportError as exc:  # pragma: no cover
        raise ImportError('pygeofetch.viz.insar needs matplotlib: pip install "pygeofetch[viz]"') from exc
    return plt, LinearSegmentedColormap, LightSource, Line2D, Patch, Rectangle


def _cmaps():
    _, LSC, *_ = _mpl()
    seq = LSC.from_list("seq_blue", ["#eaf1fb", "#2a78d6", "#0d2f63"])
    div = LSC.from_list("div_orange_blue", ["#eb6834", "#f2f1ee", "#2a78d6"])
    return seq, div


@dataclass(frozen=True)
class SarGeometry:
    """Radar constants used to put axes in kilometres and phase in millimetres.

    Defaults are Sentinel-1 IW (C-band, 2.3 m slant-range and 14.1 m azimuth pixels).
    Incidence is a scene-centre value; change it for another swath.
    """

    wavelength_m: float = 0.05546576
    incidence_deg: float = 39.0
    range_pixel_m: float = 2.3
    azimuth_pixel_m: float = 14.1
    burst_requirement_ms: float = 5.0

    @property
    def ground_range_pixel_m(self) -> float:
        return self.range_pixel_m / float(np.sin(np.radians(self.incidence_deg)))

    @property
    def mm_per_rad(self) -> float:
        return self.wavelength_m / (4 * np.pi) * 1000.0

    @property
    def mm_per_fringe(self) -> float:
        return self.wavelength_m / 2 * 1000.0


# ---------------------------------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------------------------------
def _clim(a, lo=2, hi=98):
    v = np.asarray(a, float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if v.size else (0.0, 1.0)


def _db(power):
    return 10 * np.log10(np.maximum(power, 1e-12))


def _block_mean(a, by, bx):
    h, w = (a.shape[0] // by) * by, (a.shape[1] // bx) * bx
    return a[:h, :w].reshape(h // by, by, w // bx, bx).mean(axis=(1, 3))


def _product_name(scene) -> str:
    props = getattr(scene, "properties", None) or {}
    n = props.get("name") or props.get("title") or getattr(scene, "name", "") or ""
    for suf in (".SAFE.zip", ".zip", ".SAFE"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n


def _scene_date(scene) -> str:
    return str(scene.datetime)[:10]


def relative_orbit(product_name: str) -> Optional[int]:
    """Relative orbit (track) from a Sentinel-1 product name: S1A (abs-73)%175+1, S1B (abs-27)%175+1."""
    try:
        absolute = int(product_name.split("_")[7])
    except (IndexError, ValueError):
        return None
    return (absolute - 73) % 175 + 1 if product_name[:3] == "S1A" else (absolute - 27) % 175 + 1


def _geom_rings(g) -> List[np.ndarray]:
    if g is None:
        return []
    try:
        from shapely.geometry import shape as _shape

        geom = _shape(g) if isinstance(g, dict) else g
        polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        return [np.asarray(p.exterior.coords) for p in polys if hasattr(p, "exterior")]
    except Exception:  # noqa: BLE001
        return []


def _spec_of(result) -> Optional[bool]:
    """True / False / None: was this pair inside the burst-timing requirement?"""
    if result is None:
        return None
    return (getattr(result, "metadata", None) or {}).get("burst_sync_within_requirement")


def _in_ipython() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except Exception:  # noqa: BLE001
        return False


def _tile_grid(shape: Tuple[int, int], tile: int, overlap: int) -> Tuple[List[Tuple[int, int, int, int]], str]:
    """Tile windows (row, col, height, width) used for inference; pyunwrap's own grid if installed."""
    try:
        from pyunwrap.data.preprocessing import compute_tile_grid

        specs = compute_tile_grid(shape, tile_size=tile, overlap=overlap)
        return [(s.row, s.col, s.height, s.width) for s in specs], "pyunwrap"
    except Exception:  # noqa: BLE001
        H, W = shape
        step = max(1, tile - overlap)
        rows = list(range(0, max(H - tile, 0) + 1, step))
        cols = list(range(0, max(W - tile, 0) + 1, step))
        if rows[-1] + tile < H:
            rows.append(H - tile)
        if cols[-1] + tile < W:
            cols.append(W - tile)
        return [(r, c, min(tile, H - r), min(tile, W - c)) for r in rows for c in cols], "fallback grid"


def _figure(method: Callable) -> Callable:
    """A plotting problem must not stop a processing run: warn and return None unless ``strict``."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        plt = _mpl()[0]
        try:
            with plt.rc_context(_RC):
                return method(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            plt.close("all")
            if self.strict:
                raise
            warnings.warn(f"pygeofetch.viz.insar: {method.__name__} skipped: {exc!r}", stacklevel=2)
            logger.warning("%s skipped: %r", method.__name__, exc)
            return None

    return wrapper


# ---------------------------------------------------------------------------------------------------
class InSARFigures:
    """Draws and saves the InSAR figures.

    Args:
        output_dir:    Where PNGs are written (created if missing).
        geometry:      :class:`SarGeometry` (Sentinel-1 IW defaults).
        looks:         (azimuth, range) looks of the interferograms (for km axes).
        unwrap_looks:  (azimuth, range) extra looks used before unwrapping.
        towns:         Optional ``{name: (lat, lon)}`` markers for orientation (approximate positions).
        dpi:           PNG resolution.
        show:          Display each figure in a notebook. ``None`` = only when running in IPython.
        strict:        Raise on plotting errors instead of warning and returning ``None``.
    """

    def __init__(
        self,
        output_dir: PathLike,
        geometry: Optional[SarGeometry] = None,
        looks: Tuple[int, int] = (8, 4),
        unwrap_looks: Tuple[int, int] = (4, 2),
        towns: Optional[Mapping[str, Tuple[float, float]]] = None,
        dpi: int = 150,
        show: Optional[bool] = None,
        strict: bool = False,
    ):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.geo = geometry or SarGeometry()
        self.looks = tuple(int(v) for v in looks)
        self.unwrap_looks = tuple(int(v) for v in unwrap_looks)
        self.towns = dict(towns or {})
        self.dpi = dpi
        self.show = _in_ipython() if show is None else show
        self.strict = strict
        self.saved: List[Path] = []

    # ------------------------------------------------------------------ plumbing
    def _save(self, fig, name: str) -> Path:
        plt = _mpl()[0]
        path = self.dir / f"{name}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        if path not in self.saved:
            self.saved.append(path)
        if self.show:
            try:
                from IPython.display import Image, display

                display(Image(filename=str(path)))
            except Exception:  # noqa: BLE001
                pass
        logger.info("saved %s", path)
        return path

    def save(self, fig, name: str) -> Path:
        """Save any matplotlib figure into the figures folder (PNG, same style of display), e.g. from other stages."""
        return self._save(fig, name)

    def list_figures(self) -> List[Path]:
        """Print and return every figure written by this object."""
        print(f"{len(self.saved)} figures in {self.dir}")
        for p in self.saved:
            print("  ", p.name)
        return list(self.saved)

    def _km_extent(self, shape, looks_az, looks_rg):
        h, w = shape
        return [0, w * looks_rg * self.geo.ground_range_pixel_m / 1000, h * looks_az * self.geo.azimuth_pixel_m / 1000, 0]

    def _spec_style(self, spec):
        if spec is True:
            return dict(color=C_IN, marker="o", mfc=C_IN, label=f"within {self.geo.burst_requirement_ms:g} ms")
        if spec is False:
            return dict(color=C_OUT, marker="^", mfc="white", label=f"outside {self.geo.burst_requirement_ms:g} ms")
        return dict(color=C_UNK, marker="s", mfc=C_UNK, label="timing unknown")

    @staticmethod
    def _tidy(ax, grid="y"):
        ax.grid(axis=grid, color=GRID_C, lw=0.6)
        ax.set_axisbelow(True)

    @staticmethod
    def _title(fig, text, size=12, y=None):
        kw = dict(x=0.01, ha="left", fontsize=size, fontweight="bold")
        if y is not None:
            kw["y"] = y
        fig.suptitle(text, **kw)

    def _pair_arrays(self, pair: Pair, interferograms: Mapping[Pair, Any]):
        r = interferograms[pair]
        amp = np.asarray(r.amplitude, float)
        coh = np.asarray(r.coherence, float)
        ph = np.angle(np.asarray(r.interferogram))
        h = min(amp.shape[0], coh.shape[0], ph.shape[0])
        w = min(amp.shape[1], coh.shape[1], ph.shape[1])
        return r, amp[:h, :w], coh[:h, :w], ph[:h, :w]

    def _caption(self, pair: Pair, r, coh) -> str:
        import pandas as pd

        m, (d1, d2) = (r.metadata or {}), pair
        parts = [f"{d1} -> {d2}", f"{(pd.Timestamp(d2) - pd.Timestamp(d1)).days} d"]
        if r.perpendicular_baseline_m is not None:
            parts.append(f"Bperp {r.perpendicular_baseline_m:.0f} m")
        if m.get("burst_sync_offset_ms") is not None:
            parts.append(
                f"burst offset {m['burst_sync_offset_ms']:+.1f} ms "
                f"({'within' if m.get('burst_sync_within_requirement') else 'outside'} {self.geo.burst_requirement_ms:g} ms)"
            )
        if m.get("coregistration_gcps_final") is not None:
            rms = m.get("coregistration_rms_mean_px")
            parts.append(
                f"coreg {m.get('coregistration_method')}, {m.get('coregistration_gcps_final')} GCPs"
                + (f", RMS {rms:.2f} px" if rms is not None else "")
            )
        parts.append(f"mean coherence {np.nanmean(coh):.3f}")
        return "   |   ".join(parts)

    @staticmethod
    def representative_pairs(interferograms: Mapping[Pair, Any]) -> Dict[str, Pair]:
        """The first pair, the most coherent and the least coherent (by mean coherence)."""
        mc = {p: float(np.nanmean(r.coherence)) for p, r in interferograms.items()}
        return dict(first=next(iter(interferograms)), best=max(mc, key=mc.get), worst=min(mc, key=mc.get))

    # ------------------------------------------------------------------ V1
    @_figure
    def acquisitions(self, scenes: Sequence[Any], aoi, name: str = "V1_acquisitions") -> Path:
        """Scene footprints against the AOI, and the acquisition timeline by satellite with gaps in days."""
        import pandas as pd

        plt, _, _, Line2D, Patch, Rectangle = _mpl()
        rows = []
        for s in scenes:
            n = _product_name(s)
            rows.append(dict(date=pd.Timestamp(_scene_date(s)), sat=n[:3], track=relative_orbit(n) if n else None,
                             geom=getattr(s, "geometry", None)))
        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        fig, (axm, axt) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1, 1.3]))
        n_ring, xs, ys = 0, [aoi.min_lon, aoi.max_lon], [aoi.min_lat, aoi.max_lat]
        for g in df.geom:
            for ring in _geom_rings(g):
                n_ring += 1
                xs += list(ring[:, 0])
                ys += list(ring[:, 1])
                axm.plot(ring[:, 0], ring[:, 1], color=C_S1A, lw=0.7, alpha=0.35)
        axm.add_patch(Rectangle((aoi.min_lon, aoi.min_lat), aoi.max_lon - aoi.min_lon, aoi.max_lat - aoi.min_lat,
                                fc=C_S1B, alpha=0.18, ec=INK, lw=1.8))
        for nm, (lat, lon) in self.towns.items():
            axm.plot(lon, lat, marker="^", ms=7, mfc="white", mec=INK, mew=1.2, ls="")
            axm.annotate(nm, (lon, lat), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=INK2)
        pad = 0.25
        axm.set_xlim(min(xs) - pad, max(xs) + pad)
        axm.set_ylim(min(ys) - pad, max(ys) + pad)
        axm.set_aspect(1 / np.cos(np.radians((aoi.min_lat + aoi.max_lat) / 2)))
        axm.set_xlabel("longitude (deg E)")
        axm.set_ylabel("latitude (deg N)")
        axm.set_title(f"Footprints of {n_ring} scenes vs the AOI")
        handles = [Line2D([], [], color=C_S1A, alpha=0.6, label="scene footprint (one outline per scene)"),
                   Patch(fc=C_S1B, alpha=0.3, ec=INK, lw=1.8, label="AOI")]
        if self.towns:
            handles.append(Line2D([], [], marker="^", ls="", mfc="white", mec=INK, label="town (approx. position)"))
        axm.legend(handles=handles, loc="lower left", frameon=False)
        self._tidy(axm, "both")
        ypos = {"S1A": 1, "S1B": 0}
        for sat, col in (("S1A", C_S1A), ("S1B", C_S1B)):
            sub = df[df.sat == sat]
            axt.vlines(sub.date, -0.45 if sat == "S1B" else 0.55, ypos[sat] + 0.35, color=col, lw=0.8, alpha=0.5)
            axt.scatter(sub.date, [ypos[sat]] * len(sub), s=46, color=col, zorder=3, label=f"{sat}  ({len(sub)} scenes)")
        gaps = df.date.diff().dt.days.dropna()
        for i in range(1, len(df)):
            axt.annotate(f"{int(gaps.iloc[i - 1])} d", (df.date[i - 1] + (df.date[i] - df.date[i - 1]) / 2, 0.5),
                         ha="center", va="center", fontsize=7, color=INK2)
        axt.set_yticks([0, 1])
        axt.set_yticklabels(["S1B", "S1A"])
        axt.set_ylim(-0.6, 1.6)
        tracks = sorted(df.track.dropna().unique().astype(int).tolist())
        axt.set_title(f"Acquisition timeline, track {tracks}, median gap {gaps.median():.0f} d" if len(gaps)
                      else "Acquisition timeline")
        axt.legend(loc="upper right", frameon=False, ncol=2)
        self._tidy(axt, "x")
        fig.autofmt_xdate()
        self._title(fig, "V1  What data do we actually have?")
        return self._save(fig, name)

    # ------------------------------------------------------------------ V2
    def compute_baselines(self, orbit_files: Mapping[str, PathLike], scenes: Sequence[Any], aoi=None,
                          lat: Optional[float] = None, lon: Optional[float] = None):
        """Satellite position at zero-Doppler time over one ground point, per date.

        Same method as the SBAS stage. Returns ``({date: position}, ground_point_ecef)``;
        dates whose orbit cannot be read are skipped with a warning.
        """
        from pygeofetch.insar.geolocation import (find_zero_doppler_time, geodetic_to_ecef,
                                                  interpolate_orbit_state, parse_orbit_file)

        if lat is None or lon is None:
            if aoi is None:
                raise ValueError("pass aoi or lat/lon")
            lat = (aoi.min_lat + aoi.max_lat) / 2 if lat is None else lat
            lon = (aoi.min_lon + aoi.max_lon) / 2 if lon is None else lon
        gp = geodetic_to_ecef(lat, lon, 0.0)
        t_guess = {_scene_date(s): s.datetime.replace(tzinfo=None) for s in scenes}
        pos: Dict[str, Any] = {}
        for d in sorted(orbit_files):
            try:
                times, positions, velocities = parse_orbit_file(orbit_files[d])
                tz = find_zero_doppler_time(times, positions, velocities, gp, initial_time_guess=t_guess[d])
                pos[d], _ = interpolate_orbit_state(times, positions, velocities, target_time=tz)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"baseline: {d} skipped ({exc})", stacklevel=2)
        return pos, gp

    @_figure
    def baseline_network(self, positions: Mapping[str, Any], ground_point, scenes: Sequence[Any],
                         max_days: int = 24, interferograms: Optional[Mapping[Pair, Any]] = None,
                         name: str = "V2_baseline_network"):
        """SNAP-style baseline plot: perpendicular baseline against time, and |Bperp| against temporal baseline.

        Pair lines are coloured by burst-timing class when ``interferograms`` (with metadata) is given.
        Returns a DataFrame with one row per pair.
        """
        import pandas as pd

        from pygeofetch.insar.geolocation import perpendicular_baseline

        plt, _, _, Line2D, *_ = _mpl()
        interferograms = interferograms or {}
        dates = sorted(positions)
        sat = {_scene_date(s): _product_name(s)[:3] for s in scenes}
        ref = dates[0]
        brel = {d: (0.0 if d == ref else float(perpendicular_baseline(positions[ref], positions[d], ground_point)))
                for d in dates}
        t = {d: pd.Timestamp(d) for d in dates}
        pairs = [(a, b) for i, a in enumerate(dates) for b in dates[i + 1:] if (t[b] - t[a]).days <= max_days]
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
        for a, b in pairs:
            st = self._spec_style(_spec_of(interferograms.get((a, b))))
            ax.plot([t[a], t[b]], [brel[a], brel[b]], color=st["color"], lw=0.9,
                    ls="--" if st["marker"] == "^" else "-", alpha=0.8)
        for s_, col in (("S1A", C_S1A), ("S1B", C_S1B)):
            ds = [d for d in dates if sat.get(d) == s_]
            ax.scatter([t[d] for d in ds], [brel[d] for d in ds], s=40, color=col, zorder=3, ec="white", lw=0.8, label=s_)
        ax.set_ylabel(f"perpendicular baseline to {ref} (m)")
        ax.set_title(f"Baseline network: {len(dates)} dates, {len(pairs)} pairs of at most {max_days} d")
        handles, _ = ax.get_legend_handles_labels()
        handles += [Line2D([], [], color=C_IN, label=f"pair within {self.geo.burst_requirement_ms:g} ms"),
                    Line2D([], [], color=C_OUT, ls="--", label=f"pair outside {self.geo.burst_requirement_ms:g} ms"),
                    Line2D([], [], color=C_UNK, label="timing not computed")]
        ax.legend(handles=handles, loc="best", frameon=False, ncol=2)
        self._tidy(ax, "y")
        rec = [dict(pair=(a, b), days=(t[b] - t[a]).days,
                    bperp=abs(float(perpendicular_baseline(positions[a], positions[b], ground_point)))) for a, b in pairs]
        for r in rec:
            st = self._spec_style(_spec_of(interferograms.get(r["pair"])))
            ax2.scatter(r["days"], r["bperp"], s=34, marker=st["marker"], facecolors=st["mfc"], edgecolors=st["color"], lw=1.2)
        ax2.set_xlabel("temporal baseline (days)")
        ax2.set_ylabel("|perpendicular baseline| (m)")
        ax2.set_title("What each pair costs in baseline")
        self._tidy(ax2, "both")
        fig.autofmt_xdate()
        self._title(fig, "V2  Orbit geometry: how far apart were the two satellite positions?")
        self._save(fig, name)
        return pd.DataFrame(rec)

    # ------------------------------------------------------------------ V3
    @_figure
    def dem(self, dem_file: PathLike, aoi, name: str = "V3_dem_overview") -> Path:
        """DEM hillshade with elevation tint, and an elevation histogram (2nd-98th percentile relief)."""
        import rasterio

        plt, _, LightSource, _, _, Rectangle = _mpl()
        seq, _ = _cmaps()
        with rasterio.open(dem_file) as src:
            z = src.read(1).astype(float)
            b = src.bounds
            nd = src.nodata
        if nd is not None:
            z[z == nd] = np.nan
        step = max(1, int(max(z.shape) // 1400))
        z = z[::step, ::step]
        lat0 = (b.top + b.bottom) / 2
        dx = (b.right - b.left) / z.shape[1] * 111320 * np.cos(np.radians(lat0))
        dy = (b.top - b.bottom) / z.shape[0] * 110540
        hs = LightSource(azdeg=315, altdeg=40).hillshade(np.nan_to_num(z, nan=np.nanmedian(z)), vert_exag=3, dx=dx, dy=dy)
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.45))
        ext = [b.left, b.right, b.bottom, b.top]
        asp = 1 / np.cos(np.radians(lat0))
        ax.imshow(hs, cmap="gray", extent=ext, vmin=0.2, vmax=1.0, aspect=asp)
        im = ax.imshow(z, cmap=seq, extent=ext, alpha=0.55, aspect=asp)
        ax.add_patch(Rectangle((aoi.min_lon, aoi.min_lat), aoi.max_lon - aoi.min_lon, aoi.max_lat - aoi.min_lat,
                               fill=False, ec=INK, lw=1.6))
        for nm, (lat, lon) in self.towns.items():
            if b.left <= lon <= b.right and b.bottom <= lat <= b.top:
                ax.plot(lon, lat, marker="^", ms=7, mfc="white", mec=INK, mew=1.2, ls="")
                ax.annotate(nm, (lon, lat), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=INK)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="elevation (m)")
        ax.set_xlabel("longitude (deg E)")
        ax.set_ylabel("latitude (deg N)")
        ax.set_title("DEM: hillshade + elevation")
        v = z[np.isfinite(z)]
        ax2.hist(v, bins=60, color=C_S1A, alpha=0.85)
        p2, p98 = np.percentile(v, [2, 98])
        ax2.axvline(p2, color=INK2, lw=0.8, ls="--")
        ax2.axvline(p98, color=INK2, lw=0.8, ls="--")
        ax2.set_xlabel("elevation (m)")
        ax2.set_ylabel("pixels")
        ax2.set_title(f"Relief: 2nd-98th percentile {p2:.0f}-{p98:.0f} m (span {p98 - p2:.0f} m)")
        self._tidy(ax2, "y")
        self._title(fig, "V3  Terrain: how much topographic phase should we expect?")
        return self._save(fig, name)

    # ------------------------------------------------------------------ V4
    def _read_power(self, path, by, bx):
        import rasterio

        with rasterio.open(path) as src:
            a = src.read(1)
        return _block_mean(np.abs(a).astype(np.float64) ** 2, by, bx), a.shape

    @_figure
    def slc_overview(self, path: PathLike, date: str, tile_km: float = 5.0, by: int = 4, bx: int = 16,
                     name: Optional[str] = None) -> Path:
        """Full-scene SLC amplitude (dB) in km with a square tile grid, brightness histogram, per-tile mean."""
        plt, *_ = _mpl()
        seq, _ = _cmaps()
        name = name or f"V4_slc_{date}"
        p, full_shape = self._read_power(path, by, bx)
        img = _db(p)
        lo, hi = _clim(img)
        g = self.geo
        ext = self._km_extent(img.shape, by, bx)
        fig = plt.figure(figsize=(13, 6.2))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1], hspace=0.38, wspace=0.42)
        ax = fig.add_subplot(gs[:, 0])
        imh = ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, extent=ext, aspect="equal")
        px_az_km, px_rg_km = by * g.azimuth_pixel_m / 1000, bx * g.ground_range_pixel_m / 1000
        th, tw = max(1, int(round(tile_km / px_az_km))), max(1, int(round(tile_km / px_rg_km)))
        for x in np.arange(0, ext[1] + 1e-6, tw * px_rg_km):
            ax.axvline(x, color=C_S1B, lw=0.6, alpha=0.8)
        for y in np.arange(0, ext[2] + 1e-6, th * px_az_km):
            ax.axhline(y, color=C_S1B, lw=0.6, alpha=0.8)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("ground range (km)")
        ax.set_ylabel("azimuth (km)")
        ax.set_title(f"{date}: SLC amplitude, {full_shape[0]} x {full_shape[1]} px, shown multilooked {by}x{bx}")
        fig.colorbar(imh, ax=ax, fraction=0.03, pad=0.02, label="amplitude (dB, relative)")
        ax.text(0.01, 0.01, f"orange grid = {tile_km:g} x {tile_km:g} km tiles", transform=ax.transAxes, color="white",
                fontsize=7.5, va="bottom", bbox=dict(fc=INK, alpha=0.55, ec="none", pad=2))
        axh = fig.add_subplot(gs[0, 1])
        axh.hist(img[np.isfinite(img)].ravel(), bins=80, color=C_S1A, alpha=0.9)
        axh.axvline(lo, color=INK2, ls="--", lw=0.8)
        axh.axvline(hi, color=INK2, ls="--", lw=0.8)
        axh.set_xlabel("amplitude (dB)")
        axh.set_ylabel("pixels")
        axh.set_title("Brightness histogram (dashed = display limits)")
        self._tidy(axh, "y")
        axt = fig.add_subplot(gs[1, 1])
        nth, ntw = max(1, img.shape[0] // th), max(1, img.shape[1] // tw)
        tm = img[: nth * th, : ntw * tw].reshape(nth, th, ntw, tw).mean(axis=(1, 3))
        it = axt.imshow(tm, cmap=seq, aspect="auto", extent=[0, ntw * tile_km, nth * tile_km, 0])
        if tm.size <= 150:
            for (i, j), v in np.ndenumerate(tm):
                axt.text((j + 0.5) * tile_km, (i + 0.5) * tile_km, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                         color="white" if v > np.nanmean(tm) else INK)
        fig.colorbar(it, ax=axt, fraction=0.04, pad=0.02, label="mean dB")
        axt.set_title("Mean brightness per tile")
        axt.set_xlabel("ground range (km)")
        axt.set_ylabel("azimuth (km)")
        self._title(fig, f"V4  Full scene {date}: what the radar sees")
        return self._save(fig, name)

    @_figure
    def slc_gallery(self, paths: Mapping[str, PathLike], max_n: int = 8, by: int = 8, bx: int = 32,
                    name: str = "V4_slc_gallery") -> Path:
        """Several dates on one brightness scale; flags image sizes that differ between dates."""
        plt, *_ = _mpl()
        dates = sorted(paths)[:max_n]
        imgs, shapes = {}, {}
        for d in dates:
            p, sh = self._read_power(paths[d], by, bx)
            imgs[d], shapes[d] = _db(p), sh
        allv = np.concatenate([i[np.isfinite(i)].ravel() for i in imgs.values()])
        lo, hi = np.percentile(allv, [2, 98])
        ncol = 4
        nrow = int(np.ceil(len(dates) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow), squeeze=False)
        for ax in axs.ravel():
            ax.axis("off")
        for ax, d in zip(axs.ravel(), dates):
            ax.axis("on")
            ax.imshow(imgs[d], cmap="gray", vmin=lo, vmax=hi, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{d}   {shapes[d][0]}x{shapes[d][1]}", fontsize=8)
            for sp in ax.spines.values():
                sp.set_visible(False)
        odd = len(set(shapes.values())) > 1
        self._title(fig, "V4  Same area, different dates (same brightness scale)"
                    + ("  |  image sizes differ between dates: they are NOT on a common pixel grid yet" if odd else ""), size=11)
        return self._save(fig, name)

    # ------------------------------------------------------------------ V5
    @_figure
    def pair_sheet(self, pair: Pair, interferograms: Mapping[Pair, Any], name: Optional[str] = None) -> Path:
        """Amplitude, coherence, wrapped phase, coherence histogram and azimuth/range coherence profiles."""
        plt, *_ = _mpl()
        seq, _ = _cmaps()
        r, amp, coh, ph = self._pair_arrays(pair, interferograms)
        ext = self._km_extent(coh.shape, *self.looks)
        fig = plt.figure(figsize=(14, 7.4))
        gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28)
        ax = fig.add_subplot(gs[0, 0])
        img_axes = [ax]
        a_db = 20 * np.log10(np.maximum(amp, 1e-6))
        lo, hi = _clim(a_db)
        im = ax.imshow(a_db, cmap="gray", vmin=lo, vmax=hi, extent=ext, aspect="equal")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="dB")
        ax.set_title("Amplitude (reference)")
        ax = fig.add_subplot(gs[0, 1])
        img_axes.append(ax)
        im = ax.imshow(coh, cmap="gray", vmin=0, vmax=1, extent=ext, aspect="equal")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="coherence")
        ax.set_title("Coherence (1 = same scatterers, 0 = unrelated)")
        ax = fig.add_subplot(gs[0, 2])
        img_axes.append(ax)
        im = ax.imshow(ph, cmap=PHASE_CMAP, vmin=-np.pi, vmax=np.pi, extent=ext, aspect="equal")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[-np.pi, 0, np.pi])
        cb.ax.set_yticklabels(["-π", "0", "+π"])
        ax.set_title("Wrapped phase (one colour cycle = one fringe)")
        for a in img_axes:
            a.set_xlabel("ground range (km)")
        img_axes[0].set_ylabel("azimuth (km)")
        c = coh[np.isfinite(coh)]
        axh = fig.add_subplot(gs[1, 0])
        axh.hist(c, bins=50, range=(0, 1), color=C_S1A, alpha=0.9)
        for k, (thr, ls) in enumerate(((0.3, "--"), (0.5, ":"))):
            axh.axvline(thr, color=INK2, ls=ls, lw=1)
            axh.text(0.98, 0.93 - 0.09 * k, f"{(c > thr).mean() * 100:.1f}% of pixels > {thr}", fontsize=8, color=INK2,
                     transform=axh.transAxes, ha="right", va="top")
        axh.set_xlabel("coherence")
        axh.set_ylabel("pixels")
        axh.set_title("Coherence histogram")
        self._tidy(axh, "y")
        axa = fig.add_subplot(gs[1, 1])
        axa.plot(np.linspace(ext[3], ext[2], coh.shape[0]), np.nanmean(coh, axis=1), color=C_S1A, lw=1.3)
        axa.axhline(np.nanmean(coh), color=INK2, ls="--", lw=0.8)
        axa.set_xlabel("azimuth (km)")
        axa.set_ylabel("mean coherence")
        axa.set_title("Along azimuth (dips can mark burst edges)")
        axa.set_ylim(0, 1)
        self._tidy(axa, "y")
        axr = fig.add_subplot(gs[1, 2])
        axr.plot(np.linspace(ext[0], ext[1], coh.shape[1]), np.nanmean(coh, axis=0), color=C_S1A, lw=1.3)
        axr.axhline(np.nanmean(coh), color=INK2, ls="--", lw=0.8)
        axr.set_xlabel("ground range (km)")
        axr.set_ylabel("mean coherence")
        axr.set_title("Along range")
        axr.set_ylim(0, 1)
        self._tidy(axr, "y")
        self._title(fig, "V5  " + self._caption(pair, r, coh), size=10.5, y=0.99)
        return self._save(fig, name or f"V5_pair_{pair[0]}_{pair[1]}")

    # ------------------------------------------------------------------ V6
    @_figure
    def tile_quality(self, pair: Pair, interferograms: Mapping[Pair, Any], tile_km: float = 4.0,
                     name: Optional[str] = None) -> Path:
        """Coherence in km-square tiles, a per-tile heat map, and phase thumbnails of the best and worst tiles."""
        plt, _, _, _, _, Rectangle = _mpl()
        seq, _ = _cmaps()
        r, amp, coh, ph = self._pair_arrays(pair, interferograms)
        la, lr = self.looks
        ext = self._km_extent(coh.shape, la, lr)
        px_az, px_rg = la * self.geo.azimuth_pixel_m / 1000, lr * self.geo.ground_range_pixel_m / 1000
        th, tw = max(4, int(round(tile_km / px_az))), max(4, int(round(tile_km / px_rg)))
        nth, ntw = max(1, coh.shape[0] // th), max(1, coh.shape[1] // tw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tm = np.array([[np.nanmean(coh[i * th:(i + 1) * th, j * tw:(j + 1) * tw]) for j in range(ntw)]
                           for i in range(nth)])
        order = np.argsort(np.where(np.isfinite(tm), tm, -1).ravel())
        worst = [np.unravel_index(k, tm.shape) for k in order[:3]]
        best = [np.unravel_index(k, tm.shape) for k in order[::-1][:3]]
        fig = plt.figure(figsize=(14, 9.4))
        gs = fig.add_gridspec(3, 6, height_ratios=[1.5, 1, 1], hspace=0.45, wspace=0.5)
        ax = fig.add_subplot(gs[0, 0:3])
        ax.imshow(coh, cmap="gray", vmin=0, vmax=1, extent=ext, aspect="equal")
        for i in range(nth + 1):
            ax.axhline(i * th * px_az, color=C_S1B, lw=0.6, alpha=0.8)
        for j in range(ntw + 1):
            ax.axvline(j * tw * px_rg, color=C_S1B, lw=0.6, alpha=0.8)
        for lab, group, col in (("B", best, C_IN), ("W", worst, "#ffffff")):
            for k, (i, j) in enumerate(group):
                ax.add_patch(Rectangle((j * tw * px_rg, i * th * px_az), tw * px_rg, th * px_az, fill=False, ec=col, lw=2))
                ax.text(j * tw * px_rg + 0.15, i * th * px_az + 0.15, f"{lab}{k + 1}", color=col, fontsize=8, va="top",
                        fontweight="bold")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("ground range (km)")
        ax.set_ylabel("azimuth (km)")
        ax.set_title(f"Coherence in {tile_km:g} x {tile_km:g} km tiles (B = best, W = worst)")
        axh = fig.add_subplot(gs[0, 3:6])
        it = axh.imshow(tm, cmap=seq, vmin=0, vmax=max(0.5, float(np.nanmax(tm))), aspect="equal",
                        extent=[0, ntw * tw * px_rg, nth * th * px_az, 0])
        if tm.size <= 140:
            for (i, j), v in np.ndenumerate(tm):
                axh.text((j + 0.5) * tw * px_rg, (i + 0.5) * th * px_az, f"{v:.2f}", ha="center", va="center", fontsize=7,
                         color="white" if v > 0.5 * np.nanmax(tm) else INK)
        fig.colorbar(it, ax=axh, fraction=0.04, pad=0.02, label="mean coherence")
        axh.set_title("Mean coherence per tile")
        axh.set_xlabel("ground range (km)")
        axh.set_ylabel("azimuth (km)")
        for row, group in enumerate((best, worst)):
            for k, (i, j) in enumerate(group):
                a = fig.add_subplot(gs[1 + row, 2 * k:2 * k + 2])
                sub = ph[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
                a.imshow(sub, cmap=PHASE_CMAP, vmin=-np.pi, vmax=np.pi, aspect="equal",
                         extent=[0, sub.shape[1] * px_rg, sub.shape[0] * px_az, 0])
                a.set_title(f"{'B' if row == 0 else 'W'}{k + 1}  tile ({i},{j})  coherence {tm[i, j]:.2f}", fontsize=8.5)
                a.set_xlabel("km")
                a.set_ylabel("km")
        self._title(fig, "V6  " + self._caption(pair, r, coh), size=10.5, y=0.99)
        return self._save(fig, name or f"V6_tiles_{pair[0]}_{pair[1]}")

    @_figure
    def inference_tiles(self, pair: Pair, interferograms: Mapping[Pair, Any], tile: int = 256, overlap: int = 128,
                        name: Optional[str] = None) -> Path:
        """The windows a tiled unwrapper sees on the unwrapping grid, the overlap count, and the padding share."""
        plt, _, _, _, _, Rectangle = _mpl()
        seq, _ = _cmaps()
        ual, url_ = self.unwrap_looks
        coh = np.asarray(interferograms[pair].coherence, float)
        coh_ml = _block_mean(np.nan_to_num(coh), ual, url_)
        H, W = coh_ml.shape
        Hp, Wp = max(H, tile), max(W, tile)
        specs, src = _tile_grid((Hp, Wp), tile, overlap)
        cover = np.zeros((Hp, Wp))
        for (r0, c0, h, w) in specs:
            cover[r0:r0 + h, c0:c0 + w] += 1
        pad_share = 100 * (1 - (H * W) / (Hp * Wp))
        pad_img = np.zeros((Hp, Wp))
        pad_img[:H, :W] = coh_ml
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 7.4), gridspec_kw=dict(height_ratios=[1.1, 1]))
        ax.imshow(pad_img, cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        for k, (r0, c0, h, w) in enumerate(specs):
            ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), w, h, fill=False, ec=C_S1A if k % 2 == 0 else C_S1B, lw=1.6))
            ax.text(c0 + 6, r0 + 8, str(k), color="white", fontsize=8, va="top")
        ax.add_patch(Rectangle((-0.5, -0.5), W, H, fill=False, ec=C_IN, lw=2, ls="--"))
        ax.set_xlim(-0.5, Wp - 0.5)
        ax.set_ylim(Hp - 0.5, -0.5)
        ax.set_ylabel("azimuth (pixels of the unwrapping grid)")
        ax.set_title(f"{len(specs)} tiles of {tile} px, overlap {overlap} px ({src}). Data {H} x {W} px (green dashed) is "
                     f"padded to {Hp} x {Wp} with coherence 0: {pad_share:.0f}% of what the network sees is padding")
        ic = ax2.imshow(cover, cmap=seq, aspect="auto", vmin=0, interpolation="nearest")
        fig.colorbar(ic, ax=ax2, fraction=0.025, pad=0.01, label="tiles covering the pixel")
        ax2.set_xlabel("range (pixels of the unwrapping grid)")
        ax2.set_ylabel("azimuth (pixels)")
        ax2.set_title("Overlap count: pixels covered by several tiles are blended, weighted by confidence")
        self._title(fig, f"V6b  How the unwrapper tiles the scene  ({pair[0]} -> {pair[1]})", size=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return self._save(fig, name or f"V6b_inference_tiles_{pair[0]}_{pair[1]}")

    # ------------------------------------------------------------------ V7
    def stack_table(self, interferograms: Mapping[Pair, Any]):
        """One row per pair: coherence statistics, burst timing, coregistration quality, baselines."""
        import pandas as pd

        rows = []
        for (d1, d2), r in interferograms.items():
            c = np.asarray(r.coherence, float)
            c = c[np.isfinite(c)]
            m = r.metadata or {}
            rows.append(dict(pair=f"{d1}_{d2}", d1=d1, d2=d2, days=(pd.Timestamp(d2) - pd.Timestamp(d1)).days,
                             mean_coh=c.mean(), median_coh=np.median(c), p90=np.percentile(c, 90),
                             frac_gt03=(c > 0.3).mean(), frac_gt05=(c > 0.5).mean(),
                             burst_dt_ms=m.get("burst_sync_offset_ms"), in_spec=m.get("burst_sync_within_requirement"),
                             gcps=m.get("coregistration_gcps_final"), rms_px=m.get("coregistration_rms_mean_px"),
                             bperp_m=r.perpendicular_baseline_m))
        return pd.DataFrame(rows)

    @_figure
    def stack_diagnostics(self, interferograms: Mapping[Pair, Any], name: str = "V7_stack_diagnostics"):
        """Is coherence explained by time, burst timing or baseline? Writes ``V7_stack_table.csv`` and returns the table."""
        plt, *_ = _mpl()
        t = self.stack_table(interferograms)
        t.to_csv(self.dir / "V7_stack_table.csv", index=False)
        req = self.geo.burst_requirement_ms
        ymax = max(0.6, float(t.mean_coh.max()) * 1.15)
        fig, axs = plt.subplots(2, 2, figsize=(13, 8.2))
        (a, b), (c_, d) = axs
        specs = [s if s in (True, False) else None for s in t.in_spec]

        def _scatter(ax, x, y):
            for sp in (True, False, None):
                idx = [i for i, s in enumerate(specs) if s is sp]
                if idx:
                    st = self._spec_style(sp)
                    ax.scatter(np.asarray(x)[idx], np.asarray(y)[idx], s=38, marker=st["marker"], facecolors=st["mfc"],
                               edgecolors=st["color"], lw=1.3, label=f"{st['label']} (n={len(idx)})")

        _scatter(a, t.days, t.mean_coh)
        a.axhline(t.mean_coh.median(), color=INK2, ls="--", lw=0.8)
        a.set_xlabel("temporal baseline (days)")
        a.set_ylabel("mean coherence")
        a.set_title("Coherence vs temporal baseline")
        a.set_ylim(0, ymax)
        a.legend(frameon=False)
        self._tidy(a, "both")
        if t.burst_dt_ms.notna().any():
            _scatter(b, t.burst_dt_ms.abs().fillna(0), t.mean_coh)
            b.axvline(req, color=INK2, ls=":", lw=1)
            b.text(req + 0.3, 0.97, f"{req:g} ms requirement", fontsize=8, color=INK2, va="top", transform=b.get_xaxis_transform())
            b.set_xlabel("|burst offset| (ms)")
            b.set_ylabel("mean coherence")
            b.set_title("Coherence vs burst-timing offset")
            b.set_ylim(0, ymax)
            try:
                from scipy.stats import spearmanr

                rho, pv = spearmanr(t.burst_dt_ms.abs(), t.mean_coh, nan_policy="omit")
                b.text(0.98, 0.04, f"Spearman rho = {rho:.2f} (p = {pv:.2g}, n = {int(t.burst_dt_ms.notna().sum())})",
                       transform=b.transAxes, ha="right", fontsize=8, color=INK2)
            except Exception:  # noqa: BLE001
                pass
            self._tidy(b, "both")
        else:
            b.axis("off")
            b.text(0.5, 0.5, "burst offset was not stored for these pairs\n(needs both orbit files at pair-processing time)",
                   ha="center", va="center", color=INK2)
        if t.bperp_m.notna().any():
            _scatter(c_, t.bperp_m.abs().fillna(0), t.mean_coh)
            c_.set_xlabel("|perpendicular baseline| (m)")
            c_.set_ylabel("mean coherence")
            c_.set_title("Coherence vs perpendicular baseline")
            c_.set_ylim(0, ymax)
            self._tidy(c_, "both")
        else:
            c_.axis("off")
            c_.text(0.5, 0.5, "perpendicular baseline was not stored on these pairs", ha="center", va="center", color=INK2)
        d.bar(range(len(t)), t.mean_coh, color=[self._spec_style(s)["color"] for s in specs], width=0.8)
        d.set_xticks(range(len(t)))
        d.set_xticklabels([f"{x[5:]}>{y[5:]}" for x, y in zip(t.d1, t.d2)], rotation=90, fontsize=6.5)
        d.set_ylabel("mean coherence")
        d.set_ylim(0, ymax)
        d.set_title(f"Every pair in time order (green = within {req:g} ms, grey = outside, dark = unknown)")
        self._tidy(d, "y")
        self._title(fig, "V7  What explains the coherence of this stack?")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        self._save(fig, name)
        return t

    @_figure
    def coherence_sheet(self, interferograms: Mapping[Pair, Any], max_n: int = 60, ncols: int = 8,
                        name: str = "V7b_coherence_sheet") -> Path:
        """Coherence of every pair as a thumbnail on one 0-1 scale, framed by burst-timing class."""
        plt, *_ = _mpl()
        pairs = list(interferograms)[:max_n]
        nrow = int(np.ceil(len(pairs) / ncols))
        fig, axs = plt.subplots(nrow, ncols, figsize=(2.1 * ncols, 1.7 * nrow), squeeze=False)
        for ax in axs.ravel():
            ax.axis("off")
        for ax, p in zip(axs.ravel(), pairs):
            r = interferograms[p]
            ax.axis("on")
            ax.imshow(np.asarray(r.coherence, float), cmap="gray", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            col = self._spec_style(_spec_of(r))["color"]
            for sp in ax.spines.values():
                sp.set_edgecolor(col)
                sp.set_linewidth(2)
            ax.set_title(f"{p[0][2:]}>{p[1][5:]}  {np.nanmean(r.coherence):.2f}", fontsize=6.5)
        self._title(fig, "V7b  Coherence of every pair, same 0-1 scale (frame: green = within spec, grey = outside, dark = unknown)",
                    size=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return self._save(fig, name)

    # ------------------------------------------------------------------ V8
    @_figure
    def fringe_profile(self, pair: Pair, interferograms: Mapping[Pair, Any], row: Optional[int] = None,
                       name: Optional[str] = None) -> Path:
        """Wrapped phase along the most coherent azimuth row (or ``row``), with a millimetre axis and the coherence."""
        plt, *_ = _mpl()
        seq, _ = _cmaps()
        r, amp, coh, ph = self._pair_arrays(pair, interferograms)
        ext = self._km_extent(coh.shape, *self.looks)
        if row is None:
            sm = np.convolve(np.nan_to_num(np.nanmean(coh, axis=1)), np.ones(5) / 5, mode="same")
            row = int(np.argmax(sm))
        x = np.linspace(ext[0], ext[1], coh.shape[1])
        mm_per_rad, mm_per_fringe = self.geo.mm_per_rad, self.geo.mm_per_fringe
        fig = plt.figure(figsize=(13.5, 8))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.2, 0.6], hspace=0.42)
        ax = fig.add_subplot(gs[0])
        ax.imshow(ph, cmap=PHASE_CMAP, vmin=-np.pi, vmax=np.pi, extent=ext, aspect="equal")
        ax.axhline((row + 0.5) * ext[2] / coh.shape[0], color=INK, lw=1.4)
        ax.set_title(f"Wrapped phase; black line = row {row}")
        ax.set_ylabel("azimuth (km)")
        ax.set_xlabel("ground range (km)")
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(x, ph[row], color=GRID_C, lw=0.8, zorder=1)
        sc = ax2.scatter(x, ph[row], c=coh[row], cmap=seq, vmin=0, vmax=1, s=9, zorder=2)
        fig.colorbar(sc, ax=ax2, fraction=0.02, pad=0.12, label="coherence")
        ax2.set_ylim(-np.pi * 1.05, np.pi * 1.05)
        ax2.set_yticks([-np.pi, 0, np.pi])
        ax2.set_yticklabels(["-π", "0", "+π"])
        ax2.set_ylabel("wrapped phase (rad)")
        sec = ax2.secondary_yaxis("right", functions=(lambda y: y * mm_per_rad, lambda y: y / mm_per_rad))
        sec.set_ylabel("same phase as LOS change (mm)")
        ax2.set_title(f"Phase along that line. One jump from +π to -π is one fringe = {mm_per_fringe:.1f} mm of "
                      "line-of-sight change. Dark dots = trustworthy phase, pale dots = noise")
        self._tidy(ax2, "y")
        ax3 = fig.add_subplot(gs[2], sharex=ax2)
        ax3.plot(x, coh[row], color=C_S1A, lw=1)
        ax3.axhline(0.3, color=INK2, ls="--", lw=0.8)
        ax3.text(1.0, 0.3, " 0.3", fontsize=8, color=INK2, va="center", transform=ax3.get_yaxis_transform())
        ax3.set_ylim(0, 1)
        ax3.set_ylabel("coherence")
        ax3.set_xlabel("ground range (km)")
        self._tidy(ax3, "y")
        self._title(fig, "V8  " + self._caption(pair, r, coh), size=10.5, y=0.99)
        return self._save(fig, name or f"V8_fringe_profile_{pair[0]}_{pair[1]}")

    # ------------------------------------------------------------------ V9
    @_figure
    def unwrap_panel(self, pair: Pair, interferograms: Mapping[Pair, Any], corrected: Mapping[Pair, Any],
                     unwrapped: Mapping[Pair, Any], conncomp: Mapping[Pair, Any],
                     pyunwrap: Optional[Mapping[Pair, Any]] = None, coh_min: float = 0.3,
                     name: Optional[str] = None) -> Path:
        """Wrapped phase, coherence, SNAPHU reliable mask, SNAPHU mm, pyunwrap mm, and their difference in cycles.

        ``corrected`` holds the complex (atmosphere-corrected) interferograms; ``unwrapped``/``conncomp`` the SNAPHU
        output on the unwrapping grid; ``pyunwrap`` the optional second unwrapping on the same grid.
        """
        from pygeofetch.insar.unwrap import multilook

        plt, LSC, _, _, Patch, _ = _mpl()
        _, div = _cmaps()
        ual, url_ = self.unwrap_looks
        la, lr = self.looks
        g = self.geo
        ph_ml = multilook(corrected[pair], ual, url_, wrapped_phase=True)
        if np.iscomplexobj(ph_ml):  # multilook of a complex interferogram returns the complex mean
            ph_ml = np.angle(ph_ml)
        coh_ml = multilook(interferograms[pair].coherence, ual, url_, wrapped_phase=False)
        unw, cc = np.asarray(unwrapped[pair], float), np.asarray(conncomp[pair])
        pyu = (pyunwrap or {}).get(pair)
        pyu = None if pyu is None else np.asarray(pyu, float)
        shapes = [ph_ml.shape, coh_ml.shape, unw.shape, cc.shape] + ([pyu.shape] if pyu is not None else [])
        h, w = min(s[0] for s in shapes), min(s[1] for s in shapes)
        ph_ml, coh_ml, unw, cc = ph_ml[:h, :w], coh_ml[:h, :w], unw[:h, :w], cc[:h, :w]
        pyu = None if pyu is None else pyu[:h, :w]
        ext = self._km_extent((h, w), la * ual, lr * url_)
        reliable = (cc != 0) & np.isfinite(unw)
        fig, axs = plt.subplots(2, 3, figsize=(14.5, 7.2))
        ax = axs[0, 0]
        ax.imshow(np.angle(np.exp(1j * ph_ml)), cmap=PHASE_CMAP, vmin=-np.pi, vmax=np.pi, extent=ext, aspect="equal")
        ax.set_title("Wrapped phase after atmospheric correction")
        axs[0, 1].imshow(coh_ml, cmap="gray", vmin=0, vmax=1, extent=ext, aspect="equal")
        axs[0, 1].set_title("Coherence on the unwrapping grid")
        ax = axs[0, 2]
        ax.imshow(reliable, cmap=LSC.from_list("rel", [C_OUT, C_IN]), vmin=0, vmax=1, extent=ext, aspect="equal")
        ax.set_title(f"SNAPHU reliable region: {reliable.mean() * 100:.1f}% of pixels")
        ax.legend(handles=[Patch(fc=C_IN, label="reliable component"), Patch(fc=C_OUT, label="unreliable (conncomp 0)")],
                  loc="lower left", frameon=True, fontsize=7)

        def _mm(a, mask):
            if mask.sum() < 20:
                return None, 1.0
            rel = a * g.mm_per_rad
            rel = rel - np.nanmedian(rel[mask])
            return np.where(mask, rel, np.nan), max(1.0, np.nanpercentile(np.abs(rel[mask]), 98))

        py_mask = None if pyu is None else (np.isfinite(pyu) & (coh_ml >= coh_min))
        for ax, arr, mask, ttl in ((axs[1, 0], unw, reliable, "SNAPHU unwrapped (LOS mm, relative)"),
                                   (axs[1, 1], pyu, py_mask, "pyunwrap (LOS mm, relative)")):
            if arr is None:
                ax.axis("off")
                ax.text(0.5, 0.5, "pyunwrap not run yet", ha="center", va="center", color=INK2)
                continue
            z, lim = _mm(arr, mask)
            if z is None:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{ttl}\nno pixel passes the reliability test", ha="center", va="center", color=INK2)
                continue
            im = ax.imshow(z, cmap=div, vmin=-lim, vmax=lim, extent=ext, aspect="equal")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="mm")
            ax.set_title(ttl)
        ax = axs[1, 2]
        if pyu is not None:
            both = reliable & np.isfinite(pyu)
            if both.sum() >= 20:
                diff = (pyu - unw) / (2 * np.pi)
                diff = diff - np.nanmedian(diff[both])
                same = (np.abs(diff[both]) < 0.5).mean() * 100
                im = ax.imshow(np.where(both, diff, np.nan), cmap=div, vmin=-3, vmax=3, extent=ext, aspect="equal")
                fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="cycles")
                ax.set_title(f"pyunwrap minus SNAPHU: same cycle at {same:.0f}% of {both.sum()} px")
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "too few pixels valid in both methods to compare", ha="center", va="center", color=INK2)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "comparison appears after the pyunwrap stage", ha="center", va="center", color=INK2)
        for a in axs.ravel():
            if a.axison:
                a.set_xlabel("ground range (km)")
        for a in axs[:, 0]:
            a.set_ylabel("azimuth (km)")
        self._title(fig, f"V9  Unwrapping  {pair[0]} -> {pair[1]}   (1 fringe = {g.mm_per_fringe:.1f} mm LOS; "
                    "no ground truth, so differences are not errors)", size=10.5, y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return self._save(fig, name or f"V9_unwrap_{pair[0]}_{pair[1]}")

    # ------------------------------------------------------------------ V10
    @_figure
    def velocity_summary(self, ts, name: str = "V10_velocity_summary") -> Path:
        """SBAS line-of-sight velocity (mm/yr), its distribution, and the per-pixel fit residual RMS.

        ``ts.velocity`` is in m/yr and ``ts.residual_rms`` in m (TimeSeriesResult). The residual map is the
        inversion misfit, not a velocity standard error. A ``velocity_std`` attribute is drawn instead if present.
        """
        plt, *_ = _mpl()
        seq, div = _cmaps()
        v = np.asarray(ts.velocity, float) * 1000.0
        fin = v[np.isfinite(v)]
        ext = self._km_extent(v.shape, self.looks[0] * self.unwrap_looks[0], self.looks[1] * self.unwrap_looks[1])
        lim = max(1.0, np.nanpercentile(np.abs(fin), 98)) if fin.size else 1.0
        third, third_label = None, ""
        for attr, label in (("velocity_std", "Velocity standard error"), ("residual_rms", "Inversion residual RMS (misfit)")):
            cand = getattr(ts, attr, None)
            if cand is not None and np.shape(cand) == v.shape:
                third, third_label = np.asarray(cand, float) * 1000.0, label
                break
        ncol = 3 if third is not None else 2
        fig, axs = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.8))
        im = axs[0].imshow(v, cmap=div, vmin=-lim, vmax=lim, extent=ext, aspect="equal")
        fig.colorbar(im, ax=axs[0], fraction=0.04, pad=0.02, label="mm/yr")
        axs[0].set_title(f"Velocity: {np.isfinite(v).mean() * 100:.1f}% of pixels carry a value")
        axs[0].set_xlabel("ground range (km)")
        axs[0].set_ylabel("azimuth (km)")
        axs[1].hist(fin, bins=60, color=C_S1A, alpha=0.9)
        axs[1].axvline(0, color=INK2, lw=0.8)
        axs[1].set_xlabel("velocity (mm/yr)")
        axs[1].set_ylabel("pixels")
        axs[1].set_title("Distribution (reliable pixels only)")
        self._tidy(axs[1], "y")
        if third is not None:
            im = axs[2].imshow(third, cmap=seq, extent=ext, aspect="equal", vmin=0)
            fig.colorbar(im, ax=axs[2], fraction=0.04, pad=0.02, label="mm" if "residual" in third_label.lower() else "mm/yr")
            axs[2].set_title(third_label)
            axs[2].set_xlabel("ground range (km)")
        self._title(fig, "V10  SBAS velocity (NaN = not reliable, shown blank)", size=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return self._save(fig, name)

    # ------------------------------------------------------------------ convenience
    def interferogram_report(self, interferograms: Mapping[Pair, Any], pairs: Optional[Mapping[str, Pair]] = None,
                             all_pairs: bool = False, inference_tile: Optional[Tuple[int, int]] = (256, 128)) -> Dict[str, Any]:
        """Every interferogram figure in one call: sheets, tiles and profiles for representative pairs
        (or all pairs), the stack diagnostics, the coherence sheet and, optionally, the unwrapper tile windows."""
        pairs = dict(pairs or self.representative_pairs(interferograms))
        chosen = list(interferograms) if all_pairs else list(dict.fromkeys(pairs.values()))
        out: Dict[str, Any] = {"pairs": pairs}
        for p in chosen:
            self.pair_sheet(p, interferograms)
            self.tile_quality(p, interferograms)
            self.fringe_profile(p, interferograms)
        if inference_tile:
            self.inference_tiles(pairs["first"], interferograms, tile=inference_tile[0], overlap=inference_tile[1])
        out["table"] = self.stack_diagnostics(interferograms)
        self.coherence_sheet(interferograms)
        return out


__all__ = ["InSARFigures", "SarGeometry", "relative_orbit"]