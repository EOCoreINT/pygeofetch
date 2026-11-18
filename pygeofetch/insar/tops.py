"""
Sentinel-1 TOPS azimuth deramping, for resampling and shifting IW SLC bursts without corrupting their phase.

Why this is needed
------------------
In a TOPS (IW/EW) SLC burst the azimuth spectrum is not at baseband: the Doppler centroid sweeps linearly
through the burst, by about 1.7 kHz per second of azimuth time (roughly 4.5-5 kHz over one IW burst, ten
times the 486 Hz azimuth sampling rate). Interpolating such a signal directly (linear, spline or sinc) or
shifting it with a Fourier phase ramp treats it as if it were at baseband, which

* attenuates it where the local Doppler is far from zero (coherence loss that varies within each burst), and
* leaves a phase error of ``2 pi f_DC(eta) * shift`` that grows linearly through the burst and jumps at burst
  boundaries: dense fringes along azimuth and straight discontinuities along the burst edges.

The standard remedy (ESA technical note S1-TN-MDA-52-7445, "Sentinel-1 TOPS SLC deramping"; Miranda 2017;
the ISCE2 and SNAP TOPS chains) is to multiply each burst by the conjugate of its known azimuth phase ramp,
interpolate or shift the deramped (baseband) signal, and multiply the result by the ramp evaluated at the
positions actually sampled ("reramp").

The ramp of burst b at azimuth time ``eta`` (relative to the burst centre) and two-way range time ``tau``::

    k_s       = 2 v / lambda * k_psi                 (Doppler rate of the antenna steering)
    alpha     = 1 - k_s / k_a(tau)                   (k_a: azimuth FM rate, negative)
    k_t       = k_s / alpha                          (Doppler centroid rate in the focused burst, ~ +1.7 kHz/s)
    eta_c     = -f_dc(tau) / k_a(tau)                (beam-centre crossing time)
    eta_ref   = eta_c(tau) - eta_c(tau_mid)
    psi       = pi k_t (eta - eta_ref)^2 + 2 pi f_dc (eta - eta_ref)

The local Doppler centroid of the burst is ``dpsi/deta / 2 pi = f_dc + k_t (eta - eta_ref)``. Which sign of
``psi`` the data carry depends on the processor's phase convention; :meth:`TopsRamp.calibrate_sign` measures it
on the data themselves (the right sign gives a baseband signal with a strong lag-one azimuth correlation, the
wrong sign doubles the sweep), so a convention error cannot silently corrupt the phase.

Rows here are full-swath rows of the measurement TIFF, in which burst ``b`` occupies rows
``b * linesPerBurst ... (b + 1) * linesPerBurst - 1`` (burst overlaps not yet removed), and columns are
full-swath range samples, the same coordinates as ``read_crop_offset`` and the coregistration offset field.
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("pygeofetch.insar.tops")

SPEED_OF_LIGHT = 299792458.0


@dataclass
class _Poly:
    """A range polynomial ``sum_k c_k (tau - t0)^k`` valid around azimuth time ``time``."""

    time: datetime
    t0: float
    coeffs: Tuple[float, ...]

    def __call__(self, tau: np.ndarray) -> np.ndarray:
        x = np.asarray(tau, dtype=np.float64) - self.t0
        out = np.zeros_like(x)
        for c in reversed(self.coeffs):
            out = out * x + c
        return out


@dataclass
class TopsRamp:
    """Azimuth phase ramp of every burst of one TOPS sub-swath, from its annotation XML."""

    radar_frequency_hz: float
    azimuth_steering_rate_rad_s: float
    azimuth_time_interval_s: float
    range_sampling_rate_hz: float
    slant_range_time_s: float
    lines_per_burst: int
    samples_per_burst: int
    burst_times: List[datetime]
    fm_rates: List[_Poly]
    dc_estimates: List[_Poly]
    orbit: List[Tuple[datetime, np.ndarray]]
    sign: float = 1.0
    _cache: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ geometry

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT / self.radar_frequency_hz

    @property
    def n_bursts(self) -> int:
        return len(self.burst_times)

    def burst_mid_time(self, b: int) -> datetime:
        from datetime import timedelta

        return self.burst_times[b] + timedelta(seconds=0.5 * self.lines_per_burst * self.azimuth_time_interval_s)

    def _nearest(self, polys: Sequence[_Poly], t: datetime) -> _Poly:
        return min(polys, key=lambda p: abs((p.time - t).total_seconds()))

    def _speed(self, t: datetime) -> float:
        if not self.orbit:
            return 7590.0  # typical Sentinel-1 earth-fixed speed; only enters k_s, weakly
        times = np.array([(o[0] - t).total_seconds() for o in self.orbit])
        speeds = np.array([float(np.linalg.norm(o[1])) for o in self.orbit])
        order = np.argsort(times)
        return float(np.interp(0.0, times[order], speeds[order]))

    def _burst_terms(self, b: int, cols: np.ndarray):
        """k_t, f_dc and eta_ref (each per column) of burst ``b``; cached per burst and column set."""
        cols = np.ascontiguousarray(cols, dtype=np.float64)
        key = (b, cols.size, hash(cols.tobytes()))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        mid = self.burst_mid_time(b)
        tau = self.slant_range_time_s + np.asarray(cols, dtype=np.float64) / self.range_sampling_rate_hz
        tau_mid = self.slant_range_time_s + 0.5 * self.samples_per_burst / self.range_sampling_rate_hz
        k_a_poly = self._nearest(self.fm_rates, mid)
        dc_poly = self._nearest(self.dc_estimates, mid)
        k_s = 2.0 * self._speed(mid) / self.wavelength_m * self.azimuth_steering_rate_rad_s
        k_a = k_a_poly(tau)
        k_t = k_s / (1.0 - k_s / k_a)
        f_dc = dc_poly(tau)
        eta_c = -f_dc / k_a
        eta_c_mid = float(-dc_poly(np.array([tau_mid]))[0] / k_a_poly(np.array([tau_mid]))[0])
        eta_ref = eta_c - eta_c_mid
        out = (k_t, f_dc, eta_ref)
        if len(self._cache) > 256:
            self._cache.clear()
        self._cache[key] = out
        return out

    def doppler_rate_hz_s(self) -> float:
        """Median Doppler centroid rate k_t over the swath (Hz/s), for diagnostics."""
        cols = np.linspace(0, self.samples_per_burst - 1, 16)
        return float(np.median(self._burst_terms(self.n_bursts // 2, cols)[0]))

    def overlap_doppler_difference_hz(self, cols: Optional[np.ndarray] = None) -> float:
        """Doppler centroid of burst i+1 minus that of burst i for the same ground in their overlap (Hz).

        A ground point at absolute time T is seen at ``eta = T - mid_b`` in each burst, so the difference is
        ``k_t (mid_i - mid_{i+1})``: about -k_t times the burst cycle time, roughly -4.5 kHz for IW.
        """
        if self.n_bursts < 2:
            return float("nan")
        if cols is None:
            cols = np.linspace(0, self.samples_per_burst - 1, 16)
        b = max(0, self.n_bursts // 2 - 1)
        k_t = self._burst_terms(b, np.asarray(cols, dtype=np.float64))[0]
        cycle = (self.burst_mid_time(b + 1) - self.burst_mid_time(b)).total_seconds()
        return float(-np.median(k_t) * cycle)

    # ------------------------------------------------------------------ ramp

    def phase(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Ramp phase psi (rad, float64) at full-swath (row, col); fractional rows/cols allowed.

        ``rows`` and ``cols`` broadcast against each other. The burst of a row is ``floor(row / linesPerBurst)``
        (clipped to the valid bursts).
        """
        rows = np.asarray(rows, dtype=np.float64)
        cols = np.asarray(cols, dtype=np.float64)
        rows, cols = np.broadcast_arrays(rows, cols)
        out = np.empty(rows.shape, dtype=np.float64)
        lpb = self.lines_per_burst
        bursts = np.clip(np.floor(rows / lpb).astype(np.int64), 0, self.n_bursts - 1)
        for b in np.unique(bursts):
            sel = bursts == b
            r = rows[sel]
            c = cols[sel]
            # Columns repeat across rows in all our callers; evaluate the range terms once per distinct column.
            uc, inv = np.unique(np.round(c, 3), return_inverse=True)
            k_t, f_dc, eta_ref = self._burst_terms(int(b), uc)
            eta = (r - b * lpb - 0.5 * lpb) * self.azimuth_time_interval_s
            d = eta - eta_ref[inv]
            out[sel] = math.pi * k_t[inv] * d * d + 2.0 * math.pi * f_dc[inv] * d
        return self.sign * out

    def deramp(self, data: np.ndarray, row_offset: float, col_offset: float, inverse: bool = False,
               chunk_rows: int = 512) -> np.ndarray:
        """Multiply ``data`` (a crop whose row 0/col 0 are full-swath row_offset/col_offset) by exp(-+ i psi), in place."""
        h, w = data.shape
        cols = np.arange(w, dtype=np.float64) + col_offset
        s = 1.0 if inverse else -1.0
        for r0 in range(0, h, chunk_rows):
            r1 = min(h, r0 + chunk_rows)
            rows = (np.arange(r0, r1, dtype=np.float64) + row_offset)[:, None]
            data[r0:r1] *= np.exp(1j * s * self.phase(rows, cols[None, :])).astype(np.complex64)
        return data

    # ------------------------------------------------------------------ sign check

    def calibrate_sign(self, data: np.ndarray, row_offset: float, col_offset: float,
                       n_cols: int = 256) -> dict:
        """Choose the sign of psi from the data: the deramped burst must be a baseband signal.

        For each sign, deramp a few columns of every burst fully inside ``data`` and measure the coherent lag-one
        azimuth correlation ``|sum z[r+1] conj(z[r])| / sum |z|^2``. A baseband signal of ~2/3 of the sampling
        bandwidth gives roughly 0.3-0.5; a signal whose Doppler still sweeps (no deramp, or the wrong sign, which
        doubles the sweep) averages to near 0. Sets ``self.sign`` and returns the scores.
        """
        h, w = data.shape
        lpb = self.lines_per_burst
        col_idx = np.unique(np.linspace(0, w - 1, min(n_cols, w)).astype(int))
        scores = {}
        for sign in (1.0, -1.0, 0.0):
            num, den = 0.0 + 0.0j, 0.0
            for b in range(self.n_bursts):
                r0 = int(math.ceil(b * lpb - row_offset))
                r1 = int(math.floor((b + 1) * lpb - row_offset))
                r0, r1 = max(r0, 0), min(r1, h)
                if r1 - r0 < 64:
                    continue
                # keep away from the burst edges (zero-filled invalid lines)
                r0, r1 = r0 + (r1 - r0) // 8, r1 - (r1 - r0) // 8
                z = data[r0:r1][:, col_idx].astype(np.complex128)
                rows = (np.arange(r0, r1, dtype=np.float64) + row_offset)[:, None]
                if sign != 0.0:
                    psi = self.phase(rows, (col_idx + col_offset)[None, :].astype(np.float64)) / self.sign
                    z = z * np.exp(-1j * sign * psi)
                num += np.sum(z[1:] * np.conj(z[:-1]))
                den += float(np.sum(np.abs(z) ** 2))
            scores[{1.0: "plus", -1.0: "minus", 0.0: "none"}[sign]] = abs(num) / den if den > 0 else float("nan")
        best = max(("plus", "minus"), key=lambda k: scores[k])
        self.sign = 1.0 if best == "plus" else -1.0
        scores["chosen_sign"] = self.sign
        scores["clear"] = bool(scores[best] > 2.0 * max(scores["none"], scores["minus" if best == "plus" else "plus"], 1e-6))
        if not scores["clear"]:
            logger.warning("TOPS deramp sign check is not clear-cut (lag-one correlation +%.3f / -%.3f / none %.3f); "
                           "using sign %+d", scores["plus"], scores["minus"], scores["none"], int(self.sign))
        else:
            logger.info("TOPS deramp sign check: lag-one correlation +%.3f / -%.3f / none %.3f -> sign %+d",
                        scores["plus"], scores["minus"], scores["none"], int(self.sign))
        return scores


# ---------------------------------------------------------------------- annotation parsing


def _poly_from(elem: ET.Element, name: str) -> Tuple[float, ...]:
    node = elem.find(name)
    if node is not None and node.text and node.text.strip():
        return tuple(float(v) for v in node.text.split())
    coeffs = []  # pre-IPF-2.4 annotation: separate c0, c1, c2 elements
    for k in range(4):
        c = elem.find(f"c{k}")
        if c is None or not c.text:
            break
        coeffs.append(float(c.text))
    if not coeffs:
        raise ValueError(f"no {name} polynomial")
    return tuple(coeffs)


def _annotation_root(safe_zip_path: Union[str, Path, Any], member_hint: Optional[str]) -> ET.Element:
    with zipfile.ZipFile(safe_zip_path) as zf:
        candidates = [
            n for n in zf.namelist()
            if "/annotation/" in n and n.lower().endswith(".xml") and "/calibration/" not in n.lower()
            and not Path(n).name.lower().startswith("rfi-") and "/rfi/" not in n.lower()
        ]
        if member_hint:
            filtered = [n for n in candidates if member_hint.lower() in n.lower()]
            if filtered:
                candidates = filtered
        if not candidates:
            raise ValueError(f"{safe_zip_path}: no annotation XML found")
        with zf.open(candidates[0]) as f:
            return ET.parse(f).getroot()


def tops_ramp_from_root(root: ET.Element) -> TopsRamp:
    """Build a :class:`TopsRamp` from a parsed annotation XML root (IW/EW SLC)."""

    def text(path: str) -> str:
        node = root.find(path)
        if node is None or node.text is None:
            raise ValueError(f"annotation field missing: {path}")
        return node.text.strip()

    mode = root.findtext(".//adsHeader/mode") or ""
    if mode and mode.strip().upper() not in ("IW", "EW"):
        raise ValueError(f"not a TOPS product (mode {mode!r})")
    steering_deg = float(text(".//generalAnnotation/productInformation/azimuthSteeringRate"))
    fm = [
        _Poly(datetime.fromisoformat(e.findtext("azimuthTime").strip()), float(e.findtext("t0")),
              _poly_from(e, "azimuthFmRatePolynomial"))
        for e in root.findall(".//generalAnnotation/azimuthFmRateList/azimuthFmRate")
    ]
    dc = [
        _Poly(datetime.fromisoformat(e.findtext("azimuthTime").strip()), float(e.findtext("t0")),
              _poly_from(e, "dataDcPolynomial"))
        for e in root.findall(".//dopplerCentroid/dcEstimateList/dcEstimate")
    ]
    if not fm or not dc:
        raise ValueError("azimuth FM rate or Doppler centroid list missing")
    orbit = []
    for o in root.findall(".//generalAnnotation/orbitList/orbit"):
        try:
            v = np.array([float(o.findtext(f"velocity/{a}")) for a in "xyz"])
            orbit.append((datetime.fromisoformat(o.findtext("time").strip()), v))
        except (TypeError, ValueError):
            continue
    bursts = [datetime.fromisoformat(b.findtext("azimuthTime").strip())
              for b in root.findall(".//swathTiming/burstList/burst")]
    if len(bursts) < 1:
        raise ValueError("no bursts in swathTiming")
    return TopsRamp(
        radar_frequency_hz=float(text(".//generalAnnotation/productInformation/radarFrequency")),
        azimuth_steering_rate_rad_s=math.radians(steering_deg),
        azimuth_time_interval_s=float(text(".//imageAnnotation/imageInformation/azimuthTimeInterval")),
        range_sampling_rate_hz=float(text(".//generalAnnotation/productInformation/rangeSamplingRate")),
        slant_range_time_s=float(text(".//imageAnnotation/imageInformation/slantRangeTime")),
        lines_per_burst=int(text(".//swathTiming/linesPerBurst")),
        samples_per_burst=int(text(".//swathTiming/samplesPerBurst")),
        burst_times=bursts,
        fm_rates=fm,
        dc_estimates=dc,
        orbit=orbit,
    )


def parse_tops_ramp(safe_zip_path: Union[str, Path, Any], member_hint: Optional[str] = None) -> TopsRamp:
    """Parse the TOPS azimuth ramp parameters of one sub-swath from a SAFE zip (same member selection as
    :func:`pygeofetch.insar.annotation.parse_slc_geometry`)."""
    ramp = tops_ramp_from_root(_annotation_root(safe_zip_path, member_hint))
    logger.info("TOPS ramp parameters: %d bursts, %d lines/burst, steering %.4f deg/s, k_t %.0f Hz/s",
                ramp.n_bursts, ramp.lines_per_burst, math.degrees(ramp.azimuth_steering_rate_rad_s),
                ramp.doppler_rate_hz_s())
    return ramp


# ---------------------------------------------------------------------- azimuth shift


def shift_azimuth_tops(data: np.ndarray, shift_px: float, content_phase, lines_per_burst: int,
                       row_offset: float, col_block: int = 512) -> np.ndarray:
    """Shift ``data`` along azimuth by ``shift_px`` (output row r = input row r - shift_px), TOPS-correctly.

    ``content_phase(rows_local, cols_local)`` gives the ramp phase carried by the data at (fractional) local
    positions. Each burst segment (rows of one full-swath burst, located with ``row_offset``) is deramped,
    shifted with a Fourier phase ramp (now valid: the signal is at baseband) and reramped with the phase at the
    positions it came from, ``r - shift_px``. Segments are shifted separately so no burst leaks into the next.
    """
    h, w = data.shape
    out = data if (data.dtype == np.complex64 and data.flags.writeable) else data.astype(np.complex64)
    starts = sorted({0, h} | {int(math.ceil(b * lines_per_burst - row_offset))
                              for b in range(int(row_offset // lines_per_burst) + 1,
                                             int((row_offset + h) // lines_per_burst) + 1)})
    starts = [s for s in starts if 0 <= s <= h]
    for r0, r1 in zip(starts[:-1], starts[1:]):
        n = r1 - r0
        if n < 2:
            continue
        freq = np.fft.fftfreq(n).reshape(-1, 1)
        ramp = np.exp(-2j * np.pi * freq * shift_px)
        rows = np.arange(r0, r1, dtype=np.float64)[:, None]
        for c0 in range(0, w, col_block):
            c1 = min(w, c0 + col_block)
            cols = np.arange(c0, c1, dtype=np.float64)[None, :]
            block = out[r0:r1, c0:c1].astype(np.complex128) * np.exp(-1j * content_phase(rows, cols))
            block = np.fft.ifft(np.fft.fft(block, axis=0) * ramp, axis=0)
            block *= np.exp(1j * content_phase(rows - shift_px, cols))
            out[r0:r1, c0:c1] = block.astype(np.complex64)
    return out


def shift_azimuth_tops_field(data: np.ndarray, shift_fn, content_phase, lines_per_burst: int, row_offset: float,
                             col_block: int = 256) -> np.ndarray:
    """Like shift_azimuth_tops, but by a smoothly varying shift ``shift_fn(rows, cols)`` (local coordinates, px).

    Within each burst segment the data are deramped; the field's mean over the array is applied with an exact
    Fourier shift, the small remainder d(r, c) with a second-order Taylor step, s(r - m - d) ~ g - d g' + d^2/2 g''
    with g the Fourier-shifted segment and g', g'' its Fourier derivatives (error O(d^3), negligible for the
    hundredths of a pixel ESD fields carry). The data are reramped with the phase at the positions they came from.
    """
    h, w = data.shape
    out = data if (data.dtype == np.complex64 and data.flags.writeable) else data.astype(np.complex64)
    rr = np.linspace(0, h - 1, 9)[:, None]
    cc = np.linspace(0, w - 1, 9)[None, :]
    mean = float(np.mean(shift_fn(rr + 0 * cc, cc + 0 * rr)))
    starts = sorted({0, h} | {int(math.ceil(b * lines_per_burst - row_offset))
                              for b in range(int(row_offset // lines_per_burst) + 1,
                                             int((row_offset + h) // lines_per_burst) + 1)})
    starts = [s for s in starts if 0 <= s <= h]
    for r0, r1 in zip(starts[:-1], starts[1:]):
        n = r1 - r0
        if n < 2:
            continue
        freq = np.fft.fftfreq(n).reshape(-1, 1)
        ramp = np.exp(-2j * np.pi * freq * mean)
        d1 = 2j * np.pi * freq
        rows = np.arange(r0, r1, dtype=np.float64)[:, None]
        for c0 in range(0, w, col_block):
            c1 = min(w, c0 + col_block)
            cols = np.arange(c0, c1, dtype=np.float64)[None, :]
            shift = np.asarray(shift_fn(rows + 0 * cols, cols + 0 * rows), dtype=np.float64)
            d = shift - mean
            spec = np.fft.fft(out[r0:r1, c0:c1].astype(np.complex128) * np.exp(-1j * content_phase(rows, cols)),
                              axis=0) * ramp
            g = np.fft.ifft(spec, axis=0)
            g1 = np.fft.ifft(spec * d1, axis=0)
            g2 = np.fft.ifft(spec * d1 * d1, axis=0)
            block = g - d * g1 + 0.5 * d * d * g2
            block *= np.exp(1j * content_phase(rows - shift, cols))
            out[r0:r1, c0:c1] = block.astype(np.complex64)
    return out
