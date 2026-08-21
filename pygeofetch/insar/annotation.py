"""
Sentinel-1 annotation XML parsing — real per-pixel acquisition timing.

This is the missing link between "a pixel at (row, col) in an extracted
SLC GeoTIFF" and "the exact acquisition time and slant range needed to
feed geolocation.py's orbit-based solver" — that information does not
exist in the GeoTIFF itself (a general raster format); it lives in a
separate XML file inside the .SAFE archive.

CRITICAL (TOPS burst timing): Sentinel-1 IW is a TOPS acquisition. The
azimuth time axis is NOT a single linear ramp across the whole scene.
Consecutive bursts overlap in time (~0.3 s for IW), so a global
`first_line_time + row * interval` model drifts by up to ~2.7 s
(~1300 rows) by the last burst. That drift is exactly what caused the
chronic cross-family misregistration / low coherence. When per-burst
azimuth times are available (parsed from swathTiming) the geometry uses
a piecewise, burst-correct model; otherwise it falls back to linear.

Field paths confirmed against ESA's own Mission Performance Centre
documentation (MPC-0392) and Product Specification (MPC-0240):
/product/imageAnnotation/imageInformation/productFirstLineUtcTime
/product/imageAnnotation/imageInformation/azimuthTimeInterval
/product/imageAnnotation/imageInformation/slantRangeTime
/product/imageAnnotation/imageInformation/numberOfLines
/product/imageAnnotation/imageInformation/numberOfSamples
/product/generalAnnotation/productInformation/rangeSamplingRate
/product/swathTiming/linesPerBurst
/product/swathTiming/burstList/burst/azimuthTime
"""
from __future__ import annotations

import bisect
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional, Union

logger = logging.getLogger("pygeofetch.insar.annotation")

SPEED_OF_LIGHT = 299792458.0


@dataclass
class SLCGeometry:
    """
    Real per-pixel acquisition geometry for one Sentinel-1 SLC measurement,
    parsed from its annotation XML — everything needed to convert a pixel
    (row, col) into an exact acquisition time and slant range.

    CRITICAL (TOPS burst timing): Sentinel-1 IW is a TOPS acquisition.
    The azimuth time axis is NOT a single linear ramp across the whole
    scene. Consecutive bursts overlap in time (~10% of burst duration
    for IW), so a global `first_line_time + row * interval` model
    drifts increasingly as row increases -- confirmed directly: this
    project's own real Mexico City data showed drift on the order of
    ~1000-1500 rows by the last burst, which was the direct cause of a
    real, previously-unexplained chronic coregistration/coherence
    problem for certain date pairs. When per-burst azimuth times are
    available (parsed from swathTiming), row<->time conversions use the
    correct piecewise model below instead; otherwise this falls back to
    the plain linear model.
    """
    first_line_time: datetime
    azimuth_time_interval_s: float
    near_range_time_s: float
    range_sampling_rate_hz: float
    n_lines: int
    n_columns: int
    burst_azimuth_times: Optional[List[datetime]] = None
    lines_per_burst: Optional[int] = None

    def _has_burst_timing(self) -> bool:
        return (
            self.burst_azimuth_times is not None
            and self.lines_per_burst is not None
            and self.lines_per_burst > 0
            and len(self.burst_azimuth_times) > 1
        )

    def azimuth_time(self, row: float) -> datetime:
        """Exact acquisition time of a given (possibly fractional) row.

        Burst-correct when burst timing is available: locate the burst
        containing `row` (row // lines_per_burst, clamped to the real
        burst count), then add the within-burst linear offset from that
        burst's own real azimuthTime. Falls back to a single linear
        ramp from first_line_time otherwise.
        """
        if self._has_burst_timing():
            L = self.lines_per_burst
            b = max(0, min(int(row // L), len(self.burst_azimuth_times) - 1))
            return self.burst_azimuth_times[b] + timedelta(
                seconds=(row - b * L) * self.azimuth_time_interval_s)
        return self.first_line_time + timedelta(
            seconds=row * self.azimuth_time_interval_s)

    def range_time(self, col: float) -> float:
        """Two-way slant range time (seconds) of a given (possibly
        fractional) column."""
        return self.near_range_time_s + col / self.range_sampling_rate_hz

    def slant_range_m(self, col: float) -> float:
        """One-way slant range (metres) of a given column."""
        return self.range_time(col) * SPEED_OF_LIGHT / 2.0

    def row_for_azimuth_time(self, t: datetime) -> float:
        """Inverse of azimuth_time(): which (fractional) row corresponds
        to a given acquisition time.

        Burst-correct when burst timing is available: find the burst
        whose real azimuthTime is the latest one <= t (via bisect on the
        sorted burst start times), then compute the within-burst row
        from that burst's own start time. This is the function whose
        previous purely-linear behaviour caused the real coregistration
        drift described in the class docstring above.
        """
        if self._has_burst_timing():
            L = self.lines_per_burst
            b = bisect.bisect_right(self.burst_azimuth_times, t) - 1
            b = max(0, min(b, len(self.burst_azimuth_times) - 1))
            return (b * L + (t - self.burst_azimuth_times[b]).total_seconds()
                    / self.azimuth_time_interval_s)
        return ((t - self.first_line_time).total_seconds()
                / self.azimuth_time_interval_s)

    def col_for_range_time(self, range_time_s: float) -> float:
        """Inverse of range_time(): which (fractional) column corresponds
        to a given two-way range time."""
        return (range_time_s - self.near_range_time_s) * self.range_sampling_rate_hz


def _parse_swath_timing(root) -> tuple[Optional[List[datetime]], Optional[int]]:
    """Extract (burst_azimuth_times, lines_per_burst) from an annotation
    XML root, or (None, None) if swathTiming is absent/incomplete."""
    try:
        lpb_el = root.find(".//swathTiming/linesPerBurst")
        burst_elems = root.findall(".//swathTiming/burstList/burst")
        if lpb_el is None or not burst_elems:
            return None, None
        lines_per_burst = int(lpb_el.text)
        burst_azimuth_times: List[datetime] = []
        for b in burst_elems:
            az = b.find("azimuthTime")
            if az is not None and az.text:
                burst_azimuth_times.append(datetime.fromisoformat(az.text.strip()))
        if not burst_azimuth_times:
            return None, None
        return burst_azimuth_times, lines_per_burst
    except Exception as exc:
        logger.warning("Could not parse swathTiming burst azimuth times: %s", exc)
        return None, None


import xml.etree.ElementTree as ET

def _extract_burst_timing_from_root(root: ET.Element) -> list[dict]:
    """
    Extract azimuth and sensing times from the burstList in the annotation XML.
    """
    burst_timings = []
    
    # Navigate to the burstList based on the ESA Sentinel-1 schema
    burst_list = root.find(".//swathTiming/burstList")
    if burst_list is None:
        return burst_timings
        
    for burst in burst_list.findall("burst"):
        az_time_str = burst.findtext("azimuthTime")
        sensing_time_str = burst.findtext("sensingTime")
        
        burst_timings.append({
            "azimuthTime": az_time_str,
            "sensingTime": sensing_time_str,
            # Add other fields like byteOffset, firstValidSample, etc. if needed
        })
        
    return burst_timings

def parse_slc_geometry(
    safe_zip_path: Union[str, Path], member_hint: Optional[str] = None
) -> SLCGeometry:
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(safe_zip_path) as zf:
        candidates = [
            n for n in zf.namelist()
            if "/annotation/" in n and n.lower().endswith(".xml")
            and "/calibration/" not in n.lower()
            and not Path(n).name.lower().startswith("rfi-")
            and "/rfi/" not in n.lower()
        ]
        if member_hint:
            filtered = [n for n in candidates if member_hint.lower() in n.lower()]
            if filtered:
                candidates = filtered
                
        if not candidates:
            raise ValueError(
                f"{safe_zip_path}: no annotation XML found — this doesn't "
                f"look like a real Sentinel-1 SAFE archive."
            )
            
        with zf.open(candidates[0]) as f:
            root = ET.parse(f).getroot()

        def get_text(path: str) -> str:
            elem = root.find(path)
            if elem is None or elem.text is None:
                raise ValueError(f"Required annotation field missing: {path}")
            return elem.text.strip()

        # 1. Extract basic geometry
        first_line_time = datetime.fromisoformat(
            get_text(".//imageAnnotation/imageInformation/productFirstLineUtcTime")
        )
        azimuth_interval = float(get_text(".//imageAnnotation/imageInformation/azimuthTimeInterval"))
        near_range = float(get_text(".//imageAnnotation/imageInformation/slantRangeTime"))
        range_rate = float(get_text(".//generalAnnotation/productInformation/rangeSamplingRate"))
        n_lines = int(get_text(".//imageAnnotation/imageInformation/numberOfLines"))
        n_columns = int(get_text(".//imageAnnotation/imageInformation/numberOfSamples"))

        # 2. Resolve burst timing (Single pass, explicit fallback)
        burst_note = ""
        burst_azimuth_times = []
        lines_per_burst = None
        
        try:
            # Extract lines per burst directly from the XML tree
            lpb_elem = root.find(".//swathTiming/linesPerBurst")
            if lpb_elem is not None and lpb_elem.text:
                lines_per_burst = int(lpb_elem.text.strip())
            else:
                raise ValueError("linesPerBurst missing from swathTiming")

            # _extract_burst_timing_from_root returns a list[dict], not an object
            timing_dicts = _extract_burst_timing_from_root(root) 
            
            if not timing_dicts:
                raise ValueError("burstList is empty or missing")

            # Parse azimuth times from the dictionaries into datetime objects
            for b in timing_dicts:
                az_str = b.get("azimuthTime")
                if az_str:
                    burst_azimuth_times.append(datetime.fromisoformat(az_str.strip()))
                    
            if not burst_azimuth_times:
                raise ValueError("No valid azimuthTime entries found in bursts")

            burst_note = f", {len(burst_azimuth_times)} bursts (burst-aware timing)"
            
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            # Expected fallback for Stripmap or malformed XML
            # Added AttributeError and TypeError to catch the exact failure mode you experienced
            burst_azimuth_times, lines_per_burst = _parse_swath_timing(root)
            burst_note = " (linear timing fallback)"
            logger.debug(f"Burst timing fallback triggered for {Path(safe_zip_path).name}: {e}")

        # 3. Instantiate cleanly
        geometry = SLCGeometry(
            first_line_time=first_line_time,
            azimuth_time_interval_s=azimuth_interval,
            near_range_time_s=near_range,
            range_sampling_rate_hz=range_rate,
            n_lines=n_lines,
            n_columns=n_columns,
            burst_azimuth_times=burst_azimuth_times,
            lines_per_burst=lines_per_burst,
        )

        logger.info(
            "Parsed real SLC geometry from %s: %d x %d, starting %s%s",
            Path(safe_zip_path).name, geometry.n_lines, geometry.n_columns,
            geometry.first_line_time, burst_note
        )
        return geometry


def parse_chirp_bandwidth(
    safe_zip_path: Union[str, Path], member_hint: Optional[str] = None
) -> float:
    """
    Real, per-product chirp bandwidth (Hz), read directly from a real
    Sentinel-1 SAFE archive's annotation XML.

    Raises:
        ValueError if the annotation XML or required fields are missing.
    """
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(safe_zip_path) as zf:
        candidates = [
            n for n in zf.namelist()
            if "/annotation/" in n and n.lower().endswith(".xml")
            and "/calibration/" not in n.lower()
            and not Path(n).name.lower().startswith("rfi-")
            and "/rfi/" not in n.lower()
        ]
        if member_hint:
            filtered = [n for n in candidates if member_hint.lower() in n.lower()]
            if filtered:
                candidates = filtered
        if not candidates:
            raise ValueError(
                f"{safe_zip_path}: no annotation XML found — cannot "
                f"read real chirp bandwidth."
            )
        with zf.open(candidates[0]) as f:
            root = ET.parse(f).getroot()

        def get_text(path: str) -> str:
            elem = root.find(path)
            if elem is None or elem.text is None:
                raise ValueError(
                    f"{safe_zip_path}: required chirp field missing: {path}"
                )
            return elem.text.strip()

        pulse_length_s = float(get_text(
            ".//generalAnnotation/downlinkInformationList/downlinkInformation/"
            "downlinkValues/txPulseLength"
        ))
        ramp_rate_hz_per_s = float(get_text(
            ".//generalAnnotation/downlinkInformationList/downlinkInformation/"
            "downlinkValues/txPulseRampRate"
        ))
        bandwidth_hz = abs(ramp_rate_hz_per_s * pulse_length_s)
        logger.info(
            "Parsed real chirp bandwidth from %s: %.2f MHz",
            Path(safe_zip_path).name, bandwidth_hz / 1e6,
        )
        return bandwidth_hz


@dataclass
class BurstInfo:
    """
    Real per-burst timing and valid-sample metadata for one Sentinel-1
    TOPS burst, parsed from the annotation XML's swathTiming element.
    """
    burst_index: int
    azimuth_time: datetime
    sensing_time: Optional[datetime]
    byte_offset: int
    first_valid_sample: Any
    last_valid_sample: Any


@dataclass
class SwathTiming:
    """
    Real burst structure for one Sentinel-1 TOPS sub-swath: uniform
    per-burst dimensions plus the real, individual timing/valid-sample
    metadata for every burst.
    """
    lines_per_burst: int
    samples_per_burst: int
    bursts: List[BurstInfo] = field(default_factory=list)




def parse_burst_info(
    safe_zip_path: Union[str, Path], member_hint: Optional[str] = None
) -> SwathTiming:
    """
    Parse real per-burst timing and valid-sample metadata from a
    Sentinel-1 SAFE archive's annotation XML.

    Returns a SwathTiming dataclass — NOT a raw list. All downstream
    consumers (deburst_array, estimate_esd_shift_per_burst_overlap)
    depend on the .bursts attribute, so this must always return the
    wrapper object.

    Real field paths:
     /product/swathTiming/linesPerBurst
     /product/swathTiming/samplesPerBurst
     /product/swathTiming/burstList/burst/azimuthTime
     /product/swathTiming/burstList/burst/sensingTime
     /product/swathTiming/burstList/burst/byteOffset
     /product/swathTiming/burstList/burst/firstValidSample
     /product/swathTiming/burstList/burst/lastValidSample
    """
    import numpy as np
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(safe_zip_path) as zf:
        candidates = [
            n for n in zf.namelist()
            if "/annotation/" in n and n.lower().endswith(".xml")
            and "/calibration/" not in n.lower()
            and not Path(n).name.lower().startswith("rfi-")
            and "/rfi/" not in n.lower()
        ]
        if member_hint:
            filtered = [n for n in candidates if member_hint.lower() in n.lower()]
            if filtered:
                candidates = filtered
        if not candidates:
            raise ValueError(
                f"{safe_zip_path}: no annotation XML found — this doesn't "
                f"look like a real Sentinel-1 SAFE archive."
            )
        with zf.open(candidates[0]) as f:
            root = ET.parse(f).getroot()

        def get_text(path: str) -> str:
            elem = root.find(path)
            if elem is None or elem.text is None:
                raise ValueError(
                    f"{safe_zip_path}: required swathTiming field missing: {path}"
                )
            return elem.text.strip()

        lines_per_burst = int(get_text(".//swathTiming/linesPerBurst"))
        samples_per_burst = int(get_text(".//swathTiming/samplesPerBurst"))

        burst_list_elem = root.find(".//swathTiming/burstList")
        if burst_list_elem is None:
            raise ValueError(f"{safe_zip_path}: no burstList found in swathTiming")
        burst_elems = burst_list_elem.findall("burst")

        bursts = []
        for i, burst_elem in enumerate(burst_elems):
            azimuth_time_elem = burst_elem.find("azimuthTime")
            if azimuth_time_elem is None or azimuth_time_elem.text is None:
                raise ValueError(
                    f"{safe_zip_path}: burst {i} missing required azimuthTime"
                )
            azimuth_time = datetime.fromisoformat(azimuth_time_elem.text.strip())

            sensing_time_elem = burst_elem.find("sensingTime")
            sensing_time = (
                datetime.fromisoformat(sensing_time_elem.text.strip())
                if sensing_time_elem is not None and sensing_time_elem.text
                else None
            )
            byte_offset_elem = burst_elem.find("byteOffset")
            byte_offset = (
                int(byte_offset_elem.text.strip())
                if byte_offset_elem is not None and byte_offset_elem.text
                else -1
            )
            first_valid_elem = burst_elem.find("firstValidSample")
            last_valid_elem = burst_elem.find("lastValidSample")
            if first_valid_elem is None or last_valid_elem is None:
                raise ValueError(
                    f"{safe_zip_path}: burst {i} missing firstValidSample/"
                    f"lastValidSample."
                )
            first_valid_sample = np.array(
                [int(v) for v in first_valid_elem.text.split()], dtype=np.int32
            )
            last_valid_sample = np.array(
                [int(v) for v in last_valid_elem.text.split()], dtype=np.int32
            )
            if len(first_valid_sample) != lines_per_burst:
                raise ValueError(
                    f"{safe_zip_path}: burst {i} firstValidSample has "
                    f"{len(first_valid_sample)} entries, expected "
                    f"linesPerBurst={lines_per_burst}."
                )
            bursts.append(BurstInfo(
                burst_index=i,
                azimuth_time=azimuth_time,
                sensing_time=sensing_time,
                byte_offset=byte_offset,
                first_valid_sample=first_valid_sample,
                last_valid_sample=last_valid_sample,
            ))

        logger.info(
            "Parsed real burst metadata from %s: %d bursts, %d lines/burst, "
            "%d samples/burst",
            Path(safe_zip_path).name, len(bursts), lines_per_burst, samples_per_burst,
        )
        # CRITICAL: always return the SwathTiming wrapper, never a raw list.
        return SwathTiming(
            lines_per_burst=lines_per_burst,
            samples_per_burst=samples_per_burst,
            bursts=bursts,
        )