"""
Sentinel-1 precise orbit file management.

Orbit files are provided by ESA at:
  https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB/  (precise, 21-day delay)
  https://step.esa.int/auxdata/orbits/Sentinel-1/RESORB/  (restituted, ~3-hour delay)

File naming convention:
  S1A_OPER_AUX_POEORB_OPOD_<processing_time>_V<validity_start>_<validity_stop>.EOF

Typical usage::

    from pygeofetch.core.orbits import fetch_orbit_file

    path = fetch_orbit_file(
        product_name="S1C_IW_SLC__1SDV_20260601T053000_20260601T053027_...",
        output_dir="./orbits/",
        orbit_type="precise",
    )
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import requests

logger = logging.getLogger("pygeofetch.orbits")

POEORB_BASE_URL = "https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB"
RESORB_BASE_URL = "https://step.esa.int/auxdata/orbits/Sentinel-1/RESORB"

# Mapping from short satellite name to ESA orbit directory prefix
_SAT_DIR = {
    "S1A": "S1A",
    "S1B": "S1B",
    "S1C": "S1C",
    "S1D": "S1D",
}


def fetch_orbit_file(
    product_name: str,
    output_dir: str = "./orbits/",
    orbit_type: str = "precise",
    timeout: int = 60,
) -> str | None:
    """
    Download the orbit file for a Sentinel-1 SLC product.

    Args:
        product_name: Full Sentinel-1 product name or scene ID.
                      Must contain the acquisition datetime string
                      e.g. "S1C_IW_SLC__1SDV_20260601T053000_..."
        output_dir:   Directory to save the orbit file.
        orbit_type:   "precise" (POEORB, 21-day delay, recommended for InSAR)
                      "restituted" (RESORB, ~3-hour delay, for near-real-time)
        timeout:      HTTP request timeout in seconds.

    Returns:
        Absolute path to downloaded orbit file, or None if not found/unavailable.

    Example::

        path = fetch_orbit_file(
            product_name="S1C_IW_SLC__1SDV_20260601T053000...",
            output_dir="./orbits/",
            orbit_type="precise",
        )
        if path:
            print(f"Orbit file ready: {path}")
        else:
            print("Precise orbit not yet published — try orbit_type='restituted'")
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    acq_dt = _parse_acquisition_datetime(product_name)
    if acq_dt is None:
        logger.error("Cannot parse acquisition datetime from: %s", product_name)
        return None

    sat = _parse_satellite(product_name)
    logger.debug(
        "Orbit lookup: satellite=%s, acq_dt=%s, type=%s", sat, acq_dt, orbit_type
    )

    # Check precise orbit availability window (published 21 days after acquisition)
    if orbit_type == "precise":
        days_since = (datetime.utcnow() - acq_dt).days
        if days_since < 21:
            logger.warning(
                "Precise orbit (POEORB) for %s typically published 21 days after "
                "acquisition. Only %d day(s) have passed — orbit may not be available yet. "
                "Consider orbit_type='restituted' for near-real-time processing.",
                sat,
                days_since,
            )

    # Check cache first
    cached = _find_cached_orbit(out_dir, sat, acq_dt, orbit_type)
    if cached:
        logger.info("Using cached orbit file: %s", cached.name)
        return str(cached)

    base_url = POEORB_BASE_URL if orbit_type == "precise" else RESORB_BASE_URL
    year_month = acq_dt.strftime("%Y/%m")
    listing_url = f"{base_url}/{_SAT_DIR.get(sat, sat)}/{year_month}/"

    logger.info("Searching orbit files at: %s", listing_url)

    try:
        resp = requests.get(listing_url, timeout=timeout)
        resp.raise_for_status()
        orbit_filename = _find_matching_orbit_file(resp.text, acq_dt)
        if not orbit_filename:
            logger.warning(
                "No %s orbit file found for %s on %s. "
                "If acquisition is < 21 days ago, precise orbit may not be published yet. "
                "Try orbit_type='restituted' instead.",
                orbit_type,
                sat,
                acq_dt.date(),
            )
            return None

        # orbit_filename is the .EOF name (used for validity-window
        # matching); the server actually serves it zipped as .EOF.zip
        download_url = f"{listing_url}{orbit_filename}.zip"
        output_path = out_dir / orbit_filename

        logger.info("Downloading orbit file: %s", orbit_filename)
        _download_file(download_url, output_path, timeout=timeout)
        logger.info(
            "Orbit file saved: %s (%.1f KB)",
            output_path,
            output_path.stat().st_size / 1024,
        )
        return str(output_path)

    except requests.exceptions.ConnectionError as exc:
        logger.error("Cannot reach orbit server %s: %s", listing_url, exc)
        return None
    except requests.exceptions.HTTPError as exc:
        logger.error("HTTP error fetching orbit listing %s: %s", listing_url, exc)
        return None
    except requests.exceptions.Timeout:
        logger.error(
            "Timeout fetching orbit listing after %ds: %s", timeout, listing_url
        )
        return None
    except requests.RequestException as exc:
        logger.error("Orbit file download failed: %s", exc)
        return None
    except OSError as exc:
        logger.error("Cannot write orbit file to %s: %s", out_dir, exc)
        return None


def _parse_acquisition_datetime(product_name: str) -> datetime | None:
    """Extract acquisition start time from Sentinel-1 product name.

    Handles both:
      - "S1C_IW_GRDH_1SDV_20260601T053000_..." (full product name)
      - Any string containing a datetime token like "20260601T053027"
    """
    pattern = r"(\d{8}T\d{6})"
    matches = re.findall(pattern, product_name)
    if matches:
        try:
            return datetime.strptime(matches[0], "%Y%m%dT%H%M%S")
        except ValueError:
            pass
    return None


def _parse_satellite(product_name: str) -> str:
    """Extract satellite short name from product name or ID.

    Returns one of: S1A, S1B, S1C, S1D.
    Defaults to S1C (current active constellation) if not identifiable.
    """
    name = product_name.upper()
    # Check explicit patterns first (most specific to least)
    for sat in ("S1D", "S1C", "S1B", "S1A"):
        if name.startswith(sat + "_"):
            return sat
    for sat in ("S1D", "S1C", "S1B", "S1A"):
        if sat in name:
            return sat
    for full, short in [
        ("SENTINEL-1D", "S1D"),
        ("SENTINEL-1C", "S1C"),
        ("SENTINEL-1B", "S1B"),
        ("SENTINEL-1A", "S1A"),
    ]:
        if full in name:
            return short
    logger.debug("Cannot determine satellite from %r — defaulting to S1C", product_name)
    return "S1C"


def _find_cached_orbit(
    directory: Path,
    satellite: str,
    acq_dt: datetime,
    orbit_type: str,
) -> Path | None:
    """Return a cached orbit file if one covering acq_dt exists."""
    ext = "POEORB" if orbit_type == "precise" else "RESORB"
    prefix = f"{satellite}_OPER_AUX_{ext}"
    for f in directory.glob(f"{prefix}*.EOF"):
        if _orbit_covers_datetime(f.name, acq_dt):
            return f
    return None


def _orbit_covers_datetime(filename: str, acq_dt: datetime) -> bool:
    """Check if an orbit filename's validity window covers acq_dt."""
    # S1C_OPER_AUX_POEORB_OPOD_..._V20260601T000000_20260603T000000.EOF
    m = re.search(r"_V(\d{8}T\d{6})_(\d{8}T\d{6})\.EOF$", filename)
    if not m:
        return False
    fmt = "%Y%m%dT%H%M%S"
    try:
        start = datetime.strptime(m.group(1), fmt)
        stop = datetime.strptime(m.group(2), fmt)
        return start <= acq_dt <= stop
    except ValueError:
        return False


def _find_matching_orbit_file(listing_html: str, acq_dt: datetime) -> str | None:
    """
    Parse an HTML directory listing and find the orbit file covering acq_dt.

    ESA's step.esa.int listing serves orbit files as ZIPPED .EOF.zip
    archives, not raw .EOF files — confirmed against the live directory
    listing (e.g. S1A_OPER_AUX_POEORB_OPOD_..._V<start>_<end>.EOF.zip).
    An earlier version of this regex required the match to end exactly at
    ".EOF" before the closing quote, which never matched any real file
    (all of them have ".zip" appended after ".EOF") — this caused every
    orbit file lookup to silently report "not found" regardless of
    whether a matching file actually existed on the server.
    """
    pattern = r'href="(S1[ABCD]_OPER_AUX_(?:POEORB|RESORB)_[^"]+\.EOF)\.zip"'
    filenames = re.findall(pattern, listing_html)
    for fname in filenames:
        # fname here is the .EOF name (without .zip) for validity-window
        # matching, matching what _orbit_covers_datetime expects
        if _orbit_covers_datetime(fname, acq_dt):
            return fname
    return None


def _download_file(url: str, output_path: Path, timeout: int = 60) -> None:
    """
    Stream-download a file to disk with progress logging.

    If the downloaded content is a zip archive (as ESA's orbit files are —
    served as .EOF.zip), it is extracted automatically and output_path is
    overwritten with the extracted .EOF file's content, so callers always
    get a usable, directly-readable orbit file on disk rather than a zip
    they'd need to extract themselves.
    """
    import zipfile

    zip_path = output_path.with_suffix(output_path.suffix + ".zip")

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total_bytes = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                f.write(chunk)
                total_bytes += len(chunk)
        logger.debug("Downloaded %d bytes to %s", total_bytes, zip_path)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if m.endswith(".EOF")]
            if not members:
                # Not actually a zip, or doesn't contain an .EOF — treat
                # the downloaded content as the orbit file directly
                zip_path.rename(output_path)
                return
            with zf.open(members[0]) as src, open(output_path, "wb") as dst:
                dst.write(src.read())
        zip_path.unlink(missing_ok=True)
    except zipfile.BadZipFile:
        # Server returned a non-zip response (e.g. plain .EOF, or an
        # error page) — use it as-is rather than failing outright
        zip_path.rename(output_path)
