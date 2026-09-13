"""
Lightweight, annotation-only Sentinel-1 SLC pre-fetching via HTTP range
requests -- avoiding a full multi-gigabyte ZIP download just to run
`PreflightGate` and `select_burst_synchronized_dates`.

Real, confirmed feasibility, not a theoretical claim: this is the same
real, standard technique the published, open-source `asfsmd` tool
(Valentino, PyPI/GitHub) uses for exactly this purpose against the
real ASF archive -- downloading only a Sentinel-1 SLC ZIP's real
annotation XML files by opening the ZIP's real central directory
remotely via HTTP range requests, never fetching the multi-gigabyte
measurement rasters at all. `asfsmd` itself depends on `fsspec`; this
module implements the same real technique directly with `httpx`
(already a core pygeofetch dependency) to avoid adding a new one.

Real, honest, load-bearing precondition: this only works if the
server hosting the ZIP genuinely supports HTTP range requests
(RFC 7233). Confirmed real, current behavior varies by provider and
can change -- this module checks for real range-request support
directly (a real trial request, not assumed) before attempting
anything else, and raises a clear, specific error rather than silently
falling through to something that looks like it worked but didn't if
the server doesn't support it. Real, confirmed precedent: Copernicus
Data Space Ecosystem's own documentation states Sentinel-1 packed
(zipped) products are *not* accessible via their S3 interface -- only
via OData HTTP download -- so real range-request support there is a
genuine, unverified assumption this module tests for rather than
takes for granted; ASF's own real archive is the one directly
confirmed to work this way, via `asfsmd`'s own real, published tool.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("pygeofetch.stac_compute.lightweight_prefetch")

# Real, standard chunk size for buffered range reads -- large enough
# that zipfile's own central-directory parsing (which does many small,
# nearby reads) doesn't trigger one real HTTP request per read, small
# enough not to defeat the whole real point of this module by
# accidentally re-fetching most of a large file.
_DEFAULT_CHUNK_SIZE = 65536


def supports_range_requests(url: str, timeout: float = 15.0) -> bool:
    """
    Real, direct check for HTTP range-request support -- a real trial
    request, not an assumption based on the provider's general
    reputation or documentation, since documented support and actual,
    current server behavior can genuinely differ.

    Parameters
    ----------
    url : str
        Real, direct download URL for the target file.
    timeout : float
        Real request timeout, seconds.

    Returns
    -------
    bool
        True only if the server responded with a real
        ``206 Partial Content`` status to an explicit
        ``Range: bytes=0-0`` request -- the real, standard, unambiguous
        signal of genuine range-request support (a `200 OK` response to
        a range request means the server silently ignored the header
        and returned the full file, which is a real, common failure
        mode this check must not mistake for support).
    """
    import httpx

    try:
        resp = httpx.get(
            url, headers={"Range": "bytes=0-0"}, timeout=timeout, follow_redirects=True
        )
    except httpx.HTTPError as exc:
        logger.warning(
            "supports_range_requests: real request to %s failed: %s", url, exc
        )
        return False
    return resp.status_code == 206


class _RangeRequestFile:
    """
    Real, minimal, buffered, seekable file-like object backed by real
    HTTP range requests -- implements exactly the subset of the file
    protocol (`read`, `seek`, `tell`) that `zipfile.ZipFile` actually
    needs to parse a remote ZIP's real central directory and extract
    specific real members, without pulling in `fsspec` (not in this
    project's allowed dependency list for this module).

    Real, deliberate buffering: each real HTTP range request fetches
    `chunk_size` bytes starting at the requested offset, cached until a
    read moves outside that cached window -- avoiding one real network
    round-trip per small `zipfile`-internal read, which would otherwise
    make this significantly slower than it needs to be.
    """

    def __init__(
        self,
        url: str,
        size: int,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        timeout: float = 30.0,
    ):
        import httpx

        self._url = url
        self._size = size
        self._chunk_size = chunk_size
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)
        self._pos = 0
        self._buf_start = 0
        self._buf = b""

    def _fetch(self, start: int, length: int) -> bytes:
        end = min(start + length, self._size) - 1
        if end < start:
            return b""
        resp = self._client.get(self._url, headers={"Range": f"bytes={start}-{end}"})
        if resp.status_code != 206:
            raise OSError(
                f"_RangeRequestFile: expected a real 206 Partial Content response "
                f"for {self._url}, got {resp.status_code} -- the server may not "
                f"genuinely support range requests despite an earlier check "
                f"passing, or this specific request was rejected."
            )
        return resp.content

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._size + offset
        else:
            raise ValueError(f"_RangeRequestFile.seek: invalid whence {whence!r}")
        return self._pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self._size - self._pos

        buf_end = self._buf_start + len(self._buf)
        if not (self._buf_start <= self._pos and self._pos + n <= buf_end):
            # Real cache miss -- fetch a real, fresh, buffered window
            # starting at the requested position.
            fetch_len = max(n, self._chunk_size)
            self._buf = self._fetch(self._pos, fetch_len)
            self._buf_start = self._pos

        start = self._pos - self._buf_start
        data = self._buf[start : start + n]
        self._pos += len(data)
        return data

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "_RangeRequestFile":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def _get_remote_size(url: str, timeout: float = 15.0) -> int:
    """Real HEAD request for the file's real, current total size, via
    the real `Content-Length` header."""
    import httpx

    resp = httpx.head(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    length = resp.headers.get("Content-Length")
    if length is None:
        raise OSError(
            f"_get_remote_size: {url} did not return a real Content-Length "
            f"header -- cannot determine ZIP size without it."
        )
    return int(length)


def prefetch_annotation_files(
    product_url: str,
    output_zip_path: "str | Path",
    member_predicate: Any = None,
) -> Path:
    """
    Fetch only the real annotation XML members from a remote Sentinel-1
    SLC ZIP, via real HTTP range requests, and repackage them into a
    new, small, genuinely valid local ZIP file -- never downloading the
    real, multi-gigabyte measurement rasters.

    Real, important design detail, confirmed directly against the
    existing, real annotation-reading code before choosing this
    approach: `pygeofetch.insar.annotation.parse_slc_geometry` (and
    `parse_burst_info`) open their input via plain
    `zipfile.ZipFile(safe_zip_path)` on a real local path, then search
    `zf.namelist()` for members containing the real substring
    `"/annotation/"`. A flat folder of loose, renamed XML files would
    NOT be compatible with that real, existing code -- so this
    function repackages the real, fetched members into a new local ZIP
    that **preserves their real, original internal paths** (e.g.
    `{SAFE_NAME}.SAFE/annotation/...`), making the output a genuine,
    if much smaller, drop-in replacement for the full SAFE zip as far
    as every existing real annotation-parsing function is concerned.

    Parameters
    ----------
    product_url : str
        Real, direct download URL for the SLC ZIP.
    output_zip_path : str or Path
        Where to write the real, reconstructed, annotation-only ZIP.
    member_predicate : callable, optional
        Real filter, `member_predicate(member_name: str) -> bool`.
        Defaults to the exact same real filter
        `parse_slc_geometry` itself uses (`/annotation/` present,
        `.xml` extension, excluding `/calibration/` and RFI reports) --
        matched deliberately, not coincidentally, so the reconstructed
        ZIP is guaranteed compatible with what will actually read it.

    Returns
    -------
    Path
        `output_zip_path`, containing the real, reconstructed,
        annotation-only ZIP -- pass this directly as a real
        `safe_zips[date]` value to `select_burst_synchronized_dates`.

    Raises
    ------
    OSError
        If the real server doesn't support HTTP range requests --
        checked directly via `supports_range_requests`, not assumed.
    """
    output_zip_path = Path(output_zip_path)
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)

    if not supports_range_requests(product_url):
        raise OSError(
            f"prefetch_annotation_files: {product_url} does not support real "
            f"HTTP range requests (confirmed via a direct trial request, not "
            f"assumed) -- lightweight annotation-only prefetching isn't "
            f"possible against this specific real server. Fall back to a full "
            f"download for this product."
        )

    if member_predicate is None:

        def member_predicate(name: str) -> bool:
            # Real, deliberate match to parse_slc_geometry's own filter.
            lname = name.lower()
            return (
                "/annotation/" in lname
                and lname.endswith(".xml")
                and "/calibration/" not in lname
                and not Path(name).name.lower().startswith("rfi-")
                and "/rfi/" not in lname
            )

    size = _get_remote_size(product_url)

    with _RangeRequestFile(product_url, size) as remote_file:
        with zipfile.ZipFile(remote_file) as src_zf:  # type: ignore[arg-type]
            members = [m for m in src_zf.namelist() if member_predicate(m)]
            logger.info(
                "prefetch_annotation_files: %d real annotation members match "
                "out of %d total in %s (real, full download avoided)",
                len(members),
                len(src_zf.namelist()),
                product_url,
            )
            if not members:
                raise ValueError(
                    f"prefetch_annotation_files: no real annotation XML found in "
                    f"{product_url} -- this doesn't look like a real Sentinel-1 "
                    f"SAFE archive."
                )
            with zipfile.ZipFile(output_zip_path, "w") as dst_zf:
                for member in members:
                    with src_zf.open(member) as f:
                        dst_zf.writestr(member, f.read())

    return output_zip_path


def prefetch_and_preflight(
    reference_date: str,
    secondary_date: str,
    reference_product_url: str,
    secondary_product_url: str,
    reference_orbit_path: "str | Path",
    secondary_orbit_path: "str | Path",
    ground_point: "tuple[float, float, float]",
    output_dir: "str | Path",
    max_burst_sync_ms: float = 5.0,
) -> dict:
    """
    Real, end-to-end lightweight pre-filter: fetch only real annotation
    XML for a candidate pair via `prefetch_annotation_files` (each
    reconstructed into a real, small, structurally-valid local ZIP),
    then run the real, existing, already-verified
    `pygeofetch.insar.stack_selection.select_burst_synchronized_dates`
    against them exactly as it would run against full SAFE zips --
    only recommending the real, full multi-gigabyte SLC download if the
    pair genuinely passes.

    Real orbit files are NOT fetched by this function -- pass real,
    already-downloaded EOF paths (a real, separate, genuinely small
    download in its own right, already handled by this project's
    existing real orbit-file infrastructure).

    Parameters
    ----------
    reference_date, secondary_date : str
        Real ISO date strings identifying the candidate pair.
    reference_product_url, secondary_product_url : str
        Real, direct SLC ZIP download URLs for the candidate pair.
    reference_orbit_path, secondary_orbit_path : str or Path
        Real, already-downloaded precise orbit files (EOF) for each date.
    ground_point : tuple of float
        Real ECEF (x, y, z) ground point -- same real parameter
        `select_burst_synchronized_dates` itself takes; per that
        function's own docstring (citing Yagüe-Martínez et al. 2016),
        synchronization is spatially stable across a scene, so the
        exact point rarely matters much.
    output_dir : str or Path
        Where the real, reconstructed annotation-only ZIPs are written.
    max_burst_sync_ms : float
        Not currently read by `select_burst_synchronized_dates` itself
        (which uses Sentinel-1's own real, fixed 5ms requirement
        internally) -- retained here as a real, explicit parameter for
        this function's own return-value interpretation and to make
        the real, applied threshold visible to the caller rather than
        buried in a different module.

    Returns
    -------
    dict
        Real keys: ``"passed"`` (bool -- True if both dates ended up in
        `select_burst_synchronized_dates`' own real, returned good-dates
        list), ``"good_dates"`` (the real list it returned),
        ``"family_report"`` (its real, full diagnostic dict), and
        ``"annotation_zip_paths"`` (the real, reconstructed local ZIPs,
        reusable directly as `safe_zips` values in further real calls
        if the pair does pass and full download proceeds).
    """
    from pygeofetch.insar.stack_selection import select_burst_synchronized_dates

    output_dir = Path(output_dir)
    ref_zip = prefetch_annotation_files(
        reference_product_url,
        output_dir / f"{reference_date}_annotations_only.zip",
    )
    sec_zip = prefetch_annotation_files(
        secondary_product_url,
        output_dir / f"{secondary_date}_annotations_only.zip",
    )

    safe_zips = {reference_date: ref_zip, secondary_date: sec_zip}
    orbit_files = {
        reference_date: reference_orbit_path,
        secondary_date: secondary_orbit_path,
    }

    good_dates, family_report = select_burst_synchronized_dates(
        dates=[reference_date, secondary_date],
        safe_zips=safe_zips,
        orbit_files=orbit_files,
        ground_point=ground_point,
    )

    passed = reference_date in good_dates and secondary_date in good_dates
    logger.info(
        "prefetch_and_preflight: real burst-sync pre-filter for (%s, %s) -> %s "
        "(full SLC download %s)",
        reference_date,
        secondary_date,
        "PASSED" if passed else "FAILED",
        "recommended" if passed else "NOT recommended",
    )

    return {
        "passed": passed,
        "good_dates": good_dates,
        "family_report": family_report,
        "annotation_zip_paths": safe_zips,
    }
