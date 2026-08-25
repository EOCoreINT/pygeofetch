"""
Copernicus Data Space OData "Nodes" navigation -- fetching specific
files from inside a product without downloading the whole product.

Real, documented capability of the CDSE OData API:

    Products(<UUID>)/Nodes(<name>)/Nodes                 -- list a folder's contents
    Products(<UUID>)/Nodes(<name>)/Nodes(<name>)/$value   -- fetch one file's bytes

(https://documentation.dataspace.copernicus.eu/APIs/OData.html;
 response shape for a Nodes listing -- {"result": [{"Name": ...,
 "ContentLength": ..., "ChildrenNumber": ...}, ...]} -- confirmed via
 CDSE's own sibling documentation, the CREODIAS EOData Catalogue API
 manual, which documents the same OData Nodes schema.)

Why this exists: insar.esd.compute_burst_synchronization() needs
SwathTiming parsed from a product's real annotation XML
(insar.annotation.parse_burst_info()) -- a tiny fraction (tens of KB to
a few MB) of a Sentinel-1 SLC product's total size (multiple GB,
almost entirely measurement/raster data). Fetching just the annotation
files makes it possible to run that real, already-validated check
BEFORE committing to a full download, instead of only after -- turning
insar.preflight's "burst-family risk unassessed" advisory into an
actual answer when this path works.

HONEST LIMITATION, stated plainly: this is built against CDSE's own
documented OData Nodes schema and this codebase's own confirmed real
download pattern (same auth session, same host, same product-UUID
addressing, from providers.copernicus.CopernicusProvider). The node-
name quoting convention (unquoted, Nodes(name)) IS now confirmed
directly against the real, live CDSE API -- a real Nodes listing
response includes the server's own self-generated navigation URI with
no quotes around the node name, settling what the documentation left
ambiguous. What remains unconfirmed: everything downstream of that
first real listing call (fetching actual file bytes via $value,
listing a nested annotation folder, the full fetch_annotation_zip()
pipeline end-to-end) has not yet been exercised against a real
product from the environment this was written in -- there is no
network path to download.dataspace.copernicus.eu there.
test_copernicus_nodes.py covers URL construction, response parsing,
and the resulting mini-zip structure against mocked HTTP responses
matching the now-confirmed real schema; it does not confirm every
downstream step works end-to-end live. Every function here fails
loudly and specifically (never silently) if a real response doesn't
match the expected shape, and insar.preflight's integration of this
treats any failure as "stay with the honest unassessed advisory,"
never as a false pass -- so a live-API surprise degrades this back to
the previous, safe behavior rather than producing a wrong answer.
Verify the full pipeline against a handful of real products before
relying on this unattended.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("pygeofetch.providers.copernicus_nodes")

ODATA_BASE = "https://download.dataspace.copernicus.eu/odata/v1"

# Bumped whenever the real, live-API-confirmed request behavior changes
# (not just any code edit) -- logged once below specifically so a real
# run's own log output settles, unambiguously, which behavior is
# actually active. Added after the SAME stale-file confusion happened
# repeatedly: real runs kept showing "quoted request got a 400,
# retrying unquoted" on every single call (an already-fixed ordering)
# after the fix had already been delivered -- there was no way to tell
# from the log alone that the running code was out of date, only by
# noticing the retry pattern itself.
# CHANGELOG:
#   v1: quote_segments=True was the default (WRONG -- every real call
#       hit a genuine 400 first).
#   v2: quote_segments=True tried first, quote_segments=False used as
#       a fallback on a real 400 (functionally correct but paid for a
#       wasted round-trip on every single call).
#   v3: quote_segments=False became the default and tried first, with
#       quote_segments=True kept as a rare fallback.
#   v4: the quoted attempt removed entirely from _fetch() --
#       two full, independent live runs (different dates/products,
#       ~25+ requests each) got the identical deterministic rejection
#       for every single quoted attempt, zero exceptions. No longer a
#       "safety net" worth the overhead; every call is now a single
#       request, and the "quoted...retrying" log noise is gone.
#   v5 (current): find_annotation_members() now sorts its returned
#       members alphabetically by filename before returning. Fixes a
#       real, confirmed non-determinism bug: across four separate live
#       runs on an identical 6-date stack, the computed burst-family
#       classification came back different every time (4/2, 4/2, 5/1,
#       2/4 majority/minority splits) for what should be a fixed,
#       reproducible real physical quantity -- traced to which
#       sub-swath (iw1/iw2/iw3) happened to land first in the mini-zip,
#       which depended on live server response order.
MODULE_VERSION = "v5-sorted-annotation-members"  # bumped whenever request/parsing behavior changes; see the comment block above
logger.info(
    "copernicus_nodes module loaded (%s) -- node-name quoting: unquoted "
    "only; annotation members sorted deterministically before use. If a "
    'real run\'s log shows "quoted... got a real 400... retrying" for '
    "ANY call, OR shows annotation members fetched in a non-alphabetical "
    "order (should always be iw1 before iw2 before iw3), the running "
    "file is NOT this version -- re-copy providers/copernicus_nodes.py "
    "before trusting anything else in that run.",
    MODULE_VERSION,
)


def _unwrap_token(token: Any) -> str:
    """
    AuthSession.access_token is documented as `Any | None` -- "str or
    SecretStr -- use .get_secret_value() if SecretStr". Handle both
    without assuming which one a given session actually holds.
    """
    if token is None:
        raise ValueError("_unwrap_token: access token is None -- not authenticated?")
    get_secret = getattr(token, "get_secret_value", None)
    if callable(get_secret):
        return get_secret()
    return str(token)


def get_bearer_token(client: Any, provider: str = "copernicus") -> str:
    """
    Real, already-authenticated bearer token for `provider`, via the
    same client.auth.get_session() this codebase's own
    providers.copernicus.CopernicusProvider relies on internally.

    Raises:
        ValueError: if there is no valid (unexpired) session for this
                    provider -- i.e. client.add_credentials(provider,
                    ...) hasn't been called, or the session expired and
                    wasn't refreshed.
    """
    session = client.auth.get_session(provider)
    if session is None:
        raise ValueError(
            f"get_bearer_token: no valid session for provider {provider!r} -- "
            f"call client.add_credentials({provider!r}, ...) first."
        )
    return _unwrap_token(session.access_token)


def _nodes_url(
    product_id: str,
    path_segments: List[str],
    list_children: bool,
    quote_segments: bool = False,
) -> str:
    """
    Build a real Nodes navigation URL.

    Args:
        product_id:     Real product UUID (SatelliteData.id).
        path_segments:  Real folder/file names to navigate through, in
                        order, e.g. ["S1A_..._8F57.SAFE", "annotation"].
        list_children:  True for a folder LISTING request (appends
                        "/Nodes"); False for a file CONTENT request
                        (appends "/$value").
        quote_segments: False (default; CONFIRMED correct) leaves node
                        names unquoted (Nodes(name)) -- confirmed
                        directly against the real, live CDSE API: a
                        real Nodes listing response includes the
                        server's own self-generated "uri" for
                        navigating further into that node, e.g.
                        "Nodes(S1B_IW_SLC__..._5E64.SAFE)/Nodes" with no
                        quotes at all. That's the server telling us the
                        real convention directly, not an inference from
                        documentation -- CDSE's Nodes endpoint does NOT
                        follow the standard OData string-literal quoting
                        convention (Nodes('name')) despite that being
                        the documented default for Edm.String keys
                        elsewhere in OData. _fetch() no longer tries the
                        quoted form at all (removed after two full live
                        runs got the identical deterministic rejection,
                        every time, for zero benefit) -- True remains
                        available here only for tests/direct callers
                        that specifically want to construct or compare
                        against the standard OData form.
    """
    url = f"{ODATA_BASE}/Products({product_id})"
    for seg in path_segments:
        if quote_segments:
            # OData string-literal escaping: a literal single quote
            # inside the value is doubled (''), not backslash-escaped.
            escaped = seg.replace("'", "''")
            url += f"/Nodes('{escaped}')"
        else:
            url += f"/Nodes({seg})"
    url += "/Nodes" if list_children else "/$value"
    return url


def _raise_with_body(resp: Any) -> None:
    """
    Like resp.raise_for_status(), but includes the real response BODY
    in the raised error. Confirmed directly this matters: httpx's own
    error message for a 400 only ever says "400 Bad Request for url
    ..." -- OData services put the actual, specific reason in the
    response body, which plain raise_for_status() silently discards.
    """
    if resp.is_success:
        return
    import httpx

    try:
        body = resp.text[:2000]
    except Exception:
        body = "<could not read response body>"
    raise httpx.HTTPStatusError(
        f"{resp.status_code} {resp.reason_phrase} for {resp.request.url} -- "
        f"real response body: {body!r}",
        request=resp.request,
        response=resp,
    )


def _fetch(
    client: Any,
    product_id: str,
    path_segments: List[str],
    list_children: bool,
    provider: str,
    timeout_seconds: float,
    accept_json: bool,
) -> Any:
    """
    Real GET against a Nodes URL, using the unquoted node-name form
    (Nodes(name)) -- CONFIRMED correct directly against the real, live
    CDSE API (see _nodes_url's own docstring: the server's own self-
    generated navigation URI in a real response uses no quotes).

    SIMPLIFIED, no fallback dance: earlier versions tried the quoted
    form first, then unquoted; a later version reversed that order but
    kept a quoted retry as a "safety net." Removed entirely after two
    full, independent live runs (different dates, different products,
    ~25+ requests each) got the exact same deterministic rejection --
    {"code":"DAT-ZIP-104","message":"Malformed OData request path"} --
    for every single quoted attempt, zero exceptions. That's not a
    flaky edge case worth hedging against; it's a hard, consistent
    rejection, and Sentinel-1 product/file names are standardized
    enough (alphanumerics, hyphens, underscores, periods only,
    confirmed across every real name seen so far) that there's no
    real remaining case for the quoted form to rescue. Keeping the
    "try the wrong thing first" dance was pure overhead: an extra
    network round-trip and two log lines per call, for zero benefit.
    """
    import httpx

    token = get_bearer_token(client, provider)
    headers = {"Authorization": f"Bearer {token}"}
    if accept_json:
        headers["Accept"] = "application/json"

    url = _nodes_url(product_id, path_segments, list_children, quote_segments=False)
    resp = httpx.get(url, headers=headers, timeout=timeout_seconds)
    _raise_with_body(resp)
    return resp


def list_nodes(
    client: Any,
    product_id: str,
    path_segments: List[str],
    provider: str = "copernicus",
    timeout_seconds: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Real listing of a product-internal folder's contents.

    Args:
        client:         Real PyGeoFetch client, already authenticated
                        for `provider`.
        product_id:     Real product UUID.
        path_segments:  Real folder path to list, e.g.
                        ["S1A_..._8F57.SAFE"] for the SAFE folder's own
                        top-level contents, or
                        ["S1A_..._8F57.SAFE", "annotation"] for the
                        annotation subfolder.
        provider:       Real provider id for the auth session.
        timeout_seconds: Real request timeout.

    Returns:
        Real list of {"Name": str, "ContentLength": int,
        "ChildrenNumber": int, ...} entries -- ChildrenNumber > 0 (or
        ContentLength == 0 with children) marks a folder; a file has
        ChildrenNumber == 0 and a real ContentLength.

    Raises:
        ValueError: if the real response doesn't contain a real,
                    recognized listing key ("result" is CDSE's
                    documented key for this endpoint; "value" is
                    accepted defensively in case of API version
                    variance, but a response with NEITHER key is
                    treated as a real, reportable failure -- never
                    silently returns an empty list, which could be
                    mistaken for "this folder is genuinely empty."
        httpx.HTTPStatusError: on a real HTTP failure, with the real
                    response body included (see _raise_with_body()).
    """
    resp = _fetch(
        client,
        product_id,
        path_segments,
        list_children=True,
        provider=provider,
        timeout_seconds=timeout_seconds,
        accept_json=True,
    )
    data = resp.json()

    for key in ("result", "value"):
        if key in data:
            return data[key]

    raise ValueError(
        f"list_nodes: real response from {resp.request.url} has neither a "
        f"'result' nor 'value' key (got keys: {list(data.keys())}) -- "
        f"CDSE's Nodes response schema may have changed; this is a real, "
        f"reportable failure, not an empty folder."
    )


def fetch_node_bytes(
    client: Any,
    product_id: str,
    path_segments: List[str],
    provider: str = "copernicus",
    timeout_seconds: float = 30.0,
) -> bytes:
    """
    Real bytes of a single product-internal file.

    Args:
        client, product_id, provider, timeout_seconds: see list_nodes().
        path_segments: Real full path to the file, e.g.
                       ["S1A_..._8F57.SAFE", "annotation", "s1a-iw1-slc-vv-....xml"].

    Returns:
        Real raw file bytes.
    """
    resp = _fetch(
        client,
        product_id,
        path_segments,
        list_children=False,
        provider=provider,
        timeout_seconds=timeout_seconds,
        accept_json=False,
    )
    return resp.content


def find_annotation_members(
    client: Any,
    satellite_data: Any,
    polarisation: Optional[str] = "vv",
    provider: str = "copernicus",
) -> List[Tuple[List[str], str]]:
    """
    Real discovery of a Sentinel-1 product's annotation XML files,
    without downloading the product -- lists the real annotation
    folder's contents via the Nodes API and filters them the same way
    insar.annotation.parse_slc_geometry()/parse_burst_info() do when
    scanning a real, fully-downloaded SAFE zip (excluding calibration
    and RFI XML files), so results are directly comparable.

    REAL BUG FOUND AND FIXED: this used to return members in whatever
    order the live Nodes listing happened to respond in -- and that
    order isn't guaranteed stable across separate requests. Confirmed
    directly this mattered: fetch_annotation_zip() writes members into
    the mini-zip in the order given here, zipfile.ZipFile preserves
    write order (not alphabetical) in namelist(), and
    annotation.parse_burst_info()/parse_slc_geometry() -- called with
    no explicit member_hint from screen_stack_burst_synchronization()
    -- just take whichever entry is first (candidates[0]). Across three
    real, live runs on the identical 6-date stack (confirmed identical
    real annotation file sizes each time -- the same real bytes were
    fetched every time), the computed Δt_acq values for the same real
    date pairs varied by up to ~950ms between runs. Root cause: which
    sub-swath (iw1/iw2/iw3) ended up first in the zip -- and therefore
    used as "the" burst-timing reference for that date -- depended on
    server response order, and because Δt_acq involves a modulo
    reduction to within half a burst cycle, even a real but modest
    sub-swath-to-sub-swath timing difference can alias into a wildly
    different final value depending on which sub-swath was picked.

    Fixed by sorting members alphabetically by filename before
    returning -- Sentinel-1's real annotation naming convention
    (s1a-iw1-..., s1a-iw2-..., s1a-iw3-...) sorts iw1 first, every
    time, regardless of server response order, so the same sub-swath
    is used as the reference consistently across runs.

    Does NOT need a root Nodes listing call: the real SAFE folder name
    is already known from search metadata
    (satellite_data.properties["name"]), saving one real HTTP call.

    Args:
        client:         Real, authenticated PyGeoFetch client.
        satellite_data: Real SatelliteData with a real .id (product
                        UUID) and .properties["name"] (SAFE folder
                        name).
        polarisation:   Real polarisation to filter to (e.g. "vv"),
                        matching Sentinel-1's real annotation filename
                        convention (s1a-iw1-slc-vv-...). None returns
                        every real annotation XML regardless of
                        polarisation.
        provider:       Real provider id.

    Returns:
        Real list of (path_segments, filename) pairs, sorted
        deterministically by filename, each directly usable with
        fetch_node_bytes().

    Raises:
        ValueError: if satellite_data has no real "name" property, or
                    if no real annotation folder / no matching real XML
                    files are found.
    """
    safe_name = (
        satellite_data.properties.get("name")
        if hasattr(satellite_data, "properties")
        else None
    )
    if not safe_name:
        raise ValueError(
            "find_annotation_members: satellite_data.properties['name'] is "
            "missing -- can't address the product's internal SAFE folder "
            "without it."
        )

    product_id = satellite_data.id
    top_level = list_nodes(client, product_id, [safe_name], provider=provider)
    annotation_entry = next(
        (n for n in top_level if str(n.get("Name", "")).lower() == "annotation"),
        None,
    )
    if annotation_entry is None:
        raise ValueError(
            f"find_annotation_members: no real 'annotation' folder found "
            f"under {safe_name} -- real top-level entries were: "
            f"{[n.get('Name') for n in top_level]}"
        )

    annotation_files = list_nodes(
        client, product_id, [safe_name, "annotation"], provider=provider
    )

    members: List[Tuple[List[str], str]] = []
    for entry in annotation_files:
        name = str(entry.get("Name", ""))
        name_lower = name.lower()
        if not name_lower.endswith(".xml"):
            continue
        if (
            "calibration" in name_lower
            or name_lower.startswith("rfi-")
            or "/rfi/" in name_lower
        ):
            continue
        if polarisation is not None and f"-{polarisation.lower()}-" not in name_lower:
            continue
        members.append(([safe_name, "annotation"], name))

    if not members:
        raise ValueError(
            f"find_annotation_members: real annotation folder under "
            f"{safe_name} contained no matching XML files (polarisation="
            f"{polarisation!r}) -- real entries were: "
            f"{[n.get('Name') for n in annotation_files]}"
        )

    # Deterministic order regardless of live server response order --
    # see the docstring's real-bug note above for why this matters.
    members.sort(key=lambda m: m[1].lower())

    return members


def fetch_annotation_zip(
    client: Any,
    satellite_data: Any,
    output_dir: Any,
    polarisation: Optional[str] = "vv",
    provider: str = "copernicus",
) -> Path:
    """
    Real, minimal zip file containing just a product's real annotation
    XML files, at the same relative paths a full real SAFE zip would
    have (`<safe_name>/annotation/<file>.xml`) -- so
    insar.annotation.parse_slc_geometry() / parse_burst_info() work on
    it completely unchanged, without needing the real, full,
    multi-gigabyte product downloaded at all.

    Args:
        client, satellite_data, polarisation, provider: see
                        find_annotation_members().
        output_dir:     Real directory to write the mini-zip into.

    Returns:
        Real Path to the written zip file (named
        `<safe_name>_annotation_only.zip`).
    """
    safe_name = satellite_data.properties["name"]
    members = find_annotation_members(
        client, satellite_data, polarisation=polarisation, provider=provider
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # BUG CAUGHT BEFORE SHIPPING: safe_name.rstrip(".SAFE") strips a set
    # of characters ('.', 'S', 'A', 'F', 'E'), not the literal suffix --
    # it happened to give the right answer for one real example name
    # only because the character before ".SAFE" wasn't itself in that
    # set. removesuffix() is the actually-correct operation here.
    base_name = safe_name.removesuffix(".SAFE")
    zip_path = output_dir / f"{base_name}_annotation_only.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path_segments, filename in members:
            content = fetch_node_bytes(
                client, satellite_data.id, path_segments + [filename], provider=provider
            )
            arcname = "/".join(path_segments + [filename])
            zf.writestr(arcname, content)
            logger.info(
                "Fetched real annotation member %s (%d bytes)", arcname, len(content)
            )

    return zip_path
