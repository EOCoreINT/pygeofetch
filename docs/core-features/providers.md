# Providers

pygeofetch ships **22** provider integrations. **"Listed" and "verified
working" are not the same thing** — treat the status column below as
the honest signal, not the presence of a provider module.

This page was rewritten from scratch after a full, provider-by-provider
audit: every provider below was checked against that provider's own
real, current API documentation (not assumed correct from an earlier
pass), and every real bug found along the way is named explicitly in
that provider's notes — not smoothed over.

## Status legend

- 🟢 **Verified** — checked against real, current API documentation, with real request/response shapes confirmed and (where practical) tested against a live account
- 🟡 **Open, no-auth** — search works without credentials; confirmed directly, not assumed
- 🔴 **Dead / decommissioned** — the upstream service is confirmed shut down; pygeofetch fails clearly rather than silently
- 🔐 **Auth required for download** — search may still be public; see notes

:::{admonition} Two providers were removed, not fixed
:class: warning

`eodag_provider` and `terrabotics` are **no longer part of pygeofetch**
as of this pass.

- **`eodag_provider`** only ever delegated to the third-party `eodag`
  package — itself a separate, general-purpose multi-provider
  aggregator, not a native connection to any single satellite data
  source. It was also never wired into the actual provider registry
  (`pygeofetch.providers.list_providers()`), so it was already
  unreachable through normal use before removal.
- **`terrabotics`** targeted a genuinely real company, but one with no
  public, verifiable API documentation to check its real endpoints
  against — their real engagement model is a bespoke, per-contract
  consulting service (a "fill out a form for a quote" site, not a
  self-serve developer portal), unlike every other provider on this
  page. The domain the old code used (`terrabotics.earth`) wasn't
  even the company's real one (`terrabotics.co.uk`). Rather than guess
  at endpoints with nothing public to verify them against, it was
  removed.

If you were using either, see [Contributing](../contributing.md) for
how to re-add a provider with real, documented API support.
:::
## Open, no-auth providers

Search works without any credentials for all of these — confirmed
directly against each provider's real behavior, not assumed from an
absence of an auth error.

| Provider ID | Status | Notes |
|---|---|---|
| `planetary_computer` | 🟢 Verified | Microsoft Planetary Computer STAC catalog. Sentinel-1/2, Landsat 8/9, MODIS, NAIP, ALOS DEM. SAS tokens auto-generated per asset. |
| `aws_earth` | 🟢 Verified | AWS Open Data Registry. Sentinel-2 COGs, Landsat Collection 2, NAIP. Direct S3 access, real STAC shapes confirmed against Element84/AWS Earth's actual responses. |
| `element84` | 🟢 Verified | Earth Search v1. Sentinel-2 L2A, Landsat Collection 2, Sentinel-1 RTC, Copernicus DEM. All COG. |
| `noaa_big_data` | 🟢 Verified | GOES-16/17/18 imagery. Real, public, unsigned S3 buckets (`noaa-goes16/17/18`) — listed and downloaded directly, not via a search API. Full rewrite from a fictional REST endpoint. |
| `alaska_satellite_facility` | 🟢 Verified | ASF DAAC. Sentinel-1 SLC/GRD, ALOS PALSAR, ERS, JERS. **Search is genuinely public** — confirmed directly against ASF's own documented example queries, which carry no authentication at all. Real endpoint is `api.daac.asf.alaska.edu/services/search/param`, using a WKT `intersectsWith` polygon (not a plain bbox) and real `platform`/`processingLevel` parameters — not the fictional REST search the previous version assumed. No `cloudCoverMax` parameter exists or is sent: SAR imagery isn't affected by cloud cover, and the old code's attempt to filter by it never made physical sense. Only *downloading* requires a free NASA Earthdata Login account (username/password, or a real EDL bearer token). |
| `digitalglobe` | 🟡 Open | Maxar Open Data Program disaster-response imagery. This is a **static, hierarchical STAC catalog on S3** (`maxar-opendata.s3.amazonaws.com`), not a queryable REST API — there's no server-side search. Search here does a real, honest two-stage filter: a cheap event-level pre-check using each event's own real spatial/temporal extent, then per-item filtering only for events that survive that check. A configurable `event_scan_limit` (default 60) caps how many events get scanned per call, since there's no index to search directly. |
| `jaxa_earth` | 🟡 Open | ALOS World 3D (AW3D30), 30m global DSM. This is a **static, deterministic 1°×1° tile grid**, not a temporal scene archive — "search" is real coordinate math (which tiles overlap your bbox), not a network call at all. No date or cloud-cover filter exists because none applies to a static elevation mosaic. Served via the free, public OpenTopography mirror (`opentopography.s3.sdsc.edu`), not JAXA's own distribution system (which requires manual account registration). |
| `isro_bhuvan` | 🔐 Free account | ISRO's real, current **Bhoonidhi** platform (`bhoonidhi-api.nrsc.gov.in`) — genuinely STAC-compliant, confirmed against ISRO's own official API specification. This replaced a previous version pointed at a real ISRO URL that serves an entirely different service (Bhuvan's thematic/GIS API — district codes, boundaries — not satellite imagery at all). Real collections: ResourceSat-2/2A, EOS-04, EOS-06, a Sentinel-1A mirror, CartoSat-1, and NISAR — not Cartosat-2/Oceansat-2, which aren't real Bhoonidhi collections. Real documented rate limits: 20 auth requests/hour/IP, 3 search requests/second/IP, 3 concurrent downloads/user/IP. |
| `inpe_cbers` | 🟡 Open | Brazil's INPE, via the real, current **Brazil Data Cube (BDC) STAC API** (`data.inpe.br/bdc/stac/v1`) — confirmed live and actively serving current data. CBERS-4, CBERS-4A, and Amazonia-1. The previous version pointed at a web portal, not an API. |
| `geoserver_generic` | 🟡 Open by default | Generic client for **any** self-hosted GeoServer instance's real OGC WFS `GetFeature` service — confirmed against GeoServer's own official documentation. This isn't one vendor's API: there's no "search all layers," you must configure both `endpoint` (the real GeoServer base URL) and `layer` (a real `typeName` that server advertises). GeoServer's standard security model is HTTP Basic Auth (not a bearer token), though an optional bearer token is supported for instances behind an OAuth2 proxy. |

## Authenticated providers

| Provider ID | Auth | Status | Notes |
|---|---|---|---|
| `usgs` | 🔐 Username/password (M2M) | 🟢 Verified | Landsat 1–9, MODIS, EO-1, and 500+ other datasets via the real USGS Machine-to-Machine (M2M) API. Real, current, time-sensitive detail: USGS deprecated password-only login for a required **M2M Application Token** on February 26, 2025 — confirmed against multiple real downstream projects that broke on that exact date. |
| `earth_explorer_additional` | 🔐 Same as `usgs` | 🟢 Verified | Declassified and historical imagery (Corona, KH-7 GAMBIT, KH-9 Hexagon) and early Landsat. This is a **thin subclass of `usgs`**, reusing its real, already-verified M2M auth/search/download methods unchanged — not a second API client. Only the default dataset codes differ. Real, confirmed dataset code: `declassii` (KH-7/KH-9 mapping camera). `declassi` and `declassiii` follow the same real USGS naming convention but weren't independently confirmed the same direct way. |
| `copernicus` | 🔐 OAuth2 | 🟢 Verified | Copernicus Data Space Ecosystem. Sentinel-1/2/3/5P. Full STAC, real burst-family and orbit-file infrastructure (see [InSAR Processing](../processing/insar.md)). |
| `nasa_earthdata` | 🔐 Earthdata Login | 🟢 Verified | NASA CMR (Common Metadata Repository). MODIS, VIIRS, ICESat-2, GEDI, ASTER. **Search is genuinely public** — confirmed consistently across NASA's own LP DAAC tutorial, the official `cmrfetch` CLI, and the `earthaccess` library, none of which attach credentials to a search request. A real, confirmed bug meant search used to crash with a `KeyError` before any authentication happened; fixed so search works with or without prior `authenticate()`. Download does require a real Earthdata Login account. |
| `nasa_earthdata_cloud` | 🔐 Earthdata Login + per-DAAC S3 | 🟢 Verified | Cloud-hosted NASA data on AWS. Real, confirmed finding: there is **no single, unified S3 credentials endpoint** — each DAAC (PO.DAAC, GES DISC, LP DAAC, ORNL DAAC, GHRC DAAC, NSIDC) has its own, and credentials from one don't work for another's buckets. Credentials are now fetched lazily, per the specific granule's real DAAC, and cached with real expiration awareness — not eagerly from one generic (and previously fictional) URL. Search is public, same fix and same reasoning as `nasa_earthdata` above. |
| `sentinel_hub` | 🔐 OAuth2 | 🟢 Verified | Real Catalog API (STAC-compliant). A serious, confirmed bug is fixed here: the previous request body used an entirely invented nested structure that doesn't match the real schema at all — every search would have been rejected. The real body is flat (`bbox`, a single `datetime` interval string, `collections` as plain STAC collection IDs like `sentinel-2-l2a`, not the old Process-API-style codes like `S2L2A`). Cloud-cover filtering now uses the real CQL2-JSON filter format. |
| `planet` | 🔐 API key | 🟢 Verified | PlanetScope, SkySat, RapidEye. HTTP Basic Auth with the API key as username, matching Planet's own documented convention exactly. Real bug fixed: the asset type requested on download was hardcoded to a PSScene-specific value (`ortho_analytic_4b_sr`) regardless of item type — SkySat has no such asset at all, confirmed against Planet's own SkySat documentation. Now uses a real per-item-type default, with a config override available. |
| `maxar_gbdx` | 🔐 Bearer token | 🟢 Verified | **GBDX itself was shut down on January 4, 2022** — confirmed directly from Maxar's own archived GBDXtools SDK README. This now targets the real, current, live replacement: Maxar's **Discovery API** (`api.maxar.com/discovery/v1`). Maxar has also corporately rebranded to **Vantor**, though the live API domain is unchanged at the time of writing. `PROVIDER_ID` stays `maxar_gbdx` for backward compatibility. |
| `opentopography` | 🔐 API key | 🟢 Verified | Global raster DEMs (SRTM, Copernicus DEM, ALOS World 3D, NASADEM) via the real `/API/globaldem` endpoint, plus two capabilities added this pass: USGS 3DEP raster access (1m/10m/30m, via the real `/API/usgsdem` endpoint — 1m is real, documented, academic-only) and real LiDAR point cloud **dataset discovery** via `/API/otCatalog`. Point cloud *extraction* is honestly not automated — OpenTopography's own documentation confirms there's no single-request bbox-clip API for point clouds, unlike rasters; extracting a subset needs that dataset's real tile-index shapefile, a real follow-up step this provider surfaces but doesn't yet perform itself. |

## CLI

```bash
# List all providers
pygeofetch providers list

# Filter by auth, capability, or satellite
pygeofetch providers list --no-auth
pygeofetch providers list --capabilities sar
pygeofetch providers list --satellite Landsat

# Detailed info for one provider
pygeofetch providers info planetary_computer

# Fuzzy search across provider names/descriptions
pygeofetch providers search "landsat"
```

## In Python

```python
from pygeofetch.providers import list_providers, list_provider_info

list_providers()        # -> sorted list of provider ID strings, currently 22
list_provider_info()     # -> rich metadata: auth type, capabilities, satellites, etc.
```

:::{note}

`PyGeoFetch` itself has no `.providers()` method — provider listing is
a module-level function on `pygeofetch.providers`, not a method on the
client class.
:::
## A note on how this list gets kept honest

Every fix above was made by first researching that provider's real,
current, official documentation — not by pattern-matching from other
providers or assuming an existing implementation was close enough.
Several genuinely serious bugs were only found this way: a request body
schema invented from scratch rather than the real one (`sentinel_hub`),
a completely wrong domain for a real company (`digitalglobe`'s old
`.earth` guess versus `maxar_gbdx`'s and formerly `terrabotics`'s real
domains), and a shut-down product with no clear failure message
(`maxar_gbdx`'s GBDX). If you find a provider on this page whose real
API has changed since these notes were written, that's expected —
satellite data APIs move — please open an issue with what changed.
