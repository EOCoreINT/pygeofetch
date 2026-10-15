# Providers

pygeofetch registers **22 provider integrations** behind one query model. Twenty are
functional. Two are registered but cannot return data: `esa_scihub` (its upstream service
was decommissioned) and `google_earth_engine` (not implemented). They are listed under
[Registered but not usable](#registered-but-not-usable).

```bash
pygeofetch providers list                 # all providers
pygeofetch providers list --no-auth       # those that need no credentials
pygeofetch providers info planetary_computer
```

## Choosing a provider

| I need... | Use |
|---|---|
| Free Sentinel-2 or Landsat, no account | `planetary_computer`, `element84`, `aws_earth` |
| To analyse a small area without downloading scenes | `planetary_computer` or `element84` with [streaming](streaming.md) |
| Sentinel-1 SLC for InSAR | `alaska_satellite_facility` (free Earthdata account to download) or `copernicus` |
| Sentinel-1/2/3/5P from the official source | `copernicus` |
| Landsat 1-9 collections and historical imagery | `usgs`, `earth_explorer_additional` |
| MODIS, VIIRS, ICESat-2, GEDI, ASTER | `nasa_earthdata`, `nasa_earthdata_cloud` |
| A DEM for a bounding box | `opentopography`, `jaxa_earth` |
| Sub-metre commercial imagery | `airbus_oneatlas`, `maxar_gbdx`, `planet` |
| Free sample high-resolution imagery for disasters | `digitalglobe` (Maxar Open Data) |
| Geostationary weather imagery | `noaa_big_data` |
| Brazilian, Indian regional data | `inpe_cbers`, `isro_bhuvan` |
| Your own GeoServer layer | `geoserver_generic` |

## Status legend

| Symbol | Meaning |
|---|---|
| 🟢 | Implemented against the provider's current API documentation. |
| 🟡 | Open data; search works without credentials. |
| 🔐 | Credentials needed (search, download, or both; see the notes). |
| 🔴 | Registered but not usable. |

Provider APIs change. If a provider behaves differently from what is written here,
please [open an issue](https://github.com/EOCoreINT/pygeofetch/issues).
Verification against live accounts is not available for every provider.

## Open, no credentials needed

| Provider ID | Status | Data and behaviour |
|---|---|---|
| `planetary_computer` | 🟢 | Microsoft Planetary Computer STAC catalogue. Sentinel-1/2, Landsat 8/9, MODIS, NAIP, ALOS DEM. SAS tokens are generated per asset. Per-band COGs, so [streaming](streaming.md) works. |
| `element84` | 🟢 | Earth Search v1. Sentinel-2 L2A, Landsat Collection 2, Sentinel-1 RTC, Copernicus DEM. All COG; [streaming](streaming.md) works. |
| `aws_earth` | 🟢 | AWS Open Data Registry. Sentinel-2 COGs, Landsat Collection 2, NAIP, with direct S3 access. |
| `noaa_big_data` | 🟢 | GOES-16/17/18 imagery from the public, unsigned `noaa-goes16/17/18` S3 buckets. Listed and downloaded directly rather than through a search API. |
| `alaska_satellite_facility` | 🟢 | ASF DAAC: Sentinel-1 SLC/GRD, ALOS PALSAR, ERS, JERS. **Search is public.** It uses a WKT `intersectsWith` polygon, not a plain bbox, and no cloud-cover parameter (SAR is not affected by cloud). **Downloading needs a free NASA Earthdata Login** (username and password, or an EDL bearer token). |
| `digitalglobe` | 🟡 | Maxar Open Data Program imagery, released for **disaster response** events only. It is a static STAC catalogue on S3, with no server-side search: pygeofetch filters events by their extent and date, then scans items. `event_scan_limit` (default 60) caps how many events are scanned per call. |
| `jaxa_earth` | 🟡 | ALOS World 3D (AW3D30), a 30 m global DSM. A static 1°×1° tile grid, so "search" computes which tiles overlap your bbox and makes no network call. There are no date or cloud filters. Tiles are served through the public OpenTopography mirror. |
| `inpe_cbers` | 🟡 | Brazil Data Cube STAC API (`data.inpe.br/bdc/stac/v1`): CBERS-4, CBERS-4A, Amazonia-1. |
| `geoserver_generic` | 🟡 | A generic client for **any** GeoServer's OGC WFS `GetFeature`. You must configure both `endpoint` (the GeoServer base URL) and `layer` (a `typeName` the server advertises). Default is no auth; HTTP Basic is the standard secured mode, and an optional bearer token supports servers behind an OAuth2 proxy. |

## Credentials required

| Provider ID | Credentials | Data and behaviour |
|---|---|---|
| `copernicus` | 🔐 Username and password (OAuth2) | Copernicus Data Space Ecosystem: Sentinel-1/2/3/5P. Scenes are delivered as `.SAFE.zip`, which cannot be [streamed](streaming.md). Includes burst-family and orbit-file support for [InSAR](../processing/sar-insar/insar.md). The processing-level filter value for Sentinel-2 L2A is `S2MSI2A`, not `L2A`. |
| `usgs` | 🔐 Username and **M2M Application Token** | Landsat 1-9, MODIS, EO-1 and more through the Machine-to-Machine API. USGS replaced password login with an M2M token on 26 February 2025. First request M2M access at `ers.cr.usgs.gov/profile/access` (access type "MACHINE"; approval can take a couple of days), then generate an Application Token from your ERS profile. A token generated before approval does not work. The token replaces the password. |
| `earth_explorer_additional` | 🔐 Same as `usgs` | Declassified and historical imagery (Corona, KH-7 GAMBIT, KH-9 Hexagon) and early Landsat. A thin subclass of `usgs` with different default dataset codes. `declassii` (KH-7/KH-9 mapping camera) is confirmed; `declassi` and `declassiii` follow the same naming but are not independently confirmed. |
| `nasa_earthdata` | 🔐 Earthdata Login | NASA CMR: MODIS, VIIRS, ICESat-2, GEDI, ASTER. Search is public; download requires an account. |
| `nasa_earthdata_cloud` | 🔐 Earthdata Login plus per-DAAC S3 credentials | Cloud-hosted NASA data on AWS. Each DAAC (PO.DAAC, GES DISC, LP DAAC, ORNL DAAC, GHRC DAAC, NSIDC) issues its own S3 credentials, and one DAAC's credentials do not work for another's buckets. They are fetched on demand for each granule's DAAC and cached until expiry. Search is public. |
| `sentinel_hub` | 🔐 OAuth2 client | Catalog API (STAC). Collections are plain STAC IDs such as `sentinel-2-l2a`, not Process-API codes such as `S2L2A`. Cloud filtering uses CQL2-JSON. |
| `opentopography` | 🔐 API key | Global DEMs (SRTM, Copernicus DEM, ALOS World 3D, NASADEM) through `/API/globaldem`; USGS 3DEP rasters (1 m / 10 m / 30 m; the 1 m product is academic-use only) through `/API/usgsdem`; and LiDAR point-cloud **dataset discovery** through `/API/otCatalog`. Point-cloud extraction is not automated: OpenTopography has no single-request bbox clip for point clouds. |
| `isro_bhuvan` | 🔐 Free account | ISRO's **Bhoonidhi** STAC platform (`bhoonidhi-api.nrsc.gov.in`): ResourceSat-2/2A, EOS-04, EOS-06, a Sentinel-1A mirror, CartoSat-1 and NISAR. Rate limits: 20 auth requests per hour per IP, 3 searches per second per IP, 3 concurrent downloads per user per IP. |

## Commercial, sub-metre imagery

These need a paid account or contract with the vendor. pygeofetch provides search and
download; the licence is the one you purchase.

| Provider ID | Credentials | Data and behaviour |
|---|---|---|
| `airbus_oneatlas` | 🔐 API key | Pléiades (50 cm), Pléiades Neo (30 cm) and SPOT 6/7 from the OneAtlas **Living Library** (archive imagery). The API key is exchanged for a token; results include footprints and download links. Create the key in your OneAtlas account. |
| `maxar_gbdx` | 🔐 Bearer token | Maxar's **Discovery API** (`api.maxar.com/discovery/v1`): WorldView-1/2/3/4. The old GBDX platform was shut down on 4 January 2022. Maxar has rebranded to Vantor; the API domain is unchanged at the time of writing. The provider ID stays `maxar_gbdx` for backward compatibility. |
| `planet` | 🔐 API key | PlanetScope, SkySat, RapidEye. HTTP Basic auth with the API key as username. The asset type requested on download depends on the item type (SkySat has no `ortho_analytic_4b_sr`); override it in config. |

True centimetre resolution is not available from satellites; the sharpest commercial optical
imagery is about 30 cm. See [Choosing very-high-resolution imagery](#choosing-very-high-resolution-imagery).

## Registered but not usable

| Provider ID | Status | What happens |
|---|---|---|
| `esa_scihub` | 🔴 | Targets the Copernicus Open Access Hub, which ESA decommissioned on 2 November 2023. `search()` and `download()` fail immediately with a message pointing to `copernicus`. |
| `google_earth_engine` | 🔴 | Not implemented. Earth Engine needs a service-account JWT flow and its own asset and computation API, neither of which exists in pygeofetch yet. Calls fail immediately with an explanation. |

The modules `providers/eodag_provider.py` and `providers/terrabotics.py` are present in the
source tree but are **not registered** and cannot be selected. `eodag_provider` only wrapped
the third-party `eodag` package; `terrabotics` has no public API to verify against.

## Choosing very-high-resolution imagery

| Source | Resolution | Cost | Notes |
|---|---|---|---|
| Maxar WorldView (including Legion) | 30 cm class | Paid, per km² | Via `maxar_gbdx`. |
| Airbus Pléiades Neo | 30 cm | Paid, per km² with minimum order sizes | Via `airbus_oneatlas`. |
| Airbus Pléiades | 50 cm | Paid | Via `airbus_oneatlas`. |
| Planet SkySat | about 50 cm | Subscription or tasking | Via `planet`. |
| Maxar Open Data | 30-50 cm | Free | Disaster events only (`digitalglobe`). |

Distinguish **archive** imagery (already captured, quick and cheaper, date not controllable)
from **tasking** (a new capture on request, current, costs more and can take weeks).
Imagery at 1-10 cm per pixel comes from drones or aircraft, not satellites.

## In Python

```python
from pygeofetch.providers import list_providers, list_provider_info

list_providers()        # sorted provider IDs (22 registered)
list_provider_info()    # metadata: auth type, satellites, SAR/STAC flags, resolution range
```

`PyGeoFetch` itself has no `.providers()` method. Provider listing is a module-level
function.

## Provider maintenance

Provider APIs change without notice. Each provider module documents the endpoints it uses
at the top of its source file. When you find a provider whose current API differs, an
issue with the failing request and response is the most useful report. See
[Contributing](../project/contributing.md) for adding or repairing a provider.
