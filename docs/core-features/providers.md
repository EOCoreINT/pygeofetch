# Providers

pygeofetch ships 24 provider integrations. **"Listed" and "verified
working" are not the same thing** — treat the status column below as
the honest signal, not the presence of a provider module.

## Status legend

- 🟢 **Verified** — tested end-to-end, including real footprint geometry
- 🟡 **Open** — no-auth access works; not independently re-verified this pass
- 🔴 **Dead** — the upstream service is confirmed decommissioned
- 🔐 **Auth required**

```{note}
A systematic bug affecting 12 providers was found and fixed in a
previous audit pass: real footprint geometry returned by a provider's
own API was being silently discarded and replaced with a bounding-box
rectangle. The fix is verified against synthetic data; each affected
provider's live API has not been independently re-confirmed since.

**Since that pass, four providers received deeper, individually
verified fixes** (real API research, not just the shared geometry
fix): `airbus_oneatlas` and `noaa_big_data` were fully rewritten
against their real, documented APIs and are now genuinely functional
end to end; `esa_scihub` and `google_earth_engine` previously crashed
on every single call (`AttributeError` on a method that doesn't exist
anywhere in the codebase) and now fail honestly instead — see each
one's notes below.
```

## Open, no-auth providers

| Provider ID | Status | Notes |
|---|---|---|
| `planetary_computer` | 🟢 Verified | Microsoft STAC catalog. Sentinel-1/2, Landsat 8/9, MODIS, NAIP, ALOS DEM. SAS tokens auto-generated. |
| `aws_earth` | 🟢 Verified | AWS Earth Open Data. Sentinel-2 COGs, Landsat Collection 2, NAIP. Direct S3 access. |
| `element84` | 🟢 Verified | Earth Search v1. Sentinel-2 L2A, Landsat Col 2, Sentinel-1 RTC, COP-DEM. All COG. |
| `noaa_big_data` | 🟢 Verified | GOES-16/17/18 imagery. Real, public, unsigned S3 buckets (`noaa-goes16/17/18`) — listed and downloaded directly, not via a search API. Fully rewritten and tested; see the note above. |
| `esa_scihub` | 🔴 Dead | Points at the Copernicus Open Access Hub, decommissioned Nov 2, 2023. Previously crashed with `AttributeError` on every call; now fails immediately with a clear message pointing to `copernicus` instead. Use `copernicus`. |
| `eodag_provider` | 🟡 Open | Gateway to 20+ providers (Theia, PEPS, Mundi, Copernicus ADS). Search works without auth; some backing providers need it for download. |
| `jaxa_earth` | 🟡 Open | JAXA ALOS 30m World 3D DSM and PALSAR-2 forest/non-forest map. |
| `isro_bhuvan` | 🟡 Open | ISRO Bhuvan. ResourceSat-2/2A (5.8m), Cartosat-1 (2.5m), Oceansat-2. |
| `inpe_cbers` | 🟡 Open | Brazil INPE CBERS-4/4A. 5m–40m optical, free download. |
| `digitalglobe` | 🟡 Open | Maxar Open Data Program. Sub-metre WorldView disaster response imagery. |
| `geoserver_generic` | 🟡 Open | Generic OGC WMS/WFS/WCS connector for any compliant endpoint. |

## Authenticated providers

| Provider ID | Auth | Status | Notes |
|---|---|---|---|
| `usgs` | 🔐 Username/password | 🟢 Verified | Landsat 1–9, ASTER, MODIS, EO-1. Machine-to-Machine API. |
| `copernicus` | 🔐 OAuth2 | 🟢 Verified | Copernicus Data Space Ecosystem. Sentinel-1/2/3/5P. Full STAC. |
| `sentinel_hub` | 🔐 OAuth2 | 🟢 Verified | Processing API. Sentinel-1/2/3, Landsat, custom evalscripts. |
| `airbus_oneatlas` | 🔐 API key | 🟢 Verified | Pléiades, Pléiades Neo, SPOT 6/7 via the real OneAtlas Living Library API. Fully rewritten: real token-exchange auth, real `/opensearch` endpoint, real asset population — previously all three were fictional/broken and `download()` always failed. |
| `nasa_earthdata` | 🔐 Username/password | Auth required | NASA CMR. MODIS, VIIRS, ICESat-2, GEDI, ASTER. Free registration. |
| `nasa_earthdata_cloud` | 🔐 + S3 creds | Auth required | Cloud-native NASA data on AWS, temporary S3 credentials auto-refreshed. |
| `opentopography` | 🔐 API key | Auth required | SRTM 30/90m, COP-DEM 30m, NASADEM, LiDAR point clouds. Free key. |
| `planet` | 🔐 API key | Auth required | PlanetScope (3m daily), SkySat (50cm), RapidEye. Subscription. |
| `maxar_gbdx` | 🔐 API token | Auth required | WorldView 1–4, GeoEye-1. 30–50cm. Commercial subscription. |
| `alaska_satellite_facility` | 🔐 Earthdata login | Auth required | ASF DAAC. Sentinel-1 SLC/GRD, ALOS PALSAR, UAVSAR. |
| `google_earth_engine` | 🔐 Service account JSON | 🔴 Not implemented | Earth Engine's real API (Google service-account JWT auth, plus its own asset/computation API — not a REST search endpoint) is structurally different from every other provider in this list. Previously crashed with `AttributeError` on every call; now fails immediately with a clear message explaining exactly what real integration work remains, rather than crashing or silently pretending to work. |
| `terrabotics` | 🔐 API key | Auth required | Archive and tasking. Sub-metre commercial imagery. |
| `earth_explorer_additional` | 🔐 Same as USGS | Auth required | USGS Earth Explorer's declassified/historical datasets, separate from the main Landsat catalog. |

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

list_providers()        # -> sorted list of provider ID strings
list_provider_info()     # -> rich metadata: auth type, capabilities, satellites, etc.
```

```{note}
`PyGeoFetch` itself has no `.providers()` method — provider listing is
a module-level function on `pygeofetch.providers`, not a method on the
client class.
```
