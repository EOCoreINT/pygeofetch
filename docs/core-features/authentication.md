# Authentication

Credentials are stored in your system keyring — never in plain-text
files. Supports username/password, API keys, and OAuth2 client
credentials.

## Adding credentials

```bash
# Username / password (USGS, NASA, Copernicus)
pygeofetch auth add usgs --username YOUR_USER --password YOUR_PASS
pygeofetch auth add copernicus --username email@example.com --password PASS
pygeofetch auth add nasa_earthdata --username USER --password PASS

# API key (Planet, OpenTopography, TerraBotics, Airbus)
pygeofetch auth add planet --api-key YOUR_API_KEY
pygeofetch auth add opentopography --api-key YOUR_KEY

# OAuth2 client credentials (Sentinel Hub)
pygeofetch auth add sentinel_hub --client-id YOUR_ID --client-secret YOUR_SECRET

# Interactive — prompts for all required fields
pygeofetch auth login copernicus
```

## Managing credentials

```bash
pygeofetch auth list
pygeofetch auth test usgs
pygeofetch auth remove planet --yes

# Export backup — WARNING: contains secrets, store securely
pygeofetch auth export --output creds_backup.json
```

## Environment variables

For CI/CD and Docker, use `pygeofetch_{PROVIDER}_{FIELD}`:

```bash
export pygeofetch_USGS_USERNAME=myuser
export pygeofetch_USGS_PASSWORD=mypass
export pygeofetch_PLANET_API_KEY=PL-abc123
export pygeofetch_COPERNICUS_USERNAME=email@example.com
export pygeofetch_COPERNICUS_PASSWORD=pass
export pygeofetch_NASA_EARTHDATA_USERNAME=user
export pygeofetch_OPENTOPOGRAPHY_API_KEY=mykey
export pygeofetch_SENTINEL_HUB_CLIENT_ID=id
export pygeofetch_SENTINEL_HUB_CLIENT_SECRET=secret
```

```{note}
**Headless Linux (Docker, SSH):** if no D-Bus/keyring daemon is
running, set `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`
and use environment variables instead of `auth add`.
```

## Provider auth types

| Provider | Auth type | Free? |
|---|---|---|
| `usgs` | Username / password | Free (registration) |
| `copernicus` | Username / password (OAuth2) | Free (registration) |
| `nasa_earthdata` | Username / password | Free (registration) |
| `nasa_earthdata_cloud` | Username / password + S3 creds | Free (registration) |
| `planet` | API key | Subscription |
| `sentinel_hub` | OAuth2 client credentials | Freemium |
| `opentopography` | API key | Free tier |
| `maxar_gbdx` | API token | Subscription |
| `airbus_oneatlas` | API key | Subscription |
| `alaska_satellite_facility` | Earthdata (same as NASA) | Free |
| `google_earth_engine` | Service account JSON | Free tier |
| `terrabotics` | API key | Subscription |

## In Python

```python
from pygeofetch import PyGeoFetch

pf = PyGeoFetch()
pf.add_credentials("usgs", username="user", password="pass")
pf.add_credentials("planet", api_key="PL_KEY")
```

`add_credentials()` writes to the system keyring (or the encrypted
file fallback) and overrides any matching environment variable for
the rest of the process.
