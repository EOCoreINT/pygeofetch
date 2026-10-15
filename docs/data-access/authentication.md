# Authentication

Credentials are stored via a `AuthManager`, backed by either a
Fernet-encrypted local file (the default) or your OS keyring
(available via `auth_backend="keyring"`) — see
[Security Model](../reference/security.md) for how the file backend's encryption
actually works, including the base64→Fernet migration for
existing users. Supports username/password, API keys, and OAuth2
client credentials, depending on what each provider needs — see
[Providers](providers.md) for which auth type each of the 22
providers uses.

:::{note}

**Verified default, worth knowing explicitly**: `PyGeoFetch()` with
no arguments uses `auth_backend="file"` — the encrypted-file backend
is what almost everyone is actually using unless they've explicitly
passed `auth_backend="keyring"`. This matters for the CI/CD section
below: `auth_backend="file"` there isn't opting into anything
unusual, it's just being explicit about the same default behavior.
:::
## Adding credentials

```bash
# Username / password (NASA, Copernicus)
pygeofetch auth add copernicus --username email@example.com --password PASS
pygeofetch auth add nasa_earthdata --username USER --password PASS

# Username + M2M Application Token (USGS) -- NOT a password. USGS
# deprecated password-based login on 2025-02-26; the token replaces
# it, it isn't used alongside it. Request M2M access first at
# ers.cr.usgs.gov/profile/access (select "MACHINE" type), then
# generate a token from your ERS profile page once approved.
pygeofetch auth add usgs --username YOUR_USER --api-key YOUR_M2M_TOKEN

# API key (Planet, OpenTopography, Airbus)
pygeofetch auth add planet --api-key YOUR_API_KEY
pygeofetch auth add opentopography --api-key YOUR_KEY

# OAuth2 client credentials (Sentinel Hub)
pygeofetch auth add sentinel_hub --client-id YOUR_ID --client-secret YOUR_SECRET

# Interactive -- prompts for all required fields
pygeofetch auth login copernicus
```

## Managing credentials

```bash
pygeofetch auth list
pygeofetch auth test usgs
pygeofetch auth remove planet --yes

# Export backup -- WARNING: contains secrets, store securely
pygeofetch auth export --output creds_backup.json
```

## CI/CD, Docker, and headless environments

:::{danger}

**Credentials are not read from environment variables.** There is no
`PYGEOFETCH_{PROVIDER}_{FIELD}` auto-loading. `Credentials` is a plain data model, and neither
`AuthManager` nor any provider reads `os.environ`. Setting `PYGEOFETCH_USGS_USERNAME` (or a
similar variable) has no effect, and authentication then fails without any hint that the
variable was ignored.
:::
The working way to authenticate in CI/CD, Docker, or any
headless environment is to read your own secrets and pass them to
`add_credentials()` explicitly, in your own Python code or a small
setup script — pygeofetch doesn't need to know your CI system's
specific secret-injection mechanism, since your own code already has
the values:

```python
import os
from pygeofetch import PyGeoFetch

pf = PyGeoFetch(auth_backend="file")   # explicit -- also the real default, avoids any OS keyring dependency in a container

pf.add_credentials(
    "usgs",
    username=os.environ["USGS_USERNAME"],
    api_key=os.environ["USGS_M2M_TOKEN"],  # a real M2M Application Token, not your ERS password -- see Providers
)
```

Store `USGS_USERNAME`/`USGS_M2M_TOKEN` (any names you like — they're
just your own environment variables, not a pygeofetch convention)
using your CI system's own secrets mechanism (GitHub Actions
Secrets, GitLab CI/CD Variables, a Docker `--env-file`, etc.), then
run this setup step once before your search/download code.

:::{note}

**Headless Linux (Docker, SSH) if you've explicitly opted into
`auth_backend="keyring"`:** if no D-Bus/keyring daemon is running,
`keyring` itself may raise an error before you ever reach
pygeofetch's own code. Set
`PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` as a real,
separate environment variable (a `keyring` library convention, not a
pygeofetch one) to avoid that -- or simply don't pass
`auth_backend="keyring"` at all, since the default
(`auth_backend="file"`) never touches a keyring daemon in the first
place.
:::
## Provider auth types

| Provider | Auth type | Free? |
|---|---|---|
| `usgs` | Username + M2M Application Token (not a password — see [Providers](providers.md)) | Free (registration) |
| `copernicus` | Username / password (OAuth2) | Free (registration) |
| `nasa_earthdata` | Search: none (public). Download: Earthdata Login | Free (registration) |
| `nasa_earthdata_cloud` | Search: none (public). Download: Earthdata Login + per-DAAC S3 credentials (auto-refreshed, no single unified endpoint) | Free (registration) |
| `planet` | API key (HTTP Basic Auth) | Subscription |
| `sentinel_hub` | OAuth2 client credentials | Freemium |
| `opentopography` | API key | Free tier |
| `maxar_gbdx` | Bearer token (Discovery API — GBDX itself shut down 2022) | Subscription |
| `airbus_oneatlas` | API key | Subscription |
| `alaska_satellite_facility` | Search: none (public). Download: Earthdata Login (same as NASA), or a bearer token | Free |
| `google_earth_engine` | Service account JSON | See note below |

:::{warning}

**`google_earth_engine` is not actually functional** regardless of
what credentials you provide — its API needs a fundamentally
different Google service-account JWT auth flow and Earth Engine's
own asset/computation API, neither of which is implemented. It now
fails immediately with a clear message rather than crashing, but
`auth add google_earth_engine` will not make search/download work.
See [Providers](providers.md).
:::
## In Python

```python
from pygeofetch import PyGeoFetch

pf = PyGeoFetch()
pf.add_credentials("usgs", username="user", api_key="your_m2m_application_token")
pf.add_credentials("planet", api_key="PL_KEY")
```

`add_credentials()` writes to the system keyring (or the
Fernet-encrypted file fallback) and overrides any matching credential
already stored for the rest of the process — but, per the section
above, it does **not** read from environment variables on its own; you
pass the values in directly, from wherever your own code sourced them.