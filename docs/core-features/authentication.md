# Authentication

Credentials are stored via a real `AuthManager`, backed by either a
Fernet-encrypted local file (the real default) or your OS keyring
(available via `auth_backend="keyring"`) — see
{doc}`/reference/security` for how the file backend's encryption
actually works, including the real base64→Fernet migration for
existing users. Supports username/password, API keys, and OAuth2
client credentials, depending on what each provider needs — see
{doc}`/core-features/providers` for which auth type each of the 24
providers uses.

```{note}
**Verified default, worth knowing explicitly**: `PyGeoFetch()` with
no arguments uses `auth_backend="file"` — the encrypted-file backend
is what almost everyone is actually using unless they've explicitly
passed `auth_backend="keyring"`. This matters for the CI/CD section
below: `auth_backend="file"` there isn't opting into anything
unusual, it's just being explicit about the same default behavior.
```

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

```{danger}
**A previous version of this page documented a
`PYGEOFETCH_{PROVIDER}_{FIELD}` environment-variable auto-loading
mechanism for credentials. It does not exist.** Verified by searching
the entire codebase for any environment-variable reading related to
credentials — there is none. `Credentials` is a plain data model with
no environment-variable support built in, and neither `AuthManager`
nor any individual provider reads `os.environ` to populate
credentials automatically. Setting `PYGEOFETCH_USGS_USERNAME` (or any
similar variable) and expecting `pygeofetch` to pick it up will not
work — your authentication will fail with no indication that the
environment variable was ever the problem.
```

The real, working way to authenticate in CI/CD, Docker, or any
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
    password=os.environ["USGS_PASSWORD"],
)
```

Store `USGS_USERNAME`/`USGS_PASSWORD` (any names you like — they're
just your own environment variables, not a pygeofetch convention)
using your CI system's own secrets mechanism (GitHub Actions
Secrets, GitLab CI/CD Variables, a Docker `--env-file`, etc.), then
run this setup step once before your real search/download code.

```{note}
**Headless Linux (Docker, SSH) if you've explicitly opted into
`auth_backend="keyring"`:** if no D-Bus/keyring daemon is running,
`keyring` itself may raise an error before you ever reach
pygeofetch's own code. Set
`PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` as a real,
separate environment variable (a `keyring` library convention, not a
pygeofetch one) to avoid that -- or simply don't pass
`auth_backend="keyring"` at all, since the real default
(`auth_backend="file"`) never touches a keyring daemon in the first
place.
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
| `google_earth_engine` | Service account JSON | See note below |
| `terrabotics` | API key | Subscription |

```{warning}
**`google_earth_engine` is not actually functional** regardless of
what credentials you provide — its real API needs a fundamentally
different Google service-account JWT auth flow and Earth Engine's
own asset/computation API, neither of which is implemented. It now
fails immediately with a clear message rather than crashing, but
`auth add google_earth_engine` will not make search/download work.
See {doc}`/core-features/providers`.
```

## In Python

```python
from pygeofetch import PyGeoFetch

pf = PyGeoFetch()
pf.add_credentials("usgs", username="user", password="pass")
pf.add_credentials("planet", api_key="PL_KEY")
```

`add_credentials()` writes to the system keyring (or the
Fernet-encrypted file fallback) and overrides any matching credential
already stored for the rest of the process — but, per the section
above, it does **not** read from environment variables on its own; you
pass the values in directly, from wherever your own code sourced them.
