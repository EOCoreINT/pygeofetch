# Docker & Containers

```{note}
**Not independently verified this pass:** the package archive audited
for this documentation contains only the Python package source, not
the repository root — so a `Dockerfile`, `docker-compose.yml`, or
published Docker Hub / GHCR images could not be confirmed to exist
from here. Confirm the image pulls successfully
(`docker pull pygeofetch/pygeofetch:latest`) before relying on it for
CI/CD; if it doesn't exist yet, build from source below.
```

## Quick start (if the published image exists)

```bash
docker pull pygeofetch/pygeofetch:latest

# Search — mount credentials and data output
docker run \
  -v ~/.pygeofetch:/root/.pygeofetch \
  -v $(pwd)/data:/data \
  pygeofetch/pygeofetch search run \
  --bbox "-74.1,40.6,-73.7,40.9" \
  --providers aws_earth \
  --output /data/results.geojson

# Download from saved results
docker run \
  -v ~/.pygeofetch:/root/.pygeofetch \
  -v $(pwd)/data:/data \
  pygeofetch/pygeofetch download run \
  --from-search /data/results.geojson \
  --output /data/scenes/ \
  --parallel 4
```

## Docker Compose for scheduled pipelines

```yaml
version: '3.8'
services:
  pygeofetch-scheduler:
    image: pygeofetch/pygeofetch:latest
    volumes:
      - ~/.pygeofetch:/root/.pygeofetch
      - ./pipelines:/pipelines
      - ./data:/data
    command: pygeofetch pipeline run /pipelines/weekly-ndvi.yaml
    restart: unless-stopped
    environment:
      - PYGEOFETCH_GENERAL__LOG_LEVEL=INFO
      - PYGEOFETCH_DOWNLOAD__PARALLEL=4
      - PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

```{danger}
**Real, verified correction**: an earlier version of this page showed
`PYGEOFETCH_PLANET_API_KEY`/`PYGEOFETCH_COPERNICUS_USERNAME`/
`PYGEOFETCH_COPERNICUS_PASSWORD` as environment variables the
container would pick up automatically for credentials. That
mechanism does not exist anywhere in pygeofetch's source — verified
by searching the entire codebase for any credential-related
environment-variable reading. `PYGEOFETCH_GENERAL__LOG_LEVEL` and
`PYGEOFETCH_DOWNLOAD__PARALLEL` above **are** real (pygeofetch's
general settings genuinely use `pydantic-settings` with
`env_prefix="PYGEOFETCH_"` and `__`-delimited nesting for config
values like log level or parallelism) — it's specifically
per-provider *credentials* that have no environment-variable path.

**The real, working pattern is the `~/.pygeofetch` volume mount
already shown above**, in the Quick Start section: run
`pygeofetch auth add planet --api-key ...` / `auth add copernicus
--username ... --password ...` once on the host (or in a one-time
setup job), so the real, Fernet-encrypted credentials file already
exists at `~/.pygeofetch/credentials.enc` before the container ever
starts — the container then just needs that directory mounted, no
credential-loading logic of its own. If you specifically need to
inject secrets from your CI system's own secret store rather than a
mounted host directory, do it explicitly in a short setup step before
your real command, e.g. as the container's entrypoint:

```bash
python -c "
from pygeofetch import PyGeoFetch
import os
pf = PyGeoFetch(auth_backend='file')
pf.add_credentials('planet', api_key=os.environ['PLANET_API_KEY'])
pf.add_credentials('copernicus', username=os.environ['COPERNICUS_USER'], password=os.environ['COPERNICUS_PASS'])
" && pygeofetch pipeline run /pipelines/weekly-ndvi.yaml
```

`PLANET_API_KEY`/`COPERNICUS_USER`/`COPERNICUS_PASS` here are your
own environment variable names (populated by Compose's `${VAR}`
substitution from your `.env` file or CI secrets, same as before) —
your own setup code reads them, not pygeofetch automatically.
```

## Build locally

```bash
git clone https://github.com/EOCoreINT/pygeofetch.git
cd pygeofetch
docker build -t pygeofetch:local .
docker run pygeofetch:local doctor
```

```{note}
Given the default credential storage caveat in {doc}`/reference/security`,
the encrypted-file backend (`auth_backend="file"`, already the real
default) combined with a mounted `~/.pygeofetch` volume, as shown
above, is the straightforward path for headless container use --
there's no environment-variable credential path to prefer instead,
per the correction above.
```
