# Docker & Containers

:::{note}

**Image availability is not guaranteed by the Python package.** The `Dockerfile`,
`docker-compose.yml` and published images live in the repository, not in the PyPI package.
Confirm that `docker pull pygeofetch/pygeofetch:latest` succeeds before relying on it in CI/CD.
If it does not, build from source as shown below.
:::
## Quick start with the published image

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

:::{danger}

**Credentials are not read from environment variables** such as `PYGEOFETCH_PLANET_API_KEY`,
`PYGEOFETCH_COPERNICUS_USERNAME` or `PYGEOFETCH_COPERNICUS_PASSWORD`. No such mechanism exists.
The general settings variables above (`PYGEOFETCH_GENERAL__LOG_LEVEL` and similar) are read by
`Settings`, which only the `config` commands use (see [Configuration](configuration.md)). Supply
credentials by mounting `~/.pygeofetch` or by calling `add_credentials()` at start-up.
:::

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

:::{note}

Given the default credential storage caveat in [Security Model](security.md),
the encrypted-file backend (`auth_backend="file"`, already the real
default) combined with a mounted `~/.pygeofetch` volume, as shown
above, is the straightforward path for headless container use --
there's no environment-variable credential path to prefer instead,
per the correction above.
:::