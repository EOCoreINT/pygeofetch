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
      - PYGEOFETCH_PLANET_API_KEY=${PLANET_API_KEY}
      - PYGEOFETCH_COPERNICUS_USERNAME=${COPERNICUS_USER}
      - PYGEOFETCH_COPERNICUS_PASSWORD=${COPERNICUS_PASS}
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
prefer environment-variable credentials (as above) over the default
file backend when running headless in a container.
```
