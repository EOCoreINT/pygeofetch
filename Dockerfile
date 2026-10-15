FROM python:3.11-slim

LABEL maintainer="PyGeoFetch Contributors"
LABEL description="Universal satellite data pipeline"

# Install system dependencies for rasterio/GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gdal-bin \
    libspatialindex-dev \
    libproj-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python package
COPY pyproject.toml setup.cfg README.md ./
COPY pygeofetch/ ./pygeofetch/

RUN pip install --no-cache-dir ".[all]"

# Create data directories
RUN mkdir -p /data /root/.pygeofetch

VOLUME ["/data", "/root/.pygeofetch"]

ENTRYPOINT ["pygeofetch"]
CMD ["--help"]
