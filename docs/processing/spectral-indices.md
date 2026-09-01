# Spectral Indices

```bash
pip install "pygeofetch[processor]"
```

17 built-in indices, or 232+ via the optional `spyndex` package.

```python
from pygeofetch.processor import SpectralIndex

si = SpectralIndex()
ndvi = si.compute("NDVI", RED=red_array, NIR=nir_array)
ndvi = si.from_files("NDVI", red="B04.tif", nir="B08.tif", output="ndvi.tif")
```

## Built-in indices

| Category | Indices |
|---|---|
| Vegetation health | NDVI, EVI, SAVI |
| Surface water | NDWI, MNDWI |
| Built-up areas | NDBI |
| Burn severity | NBR, dNBR |
| Other | NDSI, NDMI, BSI, ARVI, GNDVI, RVI, VCI, CRI1, PSRI |

```python
si.available()          # -> list of all built-in index names
si.info("NDVI")          # -> formula, required bands, valid range
```

Passing `prefer_spyndex=True` (the default) to `SpectralIndex()` uses
the `spyndex` package's much larger catalogue when it's installed,
falling back to the 17 built-ins otherwise.
