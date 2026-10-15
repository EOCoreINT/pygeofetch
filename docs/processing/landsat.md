# Landsat Extraction

```python
from pygeofetch.processor import LandsatExtractor

extractor = LandsatExtractor(mask_clouds=True)
scene = extractor.process_scene(download_result, output_dir="./data")
print(scene.sensor)  # "OLI" or "TM"
red, nir = scene.get("red"), scene.get("nir")
print(scene.available_bands())
```

Not wired into `PyGeoFetch` (there's no `client.landsat`) — always a
standalone import, unlike `client.indices`/`client.sar`.

## Two easy-to-get-wrong things this handles

- **Band mapping differs by sensor.** OLI (Landsat 8/9) and TM/ETM+
  (4/5/7) use different band numbers for the same wavelength —
  `SR_B4` is Red on OLI, NIR on TM/ETM+. Sensor is auto-detected from
  the filename (`_detect_sensor()`) and the correct map applied.
- **Delivery format differs by provider.** Not every provider delivers
  one archive — Planetary Computer downloads each band as its own
  separate asset file (`SR_B4.TIF`, `SR_B5.TIF`, `QA_PIXEL.TIF`, ...)
  rather than one `.tar` bundle, so `DownloadResult.output_paths`
  holds many files, not one. `process_scene()` detects this
  automatically (`_resolve_individual_files()`) and uses those files
  directly, skipping tar extraction entirely — the same call works
  regardless of which delivery style the provider used.

## `process_scene()` — the full chain

```python
scene = extractor.process_scene(
    source, output_dir, bands=None, label="", mask_clouds=None,
)
```

The full chain in one call: extract bundle → scale bands → cloud-mask
→ return. `source` accepts a `DownloadResult` (preferred — uses
`.output_path`/`.output_paths` directly) or a direct path to a `.tar`
bundle. `bands` defaults to
`["blue","green","red","nir","swir1","swir2"]` — pass fewer for
faster processing when you only need, e.g., `["red","nir"]` for NDVI.
`label` namespaces the extraction subfolder when processing multiple
scenes into the same `output_dir` (e.g. `label="before"` /
`label="after"` for change detection). `mask_clouds` overrides the
instance-level setting for just this call.

## Individual building-block methods

`process_scene()` is a convenience wrapper — each step is also
independently callable:

| Method | What it does |
|---|---|
| `extract_bundle(tar_path, output_dir)` | Extract every band + QA GeoTIFF from a Landsat C2L2 `.tar` bundle. Returns `{member_filename: extracted_path}`. |
| `load_scaled_band(path)` | Load one band and apply the correct **official USGS scale factor + offset**, auto-selected by band type (surface reflectance `SR_B*` vs. surface temperature `ST_B*`). Fill pixels (`DN=0`, the real Collection 2 fill value) become NaN. Returns `(scaled_array, rasterio_profile)`. |
| `cloud_mask(qa_pixel_path, shape=None)` | Decode Landsat Collection 2 `QA_PIXEL` bit flags into a cloud/shadow mask (`True` = masked out). Checks bits 1 (Dilated Cloud), 3 (Cloud), and 4 (Cloud Shadow) per the official Landsat 8-9 C2 Level-2 Science Product Guide, Table 6-2/6-3 — the same bit positions apply across the full Landsat 4-9 Collection 2 `QA_PIXEL` spec. |
| `find_band(extracted_files, band_suffix)` | Look up an extracted file by band suffix, e.g. `"SR_B4"` or `"QA_PIXEL"`. |

```python
# Using the pieces directly, e.g. to add a custom step between them
extracted = extractor.extract_bundle("LC08_..._SR.tar", "./raw/")
red_path = extractor.find_band(extracted, "SR_B4")
red, profile = extractor.load_scaled_band(red_path)

qa_path = extractor.find_band(extracted, "QA_PIXEL")
cloud_mask = extractor.cloud_mask(qa_path, shape=red.shape)
red_masked = red.copy()
red_masked[cloud_mask] = float("nan")
```

## `LandsatScene` — the returned object

```python
scene.get("red")             # -> scaled numpy array, or None if that band wasn't loaded
scene.available_bands()      # -> list of band names actually present on this scene
scene.sensor                 # -> "OLI" (Landsat 8/9) or "TM" (Landsat 4/5/7)
```

## Verification basis

Radiometric scaling verified against USGS's own worked examples: DN
18639 → 0.313 reflectance; DN 44947 → 302.6 K. Cloud masking decodes
real `QA_PIXEL` bits per the official Science Product Guide, not an
approximated threshold.
