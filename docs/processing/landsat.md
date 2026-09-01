# Landsat Extraction

```python
from pygeofetch.processor import LandsatExtractor

extractor = LandsatExtractor()
scene = extractor.process_scene(download_result, output_dir="./data")
print(scene.sensor)  # "OLI" or "TM"
red, nir = scene.get("red"), scene.get("nir")
```

Two easy-to-get-wrong things this handles:

- **Band mapping differs by sensor.** OLI (Landsat 8/9) and TM/ETM+
  (4/5/7) use different band numbers for the same wavelength —
  `SR_B4` is Red on OLI, NIR on TM/ETM+. Sensor is auto-detected and
  the correct map applied.
- **Delivery format differs by provider.** Not every provider delivers
  one archive — Planetary Computer downloads each band separately.
  `process_scene()` handles both delivery styles transparently.

Radiometric scaling verified against USGS's own worked examples (DN
18639 → 0.313 reflectance; DN 44947 → 302.6K). Cloud masking decodes
`QA_PIXEL` bits per the official Science Product Guide.
