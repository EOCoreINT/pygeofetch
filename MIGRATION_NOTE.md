# Migration & Feature Note

## Optical validator — status against real STAC catalogs

Three of the four reported issues were re-verified against the current
source and a real, fresh checkout before any code was touched:

- **Band naming (Element84/AWS Earth semantic names, `eo:bands`
  nesting)** — already fixed. `validate_bands()` resolves through
  `pygeofetch.models.satellite_data._ALIAS_TO_CANONICAL`, so `red`/
  `green`/`blue`/`nir` and `eo:bands`-nested common names are all
  recognised as their real Sentinel-2 band codes.
- **Processing baseline vs. level confusion** — already fixed.
  `_scene_processing_level()` no longer reads `s2:processing_baseline`
  as a level value; `SatelliteData.from_stac_item()` derives a real
  `ProcessingLevel` from the STAC collection id (e.g.
  `sentinel-2-l2a` → `L2A`), and `S2MSI2A`/`L2A`/`Level-2A` compare
  equal via alias matching.
- **Cloud cover hard-failure default** — already correct.
  `cloud_cover_is_hard_failure=False` (the default) produces a
  `WARNING`, not an `ERROR`; the scene remains in `run_preflight()`'s
  returned results. The originally-reported failure mode does not
  reproduce against current source.

**One item, reframed rather than "fixed"**: AOI coverage for tiled
satellites is real, correct geometry, not a bug — a scene that
genuinely covers only 20% of a wider AOI should report 20% coverage.
This is already fully configurable (`min_coverage_ratio`); the fix for
a multi-tile mosaic workflow is to lower that value explicitly, not to
silently change the default's meaning. Auto-detecting "mosaic mode"
and applying a different implicit threshold was considered and
rejected — it would make the check's behavior depend on how many
scenes happen to be in a batch, which is a worse, harder-to-reason-about
default than an explicit, user-set threshold.

New regression tests: `tests/test_optical_validator_stac.py` (14
tests), built against real Element84/AWS Earth STAC response shapes,
including the `eo:bands` nested-extension variant.

## New: Optical Pixel Offset Tracking (POT)

`pygeofetch.optical` — new module, real 2D ground displacement
measurement via optical image correlation, for motion that exceeds
phase InSAR's coherence limit (mining subsidence, fast landslides,
large co-seismic deformation).

```python
from pygeofetch.optical import prepare_optical_pair, compute_pixel_offsets

ref, sec, profile = prepare_optical_pair(
    "before.tif", "after.tif",
    cloud_mask_path="before_SCL.tif",
)
result = compute_pixel_offsets(
    ref, sec, pixel_size_m=abs(profile["transform"].a),
    window_size=64, step_size=16,
)
result.export_geotiff("displacement.tif", output_profile)
```

```bash
pygeofetch optical offset-track --ref before.tif --sec after.tif \
    --cloud-mask before_SCL.tif --output displacement.tif
```

**Built on the existing, already-verified `pygeofetch.insar.offset_tracking`
correlation engine** (real NCC via FFT, parabolic sub-pixel refinement,
SNR-based confidence — Lewis 1995 / Debella-Gilo & Kääb 2011 /
ampcor-style formulations, already cited and tested there) rather than
a second, independent reimplementation of the same core math. What's
genuinely new: CRS-aware reprojection/alignment (`rasterio.warp`),
real cloud/water masking (SCL or NDWI-threshold fallback), and
pixel-to-metres conversion valid for optical imagery in a real
projected CRS with square pixels — explicitly distinct from SAR
range/azimuth offsets, which need real orbit geometry to convert to
ground distance and are deliberately left to a separate module there.

**Empirically measured precision** (not assumed): recovers a known
2.5px / -1.3px synthetic sub-pixel shift to within 0.012px / 0.059px
on a realistic, non-periodic texture.

**A real, honest caveat documented in the module itself, not hidden**:
the reused SNR metric can score literal uniform random noise as
artificially reliable at realistic window sizes, due to extreme-value
statistics over many candidate search offsets — measured directly
(mean SNR 8.8 for synthetic white noise vs. 6.1 for a real correlated
match). This doesn't affect real optical imagery, which never looks
like uniform white noise, but it means SNR alone shouldn't be treated
as an absolute signal without masking known-bad regions (cloud, water,
nodata) first via `prepare_optical_pair`.

New tests: `tests/test_optical_offset_tracking.py` (11 tests) —
synthetic sub-pixel shift recovery (two shift magnitudes), NaN/nodata
handling, SNR-based comparative filtering (using a realistically
different texture, not adversarial iid noise), and result-shape/
metadata checks.

New CLI group: `pygeofetch optical offset-track` — registered in
`pygeofetch/cli/main.py` alongside the existing `sar`/`preprocess`/
`index`/`post` groups, following the same structure.

## Not changed

`pygeofetch.insar.offset_tracking` (the SAR/amplitude offset tracker)
is untouched — the new optical module imports and reuses it, rather
than modifying its already-tested internals.
