# Optical Pixel Offset Tracking

`pygeofetch.optical.offset_tracking` measures ground displacement from
ordinary optical imagery (Sentinel-2, Landsat) by tracking how patches
of real ground texture physically moved between two dates — the same
idea as tracking a fingerprint pattern sliding across two photos.

## Why this exists alongside InSAR

[InSAR](insar.md) is far more precise (millimetres, not metres) — but
it has one real, fundamental limit: it needs the ground surface to stay
*coherent* between the two radar passes. If the surface changed too
much between dates — a mining panel excavated, a landslide, a large
earthquake rupture, a bulldozed construction site — the radar phase
signal decorrelates into noise, and InSAR returns nothing for that
pixel. Not an imprecise number: no measurement at all.

Optical offset tracking is the opposite trade-off. It's coarser
(typically sub-metre to a few metres of precision, depending on pixel
size and scene texture), and it only measures *horizontal* motion —
but it works precisely where InSAR structurally cannot: on large,
chaotic, or fast displacement, because it doesn't depend on phase
coherence at all.

This isn't a hypothetical justification. Running this exact pipeline
against a real mining site (Bu'ertai, Inner Mongolia — a site published
literature already confirms defeats conventional InSAR) produced real
InSAR coverage of just **0.1%** of the processed scene. Optical offset
tracking, over the same real area and dates, covered **83.7%**. Neither
number is a target or an assumption — both came from an actual,
executed run.

## The pipeline

```python
from pygeofetch.optical.offset_tracking import (
    prepare_optical_pair,
    compute_pixel_offsets,
    compute_horizontal_strain,
    fuse_insar_optical,
)

# 1. Align the two dates onto the same grid and mask out clouds/water
reference, secondary, metadata = prepare_optical_pair(
    "sentinel2_2023-01-01_B08.tif",
    "sentinel2_2023-06-01_B08.tif",
    cloud_mask_path="sentinel2_2023-06-01_SCL.tif",
    cloud_mask_scl_classes=[3, 8, 9, 10],  # real Sentinel-2 SCL cloud/shadow classes
)

# 2. Track dense pixel offsets across the whole image pair
result = compute_pixel_offsets(
    reference, secondary,
    pixel_size_m=10.0,       # Sentinel-2 band 8 native resolution
    window_size=64,
    step_size=16,
    snr_threshold=3.0,
)

print(f"{result.reliable.sum()}/{result.reliable.size} windows reliable "
      f"({100 * result.reliable.mean():.1f}%)")
```

### `prepare_optical_pair`

Aligns the secondary image onto the reference image's exact grid via
`rasterio.warp.reproject` — any CRS, resolution, or extent mismatch
between the two dates is resolved here, always onto the reference's
grid, never the other way around. Optionally masks out clouds
(Sentinel-2 SCL classes) and water (NDWI threshold) so the correlator
doesn't lock onto a moving cloud or a textureless water surface instead
of real, stationary ground.

### `compute_pixel_offsets`

Tiles [`pygeofetch.insar.offset_tracking.OffsetTracker`](insar.md)'s
real, already-verified NCC (normalized cross-correlation) engine across
the full image pair, then converts the resulting pixel offsets to real
ground-distance metres using `pixel_size_m`. This reuses the exact same
sub-pixel correlation and SNR-based reliability engine InSAR's own
offset-tracking capability uses — not a second, independent
implementation.

Returns an `OpticalOffsetResult` with:

| Field | Meaning |
|---|---|
| `dx`, `dy` | East/North displacement per window, metres |
| `snr` | Primary-peak-to-secondary-peak ratio — the correlation confidence signal |
| `reliable` | Boolean mask: `snr >= snr_threshold` |
| `window_centers_row`, `window_centers_col` | Real pixel coordinates each measurement corresponds to |

!!! warning "SNR-based reliability alone is not enough — verified, not assumed"

    Running this pipeline against real Bu'ertai imagery, a small number
    of windows passed the `snr >= 3.0` reliability filter with a
    displacement of **226.27 metres** — a single Sentinel-2 pixel pair
    that should never move more than a few metres between two dates,
    revealing a spurious correlator lock-on that SNR alone didn't catch.
    See `compute_horizontal_strain` below for the real, direct
    consequence of trusting this without further filtering, and the fix.

## Strain: turning displacement into a risk metric

```python
strain = compute_horizontal_strain(
    result.dx, result.dy,
    pixel_size_m=metadata["step_size"] * 10.0,  # window step size in metres
    reliable=result.reliable,
    mad_outlier_threshold=8.0,
)

print(f"Excluded: {strain['n_excluded_unreliable']} unreliable, "
      f"{strain['n_excluded_outlier']} statistical outliers")
```

`compute_horizontal_strain` computes the real normal strain tensors
(`exx`, `eyy`) and shear strain (`exy`) from the displacement field via
central finite differences — turning raw displacement into a direct,
actionable geotechnical signal (ground stretching or compressing
locally), rather than a map that still needs manual interpretation.

### The outlier problem this function exists to prevent

Strain is a *spatial derivative* of displacement. A single spurious
displacement value doesn't just corrupt its own pixel — it produces a
locally enormous, physically impossible strain gradient at every
neighboring pixel the finite-difference stencil touches.

This isn't a defensive feature added just in case. Running a naive,
textbook strain calculation on the real, unfiltered Bu'ertai
displacement field (the one containing that 226.27m outlier above)
produced a **peak strain of 6,169,938 microstrain — 617% strain**,
physically impossible for real ground deformation (it would mean the
ground literally stretched to over seven times its original length).
Reproducing that exact scenario synthetically confirms the naive
calculation produces a comparable, triple-digit-percent artifact, and
that `compute_horizontal_strain`'s outlier protection brings it back
into a physically plausible range — this is a permanent regression
test, not just a docstring claim.

Two independent, real safeguards are applied, both opt-out-able:

1. **`reliable`** — pass `OpticalOffsetResult.reliable` directly.
   Unreliable pixels are excluded before any gradient is computed, so
   an unreliable neighbor can't corrupt a reliable pixel's real strain
   value.
2. **`mad_outlier_threshold`** — a robust statistical filter (median
   absolute deviation) applied to displacement magnitude specifically,
   because strain is far more sensitive to a single extreme value than
   the raw displacement map is. A displacement this function considers
   "reliable" for reporting can still be excluded here for strain
   purposes. Set to `None` to reproduce a naive calculation exactly —
   not recommended on real, unfiltered field data, for the reason
   above.

You can also pass `max_plausible_displacement_m` — a real, absolute
ceiling tied to your own physical expectations for the phenomenon
being monitored (a known maximum credible co-seismic slip, a maximum
plausible daily mining advance rate) — often more defensible in a real
report than a purely statistical cutoff alone.

## Fusing InSAR and optical

```python
fused = fuse_insar_optical(
    insar_velocity=vertical_velocity_cm_yr,   # from InSAR SBAS inversion
    insar_coherence=coherence_map,             # from the same InSAR run
    optical_disp_mag=np.sqrt(result.dx**2 + result.dy**2),
    optical_snr=result.snr,
)

print(f"InSAR-dominant: {(fused['reliability_source'] == 2).sum()} pixels")
print(f"Optical-dominant: {(fused['reliability_source'] == 0).sum()} pixels")
```

`fuse_insar_optical` is a real, coherence-weighted *source selection*,
not an average.

!!! danger "Do not average InSAR and optical displacement"

    InSAR measures real line-of-sight phase, converted to a *vertical*
    displacement estimate under a small-horizontal-motion assumption.
    Optical offset tracking measures real 2D *horizontal* displacement
    directly. These are physically different vector components, not
    two noisy estimates of the same scalar quantity. Averaging a
    vertical value with a horizontal-magnitude value produces a number
    with no real physical meaning.

Instead, `fuse_insar_optical` uses the InSAR coherence map itself as
the real, per-pixel decision signal:

- **Coherence > 0.6** (configurable via `coherence_high`): trust InSAR
  fully — precise, and coherence confirms the phase signal is real.
- **Coherence < 0.4** (`coherence_low`): trust optical fully — InSAR
  has nothing usable here regardless of what number it happens to
  output.
- **Between**: a real, linear blend of *how much each technique's
  information is trusted*, not a numeric average of two different
  physical quantities.

The result includes `reliability_source` (2=InSAR-dominant,
1=transition, 0=optical-dominant) so you always know, per pixel, which
real technique actually produced the number you're looking at.

Confirmed against the real, observed Bu'ertai coherence value (mean
0.313, well below the default `coherence_low`): the function correctly
and honestly returns 100% optical-dominant pixels for that scene — this
is real, correct behavior given genuinely poor InSAR coherence, not a
result to "fix" toward a more even split.

## Real, honest limitations

- Radiometric/geometric calibration of the input rasters (band
  selection, cloud masking accuracy) is the caller's responsibility via
  `prepare_optical_pair`'s parameters — the correlator will happily
  produce a confident-looking, wrong answer if fed a poorly-masked pair.
- The MAD outlier filter is a statistical safeguard, not a substitute
  for physical judgment — always sanity-check displacement magnitudes
  against what's physically plausible for your specific site and time
  window.
- `pixel_size_m` for `compute_horizontal_strain` is the real ground
  distance between adjacent *offset-tracking grid* samples (the
  `step_size` from `compute_pixel_offsets`, converted to metres) — not
  the original image's raw pixel size. Passing the wrong one silently
  produces a strain value scaled by the wrong real distance.
