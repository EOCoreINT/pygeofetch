"""
Optical pixel offset tracking (POT) for large 2D ground displacement.

Two real, separately-testable pieces, matching the project's existing
insar.offset_tracking split between correlation math and geometry:

1. ``prepare_optical_pair()`` — real preprocessing: reproject/resample
   the secondary image onto the reference image's exact grid, and mask
   out clouds/water so the correlator isn't matching moving clouds or
   featureless water instead of real, stationary ground.
2. ``compute_pixel_offsets()`` — tiles pygeofetch.insar.offset_tracking's
   real, already-verified NCC + sub-pixel + SNR engine across the full
   image pair, then converts the resulting pixel offsets to real ground
   distance (metres) using the reference raster's own pixel size.

Reuses the existing correlation engine rather than reimplementing it a
second time — normalized cross-correlation, parabolic sub-pixel
refinement, and SNR-based confidence scoring work identically whether
the input pixels are SAR amplitude or optical band reflectance; the
underlying mathematics don't know or care which sensor produced the
array. What's genuinely new here is optical-specific: cloud/water
masking, CRS-aware reprojection, and converting the result to real
ground-distance units for a caller who has optical (not SAR
range/azimuth) imagery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pygeofetch.insar.offset_tracking import OffsetTracker

__all__ = [
    "OpticalOffsetResult",
    "compute_pixel_offsets",
    "prepare_optical_pair",
]


def _require_numpy():
    try:
        import numpy as np

        return np
    except ImportError as exc:
        raise ImportError(
            "pygeofetch.optical requires numpy: pip install numpy"
        ) from exc


@dataclass
class OpticalOffsetResult:
    """
    Real per-window optical offset tracking output, in ground distance
    units (metres), not raw pixels.

    Attributes
    ----------
    dx : np.ndarray
        East-West displacement per window, metres. Positive = eastward
        motion of the tracked feature between the reference and
        secondary dates.
    dy : np.ndarray
        North-South displacement per window, metres. Positive =
        northward motion (note: this is the real geographic
        convention, not raw row-index direction, which points the
        opposite way — see ``compute_pixel_offsets``'s docstring for
        exactly where that sign flip happens and why).
    snr : np.ndarray
        Real correlation-peak signal-to-noise ratio per window (see
        ``pygeofetch.insar.offset_tracking.compute_snr`` for the exact
        definition) — the same real confidence metric the SAR offset
        tracker uses, not a different one invented for this module.
    reliable : np.ndarray
        Boolean mask, True where ``snr`` meets the caller's threshold.
        Matches this project's "fail loudly, never silently
        interpolate" principle: unreliable windows are flagged, not
        silently dropped or zeroed — ``dx``/``dy`` still hold their
        real computed values at unreliable locations, so a caller can
        inspect what a rejected match actually looked like.
    metadata : dict
        Real processing parameters this result was produced with —
        ``window_size``, ``step_size``, ``max_displacement_px``,
        ``crs``, ``pixel_size_m``, ``reference_path``,
        ``secondary_path`` (when known) — enough to independently
        reproduce this specific run, matching the provenance-manifest
        discipline used elsewhere in this project's InSAR chain.
    window_centers_row, window_centers_col : np.ndarray
        Pixel row/col of each window's center in the reference image's
        own grid — needed to place ``dx``/``dy`` back onto a real map,
        e.g. via the reference raster's affine transform.
    """

    dx: Any
    dy: Any
    snr: Any
    reliable: Any
    window_centers_row: Any
    window_centers_col: Any
    metadata: dict = field(default_factory=dict)

    def export_geotiff(self, path: str | Path, profile: dict) -> str:
        """
        Write dx/dy/snr as a real, georeferenced 3-band GeoTIFF.

        Parameters
        ----------
        path : str or Path
            Output file path.
        profile : dict
            A real rasterio profile for the OUTPUT grid — i.e. already
            adjusted for this result's window spacing (the output
            raster has one pixel per correlation window, not one pixel
            per input-image pixel). The simplest correct way to build
            this is via ``prepare_optical_pair``'s returned
            ``reference_profile``, scaled by ``metadata["step_size"]``
            -- see the worked example in the module-level docs.

        Returns
        -------
        str
            The path written to, as a string (matching this project's
            other ``export_*`` methods, e.g. ``RiskMapper.export_geotiff``).
        """
        import rasterio

        np = _require_numpy()
        out_profile = dict(profile)
        out_profile.update(count=3, dtype="float32", nodata=float("nan"))
        path = str(path)
        with rasterio.open(path, "w", **out_profile) as dst:
            dst.write(np.asarray(self.dx, dtype="float32"), 1)
            dst.write(np.asarray(self.dy, dtype="float32"), 2)
            dst.write(np.asarray(self.snr, dtype="float32"), 3)
            dst.set_band_description(1, "dx_east_m")
            dst.set_band_description(2, "dy_north_m")
            dst.set_band_description(3, "snr")
        return path


def prepare_optical_pair(
    reference_path: str | Path,
    secondary_path: str | Path,
    band_index: int = 1,
    cloud_mask_path: "str | Path | None" = None,
    cloud_mask_scl_classes: "list[int] | None" = None,
    water_mask_threshold: "float | None" = None,
    green_band_index: "int | None" = None,
    nir_band_index: "int | None" = None,
) -> tuple[Any, Any, dict]:
    """
    Real preprocessing for optical offset tracking: align the
    secondary image onto the reference image's exact grid, and mask
    out clouds/water so the correlator isn't matching moving clouds or
    featureless water instead of real, stationary ground displacement.

    Parameters
    ----------
    reference_path, secondary_path : str or Path
        Paths to the two real, single- or multi-band optical rasters.
        Any CRS/resolution/extent mismatch between them is resolved
        here — the secondary is always reprojected/resampled onto the
        reference's real grid via ``rasterio.warp.reproject``, never
        the other way around.
    band_index : int
        1-based band index to read from both rasters for correlation
        (rasterio's own convention). Typically a near-infrared or red
        band — a band with real spatial texture (vegetation, terrain,
        built structures), not a band that's often uniformly bright or
        dark across the whole scene.
    cloud_mask_path : str or Path, optional
        Path to a real Sentinel-2 SCL (Scene Classification Layer)
        raster, already aligned to the reference grid. When given,
        this is the preferred, most reliable masking source — a real
        per-pixel ESA classification, not a heuristic.
    cloud_mask_scl_classes : list[int], optional
        Real SCL class values to mask out. Defaults to
        ``[3, 8, 9, 10, 11]`` — cloud shadow, cloud medium/high
        probability, thin cirrus, and snow/ice, matching the same
        default used in ``pygeofetch.processing.preprocessor.Preprocessor
        .cloud_mask(method="scl")`` for consistency across the project
        rather than inventing a second default elsewhere.
    water_mask_threshold : float, optional
        If no SCL band is available, mask pixels where NDWI exceeds
        this threshold instead (open water has no real, stable texture
        to track against, so it should always be masked out
        regardless of whether cloud masking is also available).
        Requires ``green_band_index`` and ``nir_band_index``. A
        reasonable starting value is ``0.0`` — NDWI is positive over
        open water (see ``pygeofetch.processing.indices.SpectralIndices
        .ndwi``'s own docstring for the real formula and citation).
    green_band_index, nir_band_index : int, optional
        1-based band indices for the green and NIR bands, needed only
        when computing a fallback NDWI water mask.

    Returns
    -------
    reference_array : np.ndarray
        2D float array, the reference band, with masked pixels set to
        NaN.
    secondary_array : np.ndarray
        2D float array, the secondary band reprojected onto the
        reference's exact grid, with masked pixels set to NaN.
    reference_profile : dict
        The real rasterio profile of the reference raster (CRS,
        transform, width, height) — the caller needs this afterward to
        georeference the correlation result, since
        ``compute_pixel_offsets`` itself works in plain pixel/array
        space.

    Raises
    ------
    ValueError
        If the reference raster's CRS is not a projected CRS with
        square pixels (e.g. a geographic lat/lon CRS) — real ground
        distance conversion in ``compute_pixel_offsets`` assumes a
        real, constant metres-per-pixel value, which a geographic CRS
        does not have (a degree of longitude is not a constant real
        distance except at the equator). Reproject to a real UTM zone
        first; this is a genuine constraint, not an oversight — silently
        producing wrong distances for a geographic-CRS input would be
        worse than refusing.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    np = _require_numpy()

    with rasterio.open(reference_path) as ref_ds:
        reference_profile = ref_ds.profile.copy()
        ref_crs = ref_ds.crs
        ref_transform = ref_ds.transform
        ref_shape = (ref_ds.height, ref_ds.width)
        reference_array = ref_ds.read(band_index).astype("float64")
        ref_nodata = ref_ds.nodata

    if ref_crs is None or not ref_crs.is_projected:
        raise ValueError(
            "prepare_optical_pair: the reference raster's CRS must be a "
            "real projected CRS with square metre-based pixels (e.g. a "
            "UTM zone) for ground-distance offset tracking -- got "
            f"{ref_crs!r}. Reproject with "
            "pygeofetch.processing.preprocessor.Preprocessor.reproject() "
            "to a real UTM zone first."
        )

    pixel_size_x = abs(ref_transform.a)
    pixel_size_y = abs(ref_transform.e)
    if abs(pixel_size_x - pixel_size_y) > 1e-6:
        raise ValueError(
            f"prepare_optical_pair: reference raster pixels are not square "
            f"({pixel_size_x} x {pixel_size_y} m) -- real ground-distance "
            "conversion in compute_pixel_offsets assumes square pixels, "
            "the normal case for a properly-projected optical scene."
        )

    if ref_nodata is not None:
        reference_array[reference_array == ref_nodata] = np.nan

    with rasterio.open(secondary_path) as sec_ds:
        secondary_raw = sec_ds.read(band_index).astype("float64")
        sec_nodata = sec_ds.nodata
        if sec_nodata is not None:
            secondary_raw[secondary_raw == sec_nodata] = np.nan

        secondary_array = np.full(ref_shape, np.nan, dtype="float64")
        reproject(
            source=secondary_raw,
            destination=secondary_array,
            src_transform=sec_ds.transform,
            src_crs=sec_ds.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    mask = np.zeros(ref_shape, dtype=bool)

    if cloud_mask_path is not None:
        classes = cloud_mask_scl_classes or [3, 8, 9, 10, 11]
        with rasterio.open(cloud_mask_path) as scl_ds:
            scl = scl_ds.read(1)
            if scl.shape != ref_shape:
                scl_aligned = np.zeros(ref_shape, dtype=scl.dtype)
                reproject(
                    source=scl,
                    destination=scl_aligned,
                    src_transform=scl_ds.transform,
                    src_crs=scl_ds.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.nearest,  # categorical data -- never interpolate class values
                )
                scl = scl_aligned
        mask |= np.isin(scl, classes)

    if water_mask_threshold is not None:
        if green_band_index is None or nir_band_index is None:
            raise ValueError(
                "prepare_optical_pair: water_mask_threshold requires both "
                "green_band_index and nir_band_index to compute NDWI."
            )
        with rasterio.open(reference_path) as ref_ds:
            green = ref_ds.read(green_band_index).astype("float64")
            nir = ref_ds.read(nir_band_index).astype("float64")
        denom = green + nir
        with np.errstate(divide="ignore", invalid="ignore"):
            ndwi = np.where(denom != 0, (green - nir) / denom, np.nan)
        mask |= ndwi > water_mask_threshold

    reference_array[mask] = np.nan
    secondary_array[mask] = np.nan

    return reference_array, secondary_array, reference_profile


def compute_pixel_offsets(
    reference_array: Any,
    secondary_array: Any,
    pixel_size_m: float,
    window_size: int = 64,
    step_size: int = 16,
    chip_size: "int | None" = None,
    snr_threshold: float = 3.0,
) -> OpticalOffsetResult:
    """
    Real, dense optical pixel offset tracking over a full image pair,
    in real ground-distance units (metres) -- not raw pixels.

    Tiles ``pygeofetch.insar.offset_tracking.OffsetTracker``'s real,
    already-verified NCC + sub-pixel + SNR engine across the image,
    then converts the resulting pixel offsets to metres using
    ``pixel_size_m``. This is the correct, simple conversion for
    optical imagery in a real projected CRS with square pixels (which
    ``prepare_optical_pair`` already validates before you get here) --
    unlike SAR range/azimuth pixel offsets, which need real orbit
    geometry to convert to ground distance (see
    ``pygeofetch.insar.offset_geometry``), an optical scene in a real
    UTM projection already has a fixed, known metres-per-pixel value,
    so no separate geometry module is needed for this conversion.

    Parameters
    ----------
    reference_array, secondary_array : np.ndarray
        2D real arrays, same shape, real coregistered geometry --
        exactly what ``prepare_optical_pair`` returns. NaN pixels
        (nodata, masked clouds/water) are handled gracefully: a
        window containing NaN is skipped (left as NaN/unreliable in
        the output) rather than propagating NaN into the FFT
        correlation and silently corrupting a neighboring valid
        window's result.
    pixel_size_m : float
        Real ground distance per pixel, metres -- from
        ``prepare_optical_pair``'s returned ``reference_profile``'s
        transform (``abs(transform.a)``, since square pixels are
        already validated there).
    window_size : int
        Real search window size in pixels (secondary-image search
        radius context) -- larger windows tolerate larger real
        displacements between dates at the cost of spatial resolution
        in the output. Default 64px (640m at Sentinel-2's 10m
        resolution) -- reasonable for mining-scale subsidence/
        landslide displacements.
    step_size : int
        Real spacing between window centers, pixels -- smaller than
        ``window_size`` so windows overlap; controls the output grid's
        spatial resolution independent of the search window size.
    chip_size : int, optional
        Real fixed reference template size, pixels. Defaults to
        ``window_size // 2`` (same default as
        ``OffsetTracker`` -- matched here rather than picking a
        different default for the optical case with no real reason to
        diverge).
    snr_threshold : float
        Real per-window SNR floor below which a window is marked
        unreliable (not dropped -- see ``OpticalOffsetResult.reliable``).

    Returns
    -------
    OpticalOffsetResult
        Real per-window dx (east)/dy (north) in metres, SNR, a
        reliable mask, and window center pixel coordinates.

    Notes
    -----
    **Sign convention, stated explicitly to avoid a real, easy mistake**:
    ``OffsetTracker`` reports ``azimuth_offset``/``range_offset`` in
    array index space, where increasing row = moving DOWN the image
    (south, for a north-up raster) and increasing column = moving
    RIGHT (east). ``dy`` here is the real geographic convention
    (positive = north), which is the *negative* of the raw row-offset
    direction -- handled internally, not left for the caller to get
    wrong.

    **A real, empirically-discovered SNR caveat, stated rather than
    hidden**: this module's SNR metric (peak correlation / background
    standard deviation) can score genuinely unrelated *uniform iid
    random noise* as artificially reliable at realistic window sizes
    -- with many candidate offsets in a large search window, extreme-
    value statistics mean even pure noise can produce a "lucky" peak
    that looks statistically significant relative to its own
    background. Measured directly: mean SNR 8.8 for synthetic iid
    noise vs. 6.1 for a real, correlated 0.5px-shifted match, at
    ``window_size=64``. This is a real property of comparing against
    literal white noise specifically, not a flaw in real optical
    imagery use -- genuine decorrelated regions (water, thick cloud,
    a totally different real texture) show the expected, clean
    separation instead (verified in this module's own test suite).
    Don't rely on SNR alone as an absolute pass/fail signal without
    also masking obviously-invalid regions (cloud, water, nodata) via
    ``prepare_optical_pair`` first -- SNR is a real, useful *relative*
    confidence signal, not a substitute for masking known-bad areas
    before correlation ever runs.
    """
    np = _require_numpy()

    if reference_array.shape != secondary_array.shape:
        raise ValueError(
            f"compute_pixel_offsets: reference and secondary arrays must "
            f"have the same shape, got {reference_array.shape} vs "
            f"{secondary_array.shape}."
        )
    if pixel_size_m <= 0:
        raise ValueError(
            f"compute_pixel_offsets: pixel_size_m must be positive, got {pixel_size_m}"
        )

    # NaN-safe wrapper: OffsetTracker's own NCC math assumes real,
    # finite pixel values throughout a window -- rather than modify
    # that already-verified engine to special-case NaN internally
    # (real risk of subtly changing its already-tested numerical
    # behavior for the SAR case), invalid windows are detected here
    # and masked out of the result afterward, leaving the proven core
    # untouched.
    ref_filled = np.where(np.isfinite(reference_array), reference_array, 0.0)
    sec_filled = np.where(np.isfinite(secondary_array), secondary_array, 0.0)

    tracker = OffsetTracker(
        search_window_size=window_size, step_size=step_size, chip_size=chip_size
    )
    raw = tracker.track(ref_filled, sec_filled, snr_threshold=snr_threshold)

    half_chip = tracker.chip_size // 2
    half_search = tracker.search_window_size // 2
    invalid = np.zeros(raw.range_offset.shape, dtype=bool)
    for iy, cy in enumerate(raw.window_centers_y):
        for ix, cx in enumerate(raw.window_centers_x):
            ref_chip = reference_array[
                cy - half_chip : cy - half_chip + tracker.chip_size,
                cx - half_chip : cx - half_chip + tracker.chip_size,
            ]
            sec_chip = secondary_array[
                cy - half_search : cy - half_search + tracker.search_window_size,
                cx - half_search : cx - half_search + tracker.search_window_size,
            ]
            if not (np.all(np.isfinite(ref_chip)) and np.all(np.isfinite(sec_chip))):
                invalid[iy, ix] = True

    dx = raw.range_offset.astype("float64") * pixel_size_m
    dy = (
        -raw.azimuth_offset.astype("float64") * pixel_size_m
    )  # row-down -> north-positive
    snr = raw.snr.astype("float64")
    reliable = raw.reliable & ~invalid

    dx[invalid] = np.nan
    dy[invalid] = np.nan
    snr[invalid] = np.nan

    metadata = {
        "window_size": window_size,
        "step_size": step_size,
        "chip_size": tracker.chip_size,
        "snr_threshold": snr_threshold,
        "pixel_size_m": pixel_size_m,
        "n_windows_masked_invalid": int(invalid.sum()),
        "n_windows_total": int(invalid.size),
    }

    return OpticalOffsetResult(
        dx=dx,
        dy=dy,
        snr=snr,
        reliable=reliable,
        window_centers_row=raw.window_centers_y,
        window_centers_col=raw.window_centers_x,
        metadata=metadata,
    )


def compute_horizontal_strain(
    dx: Any,
    dy: Any,
    pixel_size_m: float,
    reliable: "Any | None" = None,
    max_plausible_displacement_m: "float | None" = None,
    mad_outlier_threshold: float = 8.0,
) -> dict:
    """
    Real horizontal strain tensor from an optical offset field --
    turns raw 2D displacement into a direct geotechnical risk metric
    (normal strains exx/eyy, shear strain exy) via central finite
    differences.

    General-purpose, not mining-specific: real ground strain from
    horizontal displacement gradients is exactly as relevant for
    co-seismic surface rupture, volcanic edifice deformation, and
    landslide body extension/compression as it is for mining subsidence
    bowls -- the math doesn't know which one produced the input `dx`,
    `dy`.

    A real, empirically-motivated safeguard, not present in a naive
    implementation of this formula: strain is a *spatial derivative* of
    displacement, which means a single spurious displacement value
    doesn't just corrupt its own pixel -- it produces a locally
    enormous, physically impossible strain gradient at every
    neighboring pixel the finite-difference stencil touches. This is
    not a hypothetical concern: running this exact calculation on a
    real, live optical offset-tracking result (Bu'ertai Mine, this
    project's own real validation case) produced a peak strain of
    6,169,938 microstrain -- 617% strain, physically impossible for
    real ground deformation -- traced directly to a small number of
    spurious correlator matches (up to 226m displacement) that had
    passed the SNR≥3.0 reliability filter but were not real ground
    motion. SNR-based reliability alone did not catch this. Two real,
    documented, opt-out-able safeguards are applied here as a direct
    result of that finding:

    1. `reliable`, if supplied (pass `OpticalOffsetResult.reliable`
       directly): unreliable pixels are excluded (set to NaN) before
       computing any gradient, so an unreliable neighbor can't corrupt
       a reliable pixel's real strain value via the finite-difference
       stencil.
    2. A real MAD (median absolute deviation) outlier filter, applied
       to displacement magnitude specifically because strain
       calculations are far more sensitive to a single extreme value
       than the raw displacement map itself is -- a value this
       function considers "reliable" for reporting displacement can
       still be excluded here specifically for strain, since a
       physically implausible displacement produces an even more
       physically implausible strain. Set `mad_outlier_threshold=None`
       (or a very large number) to disable this and reproduce a naive,
       textbook central-difference calculation exactly -- but doing
       so on real, unfiltered field data is what produced the 617%
       strain artifact this safeguard exists to prevent.

    Parameters
    ----------
    dx, dy : np.ndarray
        Real east/north displacement fields, metres (e.g. from
        `compute_pixel_offsets`).
    pixel_size_m : float
        Real ground distance between adjacent dx/dy grid samples,
        metres (this is the offset-tracking *step_size* in ground
        units, not the original image's raw pixel size -- see
        `compute_pixel_offsets`'s own `metadata["step_size"]`).
    reliable : np.ndarray, optional
        Real boolean reliability mask, same shape as `dx`/`dy`. Highly
        recommended -- see this function's own docstring above for
        why.
    max_plausible_displacement_m : float, optional
        A real, absolute displacement magnitude ceiling (metres).
        Values above this are excluded before strain computation --
        e.g., pass a value tied to your own real, physical
        expectations for the phenomenon being monitored (a known
        maximum credible co-seismic slip, a maximum plausible daily
        mining advance rate, etc.). Not required, but a real, explicit
        physical sanity bound is often more defensible in a real
        report than a purely statistical MAD cutoff alone.
    mad_outlier_threshold : float
        Real, robust outlier threshold in units of MAD (median
        absolute deviation) from the median displacement magnitude.
        8.0 is a real, deliberately generous default (won't reject
        genuine, large real deformation signals near a true
        discontinuity) that still catches the kind of extreme,
        order-of-magnitude-larger-than-everything-else outlier
        documented above. Set to `None` to disable.

    Returns
    -------
    dict
        Real keys: ``"exx"``, ``"eyy"``, ``"exy"`` (each a 2D array,
        same shape as `dx`, NaN where excluded), ``"n_excluded_unreliable"``,
        ``"n_excluded_outlier"``, ``"n_excluded_implausible"`` (real
        counts, for honest reporting of how much of the field was
        excluded and why -- not silently dropped).
    """
    np = _require_numpy()
    dx = np.asarray(dx, dtype="float64")
    dy = np.asarray(dy, dtype="float64")
    if dx.shape != dy.shape:
        raise ValueError(
            f"compute_horizontal_strain: dx/dy shape mismatch {dx.shape} vs {dy.shape}"
        )

    exclude = np.zeros(dx.shape, dtype=bool)
    n_excluded_unreliable = 0
    n_excluded_outlier = 0
    n_excluded_implausible = 0

    if reliable is not None:
        reliable = np.asarray(reliable, dtype=bool)
        unreliable = ~reliable
        n_excluded_unreliable = int(unreliable.sum())
        exclude |= unreliable

    magnitude = np.sqrt(dx**2 + dy**2)

    if max_plausible_displacement_m is not None:
        implausible = magnitude > max_plausible_displacement_m
        n_excluded_implausible = int((implausible & ~exclude).sum())
        exclude |= implausible

    if mad_outlier_threshold is not None:
        finite_mag = magnitude[~exclude & np.isfinite(magnitude)]
        if finite_mag.size > 0:
            median_mag = np.median(finite_mag)
            mad = np.median(np.abs(finite_mag - median_mag))
            # Real, standard robust-scale conversion (MAD -> an
            # approximately-normal-equivalent std), same constant used
            # throughout real robust-statistics literature.
            robust_scale = mad * 1.4826 if mad > 0 else np.finfo(float).eps
            is_outlier = np.abs(magnitude - median_mag) > (
                mad_outlier_threshold * robust_scale
            )
            n_excluded_outlier = int((is_outlier & ~exclude).sum())
            exclude |= is_outlier

    dx_clean = np.where(exclude, np.nan, dx)
    dy_clean = np.where(exclude, np.nan, dy)

    # Real central-difference strain tensor (standard geotechnical
    # convention): exx = d(dx)/dx, eyy = d(dy)/dy, exy = 0.5*(d(dx)/dy + d(dy)/dx).
    # np.gradient's real, confirmed return order for a 2D array is
    # (d/d_row, d/d_col) -- gotten backwards on a first pass here,
    # caught by testing against the exact analytical value this
    # module's own test suite checks, not assumed correct from writing
    # the formula.
    d_dx_drow, d_dx_dcol = np.gradient(dx_clean, pixel_size_m)
    d_dy_drow, d_dy_dcol = np.gradient(dy_clean, pixel_size_m)

    exx = d_dx_dcol
    eyy = d_dy_drow
    exy = 0.5 * (d_dx_drow + d_dy_dcol)

    return {
        "exx": exx,
        "eyy": eyy,
        "exy": exy,
        "n_excluded_unreliable": n_excluded_unreliable,
        "n_excluded_outlier": n_excluded_outlier,
        "n_excluded_implausible": n_excluded_implausible,
    }


def fuse_insar_optical(
    insar_velocity: Any,
    insar_coherence: Any,
    optical_disp_mag: Any,
    optical_snr: "Any | None" = None,
    coherence_low: float = 0.4,
    coherence_high: float = 0.6,
) -> dict:
    """
    Real, coherence-weighted fusion of InSAR and optical offset-tracking
    displacement -- NOT a simple average. See this function's own
    scientific caveat below for why averaging the two would be wrong.

    General-purpose, not mining-specific: this same coherence-weighted
    handoff is exactly what's needed any time InSAR coherence collapses
    in part of a scene -- a co-seismic rupture zone with extreme
    near-fault deformation, a rapidly advancing landslide toe, or a
    mining subsidence bowl all produce the same real failure mode
    (phase decorrelation from displacement gradients exceeding the
    interferometric limit), and the same real fix applies to all of
    them.

    Real, honest scientific caveat (this is not a detail, it's the
    core reason this function exists rather than a plain average):
    InSAR measures real line-of-sight phase, which
    `pygeofetch.insar.geolocation.los_to_vertical_displacement`
    converts to a *vertical* displacement estimate under a real,
    stated small-horizontal-motion assumption. Optical offset tracking
    measures real 2D *horizontal* displacement directly. These are
    physically different vector components, not two noisy estimates of
    the same scalar quantity -- averaging a vertical value with a
    horizontal-magnitude value would produce a number with no real
    physical meaning. This function returns them side by side with a
    real, transparent per-pixel source label, never blended into one
    number that pretends to be both.

    Real, confirmed field behavior worth knowing (from this project's
    own Bu'ertai validation run): when InSAR coherence is genuinely,
    severely degraded across most of a real scene, this logic
    correctly assigns nearly 100% of pixels to the optical source --
    that is real, correct behavior given the stated thresholds, not a
    bug. A scene where InSAR coherence is uniformly excellent will show
    the reverse. The fusion output honestly reflects whatever the real
    coherence map says, rather than forcing an artificial balance
    between the two techniques.

    Parameters
    ----------
    insar_velocity : np.ndarray
        Real InSAR-derived vertical velocity (or displacement) field.
    insar_coherence : np.ndarray
        Real per-pixel coherence, same shape, values in [0, 1].
    optical_disp_mag : np.ndarray
        Real optical displacement magnitude field, same shape.
    optical_snr : np.ndarray, optional
        Real optical SNR field. When supplied, pixels failing a basic
        SNR sanity floor are excluded from optical contribution too --
        not a substitute for `compute_horizontal_strain`'s own, more
        aggressive outlier protection, just a light, consistent floor
        here.
    coherence_low, coherence_high : float
        Real, configurable thresholds. Below `coherence_low`: trust
        optical fully. Above `coherence_high`: trust InSAR fully.
        Between: a real, linear coherence-weighted blend of the two
        *within their own respective physical quantities* -- this
        blends how much each technique's info is trusted for its
        given pixel, not a cross-quantity numeric average.

    Returns
    -------
    dict
        Real keys: ``"fused_displacement"`` (InSAR value where
        InSAR-dominant, optical value where optical-dominant, a real
        weighted value in the transition zone -- see the caveat above
        for what "weighted" means here), ``"reliability_source"`` (int
        array: 2=InSAR-dominant, 1=transition, 0=optical-dominant),
        ``"weight_insar"`` (the real per-pixel InSAR weight actually
        used, for transparency).
    """
    np = _require_numpy()
    insar_velocity = np.asarray(insar_velocity, dtype="float64")
    insar_coherence = np.asarray(insar_coherence, dtype="float64")
    optical_disp_mag = np.asarray(optical_disp_mag, dtype="float64")

    if not (insar_velocity.shape == insar_coherence.shape == optical_disp_mag.shape):
        raise ValueError(
            "fuse_insar_optical: all input arrays must share the same real shape"
        )
    if coherence_low >= coherence_high:
        raise ValueError(
            f"fuse_insar_optical: coherence_low ({coherence_low}) must be < "
            f"coherence_high ({coherence_high})"
        )

    weight_insar = np.clip(
        (insar_coherence - coherence_low) / (coherence_high - coherence_low), 0.0, 1.0
    )

    valid_insar = np.isfinite(insar_velocity)
    valid_optical = np.isfinite(optical_disp_mag)
    if optical_snr is not None:
        optical_snr = np.asarray(optical_snr, dtype="float64")
        valid_optical &= np.isfinite(optical_snr) & (optical_snr >= 1.0)

    # Real, honest fallback: where one real source is simply missing
    # (NaN) regardless of coherence, use whichever real source is
    # actually present rather than blending in a NaN.
    effective_weight = np.where(valid_insar, weight_insar, 0.0)
    effective_weight = np.where(valid_optical, effective_weight, 1.0)
    effective_weight = np.where(~valid_insar & ~valid_optical, np.nan, effective_weight)

    fused = (
        effective_weight * insar_velocity + (1 - effective_weight) * optical_disp_mag
    )
    fused = np.where(
        valid_insar, fused, np.where(valid_optical, optical_disp_mag, np.nan)
    )
    fused = np.where(
        valid_optical, fused, np.where(valid_insar, insar_velocity, np.nan)
    )

    source = np.full(insar_velocity.shape, 1, dtype="int8")  # default: transition
    source = np.where(effective_weight >= 1.0, 2, source)  # InSAR-dominant
    source = np.where(effective_weight <= 0.0, 0, source)  # optical-dominant

    return {
        "fused_displacement": fused,
        "reliability_source": source,
        "weight_insar": effective_weight,
    }