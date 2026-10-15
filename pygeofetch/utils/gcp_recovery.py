"""
GCP (Ground Control Point) recovery utilities for PyGeoFetch.

When rasterio/GDAL writes an identity or pixel-space transform while the file
contains embedded GCPs, this module can recover the correct georeference by
deriving a proper affine transform from those GCPs.

Usage::

    from pygeofetch.utils.gcp_recovery import recover_georeference, needs_recovery

    if needs_recovery("scene.tif"):
        recovered_path = recover_georeference("scene.tif", "scene_fixed.tif")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("pygeofetch.gcp_recovery")


def _is_identity_or_pixel_space(path: Path) -> Tuple[bool, str]:
    """
    Check whether a raster has an identity or pixel-space transform.

    Returns (is_corrupt, reason_string).
    """
    try:
        import rasterio

        with rasterio.open(path) as src:
            t = src.transform
            crs = src.crs

            # Case 1: pixel width = 1.0 and origin at (0,0) in a projected CRS
            if (
                abs(t.a) == 1.0
                and abs(t.e) == 1.0
                and t.c == 0.0
                and crs is not None
                and crs.is_projected
            ):
                return (
                    True,
                    f"identity transform in projected CRS (a={t.a}, origin=({t.c},{t.f}))",
                )

            # Case 2: transform is None or default
            if t.a == 1.0 and t.e == -1.0 and t.c == 0.0 and t.f == 0.0:
                return True, f"default pixel-space transform (a={t.a}, e={t.e})"

            # Case 3: pixel size implausibly large for the stated CRS
            if crs and crs.is_geographic:
                # Geographic CRS: pixel size should be < 1 degree
                if abs(t.a) > 1.0 or abs(t.e) > 1.0:
                    return True, f"pixel size > 1 degree in geographic CRS ({t.a:.4f})"

            return False, ""
    except Exception as exc:
        return False, f"could not check: {exc}"


def has_embedded_gcps(path: Path) -> bool:
    """Return True if the file has embedded GCPs that can be used for recovery."""
    try:
        import rasterio

        with rasterio.open(path) as src:
            gcps, _ = src.gcps
            return len(gcps) >= 4  # need at least 4 GCPs for a proper transform
    except Exception:
        return False


def needs_recovery(path: str | Path) -> bool:
    """
    Return True if this raster has a corrupt georeference AND embedded GCPs
    that can be used to recover it.
    """
    p = Path(path)
    if not p.exists():
        return False
    corrupt, _ = _is_identity_or_pixel_space(p)
    return corrupt and has_embedded_gcps(p)


def recover_georeference(
    source_path: str | Path,
    output_path: Optional[str | Path] = None,
    overwrite: bool = False,
) -> Optional[Path]:
    """
    Recover correct georeferencing from embedded GCPs.

    When reprojection produces an identity transform but GCPs are present,
    this function:
    1. Reads the embedded GCPs and their CRS
    2. Derives a proper affine transform using ``rasterio.transform.from_gcps()``
    3. Writes a corrected copy with the proper transform and CRS
    4. Validates the output

    Args:
        source_path:  Path to the raster with corrupt georeference.
        output_path:  Where to write the corrected raster.
                      Defaults to ``<source>_georef_recovered.tif``.
        overwrite:    Overwrite output_path if it exists.

    Returns:
        Path to recovered file, or None if recovery failed or was not needed.

    Example::

        recovered = recover_georeference("corrupt_scene.tif")
        if recovered:
            print(f"Recovered: {recovered}")
        else:
            print("Recovery not possible — no GCPs or already georeferenced")
    """
    try:
        import rasterio
        from rasterio.transform import from_gcps

    except ImportError as exc:
        logger.error("rasterio/numpy required for GCP recovery: %s", exc)
        return None

    src_path = Path(source_path)
    if not src_path.exists():
        logger.error("Source file not found: %s", src_path)
        return None

    # Check if recovery is needed
    corrupt, reason = _is_identity_or_pixel_space(src_path)
    if not corrupt:
        logger.debug(
            "%s does not need georef recovery (transform looks valid)", src_path.name
        )
        return None

    logger.warning(
        "Corrupt georeference detected in %s: %s — attempting GCP recovery",
        src_path.name,
        reason,
    )

    # Resolve output path
    if output_path is None:
        out = src_path.parent / f"{src_path.stem}_georef_recovered.tif"
    else:
        out = Path(output_path)

    if out.exists() and not overwrite:
        logger.info("Recovery output already exists: %s", out)
        return out

    out.parent.mkdir(parents=True, exist_ok=True)

    # Attempt GCP-based recovery
    try:
        with rasterio.open(src_path) as src:
            gcps, gcp_crs = src.gcps

            if not gcps or len(gcps) < 4:
                logger.warning(
                    "%s has no embedded GCPs (found %d) — recovery not possible",
                    src_path.name,
                    len(gcps),
                )
                return None

            logger.info(
                "Recovering georeference using %d GCPs (CRS: %s)", len(gcps), gcp_crs
            )

            # Derive affine transform from GCPs
            recovered_transform = from_gcps(gcps)
            recovered_crs = gcp_crs or src.crs

            # Build clean output profile
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                crs=recovered_crs,
                transform=recovered_transform,
                compress="deflate",
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )

            # Write corrected file
            with rasterio.open(out, "w", **profile) as dst:
                dst.write(src.read())

        # Validate the recovered file
        still_corrupt, reason2 = _is_identity_or_pixel_space(out)
        if still_corrupt:
            out.unlink(missing_ok=True)
            logger.error(
                "GCP recovery produced another bad transform for %s: %s",
                src_path.name,
                reason2,
            )
            return None

        logger.info(
            "Georeference recovery successful: %s → %s", src_path.name, out.name
        )
        return out

    except Exception as exc:
        logger.error(
            "GCP recovery failed for %s: %s", src_path.name, exc, exc_info=True
        )
        if out.exists():
            out.unlink(missing_ok=True)
        return None


def validate_georeference(path: str | Path) -> dict:
    """
    Validate the georeference of a raster and return a detailed report.

    Returns a dict with:
        valid (bool): Whether the georeference is valid
        crs (str):    CRS string
        transform:    Affine transform
        pixel_size:   (x_res, y_res) in CRS units
        origin:       (x, y) origin coordinates
        bounds:       (left, bottom, right, top)
        has_gcps:     Whether embedded GCPs are present
        can_recover:  Whether GCP recovery is possible
        issues:       List of detected issues
        recommendation: What to do

    Example::

        report = validate_georeference("scene.tif")
        if not report["valid"]:
            print(report["recommendation"])
    """
    try:
        import rasterio
    except ImportError:
        return {"valid": False, "issues": ["rasterio not installed"]}

    p = Path(path)
    if not p.exists():
        return {"valid": False, "issues": [f"File not found: {p}"]}

    issues = []

    try:
        with rasterio.open(p) as src:
            t = src.transform
            crs = src.crs
            gcps, gcp_crs = src.gcps

            report = {
                "valid": True,
                "crs": str(crs) if crs else None,
                "crs_type": (
                    "geographic"
                    if (crs and crs.is_geographic)
                    else "projected" if (crs and crs.is_projected) else "unknown"
                ),
                "transform": t,
                "pixel_size": (abs(t.a), abs(t.e)),
                "origin": (t.c, t.f),
                "bounds": src.bounds,
                "width": src.width,
                "height": src.height,
                "has_gcps": len(gcps) >= 4,
                "gcp_count": len(gcps),
                "can_recover": False,
                "issues": [],
                "recommendation": "File georeference appears valid.",
            }

            # Check 1: No CRS
            if crs is None:
                issues.append("No CRS defined")
                report["valid"] = False

            # Check 2: Identity transform in projected CRS
            if crs and crs.is_projected and abs(t.a) == 1.0 and t.c == 0.0:
                issues.append(
                    f"Identity/pixel-space transform in projected CRS "
                    f"(pixel_size={abs(t.a)}, origin=({t.c:.1f},{t.f:.1f}))"
                )
                report["valid"] = False

            # Check 3: Pixel size implausibly large
            if crs and crs.is_geographic and abs(t.a) > 1.0:
                issues.append(
                    f"Pixel size > 1 degree in geographic CRS ({abs(t.a):.6f}°)"
                )
                report["valid"] = False

            # Check 4: Origin at exact (0, 0) in projected CRS
            if crs and crs.is_projected and t.c == 0.0 and t.f == 0.0:
                issues.append("Origin exactly at (0, 0) in projected CRS — suspicious")

            # Check 5: Bounds sanity
            if crs and crs.is_geographic:
                b = src.bounds
                if not (-180 <= b.left <= 180 and -90 <= b.bottom <= 90):
                    issues.append(f"Bounds outside valid geographic range: {b}")
                    report["valid"] = False

            # Recovery potential
            if not report["valid"] and len(gcps) >= 4:
                report["can_recover"] = True
                issues.append(
                    f"{len(gcps)} embedded GCPs available — "
                    "call recover_georeference() to fix"
                )
                report["recommendation"] = (
                    f"Georeference is corrupt but {len(gcps)} GCPs are available. "
                    "Run: recover_georeference(path) to restore correct georeferencing."
                )
            elif not report["valid"]:
                report["recommendation"] = (
                    "Georeference is corrupt and no GCPs are available for recovery. "
                    "Re-download the original file."
                )

            report["issues"] = issues
            return report

    except Exception as exc:
        return {
            "valid": False,
            "issues": [f"Could not open file: {exc}"],
            "recommendation": "File may be corrupt or not a valid raster.",
        }
