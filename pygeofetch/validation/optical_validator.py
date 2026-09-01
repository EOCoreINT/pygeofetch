"""
Pre-download data-quality gate for optical imagery (Sentinel-2,
Landsat, and any other provider's SatelliteData results).

Runs AFTER search, BEFORE download -- the same position in the
pipeline as pygeofetch.insar.preflight, and built to the same
philosophy: catch real, confirmed acquisition-planning mistakes before
bandwidth is spent, distinguish HARD failures (reject the scene) from
WARNINGS (log and proceed), and make every check independently
toggleable so a user who doesn't need a given check pays nothing for
it.

Typical use:

    from pygeofetch import PyGeoFetch
    from pygeofetch.validation import OpticalPreflightValidator, OpticalValidationConfig

    pf = PyGeoFetch()
    results = pf.search(query, providers=["copernicus"])

    validator = OpticalPreflightValidator(OpticalValidationConfig())
    safe_results = validator.run_preflight(results, aoi=my_aoi_polygon)

    downloads = pf.download(safe_results, destination="./data")

Accepts real pygeofetch `SatelliteData` objects (the normal shape
`PyGeoFetch.search()` already returns) as well as plain STAC-like
dicts, so it works equally well against live search results or
against test fixtures / other catalog clients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Union

from pydantic import BaseModel, Field
from shapely.geometry import Polygon, shape

from pygeofetch.core.logging import get_logger
from pygeofetch.models.satellite_data import ProcessingLevel, SatelliteData

logger = get_logger(__name__)

# A scene can be a real SatelliteData instance or a plain
# STAC-like/dict record -- both are accepted throughout this module.
SceneLike = Union[SatelliteData, dict]


# ══════════════════════════════════════════════════════════════════════════
#  Exceptions
# ══════════════════════════════════════════════════════════════════════════


class OpticalValidationError(ValueError):
    """
    Raised for a HARD validation failure -- a scene that cannot be
    safely downloaded/used regardless of configuration (e.g. a
    required band is genuinely absent, or AOI coverage is 0%).

    A ValueError subclass, so existing ``except ValueError`` call
    sites keep working; carries structured attributes (``scene_id``,
    ``code``, ``reason``) so callers can build actionable messages
    without re-parsing the exception text.

    Parameters
    ----------
    scene_id : str
        Identifier of the scene that failed validation.
    code : str
        Machine-readable failure code, e.g. ``"MISSING_BANDS"``.
    reason : str
        Human-readable explanation of the failure.

    Examples
    --------
    >>> try:
    ...     raise OpticalValidationError("S2A_...", "MISSING_BANDS", "missing B08")
    ... except OpticalValidationError as exc:
    ...     print(f"Scene {exc.scene_id} rejected: {exc.reason}")
    Scene S2A_... rejected: missing B08
    """

    def __init__(self, scene_id: str, code: str, reason: str) -> None:
        self.scene_id = scene_id
        self.code = code
        self.reason = reason
        super().__init__(f"Scene {scene_id} rejected [{code}]: {reason}")


# ══════════════════════════════════════════════════════════════════════════
#  Report data model (mirrors pygeofetch.insar.preflight.PreflightIssue)
# ══════════════════════════════════════════════════════════════════════════

SEVERITY_ERROR = "ERROR"  # hard failure -- scene excluded
SEVERITY_WARNING = "WARNING"  # logged, scene kept unless config says otherwise


@dataclass
class ValidationIssue:
    """
    A single problem found while validating one scene.

    Attributes
    ----------
    code : str
        Machine-readable identifier, e.g. ``"CLOUD_COVER_EXCEEDED"``.
    severity : str
        One of ``SEVERITY_ERROR`` or ``SEVERITY_WARNING``.
    message : str
        Human-readable explanation.
    """

    code: str
    severity: str
    message: str


@dataclass
class SceneValidationReport:
    """
    The outcome of running all enabled checks against one scene.

    Attributes
    ----------
    scene_id : str
        Identifier of the validated scene.
    passed : bool
        True if the scene has no ERROR-severity issues (and, when
        ``OpticalValidationConfig.treat_warnings_as_errors`` is set,
        no WARNING-severity issues either).
    issues : list[ValidationIssue]
        Every issue found, both errors and warnings.
    metrics : dict[str, float]
        Raw computed values (e.g. ``{"aoi_coverage": 0.42,
        "cloud_cover_pct": 12.5}``) -- useful for ranking/sorting scenes
        that all pass, not just for pass/fail.
    """

    scene_id: str
    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == SEVERITY_ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == SEVERITY_WARNING]


# ══════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════


class OpticalValidationConfig(BaseModel):
    """
    Toggles and thresholds for every optical preflight check.

    The most consequential checks default on; checks that are either
    computationally heavier or more often a matter of taste than
    correctness default off. Every ``check_*`` flag can be flipped
    independently -- a disabled check costs nothing (its
    ``validate_*`` method still exists and can be called directly, it
    is simply skipped by ``run_preflight``).

    Attributes
    ----------
    check_aoi_coverage : bool
        Reject scenes whose footprint doesn't sufficiently overlap the
        AOI. Default ``True`` -- this is usually the single most
        common cause of "why did I download the wrong tile".
    min_coverage_ratio : float
        Minimum fraction of the AOI's area that must be covered by the
        scene footprint, in ``[0, 1]``. Default ``0.8``.
    check_cloud_cover : bool
        Flag scenes whose cloud cover exceeds ``max_cloud_cover_pct``.
        Default ``True``.
    max_cloud_cover_pct : float
        Maximum acceptable cloud cover, in percent. Default ``20.0``.
    cloud_cover_is_hard_failure : bool
        If ``True``, cloud cover above the threshold is a HARD failure
        (scene excluded) rather than a warning. Default ``False`` --
        a cloudy scene is still often usable (e.g. for a mosaic), so
        this defaults to warn-and-keep, matching the "Graceful
        Degradation" principle.
    check_snow_ice_cover : bool
        Flag scenes whose snow/ice cover exceeds
        ``max_snow_ice_pct``. Default ``False`` -- irrelevant for most
        AOIs and most providers don't report it consistently.
    max_snow_ice_pct : float
        Maximum acceptable snow/ice cover, in percent. Default
        ``10.0``.
    check_required_bands : bool
        Reject scenes missing any of ``required_bands``. Default
        ``True`` -- a scene missing a required band is unusable, not
        just suboptimal.
    required_bands : list[str]
        Band/asset keys that must be present. Matched case-insensitively
        against the scene's available asset keys. Default: the common
        Sentinel-2 L2A RGB+NIR+SCL set.
    check_processing_level : bool
        Reject scenes not at ``expected_level``. Default ``True``.
    expected_level : str
        Expected processing level, matched loosely against common
        aliases (``"Level-2A"``, ``"L2A"``, ``"S2MSI2A"`` all match a
        pygeofetch ``ProcessingLevel.L2A`` scene). Default ``"Level-2A"``.
    check_nodata_margins : bool
        Flag scenes where the AOI sits mostly in the scene's edge
        margin (a common cause of a mostly-black/no-data clip result).
        Default ``False`` -- a real but heuristic, geometry-only check
        (no raster access at preflight time); off by default to avoid
        false positives on legitimately edge-adjacent AOIs.
    nodata_margin_buffer_deg : float
        How far to shrink the scene footprint inward, in degrees,
        before checking AOI overlap against that shrunk footprint.
        Default ``0.01`` (roughly ~1 km at the equator).
    nodata_margin_min_aoi_fraction : float
        Minimum fraction of the AOI's area that must fall within the
        shrunk footprint to avoid a nodata-margin warning. Default
        ``0.5`` (matching ">50% of AOI" in the original spec).
    check_temporal_bounds : bool
        Reject scenes whose acquisition date falls outside
        ``[start_date, end_date]`` as passed to ``run_preflight``.
        Default ``True``.
    treat_warnings_as_errors : bool
        If ``True``, any WARNING-severity issue also excludes the
        scene from ``run_preflight``'s returned list (still logged and
        still present in the full per-scene report). Default
        ``False``.

    Examples
    --------
    >>> cfg = OpticalValidationConfig(max_cloud_cover_pct=10.0, check_snow_ice_cover=True)
    >>> cfg.check_aoi_coverage
    True
    """

    model_config = {"extra": "forbid"}

    check_aoi_coverage: bool = True
    min_coverage_ratio: float = Field(0.8, ge=0.0, le=1.0)

    check_cloud_cover: bool = True
    max_cloud_cover_pct: float = Field(20.0, ge=0.0, le=100.0)
    cloud_cover_is_hard_failure: bool = False

    check_snow_ice_cover: bool = False
    max_snow_ice_pct: float = Field(10.0, ge=0.0, le=100.0)

    check_required_bands: bool = True
    required_bands: list[str] = Field(
        default_factory=lambda: ["B02", "B03", "B04", "B08", "SCL"]
    )

    check_processing_level: bool = True
    expected_level: str = "Level-2A"

    check_nodata_margins: bool = False
    nodata_margin_buffer_deg: float = Field(0.01, ge=0.0)
    nodata_margin_min_aoi_fraction: float = Field(0.5, ge=0.0, le=1.0)

    check_temporal_bounds: bool = True

    treat_warnings_as_errors: bool = False


# Loose aliasing so "Level-2A", "L2A", "S2MSI2A", and pygeofetch's own
# ProcessingLevel.L2A all compare equal -- real providers are wildly
# inconsistent about how they spell this.
_LEVEL_ALIASES: dict[str, set[str]] = {
    "L2A": {"L2A", "LEVEL-2A", "LEVEL2A", "S2MSI2A", "L2SP"},
    "L1C": {"L1C", "LEVEL-1C", "LEVEL1C", "S2MSI1C"},
    "L1T": {"L1T", "LEVEL-1T", "LEVEL1T"},
    "L2": {"L2", "LEVEL-2", "LEVEL2"},
    "L1": {"L1", "LEVEL-1", "LEVEL1"},
}


def _normalize_level(value: str) -> str:
    return value.strip().upper().replace(" ", "").replace("_", "-")


def _levels_match(actual: str, expected: str) -> bool:
    a, e = _normalize_level(actual), _normalize_level(expected)
    if a == e:
        return True
    for canonical, aliases in _LEVEL_ALIASES.items():
        if a in aliases and e in aliases:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════
#  Scene metadata extraction (SatelliteData or plain dict -> common shape)
# ══════════════════════════════════════════════════════════════════════════


def _scene_id(scene: SceneLike) -> str:
    if isinstance(scene, SatelliteData):
        return scene.id
    return str(
        scene.get("id")
        or scene.get("scene_id")
        or scene.get("identifier")
        or "<unknown>"
    )


def _scene_footprint(scene: SceneLike) -> Polygon | None:
    """Real footprint geometry when available, falling back to a bbox
    rectangle -- the same fallback pattern used throughout the rest of
    pygeofetch's provider layer for bbox-only providers."""
    geometry = (
        scene.geometry if isinstance(scene, SatelliteData) else scene.get("geometry")
    )
    if isinstance(geometry, dict) and geometry.get("coordinates"):
        try:
            geom = shape(geometry)
            return geom if isinstance(geom, Polygon) else geom.convex_hull
        except Exception:
            pass
    bbox = scene.bbox if isinstance(scene, SatelliteData) else scene.get("bbox")
    if bbox and len(bbox) == 4:
        from shapely.geometry import box

        return box(*bbox)
    return None


def _scene_cloud_cover(scene: SceneLike) -> float | None:
    if isinstance(scene, SatelliteData):
        if scene.cloud_cover is not None:
            return scene.cloud_cover
        props = scene.properties or {}
    else:
        if scene.get("cloud_cover") is not None:
            return float(scene["cloud_cover"])
        props = scene.get("properties") or scene
    for key in (
        "eo:cloud_cover",
        "cloudCover",
        "cloud_cover",
        "cloud_cover_percentage",
    ):
        if key in props and props[key] is not None:
            return float(props[key])
    return None


def _scene_snow_ice_cover(scene: SceneLike) -> float | None:
    props = (
        scene.properties
        if isinstance(scene, SatelliteData)
        else scene.get("properties")
    ) or {}
    for key in (
        "s2:snow_ice_percentage",
        "snowIceCover",
        "snow_ice_cover",
        "snow_ice_percentage",
    ):
        if key in props and props[key] is not None:
            return float(props[key])
    return None


def _scene_available_assets(scene: SceneLike) -> list[str]:
    if isinstance(scene, SatelliteData):
        return list(scene.assets.keys())
    assets = scene.get("assets") or {}
    return list(assets.keys())


# def _scene_processing_level(scene: SceneLike) -> str | None:
#     if isinstance(scene, SatelliteData):
#         level = scene.processing_level
#         if level and level != ProcessingLevel.UNKNOWN:
#             return level.value
#         props = scene.properties or {}
#     else:
#         if scene.get("processing_level"):
#             return str(scene["processing_level"])
#         props = scene.get("properties") or scene
#     for key in ("processing:level", "processingLevel", "processing_level", "s2:processing_baseline"):
#         if key in props and props[key]:
#             return str(props[key])
#     return None


def _scene_processing_level(scene: SceneLike) -> str | None:
    if isinstance(scene, SatelliteData):
        level = scene.processing_level
        if level and level != ProcessingLevel.UNKNOWN:
            return level.value
        props = scene.properties or {}
    else:
        if scene.get("processing_level"):
            return str(scene["processing_level"])
        props = scene.get("properties") or scene

    # FIX: Removed 's2:processing_baseline' (which is a version like '05.10').
    # Added 's2:product_type' (which holds 'S2MSI2A') and standard STAC keys.
    for key in (
        "processing:level",
        "processingLevel",
        "processing_level",
        "s2:product_type",
        "s2:product_uri",
    ):
        if key in props and props[key]:
            return str(props[key])
    return None


def _scene_datetime(scene: SceneLike) -> datetime | None:
    if isinstance(scene, SatelliteData):
        return scene.datetime
    raw = (
        scene.get("datetime")
        or scene.get("acquisitionDate")
        or scene.get("acquisition_date")
    )
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


# ══════════════════════════════════════════════════════════════════════════
#  Validator
# ══════════════════════════════════════════════════════════════════════════


class OpticalPreflightValidator:
    """
    Runs configured pre-download checks against optical scene
    metadata, distinguishing hard failures from warnings.

    Parameters
    ----------
    config : OpticalValidationConfig
        Which checks to run and their thresholds.

    Examples
    --------
    >>> validator = OpticalPreflightValidator(OpticalValidationConfig())
    >>> from shapely.geometry import box
    >>> aoi = box(-74.1, 40.6, -73.7, 40.9)
    >>> scene = {
    ...     "id": "S2A_TEST",
    ...     "bbox": [-74.2, 40.5, -73.6, 41.0],
    ...     "cloud_cover": 5.0,
    ...     "assets": {"B02": {}, "B03": {}, "B04": {}, "B08": {}, "SCL": {}},
    ...     "processing_level": "Level-2A",
    ... }
    >>> report = validator.validate_scene(scene, aoi)
    >>> report.passed
    True
    """

    def __init__(self, config: OpticalValidationConfig | None = None) -> None:
        self.config = config or OpticalValidationConfig()

    # ── Individual checks ───────────────────────────────────────────────

    def validate_aoi_coverage(self, scene_footprint: Polygon, aoi: Polygon) -> float:
        """
        Fraction of the AOI's area covered by the scene footprint.

        Parameters
        ----------
        scene_footprint : shapely.geometry.Polygon
            The scene's real footprint (or bbox rectangle fallback).
        aoi : shapely.geometry.Polygon
            The user's area of interest.

        Returns
        -------
        float
            ``intersection(scene_footprint, aoi).area / aoi.area``, in
            ``[0, 1]``. Returns ``0.0`` if the AOI has zero area or the
            geometries don't intersect at all.

        Notes
        -----
        This is coverage of the AOI by the scene, not a symmetric IoU
        (Intersection over Union) -- IoU would be misleadingly low for
        a small AOI inside a much larger scene footprint, which is the
        normal, desired case here, not a defect.
        """
        if aoi.area <= 0:
            return 0.0
        if not scene_footprint.is_valid:
            scene_footprint = scene_footprint.buffer(0)
        if not aoi.is_valid:
            aoi = aoi.buffer(0)
        if not scene_footprint.intersects(aoi):
            return 0.0
        return scene_footprint.intersection(aoi).area / aoi.area

    def validate_cloud_cover(self, metadata: SceneLike) -> float:
        """
        Extract cloud cover percentage from scene metadata.

        Parameters
        ----------
        metadata : dict
            Scene properties/metadata dict (or a full scene dict --
            common cloud-cover keys are checked at both the top level
            and inside a nested ``"properties"`` dict).

        Returns
        -------
        float
            Cloud cover percentage in ``[0, 100]``. Returns ``0.0`` if
            no recognised cloud-cover field is present (treated as
            "unknown, assume clear" rather than raising, since many
            providers simply omit this for genuinely clear scenes).
        """
        value = _scene_cloud_cover(metadata)
        return value if value is not None else 0.0

    def validate_snow_ice_cover(self, metadata: SceneLike) -> float:
        """
        Extract snow/ice cover percentage from scene metadata.

        Parameters
        ----------
        metadata : dict
            Scene properties/metadata dict.

        Returns
        -------
        float
            Snow/ice cover percentage in ``[0, 100]``, or ``0.0`` if
            not reported.
        """
        value = _scene_snow_ice_cover(metadata)
        return value if value is not None else 0.0

    def validate_bands(self, available_assets: list[str]) -> list[str]:
        """
        Check that every required band is present.

        Parameters
        ----------
        available_assets : list[str]
            Asset/band keys actually present on the scene.

        Returns
        -------
        list[str]
            The subset of ``required_bands`` that ARE present (for
            informational use).

        Raises
        ------
        OpticalValidationError
            If any required band is missing. ``OpticalValidationError``
            is a ``ValueError`` subclass, so ``except ValueError`` at
            call sites keeps working unchanged.
        """
        available_upper = {a.upper() for a in available_assets}
        required = self.config.required_bands
        present = [b for b in required if b.upper() in available_upper]
        missing = [b for b in required if b.upper() not in available_upper]
        if missing:
            raise OpticalValidationError(
                "<pending>",
                "MISSING_BANDS",
                f"missing required band(s): {', '.join(missing)}",
            )
        return present

    def validate_processing_level(self, metadata: SceneLike) -> bool:
        """
        Check the scene's processing level against
        ``config.expected_level``, using loose alias matching (e.g.
        ``"L2A"`` matches ``"Level-2A"``).

        Parameters
        ----------
        metadata : dict
            Scene properties/metadata dict, or a full scene dict.

        Returns
        -------
        bool
            ``True`` if the level matches (or is unknown/unreported --
            treated as "can't disprove it", not a failure); ``False``
            if a level was reported and it doesn't match.
        """
        level = _scene_processing_level(metadata)
        if level is None:
            return True
        return _levels_match(level, self.config.expected_level)

    def validate_nodata_margins(self, scene_footprint: Polygon, aoi: Polygon) -> bool:
        """
        Heuristic check for whether the AOI sits mostly in the scene's
        edge margin (a common cause of a mostly-black/no-data clip
        result) -- without needing raster access at preflight time.

        Shrinks ``scene_footprint`` inward by
        ``config.nodata_margin_buffer_deg`` and checks what fraction
        of the AOI's area still falls within that shrunk footprint.

        Parameters
        ----------
        scene_footprint : shapely.geometry.Polygon
            The scene's real footprint.
        aoi : shapely.geometry.Polygon
            The user's area of interest.

        Returns
        -------
        bool
            ``True`` if at least
            ``config.nodata_margin_min_aoi_fraction`` of the AOI's
            area falls within the margin-shrunk footprint (safe);
            ``False`` if the AOI is mostly sitting in the scene's edge
            margin.

        Notes
        -----
        This is a real but inherently approximate, geometry-only
        heuristic -- it cannot see actual per-pixel no-data masks
        (those don't exist until the file is downloaded). It catches
        the common "AOI clips the corner of the swath" case, not
        every possible no-data pattern (e.g. cloud-masked interior
        gaps aren't a "margin" issue and aren't this check's job).
        """
        if aoi.area <= 0:
            return True
        shrunk = scene_footprint.buffer(-self.config.nodata_margin_buffer_deg)
        if shrunk.is_empty:
            return False
        if not shrunk.intersects(aoi):
            return False
        covered_fraction = shrunk.intersection(aoi).area / aoi.area
        return covered_fraction >= self.config.nodata_margin_min_aoi_fraction

    def validate_temporal_bounds(
        self,
        scene_datetime: datetime | None,
        start_date: date | datetime | None,
        end_date: date | datetime | None,
    ) -> bool:
        """
        Check the scene's acquisition date falls within
        ``[start_date, end_date]``.

        Parameters
        ----------
        scene_datetime : datetime or None
            The scene's acquisition datetime.
        start_date, end_date : date, datetime, or None
            Inclusive bounds. Either may be ``None`` to leave that side
            unbounded.

        Returns
        -------
        bool
            ``True`` if within bounds, or if ``scene_datetime`` is
            unknown (a missing date can't be disproven as out of
            range, so it isn't treated as a failure here); ``False``
            otherwise.
        """
        if scene_datetime is None:
            return True
        scene_date = (
            scene_datetime.date()
            if isinstance(scene_datetime, datetime)
            else scene_datetime
        )
        if start_date is not None:
            start = (
                start_date.date() if isinstance(start_date, datetime) else start_date
            )
            if scene_date < start:
                return False
        if end_date is not None:
            end = end_date.date() if isinstance(end_date, datetime) else end_date
            if scene_date > end:
                return False
        return True

    # ── Orchestration ───────────────────────────────────────────────────

    def validate_scene(
        self,
        scene: SceneLike,
        aoi: Polygon | None,
        start_date: date | datetime | None = None,
        end_date: date | datetime | None = None,
    ) -> SceneValidationReport:
        """
        Run every enabled check against one scene.

        Parameters
        ----------
        scene : SatelliteData or dict
            The candidate scene.
        aoi : shapely.geometry.Polygon or None
            The user's area of interest. May be ``None`` when no AOI
            is available at the call site (e.g. validating already-
            downloaded-metadata scenes right before download, where
            the original search AOI wasn't threaded through) -- in
            that case, AOI-dependent checks (``check_aoi_coverage``,
            ``check_nodata_margins``) are skipped with a debug log
            rather than raising, even if enabled in config. Every
            other check still runs normally.
        start_date, end_date : date, datetime, or None
            Temporal bounds, only used if
            ``config.check_temporal_bounds`` is enabled.

        Returns
        -------
        SceneValidationReport
            Every issue found (errors and warnings) plus raw metrics,
            regardless of whether the scene ultimately passes.
        """
        cfg = self.config
        scene_id = _scene_id(scene)
        issues: list[ValidationIssue] = []
        metrics: dict[str, float] = {}

        if cfg.check_aoi_coverage and aoi is None:
            logger.debug(
                f"Scene {scene_id}: check_aoi_coverage is enabled but no AOI "
                "was provided to validate_scene() -- skipping this check."
            )
        if cfg.check_aoi_coverage and aoi is not None:
            footprint = _scene_footprint(scene)
            if footprint is None:
                issues.append(
                    ValidationIssue(
                        "NO_FOOTPRINT",
                        SEVERITY_ERROR,
                        "scene has neither geometry nor bbox -- cannot verify AOI coverage",
                    )
                )
            else:
                coverage = self.validate_aoi_coverage(footprint, aoi)
                metrics["aoi_coverage"] = coverage
                if coverage < cfg.min_coverage_ratio:
                    issues.append(
                        ValidationIssue(
                            "LOW_AOI_COVERAGE",
                            SEVERITY_ERROR,
                            f"AOI coverage {coverage:.1%} is below the required "
                            f"{cfg.min_coverage_ratio:.1%}",
                        )
                    )

        if cfg.check_cloud_cover:
            cloud_cover = self.validate_cloud_cover(scene)
            metrics["cloud_cover_pct"] = cloud_cover
            if cloud_cover > cfg.max_cloud_cover_pct:
                severity = (
                    SEVERITY_ERROR
                    if cfg.cloud_cover_is_hard_failure
                    else SEVERITY_WARNING
                )
                issues.append(
                    ValidationIssue(
                        "CLOUD_COVER_EXCEEDED",
                        severity,
                        f"cloud cover {cloud_cover:.1f}% exceeds threshold "
                        f"{cfg.max_cloud_cover_pct:.1f}%",
                    )
                )

        if cfg.check_snow_ice_cover:
            snow_ice = self.validate_snow_ice_cover(scene)
            metrics["snow_ice_cover_pct"] = snow_ice
            if snow_ice > cfg.max_snow_ice_pct:
                issues.append(
                    ValidationIssue(
                        "SNOW_ICE_COVER_EXCEEDED",
                        SEVERITY_WARNING,
                        f"snow/ice cover {snow_ice:.1f}% exceeds threshold "
                        f"{cfg.max_snow_ice_pct:.1f}%",
                    )
                )

        if cfg.check_required_bands:
            available = _scene_available_assets(scene)
            try:
                self.validate_bands(available)
            except OpticalValidationError as exc:
                issues.append(
                    ValidationIssue("MISSING_BANDS", SEVERITY_ERROR, exc.reason)
                )

        if cfg.check_processing_level:
            if not self.validate_processing_level(scene):
                actual = _scene_processing_level(scene) or "unknown"
                issues.append(
                    ValidationIssue(
                        "PROCESSING_LEVEL_MISMATCH",
                        SEVERITY_ERROR,
                        f"processing level {actual!r} does not match expected "
                        f"{cfg.expected_level!r}",
                    )
                )

        if cfg.check_nodata_margins and aoi is None:
            logger.debug(
                f"Scene {scene_id}: check_nodata_margins is enabled but no AOI "
                "was provided to validate_scene() -- skipping this check."
            )
        if cfg.check_nodata_margins and aoi is not None:
            footprint = _scene_footprint(scene)
            if footprint is not None and not self.validate_nodata_margins(
                footprint, aoi
            ):
                issues.append(
                    ValidationIssue(
                        "NODATA_MARGIN_RISK",
                        SEVERITY_WARNING,
                        "AOI sits mostly in the scene's edge margin -- the "
                        "clipped result may be mostly no-data",
                    )
                )

        if cfg.check_temporal_bounds:
            scene_dt = _scene_datetime(scene)
            if not self.validate_temporal_bounds(scene_dt, start_date, end_date):
                issues.append(
                    ValidationIssue(
                        "OUT_OF_TEMPORAL_BOUNDS",
                        SEVERITY_ERROR,
                        f"acquisition date {scene_dt} is outside "
                        f"[{start_date}, {end_date}]",
                    )
                )

        has_errors = any(i.severity == SEVERITY_ERROR for i in issues)
        has_warnings = any(i.severity == SEVERITY_WARNING for i in issues)
        passed = not has_errors and not (cfg.treat_warnings_as_errors and has_warnings)

        return SceneValidationReport(
            scene_id=scene_id, passed=passed, issues=issues, metrics=metrics
        )

    def run_preflight(
        self,
        catalog_results: list[SceneLike],
        aoi: Polygon | None,
        start_date: date | datetime | None = None,
        end_date: date | datetime | None = None,
    ) -> list[SceneLike]:
        """
        Validate every candidate scene and return only the safe ones.

        Iterates ``catalog_results``, applies every enabled check via
        :meth:`validate_scene`, logs a warning for every non-fatal
        issue found, logs an error and drops the scene for every hard
        failure, and returns the scenes that passed -- in their
        original form (``SatelliteData`` in, ``SatelliteData`` out; a
        plain dict in, the same dict out), so this slots directly
        between ``PyGeoFetch.search()`` and ``PyGeoFetch.download()``
        with no conversion needed on either side.

        Parameters
        ----------
        catalog_results : list[SatelliteData or dict]
            Candidate scenes, as returned by ``PyGeoFetch.search()``
            or any STAC-like client.
        aoi : shapely.geometry.Polygon or None
            The user's area of interest. May be ``None``, in which
            case AOI-dependent checks are skipped for every scene
            (see :meth:`validate_scene`); every other check still
            runs normally.
        start_date, end_date : date, datetime, or None
            Temporal bounds, only used if
            ``config.check_temporal_bounds`` is enabled.

        Returns
        -------
        list[SatelliteData or dict]
            The subset of ``catalog_results`` that passed validation,
            in the same order and same representation they arrived in.
        """
        safe: list[SceneLike] = []
        for scene in catalog_results:
            report = self.validate_scene(scene, aoi, start_date, end_date)

            for issue in report.issues:
                log = (
                    logger.error if issue.severity == SEVERITY_ERROR else logger.warning
                )
                log(f"Scene {report.scene_id} [{issue.code}]: {issue.message}")

            if report.passed:
                safe.append(scene)
            else:
                first_error = report.errors[0] if report.errors else report.warnings[0]
                logger.error(
                    f"Scene {report.scene_id} rejected [{first_error.code}]: "
                    f"{first_error.message}"
                )

        logger.info(
            f"Optical preflight: {len(safe)}/{len(catalog_results)} scenes "
            "passed validation"
        )
        return safe
