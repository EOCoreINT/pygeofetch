"""
pygeofetch.core.data_organizer — post-download organization.

A flat download directory of dozens of scenes across dates, orbits, and
sensors is genuinely unusable for downstream InSAR or multi-date optical
work — this module turns that into a real, structured hierarchy plus a
real manifest mapping the structure back to the underlying scene metadata.

Real design choices, stated rather than left implicit:

- Grouping never *loses* the original files by default: `file_operation`
  defaults to "copy", not "move". A user who wants to reclaim disk space
  can opt into "move" or "symlink" explicitly, but the safe default
  doesn't require trusting the grouping logic before deleting the
  original flat download.
- `prepare_insar_stack()` reuses the existing, already-verified
  `pygeofetch.insar.stack_selection.select_burst_synchronized_dates`
  engine for burst-timing-family classification rather than
  reimplementing burst-sync checking a second time — the same real
  engine the InSAR tutorials use directly. It only runs when real orbit
  files are supplied; without them, this method still does real
  track/date grouping, but honestly reports that burst-sync compatibility
  could not be checked, rather than silently skipping the check.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from pygeofetch.models.download_task import DownloadResult
from pygeofetch.models.satellite_data import SatelliteData

logger = logging.getLogger("pygeofetch.data_organizer")

__all__ = [
    "OrganizerConfig",
    "OrganizeReport",
    "InSARStackReport",
    "DataOrganizer",
]

GroupCriterion = Literal["date", "orbit", "satellite", "processing_level"]


class OrganizerConfig(BaseModel):
    """
    Configuration for `DataOrganizer`.

    Attributes
    ----------
    group_by : list[GroupCriterion]
        Real grouping hierarchy, applied in order — e.g.
        ``["satellite", "date"]`` produces ``S2A/2024-06-15/scene.tif``,
        while ``["date", "orbit"]`` produces ``2024-06-15/R054/scene.tif``.
        Defaults to ``["date"]`` — the one grouping every real workflow
        in this project (InSAR, optical time series) needs regardless of
        anything else.
    file_operation : "copy" | "symlink" | "move"
        How files are placed into the grouped hierarchy. "copy" (the
        default) never touches the original flat download; "symlink"
        avoids duplicating large scenes on disk while keeping the
        original layout intact too; "move" reclaims disk space but is
        destructive if the grouping logic ever gets something wrong —
        opt in deliberately, it is never the default.
    manifest_filename : str
        Real manifest file written to `output_dir` after organizing.
    """

    group_by: list[GroupCriterion] = Field(default_factory=lambda: ["date"])
    file_operation: Literal["copy", "symlink", "move"] = "copy"
    manifest_filename: str = "manifest.json"


@dataclass
class OrganizeReport:
    """Real result of an `organize()` call."""

    grouped: int = 0
    skipped_no_output: int = 0
    skipped_failed_download: int = 0
    manifest_path: Path | None = None
    groups: dict[str, list[str]] = field(
        default_factory=dict
    )  # group path -> scene ids


@dataclass
class InSARStackReport:
    """Real result of a `prepare_insar_stack()` call."""

    tracks: dict[int, list[str]] = field(
        default_factory=dict
    )  # relative_orbit -> dates
    orphaned: list[str] = field(default_factory=list)  # scenes with no relative_orbit
    burst_sync_checked: bool = False
    burst_family_report: dict[str, Any] | None = (
        None  # from select_burst_synchronized_dates
    )
    ready_paths: dict[str, Path] = field(default_factory=dict)  # date -> real SLC path


def _group_key_for(scene: SatelliteData, criterion: GroupCriterion) -> str:
    """Real, single-criterion grouping key for one scene. Every branch
    has an explicit, honest fallback for missing metadata rather than
    raising or silently dropping the scene from grouping."""
    if criterion == "date":
        if scene.datetime is None:
            return "unknown-date"
        dt = scene.datetime
        d = dt.date() if isinstance(dt, datetime) else dt
        return d.isoformat() if isinstance(d, date) else str(dt)[:10]
    if criterion == "orbit":
        return (
            f"R{scene.relative_orbit:03d}"
            if scene.relative_orbit is not None
            else "unknown-orbit"
        )
    if criterion == "satellite":
        return (scene.satellite or "unknown-satellite").replace(" ", "_")
    if criterion == "processing_level":
        return str(
            scene.processing_level.value
            if hasattr(scene.processing_level, "value")
            else scene.processing_level
        )
    raise ValueError(f"Unknown group_by criterion: {criterion!r}")  # pragma: no cover


class DataOrganizer:
    """
    Groups downloaded scenes into a real, structured hierarchy and
    generates a manifest mapping that structure back to scene metadata.

    Examples
    --------
    >>> organizer = DataOrganizer(OrganizerConfig(group_by=["satellite", "date"]))
    >>> report = organizer.organize(list(zip(scenes, download_results)), output_dir="./organized")
    >>> print(f"{report.grouped} scenes organized -> {report.manifest_path}")
    """

    def __init__(self, config: OrganizerConfig | None = None) -> None:
        self.config = config or OrganizerConfig()

    def _group_dir_for(self, scene: SatelliteData) -> Path:
        parts = [_group_key_for(scene, c) for c in self.config.group_by]
        return Path(*parts) if parts else Path(".")

    def _place_file(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return  # real idempotency: re-running organize() doesn't re-copy/error
        op = self.config.file_operation
        if op == "copy":
            shutil.copy2(src, dst)
        elif op == "symlink":
            dst.symlink_to(src.resolve())
        elif op == "move":
            shutil.move(str(src), str(dst))

    def organize(
        self,
        results: list[tuple[SatelliteData, DownloadResult]],
        output_dir: str | Path,
    ) -> OrganizeReport:
        """
        Group real downloaded files into `output_dir` per `config.group_by`,
        and write a real manifest.

        Parameters
        ----------
        results : list of (SatelliteData, DownloadResult)
            Real, matched pairs — typically built by zipping the list
            returned from `client.search()` with the list returned from
            `client.download()`, in the same order both calls preserve.
        output_dir : str or Path
            Root of the real, organized hierarchy.

        Returns
        -------
        OrganizeReport
            Real counts and the manifest path — inspect `.skipped_*` to
            see how many inputs were genuinely un-organizable (failed
            downloads, or successful downloads with no real output path)
            rather than silently losing that information.
        """
        output_dir = Path(output_dir)
        manifest_entries: list[dict[str, Any]] = []
        groups: dict[str, list[str]] = defaultdict(list)
        report = OrganizeReport()

        for scene, dl in results:
            if not dl.success:
                report.skipped_failed_download += 1
                continue
            real_paths = (
                list(dl.output_paths)
                if dl.output_paths
                else ([dl.output_path] if dl.output_path else [])
            )
            if not real_paths:
                report.skipped_no_output += 1
                continue

            group_dir = self._group_dir_for(scene)
            dest_dir = output_dir / group_dir
            new_paths: list[str] = []
            for src in real_paths:
                src = Path(src)
                if not src.exists():
                    logger.warning(
                        "organize(): %s -- recorded output path does not exist on "
                        "disk (%s), skipping this file but still manifesting the "
                        "scene's other real files",
                        scene.id,
                        src,
                    )
                    continue
                dst = dest_dir / src.name
                self._place_file(src, dst)
                new_paths.append(str(dst))

            report.grouped += 1
            groups[str(group_dir)].append(scene.id)
            manifest_entries.append(
                {
                    "id": scene.id,
                    "provider": scene.provider,
                    "satellite": scene.satellite,
                    "sensor": scene.sensor,
                    "datetime": str(scene.datetime) if scene.datetime else None,
                    "bbox": list(scene.bbox) if scene.bbox else None,
                    "cloud_cover": scene.cloud_cover,
                    "processing_level": str(
                        scene.processing_level.value
                        if hasattr(scene.processing_level, "value")
                        else scene.processing_level
                    ),
                    "relative_orbit": scene.relative_orbit,
                    "orbit_number": scene.orbit_number,
                    "pass_direction": scene.pass_direction,
                    "product_type": scene.product_type,
                    "polarisation": scene.polarisation,
                    "group": str(group_dir),
                    "paths": new_paths,
                }
            )

        report.groups = dict(groups)
        report.manifest_path = self._write_manifest(manifest_entries, output_dir)
        return report

    def _write_manifest(self, entries: list[dict[str, Any]], output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "pygeofetch_manifest_version": 1,
            "generated": datetime.now(timezone.utc).isoformat(),
            "group_by": self.config.group_by,
            "scene_count": len(entries),
            "scenes": entries,
        }
        path = output_dir / self.config.manifest_filename
        path.write_text(json.dumps(manifest, indent=2, default=str))
        return path

    def prepare_insar_stack(
        self,
        results: list[tuple[SatelliteData, DownloadResult]],
        orbit_files: dict[str, Path] | None = None,
        ground_point: tuple[float, float, float] | None = None,
        **burst_sync_kwargs: Any,
    ) -> InSARStackReport:
        """
        Group downloaded SLCs by real track/relative-orbit and, when
        real orbit files are supplied, check burst-timing-family
        compatibility using the already-verified
        `select_burst_synchronized_dates` engine — not a second,
        independent reimplementation of that check.

        Parameters
        ----------
        results : list of (SatelliteData, DownloadResult)
            Real, matched SLC scenes and their download results.
        orbit_files : dict[str, Path], optional
            Real `{date: orbit_file_path}` mapping, e.g. from
            `fetch_orbit_file()` for each date. Burst-sync checking is
            skipped (honestly, not silently) when this is omitted --
            it's a real, separate download step, not something this
            method can fabricate.
        ground_point : tuple[float, float, float], optional
            Real ECEF ground point for the AOI center, required only
            when `orbit_files` is supplied (needed by the underlying
            burst-sync engine's zero-Doppler geometry).
        **burst_sync_kwargs
            Forwarded to `select_burst_synchronized_dates()` --
            `swath_hints`, `min_majority_dates`, `redundancy`, etc.

        Returns
        -------
        InSARStackReport
            Real per-track grouping, orphaned scenes (no relative_orbit
            metadata -- can't be placed in any track), and, when
            burst-sync was checked, the real classification report from
            `select_burst_synchronized_dates`.
        """
        report = InSARStackReport()
        by_date: dict[str, Path] = {}
        by_track: dict[int, list[str]] = defaultdict(list)

        for scene, dl in results:
            if not dl.success or scene.datetime is None:
                continue
            date_str = _group_key_for(scene, "date")
            real_path = dl.output_path or (
                dl.output_paths[0] if dl.output_paths else None
            )
            if real_path is None:
                continue
            by_date[date_str] = Path(real_path)
            if scene.relative_orbit is not None:
                by_track[scene.relative_orbit].append(date_str)
            else:
                report.orphaned.append(scene.id)

        report.tracks = dict(by_track)
        report.ready_paths = by_date

        if orbit_files and ground_point is not None:
            from pygeofetch.insar.stack_selection import select_burst_synchronized_dates

            dates = sorted(by_date.keys())
            safe_zips = {d: by_date[d] for d in dates}
            usable_orbits = {d: orbit_files[d] for d in dates if d in orbit_files}
            if len(usable_orbits) < len(dates):
                logger.warning(
                    "prepare_insar_stack(): %d/%d dates have a real orbit file -- "
                    "burst-sync classification will only consider those; widen "
                    "orbit_files to cover the rest for a complete check.",
                    len(usable_orbits),
                    len(dates),
                )
            good_dates, family_report = select_burst_synchronized_dates(
                dates, safe_zips, usable_orbits, ground_point, **burst_sync_kwargs
            )
            report.burst_sync_checked = True
            report.burst_family_report = family_report
            report.ready_paths = {d: by_date[d] for d in good_dates}
        else:
            logger.info(
                "prepare_insar_stack(): no orbit_files/ground_point supplied -- "
                "returning real track/date grouping only, burst-sync "
                "compatibility was not checked. Pass real orbit files "
                "(e.g. from fetch_orbit_file()) to enable that check."
            )

        return report
