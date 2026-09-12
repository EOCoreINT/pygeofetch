# Organizing Downloaded Data

A flat directory of dozens of downloaded scenes across dates, orbits,
and sensors is genuinely unusable for downstream InSAR or multi-date
optical work. `pygeofetch.core.data_organizer.DataOrganizer` turns that
into a real, structured hierarchy, plus a real manifest mapping the
structure back to the underlying scene metadata.

## Basic usage

```python
from pygeofetch import PyGeoFetch
from pygeofetch.core.data_organizer import DataOrganizer, OrganizerConfig

client = PyGeoFetch()
scenes = client.search(query)
downloads = client.download(scenes, destination="./raw")

organizer = DataOrganizer(OrganizerConfig(group_by=["satellite", "date"]))
report = organizer.organize(list(zip(scenes, downloads)), output_dir="./organized")

print(f"{report.grouped} scenes organized -> {report.manifest_path}")
```

With `group_by=["satellite", "date"]`, this produces:

```
organized/
├── Sentinel-2A/
│   └── 2024-06-15/
│       └── S2A_MSIL2A_...tif
├── Sentinel-1A/
│   └── 2024-06-20/
│       └── S1A_IW_SLC_...zip
└── manifest.json
```

## Configuration

`OrganizerConfig` controls the real grouping hierarchy and how files
are placed into it:

| Field | Default | Meaning |
|---|---|---|
| `group_by` | `["date"]` | Ordered list of `"date"`, `"orbit"`, `"satellite"`, `"processing_level"` — applied in order, so `["satellite", "date"]` nests date folders inside satellite folders. |
| `file_operation` | `"copy"` | `"copy"`, `"symlink"`, or `"move"`. |
| `manifest_filename` | `"manifest.json"` | Written to the root of `output_dir` after organizing. |

:::{admonition} Copy is the default for a real reason
:class: note

`file_operation` defaults to `"copy"`, not `"move"` — verified
directly that the original downloaded file survives an `organize()`
call. A grouping bug that also deleted your only copy of an
already-downloaded scene would be a serious, unrecoverable failure
mode. Opt into `"move"` explicitly once you trust your own grouping
configuration, or use `"symlink"` to avoid duplicating large scenes
on disk while keeping the original flat layout intact too.
:::
`organize()` is also idempotent — running it twice on the same input
doesn't re-copy, error, or double-count.

## The manifest

`manifest.json` maps every organized scene back to its real metadata:

```json
{
  "pygeofetch_manifest_version": 1,
  "generated": "2026-09-12T10:00:00+00:00",
  "group_by": ["satellite", "date"],
  "scene_count": 15,
  "scenes": [
    {
      "id": "S1A_IW_SLC__1SDV_20240615...",
      "satellite": "Sentinel-1A",
      "datetime": "2024-06-15 10:38:43",
      "bbox": [109.9, 39.28, 110.12, 39.48],
      "cloud_cover": null,
      "relative_orbit": 11,
      "group": "Sentinel-1A/2024-06-15",
      "paths": ["organized/Sentinel-1A/2024-06-15/S1A_IW_SLC_....zip"]
    }
  ]
}
```

`OrganizeReport` (returned by `organize()`) also reports what *couldn't*
be organized, rather than silently dropping it: `skipped_no_output`
(a successful download with no real file path attached) and
`skipped_failed_download` (the download itself failed) are both real,
separate counts — inspect them if `report.grouped` is lower than you
expect.

## Preparing an InSAR stack

```python
report = organizer.prepare_insar_stack(
    list(zip(scenes, downloads)),
    orbit_files=orbit_file_paths,       # optional: {date: Path}, from fetch_orbit_file()
    ground_point=aoi_center_ecef,       # optional: (x, y, z) ECEF, required with orbit_files
)

print(f"Tracks: {list(report.tracks.keys())}")
print(f"Orphaned (no relative_orbit): {report.orphaned}")
print(f"Burst-sync checked: {report.burst_sync_checked}")
```

`prepare_insar_stack()` groups downloaded SLCs by real track/relative
orbit and flags orphaned scenes (no `relative_orbit` metadata at all,
so they can't be placed in any track).

When you supply real orbit files and a ground point, it goes further:
it reuses [`pygeofetch.insar.stack_selection.select_burst_synchronized_dates`](../processing/insar.md)
— the same real, already-verified burst-timing classification the
InSAR pipeline itself uses — rather than a second, independent
implementation of that check. `report.ready_paths` then only includes
the dates that actually pass burst-sync classification.

:::{admonition} Honest behavior without orbit files
:class: note

Without real orbit files, burst-sync compatibility genuinely cannot
be checked — it's a real, separate download step, not something
this method can fabricate. `prepare_insar_stack()` still returns
real track/date grouping in that case, but `report.burst_sync_checked`
is `False`, and `report.ready_paths` includes every date rather than
only the burst-sync-compatible ones. Check `burst_sync_checked`
before assuming the returned stack is ready for coregistration.
:::