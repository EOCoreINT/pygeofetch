"""
Download-budget thinning that never breaks the interferometric network.

A full Sentinel-1 SLC is ~7.5 GB, so a 4-year, 6-day-repeat stack is >1.6 TB.
Most time-series work does not need every acquisition, so this module picks an
evenly spaced subset of acquisition *dates* -- and guarantees the result is
still a connected SBAS network:

* every kept date has at least one partner within ``max_temporal_baseline_days``
  (and, where possible, ``min_redundancy`` partners);
* the number of disconnected components is never larger than in the full
  stack (data gaps in the archive cannot be repaired, but thinning will not
  create new ones);
* if a download budget cannot be met without disconnecting the network, the
  network wins and the report says by how much the budget was exceeded.

Pure functions, no network, no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date


def _d(iso: str) -> date:
    return date.fromisoformat(iso[:10])


def _days(a: str, b: str) -> int:
    return (_d(b) - _d(a)).days


@dataclass
class NetworkStats:
    """Connectivity of the temporal-baseline-limited pair graph."""

    n_dates: int
    n_pairs: int
    n_components: int
    min_redundancy: int  # smallest number of partners any date has
    weak_dates: list[str] = field(default_factory=list)  # dates below target


@dataclass
class ThinningResult:
    kept: list[str]
    dropped: list[str]
    requested_days: int | None
    effective_days: int | None
    gb_per_scene: float
    gb_before: float
    gb_after: float
    max_gb: float | None
    budget_met: bool
    stats_before: NetworkStats
    stats_after: NetworkStats
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.dropped)


def network_stats(
    dates: Sequence[str],
    max_temporal_baseline_days: int,
    min_redundancy: int = 2,
) -> NetworkStats:
    """Union-find over pairs closer than ``max_temporal_baseline_days``."""
    ds = sorted(set(dates))
    n = len(ds)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    degree = [0] * n
    pairs = 0
    ords = [_d(x).toordinal() for x in ds]
    for i in range(n):
        j = i + 1
        while j < n and ords[j] - ords[i] <= max_temporal_baseline_days:
            pairs += 1
            degree[i] += 1
            degree[j] += 1
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
            j += 1
    comps = len({find(i) for i in range(n)}) if n else 0
    weak = [ds[i] for i in range(n) if degree[i] < min_redundancy]
    return NetworkStats(
        n_dates=n,
        n_pairs=pairs,
        n_components=comps,
        min_redundancy=min(degree) if n else 0,
        weak_dates=weak,
    )


def _cadence_pick(dates: list[str], every_days: int) -> list[str]:
    """Evenly spaced subset: after each kept date take the acquisition closest
    to ``last + every_days`` (never nearer than half a step)."""
    if not dates or every_days <= 1:
        return list(dates)
    ords = [_d(x).toordinal() for x in dates]
    kept_idx = [0]
    while True:
        last = ords[kept_idx[-1]]
        target = last + every_days
        floor = last + max(1, every_days // 2)
        best: int | None = None
        for j in range(kept_idx[-1] + 1, len(dates)):
            if ords[j] < floor:
                continue
            if best is None or abs(ords[j] - target) < abs(ords[best] - target):
                best = j
            if ords[j] > target:
                break
        if best is None:
            break
        kept_idx.append(best)
    kept = [dates[i] for i in kept_idx]
    # keep the final acquisition so the time span is not silently shortened
    if kept[-1] != dates[-1] and _days(kept[-1], dates[-1]) >= every_days // 2:
        kept.append(dates[-1])
    return kept


def _repair_connectivity(
    kept: list[str],
    all_dates: list[str],
    max_tb: int,
    target_components: int,
) -> tuple[list[str], list[str]]:
    """Add back the dropped dates that reconnect the network."""
    kept_set = set(kept)
    added: list[str] = []
    while network_stats(sorted(kept_set), max_tb).n_components > target_components:
        cur = network_stats(sorted(kept_set), max_tb).n_components
        best, best_c = None, cur
        for d in all_dates:
            if d in kept_set:
                continue
            c = network_stats(sorted(kept_set | {d}), max_tb).n_components
            if c < best_c:
                best, best_c = d, c
        if best is None:
            # no single date helps (gap wider than 2x baseline): bridge the
            # widest remaining gap with its midpoint-most dropped date
            ks = sorted(kept_set)
            gaps = [
                (_days(a, b), a, b) for a, b in zip(ks, ks[1:]) if _days(a, b) > max_tb
            ]
            if not gaps:
                break
            _, a, b = max(gaps)
            mid = (_d(a).toordinal() + _d(b).toordinal()) / 2
            cand = [d for d in all_dates if a < d < b and d not in kept_set]
            if not cand:
                break
            best = min(cand, key=lambda d: abs(_d(d).toordinal() - mid))
        kept_set.add(best)
        added.append(best)
    return sorted(kept_set), added


def thin_dates(
    dates: Sequence[str],
    *,
    max_temporal_baseline_days: int,
    thin_to_days: int | None = None,
    max_download_gb: float | None = None,
    gb_per_scene: float = 7.5,
    min_redundancy: int = 2,
) -> ThinningResult:
    """Choose a connected, evenly spaced subset of ``dates``."""
    all_dates = sorted(set(dates))
    before = network_stats(all_dates, max_temporal_baseline_days, min_redundancy)
    gb_before = len(all_dates) * gb_per_scene
    notes: list[str] = []

    def finish(kept: list[str], eff: int | None, met: bool) -> ThinningResult:
        kept = sorted(kept)
        after = network_stats(kept, max_temporal_baseline_days, min_redundancy)
        return ThinningResult(
            kept=kept,
            dropped=[d for d in all_dates if d not in set(kept)],
            requested_days=thin_to_days,
            effective_days=eff,
            gb_per_scene=gb_per_scene,
            gb_before=gb_before,
            gb_after=len(kept) * gb_per_scene,
            max_gb=max_download_gb,
            budget_met=met,
            stats_before=before,
            stats_after=after,
            notes=notes,
        )

    if len(all_dates) < 3 or (thin_to_days is None and max_download_gb is None):
        return finish(all_dates, None, True)

    span = _days(all_dates[0], all_dates[-1])
    # Largest step that still gives every date `min_redundancy` partners.
    step_cap = max(1, max_temporal_baseline_days // max(1, min_redundancy))

    step = thin_to_days or 1
    if thin_to_days and thin_to_days > step_cap:
        notes.append(
            f"thin_to_days={thin_to_days} lowered to {step_cap}: with a "
            f"{max_temporal_baseline_days}-day temporal baseline a coarser "
            f"cadence leaves dates with fewer than {min_redundancy} partners."
        )
        step = step_cap

    budget_n: int | None = None
    if max_download_gb is not None:
        budget_n = max(2, int(max_download_gb // gb_per_scene))
        needed = math.ceil(span / max(1, step)) + 1
        if needed > budget_n:
            step = max(step, math.ceil(span / max(1, budget_n - 1)))

    # The network must stay connected: step can never exceed the baseline.
    hard_cap = max(1, max_temporal_baseline_days)
    if step > hard_cap:
        notes.append(
            f"A {step}-day cadence would disconnect the network (temporal "
            f"baseline {max_temporal_baseline_days} d); using {hard_cap} d."
        )
        step = hard_cap

    kept = _cadence_pick(all_dates, step)
    kept, added = _repair_connectivity(
        kept, all_dates, max_temporal_baseline_days, before.n_components
    )
    if added:
        notes.append(
            f"{len(added)} date(s) added back to keep the network connected: "
            f"{', '.join(added[:5])}{' ...' if len(added) > 5 else ''}."
        )

    # Trim only if over budget AND removal cannot disconnect anything.
    if budget_n is not None and len(kept) > budget_n:
        base = network_stats(kept, max_temporal_baseline_days).n_components
        while len(kept) > budget_n and len(kept) > 2:
            order = sorted(
                range(1, len(kept) - 1),
                key=lambda i: _days(kept[i - 1], kept[i + 1]),
            )
            for i in order:
                trial = kept[:i] + kept[i + 1 :]
                if (
                    network_stats(trial, max_temporal_baseline_days).n_components
                    <= base
                ):
                    kept = trial
                    break
            else:
                break

    met = budget_n is None or len(kept) <= budget_n
    if not met:
        notes.append(
            f"Budget of {max_download_gb:g} GB would need <= {budget_n} scenes, "
            f"but keeping the network connected needs {len(kept)} "
            f"({len(kept) * gb_per_scene:.0f} GB). Raise the budget or the "
            f"temporal baseline (max_temporal_baseline_days)."
        )
    return finish(kept, step, met)


def date_groups(scenes: Sequence[object]) -> dict[str, list[object]]:
    """Group scene objects by acquisition date (``scene.datetime``)."""
    out: dict[str, list[object]] = {}
    for s in scenes:
        key = str(getattr(s, "datetime", ""))[:10]
        out.setdefault(key, []).append(s)
    return out
