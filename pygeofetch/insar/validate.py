"""
DataValidator — sanity checks for InSAR pipeline inputs and intermediate
products, so a malformed SLC, a broken coherence map, or a disconnected
SBAS network fails loudly and specifically at the point of entry, not
as a confusing downstream numerical artifact hours into a long
unwrapping or inversion run.

Usage::

    from pygeofetch.insar import DataValidator

    DataValidator.validate_slc(slc_array)
    DataValidator.validate_coherence(coherence_array)
    DataValidator.validate_sbas_network(pairs, dates)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Sequence

logger = logging.getLogger("pygeofetch.insar.validate")


@dataclass
class ValidationResult:
    """Outcome of a validation check — never raises on its own; callers
    decide whether to treat warnings as fatal."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise ValueError(
                f"Validation failed: {'; '.join(self.errors)}"
                + (f" (warnings: {'; '.join(self.warnings)})" if self.warnings else "")
            )

    def __bool__(self) -> bool:
        return self.valid


class DataValidator:
    """Static sanity checks for SLC data, coherence maps, and SBAS
    interferogram networks. All methods return a ValidationResult
    rather than raising directly, so callers can log-and-continue or
    call .raise_if_invalid() depending on how strict they want to be."""

    @staticmethod
    def validate_slc(data: Any, name: str = "SLC") -> ValidationResult:
        """
        Sanity-check a Single Look Complex array before it enters the
        coregistration/interferogram chain.

        Checks:
          - is a complex dtype (a real-valued array fed in here is
            almost always a sign the wrong band/product was loaded)
          - no NaN or Inf values
          - not all-zero (a common signature of a failed/empty read)
          - dynamic range isn't degenerate (near-constant amplitude
            usually means corrupted or placeholder data, not a real SLC)
        """
        np = _require_numpy()
        errors: List[str] = []
        warnings: List[str] = []

        arr = np.asarray(data)

        if not np.iscomplexobj(arr):
            errors.append(
                f"{name} is not complex-valued (dtype={arr.dtype}) — SLC data "
                f"must be complex (real + imaginary, or amplitude + phase "
                f"encoded as complex). A real-valued array here usually means "
                f"the wrong band or an already-detected (non-SLC) product was loaded."
            )

        if np.iscomplexobj(arr):
            finite_mask = np.isfinite(arr.real) & np.isfinite(arr.imag)
        else:
            finite_mask = np.isfinite(arr)

        n_total = arr.size
        n_nonfinite = n_total - int(np.sum(finite_mask))
        if n_nonfinite > 0:
            pct = 100 * n_nonfinite / n_total
            msg = f"{name} contains {n_nonfinite} NaN/Inf values ({pct:.2f}% of the array)"
            if pct > 50:
                errors.append(msg + " — over half the data is non-finite.")
            else:
                warnings.append(msg)

        if n_total > 0 and np.all(arr == 0):
            errors.append(
                f"{name} is entirely zero — this is almost always a failed or empty read, not real data."
            )

        if np.iscomplexobj(arr) and finite_mask.any():
            amplitude = np.abs(arr[finite_mask])
            if amplitude.size > 0:
                amp_std = float(np.std(amplitude))
                amp_mean = float(np.mean(amplitude))
                if amp_mean > 0 and amp_std / amp_mean < 1e-6:
                    warnings.append(
                        f"{name} amplitude is essentially constant "
                        f"(mean={amp_mean:.4g}, std={amp_std:.4g}) — real SAR "
                        f"amplitude always has speckle variation; this pattern "
                        f"usually means placeholder or corrupted data."
                    )

            imag_part = arr[finite_mask].imag
            if imag_part.size > 0 and float(np.max(np.abs(imag_part))) < 1e-10:
                errors.append(
                    f"{name} has a technically-complex dtype but zero imaginary "
                    f"part everywhere — this is the exact signature of amplitude-"
                    f"only data cast to a complex type (real + 0j), not genuine "
                    f"phase-preserving SLC data. Real SAR phase is never uniformly "
                    f"zero. This usually means the source product had no phase "
                    f"information (a detected/amplitude product was loaded instead "
                    f"of an SLC), not a real InSAR-ready input."
                )

        return ValidationResult(
            valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    @staticmethod
    def validate_coherence(data: Any, name: str = "coherence") -> ValidationResult:
        """
        Sanity-check a coherence map. Real coherence is bounded [0, 1]
        by definition (Touzi et al. 1999) — any value outside that
        range means an upstream computation error, not a legitimate
        edge case to silently clip and ignore.
        """
        np = _require_numpy()
        errors: List[str] = []
        warnings: List[str] = []

        arr = np.asarray(data)
        finite = arr[np.isfinite(arr)]

        if finite.size == 0:
            errors.append(f"{name} contains no finite values at all.")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        below_zero = float(np.min(finite))
        above_one = float(np.max(finite))
        if below_zero < -1e-6:
            errors.append(
                f"{name} has values below 0 (min={below_zero:.4f}) — coherence is bounded [0,1] by definition."
            )
        if above_one > 1 + 1e-6:
            errors.append(
                f"{name} has values above 1 (max={above_one:.4f}) — coherence is bounded [0,1] by definition."
            )

        if np.all(finite == finite.flat[0]):
            warnings.append(
                f"{name} is a single constant value ({finite.flat[0]:.4f}) across "
                f"the whole array — real coherence always varies spatially; this "
                f"usually means a computation didn't run or a placeholder was used."
            )

        return ValidationResult(
            valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    @staticmethod
    def validate_sbas_network(
        pairs: Sequence, dates: Sequence[str]
    ) -> ValidationResult:
        """
        Verify an SBAS interferogram network is fully connected — every
        acquisition date must be reachable from every other through the
        pair graph, or the least-squares inversion (Berardino et al.
        2002) is either singular or silently splits into independent,
        unrelated sub-networks with no shared reference. This is a
        graph-connectivity check, not a numerical one: it catches the
        network design problem before it ever reaches the (expensive)
        inversion step.

        Args:
            pairs: Sequence of (date1, date2) tuples/objects with
                  .date1/.date2 attributes representing each interferogram.
            dates: All acquisition dates that are supposed to be covered.
        """
        errors: List[str] = []
        warnings: List[str] = []

        date_set = set(dates)
        edges = []
        for p in pairs:
            d1 = (
                getattr(p, "reference_date", None)
                or getattr(p, "date1", None)
                or (p[0] if isinstance(p, (tuple, list)) else None)
            )
            d2 = (
                getattr(p, "secondary_date", None)
                or getattr(p, "date2", None)
                or (p[1] if isinstance(p, (tuple, list)) else None)
            )
            if d1 is None or d2 is None:
                errors.append(
                    f"Could not extract reference_date/secondary_date from pair {p!r}"
                )
                continue
            edges.append((d1, d2))

        if errors:
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        pair_dates = {d for edge in edges for d in edge}
        missing_from_pairs = date_set - pair_dates
        if missing_from_pairs:
            warnings.append(
                f"{len(missing_from_pairs)} date(s) have no interferogram pair at "
                f"all and will be unreachable in the inversion: {sorted(missing_from_pairs)}"
            )

        # Union-find connectivity check over the pair graph
        parent = {d: d for d in date_set}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for d1, d2 in edges:
            if d1 in parent and d2 in parent:
                union(d1, d2)

        components: dict = {}
        for d in date_set:
            root = find(d)
            components.setdefault(root, []).append(d)

        if len(components) > 1:
            comp_sizes = sorted((len(v) for v in components.values()), reverse=True)
            errors.append(
                f"SBAS network is NOT fully connected — {len(components)} "
                f"disconnected component(s) found (sizes: {comp_sizes}). "
                f"The inversion will either fail or silently treat these as "
                f"unrelated sub-networks with no shared displacement reference. "
                f"Add at least one interferogram pair bridging each component."
            )

        return ValidationResult(
            valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    @staticmethod
    def classify_pairs(
        pairs: Sequence,
        dates: Sequence[str],
        coherence_threshold: float = 0.3,
    ) -> "PairClassification":
        """
        Classify SBAS pairs into good/bridge/excluded, using real graph
        theory rather than a heuristic.

        A "bridge" here is the standard graph-theoretic definition: an
        edge whose removal increases the number of connected components.
        A low-coherence pair that is ALSO a bridge is topologically
        necessary — removing it would fracture the network into
        disconnected islands, the exact failure mode this project has
        directly hit and diagnosed before (a real network reported as
        "58/64 dates connected" by its own coherence-based selection
        step fractured into 17 pieces the moment a handful of
        low-quality pairs were removed naively). A low-coherence pair
        that is NOT a bridge is safe to drop outright — other paths
        already connect its endpoints.

        Args:
            pairs: Sequence of pair objects with .reference_date,
                  .secondary_date, and .coherence (a real coherence
                  array — its spatial mean is used as this pair's
                  scalar quality summary).
            dates: All acquisition dates expected in the network.
            coherence_threshold: Pairs at or above this mean coherence
                  are always "good", regardless of bridge status.

        Returns:
            PairClassification with three real, disjoint lists:
            good_pairs, bridge_pairs, excluded_pairs. Every input pair
            appears in exactly one of the three.
        """
        np = _require_numpy()

        def pair_dates(p):
            d1 = getattr(p, "reference_date", None) or (
                p[0] if isinstance(p, (tuple, list)) else None
            )
            d2 = getattr(p, "secondary_date", None) or (
                p[1] if isinstance(p, (tuple, list)) else None
            )
            return d1, d2

        def mean_coherence(p):
            coh = getattr(p, "coherence", None)
            if coh is None:
                return 0.0
            arr = np.asarray(coh)
            finite = arr[np.isfinite(arr)]
            return float(finite.mean()) if finite.size else 0.0

        date_set = set(dates)
        all_edges = [pair_dates(p) for p in pairs]

        def is_connected(edges):
            """Real union-find connectivity check over the given edge list."""
            parent = {d: d for d in date_set}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for d1, d2 in edges:
                if d1 in parent and d2 in parent:
                    r1, r2 = find(d1), find(d2)
                    if r1 != r2:
                        parent[r1] = r2
            roots = {find(d) for d in date_set}
            return len(roots) <= 1

        good_pairs, bridge_pairs, excluded_pairs = [], [], []

        for i, p in enumerate(pairs):
            coh = mean_coherence(p)
            if coh >= coherence_threshold:
                good_pairs.append(p)
                continue

            # Real bridge check: is the network still connected with
            # exactly this one edge removed from the full graph?
            edges_without_this = all_edges[:i] + all_edges[i + 1 :]
            if is_connected(edges_without_this):
                excluded_pairs.append(p)  # other paths exist -- safe to drop
            else:
                bridge_pairs.append(p)  # topologically necessary despite low quality

        return PairClassification(
            good_pairs=good_pairs,
            bridge_pairs=bridge_pairs,
            excluded_pairs=excluded_pairs,
        )


@dataclass
class PairClassification:
    """
    Real output of DataValidator.classify_pairs() — every input pair
    appears in exactly one list. bridge_pairs are real, measured
    low-coherence pairs kept only because removing them would
    disconnect the network (see classify_pairs' own docstring for the
    exact graph-theoretic definition used).
    """

    good_pairs: List[Any] = field(default_factory=list)
    bridge_pairs: List[Any] = field(default_factory=list)
    excluded_pairs: List[Any] = field(default_factory=list)

    def summary(self) -> str:
        total = len(self.good_pairs) + len(self.bridge_pairs) + len(self.excluded_pairs)
        return (
            f"{len(self.good_pairs)}/{total} good, "
            f"{len(self.bridge_pairs)}/{total} bridge (kept, down-weighted), "
            f"{len(self.excluded_pairs)}/{total} excluded (redundant, safe to drop)"
        )


def _require_numpy():
    try:
        import numpy as np

        return np
    except ImportError as exc:
        raise ImportError("DataValidator requires numpy: pip install numpy") from exc
