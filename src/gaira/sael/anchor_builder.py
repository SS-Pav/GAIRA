"""Part B — SAEL anchor-window construction.

Cluster extracted anchor evidence rows into windows, aggregate metadata, and
classify. Different from `gaira.expected.anchor_windows` because SAEL carries
contrast + direction + substrate + matrix per evidence row and records them
per window.

Clustering rule:
  - peak_cm1 values are sorted within a candidate axis; a new cluster starts
    when two consecutive values differ by more than GAP_CM (default 25).
  - region_lo/hi entries expand the cluster bounds but do not seed new
    clusters — they're already wide.
  - Clusters are padded by ±PAD_CM (5).

Classification (simple, transparent):
  - anchor       : >=2 distinct sources AND ambiguity_score <= 0.4 AND has direction signal OR multi-source assignment support
  - secondary    : >=1 source, some agreement, or matches an axis-anchor hint range
  - ambiguous    : ambiguity > 0.4, OR conflicting up/down from >1 sources
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path

import pandas as pd

from gaira.expected.axis_mapping import (
    AXIS_ANCHOR_HINTS, BSV_AXES, assigned_row_to_axis,
)


GAP_CM = 25.0
PAD_CM = 5.0
MAX_AMBIG_ANCHOR = 0.4


@dataclass
class AnchorWindowV1:
    window_id: str
    start_cm1: float
    end_cm1: float
    central_cm1: float
    source_count: int
    condition_count: int
    direction_distribution: dict[str, int]
    matrix_distribution: dict[str, int]
    substrate_distribution: dict[str, int]
    candidate_axes: list[str]
    primary_axis: str | None
    ambiguity_score: float          # share of peaks mapping to a different axis
    locality_score: float            # peak-based / (peak+region) signals
    classification: str               # anchor | secondary | ambiguous
    supporting_source_ids: list[str] = field(default_factory=list)
    priority_tags: list[str] = field(default_factory=list)
    notes: str = ""


def _row_axis(row: pd.Series) -> str | None:
    """Infer the BSV axis for a single anchor evidence row.

    Uses the assigned_molecule + priority_tags to pick a likely axis via the
    existing axis_mapping. Falls back to None (ambiguous) if unclear.
    """
    mol = row.get("assigned_molecule") or ""
    tags = (row.get("priority_tags") or "").lower()
    # Build a pseudo "assigned_group" token string for axis_mapping:
    # axis_mapping works on assigned_group first. We fake it using priority_tags.
    if "purine" in tags:
        return "purine_nucleotide"
    if "pyrimidine" in tags:
        return "pyrimidine_nucleotide"
    if "aromatic_aa" in tags:
        return "aromatic_amino_acid"
    if "lipid" in tags:
        return "membrane_lipid"
    if "protein" in tags:
        return "protein_backbone"
    if "glycan" in tags:
        return "glycan_carbohydrate"
    if "redox_sulfur" in tags:
        return "redox_metabolite"
    if "nucleic_bb" in tags:
        return "nucleic_acid_backbone"
    # No hit — return None (ambiguous).
    return None


def _cluster_peaks(vals: list[float], gap: float = GAP_CM) -> list[list[int]]:
    if not vals:
        return []
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out: list[list[int]] = [[order[0]]]
    for k in order[1:]:
        if vals[k] - vals[out[-1][-1]] > gap:
            out.append([k])
        else:
            out[-1].append(k)
    return out


def _axes_hinted_by_range(start: float, end: float) -> list[str]:
    hits = []
    for axis, hints in AXIS_ANCHOR_HINTS.items():
        for lo, hi, _ in hints:
            if end >= lo and start <= hi:
                hits.append(axis)
                break
    return hits


def _ambiguity_score(
    start: float, end: float, axis: str, all_peaks: pd.DataFrame,
) -> float:
    """Fraction of peaks in [start, end] assigned to a DIFFERENT axis."""
    band = all_peaks[
        (all_peaks["peak_cm1"].notna())
        & (all_peaks["peak_cm1"] >= start)
        & (all_peaks["peak_cm1"] <= end)
    ]
    if band.empty:
        return 0.0
    other = band[(band["_row_axis"].notna()) & (band["_row_axis"] != axis)]
    return round(len(other) / len(band), 3)


def _classify(
    direction_counts: dict[str, int],
    source_count: int,
    ambiguity: float,
    hint_matched: bool,
) -> str:
    up = direction_counts.get("up", 0)
    down = direction_counts.get("down", 0)
    # Conflict: both up and down reported in the same window.
    conflicting = up > 0 and down > 0

    if ambiguity > MAX_AMBIG_ANCHOR:
        return "ambiguous"
    if conflicting:
        return "ambiguous"
    if source_count >= 2 and (up + down) >= 1:
        return "anchor"
    if source_count >= 2 and hint_matched:
        return "anchor"
    if source_count >= 1 and hint_matched:
        return "secondary"
    if source_count >= 2:
        return "secondary"
    return "ambiguous"


def build_sael_anchor_windows(
    evidence_df: pd.DataFrame,
) -> pd.DataFrame:
    """Cluster anchor evidence rows into windows and classify each.

    Inputs: evidence_df from extract_anchor_evidence().
    Returns: DataFrame of AnchorWindowV1 rows.
    """
    # Annotate each row with a candidate axis for clustering.
    df = evidence_df.copy()
    df["_row_axis"] = df.apply(_row_axis, axis=1)

    # We cluster within axis groups (including None/unmapped as its own
    # bucket). Rows without a peak_cm1 but with a region contribute only to
    # the aggregation of windows they intersect; they don't seed clusters.
    windows: list[AnchorWindowV1] = []

    peak_rows = df[df["peak_cm1"].notna()]
    region_rows = df[(df["peak_cm1"].isna())
                       & (df["region_lo_cm1"].notna())
                       & (df["region_hi_cm1"].notna())]

    for axis in list(BSV_AXES) + [None]:
        axis_name = axis or "_unmapped"
        axis_peaks = peak_rows[peak_rows["_row_axis"] == axis].reset_index(drop=True)
        if axis_peaks.empty:
            continue
        clusters = _cluster_peaks(axis_peaks["peak_cm1"].tolist(), GAP_CM)
        for cluster in clusters:
            sub = axis_peaks.iloc[cluster]
            start = float(sub["peak_cm1"].min() - PAD_CM)
            end = float(sub["peak_cm1"].max() + PAD_CM)
            # Fold in region-type rows that intersect this window.
            regions_in = region_rows[
                (region_rows["region_hi_cm1"] >= start)
                & (region_rows["region_lo_cm1"] <= end)
            ]
            all_rows = pd.concat([sub, regions_in], ignore_index=True)

            direction_counts = dict(
                Counter(d for d in all_rows["direction"].tolist()
                         if d in ("up", "down", "mixed"))
            )
            matrix_counts = dict(
                Counter(m for m in all_rows["matrix"].tolist() if m)
            )
            substrate_counts = dict(
                Counter(s for s in all_rows["substrate"].tolist() if s)
            )
            conditions = [c for c in all_rows["condition_a"].tolist() if c]
            supporting_sources = sorted(
                {s for s in all_rows["source_id"].tolist() if s}
            )

            ambig = _ambiguity_score(
                start, end, axis, df[df["peak_cm1"].notna()],
            )
            hint_axes = _axes_hinted_by_range(start, end)
            hint_matched = axis in hint_axes if axis else False
            candidates = list(dict.fromkeys(
                ([axis] if axis else []) + hint_axes
            ))
            primary = axis if axis else (hint_axes[0] if hint_axes else None)
            cls = _classify(direction_counts, len(supporting_sources), ambig, hint_matched)

            priority_tags: Counter[str] = Counter()
            for t in all_rows["priority_tags"].fillna("").tolist():
                for tok in (x.strip() for x in t.split(";") if x.strip()):
                    priority_tags[tok] += 1

            windows.append(AnchorWindowV1(
                window_id=f"{axis_name}:{round(start):04d}-{round(end):04d}",
                start_cm1=round(start, 1), end_cm1=round(end, 1),
                central_cm1=round((start + end) / 2, 1),
                source_count=len(supporting_sources),
                condition_count=len(set(conditions)),
                direction_distribution=direction_counts,
                matrix_distribution=matrix_counts,
                substrate_distribution=substrate_counts,
                candidate_axes=candidates,
                primary_axis=primary,
                ambiguity_score=ambig,
                locality_score=round(len(sub) / max(len(all_rows), 1), 3),
                classification=cls,
                supporting_source_ids=supporting_sources,
                priority_tags=[t for t, _ in priority_tags.most_common(5)],
                notes="",
            ))

    # Also emit windows seeded purely by region rows that didn't intersect any
    # peak cluster. These are the broad "spectral_region"-style entries (e.g.
    # biomarker_claims 1300-1500). They never qualify as anchor but can serve
    # as secondary/ambiguous scaffolding.
    covered = []
    for w in windows:
        covered.append((w.start_cm1, w.end_cm1))
    for _, r in region_rows.iterrows():
        lo, hi = float(r["region_lo_cm1"]), float(r["region_hi_cm1"])
        if any(lo <= c[1] and hi >= c[0] for c in covered):
            continue
        direction_counts = {r["direction"]: 1} if r["direction"] in ("up", "down", "mixed") else {}
        matrix_counts = {r["matrix"]: 1} if r["matrix"] else {}
        windows.append(AnchorWindowV1(
            window_id=f"_region_only:{int(lo):04d}-{int(hi):04d}",
            start_cm1=lo, end_cm1=hi,
            central_cm1=round((lo + hi) / 2, 1),
            source_count=1 if r.get("source_id") else 0,
            condition_count=1 if r["condition_a"] else 0,
            direction_distribution=direction_counts,
            matrix_distribution=matrix_counts,
            substrate_distribution={},
            candidate_axes=_axes_hinted_by_range(lo, hi),
            primary_axis=(_axes_hinted_by_range(lo, hi) or [None])[0],
            ambiguity_score=0.0,
            locality_score=0.0,
            classification="secondary" if _axes_hinted_by_range(lo, hi) else "ambiguous",
            supporting_source_ids=[r["source_id"]] if r.get("source_id") else [],
            priority_tags=[t.strip() for t in (r.get("priority_tags") or "").split(";") if t.strip()],
            notes="region-only evidence (no peak-level clustering); width >50 cm⁻¹",
        ))

    records = []
    for w in windows:
        d = asdict(w)
        # Serialise dicts/lists to strings for CSV round-trip friendliness.
        d["direction_distribution"] = "; ".join(
            f"{k}={v}" for k, v in w.direction_distribution.items()
        ) or "—"
        d["matrix_distribution"] = "; ".join(
            f"{k}={v}" for k, v in w.matrix_distribution.items()
        ) or "—"
        d["substrate_distribution"] = "; ".join(
            f"{k}={v}" for k, v in w.substrate_distribution.items()
        ) or "—"
        d["candidate_axes"] = "; ".join(w.candidate_axes) or "—"
        d["supporting_source_ids"] = "; ".join(w.supporting_source_ids) or "—"
        d["priority_tags"] = "; ".join(w.priority_tags) or "—"
        records.append(d)

    out = pd.DataFrame(records)
    if not out.empty:
        order_cls = {"anchor": 0, "secondary": 1, "ambiguous": 2}
        out["_ord"] = out["classification"].map(order_cls)
        out = out.sort_values(["_ord", "primary_axis", "start_cm1"]).drop(columns=["_ord"])
        out = out.reset_index(drop=True)
    return out
