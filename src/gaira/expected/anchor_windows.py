"""Anchor-window extraction from peak_assignments.

For each BSV axis, cluster peak_cm values into candidate windows and score:
  - source count (distinct papers supporting the window)
  - molecule count (distinct molecules)
  - locality score (median width / tolerance ratio)
  - anchor-hint match (does the cluster intersect a canonical anchor range?)
  - ambiguity score (how much the cluster cm range overlaps peaks on OTHER
    axes — the higher, the more ambiguous)

Classification:
  - anchor:    ≥ 3 sources AND ≥ 2 molecules AND ambiguity_score ≤ 0.4
  - secondary: ≥ 2 sources, OR anchor-quality range but thinner evidence
  - ambiguous: ambiguity_score > 0.4 or strong cross-axis peak overlap

Clustering is intentionally simple: sort peak_cm within axis, split when two
consecutive values differ by more than `GAP_CM` (default 25 cm⁻¹), and treat
the resulting groups as candidate windows. Widened by a small pad so they
match the 22-window panel's granularity without claiming false precision.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from gaira.expected.axis_mapping import (
    AXIS_ANCHOR_HINTS, BSV_AXES, assigned_row_to_axis,
)


DEFAULT_DB_PATH = Path("/Volumes/SSD_Rad/GAIRA_DATA/interim/gaira.duckdb")

GAP_CM = 25.0          # split clusters when two peaks differ by more than this
PAD_CM = 5.0           # expand cluster edges by this much
MIN_SOURCES_ANCHOR = 3
MIN_MOLECULES_ANCHOR = 2
MAX_AMBIG_ANCHOR = 0.4


@dataclass
class AnchorWindow:
    axis: str
    start_cm: float
    end_cm: float
    width_cm: float
    peak_cm_median: float
    n_peak_rows: int
    n_sources: int
    n_molecules: int
    top_molecules: str
    share_high_plus_medium_conf: float
    anchor_hint_match: str             # empty | "matched <hint>"
    ambiguity_score: float              # [0,1]; share of peaks from other axes inside the window
    classification: str                 # "anchor" | "secondary" | "ambiguous"
    contrast_support_note: str          # free-form note about conditions that show a delta here
    notes: str


def _load_peak_assignments(db_path: Path) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute("""
            SELECT assignment_id, source_id, peak_cm, tolerance_cm,
                   assigned_molecule, assigned_group,
                   matrix_context, confidence_text, evidence_text
            FROM peak_assignments
            WHERE peak_cm IS NOT NULL
        """).df()
    df["axis"] = df.apply(
        lambda r: assigned_row_to_axis(
            r["assigned_group"], r["assigned_molecule"], r["evidence_text"],
        ),
        axis=1,
    )
    return df


def _cluster_peaks(peaks: list[float], gap: float = GAP_CM) -> list[list[int]]:
    """Return list of clusters, each a list of original indices."""
    if not peaks:
        return []
    order = sorted(range(len(peaks)), key=lambda i: peaks[i])
    clusters: list[list[int]] = []
    current: list[int] = [order[0]]
    for k in order[1:]:
        prev_peak = peaks[current[-1]]
        if peaks[k] - prev_peak > gap:
            clusters.append(current)
            current = [k]
        else:
            current.append(k)
    clusters.append(current)
    return clusters


def _anchor_hint_for(axis: str, start: float, end: float) -> str:
    for lo, hi, label in AXIS_ANCHOR_HINTS.get(axis, []):
        if end >= lo and start <= hi:
            return f"matched {lo}–{hi}: {label}"
    return ""


def _contrast_note(axis: str, diff_df: pd.DataFrame) -> str:
    rows = diff_df[diff_df["component"] == axis]
    if rows.empty:
        return ""
    up = rows[rows["direction"] == "up"]["condition"].tolist()
    dn = rows[rows["direction"] == "down"]["condition"].tolist()
    parts = []
    if up:
        parts.append("↑ in " + ", ".join(up))
    if dn:
        parts.append("↓ in " + ", ".join(dn))
    return "; ".join(parts) if parts else ""


def _ambiguity_score(
    start: float, end: float, axis: str, all_peaks: pd.DataFrame,
) -> float:
    """Fraction of peaks in [start, end] that are attributed to a DIFFERENT axis."""
    band = all_peaks[(all_peaks["peak_cm"] >= start) & (all_peaks["peak_cm"] <= end)]
    if band.empty:
        return 0.0
    other = band[(band["axis"].notna()) & (band["axis"] != axis)]
    return round(len(other) / len(band), 3)


def _classify(
    n_sources: int, n_molecules: int, ambiguity: float, hint_matched: bool,
) -> str:
    if n_sources >= MIN_SOURCES_ANCHOR and n_molecules >= MIN_MOLECULES_ANCHOR \
            and ambiguity <= MAX_AMBIG_ANCHOR:
        return "anchor"
    if ambiguity > MAX_AMBIG_ANCHOR:
        return "ambiguous"
    if n_sources >= 2 or hint_matched:
        return "secondary"
    return "ambiguous"


def build_anchor_window_registry(
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    peaks = _load_peak_assignments(db_path)

    # Load differential profile for contrast notes.
    p = Path(__file__).resolve().parents[3] / "outputs" / "landscape_v4" / "condition_differential_profile.csv"
    diff_df = pd.read_csv(p) if p.exists() else pd.DataFrame(
        columns=["condition", "component", "direction"],
    )

    windows: list[AnchorWindow] = []

    for axis in BSV_AXES:
        axis_df = peaks[peaks["axis"] == axis].reset_index(drop=True)
        if axis_df.empty:
            continue

        clusters = _cluster_peaks(axis_df["peak_cm"].tolist(), gap=GAP_CM)
        for cluster_idx in clusters:
            sub = axis_df.iloc[cluster_idx]
            start = float(sub["peak_cm"].min() - PAD_CM)
            end = float(sub["peak_cm"].max() + PAD_CM)
            width = round(end - start, 1)
            pk_med = round(float(sub["peak_cm"].median()), 1)
            n_peak = len(sub)
            n_src = int(sub["source_id"].nunique())
            n_mol = int(sub["assigned_molecule"].nunique())
            top_mol = "; ".join(sub["assigned_molecule"].value_counts().head(3).index.tolist())
            conf = sub["confidence_text"].fillna("unknown").value_counts(normalize=True)
            share_hm = round(float(conf.get("high", 0.0) + conf.get("medium", 0.0)), 3)
            hint = _anchor_hint_for(axis, start, end)
            ambig = _ambiguity_score(start, end, axis, peaks)
            cls = _classify(n_src, n_mol, ambig, bool(hint))
            windows.append(AnchorWindow(
                axis=axis,
                start_cm=round(start, 1),
                end_cm=round(end, 1),
                width_cm=width,
                peak_cm_median=pk_med,
                n_peak_rows=n_peak,
                n_sources=n_src,
                n_molecules=n_mol,
                top_molecules=top_mol,
                share_high_plus_medium_conf=share_hm,
                anchor_hint_match=hint,
                ambiguity_score=ambig,
                classification=cls,
                contrast_support_note=_contrast_note(axis, diff_df),
                notes="",
            ))

    df = pd.DataFrame([asdict(w) for w in windows])
    # Sort so anchors come first per axis, then secondary, then ambiguous.
    order = {"anchor": 0, "secondary": 1, "ambiguous": 2}
    df["_order"] = df["classification"].map(order)
    df = df.sort_values(["axis", "_order", "n_sources"], ascending=[True, True, False])
    df = df.drop(columns=["_order"]).reset_index(drop=True)
    return df
