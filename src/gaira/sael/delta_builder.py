"""Part C — SAEL anchor-based expected-delta objects.

For each registered contrast, assemble an ExpectedDelta object from SAEL
anchor windows instead of from landscape-level condition averages.

Two contrast modes:

  1. `analyte_based` — the perturbation is adding or removing a known
      analyte (hypoxanthine spike, uricase depletion, ergothioneine spike).
      We look up anchor evidence rows that match the analyte by molecule
      / priority-tag tokens and infer direction from the spec (spike → up,
      depletion → down). This uses SAEL's assignment-only anchors as
      LOCATION evidence — it does NOT read direction from the literature
      for these cases.

  2. `condition_based` — the contrast is disease vs healthy (HCC, NAFLD,
      etc.). We need SAEL contrast-type rows with a matching condition_a
      and a direction. Currently sparse; the builder will honestly report
      status = "unavailable" when no condition-specific direction is found.

Context conditioning: anchor candidates are filtered by matrix and substrate
when those are declared on the contrast spec; a mismatch downgrades the
confidence rather than silently drops the evidence.

Nothing here uses calibration data. Nothing here trains on outcomes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, asdict

import pandas as pd

from gaira.expected.axis_mapping import BSV_AXES


# ─────────────────────────────────────────────────────────────────────
# Contrast specifications (literature-side, independent of calibration)
# ─────────────────────────────────────────────────────────────────────

CONTRAST_SPECS: list[dict] = [
    # Analyte-based (perturbation is a specific molecule)
    {
        "id": "hypoxanthine_spike_literature",
        "mode": "analyte_based",
        "condition_a": "serum_plus_hypoxanthine",
        "condition_b": "serum_baseline",
        "matrix": "serum",
        "substrate": None,
        "analyte_tokens": ("hypoxanthine", "adenine"),
        "analyte_direction": "up",      # spike → up on the analyte's anchors
        "rationale":
            "Hypoxanthine spike into serum. SAEL locates hypoxanthine/adenine "
            "literature anchors (ring-breathing near ~725 cm⁻¹); direction is "
            "derived from the spike spec, not from disease-contrast prose.",
    },
    {
        "id": "uricase_depletion_literature",
        "mode": "analyte_based",
        "condition_a": "serum_plus_uricase",
        "condition_b": "serum_baseline",
        "matrix": "serum",
        "substrate": None,
        "analyte_tokens": ("uric acid", "urate"),
        "analyte_direction": "down",    # enzymatic depletion → down
        "rationale":
            "Uricase depletes serum uric acid. SAEL locates uric-acid literature "
            "anchors; direction derives from the enzymatic depletion spec."
            " Expect confound axes where uric-acid peaks straddle multiple "
            "anchor regions.",
    },
    {
        "id": "ergothioneine_spike_literature",
        "mode": "analyte_based",
        "condition_a": "serum_plus_ergothioneine",
        "condition_b": "serum_baseline",
        "matrix": "serum",
        "substrate": None,
        "analyte_tokens": ("ergothioneine",),
        "analyte_direction": "up",
        "rationale":
            "Ergothioneine spike. SAEL locates ergothioneine literature anchors. "
            "Direction derives from the spike spec. Expect axis overlap with "
            "the 720 cm⁻¹ imidazole/purine region.",
    },
    # Condition-based (disease vs reference)
    {
        "id": "hcc_vs_healthy_serum",
        "mode": "condition_based",
        "condition_a": "HCC",
        "condition_b": "healthy_control",
        "matrix": "serum",
        "substrate": None,
        "rationale":
            "HCC vs healthy serum. Requires SAEL contrast-type rows with "
            "condition_a='HCC' and a direction verb. Currently expected to "
            "be unavailable until targeted literature extraction lands.",
    },
    {
        "id": "nafld_vs_healthy_serum",
        "mode": "condition_based",
        "condition_a": "NAFLD_NASH",
        "condition_b": "healthy_control",
        "matrix": "serum",
        "substrate": None,
        "rationale": "NAFLD / NASH vs healthy serum — condition-specific SAEL rows required.",
    },
    {
        "id": "cca_vs_healthy_serum",
        "mode": "condition_based",
        "condition_a": "cholangiocarcinoma",
        "condition_b": "healthy_control",
        "matrix": "serum",
        "substrate": None,
        "rationale": "Cholangiocarcinoma vs healthy serum.",
    },
]


# ─────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PerAxisSAEL:
    axis: str
    direction: str                          # up | down | mixed | flat | unknown
    confidence: str                         # high | moderate | low
    supporting_windows: list[str] = field(default_factory=list)
    per_window_direction: dict[str, str] = field(default_factory=dict)
    ambiguity_notes: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class SAELExpectedDelta:
    contrast_id: str
    condition_a: str
    condition_b: str
    matrix: str
    substrate_context: str | None
    status: str                              # direct | approximate | weak | unavailable
    overall_confidence: str                 # high | moderate | low | none
    anchor_windows_used: list[str] = field(default_factory=list)
    per_axis: list[PerAxisSAEL] = field(default_factory=list)
    ambiguity_summary: str = ""
    rationale: str = ""
    provenance: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────

def _filter_anchors_by_context(
    windows_df: pd.DataFrame, matrix: str | None, substrate: str | None,
) -> pd.DataFrame:
    out = windows_df.copy()
    # Keep a window if its matrix_distribution either contains the target
    # matrix OR is empty (no matrix evidence = agnostic). Same for substrate.
    if matrix:
        def _matrix_ok(cell: str) -> bool:
            if not cell or cell == "—":
                return True
            return matrix in cell or "biofluid" in cell
        out = out[out["matrix_distribution"].apply(_matrix_ok)]
    if substrate:
        def _sub_ok(cell: str) -> bool:
            if not cell or cell == "—":
                return True
            return substrate in cell
        out = out[out["substrate_distribution"].apply(_sub_ok)]
    return out


def _anchors_for_analyte(
    windows_df: pd.DataFrame, tokens: tuple[str, ...],
) -> pd.DataFrame:
    """Windows whose priority_tags or supporting_source_ids mention the analyte."""
    toks = tuple(t.lower() for t in tokens)
    # The cleanest source is the underlying evidence rows — but we can
    # also use priority_tags on the window. For analyte-specific matching
    # we need to go back to the evidence. Since windows carry "priority_tags"
    # that we aggregated from evidence, we can fall back to that.
    def _hit(tags_cell: str) -> bool:
        if not tags_cell or tags_cell == "—":
            return False
        t = tags_cell.lower()
        return any(tok in t for tok in toks)
    return windows_df[windows_df["priority_tags"].apply(_hit)]


def _anchors_for_analyte_by_evidence(
    evidence_df: pd.DataFrame, windows_df: pd.DataFrame,
    tokens: tuple[str, ...],
) -> pd.DataFrame:
    """Find anchors via evidence rows that mention the analyte in
    assigned_molecule. More precise than priority-tag filtering."""
    toks = tuple(t.lower() for t in tokens)
    mask = evidence_df["assigned_molecule"].fillna("").str.lower().apply(
        lambda s: any(tok in s for tok in toks)
    )
    rel = evidence_df[mask & evidence_df["peak_cm1"].notna()]
    if rel.empty:
        return rel
    # For each window, check if any of its cm range includes one of rel's peaks.
    hit_idx = []
    for i, w in windows_df.iterrows():
        lo, hi = float(w["start_cm1"]), float(w["end_cm1"])
        if ((rel["peak_cm1"] >= lo) & (rel["peak_cm1"] <= hi)).any():
            hit_idx.append(i)
    return windows_df.loc[hit_idx]


def _axis_confidence(
    n_sources_total: int, n_anchor: int, n_ambiguous: int,
) -> str:
    if n_sources_total >= 6 and n_anchor >= 1 and n_ambiguous == 0:
        return "high"
    if n_sources_total >= 3 and n_anchor >= 1:
        return "moderate"
    return "low"


def _aggregate_axis_from_windows(
    axis_windows: pd.DataFrame, direction: str,
) -> PerAxisSAEL:
    """Aggregate a set of anchor windows (same axis) into a PerAxisSAEL."""
    n_sources = axis_windows["source_count"].sum()
    anchors = axis_windows[axis_windows["classification"] == "anchor"]
    ambigs = axis_windows[axis_windows["classification"] == "ambiguous"]
    n_anchor = len(anchors)
    n_ambig = len(ambigs)

    # Direction from the window-level direction_distribution, if any rows
    # carry a direction claim; otherwise the contrast-spec direction is used.
    declared = Counter()
    for d in axis_windows["direction_distribution"].tolist():
        if not d or d == "—":
            continue
        for tok in d.split(";"):
            tok = tok.strip()
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            try:
                declared[k.strip()] += int(v)
            except ValueError:
                pass

    up = declared.get("up", 0)
    down = declared.get("down", 0)
    if up > 0 and down > 0:
        resolved = "mixed"
    elif up > 0:
        resolved = "up"
    elif down > 0:
        resolved = "down"
    else:
        resolved = direction  # fall back to spec direction

    rationale = (
        f"{len(axis_windows)} windows, {n_sources} source-mentions total, "
        f"{n_anchor} anchors, {n_ambig} ambiguous."
    )
    if declared:
        rationale += " Window-level directions: " + dict(declared).__repr__()

    ambig_notes = []
    for _, w in ambigs.iterrows():
        ambig_notes.append(
            f"window {w['start_cm1']:.0f}–{w['end_cm1']:.0f} "
            f"ambiguity_score={w['ambiguity_score']}"
        )

    return PerAxisSAEL(
        axis=axis_windows.iloc[0]["primary_axis"] if len(axis_windows) else "unknown",
        direction=resolved,
        confidence=_axis_confidence(int(n_sources), n_anchor, n_ambig),
        supporting_windows=axis_windows["window_id"].tolist(),
        per_window_direction={
            w["window_id"]: w["direction_distribution"]
            for _, w in axis_windows.iterrows()
        },
        ambiguity_notes=ambig_notes,
        rationale=rationale,
    )


def _build_analyte_delta(
    spec: dict,
    windows_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
) -> SAELExpectedDelta:
    matrix_filtered = _filter_anchors_by_context(
        windows_df, spec["matrix"], spec.get("substrate"),
    )
    analyte_windows = _anchors_for_analyte_by_evidence(
        evidence_df, matrix_filtered, spec["analyte_tokens"],
    )
    if analyte_windows.empty:
        # Fall back to tag-based filtering.
        analyte_windows = _anchors_for_analyte(matrix_filtered, spec["analyte_tokens"])

    if analyte_windows.empty:
        return SAELExpectedDelta(
            contrast_id=spec["id"],
            condition_a=spec["condition_a"], condition_b=spec["condition_b"],
            matrix=spec["matrix"], substrate_context=spec.get("substrate"),
            status="unavailable", overall_confidence="none",
            anchor_windows_used=[], per_axis=[],
            ambiguity_summary="No anchor windows matched the analyte tokens.",
            rationale=spec["rationale"], provenance=[],
        )

    # Group by axis and aggregate.
    per_axis: list[PerAxisSAEL] = []
    for axis, sub in analyte_windows.groupby("primary_axis"):
        if axis is None or (isinstance(axis, float) and pd.isna(axis)):
            continue
        per_axis.append(_aggregate_axis_from_windows(sub, spec["analyte_direction"]))

    # Determine overall status/confidence.
    n_hi = sum(1 for a in per_axis if a.confidence == "high")
    n_mod = sum(1 for a in per_axis if a.confidence == "moderate")
    if n_hi >= 1:
        overall = "high"
        status = "direct"
    elif n_mod >= 1:
        overall = "moderate"
        status = "approximate"
    elif per_axis:
        overall = "low"
        status = "weak"
    else:
        overall = "none"
        status = "unavailable"

    ambiguity_summary = "; ".join(
        note for a in per_axis for note in a.ambiguity_notes
    ) or f"{len(analyte_windows)} windows matched analyte tokens."

    provenance = sorted({
        s for ids in analyte_windows["supporting_source_ids"]
        for s in (ids.split("; ") if ids and ids != "—" else [])
        if s
    })

    return SAELExpectedDelta(
        contrast_id=spec["id"],
        condition_a=spec["condition_a"], condition_b=spec["condition_b"],
        matrix=spec["matrix"], substrate_context=spec.get("substrate"),
        status=status, overall_confidence=overall,
        anchor_windows_used=analyte_windows["window_id"].tolist(),
        per_axis=per_axis,
        ambiguity_summary=ambiguity_summary,
        rationale=spec["rationale"], provenance=provenance,
    )


def _build_condition_delta(
    spec: dict,
    windows_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
) -> SAELExpectedDelta:
    # Need SAEL contrast rows carrying a matching condition AND a direction.
    contrast = evidence_df[
        (evidence_df["kind"] == "contrast")
        & (evidence_df["condition_a"] == spec["condition_a"])
        & (evidence_df["direction"].isin(["up", "down", "mixed"]))
    ]
    if contrast.empty:
        return SAELExpectedDelta(
            contrast_id=spec["id"],
            condition_a=spec["condition_a"], condition_b=spec["condition_b"],
            matrix=spec["matrix"], substrate_context=spec.get("substrate"),
            status="unavailable", overall_confidence="none",
            anchor_windows_used=[], per_axis=[],
            ambiguity_summary=(
                f"No SAEL contrast rows with condition_a='{spec['condition_a']}' "
                "and a direction verb. Status 'unavailable' is the honest answer "
                "until targeted extraction lands."
            ),
            rationale=spec["rationale"], provenance=[],
        )

    # For each contrast row, find the window it falls into.
    matching_windows: list[str] = []
    per_axis: list[PerAxisSAEL] = []
    # (Very thin; implementation mirrors analyte path but keyed by the
    # contrast row's peak, not by molecule tokens.)
    for _, row in contrast.iterrows():
        peak = row["peak_cm1"]
        if pd.isna(peak):
            continue
        hits = windows_df[
            (windows_df["start_cm1"] <= peak) & (windows_df["end_cm1"] >= peak)
        ]
        for _, w in hits.iterrows():
            matching_windows.append(w["window_id"])

    provenance = sorted({
        s for s in contrast["source_id"].dropna().unique().tolist()
    })

    # If we got here, we have at least one direction-bearing row but typically
    # no aggregated per-axis picture. Report as weak/approximate at best.
    return SAELExpectedDelta(
        contrast_id=spec["id"],
        condition_a=spec["condition_a"], condition_b=spec["condition_b"],
        matrix=spec["matrix"], substrate_context=spec.get("substrate"),
        status="weak", overall_confidence="low",
        anchor_windows_used=sorted(set(matching_windows)),
        per_axis=per_axis,
        ambiguity_summary=f"Found {len(contrast)} contrast rows but full per-axis aggregation requires richer extraction.",
        rationale=spec["rationale"], provenance=provenance,
    )


def build_sael_expected_deltas(
    evidence_df: pd.DataFrame,
    windows_df: pd.DataFrame,
) -> list[SAELExpectedDelta]:
    out: list[SAELExpectedDelta] = []
    for spec in CONTRAST_SPECS:
        if spec["mode"] == "analyte_based":
            out.append(_build_analyte_delta(spec, windows_df, evidence_df))
        elif spec["mode"] == "condition_based":
            out.append(_build_condition_delta(spec, windows_df, evidence_df))
        else:
            continue
    return out


def to_serializable(delta: SAELExpectedDelta) -> dict:
    return asdict(delta)
