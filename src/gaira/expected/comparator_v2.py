"""Ambiguity-aware expected comparator builder (v2).

Replaces the broad support-weighted axis-value lookup with a comparator object
that carries per-axis confidence, anchor windows, and ambiguity notes. It does
NOT fit to spectra. It produces a literature-side object that downstream
consumers (e.g. calibration eval, spectral query) can compare against.

Typical call:
    comp = build_expected_comparator_v2("hcc_vs_healthy_serum", ...)
    comp.expected_delta              # {axis: "up" | "down" | ...}
    comp.per_axis_confidence         # {axis: "high" | "moderate" | "low"}
    comp.anchor_windows_used         # {axis: [(start, end), ...]}
    comp.ambiguity_summary           # str
    comp.rationale                    # str

Status semantics:
    "direct"       — contrast-specific row exists in literature evidence
    "approximate"  — contrast inferred from analyte fingerprint, not direct
    "unavailable"  — no literature support; downstream should not treat this
                     as a real comparator
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from gaira.expected.anchor_windows import build_anchor_window_registry
from gaira.expected.delta_objects import (
    CONTRAST_REGISTRY, ExpectedDelta, build_expected_delta_objects,
)
from gaira.expected.axis_audit import DEFAULT_DB_PATH
import duckdb


@dataclass
class ExpectedComparatorV2:
    contrast_id: str
    label: str
    status: str                              # direct | approximate | unavailable
    overall_confidence: str                  # high | moderate | low | none
    expected_delta: dict[str, str]           # axis → direction
    per_axis_confidence: dict[str, str]      # axis → confidence
    anchor_windows_used: dict[str, list[tuple[float, float]]]
    ambiguity_summary: str
    rationale: str
    provenance: list[str] = field(default_factory=list)

    # Optional signed delta vector (only when it can be defended — e.g. for
    # disease contrasts where landscape-v4 already encodes a numeric delta).
    signed_delta_vector: dict[str, float] | None = None

    def to_dict(self) -> dict:
        d = {
            "contrast_id": self.contrast_id,
            "label": self.label,
            "status": self.status,
            "overall_confidence": self.overall_confidence,
            "expected_delta": dict(self.expected_delta),
            "per_axis_confidence": dict(self.per_axis_confidence),
            "anchor_windows_used": {
                k: [list(w) for w in v] for k, v in self.anchor_windows_used.items()
            },
            "ambiguity_summary": self.ambiguity_summary,
            "rationale": self.rationale,
            "provenance": list(self.provenance),
        }
        if self.signed_delta_vector is not None:
            d["signed_delta_vector"] = dict(self.signed_delta_vector)
        return d


def _load_peaks(db_path=DEFAULT_DB_PATH) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute(
            "SELECT * FROM peak_assignments WHERE peak_cm IS NOT NULL"
        ).df()


def _load_signed_delta_vector(contrast_id: str, condition_a: str) -> dict[str, float] | None:
    """Only disease contrasts have a numeric landscape-v4 delta. Everything
    else returns None so downstream doesn't treat approximate contrasts as
    precise numeric comparators."""
    from pathlib import Path
    p = Path(__file__).resolve().parents[3] / "outputs" / "landscape_v4" / "condition_differential_profile.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    sub = df[df["condition"] == condition_a]
    if sub.empty:
        return None
    return {row["component"]: float(row["delta"]) for _, row in sub.iterrows()}


def _label_for(spec: dict) -> str:
    a, b = spec["a"], spec["b"]
    return f"{a} vs {b} ({spec['matrix']})"


def _to_comparator(delta: ExpectedDelta, spec: dict) -> ExpectedComparatorV2:
    expected_delta = {a.axis: a.direction for a in delta.expected_axes}
    per_axis_conf = {a.axis: a.confidence for a in delta.expected_axes}
    anchors = {a.axis: list(a.anchor_windows) for a in delta.expected_axes}
    signed = None
    if delta.status == "direct":
        signed = _load_signed_delta_vector(delta.contrast_id, delta.condition_a)

    rationale_parts = [delta.rationale]
    if signed is not None:
        rationale_parts.append(
            "Signed delta vector attached (landscape v4 numeric delta)."
        )
    rationale = " ".join(p for p in rationale_parts if p)

    return ExpectedComparatorV2(
        contrast_id=delta.contrast_id,
        label=_label_for(spec),
        status=delta.status,
        overall_confidence=delta.overall_confidence,
        expected_delta=expected_delta,
        per_axis_confidence=per_axis_conf,
        anchor_windows_used=anchors,
        ambiguity_summary=delta.ambiguity_summary,
        rationale=rationale,
        provenance=list(delta.provenance),
        signed_delta_vector=signed,
    )


def build_expected_comparator_v2(
    contrast_id: str,
    anchor_df: pd.DataFrame | None = None,
    peaks_df: pd.DataFrame | None = None,
) -> ExpectedComparatorV2:
    """Build one comparator by contrast_id.

    Passing pre-computed anchor_df / peaks_df avoids re-running the
    extraction when building many comparators in a batch.
    """
    specs_by_id = {s["id"]: s for s in CONTRAST_REGISTRY}
    if contrast_id not in specs_by_id:
        raise KeyError(
            f"Unknown contrast_id '{contrast_id}'. "
            f"Registered: {sorted(specs_by_id)}"
        )

    if anchor_df is None:
        anchor_df = build_anchor_window_registry()
    if peaks_df is None:
        peaks_df = _load_peaks()

    deltas = build_expected_delta_objects(anchor_df, peaks_df)
    delta = next(d for d in deltas if d.contrast_id == contrast_id)
    return _to_comparator(delta, specs_by_id[contrast_id])


def build_all_expected_comparators_v2(
    anchor_df: pd.DataFrame | None = None,
    peaks_df: pd.DataFrame | None = None,
) -> list[ExpectedComparatorV2]:
    if anchor_df is None:
        anchor_df = build_anchor_window_registry()
    if peaks_df is None:
        peaks_df = _load_peaks()
    specs_by_id = {s["id"]: s for s in CONTRAST_REGISTRY}
    deltas = build_expected_delta_objects(anchor_df, peaks_df)
    return [_to_comparator(d, specs_by_id[d.contrast_id]) for d in deltas]
