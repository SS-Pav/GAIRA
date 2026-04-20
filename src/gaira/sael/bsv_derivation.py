"""Part D — Derive expected BSV from SAEL anchor-based deltas.

Produces SAELExpectedComparator objects:

  contrast_id
  label
  status                 direct | approximate | weak | unavailable
  overall_confidence     high | moderate | low | none
  expected_delta         {axis: "up" | "down" | "mixed" | "flat" | "unknown"}
  per_axis_confidence    {axis: "high" | "moderate" | "low"}
  anchor_support         {axis: [window_id, ...]}
  ambiguity_summary      str
  rationale              str
  provenance             [source_id, ...]

BSV axes that received NO anchor support are NOT silently filled with "flat"
— they remain "unknown". A downstream consumer that wants a complete 8-axis
vector must explicitly handle that, not assume the absence of evidence is
evidence of absence.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from gaira.expected.axis_mapping import BSV_AXES
from gaira.sael.delta_builder import SAELExpectedDelta


@dataclass
class SAELExpectedComparator:
    contrast_id: str
    label: str
    status: str
    overall_confidence: str
    expected_delta: dict[str, str]
    per_axis_confidence: dict[str, str]
    anchor_support: dict[str, list[str]]
    ambiguity_summary: str
    rationale: str
    provenance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _label_for(d: SAELExpectedDelta) -> str:
    return f"{d.condition_a} vs {d.condition_b} ({d.matrix}, SAEL v1)"


def derive_expected_comparators(
    deltas: list[SAELExpectedDelta],
) -> list[SAELExpectedComparator]:
    out: list[SAELExpectedComparator] = []
    for d in deltas:
        expected_delta: dict[str, str] = {ax: "unknown" for ax in BSV_AXES}
        per_axis_conf: dict[str, str] = {ax: "low" for ax in BSV_AXES}
        anchor_support: dict[str, list[str]] = {ax: [] for ax in BSV_AXES}

        for a in d.per_axis:
            if a.axis in expected_delta:
                expected_delta[a.axis] = a.direction
                per_axis_conf[a.axis] = a.confidence
                anchor_support[a.axis] = list(a.supporting_windows)

        out.append(SAELExpectedComparator(
            contrast_id=d.contrast_id,
            label=_label_for(d),
            status=d.status,
            overall_confidence=d.overall_confidence,
            expected_delta=expected_delta,
            per_axis_confidence=per_axis_conf,
            anchor_support=anchor_support,
            ambiguity_summary=d.ambiguity_summary,
            rationale=d.rationale,
            provenance=list(d.provenance),
        ))
    return out
