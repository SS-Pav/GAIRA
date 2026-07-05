"""11-axis → 8-axis projection for gaira_base_2.

MAX combiner (locked per scoring pressure test §5) — conservative against
double-counting biology-related axes (purine_nucleotide + purine_metabolite
share pathway; they are NOT independent evidence sources at the projection
layer).
"""
from __future__ import annotations

from gaira.base2.schema import (
    AxisScore,
    GAIRA_BASE_AXES_V8,
    PROJECTION_V11_TO_V8,
)


def project_to_8_axes(
    axis11_scores: list[AxisScore],
) -> list[AxisScore]:
    """Project 11 biology axis scores onto the 8 gaira_base axes via MAX.

    The projection map is locked in schema.PROJECTION_V11_TO_V8.
    """
    by_id = {a.axis_id: a for a in axis11_scores}
    out: list[AxisScore] = []
    for axis8 in GAIRA_BASE_AXES_V8:
        contributing_11 = PROJECTION_V11_TO_V8[axis8]
        scores11 = [by_id[a] for a in contributing_11 if a in by_id]
        if not scores11:
            out.append(AxisScore(
                axis_id=axis8,
                core_evidence=0.0,
                regime_evidence=0.0,
                contributing_motifs=(),
            ))
            continue
        core_max = max(s.core_evidence for s in scores11)
        regime_max = max(s.regime_evidence for s in scores11)
        # union of contributing motifs across the mapped 11-axes
        contrib = tuple(
            sorted({m for s in scores11 for m in s.contributing_motifs})
        )
        out.append(AxisScore(
            axis_id=axis8,
            core_evidence=float(core_max),
            regime_evidence=float(regime_max),
            contributing_motifs=contrib,
        ))
    return out
