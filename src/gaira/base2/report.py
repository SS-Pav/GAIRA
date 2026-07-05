"""Report / top-level orchestration for gaira_base_2.

Takes a preprocessed spectrum on the canonical master axis and produces
a SpectrumResult containing motif scores, 11-axis scores, 8-axis
projection, and ambiguity lane.

This is the public API for one-spectrum inference.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from gaira.base2.ambiguity import compute_ambiguity_lane
from gaira.base2.axis_engine import aggregate_to_11_axes
from gaira.base2.motif_engine import compute_motif_score
from gaira.base2.projection import project_to_8_axes
from gaira.base2.registry import (
    load_active_registry,
)
from gaira.base2.schema import (
    AxisMapping,
    MotifDualStatus,
    MotifScore,
    MotifSpec,
    SpectrumResult,
)


def score_spectrum(
    spectrum: np.ndarray,
    master_x: np.ndarray,
    motifs: dict[str, MotifSpec],
    mappings: dict[str, AxisMapping],
    dual_status: dict[str, MotifDualStatus],
    spectrum_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> SpectrumResult:
    """Full gaira_base_2 scoring on a single spectrum."""
    motif_scores: dict[str, MotifScore] = {}
    for mid, spec in motifs.items():
        mapping = mappings.get(mid)
        status = dual_status.get(mid)
        motif_scores[mid] = compute_motif_score(
            spec, mapping, status, spectrum, master_x,
        )

    axis11 = aggregate_to_11_axes(motif_scores, motifs, mappings)
    axis8 = project_to_8_axes(axis11)
    ambiguity = compute_ambiguity_lane(motif_scores, mappings)

    return SpectrumResult(
        spectrum_id=spectrum_id,
        motif_scores=tuple(motif_scores.values()),
        axis11_scores=tuple(axis11),
        axis8_projection=tuple(axis8),
        ambiguity=ambiguity,
        metadata={
            "n_motifs_evaluated": len(motif_scores),
            "n_motifs_active_in_registry": len(motifs),
            **(metadata or {}),
        },
    )


def result_to_flat_dict(result: SpectrumResult) -> dict[str, float]:
    """Flatten a SpectrumResult into a flat column→value dict for CSV output."""
    row: dict[str, Any] = {"spectrum_id": result.spectrum_id}
    for m in result.motif_scores:
        row[f"motif_activation.{m.motif_id}"] = round(m.activation, 4)
        row[f"motif_core.{m.motif_id}"] = round(m.core_weight, 4)
        row[f"motif_regime.{m.motif_id}"] = round(m.regime_weight, 4)
    for a in result.axis11_scores:
        row[f"axis11_core.{a.axis_id}"] = round(a.core_evidence, 4)
        row[f"axis11_regime.{a.axis_id}"] = round(a.regime_evidence, 4)
    for a in result.axis8_projection:
        row[f"axis8_core_proj.{a.axis_id}"] = round(a.core_evidence, 4)
        row[f"axis8_regime_proj.{a.axis_id}"] = round(a.regime_evidence, 4)
    row["ambiguity_core"] = round(result.ambiguity.core_evidence, 4)
    row["ambiguity_regime"] = round(result.ambiguity.regime_evidence, 4)
    return row


def load_engine():
    """One-shot engine loader returning (motifs, mappings, dual_status).

    Convenience wrapper over ``registry.load_active_registry`` for
    callers that want a single handle.
    """
    return load_active_registry()
