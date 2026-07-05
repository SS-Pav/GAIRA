"""motif → 11-axis aggregation for gaira_base_2.

Bounded noisy-OR combiner at the motif → axis layer:

    axis_score = 1 − ∏ (1 − motif_weight × mapping_weight(axis))

with all terms clipped to [0, 1] before the product. Independent of
target axis name — mapping_weight is resolved per-axis-per-motif.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from gaira.base2.motif_engine import resolve_mapping_weight
from gaira.base2.schema import (
    AxisMapping,
    AxisScore,
    BIOLOGY_AXES_V11,
    MotifScore,
    MotifSpec,
)


def _noisy_or(weights: Iterable[float]) -> float:
    p = 1.0
    any_contrib = False
    for w in weights:
        wc = float(np.clip(w, 0.0, 1.0))
        if wc > 0:
            any_contrib = True
        p *= (1.0 - wc)
    if not any_contrib:
        return 0.0
    return float(1.0 - p)


def aggregate_to_11_axes(
    motif_scores: dict[str, MotifScore],
    motif_specs: dict[str, MotifSpec],
    mappings: dict[str, AxisMapping],
) -> list[AxisScore]:
    """Produce AxisScore for each of the 11 biology axes.

    For each axis:
      1. Collect motifs whose mapping includes this axis (mapping_weight > 0)
      2. For each contributing motif, per-axis weight = motif.self_weight × mapping_weight(axis)
      3. Apply bounded noisy-OR
    """
    out: list[AxisScore] = []
    for axis_id in BIOLOGY_AXES_V11:
        contrib_ids: list[str] = []
        core_terms: list[float] = []
        regime_terms: list[float] = []
        for mid, mscore in motif_scores.items():
            mapping = mappings.get(mid)
            if mapping is None:
                continue
            mw = resolve_mapping_weight(mapping, axis_id)
            if mw <= 0.0:
                continue
            contrib_ids.append(mid)
            core_terms.append(mscore.core_weight * mw)
            regime_terms.append(mscore.regime_weight * mw)
        out.append(AxisScore(
            axis_id=axis_id,
            core_evidence=_noisy_or(core_terms),
            regime_evidence=_noisy_or(regime_terms),
            contributing_motifs=tuple(contrib_ids),
        ))
    return out
