"""Ambiguity / artifact control-lane for gaira_base_2.

Receives contributions from:
  - AMBIGUITY_ONLY-mapped motifs
  - AMBIGUITY_MOTIF / ARTIFACT_MOTIF typed motifs
  - motifs whose CROSS_AXIS mapping lists "ambiguity_artifact" as a target

Uses the same bounded noisy-OR combiner as the biology axes (motifs
contributing to the ambiguity lane are treated as independent evidence
sources of ambiguity).

The ambiguity lane does NOT flow into biology axis scores. It is a
parallel reporting channel.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from gaira.base2.schema import (
    AmbiguityLane,
    AxisMapping,
    MotifScore,
)


def _noisy_or(weights: Iterable[float]) -> float:
    p = 1.0
    any_contrib = False
    for w in weights:
        wc = float(np.clip(w, 0.0, 1.0))
        if wc > 0:
            any_contrib = True
        p *= (1.0 - wc)
    return 0.0 if not any_contrib else float(1.0 - p)


def compute_ambiguity_lane(
    motif_scores: dict[str, MotifScore],
    mappings: dict[str, AxisMapping],
) -> AmbiguityLane:
    contrib_ids: list[str] = []
    core_terms: list[float] = []
    regime_terms: list[float] = []
    for mid, mscore in motif_scores.items():
        if not mscore.contributes_to_ambiguity:
            continue
        mapping = mappings.get(mid)
        if mapping is None:
            continue
        # Effective mapping weight for ambiguity lane:
        #   AMBIGUITY_ONLY → 1.0
        #   CROSS_AXIS listing ambiguity_artifact → 0.70
        #   AMBIGUITY_MOTIF / ARTIFACT_MOTIF motif_type → 1.0
        if mapping.mapping_type == "AMBIGUITY_ONLY":
            mw = 1.0
        elif "ambiguity_artifact" in (mapping.primary_axis, *mapping.secondary_axes):
            mw = 0.70
        else:
            # typed motif but no mapping entry → conservative 0.70
            mw = 0.70
        contrib_ids.append(mid)
        core_terms.append(mscore.core_weight * mw)
        regime_terms.append(mscore.regime_weight * mw)
    return AmbiguityLane(
        core_evidence=_noisy_or(core_terms),
        regime_evidence=_noisy_or(regime_terms),
        contributing_motifs=tuple(contrib_ids),
    )
