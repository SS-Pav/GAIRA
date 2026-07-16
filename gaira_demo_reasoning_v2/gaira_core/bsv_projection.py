"""GAIRA Demo v1 — motif/MSS → 11-axis BSV projection.

Projects motif fire scores (and a small MSS supporting contribution) into
the 11-axis biochemical state vector using a noisy-OR-style aggregation.
"""
from __future__ import annotations

import numpy as np

from . import config as cfg
from .motif_scoring import MOTIFS
from .mss_scoring import molecule_axis_contributions


def _noisy_or(scores: list[float]) -> float:
    """noisy-OR: 1 - prod(1 - clip(s, 0, 1))."""
    if not scores:
        return 0.0
    p = 1.0
    for s in scores:
        p *= (1.0 - float(np.clip(s, 0.0, 1.0)))
    return float(1.0 - p)


def project_to_bsv(motif_scores: dict[str, float],
                     mss_fires: dict | None = None,
                     *, mss_weight: float = 0.25) -> dict[str, float]:
    """Aggregate motif scores into 11-axis BSV.

    mss_fires (optional): dict[mol_id -> MSSFire]. Each contributes through
    its `molecule_axis_contributions` profile, scaled by mss_weight.
    """
    bsv = {a: 0.0 for a in cfg.BSV_AXES}

    # Motif → axis (primary axis only; this is intentionally simple)
    per_axis_motifs: dict[str, list[float]] = {a: [] for a in cfg.BSV_AXES}
    for m in MOTIFS:
        s = float(motif_scores.get(m.motif_id, 0.0))
        if s > 0:
            per_axis_motifs[m.primary_axis].append(s)

    for axis, scores in per_axis_motifs.items():
        bsv[axis] = _noisy_or(scores)

    # MSS supporting evidence — light additional push
    if mss_fires:
        for mol_id, fire in mss_fires.items():
            f = getattr(fire, "fire", 0.0)
            if f <= 0:
                continue
            contrib = molecule_axis_contributions(mol_id)
            for axis, w in contrib.items():
                bsv[axis] = min(1.0, bsv[axis] + mss_weight * f * w)

    return bsv


def delta_bsv(reference: dict[str, float], condition: dict[str, float]) -> dict[str, float]:
    return {a: float(condition.get(a, 0.0) - reference.get(a, 0.0)) for a in cfg.BSV_AXES}


def top_axes(bsv: dict[str, float], k: int = 3, mode: str = "abs") -> list[dict]:
    """Return top-k axes ranked by absolute (mode='abs') or signed value."""
    items = list(bsv.items())
    if mode == "abs":
        items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    else:
        items.sort(key=lambda kv: kv[1], reverse=True)
    out = []
    for axis, val in items[:k]:
        direction = "increased" if val > 0 else ("decreased" if val < 0 else "flat")
        # Heuristic evidence strength
        m = abs(val)
        if m >= 0.30:
            strength = "high"
        elif m >= 0.12:
            strength = "moderate"
        else:
            strength = "low"
        out.append({"axis": axis, "value": float(val),
                     "direction": direction, "evidence_strength": strength})
    return out
