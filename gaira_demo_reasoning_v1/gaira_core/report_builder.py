"""GAIRA Demo v1 — GAIRAReport builder (end-to-end).

Orchestrates preprocessing → primitives → MSS → motifs → substrate →
BSV → evidence → caveats and packages the result into a single dict
that downstream UI tabs render.
"""
from __future__ import annotations

import numpy as np

from . import config as cfg
from .bsv_projection import project_to_bsv, delta_bsv, top_axes
from .evidence_synthesis import (
    synthesize_evidence, synthesize_caveats, overall_confidence,
)
from .mss_scoring import score_all
from .motif_scoring import score_motifs, MOTIFS
from .preprocessing import preprocess
from .primitive_extraction import primitives_from
from .substrate_physics import apply_substrate_corrections


def build_report(
    *, sample_id: str, title: str,
    domain: str, substrate: str,
    wavenumber: np.ndarray, intensity: np.ndarray,
    reference_bsv: dict[str, float] | None = None,
) -> dict:
    pp = preprocess(wavenumber, intensity)
    prims = primitives_from(pp["wavenumber"], pp["processed_intensity"])

    mss_fires = score_all(pp["wavenumber"], pp["processed_intensity"])
    motif_scores_raw = score_motifs(pp["wavenumber"], pp["processed_intensity"])
    motif_scores, sub_events = apply_substrate_corrections(motif_scores_raw, substrate=substrate)

    bsv = project_to_bsv(motif_scores, mss_fires=mss_fires)

    # Anchors / supports / anti-evidence for display
    # Use the motifs that fired ≥ 0.05 as anchors (their bands)
    anchors, supports = [], []
    for m in MOTIFS:
        s = float(motif_scores.get(m.motif_id, 0.0))
        if s >= 0.05:
            for lo, hi in m.bands:
                anchors.append({"motif_id": m.motif_id, "low": lo, "high": hi, "score": s})
        elif s > 0.01:
            for lo, hi in m.bands:
                supports.append({"motif_id": m.motif_id, "low": lo, "high": hi, "score": s})

    evidence = synthesize_evidence(motif_scores, bsv, substrate=substrate)
    caveats = synthesize_caveats(bsv, substrate=substrate)
    conf = overall_confidence(bsv, substrate=substrate)

    delta = {a: 0.0 for a in cfg.BSV_AXES}
    if reference_bsv is not None:
        delta = delta_bsv(reference_bsv, bsv)
    top_signed = top_axes(delta if reference_bsv is not None else bsv, k=3,
                            mode="abs" if reference_bsv is not None else "abs")

    return {
        "sample_id": sample_id,
        "title": title,
        "domain": domain,
        "substrate": substrate,
        "preprocessing": pp["summary"],
        "spectrum": {
            "wavenumber": pp["wavenumber"].tolist(),
            "raw_intensity": pp["raw_intensity"].tolist(),
            "processed_intensity": pp["processed_intensity"].tolist(),
        },
        "features": {
            "anchors": anchors,
            "support": supports,
            "anti_evidence": [],
            "n_peaks": prims["n_peaks"],
        },
        "motif_scores_raw": motif_scores_raw,
        "motif_scores_adjusted": motif_scores,
        "substrate_events": sub_events,
        "mss_fires": {k: {"fire": v.fire, "anchor": v.anchor_score,
                            "support": v.support_score, "anti": v.anti_score}
                       for k, v in mss_fires.items()},
        "bsv": bsv,
        "delta_bsv": delta,
        "top_axes": top_signed,
        "evidence": evidence,
        "caveats": caveats,
        "confidence": conf,
    }
