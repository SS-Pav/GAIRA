"""Diabetes-audit overrides — modular additions that do NOT modify GAIRA core.

Three fixes applied by this module (all reversible, all self-contained):

1. **G10 motif window tightened**
   - Original: `thione_c_s_490_500` fires on 480–510 cm⁻¹ (30 cm⁻¹ window).
   - Ag-colloid + plasma matrix commonly show broad background in 450–520 cm⁻¹
     (citrate breathing modes, aggregate cluster modes, Ag oxide).
   - Tighter window (490–505 cm⁻¹, 15 cm⁻¹) targets the ergothioneine / thione
     C–S stretch specifically and reduces substrate/matrix pickup ~50%.

2. **Co-band-gated Ag-SERS thiol boost**
   - Original: `ag_sers_thiol_amplify` unconditionally multiplies the thione motif
     score by ×1.20 on any Ag-SERS spectrum.
   - New: the ×1.20 boost is applied ONLY when the ergothioneine 720 cm⁻¹
     imidazole support co-band actually fires above a small floor. If 720
     is silent, boosting the anchor at 495 alone is unjustified inflation.

3. **Z-score axis reporting** (implemented in the caller, not here)
   - Report cohort mean on each axis as z = (cohort_mean − pooled_mean) / pooled_SD.
   - Normalizes out the ×0.65 vs ×1.20 asymmetric substrate rules and the
     inter-axis magnitude differences that come from broad vs narrow motif windows.

Consumers call `build_report_diabetes(...)` instead of `gaira_core.report_builder.build_report(...)`.
The output dict shape is identical to `build_report`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaira_core import config as _cfg
from gaira_core.bsv_projection import delta_bsv, project_to_bsv, top_axes
from gaira_core.evidence_synthesis import (
    overall_confidence, synthesize_caveats, synthesize_evidence,
)
from gaira_core.motif_scoring import MOTIFS as _ORIG_MOTIFS, Motif as _Motif
from gaira_core.mss_scoring import score_all
from gaira_core.preprocessing import preprocess
from gaira_core.primitive_extraction import primitives_from
from gaira_core.substrate_physics import RULES as _RULES


# ── FIX 1 — tightened G10 motif window ──────────────────────────────
def _tighten_g10(motif: _Motif) -> _Motif:
    if motif.motif_id != "thione_c_s_490_500":
        return motif
    return _Motif(
        motif_id="thione_c_s_490_500",
        bands=((490.0, 505.0),),      # was (480, 510)
        primary_axis="G10_sulfur_thiol_redox",
        description=(motif.description
                        + " [diabetes-audit: tightened to 490–505]"),
    )

DIABETES_MOTIFS: tuple[_Motif, ...] = tuple(_tighten_g10(m) for m in _ORIG_MOTIFS)


def _band_score(wn: np.ndarray, intensity: np.ndarray, band: tuple[float, float]) -> float:
    lo, hi = band
    mask = (wn >= lo) & (wn <= hi)
    return float(intensity[mask].max()) if mask.any() else 0.0


def score_motifs_diabetes(wavenumber: np.ndarray, intensity: np.ndarray) -> dict[str, float]:
    """Same as gaira_core.motif_scoring.score_motifs, but uses DIABETES_MOTIFS."""
    out = {}
    for m in DIABETES_MOTIFS:
        scores = [_band_score(wavenumber, intensity, b) for b in m.bands]
        if any(s <= 0 for s in scores):
            out[m.motif_id] = 0.0
        else:
            out[m.motif_id] = float(np.exp(np.mean(np.log(scores))))
    return out


# ── FIX 2 — co-band-gated Ag-SERS thiol boost ───────────────────────
ERGO_IMIDAZOLE_BAND: tuple[float, float] = (715.0, 735.0)
ERGO_IMIDAZOLE_FLOOR: float = 0.010

def apply_substrate_gated(motif_scores: dict[str, float], *, substrate: str,
                             wavenumber: np.ndarray, intensity: np.ndarray
                            ) -> tuple[dict[str, float], list[dict]]:
    """Co-band-gated Ag-SERS substrate correction.

    Iterates the original substrate rule list; for the thiol-amplify rule
    only, checks that the ergothioneine 720 cm⁻¹ imidazole support fires
    above `ERGO_IMIDAZOLE_FLOOR` before applying the ×1.20 boost. All
    other substrate rules apply as in the demo (e.g., the purine dampen
    at ×0.65 remains unchanged, since it is applied specifically to
    counter Ag-SERS purine over-amplification and does not have a
    co-band gating requirement)."""
    adjusted = dict(motif_scores)
    events   = []
    imi_max  = _band_score(wavenumber, intensity, ERGO_IMIDAZOLE_BAND)
    imi_present = imi_max >= ERGO_IMIDAZOLE_FLOOR

    for rule in _RULES:
        if rule.substrate != substrate:      continue
        if rule.motif_id is None:            continue
        if rule.motif_id not in adjusted:    continue

        if rule.rule_id == "ag_sers_thiol_amplify":
            if not imi_present:
                events.append({
                    "rule_id":       rule.rule_id,
                    "motif_id":      rule.motif_id,
                    "before":        adjusted[rule.motif_id],
                    "after":         adjusted[rule.motif_id],
                    "multiplier":    1.0,
                    "note":          (f"boost SKIPPED — 720 cm⁻¹ imidazole "
                                        f"support ({imi_max:.4f}) below floor "
                                        f"{ERGO_IMIDAZOLE_FLOOR:.3f}. Anchor at "
                                        f"495 firing alone is not sufficient for "
                                        f"the substrate-driven thiol boost."),
                    "gate":          "co-band-required (skipped)",
                    "imi_max":       imi_max,
                })
                continue
            before = adjusted[rule.motif_id]
            adjusted[rule.motif_id] = before * rule.multiplier
            events.append({
                "rule_id":       rule.rule_id,
                "motif_id":      rule.motif_id,
                "before":        before,
                "after":         adjusted[rule.motif_id],
                "multiplier":    rule.multiplier,
                "note":          (f"boost APPLIED — 720 cm⁻¹ imidazole "
                                    f"support ({imi_max:.4f}) confirms thione "
                                    f"co-band evidence."),
                "gate":          "co-band-required (applied)",
                "imi_max":       imi_max,
            })
            continue

        before = adjusted[rule.motif_id]
        adjusted[rule.motif_id] = before * rule.multiplier
        events.append({
            "rule_id":    rule.rule_id, "motif_id":  rule.motif_id,
            "before":     before, "after":     adjusted[rule.motif_id],
            "multiplier": rule.multiplier, "note":     rule.note,
            "gate": "unconditional",
        })
    return adjusted, events


# ── Full pipeline (drop-in replacement for build_report) ────────────
def build_report_diabetes(*, sample_id: str, title: str, domain: str,
                             substrate: str, wavenumber: np.ndarray,
                             intensity: np.ndarray,
                             reference_bsv: dict[str, float] | None = None) -> dict:
    """Diabetes-audit build_report. Same output shape as
    `gaira_core.report_builder.build_report`, but uses:
        - tightened G10 motif (490–505 cm⁻¹)
        - co-band-gated Ag-SERS thiol boost (requires 720 cm⁻¹ imidazole)
    """
    pp = preprocess(wavenumber, intensity)
    prims = primitives_from(pp["wavenumber"], pp["processed_intensity"])
    mss_fires = score_all(pp["wavenumber"], pp["processed_intensity"])

    motif_scores_raw = score_motifs_diabetes(pp["wavenumber"], pp["processed_intensity"])
    motif_scores, sub_events = apply_substrate_gated(
        motif_scores_raw, substrate=substrate,
        wavenumber=pp["wavenumber"], intensity=pp["processed_intensity"],
    )

    bsv = project_to_bsv(motif_scores, mss_fires=mss_fires)

    anchors, supports = [], []
    for m in DIABETES_MOTIFS:
        s = float(motif_scores.get(m.motif_id, 0.0))
        if s >= 0.05:
            for lo, hi in m.bands:
                anchors.append({"motif_id": m.motif_id, "low": lo, "high": hi, "score": s})
        elif s > 0.01:
            for lo, hi in m.bands:
                supports.append({"motif_id": m.motif_id, "low": lo, "high": hi, "score": s})

    evidence = synthesize_evidence(motif_scores, bsv, substrate=substrate)
    caveats  = synthesize_caveats(bsv, substrate=substrate)
    conf     = overall_confidence(bsv, substrate=substrate)

    delta = {a: 0.0 for a in _cfg.BSV_AXES}
    if reference_bsv is not None:
        delta = delta_bsv(reference_bsv, bsv)
    top_signed = top_axes(delta if reference_bsv is not None else bsv, k=3, mode="abs")

    return {
        "sample_id":              sample_id,
        "title":                  title,
        "domain":                 domain,
        "substrate":              substrate,
        "preprocessing":          pp["summary"],
        "spectrum":  {
            "wavenumber":         pp["wavenumber"].tolist(),
            "raw_intensity":      pp["raw_intensity"].tolist(),
            "processed_intensity": pp["processed_intensity"].tolist(),
        },
        "features":  {
            "anchors":            anchors,
            "support":            supports,
            "anti_evidence":      [],
            "n_peaks":            prims["n_peaks"],
        },
        "motif_scores_raw":       motif_scores_raw,
        "motif_scores_adjusted":  motif_scores,
        "substrate_events":       sub_events,
        "mss_fires": {k: {"fire": v.fire, "anchor": v.anchor_score,
                              "support": v.support_score, "anti": v.anti_score}
                       for k, v in mss_fires.items()},
        "bsv":                    bsv,
        "delta_bsv":              delta,
        "top_axes":               top_signed,
        "evidence":               evidence,
        "caveats":                caveats,
        "confidence":             conf,
        "diabetes_audit_flags":  {
            "g10_motif_window":  "(490, 505)  [was (480, 510)]",
            "thiol_boost_gate":   "requires 720 cm⁻¹ imidazole co-band ≥ 0.010",
            "imidazole_720_intensity": _band_score(
                pp["wavenumber"], pp["processed_intensity"], ERGO_IMIDAZOLE_BAND),
        },
    }
