"""GAIRA Demo v1 — evidence synthesis.

Converts motif scores, MSS fires, and substrate context into the
evidence + caveats blocks that feed the GAIRAReport.

Language rules (enforced via templates):
- Use "consistent with" not "is".
- Default to class-level claims unless co-bands corroborate.
- Substrate sensitivity flagged explicitly for SERS regimes.
"""
from __future__ import annotations

from .config import (
    AXIS_LABEL_LONG, BSV_AXES, axis_long, axis_short,
)
from .motif_scoring import MOTIFS, get_motif
from .substrate_physics import axis_level_rules


def synthesize_evidence(motif_scores: dict[str, float],
                          bsv: dict[str, float],
                          *, substrate: str,
                          strong_threshold: float = 0.25,
                          mod_threshold: float = 0.10) -> list[dict]:
    """Build a list of evidence entries (one per supported axis)."""
    evidence: list[dict] = []
    # Per-axis evidence rolled up from motifs
    per_axis_motifs: dict[str, list[tuple[str, float]]] = {a: [] for a in BSV_AXES}
    for m in MOTIFS:
        s = float(motif_scores.get(m.motif_id, 0.0))
        if s > 0:
            per_axis_motifs[m.primary_axis].append((m.motif_id, s))

    for axis in BSV_AXES:
        val = float(bsv.get(axis, 0.0))
        if val < mod_threshold:
            continue
        confidence = "high" if val >= strong_threshold else "moderate"
        motifs_on_axis = sorted(per_axis_motifs[axis], key=lambda x: x[1], reverse=True)
        bands: list[str] = []
        for motif_id, score in motifs_on_axis:
            m = get_motif(motif_id)
            if m is None:
                continue
            for lo, hi in m.bands:
                bands.append(f"{lo:.0f}–{hi:.0f} cm⁻¹")
        if motifs_on_axis:
            summary = (
                f"Spectral features consistent with {axis_long(axis)}. "
                f"Primary motif: {motifs_on_axis[0][0]}."
            )
            evidence.append({
                "axis": axis,
                "evidence_type": "direct_spectral",
                "bands": bands,
                "summary": summary,
                "confidence": confidence,
                "motif_id": motifs_on_axis[0][0],
            })
        else:
            evidence.append({
                "axis": axis,
                "evidence_type": "supporting",
                "bands": [],
                "summary": f"Supporting / cross-axis contribution to {axis_long(axis)}.",
                "confidence": "low",
                "motif_id": None,
            })
    return evidence


def synthesize_caveats(bsv: dict[str, float], *, substrate: str) -> list[dict]:
    caveats: list[dict] = []
    # Substrate caveats
    rules = axis_level_rules(substrate)
    for r in rules:
        if r.axis in bsv and bsv[r.axis] > 0.05:
            caveats.append({
                "type": "substrate",
                "summary": f"[{substrate}] {r.note}",
                "axis": r.axis,
            })

    # Ambiguity caveats for tightly-coupled axes
    coupled = [
        ("G01_purine_nucleotide", "G02_purine_metabolite",
            "Purine ring breathing 720–740 cm⁻¹ is shared between nucleotide and metabolite chemistry — "
            "molecule-level call requires co-bands at 1335 cm⁻¹ (adenine) or 640+891 cm⁻¹ (uric acid)."),
        ("G08_lipid_acyl_membrane", "G09_sterol_neutral_lipid",
            "Lipid acyl and sterol share 1440 cm⁻¹ CH₂ deformation — sterol-specific 548 cm⁻¹ "
            "must co-fire for sterol-specific call."),
        ("G06_protein_peptide_backbone", "G07_aromatic_residue",
            "Aromatic residues drive a fraction of the protein amide/aromatic envelope — overlap expected."),
    ]
    for a1, a2, msg in coupled:
        if bsv.get(a1, 0.0) > 0.10 and bsv.get(a2, 0.0) > 0.10:
            caveats.append({
                "type": "ambiguity",
                "summary": msg,
                "axes": [a1, a2],
            })

    return caveats


def overall_confidence(bsv: dict[str, float], *, substrate: str) -> dict:
    top = max(bsv.values()) if bsv else 0.0
    second = sorted(bsv.values(), reverse=True)[1] if len(bsv) >= 2 else 0.0
    margin = top - second

    if top >= 0.30 and margin >= 0.10:
        overall = "moderate-high"
        specificity = "class-level"
    elif top >= 0.20:
        overall = "moderate"
        specificity = "class-level"
    else:
        overall = "low"
        specificity = "not supported"

    sub_sens = {
        "Ag colloid SERS": "high",
        "Au SERS":         "high",
        "SERS":            "high",
        "Raman":           "low",
        "Spike-in":        "low",
    }.get(substrate, "medium")

    return {
        "overall": overall,
        "substrate_sensitivity": sub_sens,
        "molecular_specificity": specificity,
    }
