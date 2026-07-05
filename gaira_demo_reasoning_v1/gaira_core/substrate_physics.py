"""GAIRA Demo v1 — substrate-physics-aware adjustment.

Encodes a *small* set of substrate-aware rules so the demo can show how
GAIRA dampens or boosts evidence given the matrix/SERS regime. These are
illustrative; production GAIRA uses a much larger rule library.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubstrateRule:
    rule_id: str
    substrate: str
    motif_id: str | None
    axis: str | None
    multiplier: float
    note: str


RULES: tuple[SubstrateRule, ...] = (
    SubstrateRule(
        "ag_sers_purine_amplify", "Ag colloid SERS",
        "purine_ring_breathing_720_735", None, 0.65,
        "Ag-SERS adsorbs purines onto the metal — 720–740 cm⁻¹ is amplified ×3–10. "
        "Dampen molecule-level purine call until co-bands corroborate."
    ),
    SubstrateRule(
        "ag_sers_carotenoid_overlap", "Ag colloid SERS",
        None, "G02_purine_metabolite", 0.85,
        "Carotenoid C=C near 1517 cm⁻¹ co-occurs with uric-acid 1517 in serum; "
        "soft-dampen G02 specificity in carotenoid-rich serum."
    ),
    SubstrateRule(
        "raman_amide_protein", "Raman",
        "protein_amide_iii_1230_1300", "G06_protein_peptide_backbone", 1.0,
        "Raman amide-III is reliable in protein-rich matrices; no adjustment."
    ),
    SubstrateRule(
        "raman_amide_i_lipid_overlap", "Raman",
        None, "G08_lipid_acyl_membrane", 0.92,
        "Amide-I 1650–1670 cm⁻¹ overlaps C=C 1655 in unsaturated lipids; "
        "minor soft-dampen in protein-dominant matrices."
    ),
    SubstrateRule(
        "ag_sers_thiol_amplify", "Ag colloid SERS",
        "thione_c_s_490_500", "G10_sulfur_thiol_redox", 1.20,
        "Thiol/thione affinity for Ag surface — small boost on Ag-SERS."
    ),
)


def apply_substrate_corrections(motif_scores: dict[str, float], *, substrate: str
                                  ) -> tuple[dict[str, float], list[dict]]:
    """Return adjusted motif scores plus list of applied rule events."""
    adjusted = dict(motif_scores)
    events = []
    for rule in RULES:
        if rule.substrate != substrate:
            continue
        if rule.motif_id is None:
            continue
        if rule.motif_id in adjusted:
            before = adjusted[rule.motif_id]
            adjusted[rule.motif_id] = before * rule.multiplier
            events.append({
                "rule_id": rule.rule_id,
                "motif_id": rule.motif_id,
                "before": before,
                "after": adjusted[rule.motif_id],
                "multiplier": rule.multiplier,
                "note": rule.note,
            })
    return adjusted, events


def axis_level_rules(substrate: str) -> list[SubstrateRule]:
    """Substrate rules that apply at axis (not motif) level — used as caveats."""
    return [r for r in RULES if r.substrate == substrate and r.axis is not None]
