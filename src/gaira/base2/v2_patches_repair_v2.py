"""gaira_base_2 repair loop v2 patches module.

Additive layer on top of v2_patches_rescue (which was additive on top of
v2_patches). Implements four targeted repairs identified in
`gaira_base_2_grounding_repair_loop_v1_miss_interpretation`:

  REPAIR 1 — cholesteryl-ester discriminator (registry-level, not here)
  REPAIR 2 — asymmetric purine routing (MOTIF_AXIS_OVERRIDES)
  REPAIR 3 — motif-specific broad-motif suppression (tighter specificity
             weights for 4 specific broad motifs)
  REPAIR 4 — metabolic axis coverage strengthening (sparse-axis boost
             metabolic_small_molecule × 1.5 instead of × 1.3; per-motif
             specificity bumps for the promoted metabolite motifs)

None of the v1 / v2 / rescue modules are modified. Original fallback paths
remain available.
"""
from __future__ import annotations

import copy
from typing import Iterable

import numpy as np

from gaira.base2 import v2_patches as _v2
from gaira.base2 import v2_patches_rescue as _rescue


# ──────────────────────────────────────────────────────────────────────
# REPAIR 2 — asymmetric purine routing
# ──────────────────────────────────────────────────────────────────────
#
# v3 data: pure adenine/guanine references often route to purine_metabolite
# because purine_ring_breathing_720_735 is CROSS_AXIS with equal 0.7 weight
# on both purine_nucleotide and purine_metabolite, and the sparse-axis
# boost (1.15×) tilts the tie toward purine_metabolite.
#
# Fix: override the (motif, axis) weight pair to asymmetrically favour
# purine_nucleotide. This is a mapping-layer correction, not an ontology
# or motif definition change.

MOTIF_AXIS_OVERRIDES: dict[tuple[str, str], float] = {
    # Drop purine_ring_breathing_720_735 → purine_metabolite to 0.3
    # (was CROSS_AXIS 0.7). Keep → purine_nucleotide at 1.0 (PRIMARY-level).
    ("purine_ring_breathing_720_735", "purine_metabolite"): 0.30,
    ("purine_ring_breathing_720_735", "purine_nucleotide"): 1.00,
}


# ──────────────────────────────────────────────────────────────────────
# REPAIR 3 — motif-specific broad-motif suppression
# ──────────────────────────────────────────────────────────────────────
#
# v3 data identified 8 BROAD_MOTIF_DOMINANCE cases where specific broad
# motifs won top-1 on spectra where they shouldn't. Further tighten their
# specificity weights (v2 rescue had them at 0.50-0.55).

SPECIFICITY_WEIGHTS_REPAIR_V2: dict[str, float] = {
    **_rescue.SPECIFICITY_WEIGHTS_RESCUE,
    # further dampening for motifs causing >1 misdirection in v3
    "amide_I_alpha_helix_beta_sheet_motif":    0.40,  # was 0.54; fires on COOH / small molecules
    "amide_III_protein_backbone_1230_1280":    0.40,  # was 0.52; fires on amino acids
    "lipid_C_H_bend_1440_1460":                0.40,  # was 0.50; fires on amino acids (Arg, Ser, Asp)
    "free_saccharide_motif":                   0.40,  # was 0.55; fires on xylanase + amino acids
    "phosphatidylcholine_choline_head_715":    0.60,  # was 0.80; fires on estrogens
    "lipid_methylene_twist_1300":              0.42,  # was 0.50; dominates on Arg / amino acids
    "cholesterol_signature":                   0.50,  # was 0.54; broad 700+1440

    # REPAIR 4 — promote glutamate + citrate specificity for metabolic axis rescue
    "glutamate_glutamine_motif":               0.95,  # was 0.80; 3-band REQUIRED deserves high spec
    "citrate_as_biology_motif":                0.92,  # was 0.80
    "sugar_phosphate_skeletal_870_900":        0.80,  # was 0.75; slight bump
    # new motif added in registry v1.4
    "cholesteryl_ester_discriminator_motif":   0.95,
}


def specificity_weight_repair_v2(motif_id: str) -> float:
    return SPECIFICITY_WEIGHTS_REPAIR_V2.get(
        motif_id, _v2.DEFAULT_SPECIFICITY_WEIGHT,
    )


# ──────────────────────────────────────────────────────────────────────
# REPAIR 4 — stronger sparse-axis boost for metabolic_small_molecule
# ──────────────────────────────────────────────────────────────────────

SPARSE_AXIS_BOOST_REPAIR_V2: dict[str, float] = {
    **_v2.SPARSE_AXIS_BOOST,
    "metabolic_small_molecule":     1.50,   # was 1.30; still sparse despite promotions
    "purine_nucleotide":            1.15,   # mild boost so guanine/adenine refs
                                              # can compete against purine_metabolite
    "aromatic_residue":             1.15,   # was unboosted; His / Phe / Tyr
                                              # vs protein_peptide_backbone
    "sterol_neutral_lipid":         1.30,   # was 1.20; increase to help
                                              # cholesteryl_ester discriminator weight
}


def apply_sparse_axis_boost_repair_v2(
    mapping_weight: float, target_axis: str, mapping_type: str,
) -> float:
    if mapping_type != "PRIMARY":
        return mapping_weight
    boost = SPARSE_AXIS_BOOST_REPAIR_V2.get(target_axis, 1.0)
    return mapping_weight * boost


# ──────────────────────────────────────────────────────────────────────
# Mapping-weight resolver combining REPAIR 2 + REPAIR 4 (+ rescue)
# ──────────────────────────────────────────────────────────────────────

def _resolve_mapping_weight_repair_v2(mapping, target_axis: str) -> float:
    # Check override first (REPAIR 2)
    key = (mapping.motif_id, target_axis)
    if key in MOTIF_AXIS_OVERRIDES:
        if not mapping.active:
            return 0.0
        return MOTIF_AXIS_OVERRIDES[key]
    # Otherwise: base mapping weight × sparse-axis boost with REPAIR-V2 boosts
    from gaira.base2.motif_engine import resolve_mapping_weight
    base = resolve_mapping_weight(mapping, target_axis)
    return apply_sparse_axis_boost_repair_v2(
        base, target_axis, mapping.mapping_type,
    )


# ──────────────────────────────────────────────────────────────────────
# Competitor-set dampening (reuse rescue variant — glycan-vs-phos removed)
# ──────────────────────────────────────────────────────────────────────

def apply_competitor_dampening_repair_v2(self_core_weights: dict) -> dict:
    return _rescue.apply_competitor_dampening_rescue(self_core_weights)


# ──────────────────────────────────────────────────────────────────────
# Patched scoring pipeline (repair-v2 variant)
# ──────────────────────────────────────────────────────────────────────

def _noisy_or(weights: Iterable[float]) -> float:
    p = 1.0
    any_contrib = False
    for w in weights:
        wc = float(np.clip(w, 0.0, 1.0))
        if wc > 0:
            any_contrib = True
        p *= (1.0 - wc)
    return 0.0 if not any_contrib else float(1.0 - p)


def patched_score_spectrum_repair_v2(
    spectrum, master_x, motifs, mappings, dual_status,
    spectrum_id="",
    apply_a=True, apply_b=True, apply_c=True, apply_d=True,
):
    """Repair-v2 patched_score_spectrum.

    - PATCH A: REPAIR-V2 specificity weights (tighter broad-motif suppression
               + metabolite-motif promotion)
    - PATCH B: rescue-variant competitor dampening (glycan-vs-phos removed)
    - PATCH C: gated ambiguity (unchanged from v2)
    - PATCH D: REPAIR-V2 sparse-axis boost (metabolic × 1.5, +purine_nuc + aromatic)
    - PURINE ROUTING: asymmetric MOTIF_AXIS_OVERRIDES
    """
    from gaira.base2.motif_engine import (
        compute_motif_activation, motif_belongs_to_ambiguity_lane,
        resolve_status_calibration_weight, resolve_status_core_weight,
    )
    from gaira.base2.projection import project_to_8_axes
    from gaira.base2.schema import (
        AxisScore, BIOLOGY_AXES_V11, MotifScore, SpectrumResult,
    )

    per_motif_self_core = {}
    per_motif_self_regime = {}
    per_motif_contrib_ambig = {}
    per_motif_activation = {}

    for mid, spec in motifs.items():
        activation = compute_motif_activation(spec, spectrum, master_x)
        per_motif_activation[mid] = activation
        status = dual_status.get(mid)
        core_status_w = resolve_status_core_weight(status)
        cal_w = resolve_status_calibration_weight(status)
        spec_w = specificity_weight_repair_v2(mid) if apply_a else 1.0
        self_core = float(np.clip(activation * core_status_w * spec_w, 0.0, 1.0))
        self_regime = float(np.clip(self_core * cal_w, 0.0, 1.0))
        per_motif_self_core[mid] = self_core
        per_motif_self_regime[mid] = self_regime
        per_motif_contrib_ambig[mid] = motif_belongs_to_ambiguity_lane(
            spec, mappings.get(mid),
        )

    if apply_b:
        per_motif_self_core = apply_competitor_dampening_repair_v2(per_motif_self_core)
        per_motif_self_regime = apply_competitor_dampening_repair_v2(per_motif_self_regime)

    motif_scores = {}
    for mid in motifs:
        motif_scores[mid] = MotifScore(
            motif_id=mid,
            activation=float(per_motif_activation[mid]),
            core_weight=float(per_motif_self_core[mid]),
            regime_weight=float(per_motif_self_regime[mid]),
            contributes_to_ambiguity=per_motif_contrib_ambig[mid],
        )

    axis11 = []
    axis11_core_dict = {}
    if apply_d:
        resolver = _resolve_mapping_weight_repair_v2
    else:
        from gaira.base2.motif_engine import resolve_mapping_weight
        resolver = resolve_mapping_weight

    for axis_id in BIOLOGY_AXES_V11:
        contrib_ids = []
        core_terms = []
        regime_terms = []
        for mid in motifs:
            mapping = mappings.get(mid)
            if mapping is None:
                continue
            mw = resolver(mapping, axis_id)
            if mw <= 0:
                continue
            contrib_ids.append(mid)
            core_terms.append(per_motif_self_core[mid] * mw)
            regime_terms.append(per_motif_self_regime[mid] * mw)
        core_ev = _noisy_or(core_terms)
        regime_ev = _noisy_or(regime_terms)
        axis11_core_dict[axis_id] = core_ev
        axis11.append(AxisScore(
            axis_id=axis_id,
            core_evidence=core_ev,
            regime_evidence=regime_ev,
            contributing_motifs=tuple(contrib_ids),
        ))

    axis8 = project_to_8_axes(axis11)

    if apply_c:
        ambiguity = _v2.compute_gated_ambiguity_lane(
            motif_scores, mappings, axis11_core_dict,
        )
    else:
        from gaira.base2.ambiguity import compute_ambiguity_lane
        ambiguity = compute_ambiguity_lane(motif_scores, mappings)

    return SpectrumResult(
        spectrum_id=spectrum_id,
        motif_scores=tuple(motif_scores.values()),
        axis11_scores=tuple(axis11),
        axis8_projection=tuple(axis8),
        ambiguity=ambiguity,
        metadata={
            "n_motifs_evaluated": len(motif_scores),
            "variant": "repair_v2",
            "repairs_applied": {
                "asymmetric_purine_routing": True,
                "broad_motif_suppression": True,
                "metabolic_axis_boost": True,
                "glycan_vs_phos_competitor_removed": True,
            },
        },
    )


REPAIR_SUMMARY = {
    "REPAIR_2": "Asymmetric purine routing — purine_ring_breathing_720_735 → purine_metabolite dampened to 0.30",
    "REPAIR_3": "Broad-motif suppression — amide_I/III, lipid_CH_bend, free_saccharide, PC_choline, methylene_twist, cholesterol_signature specificity further reduced",
    "REPAIR_4": "Metabolic axis boost — sparse-axis multiplier ×1.5 for metabolic_small_molecule (was ×1.3); +glutamate/citrate specificity promotion",
    "NEW_MAPPING": "sterol_neutral_lipid boost ×1.3 (was ×1.2); purine_nucleotide +×1.15; aromatic_residue +×1.15",
}
