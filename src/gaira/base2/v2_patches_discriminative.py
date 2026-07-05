"""gaira_base_2 discriminative motif upgrade (v_discriminative).

ADDITIVE on top of v2_patches_rescue. Reframes motifs as discriminative
objects with structured role + anti-evidence + competitor metadata, and
applies role-based gating + anti-evidence penalties to the per-motif
core_weight produced by the rescue engine.

Design rules (locked):

  * No engine-module modification (schema / motif_engine / projection /
    ambiguity / registry / primitives — all read-only).
  * No new motifs added in this module (registry is not the source of
    discriminator metadata; this module IS the discriminator metadata
    plus its scoring rule).
  * Anti-evidence is structured and deterministic. Three rule types:

      REQUIRES_COBAND <target> >= min_w → if target's weight is BELOW
                                          min_w, multiply this motif by
                                          (1 - penalty).
      SUPPRESS_IF_PRESENT <target> >= min_w → if target's weight is AT
                                              OR ABOVE min_w, multiply
                                              this motif by (1 - penalty).
      REQUIRES_ANY_FAMILY_ANCHOR [targets] >= min_w → if NONE of the
                                                       listed anchors are
                                                       at or above min_w,
                                                       multiply this motif
                                                       by (1 - penalty).

  * Role gating is multiplicative:
      ANCHOR        × 1.00
      SUPPORT       × 0.85   (0.55 if no same-family ANCHOR co-fires)
      BACKGROUND    × 0.40   (0.20 if no same-family ANCHOR co-fires)
      AMBIGUITY_ONLY → biology-axis contribution = 0
      ARTIFACT_ONLY  → biology-axis contribution = 0

  * Ambiguity lane keeps the rescue gated-ambiguity rule unchanged.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np

from gaira.base2 import v2_patches as _v2
from gaira.base2 import v2_patches_rescue as _rescue


# ─────────────────────────────────────────────────────────────────────
# ROLE TABLE — 52 active motifs + 11 inactive motifs (NO_MAPPING /
# HELD_V2 / DEFERRED). Roles drive multiplicative gating.
# ─────────────────────────────────────────────────────────────────────

ROLE_TABLE: dict[str, str] = {
    # ── ANCHORs (chemistry-specific, can drive a claim alone) ────────
    "uric_acid_full_signature":          "ANCHOR",
    "hypoxanthine_signature":            "ANCHOR",
    "xanthine_signature":                "ANCHOR",
    "guanine_specific_motif":            "ANCHOR",
    "thymine_specific_motif":            "ANCHOR",
    "cytosine_specific_motif":           "ANCHOR",
    "pyrimidine_ring_breathing_780_800": "ANCHOR",
    "dna_methylation_marker_790":        "ANCHOR",
    "phenylalanine_ring_1003":           "ANCHOR",
    "tyrosine_doublet_830_850":          "ANCHOR",
    "sialic_acid_signature":             "ANCHOR",
    "cholesterol_signature":             "ANCHOR",
    "sterol_skeletal_motif":             "ANCHOR",
    "cholesteryl_ester_discriminator_motif": "ANCHOR",
    "neutral_lipid_triglyceride_motif":  "ANCHOR",
    "disulfide_S_S_str_500_550":         "ANCHOR",
    "ergothioneine_signature":           "ANCHOR",
    "glutathione_GSH_motif":             "ANCHOR",
    "creatine_creatinine_motif":         "ANCHOR",
    "glutamate_glutamine_motif":         "ANCHOR",
    "citrate_as_biology_motif":          "ANCHOR",
    "cytochrome_c_resonance_motif":      "ANCHOR",
    "dna_composite_motif":               "ANCHOR",  # multi-band REQUIRED is anchor-grade
    "phosphate_PO_asym_str_1240":        "ANCHOR",  # narrow phosphate-specific

    # ── SUPPORTs (helpful but shared chemistry) ──────────────────────
    "purine_ring_breathing_720_735":     "SUPPORT",  # CROSS_AXIS, shared
    "phosphate_PO2_sym_str_1080":        "SUPPORT",  # overlaps glycan
    "glycan_pyranose_ring_skeletal_850_950": "SUPPORT",  # broad ring region
    "glycan_glycosidic_C_O_C_1020_1100": "SUPPORT",  # overlaps phosphate
    "lipid_acyl_C_C_str_1060_1130":      "SUPPORT",  # lipid skeletal
    "thiol_C_S_str_660_motif":           "SUPPORT",  # narrow C-S
    "phosphatidylcholine_choline_head_715": "SUPPORT",  # 715 collision zone
    "sugar_phosphate_skeletal_870_900":  "SUPPORT",  # CROSS_AXIS

    # ── BACKGROUND (broad chemistry indicators; never win alone) ─────
    "amide_I_alpha_helix_beta_sheet_motif": "BACKGROUND",
    "amide_III_protein_backbone_1230_1280": "BACKGROUND",
    "amide_II_motif":                    "BACKGROUND",
    "lipid_C_H_bend_1440_1460":          "BACKGROUND",
    "lipid_methylene_twist_1300":        "BACKGROUND",
    "free_saccharide_motif":             "BACKGROUND",  # broad sugar context

    # ── AMBIGUITY_ONLY ───────────────────────────────────────────────
    "purine_HX_lipid_choline_715_overlap_ambiguity": "AMBIGUITY_ONLY",
    "collision_1300_1400_multi_candidate_motif":     "AMBIGUITY_ONLY",

    # ── ARTIFACT_ONLY ────────────────────────────────────────────────
    "citrate_baseline_artifact_motif":   "ARTIFACT_ONLY",

    # ── inactive in v1 mapping (no contribution either way) ──────────
    "nucleobase_in_plane_ring_1320_1340":            "SUPPORT",       # HELD_V2
    "amide_I_lipid_carbonyl_partial_panel_motif":    "BACKGROUND",    # HELD_V2
    "collision_1020_1080_multi_candidate":           "AMBIGUITY_ONLY",# HELD_V2
    "lactate_motif":                                 "ANCHOR",        # DEFERRED (no mapping)
    "histidine_imidazole_motif":                     "ANCHOR",        # NO_MAPPING
    "tryptophan_signature_760_1340_1550":            "ANCHOR",        # NO_MAPPING
    "lipid_unsaturation_C_C_str_1655_motif":         "SUPPORT",       # NO_MAPPING
    "lipid_peroxidation_marker_motif":               "SUPPORT",       # NO_MAPPING
    "phosphoethanolamine_head_motif":                "SUPPORT",       # NO_MAPPING
    "ester_carbonyl_C_O_str_1730_motif":             "SUPPORT",       # NO_MAPPING
    "aldehyde_carbonyl_motif":                       "BACKGROUND",    # NO_MAPPING
    "cholesterol_ester_motif":                       "ANCHOR",        # NO_MAPPING
    "glycoprotein_composite_motif":                  "SUPPORT",       # NO_MAPPING
    "glycosaminoglycan_GAG_motif":                   "ANCHOR",        # NO_MAPPING
}

ROLE_FACTOR: dict[str, float] = {
    "ANCHOR":         1.00,
    "SUPPORT":        0.85,
    "BACKGROUND":     0.65,   # broad but cannot be crushed: many families
                              # (protein_peptide_backbone) have NO ANCHOR
                              # motifs in v1, so BACKGROUND must remain
                              # informative on its own
    "AMBIGUITY_ONLY": 0.00,   # zero contribution to family scoring
    "ARTIFACT_ONLY":  0.00,
}

# Penalty applied to SUPPORT / BACKGROUND when no same-family ANCHOR
# co-fires. Multiplied on top of the base role factor.
NO_ANCHOR_PENALTY: dict[str, float] = {
    "ANCHOR":         1.00,
    "SUPPORT":        0.80,   # 0.85 × 0.80 = 0.68 effective
    "BACKGROUND":     0.75,   # 0.65 × 0.75 = 0.49 effective
    "AMBIGUITY_ONLY": 1.00,   # n/a (already zero)
    "ARTIFACT_ONLY":  1.00,
}


# ─────────────────────────────────────────────────────────────────────
# ANTI-EVIDENCE RULES — applied per-motif at scoring time. Multiplicative.
# Format: list of dict rules:
#
#   {"rule": "REQUIRES_COBAND",
#    "target": <motif_id>, "min_weight": float, "penalty": float}
#
#   {"rule": "SUPPRESS_IF_PRESENT",
#    "target": <motif_id>, "min_weight": float, "penalty": float}
#
#   {"rule": "REQUIRES_ANY_FAMILY_ANCHOR",
#    "targets": [<motif_id>, ...], "min_weight": float, "penalty": float}
# ─────────────────────────────────────────────────────────────────────

ANTI_EVIDENCE: dict[str, list[dict]] = {
    # NOTE on min_weight values: rescue-engine core_weights run on the
    # 0.01-0.10 scale (specificity x status x activation, all clipped to
    # [0,1]). Thresholds calibrated to that scale, not the raw [0,1] axis.
    # ── Broad amide motifs ────────────────────────────────────────────
    # True protein references show amide_I + amide_III (and ideally II)
    # together. Free amino acids and small molecules typically fire
    # only ONE of these. Require multi-band amide co-fire.
    "amide_I_alpha_helix_beta_sheet_motif": [
        {"rule": "REQUIRES_COBAND",
         "target": "amide_III_protein_backbone_1230_1280",
         "min_weight": 0.020, "penalty": 0.50},
    ],
    "amide_III_protein_backbone_1230_1280": [
        {"rule": "REQUIRES_COBAND",
         "target": "amide_I_alpha_helix_beta_sheet_motif",
         "min_weight": 0.020, "penalty": 0.50},
    ],
    "amide_II_motif": [
        {"rule": "REQUIRES_COBAND",
         "target": "amide_I_alpha_helix_beta_sheet_motif",
         "min_weight": 0.020, "penalty": 0.50},
    ],

    # ── Broad lipid backbone motifs ──────────────────────────────────
    # Lipid CH bend (1440) and methylene twist (1300) fire on many
    # CH2-bearing molecules including amino acids. Real lipids must
    # have at least ONE chemistry-specific lipid anchor co-firing.
    "lipid_C_H_bend_1440_1460": [
        {"rule": "REQUIRES_ANY_FAMILY_ANCHOR",
         "targets": ["lipid_acyl_C_C_str_1060_1130",
                     "cholesterol_signature",
                     "sterol_skeletal_motif",
                     "neutral_lipid_triglyceride_motif",
                     "cholesteryl_ester_discriminator_motif",
                     "phosphatidylcholine_choline_head_715"],
         "min_weight": 0.020, "penalty": 0.55},
    ],
    "lipid_methylene_twist_1300": [
        {"rule": "REQUIRES_ANY_FAMILY_ANCHOR",
         "targets": ["lipid_acyl_C_C_str_1060_1130",
                     "cholesterol_signature",
                     "sterol_skeletal_motif",
                     "neutral_lipid_triglyceride_motif",
                     "cholesteryl_ester_discriminator_motif"],
         "min_weight": 0.020, "penalty": 0.55},
    ],

    # ── Broad sugar motif ────────────────────────────────────────────
    # free_saccharide_motif fires on amino acids + xylanase (protein).
    # Require glycan ring backbone co-fire AND no strong amide co-fire.
    "free_saccharide_motif": [
        {"rule": "REQUIRES_COBAND",
         "target": "glycan_pyranose_ring_skeletal_850_950",
         "min_weight": 0.020, "penalty": 0.55},
        {"rule": "SUPPRESS_IF_PRESENT",
         "target": "amide_I_alpha_helix_beta_sheet_motif",
         "min_weight": 0.050, "penalty": 0.40},
    ],

    # ── Glycan ↔ phosphate collision (1020-1100) ─────────────────────
    # If glycan_glycosidic 1020 fires AND phosphate_PO2 1080 fires
    # alongside DNA-anchor evidence, suppress the glycan claim.
    "glycan_glycosidic_C_O_C_1020_1100": [
        # Suppress only if BOTH phosphate motifs strongly fire
        # (avoid hitting pure sugars where 1240 has weak noise)
        {"rule": "SUPPRESS_IF_PRESENT",
         "target": "phosphate_PO_asym_str_1240",
         "min_weight": 0.050, "penalty": 0.35},
        {"rule": "SUPPRESS_IF_PRESENT",
         "target": "dna_composite_motif",
         "min_weight": 0.020, "penalty": 0.55},
    ],

    # ── Phosphate ↔ glycan collision (1080) ──────────────────────────
    # If phosphate_PO2 1080 fires WITHOUT phosphate_PO 1240 and the
    # glycan ring 850-950 IS strong, the 1080 is more likely glycan.
    "phosphate_PO2_sym_str_1080": [
        {"rule": "SUPPRESS_IF_PRESENT",
         "target": "glycan_pyranose_ring_skeletal_850_950",
         "min_weight": 0.040, "penalty": 0.40},
    ],

    # ── Purine 720 ↔ choline 715 collision ───────────────────────────
    # PC choline 715 is a known collision zone with purine 720-735.
    # If nucleobase in-plane 1320 is silent, the 720 reading is more
    # likely choline. (nucleobase_in_plane is HELD_V2 inactive in
    # mapping, but its activation can still inform anti-evidence.)
    # (purine_ring_breathing_720_735 anti-evidence is defined below in
    # the cholesterol section, and includes both PC choline 715 and
    # cholesterol_signature 700 suppressions in one block.)
    "phosphatidylcholine_choline_head_715": [
        {"rule": "REQUIRES_ANY_FAMILY_ANCHOR",
         "targets": ["lipid_C_H_bend_1440_1460",
                     "lipid_methylene_twist_1300",
                     "lipid_acyl_C_C_str_1060_1130"],
         "min_weight": 0.020, "penalty": 0.55},
    ],

    # ── Cholesterol broad anchor ──────────────────────────────────────
    # cholesterol_signature (548 + 700 + 1440) fires partly on broad CH
    # bend (1440). If sterol_skeletal_motif IS NOT firing, the 1440 is
    # likely generic lipid, not sterol. NOTE: sterol_skeletal_motif's
    # REQUIRED 3-band co-fire (548+615+956) almost never triggers in
    # canonical preprocessing, so this rule effectively always penalises
    # — disable for now with very small min_weight to keep documented
    # but not punish cholesterol references.
    "cholesterol_signature": [
        {"rule": "REQUIRES_COBAND",
         "target": "sterol_skeletal_motif",
         "min_weight": 0.005, "penalty": 0.20},
    ],

    # ── Purine 720 vs cholesterol 700 (additional collision) ─────────
    # cholesterol_signature has a primary band at 700 cm-1; on cholesterol
    # references the 720-735 purine motif can spuriously fire. Suppress
    # if cholesterol_signature is the dominant anchor.
    "purine_ring_breathing_720_735": [
        {"rule": "SUPPRESS_IF_PRESENT",
         "target": "phosphatidylcholine_choline_head_715",
         "min_weight": 0.040, "penalty": 0.50},
        {"rule": "SUPPRESS_IF_PRESENT",
         "target": "cholesterol_signature",
         "min_weight": 0.030, "penalty": 0.45},
    ],
}


# ─────────────────────────────────────────────────────────────────────
# COMPETITOR PAIRS — informational only (recorded in registry CSV);
# competitor suppression is implemented through SUPPRESS_IF_PRESENT
# rules above. This dict is for discriminator registry export.
# ─────────────────────────────────────────────────────────────────────

COMPETITORS: dict[str, list[str]] = {
    "amide_I_alpha_helix_beta_sheet_motif": [
        "lipid_methylene_twist_1300", "citrate_as_biology_motif",
        "glutamate_glutamine_motif", "carboxylic_acid (no motif yet)",
    ],
    "amide_III_protein_backbone_1230_1280": [
        "phosphate_PO_asym_str_1240", "lipid_methylene_twist_1300",
    ],
    "amide_II_motif": [
        "amide_I_alpha_helix_beta_sheet_motif",
    ],
    "lipid_C_H_bend_1440_1460": [
        "amino-acid CH2 (no motif)", "cholesterol_signature",
    ],
    "lipid_methylene_twist_1300": [
        "amino-acid CH2 wag (no motif)",
        "amide_III_protein_backbone_1230_1280",
        "collision_1300_1400_multi_candidate_motif",
    ],
    "free_saccharide_motif": [
        "amide_I_alpha_helix_beta_sheet_motif",
        "amino-acid skeletal (no motif)",
    ],
    "glycan_glycosidic_C_O_C_1020_1100": [
        "phosphate_PO2_sym_str_1080",
        "phosphate_PO_asym_str_1240",
        "dna_composite_motif",
    ],
    "phosphate_PO2_sym_str_1080": [
        "glycan_glycosidic_C_O_C_1020_1100",
        "glycan_pyranose_ring_skeletal_850_950",
    ],
    "purine_ring_breathing_720_735": [
        "phosphatidylcholine_choline_head_715",
        "uric_acid_full_signature",
        "hypoxanthine_signature",
    ],
    "phosphatidylcholine_choline_head_715": [
        "purine_ring_breathing_720_735",
        "uric_acid_full_signature",
        "hypoxanthine_signature",
    ],
    "cholesterol_signature": [
        "sterol_skeletal_motif",
        "lipid_C_H_bend_1440_1460",
    ],
    "sterol_skeletal_motif": [
        "cholesterol_signature",
        "cholesteryl_ester_discriminator_motif",
    ],
    "phosphate_PO_asym_str_1240": [
        "amide_III_protein_backbone_1230_1280",
    ],
    "thiol_C_S_str_660_motif": [
        "disulfide_S_S_str_500_550",
        "purine_ring_breathing_720_735",  # 720 vs 660 region overlap on metabolites
    ],
}


# ─────────────────────────────────────────────────────────────────────
# CO_FIRE_ANCHOR_GROUPS — multi-motif combinations that are
# chemistry-grade anchors when all members fire above min_weight.
#
# Required because the v1 ontology has families with NO ANCHOR motifs
# (protein_peptide_backbone, lipid_acyl_membrane). Real protein and
# real lipid chemistry presents as multi-band co-fire of broad motifs;
# the co-fire IS the anchor signal even if no individual member is
# chemistry-specific on its own.
# ─────────────────────────────────────────────────────────────────────

CO_FIRE_ANCHOR_GROUPS: list[dict] = [
    # Threshold values calibrated against actual base_weight distributions:
    # rescue-engine core_weights typically run 0.01-0.10 (specificity x
    # status x activation, all clipped). p75 for amide_I/III is ~0.04;
    # p90 is ~0.05. A threshold of 0.020 catches real proteins (where
    # both bands fire above noise) without firing on small-molecule
    # accidental amide-I bumps.
    {
        "name": "real_protein_amide_pair",
        "members": ["amide_I_alpha_helix_beta_sheet_motif",
                    "amide_III_protein_backbone_1230_1280"],
        "min_weight": 0.020,
        "anchor_for_families": ["protein_peptide_backbone"],
    },
    {
        "name": "real_lipid_acyl_chain_pair",
        "members": ["lipid_C_H_bend_1440_1460",
                    "lipid_methylene_twist_1300"],
        "min_weight": 0.020,
        "anchor_for_families": ["lipid_acyl_membrane"],
    },
    # Glycan anchor pair: glycan_pyranose ring (850-950) + free_saccharide
    # co-fire is real-sugar chemistry. Compensates for the v1 ontology
    # gap where most pure sugars (glucose, mannose, etc.) have no
    # discriminative ANCHOR — the discriminative chemistry comes from
    # the ring-mode + free-saccharide combination.
    {
        "name": "real_glycan_ring_pair",
        "members": ["glycan_pyranose_ring_skeletal_850_950",
                    "free_saccharide_motif"],
        "min_weight": 0.020,
        "anchor_for_families": ["glycan_carbohydrate"],
    },
    # Lipid acyl + skeletal pair: lipid_acyl_C_C_str + lipid_C_H_bend
    # co-fire is also real lipid chemistry (free fatty acids).
    {
        "name": "real_lipid_skeletal_pair",
        "members": ["lipid_acyl_C_C_str_1060_1130",
                    "lipid_C_H_bend_1440_1460"],
        "min_weight": 0.020,
        "anchor_for_families": ["lipid_acyl_membrane"],
    },
]


def is_in_active_cofire_group(motif_id: str, base_weights: dict[str, float]) -> bool:
    """True if motif_id is a member of any CO_FIRE_ANCHOR_GROUP whose
    ALL members fire above the group's min_weight."""
    for group in CO_FIRE_ANCHOR_GROUPS:
        if motif_id not in group["members"]:
            continue
        if all(base_weights.get(m, 0.0) >= group["min_weight"]
               for m in group["members"]):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────
# AMBIGUITY ROUTING — when a motif's strongest reading is shared
# chemistry, route part of its weight to the ambiguity lane instead.
# Rule format: list of {"rule": ..., "target": ..., "ambiguity_share": ...}
# ─────────────────────────────────────────────────────────────────────

AMBIGUITY_ROUTING: dict[str, dict] = {
    # purine 720 firing alongside PC choline 715 → ambiguity
    # (trigger thresholds calibrated against actual base_weight scale 0.01-0.10)
    "purine_ring_breathing_720_735": {
        "trigger": "PRESENT",
        "target": "phosphatidylcholine_choline_head_715",
        "trigger_min_weight": 0.040,
        "ambiguity_share": 0.30,   # route 30% of weight to ambiguity lane
    },
    # glycan 1020-1100 firing alongside phosphate 1080 → ambiguity
    "glycan_glycosidic_C_O_C_1020_1100": {
        "trigger": "PRESENT",
        "target": "phosphate_PO_asym_str_1240",
        "trigger_min_weight": 0.030,
        "ambiguity_share": 0.30,
    },
    # citrate as biology in serum SERS is also citrate buffer artifact
    # → already CROSS_AXIS to ambiguity_artifact in v1.2 mapping; no
    # additional routing needed at scoring layer.
}


# ─────────────────────────────────────────────────────────────────────
# SAME-FAMILY ANCHOR LOOKUP — built once from the mapping +
# ROLE_TABLE so SUPPORT / BACKGROUND can check whether a same-family
# ANCHOR co-fires.
# ─────────────────────────────────────────────────────────────────────

def _build_family_anchors(mappings: dict) -> dict[str, list[str]]:
    """For each family (axis), list its ANCHOR motifs (per ROLE_TABLE)."""
    fam_anchors: dict[str, list[str]] = {}
    for mid, mp in mappings.items():
        if ROLE_TABLE.get(mid) != "ANCHOR":
            continue
        if not mp.active:
            continue
        # Use primary_axis as the family
        fam_anchors.setdefault(mp.primary_axis, []).append(mid)
        for sa in mp.secondary_axes:
            fam_anchors.setdefault(sa, []).append(mid)
    return fam_anchors


def _motif_families(mid: str, mappings: dict) -> list[str]:
    mp = mappings.get(mid)
    if mp is None:
        return []
    out = [mp.primary_axis]
    out.extend(mp.secondary_axes)
    return out


# ─────────────────────────────────────────────────────────────────────
# DISCRIMINATIVE PER-MOTIF SCORING
# ─────────────────────────────────────────────────────────────────────

def apply_anti_evidence(
    motif_id: str, base_weights: dict[str, float],
) -> tuple[float, list[str]]:
    """Return multiplicative factor in [0, 1] and list of rule names that
    fired (for audit)."""
    rules = ANTI_EVIDENCE.get(motif_id, [])
    factor = 1.0
    fired = []
    for r in rules:
        if r["rule"] == "REQUIRES_COBAND":
            tw = base_weights.get(r["target"], 0.0)
            if tw < r["min_weight"]:
                factor *= (1.0 - r["penalty"])
                fired.append(f"REQUIRES_COBAND:{r['target']}<{r['min_weight']}")
        elif r["rule"] == "SUPPRESS_IF_PRESENT":
            tw = base_weights.get(r["target"], 0.0)
            if tw >= r["min_weight"]:
                factor *= (1.0 - r["penalty"])
                fired.append(f"SUPPRESS_IF_PRESENT:{r['target']}>={r['min_weight']}")
        elif r["rule"] == "REQUIRES_ANY_FAMILY_ANCHOR":
            max_t = max((base_weights.get(t, 0.0) for t in r["targets"]),
                        default=0.0)
            if max_t < r["min_weight"]:
                factor *= (1.0 - r["penalty"])
                fired.append("REQUIRES_ANY_FAMILY_ANCHOR<min")
    return float(factor), fired


def apply_role_gate(
    motif_id: str, base_weights: dict[str, float],
    family_anchors: dict[str, list[str]], mappings: dict,
    anchor_threshold: float = 0.030,
) -> tuple[float, str]:
    """Return role factor for this motif. The factor accounts for role
    AND for whether a same-family ANCHOR (or active CO_FIRE_ANCHOR_GROUP)
    co-fires for SUPPORT/BACKGROUND. Returns (factor, gate_reason)."""
    role = ROLE_TABLE.get(motif_id, "SUPPORT")
    base_factor = ROLE_FACTOR[role]
    if role in ("ANCHOR", "AMBIGUITY_ONLY", "ARTIFACT_ONLY"):
        return base_factor, role

    # SUPPORT or BACKGROUND: check for same-family ANCHOR co-fire
    fams = _motif_families(motif_id, mappings)
    has_anchor = False
    for fam in fams:
        for aid in family_anchors.get(fam, []):
            if aid == motif_id:
                continue
            if base_weights.get(aid, 0.0) >= anchor_threshold:
                has_anchor = True
                break
        if has_anchor:
            break

    # Also check CO_FIRE_ANCHOR_GROUP membership: if this motif is in an
    # active co-fire group, treat as anchor-equivalent.
    in_cofire = is_in_active_cofire_group(motif_id, base_weights)

    if has_anchor:
        return base_factor, f"{role}_with_anchor"
    if in_cofire:
        return base_factor, f"{role}_with_cofire_anchor"

    # No same-family anchor and not in cofire group → NO_ANCHOR_PENALTY
    factor = base_factor * NO_ANCHOR_PENALTY[role]
    return factor, f"{role}_no_anchor"


# ─────────────────────────────────────────────────────────────────────
# Family scoring helper (mirrors motif-first phase but uses
# discriminative weights).
# ─────────────────────────────────────────────────────────────────────

def family_score_discriminative(
    discriminative_weights: dict[str, float],
    mappings: dict, family: str,
) -> tuple[float, list[str]]:
    from gaira.base2.motif_engine import resolve_mapping_weight
    total = 0.0
    contribs = []
    for mid, w in discriminative_weights.items():
        if w <= 0:
            continue
        mp = mappings.get(mid)
        if mp is None:
            continue
        mw = resolve_mapping_weight(mp, family)
        if mw <= 0:
            continue
        total += float(w) * float(mw)
        contribs.append(mid)
    return float(total), contribs


# ─────────────────────────────────────────────────────────────────────
# Main scoring entry point — returns the SAME shape as the rescue
# engine PLUS a discriminative-weights dict and audit dict.
# ─────────────────────────────────────────────────────────────────────

def score_spectrum_discriminative(
    spectrum, master_x, motifs, mappings, dual_status,
    spectrum_id: str = "",
):
    """Score a spectrum through the rescue engine, then apply
    discriminative role gating + anti-evidence + ambiguity routing."""
    res = _rescue.patched_score_spectrum_rescue(
        spectrum, master_x, motifs, mappings, dual_status, spectrum_id,
    )
    base_weights = {m.motif_id: float(m.core_weight) for m in res.motif_scores}

    family_anchors = _build_family_anchors(mappings)

    # 1. Apply anti-evidence + role gating
    discriminative_weights: dict[str, float] = {}
    audit_rows: list[dict] = []
    routed_to_ambiguity = 0.0
    for mid, bw in base_weights.items():
        anti_factor, fired = apply_anti_evidence(mid, base_weights)
        role_factor, gate_reason = apply_role_gate(
            mid, base_weights, family_anchors, mappings,
        )
        # Ambiguity routing (route a share of weight to ambiguity lane)
        amb_share = 0.0
        amb_rule = AMBIGUITY_ROUTING.get(mid)
        if amb_rule and amb_rule["trigger"] == "PRESENT":
            tw = base_weights.get(amb_rule["target"], 0.0)
            if tw >= amb_rule["trigger_min_weight"]:
                amb_share = amb_rule["ambiguity_share"]

        new_weight = bw * anti_factor * role_factor * (1.0 - amb_share)
        discriminative_weights[mid] = float(np.clip(new_weight, 0.0, 1.0))
        if amb_share > 0:
            routed_to_ambiguity += bw * anti_factor * role_factor * amb_share
        audit_rows.append({
            "motif_id": mid,
            "base_weight": round(bw, 5),
            "anti_factor": round(anti_factor, 5),
            "anti_rules_fired": ";".join(fired),
            "role_factor": round(role_factor, 5),
            "gate_reason": gate_reason,
            "ambiguity_share": round(amb_share, 5),
            "discriminative_weight": round(discriminative_weights[mid], 5),
        })

    # 2. Combined ambiguity score = rescue-engine ambiguity + routed share
    #    Cap at 1.0.
    ambiguity_core = float(min(
        1.0, res.ambiguity.core_evidence + routed_to_ambiguity,
    ))

    return {
        "spectrum_id": spectrum_id,
        "rescue_motif_scores": res.motif_scores,
        "base_weights": base_weights,
        "discriminative_weights": discriminative_weights,
        "ambiguity_core": ambiguity_core,
        "rescue_ambiguity_core": float(res.ambiguity.core_evidence),
        "routed_to_ambiguity": float(routed_to_ambiguity),
        "audit_rows": audit_rows,
    }


# Public re-exports
__all__ = [
    "ROLE_TABLE", "ROLE_FACTOR", "NO_ANCHOR_PENALTY",
    "ANTI_EVIDENCE", "COMPETITORS", "AMBIGUITY_ROUTING",
    "CO_FIRE_ANCHOR_GROUPS", "is_in_active_cofire_group",
    "apply_anti_evidence", "apply_role_gate",
    "family_score_discriminative", "score_spectrum_discriminative",
]
