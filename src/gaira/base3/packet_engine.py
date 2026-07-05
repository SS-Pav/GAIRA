"""gaira_base_3 packet engine.

Defines the packet (subfamily) ontology layer between motif and family.

Each packet is a chemically coherent group of motifs with:
  - anchor_motifs: chemistry-specific; drive packet identity
  - support_motifs: helpful but not enough alone
  - background_motifs: broad indicators; capped contribution
  - anti_evidence_rules: motif-level conditions that suppress packet
  - competitor_packets: packets commonly confused; competition-aware
  - allowed_coexistence_packets: packets that legitimately co-fire
  - ambiguity_routes: when packet should route weight to ambiguity instead

Scoring rule:
  packet_score = (anchor_sum + 0.5*support_sum + 0.2*background_sum)
               × anti_evidence_factor
               × competitor_suppression
  capped to [0, 1]

A packet is "validly active" when it has at least one anchor motif firing
above PACKET_ANCHOR_VALID_THRESHOLD (default 0.015). Packets without a
valid anchor fire are dampened by NO_ANCHOR_PACKET_CAP.

Family scoring: family_score = sum over packets that map to family of
(packet_score × packet_to_family_weight).

This module is ADDITIVE on top of gaira_base_2. It does NOT modify any
gaira_base / gaira_base_2 module.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

# A packet's anchor motif weight must reach this to count as a valid
# anchor fire. Raised from 0.015 to 0.030 after first iteration showed
# that single-band motifs like thiol_C_S_660 and disulfide_S_S_510 fire
# at 0.04-0.08 on non-sulfur references (engine BAND_FLOOR=1e-3 is
# permissive). At 0.030, only genuinely-firing chemistry-specific anchors
# pass the validity check.
PACKET_ANCHOR_VALID_THRESHOLD: float = 0.030

# Packets without any valid anchor get this cap on their final score.
# Tightened from 0.50 to 0.35 to push non-anchored packets further down
# the ranking.
NO_ANCHOR_PACKET_CAP: float = 0.35

# Aggregation strategy inside a packet:
#   - anchors: MAX (was SUM) — prevents weak fires of multiple anchors
#     in a single packet from accumulating into a winning score
#   - supports: SUM × SUPPORT_WEIGHT_IN_PACKET (unchanged)
#   - backgrounds: SUM × BACKGROUND_WEIGHT_IN_PACKET (unchanged)
ANCHOR_AGGREGATION_MODE:    str   = "MAX"   # "SUM" or "MAX"
ANCHOR_WEIGHT_IN_PACKET:    float = 1.00
SUPPORT_WEIGHT_IN_PACKET:   float = 0.50   # second iteration tested 0.25 but
                                              # family-derived hit rates dropped;
                                              # keep at 0.50 for family fidelity
BACKGROUND_WEIGHT_IN_PACKET: float = 0.20

# Competitor suppression: if a competitor packet's preliminary score is
# substantially higher than this packet's, suppress this packet.
# Tightened from 1.50/0.55 to 1.30/0.40 — competitor wins more easily.
COMPETITOR_DOMINANCE_RATIO:  float = 1.30
COMPETITOR_SUPPRESSION_FACTOR: float = 0.40


# ─────────────────────────────────────────────────────────────────────
# PACKET ONTOLOGY — 31 packets covering current motif registry
# ─────────────────────────────────────────────────────────────────────
#
# Notation:
#   anchor_motifs      list of motif_ids that DEFINE the packet identity
#   support_motifs     helpful but shared chemistry
#   background_motifs  broad indicators (e.g. amide_I, lipid_C_H_bend)
#
#   anti_evidence_rules:
#     {"rule": "REQUIRES_MOTIF_COBAND",  "target": <motif_id>, "min": w, "penalty": p}
#     {"rule": "SUPPRESS_IF_MOTIF",       "target": <motif_id>, "min": w, "penalty": p}
#
#   competitor_packets: packet_ids that are commonly confused with this one
#   allowed_coexistence_packets: packet_ids that may legitimately co-fire
#   ambiguity_routes: explicit ambiguity routing rules

PACKET_REGISTRY: dict[str, dict] = {
    # ── PURINE SYSTEM ────────────────────────────────────────────────
    "purine_adenine_packet": {
        "description": "Adenine-specific purine chemistry "
                       "(728 ring + 1255 C-N + 1480 ring stretch)",
        "anchor_motifs": ["adenine_specific_anchor_motif"],
        "support_motifs": ["purine_ring_breathing_720_735",
                            "nucleobase_in_plane_ring_1320_1340"],
        "background_motifs": [],
        "anti_evidence_rules": [
            {"rule": "SUPPRESS_IF_MOTIF",
             "target": "uric_acid_full_signature",
             "min": 0.020, "penalty": 0.70},
            {"rule": "SUPPRESS_IF_MOTIF",
             "target": "hypoxanthine_signature",
             "min": 0.020, "penalty": 0.65},
        ],
        "competitor_packets": ["purine_metabolite_ua_packet",
                                "purine_metabolite_hx_packet",
                                "purine_metabolite_xanth_packet",
                                "aromatic_residue_packet"],
        "allowed_coexistence_packets": ["purine_shared_ring_packet",
                                          "phosphate_backbone_packet"],
        "ambiguity_routes": [],
    },
    "purine_guanine_packet": {
        "description": "Guanine-specific purine chemistry "
                       "(651 + 1326 + 1486 ring marks)",
        "anchor_motifs": ["guanine_specific_motif"],
        "support_motifs": ["purine_ring_breathing_720_735",
                            "nucleobase_in_plane_ring_1320_1340"],
        "background_motifs": [],
        "anti_evidence_rules": [
            {"rule": "SUPPRESS_IF_MOTIF",
             "target": "uric_acid_full_signature",
             "min": 0.020, "penalty": 0.65},
        ],
        "competitor_packets": ["purine_metabolite_ua_packet"],
        "allowed_coexistence_packets": ["purine_shared_ring_packet"],
        "ambiguity_routes": [],
    },
    "purine_metabolite_ua_packet": {
        "description": "Uric acid (635 + 891 + 1006 + 1340 4-band signature)",
        "anchor_motifs": ["uric_acid_full_signature"],
        "support_motifs": ["purine_ring_breathing_720_735",
                            "nucleobase_in_plane_ring_1320_1340"],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": ["purine_adenine_packet",
                                "purine_guanine_packet"],
        "allowed_coexistence_packets": ["purine_shared_ring_packet",
                                          "purine_metabolite_hx_packet"],
        "ambiguity_routes": [],
    },
    "purine_metabolite_hx_packet": {
        "description": "Hypoxanthine (635 + 891 ring + 1340 marks)",
        "anchor_motifs": ["hypoxanthine_signature"],
        "support_motifs": ["purine_ring_breathing_720_735"],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": ["purine_adenine_packet"],
        "allowed_coexistence_packets": ["purine_shared_ring_packet",
                                          "purine_metabolite_ua_packet"],
        "ambiguity_routes": [],
    },
    "purine_metabolite_xanth_packet": {
        "description": "Xanthine (650 + 1340 ring marks)",
        "anchor_motifs": ["xanthine_signature"],
        "support_motifs": ["purine_ring_breathing_720_735"],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": ["purine_adenine_packet"],
        "allowed_coexistence_packets": ["purine_shared_ring_packet"],
        "ambiguity_routes": [],
    },
    "purine_shared_ring_packet": {
        "description": "Shared purine ring chemistry (720-735 ring breathing); "
                       "ambiguity-support packet that contributes to multiple "
                       "purine packets without claiming a specific identity",
        "anchor_motifs": [],   # NO single-motif anchor; this packet IS the ambiguity
        "support_motifs": ["purine_ring_breathing_720_735"],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": [],
        "allowed_coexistence_packets": ["purine_adenine_packet",
                                          "purine_guanine_packet",
                                          "purine_metabolite_ua_packet",
                                          "purine_metabolite_hx_packet",
                                          "purine_metabolite_xanth_packet"],
        "ambiguity_routes": [
            {"rule": "ROUTE_TO_AMBIGUITY_IF_LONELY", "share": 0.50},
        ],
    },

    # ── PYRIMIDINE SYSTEM ────────────────────────────────────────────
    "pyrimidine_thymine_packet": {
        "description": "Thymine (790 + 1245 + 1376 ring + methyl marks)",
        "anchor_motifs": ["thymine_specific_motif", "dna_methylation_marker_790"],
        "support_motifs": ["pyrimidine_ring_breathing_780_800",
                            "nucleobase_in_plane_ring_1320_1340"],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": ["pyrimidine_cytosine_packet",
                                "pyrimidine_uracil_like_packet"],
        "allowed_coexistence_packets": [],
        "ambiguity_routes": [],
    },
    "pyrimidine_cytosine_packet": {
        "description": "Cytosine (785 + 1230 + 1295 ring marks)",
        "anchor_motifs": ["cytosine_specific_motif"],
        "support_motifs": ["pyrimidine_ring_breathing_780_800",
                            "nucleobase_in_plane_ring_1320_1340"],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": ["pyrimidine_thymine_packet"],
        "allowed_coexistence_packets": [],
        "ambiguity_routes": [],
    },
    "pyrimidine_uracil_like_packet": {
        "description": "Uracil-like pyrimidine (780-800 ring) without thymine "
                       "methyl or cytosine NH2 — generic pyrimidine reading",
        "anchor_motifs": ["pyrimidine_ring_breathing_780_800"],
        "support_motifs": ["nucleobase_in_plane_ring_1320_1340"],
        "background_motifs": [],
        "anti_evidence_rules": [
            {"rule": "SUPPRESS_IF_MOTIF",
             "target": "thymine_specific_motif",
             "min": 0.020, "penalty": 0.50},
            {"rule": "SUPPRESS_IF_MOTIF",
             "target": "cytosine_specific_motif",
             "min": 0.020, "penalty": 0.50},
        ],
        "competitor_packets": ["pyrimidine_thymine_packet",
                                "pyrimidine_cytosine_packet"],
        "allowed_coexistence_packets": [],
        "ambiguity_routes": [],
    },

    # ── LIPID / STEROL SYSTEM ────────────────────────────────────────
    "lipid_acyl_chain_packet": {
        "description": "Generic acyl-chain lipid chemistry "
                       "(1060 C-C + 1300 CH2 twist + 1440 CH2 bend)",
        "anchor_motifs": [],   # cofire-pair anchor handled via base2
        "support_motifs": ["lipid_acyl_C_C_str_1060_1130"],
        "background_motifs": ["lipid_C_H_bend_1440_1460",
                                "lipid_methylene_twist_1300"],
        "anti_evidence_rules": [],
        "competitor_packets": ["sterol_skeleton_packet",
                                "free_fatty_acid_packet"],
        "allowed_coexistence_packets": ["sterol_skeleton_packet",
                                          "free_fatty_acid_packet",
                                          "cholesteryl_ester_packet"],
        "ambiguity_routes": [],
    },
    "free_fatty_acid_packet": {
        "description": "Free fatty acid chemistry "
                       "(1300 + 1440 + 1700 free COOH 3-band)",
        "anchor_motifs": ["free_fatty_acid_carboxyl_anchor_motif"],
        "support_motifs": ["lipid_acyl_C_C_str_1060_1130"],
        "background_motifs": ["lipid_C_H_bend_1440_1460",
                                "lipid_methylene_twist_1300"],
        "anti_evidence_rules": [
            {"rule": "SUPPRESS_IF_MOTIF",
             "target": "amide_I_alpha_helix_beta_sheet_motif",
             "min": 0.025, "penalty": 0.60},
            {"rule": "SUPPRESS_IF_MOTIF",
             "target": "cholesterol_signature",
             "min": 0.025, "penalty": 0.50},
        ],
        "competitor_packets": ["sterol_skeleton_packet",
                                "cholesteryl_ester_packet",
                                "peptide_backbone_packet"],
        "allowed_coexistence_packets": ["lipid_acyl_chain_packet"],
        "ambiguity_routes": [],
    },
    "sterol_skeleton_packet": {
        "description": "Sterol ring chemistry (548 + 615 + 956 sterol-discriminative trio)",
        "anchor_motifs": ["sterol_skeletal_motif", "cholesterol_signature"],
        "support_motifs": [],
        "background_motifs": ["lipid_C_H_bend_1440_1460"],
        "anti_evidence_rules": [],
        "competitor_packets": ["lipid_acyl_chain_packet"],
        "allowed_coexistence_packets": ["lipid_acyl_chain_packet",
                                          "cholesteryl_ester_packet",
                                          "mixed_sterol_lipid_packet"],
        "ambiguity_routes": [],
    },
    "cholesteryl_ester_packet": {
        "description": "Cholesteryl ester chemistry (548+615+1730 sterol+ester 3-band)",
        "anchor_motifs": ["cholesteryl_ester_discriminator_motif"],
        "support_motifs": ["sterol_skeletal_motif", "cholesterol_signature",
                            "lipid_acyl_C_C_str_1060_1130"],
        "background_motifs": ["lipid_C_H_bend_1440_1460",
                                "lipid_methylene_twist_1300"],
        "anti_evidence_rules": [],
        "competitor_packets": ["sterol_skeleton_packet",
                                "free_fatty_acid_packet"],
        "allowed_coexistence_packets": ["sterol_skeleton_packet",
                                          "lipid_acyl_chain_packet",
                                          "mixed_sterol_lipid_packet"],
        "ambiguity_routes": [],
    },
    "mixed_sterol_lipid_packet": {
        "description": "Triglyceride / mixed sterol+acyl chemistry "
                       "(neutral_lipid_triglyceride + sterol bands)",
        "anchor_motifs": ["neutral_lipid_triglyceride_motif"],
        "support_motifs": ["lipid_acyl_C_C_str_1060_1130",
                            "sterol_skeletal_motif"],
        "background_motifs": ["lipid_C_H_bend_1440_1460",
                                "lipid_methylene_twist_1300"],
        "anti_evidence_rules": [],
        "competitor_packets": ["sterol_skeleton_packet",
                                "cholesteryl_ester_packet"],
        "allowed_coexistence_packets": ["sterol_skeleton_packet",
                                          "cholesteryl_ester_packet",
                                          "lipid_acyl_chain_packet"],
        "ambiguity_routes": [],
    },

    # ── GLYCAN / PHOSPHATE SYSTEM ────────────────────────────────────
    "monosaccharide_packet": {
        "description": "Monosaccharide chemistry "
                       "(850 alpha + 905 beta + 1130 C-O 3-band)",
        "anchor_motifs": ["monosaccharide_anomeric_anchor_motif"],
        "support_motifs": ["glycan_pyranose_ring_skeletal_850_950",
                            "free_saccharide_motif"],
        "background_motifs": [],
        "anti_evidence_rules": [
            {"rule": "SUPPRESS_IF_MOTIF",
             "target": "amide_III_protein_backbone_1230_1280",
             "min": 0.025, "penalty": 0.55},
        ],
        "competitor_packets": ["glycan_polysaccharide_packet"],
        "allowed_coexistence_packets": ["glycan_polysaccharide_packet"],
        "ambiguity_routes": [],
    },
    "glycan_polysaccharide_packet": {
        "description": "Polysaccharide / disaccharide chemistry "
                       "(glycan_pyranose ring + glycosidic C-O-C)",
        "anchor_motifs": [],   # cofire pair (pyranose+free_sacch) handles this
        "support_motifs": ["glycan_pyranose_ring_skeletal_850_950",
                            "glycan_glycosidic_C_O_C_1020_1100",
                            "free_saccharide_motif"],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": ["monosaccharide_packet",
                                "phosphate_backbone_packet"],
        "allowed_coexistence_packets": ["monosaccharide_packet",
                                          "sugar_phosphate_packet"],
        "ambiguity_routes": [],
    },
    "sugar_phosphate_packet": {
        "description": "Sugar-phosphate skeletal chemistry (870-900 + phosphate)",
        "anchor_motifs": [],   # no single anchor; chemistry IS the cross-axis
        "support_motifs": ["sugar_phosphate_skeletal_870_900",
                            "phosphate_PO_asym_str_1240"],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": ["phosphate_backbone_packet",
                                "monosaccharide_packet"],
        "allowed_coexistence_packets": ["phosphate_backbone_packet",
                                          "glycan_polysaccharide_packet"],
        "ambiguity_routes": [],
    },
    "phosphate_backbone_packet": {
        "description": "DNA/RNA phosphate backbone (1080 + 1240 + DNA composite)",
        "anchor_motifs": ["dna_composite_motif",
                            "phosphate_PO_asym_str_1240"],
        "support_motifs": ["phosphate_PO2_sym_str_1080"],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": ["sugar_phosphate_packet",
                                "glycan_polysaccharide_packet"],
        "allowed_coexistence_packets": ["sugar_phosphate_packet",
                                          "purine_adenine_packet",
                                          "purine_guanine_packet",
                                          "pyrimidine_thymine_packet",
                                          "pyrimidine_cytosine_packet"],
        "ambiguity_routes": [],
    },
    "glycan_phosphate_ambiguity_packet": {
        "description": "1020-1100 collision zone (glycan glycosidic vs phosphate)",
        "anchor_motifs": [],
        "support_motifs": ["glycan_glycosidic_C_O_C_1020_1100",
                            "phosphate_PO2_sym_str_1080"],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": [],
        "allowed_coexistence_packets": ["glycan_polysaccharide_packet",
                                          "phosphate_backbone_packet"],
        "ambiguity_routes": [
            {"rule": "ROUTE_TO_AMBIGUITY_IF_LONELY", "share": 0.50},
        ],
    },

    # ── PROTEIN / AMINO ACID SYSTEM ──────────────────────────────────
    "peptide_backbone_packet": {
        "description": "Polypeptide backbone (amide_I + amide_III + amide_II co-fire)",
        "anchor_motifs": [],   # no single ANCHOR; cofire pair IS the anchor
        "support_motifs": ["amide_I_alpha_helix_beta_sheet_motif",
                            "amide_III_protein_backbone_1230_1280",
                            "amide_II_motif"],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": ["free_amino_acid_packet"],
        "allowed_coexistence_packets": ["aromatic_residue_packet",
                                          "sulfur_amino_acid_packet",
                                          "amide_aromatic_overlap_packet"],
        "ambiguity_routes": [],
    },
    "aromatic_residue_packet": {
        "description": "Aromatic side chains (Phe 1003 + Tyr 830/850 doublet)",
        "anchor_motifs": ["phenylalanine_ring_1003",
                            "tyrosine_doublet_830_850"],
        "support_motifs": [],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": ["purine_adenine_packet"],
        "allowed_coexistence_packets": ["peptide_backbone_packet",
                                          "free_amino_acid_packet",
                                          "amide_aromatic_overlap_packet"],
        "ambiguity_routes": [],
    },
    "free_amino_acid_packet": {
        "description": "Free amino acid (amide-bearing small molecule); "
                       "amide_III without strong amide_I OR isolated amide_I",
        "anchor_motifs": [],   # no single ANCHOR; pattern IS amide w/o full peptide
        "support_motifs": ["amide_III_protein_backbone_1230_1280",
                            "amide_I_alpha_helix_beta_sheet_motif"],
        "background_motifs": [],
        "anti_evidence_rules": [
            # If real protein cofire active (amide_I + amide_III + amide_II co-fire
            # at high level), this is peptide, not free AA
            {"rule": "SUPPRESS_IF_MOTIF",
             "target": "amide_II_motif",
             "min": 0.030, "penalty": 0.60},
        ],
        "competitor_packets": ["peptide_backbone_packet"],
        "allowed_coexistence_packets": ["aromatic_residue_packet",
                                          "sulfur_amino_acid_packet",
                                          "glutamate_packet"],
        "ambiguity_routes": [],
    },
    "sulfur_amino_acid_packet": {
        "description": "Sulfur-containing AA (Cys, Met, GSH) — C-S 660 + S-S 510",
        "anchor_motifs": ["thiol_C_S_str_660_motif",
                            "disulfide_S_S_str_500_550",
                            "glutathione_GSH_motif"],
        "support_motifs": ["amide_III_protein_backbone_1230_1280"],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": [],
        "allowed_coexistence_packets": ["peptide_backbone_packet",
                                          "free_amino_acid_packet",
                                          "ergothioneine_packet"],
        "ambiguity_routes": [],
    },
    "amide_aromatic_overlap_packet": {
        "description": "Real protein with aromatic side chains "
                       "(amide cofire + aromatic anchor)",
        "anchor_motifs": [],
        "support_motifs": ["phenylalanine_ring_1003",
                            "tyrosine_doublet_830_850",
                            "amide_I_alpha_helix_beta_sheet_motif",
                            "amide_III_protein_backbone_1230_1280"],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": [],
        "allowed_coexistence_packets": ["peptide_backbone_packet",
                                          "aromatic_residue_packet"],
        "ambiguity_routes": [],
    },

    # ── METABOLIC SMALL MOLECULES ────────────────────────────────────
    "creatine_creatinine_packet": {
        "description": "Creatine / creatinine (605 + 685 ring doublet)",
        "anchor_motifs": ["creatine_creatinine_motif"],
        "support_motifs": [],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": [],
        "allowed_coexistence_packets": [],
        "ambiguity_routes": [],
    },
    "ergothioneine_packet": {
        "description": "Ergothioneine (imidazole-thiol diet metabolite)",
        "anchor_motifs": ["ergothioneine_signature"],
        "support_motifs": ["thiol_C_S_str_660_motif"],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": ["sulfur_amino_acid_packet"],
        "allowed_coexistence_packets": ["sulfur_amino_acid_packet"],
        "ambiguity_routes": [],
    },
    "glutamate_packet": {
        "description": "Glutamate / glutamine (870+1340+1410 Glx 3-band)",
        "anchor_motifs": ["glutamate_glutamine_motif"],
        "support_motifs": ["amide_III_protein_backbone_1230_1280"],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": ["free_amino_acid_packet"],
        "allowed_coexistence_packets": ["free_amino_acid_packet"],
        "ambiguity_routes": [],
    },
    "citrate_packet": {
        "description": "Citrate / TCA-cycle dicarboxylate (950 + 1390)",
        "anchor_motifs": ["citrate_as_biology_motif"],
        "support_motifs": [],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": [],
        "allowed_coexistence_packets": [],
        "ambiguity_routes": [
            {"rule": "ROUTE_TO_AMBIGUITY_IF_SUBSTRATE_CONTEXT", "share": 0.0},
        ],
    },

    # ── HEME / RESONANT ──────────────────────────────────────────────
    "heme_resonance_packet": {
        "description": "Cytochrome c heme resonance (Raman-resonant heme bands)",
        "anchor_motifs": ["cytochrome_c_resonance_motif"],
        "support_motifs": [],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": [],
        "allowed_coexistence_packets": ["peptide_backbone_packet",
                                          "sulfur_amino_acid_packet"],
        "ambiguity_routes": [],
    },

    # ── CONTROL / AMBIGUITY ──────────────────────────────────────────
    "collision_packet_1020_1080": {
        "description": "Multi-candidate 1020-1080 collision zone "
                       "(glycan/phos/lipid C-O overlap)",
        "anchor_motifs": [],
        "support_motifs": [],
        "background_motifs": ["glycan_glycosidic_C_O_C_1020_1100",
                                "phosphate_PO2_sym_str_1080"],
        "anti_evidence_rules": [],
        "competitor_packets": [],
        "allowed_coexistence_packets": [],
        "ambiguity_routes": [
            {"rule": "ROUTE_TO_AMBIGUITY_IF_LONELY", "share": 1.0},
        ],
    },
    "collision_packet_1300_1400": {
        "description": "Multi-candidate 1300-1400 collision zone",
        "anchor_motifs": ["collision_1300_1400_multi_candidate_motif"],
        "support_motifs": [],
        "background_motifs": [],
        "anti_evidence_rules": [],
        "competitor_packets": [],
        "allowed_coexistence_packets": [],
        "ambiguity_routes": [
            {"rule": "ROUTE_TO_AMBIGUITY_IF_LONELY", "share": 1.0},
        ],
    },
    "generic_ambiguity_packet": {
        "description": "Substrate / buffer artifacts and other non-biology signal",
        "anchor_motifs": [],
        "support_motifs": [],
        "background_motifs": ["citrate_baseline_artifact_motif",
                                "purine_HX_lipid_choline_715_overlap_ambiguity",
                                "phosphatidylcholine_choline_head_715"],
        "anti_evidence_rules": [],
        "competitor_packets": [],
        "allowed_coexistence_packets": [],
        "ambiguity_routes": [],
    },
}


# ─────────────────────────────────────────────────────────────────────
# PACKET → FAMILY MAPPING (packets are the new primary; families derive)
# ─────────────────────────────────────────────────────────────────────

PACKET_TO_FAMILY: dict[str, list[tuple[str, float]]] = {
    # purine
    "purine_adenine_packet":          [("purine_nucleotide", 1.0)],
    "purine_guanine_packet":          [("purine_nucleotide", 1.0)],
    "purine_metabolite_ua_packet":    [("purine_metabolite", 1.0)],
    "purine_metabolite_hx_packet":    [("purine_metabolite", 1.0)],
    "purine_metabolite_xanth_packet": [("purine_metabolite", 1.0)],
    "purine_shared_ring_packet":      [("purine_nucleotide", 0.40),
                                         ("purine_metabolite", 0.40),
                                         ("ambiguity_artifact", 0.30)],

    # pyrimidine
    "pyrimidine_thymine_packet":   [("pyrimidine_nucleotide", 1.0)],
    "pyrimidine_cytosine_packet":  [("pyrimidine_nucleotide", 1.0)],
    "pyrimidine_uracil_like_packet": [("pyrimidine_nucleotide", 1.0)],

    # lipid / sterol
    "lipid_acyl_chain_packet":     [("lipid_acyl_membrane", 1.0)],
    "free_fatty_acid_packet":      [("lipid_acyl_membrane", 1.0)],
    "sterol_skeleton_packet":      [("sterol_neutral_lipid", 1.0)],
    "cholesteryl_ester_packet":    [("sterol_neutral_lipid", 0.70),
                                      ("lipid_acyl_membrane", 0.50)],
    "mixed_sterol_lipid_packet":   [("sterol_neutral_lipid", 0.70),
                                      ("lipid_acyl_membrane", 0.40)],

    # glycan / phosphate
    "monosaccharide_packet":           [("glycan_carbohydrate", 1.0)],
    "glycan_polysaccharide_packet":    [("glycan_carbohydrate", 1.0)],
    "sugar_phosphate_packet":          [("phosphate_nucleic_adjacent", 0.70),
                                          ("glycan_carbohydrate", 0.40)],
    "phosphate_backbone_packet":       [("phosphate_nucleic_adjacent", 1.0)],
    "glycan_phosphate_ambiguity_packet": [("ambiguity_artifact", 0.70),
                                            ("glycan_carbohydrate", 0.30),
                                            ("phosphate_nucleic_adjacent", 0.30)],

    # protein / AA
    "peptide_backbone_packet":     [("protein_peptide_backbone", 1.0)],
    "aromatic_residue_packet":     [("aromatic_residue", 1.0)],
    "free_amino_acid_packet":      [("metabolic_small_molecule", 0.70),
                                      ("protein_peptide_backbone", 0.40)],
    "sulfur_amino_acid_packet":    [("sulfur_thiol_redox", 1.0),
                                      ("protein_peptide_backbone", 0.30)],
    "amide_aromatic_overlap_packet": [("protein_peptide_backbone", 0.70),
                                        ("aromatic_residue", 0.50)],

    # metabolic small molecules
    "creatine_creatinine_packet": [("metabolic_small_molecule", 1.0)],
    "ergothioneine_packet":       [("sulfur_thiol_redox", 0.70),
                                     ("metabolic_small_molecule", 0.50)],
    "glutamate_packet":           [("metabolic_small_molecule", 1.0),
                                     ("protein_peptide_backbone", 0.30)],
    "citrate_packet":             [("metabolic_small_molecule", 1.0),
                                     ("ambiguity_artifact", 0.30)],

    # heme
    "heme_resonance_packet":      [("protein_peptide_backbone", 0.50),
                                     ("sulfur_thiol_redox", 0.50)],

    # ambiguity
    "collision_packet_1020_1080": [("ambiguity_artifact", 1.0)],
    "collision_packet_1300_1400": [("ambiguity_artifact", 1.0)],
    "generic_ambiguity_packet":   [("ambiguity_artifact", 1.0)],
}


# ─────────────────────────────────────────────────────────────────────
# Derived: motif → packets
# ─────────────────────────────────────────────────────────────────────

def build_motif_to_packet() -> dict[str, list[tuple[str, str]]]:
    """For each motif, list (packet_id, role_in_packet) entries.
    role is one of: ANCHOR / SUPPORT / BACKGROUND."""
    out: dict[str, list[tuple[str, str]]] = {}
    for pid, p in PACKET_REGISTRY.items():
        for mid in p.get("anchor_motifs", []):
            out.setdefault(mid, []).append((pid, "ANCHOR"))
        for mid in p.get("support_motifs", []):
            out.setdefault(mid, []).append((pid, "SUPPORT"))
        for mid in p.get("background_motifs", []):
            out.setdefault(mid, []).append((pid, "BACKGROUND"))
    return out


# ─────────────────────────────────────────────────────────────────────
# PACKET SCORING
# ─────────────────────────────────────────────────────────────────────

def _packet_anti_evidence_factor(
    packet_id: str, motif_weights: dict[str, float],
) -> tuple[float, list[str]]:
    """Apply packet-level anti-evidence rules. Returns (factor, fired)."""
    pkt = PACKET_REGISTRY.get(packet_id, {})
    rules = pkt.get("anti_evidence_rules", [])
    factor = 1.0
    fired: list[str] = []
    for r in rules:
        if r["rule"] == "REQUIRES_MOTIF_COBAND":
            tw = motif_weights.get(r["target"], 0.0)
            if tw < r["min"]:
                factor *= (1.0 - r["penalty"])
                fired.append(f"REQUIRES_COBAND:{r['target']}")
        elif r["rule"] == "SUPPRESS_IF_MOTIF":
            tw = motif_weights.get(r["target"], 0.0)
            if tw >= r["min"]:
                factor *= (1.0 - r["penalty"])
                fired.append(f"SUPPRESS_IF_MOTIF:{r['target']}")
    return float(factor), fired


def _packet_has_valid_anchor(
    packet_id: str, motif_weights: dict[str, float],
) -> tuple[bool, list[str]]:
    """True iff at least one anchor motif fires above PACKET_ANCHOR_VALID_THRESHOLD."""
    pkt = PACKET_REGISTRY.get(packet_id, {})
    fired_anchors = []
    for mid in pkt.get("anchor_motifs", []):
        if motif_weights.get(mid, 0.0) >= PACKET_ANCHOR_VALID_THRESHOLD:
            fired_anchors.append(mid)
    return bool(fired_anchors), fired_anchors


def compute_packet_scores(
    motif_weights: dict[str, float],
) -> dict[str, dict]:
    """Compute packet scores from motif weights.

    Returns dict[packet_id, {
        "score": float,
        "anchor_sum": float,
        "support_sum": float,
        "background_sum": float,
        "has_valid_anchor": bool,
        "anti_factor": float,
        "anti_fired": [...],
        "competitor_factor": float,  # filled in below after first pass
        "ambiguity_routed": float,
        "fired_anchors": [...],
    }]
    """
    # First pass: per-packet preliminary score
    prelim: dict[str, dict] = {}
    for pid, p in PACKET_REGISTRY.items():
        anchor_weights_list = [motif_weights.get(m, 0.0)
                                for m in p.get("anchor_motifs", [])]
        if ANCHOR_AGGREGATION_MODE == "MAX":
            anchor_aggregate = max(anchor_weights_list, default=0.0)
        else:
            anchor_aggregate = sum(anchor_weights_list)
        anchor_sum = sum(anchor_weights_list)  # for audit
        support_sum = sum(motif_weights.get(m, 0.0)
                          for m in p.get("support_motifs", []))
        bg_sum = sum(motif_weights.get(m, 0.0)
                     for m in p.get("background_motifs", []))
        has_anchor, fired_anchors = _packet_has_valid_anchor(pid, motif_weights)
        anti_factor, anti_fired = _packet_anti_evidence_factor(pid, motif_weights)

        raw = (
            ANCHOR_WEIGHT_IN_PACKET * anchor_aggregate
            + SUPPORT_WEIGHT_IN_PACKET * support_sum
            + BACKGROUND_WEIGHT_IN_PACKET * bg_sum
        )
        # No-anchor cap (no single anchor above threshold AND no anchor motifs defined)
        if not has_anchor:
            raw *= NO_ANCHOR_PACKET_CAP
        raw *= anti_factor

        prelim[pid] = {
            "raw_score": float(np.clip(raw, 0.0, 1.0)),
            "anchor_sum": float(anchor_sum),
            "anchor_max": float(anchor_aggregate),
            "support_sum": float(support_sum),
            "background_sum": float(bg_sum),
            "has_valid_anchor": bool(has_anchor),
            "fired_anchors": fired_anchors,
            "anti_factor": float(anti_factor),
            "anti_fired": anti_fired,
        }

    # Second pass: competitor suppression. If competitor packet is
    # COMPETITOR_DOMINANCE_RATIO times stronger AND has anchor, suppress
    # this packet.
    final: dict[str, dict] = {}
    for pid, p in PACKET_REGISTRY.items():
        my = prelim[pid]
        my_score = my["raw_score"]
        suppressed = False
        suppress_factor = 1.0
        suppressed_by = []
        for cpid in p.get("competitor_packets", []):
            comp = prelim.get(cpid)
            if comp is None:
                continue
            if not comp["has_valid_anchor"]:
                continue
            if comp["raw_score"] >= COMPETITOR_DOMINANCE_RATIO * my_score and my_score > 0:
                suppress_factor *= COMPETITOR_SUPPRESSION_FACTOR
                suppressed = True
                suppressed_by.append(cpid)
        # ambiguity routing
        ambig_routed = 0.0
        for r in p.get("ambiguity_routes", []):
            if r["rule"] == "ROUTE_TO_AMBIGUITY_IF_LONELY":
                # If no allied packet has anchor active, this packet's
                # signal is "lonely" — route a share to ambiguity.
                allies = p.get("allowed_coexistence_packets", [])
                ally_has_anchor = any(
                    prelim.get(a, {}).get("has_valid_anchor", False)
                    for a in allies
                )
                if not ally_has_anchor:
                    ambig_routed += my_score * r["share"]
        final_score = float(np.clip(my_score * suppress_factor - ambig_routed,
                                       0.0, 1.0))

        final[pid] = dict(my)
        final[pid].update({
            "score": final_score,
            "competitor_factor": float(suppress_factor),
            "competitor_suppressed_by": suppressed_by,
            "ambiguity_routed": float(ambig_routed),
        })
    return final


# ─────────────────────────────────────────────────────────────────────
# PACKET → FAMILY AGGREGATION
# ─────────────────────────────────────────────────────────────────────

def compute_family_scores_from_packets(
    packet_scores: dict[str, dict],
) -> dict[str, dict]:
    """Aggregate packet scores into family scores.
    family_score = sum_{p mapping to family} (packet_score × packet_to_family_weight)
    """
    fam_scores: dict[str, dict] = {}
    for pid, info in packet_scores.items():
        s = info["score"]
        if s <= 0:
            continue
        for fam, w in PACKET_TO_FAMILY.get(pid, []):
            row = fam_scores.setdefault(fam, {
                "score": 0.0, "contributing_packets": []
            })
            row["score"] += s * w
            row["contributing_packets"].append((pid, round(s * w, 5)))
    # Cap family scores to [0, 1] for ranking comparability
    for fam in fam_scores:
        fam_scores[fam]["score"] = float(np.clip(fam_scores[fam]["score"], 0.0, 1.0))
    return fam_scores


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

__all__ = [
    "PACKET_REGISTRY", "PACKET_TO_FAMILY",
    "PACKET_ANCHOR_VALID_THRESHOLD", "NO_ANCHOR_PACKET_CAP",
    "ANCHOR_WEIGHT_IN_PACKET", "SUPPORT_WEIGHT_IN_PACKET",
    "BACKGROUND_WEIGHT_IN_PACKET",
    "COMPETITOR_DOMINANCE_RATIO", "COMPETITOR_SUPPRESSION_FACTOR",
    "build_motif_to_packet",
    "compute_packet_scores", "compute_family_scores_from_packets",
]
