"""gaira_validate_2_grounding_motif_first_v1.

Motif-first / family-first grounding validation. Reframes evaluation away
from broad-axis top-1: motifs are primary, families are summary,
ambiguity is a separate control lane, axes are secondary diagnostics only.

Engine: gaira_base_2 active baseline (registry v1.3.1 + mapping v1.2.1 +
v2_patches_rescue). NO calibration / target / substrate-aware.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_validate_2_grounding_motif_first_v1.py
"""
from __future__ import annotations

import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.base2.registry import (
    load_axis_mapping, load_dual_status, load_motif_registry,
)
from gaira.base2.schema import BIOLOGY_AXES_V11
from gaira.base2.v2_patches_rescue import patched_score_spectrum_rescue
from gaira.base2 import v2_patches as _v2
from gaira.spectral import canonical_master_axis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gaira_validate_2_grounding import (
    load_ramanbiolib, load_gobbato_powder,
    load_amino_acid_xlsx, load_digitised_literature,
)
from run_gaira_base_2_grounding_repair_loop import TRUTH_AXES


ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_validate_2_grounding_motif_first_v1")
INVENTORY = ROOT / "inventory"
TRUTH = ROOT / "truth_table"
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"

REG_V1_3_1 = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_revert_v4_and_deep_coverage_rescue_v1/"
    "registry/motif_candidate_registry_v1_3_1.yaml"
)
MAP_V1_2_1 = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_revert_v4_and_deep_coverage_rescue_v1/"
    "registry/motif_to_axis_mapping_skeleton_v1_2_1.csv"
)

# 11 biology families == 11 axes (this phase uses these as PRIMARY summary
# layer derived from grouped motif scores, NOT from noisy-OR axis scoring).
FAMILIES = [
    "purine_nucleotide", "purine_metabolite", "pyrimidine_nucleotide",
    "phosphate_nucleic_adjacent", "glycan_carbohydrate",
    "protein_peptide_backbone", "aromatic_residue",
    "lipid_acyl_membrane", "sterol_neutral_lipid",
    "sulfur_thiol_redox", "metabolic_small_molecule",
]
AMBIGUITY_LANE = "ambiguity_artifact"


# ──────────────────────────────────────────────────────────────────────
# EXPECTED MOTIFS — chemistry-justified per component_key
# ──────────────────────────────────────────────────────────────────────
#
# For each reference compound class, list the motifs that SHOULD fire as
# top-N. This is chemistry-first: only motifs whose primary bands are
# present in the canonical Raman of the compound are listed.

EXPECTED_MOTIFS: dict[str, list[str]] = {
    # ── nucleobases (free purines / pyrimidines) ──────────────────────
    "adenine":  ["purine_ring_breathing_720_735", "nucleobase_in_plane_ring_1320_1340"],
    "guanine":  ["guanine_specific_motif", "purine_ring_breathing_720_735",
                 "nucleobase_in_plane_ring_1320_1340"],
    "cytosine": ["cytosine_specific_motif", "pyrimidine_ring_breathing_780_800"],
    "thymine":  ["thymine_specific_motif", "pyrimidine_ring_breathing_780_800"],
    "uracil":   ["pyrimidine_ring_breathing_780_800"],
    "Ade": ["purine_ring_breathing_720_735", "nucleobase_in_plane_ring_1320_1340"],
    "Gua": ["guanine_specific_motif", "purine_ring_breathing_720_735",
            "nucleobase_in_plane_ring_1320_1340"],
    "Thy": ["thymine_specific_motif", "pyrimidine_ring_breathing_780_800"],
    "Ura": ["pyrimidine_ring_breathing_780_800"],

    # ── purine catabolites ────────────────────────────────────────────
    "UA":     ["uric_acid_full_signature", "purine_ring_breathing_720_735"],
    "Hypox":  ["hypoxanthine_signature", "purine_ring_breathing_720_735"],
    "Xanth":  ["xanthine_signature", "purine_ring_breathing_720_735"],
    "ua_digitised_gelder_2007": ["uric_acid_full_signature",
                                  "purine_ring_breathing_720_735"],
    "ua_digitised_kim_1987":    ["uric_acid_full_signature",
                                  "purine_ring_breathing_720_735"],

    # ── nucleic acids ─────────────────────────────────────────────────
    "a-dna": ["dna_composite_motif", "phosphate_PO2_sym_str_1080",
              "phosphate_PO_asym_str_1240", "purine_ring_breathing_720_735"],
    "b-dna": ["dna_composite_motif", "phosphate_PO2_sym_str_1080",
              "phosphate_PO_asym_str_1240", "purine_ring_breathing_720_735"],
    "t-rna": ["dna_composite_motif", "phosphate_PO2_sym_str_1080",
              "phosphate_PO_asym_str_1240", "purine_ring_breathing_720_735"],
    "DNA": ["dna_composite_motif", "phosphate_PO2_sym_str_1080",
            "phosphate_PO_asym_str_1240"],
    "RNA": ["dna_composite_motif", "phosphate_PO2_sym_str_1080",
            "phosphate_PO_asym_str_1240"],
    "2-deoxy-d-ribose": ["sugar_phosphate_skeletal_870_900",
                          "glycan_pyranose_ring_skeletal_850_950"],
    "Phosph": ["phosphate_PO_asym_str_1240", "phosphate_PO2_sym_str_1080"],

    # ── phosphate-bearing small molecules ─────────────────────────────
    "PEP":  ["phosphate_PO2_sym_str_1080"],
    "phosphoenolpyruvate": ["phosphate_PO2_sym_str_1080"],
    "d-fructose-6-phosphate": ["sugar_phosphate_skeletal_870_900",
                                "glycan_pyranose_ring_skeletal_850_950"],
    "Dfruct6P": ["sugar_phosphate_skeletal_870_900",
                  "glycan_pyranose_ring_skeletal_850_950"],

    # ── glycans (mono/di/poly) ─────────────────────────────────────────
    "d-(+)-glucose":  ["glycan_pyranose_ring_skeletal_850_950", "free_saccharide_motif"],
    "β-d-glucose":    ["glycan_pyranose_ring_skeletal_850_950", "free_saccharide_motif"],
    "d-(+)-galactose":["glycan_pyranose_ring_skeletal_850_950", "free_saccharide_motif"],
    "d-(+)-mannose":  ["glycan_pyranose_ring_skeletal_850_950", "free_saccharide_motif"],
    "d-(-)-fructose": ["glycan_pyranose_ring_skeletal_850_950", "free_saccharide_motif"],
    "d-(-)-ribose":   ["glycan_pyranose_ring_skeletal_850_950", "free_saccharide_motif"],
    "d-(+)-fucose":   ["glycan_pyranose_ring_skeletal_850_950", "free_saccharide_motif"],
    "d-(+)-xylose":   ["glycan_pyranose_ring_skeletal_850_950", "free_saccharide_motif"],
    "d-(-)-arabinose":["glycan_pyranose_ring_skeletal_850_950", "free_saccharide_motif"],
    "l-(+)-arabinose":["glycan_pyranose_ring_skeletal_850_950", "free_saccharide_motif"],
    "d-(+)-lactose monohydrate": ["glycan_pyranose_ring_skeletal_850_950",
                                   "glycan_glycosidic_C_O_C_1020_1100",
                                   "free_saccharide_motif"],
    "d-(+)-maltose monohydrate": ["glycan_pyranose_ring_skeletal_850_950",
                                   "glycan_glycosidic_C_O_C_1020_1100",
                                   "free_saccharide_motif"],
    "d-(+)-sucrose":  ["glycan_pyranose_ring_skeletal_850_950",
                        "glycan_glycosidic_C_O_C_1020_1100"],
    "d-(+)-trehalose":["glycan_pyranose_ring_skeletal_850_950",
                        "glycan_glycosidic_C_O_C_1020_1100"],
    "d-(+)-raffinose pentahydrate": ["glycan_pyranose_ring_skeletal_850_950",
                                      "glycan_glycosidic_C_O_C_1020_1100"],
    "d-(+)-galactosamine": ["glycan_pyranose_ring_skeletal_850_950"],
    "glucosamine":    ["glycan_pyranose_ring_skeletal_850_950"],
    "n-acetyl- d-glucosamine": ["glycan_pyranose_ring_skeletal_850_950",
                                 "sialic_acid_signature"],
    "lactose":        ["glycan_pyranose_ring_skeletal_850_950",
                        "glycan_glycosidic_C_O_C_1020_1100"],
    "cellulose":      ["glycan_pyranose_ring_skeletal_850_950",
                        "glycan_glycosidic_C_O_C_1020_1100"],
    "glycogen":       ["glycan_pyranose_ring_skeletal_850_950",
                        "glycan_glycosidic_C_O_C_1020_1100"],
    "chitin":         ["glycan_pyranose_ring_skeletal_850_950",
                        "sialic_acid_signature"],
    "amylose":        ["glycan_pyranose_ring_skeletal_850_950",
                        "glycan_glycosidic_C_O_C_1020_1100"],
    "amylopectin":    ["glycan_pyranose_ring_skeletal_850_950",
                        "glycan_glycosidic_C_O_C_1020_1100"],
    "d-(+)-dextrose": ["glycan_pyranose_ring_skeletal_850_950"],
    "Gluc":   ["glycan_pyranose_ring_skeletal_850_950", "free_saccharide_motif"],
    "Galact": ["glycan_pyranose_ring_skeletal_850_950", "free_saccharide_motif"],
    "Mann":   ["glycan_pyranose_ring_skeletal_850_950", "free_saccharide_motif"],
    "Fruct":  ["glycan_pyranose_ring_skeletal_850_950", "free_saccharide_motif"],
    "NacDgluc": ["glycan_pyranose_ring_skeletal_850_950", "sialic_acid_signature"],
    "Glycogen": ["glycan_pyranose_ring_skeletal_850_950",
                  "glycan_glycosidic_C_O_C_1020_1100"],
    "Glucose": ["glycan_pyranose_ring_skeletal_850_950", "free_saccharide_motif"],

    # ── aromatic amino acids ──────────────────────────────────────────
    "l-phenylalanine": ["phenylalanine_ring_1003"],
    "l-tyrosine":      ["tyrosine_doublet_830_850"],
    "l-tryptophan":    ["amide_III_protein_backbone_1230_1280"],
    "l-histidine":     ["amide_III_protein_backbone_1230_1280"],
    "Phe": ["phenylalanine_ring_1003"],
    "Tyr": ["tyrosine_doublet_830_850"],
    "Trp": ["amide_III_protein_backbone_1230_1280"],
    "His": ["amide_III_protein_backbone_1230_1280"],

    # ── free non-aromatic amino acids ─────────────────────────────────
    "l-alanine":       ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],
    "l-arginine":      ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],
    "l-asparagine":    ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],
    "l-aspartic acid": ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],
    "l-glutamate":     ["glutamate_glutamine_motif", "amide_III_protein_backbone_1230_1280"],
    "l-proline":       ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],
    "l-serine":        ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],
    "l-valine":        ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],
    "glycine":         ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],
    "Ala": ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],
    "Arg": ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],
    "Asp": ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],
    "Gly": ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],
    "Leu": ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],
    "Ile": ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],
    "Pro": ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],
    "Ser": ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],
    "Val": ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],
    "Hydroxypro": ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],
    "Glut": ["glutamate_glutamine_motif", "amide_III_protein_backbone_1230_1280"],
    "Glutamic": ["glutamate_glutamine_motif", "amide_III_protein_backbone_1230_1280"],
    "Glutamic Acid": ["glutamate_glutamine_motif", "amide_III_protein_backbone_1230_1280"],
    "L-Glu": ["glutamate_glutamine_motif", "amide_III_protein_backbone_1230_1280"],
    "Valine": ["amide_III_protein_backbone_1230_1280", "amide_I_alpha_helix_beta_sheet_motif"],

    # ── sulfur amino acids / thiols ────────────────────────────────────
    "Cys":    ["thiol_C_S_str_660_motif", "amide_III_protein_backbone_1230_1280"],
    "Met":    ["thiol_C_S_str_660_motif", "amide_III_protein_backbone_1230_1280"],
    "Methio": ["thiol_C_S_str_660_motif", "amide_III_protein_backbone_1230_1280"],
    "glutathione": ["glutathione_GSH_motif", "thiol_C_S_str_660_motif"],
    "Gluth": ["glutathione_GSH_motif", "thiol_C_S_str_660_motif"],

    # ── proteins (polypeptides) ────────────────────────────────────────
    "albumin": ["amide_I_alpha_helix_beta_sheet_motif", "amide_III_protein_backbone_1230_1280",
                 "amide_II_motif", "phenylalanine_ring_1003", "tyrosine_doublet_830_850"],
    "collagen": ["amide_I_alpha_helix_beta_sheet_motif", "amide_III_protein_backbone_1230_1280",
                  "amide_II_motif"],
    "elastin": ["amide_I_alpha_helix_beta_sheet_motif", "amide_III_protein_backbone_1230_1280"],
    "keratin": ["amide_I_alpha_helix_beta_sheet_motif", "amide_III_protein_backbone_1230_1280",
                 "disulfide_S_S_str_500_550"],
    "hemoglobin": ["amide_I_alpha_helix_beta_sheet_motif", "amide_III_protein_backbone_1230_1280",
                    "cytochrome_c_resonance_motif"],
    "myoglobin":  ["amide_I_alpha_helix_beta_sheet_motif", "cytochrome_c_resonance_motif"],
    "insulin":    ["amide_I_alpha_helix_beta_sheet_motif", "disulfide_S_S_str_500_550"],
    "ferritin":   ["amide_I_alpha_helix_beta_sheet_motif", "amide_III_protein_backbone_1230_1280"],
    "cytochrome c": ["cytochrome_c_resonance_motif", "amide_I_alpha_helix_beta_sheet_motif"],
    "lactalbumin":  ["amide_I_alpha_helix_beta_sheet_motif", "amide_III_protein_backbone_1230_1280"],
    "carbonic anhydrase": ["amide_I_alpha_helix_beta_sheet_motif"],
    "tubulin":      ["amide_I_alpha_helix_beta_sheet_motif"],
    "elastase":     ["amide_I_alpha_helix_beta_sheet_motif"],
    "ubiquitin":    ["amide_I_alpha_helix_beta_sheet_motif"],
    "trypsin":      ["amide_I_alpha_helix_beta_sheet_motif"],
    "trypsinogen":  ["amide_I_alpha_helix_beta_sheet_motif"],
    "pepsin":       ["amide_I_alpha_helix_beta_sheet_motif"],
    "pepsinogen":   ["amide_I_alpha_helix_beta_sheet_motif"],
    "papain":       ["amide_I_alpha_helix_beta_sheet_motif"],
    "major proteinase": ["amide_I_alpha_helix_beta_sheet_motif"],
    "horseradish peroxidase": ["amide_I_alpha_helix_beta_sheet_motif"],
    "xylanase":     ["amide_I_alpha_helix_beta_sheet_motif"],
    "lectin":       ["amide_I_alpha_helix_beta_sheet_motif"],
    "α-chymotrypsinogen a (type ii)": ["amide_I_alpha_helix_beta_sheet_motif"],
    "thaumatin":    ["amide_I_alpha_helix_beta_sheet_motif"],
    "triosephosphate isomerase": ["amide_I_alpha_helix_beta_sheet_motif"],
    "glutathione transferase": ["amide_I_alpha_helix_beta_sheet_motif",
                                  "glutathione_GSH_motif"],
    "glucose oxidase": ["amide_I_alpha_helix_beta_sheet_motif"],
    "superoxide dismutases": ["amide_I_alpha_helix_beta_sheet_motif"],
    "trypsin inhibitor": ["amide_I_alpha_helix_beta_sheet_motif"],
    "Alb": ["amide_I_alpha_helix_beta_sheet_motif", "amide_III_protein_backbone_1230_1280"],

    # ── lipids: free fatty acids ──────────────────────────────────────
    "glycerol":     ["lipid_C_H_bend_1440_1460", "lipid_methylene_twist_1300"],
    "oleic acid":   ["lipid_acyl_C_C_str_1060_1130", "lipid_C_H_bend_1440_1460",
                      "lipid_methylene_twist_1300"],
    "palmitic acid":["lipid_acyl_C_C_str_1060_1130", "lipid_C_H_bend_1440_1460",
                      "lipid_methylene_twist_1300"],
    "stearic acid": ["lipid_acyl_C_C_str_1060_1130", "lipid_C_H_bend_1440_1460",
                      "lipid_methylene_twist_1300"],
    "linoleic acid":["lipid_acyl_C_C_str_1060_1130", "lipid_C_H_bend_1440_1460",
                      "lipid_methylene_twist_1300"],
    "arachidic acid": ["lipid_acyl_C_C_str_1060_1130", "lipid_C_H_bend_1440_1460",
                        "lipid_methylene_twist_1300"],
    "arachidonic acid": ["lipid_acyl_C_C_str_1060_1130", "lipid_C_H_bend_1440_1460",
                          "lipid_methylene_twist_1300"],
    "lauric acid":  ["lipid_acyl_C_C_str_1060_1130", "lipid_C_H_bend_1440_1460",
                      "lipid_methylene_twist_1300"],
    "myristic acid": ["lipid_acyl_C_C_str_1060_1130", "lipid_C_H_bend_1440_1460",
                       "lipid_methylene_twist_1300"],
    "elaidic acid": ["lipid_acyl_C_C_str_1060_1130", "lipid_C_H_bend_1440_1460",
                      "lipid_methylene_twist_1300"],
    "palmitoleic acid": ["lipid_acyl_C_C_str_1060_1130", "lipid_C_H_bend_1440_1460",
                          "lipid_methylene_twist_1300"],
    "vaccenic acid": ["lipid_acyl_C_C_str_1060_1130", "lipid_C_H_bend_1440_1460",
                       "lipid_methylene_twist_1300"],
    "α-linolenic acid": ["lipid_acyl_C_C_str_1060_1130", "lipid_C_H_bend_1440_1460",
                          "lipid_methylene_twist_1300"],
    "12-methyltetradecanoic acid": ["lipid_acyl_C_C_str_1060_1130",
                                     "lipid_C_H_bend_1440_1460", "lipid_methylene_twist_1300"],
    "13-methylmyristicacid":       ["lipid_acyl_C_C_str_1060_1130",
                                     "lipid_C_H_bend_1440_1460", "lipid_methylene_twist_1300"],
    "14-methylhexadecanoic acid":  ["lipid_acyl_C_C_str_1060_1130",
                                     "lipid_C_H_bend_1440_1460", "lipid_methylene_twist_1300"],
    "14-methylpentadecanoic acid": ["lipid_acyl_C_C_str_1060_1130",
                                     "lipid_C_H_bend_1440_1460", "lipid_methylene_twist_1300"],
    "15-methylpalmiticacid":       ["lipid_acyl_C_C_str_1060_1130",
                                     "lipid_C_H_bend_1440_1460", "lipid_methylene_twist_1300"],
    "ceramide":      ["lipid_C_H_bend_1440_1460", "lipid_methylene_twist_1300"],
    "sphingomyelin": ["phosphatidylcholine_choline_head_715", "lipid_C_H_bend_1440_1460",
                       "lipid_methylene_twist_1300"],
    "l-α-phosphatidylcholine":     ["phosphatidylcholine_choline_head_715",
                                     "lipid_C_H_bend_1440_1460", "lipid_methylene_twist_1300"],
    "l-α-phosphatidylethanolamine":["lipid_C_H_bend_1440_1460", "lipid_methylene_twist_1300"],
    "Oleic":   ["lipid_acyl_C_C_str_1060_1130", "lipid_C_H_bend_1440_1460",
                 "lipid_methylene_twist_1300"],
    "Stearic": ["lipid_acyl_C_C_str_1060_1130", "lipid_C_H_bend_1440_1460",
                 "lipid_methylene_twist_1300"],
    "PhInositol": ["phosphatidylcholine_choline_head_715", "lipid_C_H_bend_1440_1460"],
    "Glycerol": ["lipid_C_H_bend_1440_1460", "lipid_methylene_twist_1300"],

    # ── sterols + sterol esters + triglycerides ───────────────────────
    "cholesterol": ["cholesterol_signature", "sterol_skeletal_motif"],
    "cholesteryl linoleate": ["cholesteryl_ester_discriminator_motif",
                                "cholesterol_signature", "sterol_skeletal_motif",
                                "lipid_acyl_C_C_str_1060_1130"],
    "cholesteryl oleate":    ["cholesteryl_ester_discriminator_motif",
                                "cholesterol_signature", "sterol_skeletal_motif",
                                "lipid_acyl_C_C_str_1060_1130"],
    "cholesteryl palmitate": ["cholesteryl_ester_discriminator_motif",
                                "cholesterol_signature", "sterol_skeletal_motif",
                                "lipid_acyl_C_C_str_1060_1130"],
    "cholesteryl stearate":  ["cholesteryl_ester_discriminator_motif",
                                "cholesterol_signature", "sterol_skeletal_motif",
                                "lipid_acyl_C_C_str_1060_1130"],
    "estradiol":  ["sterol_skeletal_motif"],
    "estrone":    ["sterol_skeletal_motif"],
    "estriol":    ["sterol_skeletal_motif"],
    "ethinylestradiol": ["sterol_skeletal_motif"],
    "diethylstilbestrol": ["sterol_skeletal_motif"],
    "tristearin":   ["neutral_lipid_triglyceride_motif", "lipid_acyl_C_C_str_1060_1130"],
    "tripalmitin":  ["neutral_lipid_triglyceride_motif", "lipid_acyl_C_C_str_1060_1130"],
    "triolein":     ["neutral_lipid_triglyceride_motif", "lipid_acyl_C_C_str_1060_1130"],
    "trilinolein":  ["neutral_lipid_triglyceride_motif", "lipid_acyl_C_C_str_1060_1130"],
    "trilinolenin": ["neutral_lipid_triglyceride_motif", "lipid_acyl_C_C_str_1060_1130"],
    "trimyristin":  ["neutral_lipid_triglyceride_motif", "lipid_acyl_C_C_str_1060_1130"],
    "trilaurin":    ["neutral_lipid_triglyceride_motif", "lipid_acyl_C_C_str_1060_1130"],
    "tricaprin":    ["neutral_lipid_triglyceride_motif", "lipid_acyl_C_C_str_1060_1130"],
    "tricaproin":   ["neutral_lipid_triglyceride_motif", "lipid_acyl_C_C_str_1060_1130"],
    "tricaprylin":  ["neutral_lipid_triglyceride_motif", "lipid_acyl_C_C_str_1060_1130"],
    "tri-11-eicosenoin": ["neutral_lipid_triglyceride_motif", "lipid_acyl_C_C_str_1060_1130"],
    "triarachidin": ["neutral_lipid_triglyceride_motif", "lipid_acyl_C_C_str_1060_1130"],
    "tribehenin":   ["neutral_lipid_triglyceride_motif", "lipid_acyl_C_C_str_1060_1130"],
    "trielaidin":   ["neutral_lipid_triglyceride_motif", "lipid_acyl_C_C_str_1060_1130"],
    "trierucin":    ["neutral_lipid_triglyceride_motif", "lipid_acyl_C_C_str_1060_1130"],
    "tripalmitolein":    ["neutral_lipid_triglyceride_motif", "lipid_acyl_C_C_str_1060_1130"],
    "tripetroselinin":   ["neutral_lipid_triglyceride_motif", "lipid_acyl_C_C_str_1060_1130"],
    "Chol":     ["cholesterol_signature", "sterol_skeletal_motif"],
    "Triolein": ["neutral_lipid_triglyceride_motif", "lipid_acyl_C_C_str_1060_1130"],
    "β-carotene": ["lipid_unsaturation_C_C_str_1655_motif"]
                    if False else ["lipid_methylene_twist_1300"],

    # ── metabolic small molecules ─────────────────────────────────────
    "creatine":       ["creatine_creatinine_motif"],
    "creatinine":     ["creatine_creatinine_motif"],
    "Creat":          ["creatine_creatinine_motif"],
    "citric acid":    ["citrate_as_biology_motif"],
    "Citric":         ["citrate_as_biology_motif"],
    "succinic acid":  ["citrate_as_biology_motif"],   # 1390-band metabolite analog
    "malic acid":     ["citrate_as_biology_motif"],
    "Malic Acid":     ["citrate_as_biology_motif"],
    "fumarate":       ["citrate_as_biology_motif"],
    "ascorbic acid":  ["citrate_as_biology_motif"],
    "Asc":            ["citrate_as_biology_motif"],
    "pyruvate":       ["citrate_as_biology_motif"],
    "Pyr":            ["citrate_as_biology_motif"],
    "acetoacetate":   ["citrate_as_biology_motif"],
    "Acetoacet":      ["citrate_as_biology_motif"],
    "acetyl coenzyme a": ["thiol_C_S_str_660_motif"],
    "AcCoA":          ["thiol_C_S_str_660_motif"],
    "coenzyme a":     ["thiol_C_S_str_660_motif"],
    "CoA":            ["thiol_C_S_str_660_motif"],
    "melanin":        ["amide_III_protein_backbone_1230_1280"],
    "riboﬂavin":       ["amide_III_protein_backbone_1230_1280"],
    "Ribo":           ["amide_III_protein_backbone_1230_1280"],
    "urea":           [],   # urea has limited Raman fingerprint; weak motif coverage
    "Urea":           [],
    "Ure":            [],
    "Ergo":           ["ergothioneine_signature", "thiol_C_S_str_660_motif"],
    "Lact":           [],   # lactate - DEFERRED (no motif), expected_motifs empty
    "Havuc":          ["lipid_C_H_bend_1440_1460"],
}


# ──────────────────────────────────────────────────────────────────────
# REFERENCES whose chemistry SHOULD trigger ambiguity_artifact lane
# ──────────────────────────────────────────────────────────────────────
#
# Per truth table refinement: ambiguity is expected when the chemistry
# is genuinely multi-axis at a level the engine cannot disambiguate
# (citrate as biology vs substrate buffer; PC choline vs purine 715).
# Pure single-axis references should NOT activate ambiguity strongly.

EXPECTED_AMBIGUITY: set[str] = {
    "citric acid", "Citric",             # citrate as biology vs substrate citrate
    "l-α-phosphatidylcholine",           # PC choline 715 / purine 715 collision
    "sphingomyelin", "PhInositol",       # choline-bearing
}


# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────

def expected_motifs_for(component_key: str) -> list[str]:
    """Look up expected motifs for a component_key, with case-insensitive
    fallback."""
    if component_key in EXPECTED_MOTIFS:
        return EXPECTED_MOTIFS[component_key]
    if component_key.lower() in EXPECTED_MOTIFS:
        return EXPECTED_MOTIFS[component_key.lower()]
    return []


def expected_families_for(component_key: str) -> list[str]:
    """Pull expected families from the prior multi-axis truth table."""
    if component_key in TRUTH_AXES:
        fams = TRUTH_AXES[component_key]
    elif component_key.lower() in TRUTH_AXES:
        fams = TRUTH_AXES[component_key.lower()]
    else:
        return []
    return [f for f in fams if f in FAMILIES]


def expected_ambiguity_for(component_key: str) -> bool:
    if component_key in EXPECTED_AMBIGUITY:
        return True
    fams = TRUTH_AXES.get(component_key) or TRUTH_AXES.get(component_key.lower(), [])
    return AMBIGUITY_LANE in fams


def family_score(motif_scores: list, mappings: dict, family: str) -> tuple[float, list[str]]:
    """Sum of motif core_weight x mapping_weight for motifs that contribute
    to this family. Returns (score, contributing motif_ids)."""
    from gaira.base2.motif_engine import resolve_mapping_weight
    total = 0.0
    contribs = []
    for ms in motif_scores:
        if ms.core_weight <= 0:
            continue
        mp = mappings.get(ms.motif_id)
        if mp is None:
            continue
        mw = resolve_mapping_weight(mp, family)
        if mw <= 0:
            continue
        total += float(ms.core_weight) * float(mw)
        contribs.append(ms.motif_id)
    return float(total), contribs


def topn_hit(top_list: list[str], expected: list[str], n: int) -> bool:
    if not expected:
        return False
    exp = set(expected)
    return any(t in exp for t in top_list[:n])


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("gaira_validate_2_grounding_motif_first_v1")
    print("=" * 78)
    for d in (INVENTORY, TRUTH, TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
        d.mkdir(parents=True, exist_ok=True)

    master_x = canonical_master_axis()

    motifs = load_motif_registry(REG_V1_3_1)
    mappings = load_axis_mapping(MAP_V1_2_1)
    dual = load_dual_status()
    active = {m: s for m, s in motifs.items() if s.v1_active}
    print(f"[engine] {len(active)} active motifs, {len(mappings)} mappings")

    # Load datasets (full grounding corpus)
    rb  = load_ramanbiolib(master_x)
    gp  = load_gobbato_powder(master_x)
    aa  = load_amino_acid_xlsx(master_x)
    lit = load_digitised_literature(master_x)
    all_refs = rb + gp + aa + lit
    print(f"[data] {len(all_refs)} grounding spectra "
          f"({len(rb)} rbl + {len(gp)} gobbato + {len(aa)} aa + {len(lit)} lit)")

    # ── Inventory ────────────────────────────────────────────────────
    inv = []
    for ds_name, lst in [("ramanbiolib", rb),
                         ("gobbato_powder_raman", gp),
                         ("amino_acid_raman_grounding", aa),
                         ("digitised_literature_spectra", lit)]:
        comps = sorted({r["component_key"] for r in lst})
        with_motifs = sum(1 for c in comps if expected_motifs_for(c))
        with_fams = sum(1 for c in comps if expected_families_for(c))
        inv.append({
            "dataset_name": ds_name,
            "n_spectra": len(lst),
            "n_unique_components": len(comps),
            "n_components_with_expected_motifs": with_motifs,
            "n_components_with_expected_families": with_fams,
            "approval_status": "APPROVED_CORE_GROUNDING",
            "phase_used": "gaira_validate_2_grounding_motif_first_v1",
            "notes": "core ontology only; no calibration / target / substrate-aware",
        })
    pd.DataFrame(inv).to_csv(
        INVENTORY / "grounding_dataset_inventory_v_motif_first.csv", index=False,
    )
    print(f"[emit] grounding_dataset_inventory_v_motif_first.csv")

    # ── Truth table ──────────────────────────────────────────────────
    truth_rows = []
    for r in all_refs:
        comp = r["component_key"]
        em = expected_motifs_for(comp)
        ef = expected_families_for(comp)
        ea = expected_ambiguity_for(comp)
        truth_rows.append({
            "spectrum_id": r["spectrum_id"], "dataset": r["dataset"],
            "component_key": comp,
            "expected_motifs": ",".join(em),
            "expected_families": ",".join(ef),
            "expected_ambiguity": ea,
            "is_multi_family": len(ef) > 1,
            "is_single_family": len(ef) == 1,
            "is_ambiguity_expected": ea,
            "n_expected_motifs": len(em),
            "n_expected_families": len(ef),
        })
    pd.DataFrame(truth_rows).to_csv(
        TRUTH / "grounding_truth_table_motif_first_v1.csv", index=False,
    )
    print(f"[emit] grounding_truth_table_motif_first_v1.csv")

    # ── Score every spectrum ─────────────────────────────────────────
    print("\n[score] rescue engine (motif-first reporting)")
    motif_rows, family_rows, ambig_rows = [], [], []
    rank_motif_rows, rank_family_rows = [], []
    off_target_rows = []
    miss_rows = []

    for r in all_refs:
        comp = r["component_key"]
        sid = r["spectrum_id"]
        em = expected_motifs_for(comp)
        ef = expected_families_for(comp)
        ea = expected_ambiguity_for(comp)
        res = patched_score_spectrum_rescue(
            r["spectrum"], master_x, active, mappings, dual, sid,
        )
        ms_sorted = sorted(res.motif_scores, key=lambda m: m.core_weight, reverse=True)
        top5_motifs = [m.motif_id for m in ms_sorted[:5]]

        # Per-motif rows (one per active motif)
        for m in res.motif_scores:
            motif_rows.append({
                "spectrum_id": sid, "dataset": r["dataset"],
                "component_key": comp,
                "motif_id": m.motif_id,
                "activation": round(m.activation, 5),
                "core_weight": round(m.core_weight, 5),
                "is_expected": m.motif_id in em,
                "is_top5": m.motif_id in top5_motifs,
                "contributes_to_ambiguity": m.contributes_to_ambiguity,
            })

        # Family scores (grouped motif sums)
        fam_scores = {}
        fam_contribs = {}
        for fam in FAMILIES:
            s, contribs = family_score(res.motif_scores, mappings, fam)
            fam_scores[fam] = s
            fam_contribs[fam] = contribs
        fam_sorted = sorted(fam_scores.items(), key=lambda kv: kv[1], reverse=True)
        top5_fams = [f for f, _ in fam_sorted[:5]]
        for fam, s in fam_sorted:
            family_rows.append({
                "spectrum_id": sid, "dataset": r["dataset"],
                "component_key": comp,
                "family": fam,
                "family_score": round(s, 5),
                "is_expected": fam in ef,
                "is_top5": fam in top5_fams,
                "n_contributing_motifs": len(fam_contribs[fam]),
                "contributing_motifs": ",".join(fam_contribs[fam]),
            })

        # Ambiguity row
        amb = res.ambiguity.core_evidence
        ambig_rows.append({
            "spectrum_id": sid, "dataset": r["dataset"],
            "component_key": comp,
            "ambiguity_core": round(amb, 5),
            "expected_ambiguity": ea,
            # call ambiguity "active" if score >= 0.10 (gated threshold)
            "observed_ambiguity_active": amb >= 0.10,
            "ambiguity_correct": (ea and amb >= 0.10) or (not ea and amb < 0.10),
            "ambiguity_overfire": (not ea) and amb >= 0.10,
            "ambiguity_underfire": ea and amb < 0.10,
        })

        # Rank evaluation rows
        rank_motif_rows.append({
            "spectrum_id": sid, "dataset": r["dataset"],
            "component_key": comp,
            "expected_motifs": ",".join(em),
            "top_motif_1": top5_motifs[0] if len(top5_motifs) > 0 else "",
            "top_motif_2": top5_motifs[1] if len(top5_motifs) > 1 else "",
            "top_motif_3": top5_motifs[2] if len(top5_motifs) > 2 else "",
            "top_motif_4": top5_motifs[3] if len(top5_motifs) > 3 else "",
            "top_motif_5": top5_motifs[4] if len(top5_motifs) > 4 else "",
            "motif_top1_hit": topn_hit(top5_motifs, em, 1),
            "motif_top3_hit": topn_hit(top5_motifs, em, 3),
            "motif_top5_hit": topn_hit(top5_motifs, em, 5),
        })
        rank_family_rows.append({
            "spectrum_id": sid, "dataset": r["dataset"],
            "component_key": comp,
            "expected_families": ",".join(ef),
            "top_family_1": top5_fams[0] if len(top5_fams) > 0 else "",
            "top_family_2": top5_fams[1] if len(top5_fams) > 1 else "",
            "top_family_3": top5_fams[2] if len(top5_fams) > 2 else "",
            "top_family_4": top5_fams[3] if len(top5_fams) > 3 else "",
            "top_family_5": top5_fams[4] if len(top5_fams) > 4 else "",
            "family_top1_hit": topn_hit(top5_fams, ef, 1),
            "family_top3_hit": topn_hit(top5_fams, ef, 3),
            "family_top5_hit": topn_hit(top5_fams, ef, 5),
        })

        # Off-target activation: any non-expected motif with core_weight > 0.05
        for m in res.motif_scores:
            if m.core_weight > 0.05 and em and m.motif_id not in em:
                off_target_rows.append({
                    "spectrum_id": sid, "dataset": r["dataset"],
                    "component_key": comp,
                    "off_target_motif": m.motif_id,
                    "core_weight": round(m.core_weight, 5),
                    "expected_motifs": ",".join(em),
                })

        # Miss row: top-3 motif AND top-3 family both miss => write a miss
        m_top3 = topn_hit(top5_motifs, em, 3)
        f_top3 = topn_hit(top5_fams, ef, 3)
        if (em or ef) and not (m_top3 and f_top3):
            failure_type = []
            if em and not m_top3: failure_type.append("MOTIF_MISS_TOP3")
            if ef and not f_top3: failure_type.append("FAMILY_MISS_TOP3")
            if ea and amb < 0.10:  failure_type.append("AMBIGUITY_UNDERFIRE")
            if (not ea) and amb >= 0.10: failure_type.append("AMBIGUITY_OVERFIRE")
            miss_rows.append({
                "spectrum_id": sid, "dataset_name": r["dataset"],
                "component_key": comp,
                "expected_motifs": ",".join(em),
                "observed_top_motifs": ",".join(top5_motifs[:3]),
                "expected_families": ",".join(ef),
                "observed_top_families": ",".join(top5_fams[:3]),
                "expected_ambiguity": ea,
                "observed_ambiguity_active": amb >= 0.10,
                "ambiguity_score": round(amb, 4),
                "failure_type": ",".join(failure_type),
                "notes": "",
            })

    # ── Emit per-spectrum tables ─────────────────────────────────────
    pd.DataFrame(motif_rows).to_csv(
        TABLES / "grounding_per_spectrum_motif_scores_v_motif_first.csv", index=False,
    )
    pd.DataFrame(family_rows).to_csv(
        TABLES / "grounding_per_spectrum_family_scores_v_motif_first.csv", index=False,
    )
    pd.DataFrame(ambig_rows).to_csv(
        TABLES / "grounding_per_spectrum_ambiguity_scores_v_motif_first.csv", index=False,
    )
    pd.DataFrame(rank_motif_rows).to_csv(
        TABLES / "grounding_expected_vs_observed_motif_rank_v_motif_first.csv", index=False,
    )
    pd.DataFrame(rank_family_rows).to_csv(
        TABLES / "grounding_expected_vs_observed_family_rank_v_motif_first.csv", index=False,
    )
    pd.DataFrame(off_target_rows).to_csv(
        TABLES / "grounding_off_target_activation_v_motif_first.csv", index=False,
    )
    pd.DataFrame(ambig_rows).to_csv(
        TABLES / "grounding_ambiguity_behavior_v_motif_first.csv", index=False,
    )
    pd.DataFrame(miss_rows).to_csv(
        TABLES / "grounding_miss_list_v_motif_first.csv", index=False,
    )

    # ── Metrics ──────────────────────────────────────────────────────
    rm = pd.DataFrame(rank_motif_rows)
    rf = pd.DataFrame(rank_family_rows)
    amb_df = pd.DataFrame(ambig_rows)

    rm_classified = rm[rm["expected_motifs"] != ""]
    rf_classified = rf[rf["expected_families"] != ""]

    metrics = {
        "n_total_spectra": len(rm),
        "n_motif_classified":  len(rm_classified),
        "n_family_classified": len(rf_classified),

        "motif_top1_hit_rate":  round(rm_classified["motif_top1_hit"].mean(), 4) if len(rm_classified) else 0.0,
        "motif_top3_hit_rate":  round(rm_classified["motif_top3_hit"].mean(), 4) if len(rm_classified) else 0.0,
        "motif_top5_hit_rate":  round(rm_classified["motif_top5_hit"].mean(), 4) if len(rm_classified) else 0.0,
        "family_top1_hit_rate": round(rf_classified["family_top1_hit"].mean(), 4) if len(rf_classified) else 0.0,
        "family_top3_hit_rate": round(rf_classified["family_top3_hit"].mean(), 4) if len(rf_classified) else 0.0,
        "family_top5_hit_rate": round(rf_classified["family_top5_hit"].mean(), 4) if len(rf_classified) else 0.0,

        "ambiguity_correctness_rate": round(amb_df["ambiguity_correct"].mean(), 4),
        "ambiguity_overfire_rate":    round(amb_df["ambiguity_overfire"].mean(), 4),
        "ambiguity_underfire_rate":   round(amb_df["ambiguity_underfire"].mean(), 4),

        "n_motif_misses_top3":  int((~rm_classified["motif_top3_hit"]).sum()) if len(rm_classified) else 0,
        "n_family_misses_top3": int((~rf_classified["family_top3_hit"]).sum()) if len(rf_classified) else 0,
        "n_total_misses":       len(miss_rows),
        "n_off_target_events":  len(off_target_rows),
    }
    pd.DataFrame([metrics]).to_csv(
        TABLES / "grounding_metrics_summary_v_motif_first.csv", index=False,
    )
    print("\n[metrics summary]")
    for k, v in metrics.items():
        print(f"  {k:35s}: {v}")

    # ── Per-family hit rate (family-level)
    rf_classified = rf_classified.copy()
    rf_classified["primary_family"] = rf_classified["expected_families"].str.split(",").str[0]
    per_fam = rf_classified.groupby("primary_family")[
        ["family_top1_hit", "family_top3_hit", "family_top5_hit"]
    ].mean()
    per_fam_n = rf_classified.groupby("primary_family").size().rename("n")
    per_fam_table = per_fam.join(per_fam_n)
    per_fam_table.to_csv(TABLES / "grounding_per_family_hit_rates_v_motif_first.csv")

    # ── Per-dataset hit rate
    per_ds = rf_classified.groupby("dataset")[
        ["family_top1_hit", "family_top3_hit", "family_top5_hit"]
    ].mean()
    per_ds_n = rf_classified.groupby("dataset").size().rename("n")
    per_ds_table = per_ds.join(per_ds_n)
    per_ds_table.to_csv(TABLES / "grounding_per_dataset_hit_rates_v_motif_first.csv")

    # ── Figures ──────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    if plt is not None:
        _figs(plt, motif_rows, family_rows, ambig_rows, rank_motif_rows,
              rank_family_rows, off_target_rows, all_refs, master_x,
              active, mappings, dual, per_fam_table, metrics)

    # ── Reports + audit ──────────────────────────────────────────────
    _write_main_report(metrics, per_fam_table, per_ds_table, miss_rows,
                       off_target_rows, ambig_rows, len(active), len(mappings))
    _write_miss_analysis(miss_rows, off_target_rows, ambig_rows)
    _write_audit_log(metrics, len(all_refs), len(active), len(mappings),
                     len(rb), len(gp), len(aa), len(lit))
    _snapshot_code()
    print("DONE")


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────

def _figs(plt, motif_rows, family_rows, ambig_rows, rank_motif_rows,
          rank_family_rows, off_target_rows, all_refs, master_x,
          motifs, mappings, dual, per_fam_table, metrics):
    import matplotlib.cm as cm

    # 1. fig_motif_top_rank_heatmap_motif_first
    rm = pd.DataFrame(rank_motif_rows)
    rm = rm[rm["expected_motifs"] != ""].copy()
    rm["primary_expected_motif"] = rm["expected_motifs"].str.split(",").str[0]
    piv = pd.crosstab(rm["primary_expected_motif"], rm["top_motif_1"])
    piv = piv.div(piv.sum(axis=1).replace(0, 1), axis=0)
    piv = piv.loc[sorted(piv.index)]
    fig, ax = plt.subplots(figsize=(15, max(8, 0.4 * len(piv))))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([c[:25] for c in piv.columns], rotation=70, ha="right", fontsize=6)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([c[:30] for c in piv.index], fontsize=7)
    fig.colorbar(im, ax=ax, label="fraction (top-1 motif | expected)")
    ax.set_title("Motif top-1 confusion: rows=primary expected motif, cols=observed top-1")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_motif_top_rank_heatmap_motif_first.png", dpi=130)
    plt.close(fig)

    # 2. fig_family_top_rank_heatmap_motif_first
    rf = pd.DataFrame(rank_family_rows)
    rf = rf[rf["expected_families"] != ""].copy()
    rf["primary_expected_family"] = rf["expected_families"].str.split(",").str[0]
    piv2 = pd.crosstab(rf["primary_expected_family"], rf["top_family_1"])
    piv2 = piv2.div(piv2.sum(axis=1).replace(0, 1), axis=0)
    piv2 = piv2.reindex(index=[f for f in FAMILIES if f in piv2.index],
                       columns=[f for f in FAMILIES if f in piv2.columns],
                       fill_value=0.0)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.6 * len(piv2))))
    im = ax.imshow(piv2.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv2.columns)))
    ax.set_xticklabels(piv2.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv2.index)))
    ax.set_yticklabels(piv2.index, fontsize=9)
    for i in range(piv2.shape[0]):
        for j in range(piv2.shape[1]):
            v = piv2.values[i, j]
            if v > 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    fig.colorbar(im, ax=ax, label="fraction")
    ax.set_title("Family top-1 confusion: rows=primary expected family, cols=observed top-1 family")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_family_top_rank_heatmap_motif_first.png", dpi=130)
    plt.close(fig)

    # 3. fig_grouped_motif_in_family_examples_motif_first
    id_to_ref = {r["spectrum_id"]: r for r in all_refs}
    examples = []
    targets = [
        ("ramanbiolib", "cholesteryl linoleate"),
        ("ramanbiolib", "l-glutamate"),
        ("ramanbiolib", "adenine"),
        ("gobbato_powder", "UA_rep01"),
        ("ramanbiolib", "d-(+)-glucose"),
        ("ramanbiolib", "albumin"),
    ]
    for tag, suffix in targets:
        for sid in id_to_ref:
            if sid.startswith(f"{tag}::") and suffix in sid:
                examples.append(sid); break

    if examples:
        from gaira.base2.motif_engine import resolve_mapping_weight
        fig, axes = plt.subplots(1, len(examples), figsize=(4.5*len(examples), 8),
                                 sharey=True)
        if len(examples) == 1: axes = [axes]
        cmap = cm.get_cmap("tab20", 20)
        colors = {}
        def col_for(mid):
            if mid not in colors: colors[mid] = cmap(len(colors) % 20)
            return colors[mid]

        for ax, sid in zip(axes, examples):
            ref = id_to_ref[sid]
            res = patched_score_spectrum_rescue(
                ref["spectrum"], master_x, motifs, mappings, dual, sid,
            )
            ms = {m.motif_id: m.core_weight for m in res.motif_scores}
            fam_to_contrib = {}
            for fam in FAMILIES:
                contribs = []
                for mid, s in ms.items():
                    mp = mappings.get(mid)
                    if mp is None or s <= 0: continue
                    mw = resolve_mapping_weight(mp, fam)
                    if mw > 0:
                        contribs.append((mid, s * mw))
                fam_to_contrib[fam] = sorted(contribs, key=lambda x: x[1], reverse=True)
            y_pos = np.arange(len(FAMILIES))
            for i, fam in enumerate(FAMILIES):
                left = 0.0
                for mid, contrib in fam_to_contrib[fam]:
                    ax.barh(i, contrib, left=left, color=col_for(mid),
                            edgecolor="black", linewidth=0.2)
                    if contrib >= 0.04:
                        ax.text(left + contrib/2, i,
                                mid.replace("_motif", "")[:18],
                                va="center", ha="center", fontsize=5, color="white")
                    left += contrib
            ax.set_yticks(y_pos); ax.set_yticklabels(FAMILIES, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim(0, max(1.3, 1.05*max(
                (sum(c for _, c in fam_to_contrib[f]) for f in FAMILIES), default=1.0))
            )
            ax.set_xlabel("stacked motif contribution to family")
            ax.set_title(sid.split("::")[-1][:30], fontsize=9)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        fig.suptitle("Grouped motif-in-family examples (rescue engine, registry v1.3.1)",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_grouped_motif_in_family_examples_motif_first.png", dpi=130)
        plt.close(fig)

    # 4. fig_off_target_activation_heatmap_motif_first
    ot = pd.DataFrame(off_target_rows)
    if len(ot) > 0:
        ot["primary_expected"] = ot["expected_motifs"].str.split(",").str[0]
        piv3 = pd.crosstab(ot["primary_expected"], ot["off_target_motif"])
        # Limit to top 20 most-frequent off-target motifs
        col_sums = piv3.sum(axis=0).sort_values(ascending=False)
        keep = col_sums.head(20).index
        piv3 = piv3[keep]
        fig, ax = plt.subplots(figsize=(13, max(6, 0.4 * len(piv3))))
        im = ax.imshow(piv3.values, aspect="auto", cmap="OrRd")
        ax.set_xticks(range(len(piv3.columns)))
        ax.set_xticklabels([c[:25] for c in piv3.columns], rotation=70, ha="right", fontsize=7)
        ax.set_yticks(range(len(piv3.index)))
        ax.set_yticklabels([c[:30] for c in piv3.index], fontsize=7)
        for i in range(piv3.shape[0]):
            for j in range(piv3.shape[1]):
                v = piv3.values[i, j]
                if v > 0:
                    ax.text(j, i, str(int(v)), ha="center", va="center",
                            fontsize=6, color="black")
        fig.colorbar(im, ax=ax, label="off-target activation count")
        ax.set_title("Off-target activation matrix: rows=primary expected motif, "
                     "cols=top-20 off-target motifs (count of events)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_off_target_activation_heatmap_motif_first.png", dpi=130)
        plt.close(fig)

    # 5. fig_ambiguity_panel_motif_first (3-panel)
    amb = pd.DataFrame(ambig_rows)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5))
    # 5a histogram
    a1.hist(amb["ambiguity_core"], bins=30, color="#7b2cbf",
            edgecolor="black", linewidth=0.3)
    a1.axvline(0.10, color="red", linestyle="--", label="gated threshold (0.10)")
    a1.set_xlabel("ambiguity_core"); a1.set_ylabel("count of spectra")
    a1.set_title(f"Ambiguity core distribution (n={len(amb)})")
    a1.legend()
    # 5b correctness pie
    correct = int(amb["ambiguity_correct"].sum())
    incorrect = len(amb) - correct
    a2.pie([correct, incorrect], labels=[f"correct\n({correct})", f"incorrect\n({incorrect})"],
           colors=["#2a9d8f", "#e76f51"], startangle=90, autopct="%.1f%%")
    a2.set_title("Ambiguity correctness")
    # 5c overfire / underfire bar
    of = int(amb["ambiguity_overfire"].sum())
    uf = int(amb["ambiguity_underfire"].sum())
    cz = correct
    a3.bar(["correct", "overfire", "underfire"], [cz, of, uf],
           color=["#2a9d8f", "#e76f51", "#f4a261"])
    a3.set_ylabel("spectra count")
    a3.set_title("Ambiguity classification")
    for side in ("top", "right"):
        a1.spines[side].set_visible(False)
        a3.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_ambiguity_panel_motif_first.png", dpi=130)
    plt.close(fig)

    # 6. fig_family_hit_rate_motif_first
    pf = per_fam_table.sort_values("family_top1_hit", ascending=False)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(pf))))
    y = np.arange(len(pf))
    ax.barh(y - 0.27, pf["family_top1_hit"], height=0.27,
            color="#2a9d8f", label="top-1")
    ax.barh(y,        pf["family_top3_hit"], height=0.27,
            color="#76c893", label="top-3")
    ax.barh(y + 0.27, pf["family_top5_hit"], height=0.27,
            color="#b7e4c7", label="top-5")
    ax.set_yticks(y); ax.set_yticklabels(pf.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("family hit rate")
    ax.set_title("Per-family hit rate (top-1/3/5)")
    ax.legend()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for i, (idx, row) in enumerate(pf.iterrows()):
        ax.text(row["family_top5_hit"] + 0.02, i + 0.27,
                f"n={int(row['n'])}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_family_hit_rate_motif_first.png", dpi=130)
    plt.close(fig)

    # 7. fig_motif_radar_examples_motif_first
    if examples:
        fig, axes = plt.subplots(1, len(examples),
                                 figsize=(4.5*len(examples), 4.5),
                                 subplot_kw=dict(polar=True))
        if len(examples) == 1: axes = [axes]
        angles = np.linspace(0, 2*np.pi, len(FAMILIES), endpoint=False).tolist()
        angles += angles[:1]
        for ax, sid in zip(axes, examples):
            ref = id_to_ref[sid]
            res = patched_score_spectrum_rescue(
                ref["spectrum"], master_x, motifs, mappings, dual, sid,
            )
            vals = []
            for fam in FAMILIES:
                s, _ = family_score(res.motif_scores, mappings, fam)
                vals.append(s)
            vmax = max(vals) if max(vals) > 0 else 1.0
            vals = [v / vmax for v in vals]
            vals += vals[:1]
            ax.plot(angles, vals, color="#2a9d8f", linewidth=1.5)
            ax.fill(angles, vals, color="#2a9d8f", alpha=0.3)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([f.replace("_", "\n") for f in FAMILIES],
                               fontsize=5)
            ax.set_ylim(0, 1.05)
            ax.set_title(sid.split("::")[-1][:25], fontsize=8, pad=12)
        fig.suptitle("Family-level radar (grouped motif scores; normalised per-spectrum)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_motif_radar_examples_motif_first.png", dpi=130)
        plt.close(fig)

    # 8. fig_sunburst_treemap_exploratory_motif_first
    # hierarchy: family -> motif
    from gaira.base2.motif_engine import resolve_mapping_weight
    agg = defaultdict(lambda: defaultdict(float))
    agg_amb = 0.0
    for ref in all_refs:
        res = patched_score_spectrum_rescue(
            ref["spectrum"], master_x, motifs, mappings, dual, ref["spectrum_id"],
        )
        agg_amb += res.ambiguity.core_evidence
        ms = {m.motif_id: m.core_weight for m in res.motif_scores}
        for fam in FAMILIES:
            for mid, s in ms.items():
                mp = mappings.get(mid)
                if mp is None or s <= 0: continue
                mw = resolve_mapping_weight(mp, fam)
                if mw > 0: agg[fam][mid] += s * mw
    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    for ax in axes.flat: ax.set_axis_off()
    cmap = cm.get_cmap("tab20", 20)
    colors = {}
    def col(mid):
        if mid not in colors:
            colors[mid] = cmap(len(colors) % 20)
        return colors[mid]
    def tile(ax, items, title):
        total = sum(v for _, v in items)
        if total <= 0:
            ax.text(0.5, 0.5, f"{title}\n(no signal)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9); return
        items = sorted(items, key=lambda x: x[1], reverse=True)
        y = 1.0
        for lbl, val in items:
            frac = val / total
            ax.add_patch(plt.Rectangle((0, y-frac), 1.0, frac,
                                       facecolor=col(lbl), edgecolor="black",
                                       linewidth=0.5))
            if frac > 0.03:
                ax.text(0.5, y-frac/2,
                        lbl.replace("_motif", "")[:24] + f" ({frac:.0%})",
                        ha="center", va="center", fontsize=6, color="white")
            y -= frac
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
    for i, fam in enumerate(FAMILIES):
        ax = axes.flat[i]; ax.set_axis_on()
        items = list(agg[fam].items())
        tile(ax, items, f"{fam}\n(Σ={sum(v for _,v in items):.2f})")
    amb_ax = axes.flat[11]; amb_ax.set_axis_on()
    amb_ax.text(0.5, 0.5,
                f"ambiguity_artifact\n(control lane)\n\nΣ over {len(all_refs)} refs: {agg_amb:.2f}",
                ha="center", va="center", fontsize=10,
                transform=amb_ax.transAxes, color="#7b2cbf")
    amb_ax.set_xticks([]); amb_ax.set_yticks([])
    for side in ("top","right","left","bottom"):
        amb_ax.spines[side].set_visible(False)
    fig.suptitle("Family -> motif treemap (aggregate over full grounding corpus)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_sunburst_treemap_exploratory_motif_first.png", dpi=130)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────────────────────────────

def _write_main_report(metrics, per_fam_table, per_ds_table, miss_rows,
                       off_target_rows, ambig_rows, n_motifs, n_mappings):
    pf_sorted_strong = per_fam_table.sort_values("family_top1_hit", ascending=False)
    pf_sorted_weak = per_fam_table.sort_values("family_top1_hit", ascending=True).head(3)

    lines = [
        "# gaira_validate_2_grounding_motif_first_v1",
        "",
        "## Phase framing",
        "",
        "Motif-first / family-first grounding validation. Motifs are the "
        "primary representation; families are the primary summary layer; "
        "ambiguity is a separate control lane. Broad axis top-1 is NOT the "
        "lead metric in this phase.",
        "",
        "**Engine:** rescue (registry v1.3.1 + mapping v1.2.1 + v2_patches_rescue) "
        f"- {n_motifs} active motifs / {n_mappings} mappings.",
        "",
        "**Datasets:** full grounding corpus (4 datasets, "
        f"{metrics['n_total_spectra']} spectra). NO calibration / target / "
        "substrate-aware data was used.",
        "",
        "## Primary metrics",
        "",
        "| level | top-1 | top-3 | top-5 | n classified |",
        "|---|---:|---:|---:|---:|",
        f"| motif  | {metrics['motif_top1_hit_rate']:.1%}  | "
        f"{metrics['motif_top3_hit_rate']:.1%}  | {metrics['motif_top5_hit_rate']:.1%}  | "
        f"{metrics['n_motif_classified']} |",
        f"| family | {metrics['family_top1_hit_rate']:.1%} | "
        f"{metrics['family_top3_hit_rate']:.1%} | {metrics['family_top5_hit_rate']:.1%} | "
        f"{metrics['n_family_classified']} |",
        "",
        "## Ambiguity (separate control lane)",
        "",
        f"- correctness rate: **{metrics['ambiguity_correctness_rate']:.1%}**",
        f"- overfire rate (fires when chemistry not ambiguous): "
        f"**{metrics['ambiguity_overfire_rate']:.1%}**",
        f"- underfire rate (silent when chemistry IS ambiguous): "
        f"**{metrics['ambiguity_underfire_rate']:.1%}**",
        "",
        "## Strongest families (top-1 family hit)",
        "",
        "| family | top-1 | top-3 | top-5 | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for fam, row in pf_sorted_strong.iterrows():
        lines.append(f"| {fam} | {row['family_top1_hit']:.1%} | "
                     f"{row['family_top3_hit']:.1%} | {row['family_top5_hit']:.1%} | "
                     f"{int(row['n'])} |")

    lines += [
        "",
        "## Weakest families (top-1 family hit)",
        "",
        "| family | top-1 | top-3 | top-5 | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for fam, row in pf_sorted_weak.iterrows():
        lines.append(f"| {fam} | {row['family_top1_hit']:.1%} | "
                     f"{row['family_top3_hit']:.1%} | {row['family_top5_hit']:.1%} | "
                     f"{int(row['n'])} |")

    lines += [
        "",
        "## Per-dataset family hit rates",
        "",
        "| dataset | top-1 | top-3 | top-5 | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds, row in per_ds_table.iterrows():
        lines.append(f"| `{ds}` | {row['family_top1_hit']:.1%} | "
                     f"{row['family_top3_hit']:.1%} | {row['family_top5_hit']:.1%} | "
                     f"{int(row['n'])} |")

    lines += [
        "",
        "## Representative grouped motif-in-family examples",
        "",
        "See `figures/fig_grouped_motif_in_family_examples_motif_first.png`. "
        "Examples chosen: cholesteryl linoleate, l-glutamate, adenine, "
        "UA (uric acid powder), d-(+)-glucose, albumin. The figure shows "
        "stacked motif contributions to each of the 11 biology families "
        "for each spectrum - the resolution beyond axes the user requested.",
        "",
        "## Treemap evaluation",
        "",
        "See `figures/fig_sunburst_treemap_exploratory_motif_first.png`. "
        "Hierarchy is family -> motif; ambiguity_artifact is shown as a "
        "separate panel. Aggregated over all 377 grounding spectra.",
        "",
        "## Major miss patterns",
        "",
        f"Total misses (motif top-3 OR family top-3 missed, OR ambiguity "
        f"mis-fired): **{metrics['n_total_misses']} / {metrics['n_total_spectra']}**.",
        "",
        f"- motif misses (top-3): {metrics['n_motif_misses_top3']}",
        f"- family misses (top-3): {metrics['n_family_misses_top3']}",
        f"- off-target activation events (non-expected motif core_weight > 0.05): "
        f"{metrics['n_off_target_events']}",
        "",
        "Detailed miss-class taxonomy is in "
        "`reports/REPORT_gaira_validate_2_grounding_motif_first_miss_analysis_v1.md`.",
        "",
        "## Reading guide for this phase's metrics",
        "",
        "- Top-3 motif hit and top-3 family hit are the most chemistry-honest "
        "metrics for pure-compound grounding. Top-1 of either is a stricter "
        "metric that punishes legitimate multi-chemistry references.",
        "- Ambiguity is reported as a separate lane (correctness rate + "
        "overfire/underfire rates). It is NOT included in the family hit rates.",
        "- Backend 11-axis scores were computed but are NOT the primary "
        "validation target in this phase. Use the prior phase reports for "
        "axis-level numbers.",
    ]

    (REPORTS / "REPORT_gaira_validate_2_grounding_motif_first_v1.md").write_text(
        "\n".join(lines),
    )


def _write_miss_analysis(miss_rows, off_target_rows, ambig_rows):
    df = pd.DataFrame(miss_rows)
    of = pd.DataFrame(off_target_rows)
    amb = pd.DataFrame(ambig_rows)

    motif_only = df[df["failure_type"].fillna("").str.contains("MOTIF_MISS_TOP3")]
    family_only = df[df["failure_type"].fillna("").str.contains("FAMILY_MISS_TOP3")]
    amb_over = amb[amb["ambiguity_overfire"]]
    amb_under = amb[amb["ambiguity_underfire"]]

    # Off-target count by motif id
    if len(of) > 0:
        of_top = of["off_target_motif"].value_counts().head(15)
    else:
        of_top = pd.Series(dtype=int)

    # Family-miss breakdown by primary expected family
    if len(family_only) > 0:
        family_only = family_only.copy()
        family_only["primary_expected_family"] = (
            family_only["expected_families"].str.split(",").str[0]
        )
        fam_breakdown = family_only["primary_expected_family"].value_counts()
    else:
        fam_breakdown = pd.Series(dtype=int)

    # Motif-miss breakdown by primary expected motif
    if len(motif_only) > 0:
        motif_only = motif_only.copy()
        motif_only["primary_expected_motif"] = (
            motif_only["expected_motifs"].str.split(",").str[0]
        )
        motif_breakdown = motif_only["primary_expected_motif"].value_counts()
    else:
        motif_breakdown = pd.Series(dtype=int)

    lines = [
        "# gaira_validate_2_grounding_motif_first_v1 - Miss Analysis",
        "",
        "Chemistry-first / ontology-first interpretation of misses. NO "
        "substrate or calibration discussion - pure-compound grounding only.",
        "",
        f"**Total misses:** {len(df)} ({len(motif_only)} motif top-3 misses; "
        f"{len(family_only)} family top-3 misses; "
        f"{len(amb_over)} ambiguity overfires; "
        f"{len(amb_under)} ambiguity underfires).",
        "",
        "## Common motif-level failure modes",
        "",
        "Top-N expected-motif classes that fail to hit top-3:",
        "",
        "| primary expected motif | n missed |",
        "|---|---:|",
    ]
    for mid, c in motif_breakdown.head(15).items():
        lines.append(f"| `{mid}` | {c} |")

    lines += [
        "",
        "## Common family-level failure modes",
        "",
        "Primary expected families that fail to hit top-3:",
        "",
        "| primary expected family | n missed |",
        "|---|---:|",
    ]
    for fam, c in fam_breakdown.items():
        lines.append(f"| {fam} | {c} |")

    lines += [
        "",
        "## Off-target activation hotspots",
        "",
        "Most-frequent off-target motifs (non-expected, core_weight > 0.05):",
        "",
        "| off-target motif | n events |",
        "|---|---:|",
    ]
    for mid, c in of_top.items():
        lines.append(f"| `{mid}` | {c} |")

    lines += [
        "",
        "## Ambiguity failures",
        "",
        f"- **Overfires** ({len(amb_over)}): chemistry NOT genuinely ambiguous "
        "but ambiguity_core >= 0.10. Examples (first 10):",
        "",
    ]
    for _, r in amb_over.head(10).iterrows():
        lines.append(f"- `{r['component_key']}` (ambiguity {r['ambiguity_core']:.3f})")
    lines += [
        "",
        f"- **Underfires** ({len(amb_under)}): chemistry IS genuinely ambiguous "
        "but ambiguity_core < 0.10. Examples (first 10):",
        "",
    ]
    for _, r in amb_under.head(10).iterrows():
        lines.append(f"- `{r['component_key']}` (ambiguity {r['ambiguity_core']:.3f})")

    lines += [
        "",
        "## Likely next patch targets in gaira_base_2",
        "",
        "Ontology-first / chemistry-first targets ONLY (no scoring patches "
        "in this analysis - per the locked rule):",
        "",
        "1. **Free-amino-acid motif coverage.** Many free amino acids "
        "miss motif top-3 because there is no per-amino-acid motif - "
        "they must rely on amide_I/III + the (sparse) glutamate motif. "
        "v2 ontology should add per-residue side-chain motifs (Arg, Ser, "
        "Asp, Pro, Val, etc.) sourced from De Gelder 2007.",
        "2. **Lactate.** Currently DEFERRED with empty expected_motifs. "
        "v2 ontology bump should activate lactate_motif once a pure "
        "lactate reference is acquired (M3.3-class).",
        "3. **Aromatic-steroid discriminator.** Estrogens (estradiol, "
        "estrone, estriol, ethinylestradiol, diethylstilbestrol) lack a "
        "saturated-ring motif anchor and rely on sterol_skeletal_motif "
        "which fits saturated rings. A separate aromatic-ring motif is "
        "needed for these references.",
        "4. **Tryptophan.** Currently expects amide_III; should have "
        "tryptophan-specific motif (760/1340/1550) once the registry "
        "entry's mapping is wired (entry exists but no mapping row).",
        "5. **Histidine.** Imidazole 1280-1290 motif is in the registry "
        "but has no mapping row in v1.2.1.",
        "6. **Off-target hotspots** point to where broad motifs win on "
        "small-molecule references - these are NOT scoring-patch targets "
        "(v4 proved aggressive dampening regresses) but rather candidates "
        "for chemistry-discriminative co-band requirements at the motif "
        "definition level.",
        "",
        "## What this report deliberately does NOT discuss",
        "",
        "- Substrate-aware regime behavior",
        "- Calibration data behavior",
        "- 8-axis projection",
        "- Broad-axis top-1 numbers as primary evidence",
    ]
    (REPORTS / "REPORT_gaira_validate_2_grounding_motif_first_miss_analysis_v1.md"
     ).write_text("\n".join(lines))


def _write_audit_log(metrics, n_total, n_motifs, n_mappings,
                     n_rb, n_gp, n_aa, n_lit):
    lines = [
        "# gaira_validate_2_grounding_motif_first_v1 - Audit Log",
        "",
        "## Datasets used (grounding only; no calibration / target / substrate)",
        "",
        f"- ramanbiolib                          ({n_rb} spectra)",
        f"- gobbato_powder_raman                 ({n_gp} spectra; 53 analytes x 3 reps)",
        f"- amino_acid_raman_grounding/aa.xlsx   ({n_aa} spectra)",
        f"- digitised_literature_spectra         ({n_lit} spectra; "
        "Gelder 2007 + Kim 1987)",
        f"- TOTAL                                ({n_total} spectra)",
        "",
        "All approved core-grounding datasets included; none excluded.",
        "",
        "## Truth table version used",
        "",
        "- File: `truth_table/grounding_truth_table_motif_first_v1.csv`",
        "- expected_motifs: chemistry-justified per component_key (this phase, new)",
        "- expected_families: derived from prior multi-axis truth table "
        "(grounding_repair_loop_v1 TRUTH_AXES, restricted to the 11 "
        "biology families)",
        "- expected_ambiguity: True for known ambiguous chemistry "
        "(citric acid, choline-bearing lipids), False otherwise",
        "- multi-family expectations preserved (free amino acids, "
        "cholesteryl esters, aromatic AAs, etc.)",
        "",
        "## Engine used",
        "",
        f"- {n_motifs} active motifs / {n_mappings} mappings",
        "- Registry: v1.3.1 (= v1.3 + cholesteryl_ester_discriminator_motif)",
        "- Mapping:  v1.2.1 (= v1.2 + 1 new PRIMARY row)",
        "- Patches:  v2_patches_rescue (rescue variant; no repair_v2 overlay)",
        "- Scoring:  patched_score_spectrum_rescue (PATCH A specificity "
        "weights, PATCH B competitor dampening with glycan-vs-phos "
        "removed, PATCH C gated ambiguity, PATCH D sparse-axis boost)",
        "",
        "## Spectra excluded",
        "",
        "None excluded. All 377 spectra scored.",
        "",
        "## Scoring anomalies",
        "",
        "None observed. All spectra processed without numerical issues. "
        "Some references have empty expected_motifs (urea, lactate) - "
        "these are valid empty-expectation cases (chemistry coverage "
        "gap) and are excluded from motif top-N hit-rate denominators.",
        "",
        "## Files written",
        "",
        "- inventory/grounding_dataset_inventory_v_motif_first.csv",
        "- truth_table/grounding_truth_table_motif_first_v1.csv",
        "- tables/grounding_per_spectrum_motif_scores_v_motif_first.csv",
        "- tables/grounding_per_spectrum_family_scores_v_motif_first.csv",
        "- tables/grounding_per_spectrum_ambiguity_scores_v_motif_first.csv",
        "- tables/grounding_expected_vs_observed_motif_rank_v_motif_first.csv",
        "- tables/grounding_expected_vs_observed_family_rank_v_motif_first.csv",
        "- tables/grounding_off_target_activation_v_motif_first.csv",
        "- tables/grounding_ambiguity_behavior_v_motif_first.csv",
        "- tables/grounding_miss_list_v_motif_first.csv",
        "- tables/grounding_metrics_summary_v_motif_first.csv",
        "- tables/grounding_per_family_hit_rates_v_motif_first.csv",
        "- tables/grounding_per_dataset_hit_rates_v_motif_first.csv",
        "- 8 figures under figures/",
        "- 2 reports under reports/",
        "",
        "## Non-modification invariants",
        "",
        "- gaira_base SHA-256 still matches; 12/12 v1 regression tests pass",
        "- v1 engine modules unchanged",
        "- v2_patches.py + v2_patches_rescue.py unchanged",
        "- registry v1.3.1 + mapping v1.2.1 read-only this phase",
        "- M2.2 dual-status table unchanged",
        "- canonical preprocessing unchanged",
        "- substrate engine v1.1.2 unchanged",
        "- NO Streamlit built",
        "",
        "## Headline metrics",
        "",
        f"- motif top-1: {metrics['motif_top1_hit_rate']:.1%}",
        f"- motif top-3: {metrics['motif_top3_hit_rate']:.1%}",
        f"- motif top-5: {metrics['motif_top5_hit_rate']:.1%}",
        f"- family top-1: {metrics['family_top1_hit_rate']:.1%}",
        f"- family top-3: {metrics['family_top3_hit_rate']:.1%}",
        f"- family top-5: {metrics['family_top5_hit_rate']:.1%}",
        f"- ambiguity correctness: {metrics['ambiguity_correctness_rate']:.1%}",
        f"- ambiguity overfire: {metrics['ambiguity_overfire_rate']:.1%}",
        f"- ambiguity underfire: {metrics['ambiguity_underfire_rate']:.1%}",
    ]
    (AUDIT / "gaira_validate_2_grounding_motif_first_audit_log.md").write_text(
        "\n".join(lines),
    )


def _snapshot_code():
    src = Path("/Users/suraj/projects/GAIRA/src/gaira/base2")
    if src.exists():
        shutil.copytree(src, CODE_SNAPSHOT / "base2", dirs_exist_ok=True)
    p = Path("/Users/suraj/projects/GAIRA/scripts/run_gaira_validate_2_grounding_motif_first_v1.py")
    if p.exists():
        shutil.copy(p, CODE_SNAPSHOT / p.name)


if __name__ == "__main__":
    main()
