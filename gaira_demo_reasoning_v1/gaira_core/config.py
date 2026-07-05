"""GAIRA Demo v1 — config (axes, labels, colors, palette, paths).

The 11-axis biochemical state vector (BSV) is the canonical interpretation
layer. UI labels and colors live here so every plot uses the same order
and palette.
"""
from __future__ import annotations

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────

DEMO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = DEMO_ROOT / "data"
ASSETS_ROOT = DEMO_ROOT / "assets"

CALIBRATION_DIR = DATA_ROOT / "calibration"
PILOTS_DIR = DATA_ROOT / "pilots"
GROUNDING_DIR = DATA_ROOT / "grounding"
CACHED_DIR = DATA_ROOT / "cached"

# Optional real-data roots (used by data_loader when present)
GAIRA_REPO_ROOT = DEMO_ROOT.parent
LEGACY_DEMO_DATA = GAIRA_REPO_ROOT / "streamlit_apps" / "gaira_demo" / "data"

# External processed-data volume (GAIRA autoresearch outputs)
GAIRA_DATA_VOLUME = Path("/Volumes/SSD_Rad/GAIRA_DATA")
AUTORESEARCH_ROOT = (GAIRA_DATA_VOLUME
                       / "processed" / "gaira_autoresearch" / "gaira_autoresearch_v1")

SHINE_DAY02_TABLES   = AUTORESEARCH_ROOT / "pilot3_shine_single_set_day0_day2" / "tables"
SHINE_DAY2_TABLES    = AUTORESEARCH_ROOT / "pilot3_shine_day2_controlanchored" / "tables"
SHINE_EV_SERS_TABLES = AUTORESEARCH_ROOT / "pilot3_shine_ev_sers" / "tables"

EV_DIABETES_TABLES = AUTORESEARCH_ROOT / "pilot2_target_validation_v1" / "tables"
LIVER_PATIENT_TABLES = AUTORESEARCH_ROOT / "pilot4_1_cca_hcc_lm_serum_patient_level" / "tables"
LIVER_COHORT_TABLES  = AUTORESEARCH_ROOT / "pilot4_cca_hcc_lm_serum_sers" / "tables"

# Family-fingerprint files (richer than the 3-axis autoresearch BSV but
# orthogonal to the v11 ontology — used as a complementary view, not remapped).
SHINE_FAMILY_FINGERPRINT = SHINE_DAY02_TABLES / "class_mean_family_fingerprint_day0_day2.csv"
EV_FAMILY_FINGERPRINT    = EV_DIABETES_TABLES / "cluster_family_fingerprint.csv"
LIVER_FAMILY_FINGERPRINT = LIVER_PATIENT_TABLES / "patient_level_family.csv"

GROUNDING_REGISTRY_TABLES = (AUTORESEARCH_ROOT
                                / "gaira_evidence_warehouse_grounding_backbone_v1"
                                / "tables")

ADENINE_RAW_DIR = GAIRA_DATA_VOLUME / "raw" / "adenine_sers_control"


# ─────────────────────────────────────────────────────────────────────
# 11-axis BSV definition
# ─────────────────────────────────────────────────────────────────────

BSV_AXES: tuple[str, ...] = (
    "G01_purine_nucleotide",
    "G02_purine_metabolite",
    "G03_pyrimidine_nucleotide",
    "G04_nucleic_acid_phosphate",
    "G05_glycan_carbohydrate",
    "G06_protein_peptide_backbone",
    "G07_aromatic_residue",
    "G08_lipid_acyl_membrane",
    "G09_sterol_neutral_lipid",
    "G10_sulfur_thiol_redox",
    "G11_metabolic_small_molecule",
)

AXIS_LABEL_SHORT: dict[str, str] = {
    "G01_purine_nucleotide":      "Purine-nuc",
    "G02_purine_metabolite":      "Purine-met",
    "G03_pyrimidine_nucleotide":  "Pyrimidine",
    "G04_nucleic_acid_phosphate": "Nuc-phosphate",
    "G05_glycan_carbohydrate":    "Glycan",
    "G06_protein_peptide_backbone": "Protein",
    "G07_aromatic_residue":       "Aromatic",
    "G08_lipid_acyl_membrane":    "Lipid",
    "G09_sterol_neutral_lipid":   "Sterol",
    "G10_sulfur_thiol_redox":     "Redox",
    "G11_metabolic_small_molecule": "Metabolite",
}

AXIS_LABEL_LONG: dict[str, str] = {
    "G01_purine_nucleotide":      "Purine nucleotide",
    "G02_purine_metabolite":      "Purine metabolite",
    "G03_pyrimidine_nucleotide":  "Pyrimidine nucleotide",
    "G04_nucleic_acid_phosphate": "Nucleic acid phosphate",
    "G05_glycan_carbohydrate":    "Glycan / carbohydrate",
    "G06_protein_peptide_backbone": "Protein peptide backbone",
    "G07_aromatic_residue":       "Aromatic residue",
    "G08_lipid_acyl_membrane":    "Lipid acyl / membrane",
    "G09_sterol_neutral_lipid":   "Sterol / neutral lipid",
    "G10_sulfur_thiol_redox":     "Sulfur / thiol / redox",
    "G11_metabolic_small_molecule": "Metabolic small molecule",
}

AXIS_COLORS: dict[str, str] = {
    "G01_purine_nucleotide":      "#F87171",
    "G02_purine_metabolite":      "#FB923C",
    "G03_pyrimidine_nucleotide":  "#22D3EE",
    "G04_nucleic_acid_phosphate": "#F472B6",
    "G05_glycan_carbohydrate":    "#C084FC",
    "G06_protein_peptide_backbone": "#FBBF24",
    "G07_aromatic_residue":       "#34D399",
    "G08_lipid_acyl_membrane":    "#60A5FA",
    "G09_sterol_neutral_lipid":   "#A78BFA",
    "G10_sulfur_thiol_redox":     "#FDE68A",
    "G11_metabolic_small_molecule": "#94A3B8",
}


# Map legacy 8-axis BSV → new 11-axis (best-effort, demo-only).
# Where a legacy axis splits into multiple v11 axes, the value is replicated
# (divided equally) across each child. This is *not* a scientifically rigorous
# projection — it exists so we can reuse old cached 8-axis BSVs in
# the new 11-axis visuals. Always flag in UI when used.
LEGACY8_TO_V11: dict[str, tuple[str, ...]] = {
    "membrane_lipid":         ("G08_lipid_acyl_membrane", "G09_sterol_neutral_lipid"),
    "protein_backbone":       ("G06_protein_peptide_backbone",),
    "aromatic_amino_acid":    ("G07_aromatic_residue",),
    "purine_nucleotide":      ("G01_purine_nucleotide", "G02_purine_metabolite"),
    "pyrimidine_nucleotide":  ("G03_pyrimidine_nucleotide",),
    "glycan_carbohydrate":    ("G05_glycan_carbohydrate",),
    "redox_metabolite":       ("G10_sulfur_thiol_redox", "G11_metabolic_small_molecule"),
    "nucleic_acid_backbone":  ("G04_nucleic_acid_phosphate",),
}

# Autoresearch (SHINE/pilot) 8-axis ontology → v11. Same disclaimer applies.
# Autoresearch axes 6/7/8 (matrix_background, substrate_adsorption_bias,
# protocol_sensitive_signal) are NOT biology — they are protocol / artifact
# axes. They are intentionally not mapped to any biological axis; instead the
# UI surfaces them as caveats.
AUTORESEARCH8_TO_V11: dict[str, tuple[str, ...]] = {
    "protein_peptide":            ("G06_protein_peptide_backbone",),
    "lipid_membrane":             ("G08_lipid_acyl_membrane", "G09_sterol_neutral_lipid"),
    "nucleic_acid":               ("G04_nucleic_acid_phosphate",),
    "carbohydrate_glycan":        ("G05_glycan_carbohydrate",),
    "small_molecule_metabolite":  ("G11_metabolic_small_molecule",),
}
# These autoresearch axes do not map to biology; carried as caveats/notes.
AUTORESEARCH_NON_BIOLOGY: tuple[str, ...] = (
    "matrix_background",
    "substrate_adsorption_bias",
    "protocol_sensitive_signal",
)


# ─────────────────────────────────────────────────────────────────────
# Theme palette (dark scientific)
# ─────────────────────────────────────────────────────────────────────

BG_PAGE     = "#0B1220"
BG_PANEL    = "#111827"
BG_PLOT     = "rgba(17, 24, 39, 0.55)"
BG_PAPER    = "rgba(0, 0, 0, 0)"

TEXT_PRIMARY   = "#F1F5F9"
TEXT_SECONDARY = "#CBD5E1"
TEXT_MUTED     = "#94A3B8"
TITLE_COLOR    = "#F8FAFC"

GRID_COLOR      = "rgba(148, 163, 184, 0.18)"
AXIS_LINE_COLOR = "#64748B"
ZERO_LINE_COLOR = "rgba(148, 163, 184, 0.55)"

LEGEND_BG     = "rgba(17, 24, 39, 0.75)"
LEGEND_BORDER = "rgba(148, 163, 184, 0.25)"

OVERLAY_COLORS = ["#60A5FA", "#FBBF24", "#34D399", "#F87171", "#22D3EE", "#C084FC"]
DIVERGE_POS = "#34D399"
DIVERGE_NEG = "#F87171"


# ─────────────────────────────────────────────────────────────────────
# Spectral conventions
# ─────────────────────────────────────────────────────────────────────

WAVENUMBER_MIN = 400.0
WAVENUMBER_MAX = 1800.0
WAVENUMBER_N   = 1401   # 1 cm-1 spacing


# Physics-aware atlas regions (the 8 high-level zones the user specified).
# Each region's downstream content lives in primitive_extraction / atlas tab.
ATLAS_REGIONS: list[dict] = [
    {"label": "Skeletal / ring / metal-ligand", "start": 400, "end": 700,
     "axes": ["G05_glycan_carbohydrate", "G09_sterol_neutral_lipid", "G11_metabolic_small_molecule"]},
    {"label": "Purine / nucleobase-sensitive",   "start": 700, "end": 760,
     "axes": ["G01_purine_nucleotide", "G02_purine_metabolite"]},
    {"label": "Ring breathing / AA / carbohydrate", "start": 760, "end": 900,
     "axes": ["G07_aromatic_residue", "G05_glycan_carbohydrate", "G06_protein_peptide_backbone"]},
    {"label": "C–C / C–O / phosphate / glycan",  "start": 900, "end": 1150,
     "axes": ["G04_nucleic_acid_phosphate", "G05_glycan_carbohydrate"]},
    {"label": "Nucleobase / protein / lipid mixed", "start": 1150, "end": 1350,
     "axes": ["G01_purine_nucleotide", "G03_pyrimidine_nucleotide", "G06_protein_peptide_backbone", "G08_lipid_acyl_membrane"]},
    {"label": "CH deformation / nucleobase / lipid", "start": 1350, "end": 1500,
     "axes": ["G06_protein_peptide_backbone", "G08_lipid_acyl_membrane", "G01_purine_nucleotide"]},
    {"label": "Aromatic / amide / C=C / unsat.",   "start": 1500, "end": 1700,
     "axes": ["G07_aromatic_residue", "G06_protein_peptide_backbone", "G08_lipid_acyl_membrane"]},
    {"label": "Carbonyl / lipid ester / oxidation", "start": 1700, "end": 1800,
     "axes": ["G08_lipid_acyl_membrane", "G09_sterol_neutral_lipid"]},
]


# ─────────────────────────────────────────────────────────────────────
# Evidence / confidence vocab
# ─────────────────────────────────────────────────────────────────────

EVIDENCE_STRENGTHS = ("high", "moderate", "low")
DOMAIN_FITS = ("strong", "caution", "weak")
SUBSTRATE_SENSITIVITIES = ("low", "medium", "high")
SPECIFICITY_LEVELS = ("class-level", "candidate-level", "not supported")


def axis_color(axis: str) -> str:
    return AXIS_COLORS.get(axis, "#94A3B8")


def axis_short(axis: str) -> str:
    return AXIS_LABEL_SHORT.get(axis, axis)


def axis_long(axis: str) -> str:
    return AXIS_LABEL_LONG.get(axis, axis)
