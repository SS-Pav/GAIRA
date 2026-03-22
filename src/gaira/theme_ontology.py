from __future__ import annotations

from dataclasses import dataclass


POSITIVE_THEMES = [
    "lipid_membrane_associated",
    "protein_peptide_associated",
    "nucleic_acid_purine_associated",
    "carbohydrate_glycan_associated",
    "oxidative_metabolic_stress_associated",
]

CAUTION_THEMES = [
    "matrix_dominance_caution",
    "probe_substrate_caution",
    "modality_mismatch_caution",
    "weak_label_or_cohort_caution",
    "low_specificity_caution",
]


@dataclass(frozen=True)
class ThemeDefinition:
    theme_name: str
    category: str
    description: str
    keywords: tuple[str, ...]
    anchor_bands_cm: tuple[float, ...]
    negative_keywords: tuple[str, ...] = ()


EVIDENCE_WEIGHTS = {
    "tier1": 1.00,
    "tier2": 0.65,
    "knowledge": 0.60,
    "semantic": 0.50,
    "context": 0.35,
    "band": 0.45,
}


THEME_ONTOLOGY: dict[str, ThemeDefinition] = {
    "lipid_membrane_associated": ThemeDefinition(
        theme_name="lipid_membrane_associated",
        category="positive",
        description="Membrane-rich, lipid-like, or CH-dominant biochemical structure.",
        keywords=(
            "lipid",
            "membrane",
            "phospholipid",
            "cholesterol",
            "fatty",
            "ch2",
            "ch3",
            "vesicle membrane",
            "phosphatidyl",
            "membrane-associated",
        ),
        anchor_bands_cm=(718.0, 1060.0, 1295.0, 1445.0, 1655.0),
        negative_keywords=(
            "amide",
            "protein",
            "peptide",
            "albumin",
            "hemoglobin",
            "adenine",
            "purine",
            "dna",
            "rna",
            "glycan",
            "glucose",
        ),
    ),
    "protein_peptide_associated": ThemeDefinition(
        theme_name="protein_peptide_associated",
        category="positive",
        description="Amide-like, peptide-like, or proteinaceous biochemical structure.",
        keywords=(
            "protein",
            "peptide",
            "amide",
            "albumin",
            "hemoglobin",
            "collagen",
            "phenylalanine",
            "tyrosine",
            "tryptophan",
            "serum protein",
        ),
        anchor_bands_cm=(725.0, 1003.0, 1240.0, 1450.0, 1660.0),
        negative_keywords=(
            "lipid",
            "membrane",
            "phospholipid",
            "cholesterol",
            "adenine",
            "purine",
            "dna",
            "rna",
            "glycan",
            "saccharide",
        ),
    ),
    "nucleic_acid_purine_associated": ThemeDefinition(
        theme_name="nucleic_acid_purine_associated",
        category="positive",
        description="Nucleic-acid-like, purine-like, or nucleobase-like support.",
        keywords=(
            "dna",
            "rna",
            "adenine",
            "purine",
            "nucleic",
            "nucleobase",
            "guanine",
            "uric",
            "hypoxanthine",
            "adenosine",
            "nucleotide",
        ),
        anchor_bands_cm=(725.0, 733.0, 782.0, 1337.0, 1485.0, 1578.0),
        negative_keywords=(
            "amide",
            "protein",
            "peptide",
            "albumin",
            "collagen",
            "lipid",
            "membrane",
            "phospholipid",
            "glycan",
            "saccharide",
        ),
    ),
    "carbohydrate_glycan_associated": ThemeDefinition(
        theme_name="carbohydrate_glycan_associated",
        category="positive",
        description="Carbohydrate-like, glycan-like, or saccharide-rich support.",
        keywords=(
            "carbohydrate",
            "glycan",
            "glyco",
            "saccharide",
            "glucose",
            "glycogen",
            "polysaccharide",
            "sugar",
            "hexose",
        ),
        anchor_bands_cm=(850.0, 915.0, 1045.0, 1080.0, 1125.0),
        negative_keywords=(
            "amide",
            "protein",
            "peptide",
            "adenine",
            "purine",
            "dna",
            "rna",
            "lipid",
            "membrane",
        ),
    ),
    "oxidative_metabolic_stress_associated": ThemeDefinition(
        theme_name="oxidative_metabolic_stress_associated",
        category="positive",
        description="Metabolite-turnover or oxidative/metabolic stress support.",
        keywords=(
            "oxidative",
            "metabolic",
            "stress",
            "lactate",
            "uric acid",
            "uric",
            "hypoxanthine",
            "metabolite",
            "turnover",
            "redox",
            "catabolic",
        ),
        anchor_bands_cm=(725.0, 733.0, 1003.0, 1200.0, 1655.0),
        negative_keywords=(
            "structural membrane",
            "stable phospholipid",
            "collagen",
            "glycan",
        ),
    ),
    "matrix_dominance_caution": ThemeDefinition(
        theme_name="matrix_dominance_caution",
        category="caution",
        description="Matrix-heavy or background-dominant interpretation caution.",
        keywords=(
            "matrix",
            "dominance",
            "background",
            "overlap region",
            "protein dominance",
            "metabolite dominance",
            "serum-local",
            "hemoglobin contamination",
            "broad overlap",
        ),
        anchor_bands_cm=(725.0, 1450.0, 1659.0),
    ),
    "probe_substrate_caution": ThemeDefinition(
        theme_name="probe_substrate_caution",
        category="caution",
        description="Probe, substrate, adsorption, or protocol sensitivity caution.",
        keywords=(
            "probe",
            "substrate",
            "batch",
            "adsorption",
            "protocol",
            "cross-substrate",
            "probe1",
            "probe2",
            "strip variability",
            "shelf-life",
            "nanopillar",
            "colloid",
        ),
        anchor_bands_cm=(725.0, 1003.0, 1450.0),
    ),
    "modality_mismatch_caution": ThemeDefinition(
        theme_name="modality_mismatch_caution",
        category="caution",
        description="Caution when spontaneous Raman and SERS evidence are mixed.",
        keywords=(
            "spontaneous raman",
            "sers-heavy",
            "modality caution",
            "modality mismatch",
            "not equivalent",
            "not interchangeable",
        ),
        anchor_bands_cm=(),
    ),
    "weak_label_or_cohort_caution": ThemeDefinition(
        theme_name="weak_label_or_cohort_caution",
        category="caution",
        description="Weak-label, small-cohort, or coarse cohort-family caution.",
        keywords=(
            "weak label",
            "weak-label",
            "cohort",
            "suspected",
            "archive-supported cohort",
            "not defensible",
            "cohort family",
            "small cohort",
            "stress-test",
        ),
        anchor_bands_cm=(),
    ),
    "low_specificity_caution": ThemeDefinition(
        theme_name="low_specificity_caution",
        category="caution",
        description="Caution when evidence is broad, conflicting, or only analog/support-level.",
        keywords=(
            "support-only",
            "broad analog",
            "not definitive",
            "do not overclaim",
            "calibration-like",
            "controlled grounding",
            "mixed signature",
            "overclaim",
            "broad molecular",
            "conservative",
        ),
        anchor_bands_cm=(),
    ),
}


def get_theme_definitions() -> dict[str, ThemeDefinition]:
    return THEME_ONTOLOGY
