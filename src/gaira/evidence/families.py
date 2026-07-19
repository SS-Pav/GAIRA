"""Curated chemical-family labels for the 785 nm grounding analytes.

EVALUATION-ONLY semantic labels from known chemistry (NOT inferred from spectra,
NOT used to train the primary Stage B models). Conservative: an analyte with no
confident assignment maps to "unknown". Non-small-molecule references are kept in
a separate flag (see datasets.NON_SMALL_MOLECULE) and excluded from family
neighborhood analyses.

Families are deliberately coarse and defensible; ambiguity is flagged, not forced.
"""
from __future__ import annotations

# canonical analyte -> coarse chemical family (known chemistry)
FAMILY = {
    # amino acids
    "alanine": "amino_acid", "arginine": "amino_acid", "asparagine": "amino_acid",
    "cysteine": "amino_acid", "cystine": "amino_acid", "cystathionine": "amino_acid",
    "glutamate": "amino_acid", "glycine": "amino_acid", "histidine": "amino_acid",
    "hydroxyproline": "amino_acid", "isoleucine": "amino_acid", "leucine": "amino_acid",
    "lysine": "amino_acid", "methionine": "amino_acid", "phenylalanine": "amino_acid",
    "proline": "amino_acid", "serine": "amino_acid", "tryptophan": "amino_acid",
    "tyrosine": "amino_acid", "valine": "amino_acid",
    # purines (bases / derivatives)
    "adenine": "purine", "guanine": "purine", "hypoxanthine": "purine",
    "xanthine": "purine", "urate": "purine",
    # pyrimidines
    "uracil": "pyrimidine", "thymine": "pyrimidine", "cytosine": "pyrimidine",
    # saccharides / sugars & derivatives
    "glucose": "saccharide", "fructose": "saccharide", "galactose": "saccharide",
    "mannose": "saccharide", "fructose-6-phosphate": "saccharide",
    "n-acetylglucosamine": "saccharide",
    # lipids / fatty acids
    "oleate": "lipid", "stearate": "lipid", "cholesterol": "lipid",
    "triolein": "lipid", "phosphatidylinositol": "lipid",
    # organic acids / small metabolites
    "citrate": "organic_acid", "lactate": "organic_acid", "pyruvate": "organic_acid",
    "acetoacetate": "organic_acid", "ascorbate": "organic_acid",
    "phosphoenolpyruvate": "organic_acid", "phosphate": "organic_acid",
    "glycerol": "polyol", "urea": "small_nitrogenous", "creatinine": "small_nitrogenous",
    # cofactors / vitamins
    "riboflavin": "cofactor", "coenzyme a": "cofactor", "acetyl-coa": "cofactor",
    "glutathione": "cofactor", "ergothioneine": "cofactor",
    # macromolecules (kept separate; non-small-molecule)
    "dna": "nucleic_acid", "rna": "nucleic_acid", "albumin": "protein",
    "glycogen": "polysaccharide",
}

# analytes whose family assignment is genuinely ambiguous / borderline
AMBIGUOUS = {"ergothioneine", "acetyl-coa", "coenzyme a", "phosphate"}


def family_of(analyte: str) -> str:
    return FAMILY.get(analyte, "unknown")


def is_ambiguous(analyte: str) -> bool:
    return analyte in AMBIGUOUS
