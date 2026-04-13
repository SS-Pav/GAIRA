"""
Spectral band explanation — motif/theme annotations for observed BSV shifts.

IMPORTANT: This module is ANNOTATION ONLY. It does not compute observed BSV.
Observed BSV is computed by direct spectral projection (bsv_projection.py).
This module maps observed BSV window drivers to candidate motifs/themes
for interpretive support.
"""
from __future__ import annotations


# Motif annotations: window → candidate molecular assignments
# Each motif is a plausible assignment, not a definitive identification
WINDOW_MOTIF_ANNOTATIONS = {
    "450-500":   [{"motif": "disulfide S-S stretch",  "theme": "redox_metabolite",       "ambiguity": "low"}],
    "500-540":   [{"motif": "disulfide / thiol",      "theme": "redox_metabolite",       "ambiguity": "low"}],
    "540-580":   [{"motif": "unresolved",             "theme": "unmapped",               "ambiguity": "high"}],
    "580-620":   [{"motif": "unresolved",             "theme": "unmapped",               "ambiguity": "high"}],
    "620-660":   [{"motif": "tyrosine ring",          "theme": "aromatic_amino_acid",    "ambiguity": "low"},
                  {"motif": "C-S stretch",            "theme": "redox_metabolite",       "ambiguity": "moderate"}],
    "660-700":   [{"motif": "guanine / adenine ring", "theme": "purine_nucleotide",      "ambiguity": "moderate"}],
    "700-740":   [{"motif": "adenine ring breathing", "theme": "purine_nucleotide",      "ambiguity": "low"}],
    "740-780":   [{"motif": "cytosine / thymine",     "theme": "pyrimidine_nucleotide",  "ambiguity": "moderate"}],
    "780-820":   [{"motif": "cytosine ring",          "theme": "pyrimidine_nucleotide",  "ambiguity": "moderate"},
                  {"motif": "O-P-O backbone",         "theme": "nucleic_acid_backbone",  "ambiguity": "moderate"}],
    "820-860":   [{"motif": "tyrosine Fermi doublet", "theme": "aromatic_amino_acid",    "ambiguity": "low"}],
    "860-920":   [{"motif": "C-O-C glycosidic",       "theme": "glycan_carbohydrate",    "ambiguity": "low"},
                  {"motif": "proline / collagen",     "theme": "protein_backbone",       "ambiguity": "moderate"}],
    "920-980":   [{"motif": "C-C backbone stretch",   "theme": "protein_backbone",       "ambiguity": "low"}],
    "980-1020":  [{"motif": "phenylalanine ring",     "theme": "aromatic_amino_acid",    "ambiguity": "low"}],
    "1020-1080": [{"motif": "PO₂⁻ symmetric stretch","theme": "nucleic_acid_backbone",  "ambiguity": "low"},
                  {"motif": "C-O stretch (carb.)",    "theme": "glycan_carbohydrate",    "ambiguity": "moderate"}],
    "1080-1140": [{"motif": "C-O / C-C (glycan)",     "theme": "glycan_carbohydrate",    "ambiguity": "moderate"},
                  {"motif": "PO₂⁻ asymmetric",       "theme": "nucleic_acid_backbone",  "ambiguity": "moderate"}],
    "1140-1200": [{"motif": "C-C / C-O lipid chain",  "theme": "membrane_lipid",         "ambiguity": "low"}],
    "1200-1260": [{"motif": "amide III / C-N",        "theme": "protein_backbone",       "ambiguity": "low"}],
    "1260-1320": [{"motif": "amide III",              "theme": "protein_backbone",       "ambiguity": "low"}],
    "1320-1380": [{"motif": "purine base vibration",  "theme": "purine_nucleotide",      "ambiguity": "moderate"},
                  {"motif": "CH₂ wag (lipid)",        "theme": "membrane_lipid",         "ambiguity": "moderate"}],
    "1380-1450": [{"motif": "CH₂/CH₃ deformation",   "theme": "membrane_lipid",         "ambiguity": "low"}],
    "1450-1520": [{"motif": "CH₂ scissoring / amide II", "theme": "protein_backbone",    "ambiguity": "moderate"}],
    "1520-1600": [{"motif": "aromatic C=C stretch",   "theme": "aromatic_amino_acid",    "ambiguity": "low"},
                  {"motif": "amide II (partial)",     "theme": "protein_backbone",       "ambiguity": "moderate"}],
}

THEME_DISPLAY = {
    "membrane_lipid": "Membrane / Lipid",
    "protein_backbone": "Protein Backbone",
    "aromatic_amino_acid": "Aromatic Amino Acids",
    "purine_nucleotide": "Purine Nucleotides",
    "pyrimidine_nucleotide": "Pyrimidine Nucleotides",
    "glycan_carbohydrate": "Glycan / Carbohydrate",
    "redox_metabolite": "Redox / Metabolite",
    "nucleic_acid_backbone": "Nucleic Acid Backbone",
    "unmapped": "Unresolved",
}


def annotate_windows(window_drivers: list[dict]) -> list[dict]:
    """Annotate a list of window drivers with motif/theme explanations.

    Takes output from band_drivers.compute_window_importance() and adds
    motif annotations. Does NOT alter any BSV values.

    Returns enriched list with 'annotations' field per window.
    """
    result = []
    for w in window_drivers:
        wid = w["window_id"]
        annotations = WINDOW_MOTIF_ANNOTATIONS.get(wid, [])
        enriched = dict(w)
        enriched["annotations"] = annotations
        enriched["primary_motif"] = annotations[0]["motif"] if annotations else "unresolved"
        enriched["primary_theme"] = annotations[0]["theme"] if annotations else "unmapped"
        enriched["multi_assignment"] = len(annotations) > 1
        result.append(enriched)
    return result
