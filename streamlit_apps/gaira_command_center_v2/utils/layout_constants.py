"""Shared layout constants for v2 Tab 2 (chemistry-grouped positions, labels)."""
from __future__ import annotations


# Manual chemistry-grouped positions for the 11 BSV families.
# Used by both the family-first cluster map and the axis-overlap network.
#
#   nucleic / purine     → upper-left
#   glycan / phosphate   → upper-centre
#   protein / aromatic   → upper-right
#   lipid / sterol       → right / lower-right
#   sulfur / metabolic   → bottom
AXIS_POSITIONS: dict[str, tuple[float, float]] = {
    "G01": (-1.20,  0.95),  # purine_nucleotide
    "G02": (-1.50,  0.20),  # purine_metabolite
    "G03": (-1.20, -0.45),  # pyrimidine_nucleotide
    "G04": (-0.30,  0.65),  # nucleic_acid_phosphate
    "G05": ( 0.30,  0.65),  # glycan_carbohydrate
    "G06": ( 1.20,  0.95),  # protein_peptide_backbone
    "G07": ( 1.50,  0.20),  # aromatic_residue
    "G08": ( 1.50, -0.50),  # lipid_acyl_membrane
    "G09": ( 1.00, -1.10),  # sterol_neutral_lipid
    "G10": (-0.55, -1.10),  # sulfur_thiol_redox
    "G11": ( 0.10, -1.30),  # metabolic_small_molecule
}


# Curated short summaries of each BSV axis. Stable independent of artifacts.
AXIS_INFO: dict[str, tuple[str, str, str, str]] = {
    "G01": ("purine_nucleotide",
            "adenine · guanine · AMP/GMP/ATP/GTP",
            "720-740 (ring breath) · 1336 · 1480 · 1576",
            "shares ring-breath with G02 metabolites"),
    "G02": ("purine_metabolite",
            "uric acid · hypoxanthine · xanthine",
            "640 (HX) · 891 (UA distinctive) · 1517",
            "UA 1517 ↔ carotenoid 1525 in serum"),
    "G03": ("pyrimidine_nucleotide",
            "cytosine · thymine · uracil",
            "780 (ring) · 1240 · 1660",
            "C/T/U very similar within family"),
    "G04": ("nucleic_acid_phosphate",
            "DNA/RNA backbone · sugar-phosphate",
            "1080-1100 (PO₂⁻) · 814 · 835",
            "1080 shared with glycan C-O-C (G05)"),
    "G05": ("glycan_carbohydrate",
            "glucose · fructose · sucrose · glycogen",
            "1080 (anomeric) · 1126 · 1340 · 1460",
            "1080 collides with phosphate (G04)"),
    "G06": ("protein_peptide_backbone",
            "albumin · collagen · peptides",
            "1003 (Phe) · 1450 · 1655 (amide-I) · 1245",
            "amide-I overlaps lipid C=C (G08)"),
    "G07": ("aromatic_residue",
            "Phe · Tyr · Trp",
            "1003 (Phe) · 853/828 (Tyr) · 759/1554 (Trp)",
            "1003 shared with G06 backbone"),
    "G08": ("lipid_acyl_membrane",
            "palmitic · oleic · linoleic · phospholipids",
            "1299 (CH₂ twist) · 1440 · 1655 · 2850-2935",
            "1655 overlaps amide-I (G06)"),
    "G09": ("sterol_neutral_lipid",
            "cholesterol · cholesteryl ester · triglyceride",
            "608 · 700 · 1745 (ester C=O) · 1080+1265",
            "triglyceride lacks 608/700; subfamily routing"),
    "G10": ("sulfur_thiol_redox",
            "cysteine · cystine · GSH · ergothioneine",
            "510 (S-S) · 660 (C-S) · 912 (GSH) · 1220 (ergo)",
            "GSH 912 overlaps sugar anomeric"),
    "G11": ("metabolic_small_molecule",
            "lactate · urea · creatinine · citrate · glutamate",
            "858 (lactate) · 1003-1014 (urea) · 605/685 (creat)",
            "sparse axis — many small molecules without atlas zones"),
}


# Major biochemical-class labels worth annotating on the MSS UMAP.
# Anything else is in the legend but not annotated, to keep the figure readable.
MAJOR_CLASS_LABELS: set[str] = {
    "protein_polypeptide",
    "free_amino_acid",
    "free_fatty_acid",
    "triglyceride",
    "sugar",
    "tryptophan_indole",
    "organic_acid_metabolite",
    "sterol",
    "purine_nucleobase",
    "sulfur_amino_acid",
}


# Canonical Raman bands shown as dashed guidelines on the saliency heatmap.
CANONICAL_BANDS: list[tuple[int, str]] = [
    (725, "purine ring"),
    (785, "pyrimidine ring"),
    (1003, "Phe / G06↔G07"),
    (1080, "phosphate / glycan"),
    (1299, "lipid CH₂ twist"),
    (1450, "amide-III / lipid bend"),
    (1655, "amide-I / lipid C=C"),
    (1745, "ester C=O (G09)"),
]
