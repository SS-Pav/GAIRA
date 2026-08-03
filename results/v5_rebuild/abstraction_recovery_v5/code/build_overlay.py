"""V5 — chemical classification overlay (evaluation only; NOT a new GAIRA ontology axis).

Assigns each of the 51 matched analytes a molecular subclass + expected MSS motif(s) + expected
theme(s), each traceable to molecular chemistry, the current MSS definitions, and the current
GAIRA themes. Multi-label where chemically justified; 'mixed'/'unassigned' allowed. Singleton
subclasses are marked exploratory. Writes tables/analyte_classification_overlay.csv.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

OUT = Path("/Users/surajpg/projects/GAIRA/results/v5_rebuild/abstraction_recovery_v5")

# analyte: (broad_family, subclass_primary, subclass_secondary, exp_mss_primary, exp_mss_secondary,
#           exp_theme_primary, exp_theme_secondary, source, confidence, notes)
M = "molecular chemistry"; MSSDEF = "MSS motif definition"; ONT = "GAIRA theme"
OVERLAY = {
    # ── purines ──
    "adenine": ("purine", "aminopurine", "", "purine_ring_breathing", "", "nucleic_purine", "", M, "high", "6-aminopurine"),
    "guanine": ("purine", "oxopurine", "aminopurine", "oxopurine_carbonyl", "purine_ring_breathing", "nucleic_purine", "", M, "high", "2-amino-6-oxopurine (multi-label)"),
    "hypoxanthine": ("purine", "oxopurine", "", "oxopurine_carbonyl", "purine_ring_breathing", "nucleic_purine", "", M, "high", "6-oxopurine"),
    "xanthine": ("purine", "oxopurine", "", "oxopurine_carbonyl", "purine_ring_breathing", "nucleic_purine", "", M, "high", "2,6-dioxopurine"),
    "urate": ("purine", "oxopurine", "", "oxopurine_carbonyl", "", "nucleic_purine", "", M, "high", "uric acid; oxopurine"),
    # ── pyrimidines ──
    "thymine": ("pyrimidine", "oxopyrimidine", "", "pyrimidine_ring", "", "nucleic_pyrimidine", "", M, "high", "5-methyl oxopyrimidine"),
    "uracil": ("pyrimidine", "oxopyrimidine", "", "pyrimidine_ring", "", "nucleic_pyrimidine", "", M, "high", "oxopyrimidine"),
    # ── amino acids: aromatic ──
    "tyrosine": ("amino_acid", "aromatic_amino_acid", "", "aromatic_ring_residue", "protein_amide_backbone", "aromatic_amino_acid", "protein_peptide", M, "high", "phenol ring"),
    "tryptophan": ("amino_acid", "aromatic_amino_acid", "", "aromatic_ring_residue", "protein_amide_backbone", "aromatic_amino_acid", "protein_peptide", M, "high", "indole ring"),
    "phenylalanine": ("amino_acid", "aromatic_amino_acid", "", "aromatic_ring_residue", "protein_amide_backbone", "aromatic_amino_acid", "protein_peptide", M, "high", "benzyl ring"),
    "histidine": ("amino_acid", "aromatic_amino_acid", "", "aromatic_ring_residue", "protein_amide_backbone", "aromatic_amino_acid", "protein_peptide", M, "medium", "imidazole (aromatic N-heterocycle)"),
    # ── amino acids: sulfur ──
    "cysteine": ("amino_acid", "sulfur_amino_acid", "", "sulfur_heterocycle_thione", "protein_amide_backbone", "sulfur_antioxidant", "protein_peptide", M, "medium", "thiol; sulfur MSS is approximate (thione-defined)"),
    "methionine": ("amino_acid", "sulfur_amino_acid", "", "sulfur_heterocycle_thione", "protein_amide_backbone", "sulfur_antioxidant", "protein_peptide", M, "medium", "thioether; sulfur MSS approximate"),
    # ── amino acids: aliphatic/polar ──
    **{a: ("amino_acid", "aliphatic_amino_acid", "", "protein_amide_backbone", "", "protein_peptide", "", M, "medium", "amide backbone dominant")
       for a in ["alanine", "glycine", "valine", "leucine", "isoleucine", "serine", "proline", "hydroxyproline"]},
    "glutamate": ("amino_acid", "acidic_amino_acid", "aliphatic_amino_acid", "protein_amide_backbone", "carboxylate_organic_acid", "protein_peptide", "organic_acid_metabolism", M, "medium", "dicarboxylic amino acid (multi-label)"),
    "arginine": ("amino_acid", "basic_amino_acid", "aliphatic_amino_acid", "protein_amide_backbone", "", "protein_peptide", "", M, "medium", "guanidino side chain"),
    # ── cofactors ──
    "acetyl-coa": ("cofactor", "purine_cofactor", "thiol_cofactor", "purine_ring_breathing", "sulfur_heterocycle_thione", "nucleic_purine", "sulfur_antioxidant", M, "high", "contains adenine + thioester; purine theme is LEGITIMATE chemistry"),
    "coenzyme a": ("cofactor", "purine_cofactor", "thiol_cofactor", "purine_ring_breathing", "sulfur_heterocycle_thione", "nucleic_purine", "sulfur_antioxidant", M, "high", "contains adenine + free thiol; purine theme legitimate"),
    "glutathione": ("cofactor", "thiol_peptide", "", "sulfur_heterocycle_thione", "protein_amide_backbone", "sulfur_antioxidant", "protein_peptide", M, "medium", "gamma-glutamyl-cys-gly thiol"),
    "ergothioneine": ("cofactor", "thione", "", "sulfur_heterocycle_thione", "", "sulfur_antioxidant", "", M, "high", "histidine-derived thione — strong MSS match"),
    "riboflavin": ("cofactor", "flavin", "", "flavin_redox_cofactor", "", "redox_broad", "", M, "high", "isoalloxazine flavin; EXPLORATORY singleton"),
    # ── lipids ──
    "cholesterol": ("lipid", "sterol", "", "sterol_ring_system", "", "sterol_membrane", "", M, "high", "sterol; EXPLORATORY singleton"),
    "oleate": ("lipid", "fatty_acid", "", "lipid_acyl_chain", "carboxylate_organic_acid", "lipid_acyl", "", M, "high", "C18:1 fatty acid"),
    "stearate": ("lipid", "fatty_acid", "", "lipid_acyl_chain", "carboxylate_organic_acid", "lipid_acyl", "", M, "high", "C18:0 fatty acid"),
    "triolein": ("lipid", "triacylglycerol", "", "lipid_acyl_chain", "", "lipid_acyl", "", M, "high", "triacylglycerol; EXPLORATORY singleton"),
    "phosphatidylinositol": ("lipid", "phospholipid", "", "lipid_acyl_chain", "", "lipid_acyl", "", M, "high", "phospholipid; EXPLORATORY singleton"),
    # ── organic acids ──
    **{a: ("organic_acid", "carboxylic_acid", "", "carboxylate_organic_acid", "", "organic_acid_metabolism", "", M, "high", "carboxylate")
       for a in ["acetoacetate", "ascorbate", "citrate", "lactate", "pyruvate"]},
    "phosphoenolpyruvate": ("organic_acid", "carboxylic_acid", "phosphorylated_metabolite", "carboxylate_organic_acid", "", "organic_acid_metabolism", "", M, "high", "phosphoenol + carboxylate"),
    "phosphate": ("organic_acid", "inorganic_phosphate", "", "", "", "organic_acid_metabolism", "", M, "low", "inorganic; no clean biochemical MSS; theme weak fit; EXPLORATORY singleton"),
    # ── saccharides ──
    **{a: ("saccharide", "monosaccharide", "", "glycan_co_network", "", "saccharide_glycan", "", M, "high", "hexose")
       for a in ["fructose", "galactose", "glucose", "mannose"]},
    "fructose-6-phosphate": ("saccharide", "phosphorylated_sugar", "", "glycan_co_network", "", "saccharide_glycan", "", M, "medium", "phosphorylated hexose; EXPLORATORY singleton"),
    "n-acetylglucosamine": ("saccharide", "amino_sugar", "", "glycan_co_network", "", "saccharide_glycan", "", M, "medium", "N-acetyl amino sugar; EXPLORATORY singleton"),
    # ── polyol / polysaccharide / protein ──
    "glycerol": ("polyol", "polyol", "", "glycan_co_network", "", "saccharide_glycan", "", M, "low", "polyol grouped with glycans; EXPLORATORY singleton"),
    "glycogen": ("polysaccharide", "polysaccharide", "", "glycan_co_network", "", "saccharide_glycan", "", M, "high", "glucose polymer; EXPLORATORY singleton"),
    "albumin": ("protein", "protein", "", "protein_amide_backbone", "aromatic_ring_residue", "protein_peptide", "aromatic_amino_acid", M, "high", "protein; EXPLORATORY singleton"),
    # ── small nitrogenous ──
    "creatinine": ("small_nitrogenous", "guanidino_small_n", "", "", "", "organic_acid_metabolism", "", M, "low", "cyclic guanidino; no strong theme; weak fit"),
    "urea": ("small_nitrogenous", "guanidino_small_n", "", "", "", "organic_acid_metabolism", "", M, "low", "urea; no strong theme; weak fit"),
}

rows = []
for a, v in OVERLAY.items():
    bf, s1, s2, m1, m2, t1, t2, src, conf, note = v
    rows.append({"canonical_analyte": a, "broad_family": bf, "subclass_primary": s1,
                 "subclass_secondary": s2, "expected_mss_primary": m1, "expected_mss_secondary": m2,
                 "expected_theme_primary": t1, "expected_theme_secondary": t2,
                 "assignment_source": src, "confidence": conf, "notes": note,
                 "min_evidence_tier": ("motif" if m1 else "theme" if t1 else "family")})
df = pd.DataFrame(rows).sort_values(["broad_family", "subclass_primary", "canonical_analyte"])
# mark exploratory (singleton) subclasses
sc = df.subclass_primary.value_counts()
df["subclass_exploratory"] = df.subclass_primary.map(lambda s: sc[s] < 2)
df["subclass_n"] = df.subclass_primary.map(sc)
df.to_csv(OUT / "tables/analyte_classification_overlay.csv", index=False)
print(f"overlay: {len(df)} analytes")
print("subclass counts:"); print(sc.to_string())
print("\nsubclasses >=2 (used for primary LOAO accuracy):", sorted(sc[sc >= 2].index))
print("exploratory singletons:", sorted(sc[sc < 2].index))
print("expected_mss unassigned:", df[df.expected_mss_primary == ""].canonical_analyte.tolist())
