# Analyte Classification Overlay — Provenance

*Every subclass / expected-MSS / expected-theme assignment in
`tables/analyte_classification_overlay.csv`, and why. This overlay is an **evaluation scaffold
only** — it is never fed back into GAIRA's ontology, BSV, or MSS scoring, and never becomes a new
representation axis. All assignments trace to molecular chemistry, the current 12 MSS motif
definitions, and the current 11 GAIRA themes. `mixed`/`unassigned` is allowed; labels are never
forced for coverage.*

## Assignment rules

- **Subclass** = molecular chemistry (functional-group / scaffold), not a spectral output.
- **Expected MSS** = the motif whose *definition* matches the analyte's chemistry AND which is
  activated on the Raman reference — NOT the motif that happens to be high in Ag-SERS.
- **Expected theme** = the curated `FAM_THEME`/`ANALYTE_THEME` map reused from V2–V4.
- **Multi-label** where chemically real (e.g. guanine amino+oxo; CoA cofactors purine+thiol).
- **Confidence** high/medium/low reflects how cleanly the chemistry maps to an existing motif/theme.

## Traces (grouped)

- **Purines** — adenine=aminopurine (`purine_ring_breathing`); guanine/hypoxanthine/xanthine/urate
  =oxopurine (`oxopurine_carbonyl` + `purine_ring_breathing`); theme `nucleic_purine`. High conf.
- **Pyrimidines** — thymine/uracil=oxopyrimidine (`pyrimidine_ring`, `nucleic_pyrimidine`). High.
- **Aromatic amino acids** — tyr/trp/phe/his (`aromatic_ring_residue`; `aromatic_amino_acid` +
  `protein_peptide`). his medium (imidazole).
- **Sulfur amino acids** — cys (thiol) / met (thioether) → `sulfur_heterocycle_thione`
  (approximate — that motif is defined for thiones/S-heterocycles), `sulfur_antioxidant`. Medium,
  flagged approximate.
- **Aliphatic/polar amino acids** — ala/gly/val/leu/ile/ser/pro/hyp (`protein_amide_backbone`,
  `protein_peptide`); glutamate=acidic (multi-label organic-acid); arginine=basic/guanidino. Medium.
- **Cofactors** — acetyl-coa / coenzyme a = **purine_cofactor** (contain adenine → a purine theme is
  *legitimate chemistry, not an attractor artifact*), multi-label with thiol; glutathione=thiol
  peptide; **ergothioneine=thione** (clean `sulfur_heterocycle_thione` match); riboflavin=flavin
  (`flavin_redox_cofactor`, `redox_broad`) — exploratory singleton.
- **Lipids** — cholesterol=sterol (`sterol_ring_system`); oleate/stearate=fatty acid, triolein=TAG,
  phosphatidylinositol=phospholipid (all `lipid_acyl_chain`, `lipid_acyl`). Several exploratory.
- **Organic acids** — acetoacetate/ascorbate/citrate/lactate/pyruvate/PEP = carboxylic acid
  (`carboxylate_organic_acid`, `organic_acid_metabolism`); **phosphate = inorganic**, no clean
  motif → MSS unassigned, theme weak-fit, low conf, exploratory.
- **Saccharides** — glucose/fructose/galactose/mannose=monosaccharide; fructose-6-phosphate
  =phosphorylated sugar; N-acetylglucosamine=amino sugar (all `glycan_co_network`,
  `saccharide_glycan`). Several exploratory.
- **Polyol / polysaccharide / protein** — glycerol=polyol (weak, grouped with glycans); glycogen
  =polysaccharide; albumin=protein (`protein_amide_backbone`, `protein_peptide`). Exploratory singletons.
- **Small nitrogenous** — creatinine / urea = guanidino small-N; no strong GAIRA theme → MSS
  unassigned, theme `organic_acid_metabolism` weak-fit, low conf.

## Low-count / exploratory subclasses
Singletons (n=1) cannot be evaluated by leave-one-analyte-out (the class centroid disappears when
the sole member is held out): acidic_amino_acid, amino_sugar, aminopurine, basic_amino_acid, flavin,
inorganic_phosphate, phospholipid, phosphorylated_sugar, polyol, polysaccharide, protein, sterol,
thiol_peptide, thione, triacylglycerol. **Reported separately; excluded from the primary
subclass-accuracy denominator.** Subclasses with ≥2 members used for LOAO accuracy: aliphatic_amino_acid,
aromatic_amino_acid, carboxylic_acid, fatty_acid, guanidino_small_n, monosaccharide, oxopurine,
oxopyrimidine, purine_cofactor, sulfur_amino_acid.

## Ambiguous / unassigned (not forced)
guanine (amino+oxopurine, multi-label); glutamate (amino acid + carboxylate); CoA cofactors (purine +
thiol); phosphate, creatinine, urea (no clean MSS → `unassigned` expected-MSS). These are documented,
never silently coerced into a single label.
