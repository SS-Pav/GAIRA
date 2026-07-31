# glycine  ·  cross-modal transfer card
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide · 5 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.2465** (Poor) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **sulfur_antioxidant** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.8118 · **distinctive -0.2531** (null -0.3689, separation 0.1158, self-nearest False)
- expected theme rank: Raman #3 → Ag-SERS #4 (top-3 retained: False)
- MSS motif cosine 0.8973 (dominant motif preserved: False)
- redistribution: JSD 0.0962, L1 0.5704; gained **nucleic_purine**, lost **sulfur_antioxidant** (motif +carboxylate_organic_acid / −flavin_redox_cofactor)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **weak** (displacement 0.0154, direction cos 0.8538, vs pure-SERS -0.0333)

*OOD(SERS) 0.1254 · confidence(SERS) 0.2117. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*