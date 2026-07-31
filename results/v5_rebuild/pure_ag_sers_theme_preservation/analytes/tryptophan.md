# tryptophan  ·  cross-modal transfer card
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide · 5 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.3394** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **nucleic_purine** → Ag-SERS **nucleic_purine**  (preserved)
- theme cosine: raw 0.9305 · **distinctive 0.2139** (null 0.3667, separation -0.1528, self-nearest False)
- expected theme rank: Raman #3 → Ag-SERS #4 (top-3 retained: False)
- MSS motif cosine 0.6793 (dominant motif preserved: False)
- redistribution: JSD 0.03, L1 0.3277; gained **saccharide_glycan**, lost **aromatic_amino_acid** (motif +sulfur_heterocycle_thione / −flavin_redox_cofactor)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.038, direction cos 0.951, vs pure-SERS -0.1942)

*OOD(SERS) 0.1089 · confidence(SERS) 0.2177. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*