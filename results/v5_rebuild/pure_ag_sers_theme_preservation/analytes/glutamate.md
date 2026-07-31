# glutamate  ·  cross-modal transfer card
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide · 6 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.3337** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **saccharide_glycan** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.9535 · **distinctive 0.1533** (null 0.1489, separation 0.0045, self-nearest False)
- expected theme rank: Raman #4 → Ag-SERS #4 (top-3 retained: False)
- MSS motif cosine 0.6553 (dominant motif preserved: False)
- redistribution: JSD 0.0235, L1 0.2738; gained **nucleic_purine**, lost **organic_acid_metabolism** (motif +sulfur_heterocycle_thione / −flavin_redox_cofactor)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **weak** (displacement 0.0156, direction cos 0.7806, vs pure-SERS -0.2801)

*OOD(SERS) 0.1491 · confidence(SERS) 0.2059. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*