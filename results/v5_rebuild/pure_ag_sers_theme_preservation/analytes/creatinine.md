# creatinine  ·  cross-modal transfer card
*Family: small_nitrogenous · expected theme: organic_acid_metabolism · 3 Raman / 5 Ag-SERS spectra · quadrant: Q1 identity preserved (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.6877** (Good) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **nucleic_purine** → Ag-SERS **nucleic_purine**  (preserved)
- theme cosine: raw 0.9659 · **distinctive 0.8747** (null 0.7143, separation 0.1604, self-nearest True)
- expected theme rank: Raman #3 → Ag-SERS #3 (top-3 retained: True)
- MSS motif cosine 0.6521 (dominant motif preserved: True)
- redistribution: JSD 0.0353, L1 0.2698; gained **nucleic_pyrimidine**, lost **protein_peptide** (motif +carboxylate_organic_acid / −sulfur_heterocycle_thione)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **weak** (displacement 0.0182, direction cos 0.8327, vs pure-SERS 0.5843)

*OOD(SERS) 0.1536 · confidence(SERS) 0.1999. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*