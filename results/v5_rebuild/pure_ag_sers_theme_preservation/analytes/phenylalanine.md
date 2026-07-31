# phenylalanine  ·  cross-modal transfer card
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide · 5 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.3007** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **sulfur_antioxidant** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.8306 · **distinctive -0.0917** (null -0.0599, separation -0.0318, self-nearest False)
- expected theme rank: Raman #2 → Ag-SERS #4 (top-3 retained: False)
- MSS motif cosine 0.5299 (dominant motif preserved: False)
- redistribution: JSD 0.0819, L1 0.4933; gained **nucleic_purine**, lost **sulfur_antioxidant** (motif +carboxylate_organic_acid / −aromatic_ring_residue)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0389, direction cos 0.9762, vs pure-SERS 0.0588)

*OOD(SERS) 0.1166 · confidence(SERS) 0.2036. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*