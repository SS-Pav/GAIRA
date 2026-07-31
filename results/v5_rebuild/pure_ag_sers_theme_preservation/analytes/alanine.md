# alanine  ·  cross-modal transfer card
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide · 5 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.2553** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **saccharide_glycan** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.8867 · **distinctive -0.4841** (null -0.4569, separation -0.0272, self-nearest False)
- expected theme rank: Raman #3 → Ag-SERS #4 (top-3 retained: False)
- MSS motif cosine 0.6624 (dominant motif preserved: False)
- redistribution: JSD 0.0537, L1 0.4234; gained **nucleic_purine**, lost **saccharide_glycan** (motif +carboxylate_organic_acid / −aromatic_ring_residue)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **weak** (displacement 0.0156, direction cos 0.7869, vs pure-SERS -0.2113)

*OOD(SERS) 0.1256 · confidence(SERS) 0.1993. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*