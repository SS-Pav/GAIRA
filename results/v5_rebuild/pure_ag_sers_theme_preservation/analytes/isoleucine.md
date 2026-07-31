# isoleucine  ·  cross-modal transfer card
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide · 3 Raman / 5 Ag-SERS spectra · quadrant: Q3 superficial coord match, theme changes*

## Level 1 — Latent fingerprint preservation
- component cosine **0.5932** (Moderate) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **saccharide_glycan** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.9379 · **distinctive -0.9043** (null -0.8225, separation -0.0818, self-nearest False)
- expected theme rank: Raman #6 → Ag-SERS #4 (top-3 retained: False)
- MSS motif cosine 0.8301 (dominant motif preserved: False)
- redistribution: JSD 0.0196, L1 0.2676; gained **nucleic_purine**, lost **saccharide_glycan** (motif +carboxylate_organic_acid / −lipid_acyl_chain)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0452, direction cos 0.927, vs pure-SERS -0.029)

*OOD(SERS) 0.1672 · confidence(SERS) 0.1963. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*