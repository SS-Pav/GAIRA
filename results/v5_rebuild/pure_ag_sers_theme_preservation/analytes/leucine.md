# leucine  ·  cross-modal transfer card
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide · 4 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.4373** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **saccharide_glycan** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.9349 · **distinctive -0.6361** (null -0.5114, separation -0.1246, self-nearest False)
- expected theme rank: Raman #3 → Ag-SERS #4 (top-3 retained: False)
- MSS motif cosine 0.7764 (dominant motif preserved: False)
- redistribution: JSD 0.0308, L1 0.2913; gained **nucleic_purine**, lost **saccharide_glycan** (motif +pyrimidine_ring / −aromatic_ring_residue)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **weak** (displacement 0.0149, direction cos 0.7077, vs pure-SERS 0.1321)

*OOD(SERS) 0.1281 · confidence(SERS) 0.2031. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*