# serine  ·  cross-modal transfer card
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide · 5 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.3536** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **saccharide_glycan** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.8496 · **distinctive -0.6139** (null -0.5968, separation -0.0171, self-nearest False)
- expected theme rank: Raman #2 → Ag-SERS #4 (top-3 retained: False)
- MSS motif cosine 0.6897 (dominant motif preserved: False)
- redistribution: JSD 0.0588, L1 0.4355; gained **nucleic_purine**, lost **saccharide_glycan** (motif +carboxylate_organic_acid / −aromatic_ring_residue)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0285, direction cos 0.6258, vs pure-SERS 0.0184)

*OOD(SERS) 0.1701 · confidence(SERS) 0.1917. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*