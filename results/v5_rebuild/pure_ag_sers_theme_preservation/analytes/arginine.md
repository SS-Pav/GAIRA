# arginine  ·  cross-modal transfer card
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide · 5 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.4943** (Moderate) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **saccharide_glycan** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.8175 · **distinctive -0.9317** (null -0.8089, separation -0.1228, self-nearest False)
- expected theme rank: Raman #6 → Ag-SERS #4 (top-3 retained: False)
- MSS motif cosine 0.7433 (dominant motif preserved: False)
- redistribution: JSD 0.0709, L1 0.452; gained **nucleic_purine**, lost **lipid_acyl** (motif +carboxylate_organic_acid / −glycan_co_network)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0432, direction cos 0.9078, vs pure-SERS -0.1194)

*OOD(SERS) 0.2065 · confidence(SERS) 0.1856. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*