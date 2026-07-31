# histidine  ·  cross-modal transfer card
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide · 5 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.4868** (Moderate) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **saccharide_glycan** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.9256 · **distinctive -0.7794** (null -0.622, separation -0.1574, self-nearest False)
- expected theme rank: Raman #3 → Ag-SERS #4 (top-3 retained: False)
- MSS motif cosine 0.8677 (dominant motif preserved: False)
- redistribution: JSD 0.0256, L1 0.2962; gained **nucleic_purine**, lost **saccharide_glycan** (motif +carboxylate_organic_acid / −glycan_co_network)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0489, direction cos 0.9612, vs pure-SERS -0.1347)

*OOD(SERS) 0.1542 · confidence(SERS) 0.1969. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*