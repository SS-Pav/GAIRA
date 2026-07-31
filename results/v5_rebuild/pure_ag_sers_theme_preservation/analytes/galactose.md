# galactose  ·  cross-modal transfer card
*Family: saccharide · expected theme: saccharide_glycan · 3 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.4462** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **saccharide_glycan** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.9241 · **distinctive -0.2446** (null -0.3684, separation 0.1237, self-nearest False)
- expected theme rank: Raman #1 → Ag-SERS #2 (top-3 retained: True)
- MSS motif cosine 0.7966 (dominant motif preserved: False)
- redistribution: JSD 0.0245, L1 0.2868; gained **protein_peptide**, lost **saccharide_glycan** (motif +sulfur_heterocycle_thione / −glycan_co_network)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0408, direction cos 0.9232, vs pure-SERS -0.0124)

*OOD(SERS) 0.1177 · confidence(SERS) 0.2074. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*