# ascorbate  ·  cross-modal transfer card
*Family: organic_acid · expected theme: organic_acid_metabolism · 4 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.3917** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **lipid_acyl** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.8605 · **distinctive -0.5823** (null -0.5302, separation -0.0521, self-nearest False)
- expected theme rank: Raman #6 → Ag-SERS #3 (top-3 retained: True)
- MSS motif cosine 0.674 (dominant motif preserved: False)
- redistribution: JSD 0.0551, L1 0.3721; gained **nucleic_purine**, lost **lipid_acyl** (motif +carboxylate_organic_acid / −lipid_acyl_chain)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **weak** (displacement 0.0146, direction cos 0.844, vs pure-SERS 0.481)

*OOD(SERS) 0.1427 · confidence(SERS) 0.2003. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*