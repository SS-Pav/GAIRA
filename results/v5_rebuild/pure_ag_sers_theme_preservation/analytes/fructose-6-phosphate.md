# fructose-6-phosphate  ·  cross-modal transfer card
*Family: saccharide · expected theme: saccharide_glycan · 4 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.3473** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **saccharide_glycan** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.8287 · **distinctive -0.7297** (null -0.7217, separation -0.008, self-nearest False)
- expected theme rank: Raman #1 → Ag-SERS #2 (top-3 retained: True)
- MSS motif cosine 0.6931 (dominant motif preserved: False)
- redistribution: JSD 0.0588, L1 0.4186; gained **nucleic_purine**, lost **saccharide_glycan** (motif +carboxylate_organic_acid / −glycan_co_network)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **weak** (displacement 0.0162, direction cos 0.6543, vs pure-SERS -0.2729)

*OOD(SERS) 0.1113 · confidence(SERS) 0.2071. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*