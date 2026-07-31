# pyruvate  ·  cross-modal transfer card
*Family: organic_acid · expected theme: organic_acid_metabolism · 4 Raman / 5 Ag-SERS spectra · quadrant: Q1 identity preserved (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.6083** (Moderate) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **nucleic_purine** → Ag-SERS **nucleic_purine**  (preserved)
- theme cosine: raw 0.9333 · **distinctive 0.6639** (null 0.6848, separation -0.0209, self-nearest False)
- expected theme rank: Raman #2 → Ag-SERS #3 (top-3 retained: True)
- MSS motif cosine 0.8026 (dominant motif preserved: False)
- redistribution: JSD 0.0326, L1 0.3138; gained **saccharide_glycan**, lost **organic_acid_metabolism** (motif +sulfur_heterocycle_thione / −carboxylate_organic_acid)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **weak** (displacement 0.0162, direction cos 0.7965, vs pure-SERS 0.0452)

*OOD(SERS) 0.1116 · confidence(SERS) 0.2097. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*