# cysteine  ·  cross-modal transfer card
*Family: amino_acid · expected theme: sulfur_antioxidant · 3 Raman / 5 Ag-SERS spectra · quadrant: Q1 identity preserved (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.58** (Moderate) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **nucleic_purine** → Ag-SERS **nucleic_purine**  (preserved)
- theme cosine: raw 0.9608 · **distinctive 0.6942** (null 0.6889, separation 0.0053, self-nearest False)
- expected theme rank: Raman #6 → Ag-SERS #5 (top-3 retained: False)
- MSS motif cosine 0.8088 (dominant motif preserved: False)
- redistribution: JSD 0.0346, L1 0.2496; gained **nucleic_pyrimidine**, lost **nucleic_purine** (motif +pyrimidine_ring / −oxopurine_carbonyl)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0252, direction cos 0.8659, vs pure-SERS -0.021)

*OOD(SERS) 0.1228 · confidence(SERS) 0.2075. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*