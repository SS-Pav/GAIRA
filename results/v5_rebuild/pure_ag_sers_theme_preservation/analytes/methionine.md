# methionine  ·  cross-modal transfer card
*Family: amino_acid · expected theme: sulfur_antioxidant · 4 Raman / 5 Ag-SERS spectra · quadrant: Q1 identity preserved (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.5575** (Moderate) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **nucleic_purine** → Ag-SERS **nucleic_purine**  (preserved)
- theme cosine: raw 0.9546 · **distinctive 0.8119** (null 0.7509, separation 0.061, self-nearest False)
- expected theme rank: Raman #7 → Ag-SERS #5 (top-3 retained: False)
- MSS motif cosine 0.8735 (dominant motif preserved: False)
- redistribution: JSD 0.0186, L1 0.257; gained **saccharide_glycan**, lost **nucleic_purine** (motif +pyrimidine_ring / −purine_ring_breathing)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **weak** (displacement 0.0186, direction cos 0.6228, vs pure-SERS -0.2341)

*OOD(SERS) 0.1276 · confidence(SERS) 0.1989. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*