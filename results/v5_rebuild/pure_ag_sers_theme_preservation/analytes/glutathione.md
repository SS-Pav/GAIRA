# glutathione  ·  cross-modal transfer card
*Family: cofactor · expected theme: sulfur_antioxidant · 4 Raman / 5 Ag-SERS spectra · quadrant: Q1 identity preserved (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.7271** (Good) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **nucleic_purine** → Ag-SERS **nucleic_purine**  (preserved)
- theme cosine: raw 0.9858 · **distinctive 0.6482** (null 0.5213, separation 0.1269, self-nearest False)
- expected theme rank: Raman #4 → Ag-SERS #5 (top-3 retained: False)
- MSS motif cosine 0.9554 (dominant motif preserved: True)
- redistribution: JSD 0.0063, L1 0.1405; gained **nucleic_purine**, lost **sulfur_antioxidant** (motif +purine_ring_breathing / −flavin_redox_cofactor)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0287, direction cos 0.9123, vs pure-SERS -0.1048)

*OOD(SERS) 0.1361 · confidence(SERS) 0.2291. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*