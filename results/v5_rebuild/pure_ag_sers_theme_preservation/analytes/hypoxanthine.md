# hypoxanthine  ·  cross-modal transfer card
*Family: purine · expected theme: nucleic_purine · 3 Raman / 5 Ag-SERS spectra · quadrant: Q1 identity preserved (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.8443** (Excellent) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **nucleic_purine** → Ag-SERS **nucleic_purine**  (preserved)
- theme cosine: raw 0.9914 · **distinctive 0.8269** (null 0.475, separation 0.3518, self-nearest True)
- expected theme rank: Raman #1 → Ag-SERS #1 (top-3 retained: True)
- MSS motif cosine 0.9642 (dominant motif preserved: False)
- redistribution: JSD 0.0068, L1 0.1248; gained **nucleic_pyrimidine**, lost **lipid_acyl** (motif +sterol_ring_system / −lipid_acyl_chain)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **strong** (displacement 0.1137, direction cos 0.9834, vs pure-SERS 0.8888)

*OOD(SERS) 0.1313 · confidence(SERS) 0.2463. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*