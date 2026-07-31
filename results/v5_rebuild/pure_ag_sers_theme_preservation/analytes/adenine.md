# adenine  ·  cross-modal transfer card
*Family: purine · expected theme: nucleic_purine · 4 Raman / 5 Ag-SERS spectra · quadrant: Q2 latent redistribution, theme survives*

## Level 1 — Latent fingerprint preservation
- component cosine **0.3592** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **nucleic_purine** → Ag-SERS **nucleic_purine**  (preserved)
- theme cosine: raw 0.9466 · **distinctive 0.733** (null 0.7069, separation 0.0261, self-nearest False)
- expected theme rank: Raman #1 → Ag-SERS #1 (top-3 retained: True)
- MSS motif cosine 0.7405 (dominant motif preserved: False)
- redistribution: JSD 0.0301, L1 0.3242; gained **organic_acid_metabolism**, lost **nucleic_purine** (motif +pyrimidine_ring / −sterol_ring_system)

## Level 3 — Perturbation sensitivity
- **concentration dose-response** on nucleic_purine: purine theme rises monotonically and saturates (Langmuir) — a reproducible dose-response
  - monotonicity_rho: 0.996
  - saturating_K_uM: 0.893
  - saturating_r2: 0.993

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **strong** (displacement 0.0745, direction cos 0.9625, vs pure-SERS 0.0796)

*OOD(SERS) 0.1388 · confidence(SERS) 0.2195. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*