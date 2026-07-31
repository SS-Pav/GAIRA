# ergothioneine  ·  cross-modal transfer card
*Family: cofactor · expected theme: sulfur_antioxidant · 3 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.2852** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **nucleic_purine** → Ag-SERS **nucleic_purine**  (preserved)
- theme cosine: raw 0.9582 · **distinctive 0.2507** (null 0.3447, separation -0.0941, self-nearest False)
- expected theme rank: Raman #4 → Ag-SERS #5 (top-3 retained: False)
- MSS motif cosine 0.7263 (dominant motif preserved: False)
- redistribution: JSD 0.0196, L1 0.1726; gained **lipid_acyl**, lost **nucleic_purine** (motif +flavin_redox_cofactor / −sulfur_heterocycle_thione)

## Level 3 — Perturbation sensitivity
- **concentration dose-response** on sulfur_antioxidant: sulfur theme rises monotonically and saturates — a reproducible dose-response
  - monotonicity_rho: 0.927
  - saturating_K_uM: 1.521
  - saturating_r2: 0.957

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **strong** (displacement 0.0979, direction cos 0.9921, vs pure-SERS 0.5342)

*OOD(SERS) 0.307 · confidence(SERS) 0.2571. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*