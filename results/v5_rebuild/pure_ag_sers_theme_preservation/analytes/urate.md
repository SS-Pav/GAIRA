# urate  ·  cross-modal transfer card
*Family: purine · expected theme: nucleic_purine · 3 Raman / 5 Ag-SERS spectra · quadrant: Q3 superficial coord match, theme changes*

## Level 1 — Latent fingerprint preservation
- component cosine **0.644** (Moderate) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **saccharide_glycan** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.9127 · **distinctive -0.5662** (null -0.1396, separation -0.4266, self-nearest False)
- expected theme rank: Raman #2 → Ag-SERS #1 (top-3 retained: True)
- MSS motif cosine 0.7488 (dominant motif preserved: False)
- redistribution: JSD 0.0355, L1 0.3011; gained **nucleic_purine**, lost **nucleic_pyrimidine** (motif +purine_ring_breathing / −carboxylate_organic_acid)

## Level 3 — Perturbation sensitivity
- **directional depletion (uricase)** on nucleic_purine (oxopurine motif): enzymatic urate removal drops the oxopurine-carbonyl MOTIF sharply (theme layer is diffuse); validates perturbation DIRECTION, not a dose score
  - delta_oxopurine_motif: -0.0602
  - delta_purine_ring_motif: -0.0005
  - purine_theme_delta: -0.0114

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **strong** (displacement 0.0524, direction cos 0.9691, vs pure-SERS 0.0691)

*OOD(SERS) 0.2944 · confidence(SERS) 0.2378. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*