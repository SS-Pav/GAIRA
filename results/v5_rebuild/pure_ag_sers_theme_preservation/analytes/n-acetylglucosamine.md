# n-acetylglucosamine  ·  cross-modal transfer card
*Family: saccharide · expected theme: saccharide_glycan · 3 Raman / 5 Ag-SERS spectra · quadrant: Q3 superficial coord match, theme changes*

## Level 1 — Latent fingerprint preservation
- component cosine **0.5987** (Moderate) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **saccharide_glycan** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.9014 · **distinctive -0.6277** (null -0.5965, separation -0.0313, self-nearest False)
- expected theme rank: Raman #1 → Ag-SERS #2 (top-3 retained: True)
- MSS motif cosine 0.8527 (dominant motif preserved: False)
- redistribution: JSD 0.0316, L1 0.3343; gained **nucleic_purine**, lost **saccharide_glycan** (motif +purine_ring_breathing / −glycan_co_network)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **weak** (displacement 0.0135, direction cos 0.7807, vs pure-SERS -0.1325)

*OOD(SERS) 0.1453 · confidence(SERS) 0.1942. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*