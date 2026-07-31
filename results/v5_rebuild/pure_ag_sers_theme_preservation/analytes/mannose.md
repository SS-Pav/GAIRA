# mannose  ·  cross-modal transfer card
*Family: saccharide · expected theme: saccharide_glycan · 3 Raman / 5 Ag-SERS spectra · quadrant: Q3 superficial coord match, theme changes*

## Level 1 — Latent fingerprint preservation
- component cosine **0.6013** (Moderate) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **saccharide_glycan** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.8841 · **distinctive -0.6059** (null -0.5032, separation -0.1027, self-nearest False)
- expected theme rank: Raman #1 → Ag-SERS #2 (top-3 retained: True)
- MSS motif cosine 0.7375 (dominant motif preserved: False)
- redistribution: JSD 0.0363, L1 0.352; gained **nucleic_purine**, lost **saccharide_glycan** (motif +carboxylate_organic_acid / −glycan_co_network)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0447, direction cos 0.9415, vs pure-SERS -0.078)

*OOD(SERS) 0.1162 · confidence(SERS) 0.2056. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*