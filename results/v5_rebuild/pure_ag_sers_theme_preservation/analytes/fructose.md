# fructose  ·  cross-modal transfer card
*Family: saccharide · expected theme: saccharide_glycan · 3 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.3411** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **saccharide_glycan** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.7526 · **distinctive -0.6003** (null -0.6484, separation 0.0481, self-nearest False)
- expected theme rank: Raman #1 → Ag-SERS #2 (top-3 retained: True)
- MSS motif cosine 0.5823 (dominant motif preserved: False)
- redistribution: JSD 0.1153, L1 0.6139; gained **nucleic_purine**, lost **saccharide_glycan** (motif +carboxylate_organic_acid / −glycan_co_network)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0278, direction cos 0.9066, vs pure-SERS 0.1197)

*OOD(SERS) 0.1429 · confidence(SERS) 0.1941. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*