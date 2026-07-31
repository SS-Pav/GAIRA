# glucose  ·  cross-modal transfer card
*Family: saccharide · expected theme: saccharide_glycan · 4 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.2555** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **saccharide_glycan** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.8458 · **distinctive -0.4605** (null -0.4806, separation 0.0201, self-nearest False)
- expected theme rank: Raman #1 → Ag-SERS #2 (top-3 retained: True)
- MSS motif cosine 0.5358 (dominant motif preserved: False)
- redistribution: JSD 0.0615, L1 0.4474; gained **organic_acid_metabolism**, lost **saccharide_glycan** (motif +carboxylate_organic_acid / −glycan_co_network)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **weak** (displacement 0.0143, direction cos 0.7681, vs pure-SERS -0.2571)

*OOD(SERS) 0.124 · confidence(SERS) 0.2052. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*