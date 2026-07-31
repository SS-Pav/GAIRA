# citrate  ·  cross-modal transfer card
*Family: organic_acid · expected theme: organic_acid_metabolism · 4 Raman / 5 Ag-SERS spectra · quadrant: Q1 identity preserved (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.6183** (Moderate) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **organic_acid_metabolism** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.8426 · **distinctive 0.8517** (null 0.7608, separation 0.0908, self-nearest False)
- expected theme rank: Raman #1 → Ag-SERS #3 (top-3 retained: True)
- MSS motif cosine 0.6956 (dominant motif preserved: True)
- redistribution: JSD 0.084, L1 0.5688; gained **saccharide_glycan**, lost **organic_acid_metabolism** (motif +glycan_co_network / −carboxylate_organic_acid)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0203, direction cos 0.7131, vs pure-SERS 0.0403)

*OOD(SERS) 0.1465 · confidence(SERS) 0.1899. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*