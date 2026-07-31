# glycogen  ·  cross-modal transfer card
*Family: polysaccharide · expected theme: saccharide_glycan · 4 Raman / 5 Ag-SERS spectra · quadrant: Q3 superficial coord match, theme changes*

## Level 1 — Latent fingerprint preservation
- component cosine **0.6837** (Good) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **saccharide_glycan** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.9572 · **distinctive 0.0647** (null 0.0727, separation -0.0081, self-nearest False)
- expected theme rank: Raman #1 → Ag-SERS #2 (top-3 retained: True)
- MSS motif cosine 0.9023 (dominant motif preserved: True)
- redistribution: JSD 0.0201, L1 0.2392; gained **nucleic_purine**, lost **saccharide_glycan** (motif +sulfur_heterocycle_thione / −glycan_co_network)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0294, direction cos 0.8857, vs pure-SERS 0.0581)

*OOD(SERS) 0.137 · confidence(SERS) 0.211. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*