# phosphatidylinositol  ·  cross-modal transfer card
*Family: lipid · expected theme: lipid_acyl|sterol_membrane · 3 Raman / 5 Ag-SERS spectra · quadrant: Q3 superficial coord match, theme changes*

## Level 1 — Latent fingerprint preservation
- component cosine **0.8339** (Excellent) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **nucleic_purine** → Ag-SERS **nucleic_purine**  (preserved)
- theme cosine: raw 0.9888 · **distinctive 0.4695** (null 0.5101, separation -0.0407, self-nearest False)
- expected theme rank: Raman #7 → Ag-SERS #7 (top-3 retained: False)
- MSS motif cosine 0.9558 (dominant motif preserved: False)
- redistribution: JSD 0.0061, L1 0.1339; gained **nucleic_pyrimidine**, lost **saccharide_glycan** (motif +porphyrin_macrocycle / −oxopurine_carbonyl)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **weak** (displacement 0.0169, direction cos 0.7265, vs pure-SERS 0.0562)

*OOD(SERS) 0.1654 · confidence(SERS) 0.1853. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*