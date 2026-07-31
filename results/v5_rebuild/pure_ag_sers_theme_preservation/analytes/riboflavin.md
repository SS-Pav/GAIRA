# riboflavin  ·  cross-modal transfer card
*Family: cofactor · expected theme: redox_broad · 3 Raman / 5 Ag-SERS spectra · quadrant: Q2 latent redistribution, theme survives*

## Level 1 — Latent fingerprint preservation
- component cosine **0.3394** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **nucleic_purine** → Ag-SERS **nucleic_purine**  (preserved)
- theme cosine: raw 0.972 · **distinctive 0.7086** (null 0.6597, separation 0.0489, self-nearest False)
- expected theme rank: Raman #9 → Ag-SERS #11 (top-3 retained: False)
- MSS motif cosine 0.6387 (dominant motif preserved: False)
- redistribution: JSD 0.0262, L1 0.2373; gained **organic_acid_metabolism**, lost **nucleic_pyrimidine** (motif +carboxylate_organic_acid / −flavin_redox_cofactor)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0298, direction cos 0.898, vs pure-SERS -0.0369)

*OOD(SERS) 0.1289 · confidence(SERS) 0.2067. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*