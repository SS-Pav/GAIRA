# stearate  ·  cross-modal transfer card
*Family: lipid · expected theme: lipid_acyl|sterol_membrane · 5 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.3209** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **lipid_acyl** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.5662 · **distinctive 0.0776** (null -0.2541, separation 0.3318, self-nearest False)
- expected theme rank: Raman #1 → Ag-SERS #5 (top-3 retained: False)
- MSS motif cosine 0.3722 (dominant motif preserved: False)
- redistribution: JSD 0.1894, L1 0.7759; gained **organic_acid_metabolism**, lost **lipid_acyl** (motif +carboxylate_organic_acid / −lipid_acyl_chain)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0423, direction cos 0.9088, vs pure-SERS 0.1114)

*OOD(SERS) 0.168 · confidence(SERS) 0.2096. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*