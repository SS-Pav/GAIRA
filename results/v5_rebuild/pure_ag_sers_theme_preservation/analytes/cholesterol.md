# cholesterol  ·  cross-modal transfer card
*Family: lipid · expected theme: lipid_acyl|sterol_membrane · 4 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.3357** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **lipid_acyl** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.9181 · **distinctive 0.1114** (null -0.255, separation 0.3664, self-nearest False)
- expected theme rank: Raman #1 → Ag-SERS #4 (top-3 retained: False)
- MSS motif cosine 0.814 (dominant motif preserved: False)
- redistribution: JSD 0.032, L1 0.3327; gained **organic_acid_metabolism**, lost **lipid_acyl** (motif +carboxylate_organic_acid / −protein_amide_backbone)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **weak** (displacement 0.0159, direction cos 0.8744, vs pure-SERS -0.1596)

*OOD(SERS) 0.3183 · confidence(SERS) 0.2197. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*