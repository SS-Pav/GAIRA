# triolein  ·  cross-modal transfer card
*Family: lipid · expected theme: lipid_acyl|sterol_membrane · 5 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.3138** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **lipid_acyl** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.5219 · **distinctive -0.3572** (null -0.4846, separation 0.1274, self-nearest False)
- expected theme rank: Raman #1 → Ag-SERS #6 (top-3 retained: False)
- MSS motif cosine 0.3694 (dominant motif preserved: False)
- redistribution: JSD 0.2197, L1 0.7858; gained **nucleic_purine**, lost **lipid_acyl** (motif +carboxylate_organic_acid / −lipid_acyl_chain)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **weak** (displacement 0.0154, direction cos 0.7683, vs pure-SERS 0.1027)

*OOD(SERS) 0.1456 · confidence(SERS) 0.2033. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*