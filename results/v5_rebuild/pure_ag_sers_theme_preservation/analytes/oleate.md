# oleate  ·  cross-modal transfer card
*Family: lipid · expected theme: lipid_acyl|sterol_membrane · 5 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.264** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **lipid_acyl** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.5072 · **distinctive -0.4675** (null -0.4814, separation 0.0139, self-nearest False)
- expected theme rank: Raman #1 → Ag-SERS #6 (top-3 retained: False)
- MSS motif cosine 0.3648 (dominant motif preserved: False)
- redistribution: JSD 0.2234, L1 0.793; gained **nucleic_purine**, lost **lipid_acyl** (motif +sulfur_heterocycle_thione / −lipid_acyl_chain)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **strong** (displacement 0.1075, direction cos 0.8987, vs pure-SERS -0.1239)

*OOD(SERS) 0.1173 · confidence(SERS) 0.2078. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*