# urea  ·  cross-modal transfer card
*Family: small_nitrogenous · expected theme: organic_acid_metabolism · 3 Raman / 5 Ag-SERS spectra · quadrant: Q1 identity preserved (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.6674** (Good) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **sulfur_antioxidant** → Ag-SERS **saccharide_glycan**  (not preserved)
- theme cosine: raw 0.8827 · **distinctive 0.9025** (null -0.0978, separation 1.0003, self-nearest True)
- expected theme rank: Raman #3 → Ag-SERS #3 (top-3 retained: True)
- MSS motif cosine 0.5222 (dominant motif preserved: False)
- redistribution: JSD 0.0674, L1 0.4546; gained **nucleic_purine**, lost **sulfur_antioxidant** (motif +sulfur_heterocycle_thione / −aromatic_ring_residue)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **weak** (displacement 0.0111, direction cos 0.6318, vs pure-SERS -0.1499)

*OOD(SERS) 0.2901 · confidence(SERS) 0.1807. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*