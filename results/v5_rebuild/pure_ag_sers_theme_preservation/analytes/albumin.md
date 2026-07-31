# albumin  ·  cross-modal transfer card
*Family: protein · expected theme: protein_peptide · 6 Raman / 5 Ag-SERS spectra · quadrant: Q3 superficial coord match, theme changes*

## Level 1 — Latent fingerprint preservation
- component cosine **0.6025** (Moderate) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **lipid_acyl** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.9286 · **distinctive -0.1934** (null -0.1117, separation -0.0817, self-nearest False)
- expected theme rank: Raman #2 → Ag-SERS #4 (top-3 retained: False)
- MSS motif cosine 0.879 (dominant motif preserved: False)
- redistribution: JSD 0.0273, L1 0.3097; gained **nucleic_purine**, lost **lipid_acyl** (motif +purine_ring_breathing / −protein_amide_backbone)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **strong** (displacement 0.0792, direction cos 0.7726, vs pure-SERS 0.0484)

*OOD(SERS) 0.1286 · confidence(SERS) 0.2398. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*