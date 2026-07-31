# guanine  ·  cross-modal transfer card
*Family: purine · expected theme: nucleic_purine · 4 Raman / 5 Ag-SERS spectra · quadrant: Q1 identity preserved (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.631** (Moderate) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **nucleic_purine** → Ag-SERS **nucleic_purine**  (preserved)
- theme cosine: raw 0.8758 · **distinctive 0.9241** (null 0.7088, separation 0.2153, self-nearest False)
- expected theme rank: Raman #1 → Ag-SERS #1 (top-3 retained: True)
- MSS motif cosine 0.8745 (dominant motif preserved: True)
- redistribution: JSD 0.0964, L1 0.5694; gained **saccharide_glycan**, lost **nucleic_purine** (motif +aromatic_ring_residue / −oxopurine_carbonyl)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **strong** (displacement 0.0803, direction cos 0.9873, vs pure-SERS 0.4021)

*OOD(SERS) 0.159 · confidence(SERS) 0.2327. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*