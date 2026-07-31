# xanthine  ·  cross-modal transfer card
*Family: purine · expected theme: nucleic_purine · 3 Raman / 5 Ag-SERS spectra · quadrant: Q1 identity preserved (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.8145** (Excellent) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **nucleic_purine** → Ag-SERS **nucleic_purine**  (preserved)
- theme cosine: raw 0.9751 · **distinctive 0.9557** (null 0.6076, separation 0.3481, self-nearest True)
- expected theme rank: Raman #1 → Ag-SERS #1 (top-3 retained: True)
- MSS motif cosine 0.9303 (dominant motif preserved: True)
- redistribution: JSD 0.0389, L1 0.3287; gained **organic_acid_metabolism**, lost **nucleic_purine** (motif +pyrimidine_ring / −oxopurine_carbonyl)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **strong** (displacement 0.1202, direction cos 0.9947, vs pure-SERS 0.7356)

*OOD(SERS) 0.1743 · confidence(SERS) 0.2908. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*