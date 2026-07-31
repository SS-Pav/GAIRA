# uracil  ·  cross-modal transfer card
*Family: pyrimidine · expected theme: nucleic_pyrimidine · 4 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.0562** (Poor) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **nucleic_pyrimidine** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.45 · **distinctive 0.1591** (null 0.0649, separation 0.0942, self-nearest False)
- expected theme rank: Raman #1 → Ag-SERS #8 (top-3 retained: False)
- MSS motif cosine 0.4858 (dominant motif preserved: False)
- redistribution: JSD 0.2629, L1 0.8929; gained **nucleic_purine**, lost **nucleic_pyrimidine** (motif +sulfur_heterocycle_thione / −pyrimidine_ring)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0353, direction cos 0.9371, vs pure-SERS 0.118)

*OOD(SERS) 0.1659 · confidence(SERS) 0.2101. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*