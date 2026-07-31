# acetoacetate  ·  cross-modal transfer card
*Family: organic_acid · expected theme: organic_acid_metabolism · 4 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.423** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **organic_acid_metabolism** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.9169 · **distinctive 0.446** (null 0.4032, separation 0.0428, self-nearest False)
- expected theme rank: Raman #1 → Ag-SERS #3 (top-3 retained: True)
- MSS motif cosine 0.7716 (dominant motif preserved: True)
- redistribution: JSD 0.0439, L1 0.3606; gained **nucleic_pyrimidine**, lost **organic_acid_metabolism** (motif +pyrimidine_ring / −aromatic_ring_residue)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0465, direction cos 0.8514, vs pure-SERS 0.3478)

*OOD(SERS) 0.2081 · confidence(SERS) 0.2019. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*