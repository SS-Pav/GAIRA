# phosphate  ·  cross-modal transfer card
*Family: organic_acid · expected theme: organic_acid_metabolism · 3 Raman / 5 Ag-SERS spectra · quadrant: Q2 latent redistribution, theme survives*

## Level 1 — Latent fingerprint preservation
- component cosine **0.361** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **organic_acid_metabolism** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.9183 · **distinctive 0.603** (null 0.54, separation 0.063, self-nearest False)
- expected theme rank: Raman #1 → Ag-SERS #3 (top-3 retained: True)
- MSS motif cosine 0.6495 (dominant motif preserved: True)
- redistribution: JSD 0.0404, L1 0.2911; gained **protein_peptide**, lost **organic_acid_metabolism** (motif +purine_ring_breathing / −carboxylate_organic_acid)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **weak** (displacement 0.0127, direction cos 0.6831, vs pure-SERS 0.0437)

*OOD(SERS) 0.1248 · confidence(SERS) 0.2017. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*