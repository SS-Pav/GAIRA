# tyrosine  ·  cross-modal transfer card
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide · 4 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.236** (Poor) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **nucleic_purine** → Ag-SERS **nucleic_purine**  (preserved)
- theme cosine: raw 0.9473 · **distinctive 0.1874** (null 0.1409, separation 0.0465, self-nearest False)
- expected theme rank: Raman #3 → Ag-SERS #4 (top-3 retained: False)
- MSS motif cosine 0.6292 (dominant motif preserved: False)
- redistribution: JSD 0.0302, L1 0.2707; gained **organic_acid_metabolism**, lost **aromatic_amino_acid** (motif +carboxylate_organic_acid / −aromatic_ring_residue)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0327, direction cos 0.8283, vs pure-SERS 0.0088)

*OOD(SERS) 0.1573 · confidence(SERS) 0.1962. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*