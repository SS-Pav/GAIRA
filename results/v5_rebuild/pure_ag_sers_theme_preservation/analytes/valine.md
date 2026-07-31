# valine  ·  cross-modal transfer card
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide · 5 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.4249** (Weak) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **saccharide_glycan** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.9668 · **distinctive 0.2074** (null 0.174, separation 0.0333, self-nearest False)
- expected theme rank: Raman #4 → Ag-SERS #4 (top-3 retained: False)
- MSS motif cosine 0.8199 (dominant motif preserved: True)
- redistribution: JSD 0.016, L1 0.2035; gained **protein_peptide**, lost **saccharide_glycan** (motif +sulfur_heterocycle_thione / −lipid_acyl_chain)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0336, direction cos 0.7394, vs pure-SERS 0.0327)

*OOD(SERS) 0.1585 · confidence(SERS) 0.199. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*