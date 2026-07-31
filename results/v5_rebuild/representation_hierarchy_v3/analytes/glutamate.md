# glutamate · Representation-hierarchy assessment (V3)
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.3337 |
| 2 · MSS motif | mss cosine | 0.6553 |
| 3 · Theme (raw) | theme cosine raw | 0.9535 |
| 3 · Theme (identity) | baseline-subtracted | 0.1533 (null 0.1489, sep 0.0045) |
| 4 · Theme rank | Spearman ρ | 0.9273 (sep 0.0198) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 1.0 |
| 6 · Argmax agreement | dominant theme | saccharide_glycan → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.0577 (0.1498 → 0.2075) |

## Layer 8 — Interpretation
Partial mid-level transfer: motif structure carries over but the identity-specific theme abstraction is weak — a surface-physics limit, not a representation error.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.15 / rank separation +0.020.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).
- Weak serum-matrix recoverability (displacement 0.016): competition on colloid suppresses recovery.

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*