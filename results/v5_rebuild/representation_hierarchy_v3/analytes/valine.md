# valine · Representation-hierarchy assessment (V3)
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.4249 |
| 2 · MSS motif | mss cosine | 0.8199 |
| 3 · Theme (raw) | theme cosine raw | 0.9668 |
| 3 · Theme (identity) | baseline-subtracted | 0.2074 (null 0.174, sep 0.0333) |
| 4 · Theme rank | Spearman ρ | 0.9455 (sep 0.0093) |
| 5 · Top-k overlap | top-2 / top-3 | 1.0 / 1.0 |
| 6 · Argmax agreement | dominant theme | saccharide_glycan → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.0076 (0.2057 → 0.2132) |

## Layer 8 — Interpretation
Partial mid-level transfer: motif structure carries over but the identity-specific theme abstraction is weak — a surface-physics limit, not a representation error.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.21 / rank separation +0.009.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*