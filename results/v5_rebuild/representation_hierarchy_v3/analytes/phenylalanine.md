# phenylalanine · Representation-hierarchy assessment (V3)
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.3007 |
| 2 · MSS motif | mss cosine | 0.5299 |
| 3 · Theme (raw) | theme cosine raw | 0.8306 |
| 3 · Theme (identity) | baseline-subtracted | -0.0917 (null -0.0599, sep -0.0318) |
| 4 · Theme rank | Spearman ρ | 0.6455 (sep 0.0218) |
| 5 · Top-k overlap | top-2 / top-3 | 0.0 / 0.333 |
| 6 · Argmax agreement | dominant theme | sulfur_antioxidant → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.1295 (0.075 → 0.2044) |

## Layer 8 — Interpretation
Adsorption-driven observation bias: on silver this analyte is pulled toward the nucleic_purine attractor, so its distinctive Raman abstraction is not recovered even where surface-level structure partially transfers.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine -0.09 / rank separation +0.022.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*