# glycine · Representation-hierarchy assessment (V3)
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.2465 |
| 2 · MSS motif | mss cosine | 0.8973 |
| 3 · Theme (raw) | theme cosine raw | 0.8118 |
| 3 · Theme (identity) | baseline-subtracted | -0.2531 (null -0.3689, sep 0.1158) |
| 4 · Theme rank | Spearman ρ | 0.7636 (sep 0.0175) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.333 |
| 6 · Argmax agreement | dominant theme | sulfur_antioxidant → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.1117 (0.0814 → 0.1931) |

## Layer 8 — Interpretation
Adsorption-driven observation bias: on silver this analyte is pulled toward the nucleic_purine attractor, so its distinctive Raman abstraction is not recovered even where surface-level structure partially transfers.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine -0.25 / rank separation +0.018.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).
- Weak serum-matrix recoverability (displacement 0.015): competition on colloid suppresses recovery.

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*