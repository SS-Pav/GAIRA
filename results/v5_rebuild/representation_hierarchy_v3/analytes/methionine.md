# methionine · Representation-hierarchy assessment (V3)
*Family: amino_acid · expected theme: sulfur_antioxidant*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.5575 |
| 2 · MSS motif | mss cosine | 0.8735 |
| 3 · Theme (raw) | theme cosine raw | 0.9546 |
| 3 · Theme (identity) | baseline-subtracted | 0.8119 (null 0.7509, sep 0.061) |
| 4 · Theme rank | Spearman ρ | 0.9364 (sep 0.0062) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 1.0 |
| 6 · Argmax agreement | dominant theme | nucleic_purine → nucleic_purine (agree) |
| 7 · Family | ΔPurine share | -0.1033 (0.3068 → 0.2035) |

## Layer 8 — Interpretation
Identity-specific preservation across the hierarchy: both the latent fingerprint and the distinctive biochemical abstraction transfer to silver.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.81 / rank separation +0.006.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).
- Weak serum-matrix recoverability (displacement 0.019): competition on colloid suppresses recovery.

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*