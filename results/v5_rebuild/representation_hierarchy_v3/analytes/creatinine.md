# creatinine · Representation-hierarchy assessment (V3)
*Family: small_nitrogenous · expected theme: organic_acid_metabolism*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.6877 |
| 2 · MSS motif | mss cosine | 0.6521 |
| 3 · Theme (raw) | theme cosine raw | 0.9659 |
| 3 · Theme (identity) | baseline-subtracted | 0.8747 (null 0.7143, sep 0.1604) |
| 4 · Theme rank | Spearman ρ | 0.9545 (sep 0.0096) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.667 |
| 6 · Argmax agreement | dominant theme | nucleic_purine → nucleic_purine (agree) |
| 7 · Family | ΔPurine share | -0.0662 (0.2676 → 0.2014) |

## Layer 8 — Interpretation
Identity-specific preservation across the hierarchy: both the latent fingerprint and the distinctive biochemical abstraction transfer to silver.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.87 / rank separation +0.010.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).
- Weak serum-matrix recoverability (displacement 0.018): competition on colloid suppresses recovery.

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*