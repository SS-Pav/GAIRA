# urea · Representation-hierarchy assessment (V3)
*Family: small_nitrogenous · expected theme: organic_acid_metabolism*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.6674 |
| 2 · MSS motif | mss cosine | 0.5222 |
| 3 · Theme (raw) | theme cosine raw | 0.8827 |
| 3 · Theme (identity) | baseline-subtracted | 0.9025 (null -0.0978, sep 1.0003) |
| 4 · Theme rank | Spearman ρ | 0.6455 (sep 0.0364) |
| 5 · Top-k overlap | top-2 / top-3 | 0.0 / 0.333 |
| 6 · Argmax agreement | dominant theme | sulfur_antioxidant → saccharide_glycan (differ) |
| 7 · Family | ΔPurine share | 0.0955 (0.0503 → 0.1459) |

## Layer 8 — Interpretation
Identity-specific preservation across the hierarchy: both the latent fingerprint and the distinctive biochemical abstraction transfer to silver.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.90 / rank separation +0.036.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).
- Weak serum-matrix recoverability (displacement 0.011): competition on colloid suppresses recovery.

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*