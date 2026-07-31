# phosphoenolpyruvate · Representation-hierarchy assessment (V3)
*Family: organic_acid · expected theme: organic_acid_metabolism*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.6177 |
| 2 · MSS motif | mss cosine | 0.8211 |
| 3 · Theme (raw) | theme cosine raw | 0.9331 |
| 3 · Theme (identity) | baseline-subtracted | 0.8668 (null 0.6833, sep 0.1835) |
| 4 · Theme rank | Spearman ρ | 0.8727 (sep 0.0733) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.667 |
| 6 · Argmax agreement | dominant theme | nucleic_purine → nucleic_purine (agree) |
| 7 · Family | ΔPurine share | -0.0154 (0.2202 → 0.2047) |

## Layer 8 — Interpretation
Identity-specific preservation across the hierarchy: both the latent fingerprint and the distinctive biochemical abstraction transfer to silver.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.87 / rank separation +0.073.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*