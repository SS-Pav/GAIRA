# cysteine · Representation-hierarchy assessment (V3)
*Family: amino_acid · expected theme: sulfur_antioxidant*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.58 |
| 2 · MSS motif | mss cosine | 0.8088 |
| 3 · Theme (raw) | theme cosine raw | 0.9608 |
| 3 · Theme (identity) | baseline-subtracted | 0.6942 (null 0.6889, sep 0.0053) |
| 4 · Theme rank | Spearman ρ | 0.9455 (sep 0.0167) |
| 5 · Top-k overlap | top-2 / top-3 | 1.0 / 1.0 |
| 6 · Argmax agreement | dominant theme | nucleic_purine → nucleic_purine (agree) |
| 7 · Family | ΔPurine share | -0.1001 (0.3034 → 0.2033) |

## Layer 8 — Interpretation
Identity-specific preservation across the hierarchy: both the latent fingerprint and the distinctive biochemical abstraction transfer to silver.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.69 / rank separation +0.017.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*