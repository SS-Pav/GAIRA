# glutathione · Representation-hierarchy assessment (V3)
*Family: cofactor · expected theme: sulfur_antioxidant*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.7271 |
| 2 · MSS motif | mss cosine | 0.9554 |
| 3 · Theme (raw) | theme cosine raw | 0.9858 |
| 3 · Theme (identity) | baseline-subtracted | 0.6482 (null 0.5213, sep 0.1269) |
| 4 · Theme rank | Spearman ρ | 0.9818 (sep 0.0176) |
| 5 · Top-k overlap | top-2 / top-3 | 1.0 / 1.0 |
| 6 · Argmax agreement | dominant theme | nucleic_purine → nucleic_purine (agree) |
| 7 · Family | ΔPurine share | 0.0419 (0.1809 → 0.2228) |

## Layer 8 — Interpretation
Identity-specific preservation across the hierarchy: both the latent fingerprint and the distinctive biochemical abstraction transfer to silver.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.65 / rank separation +0.018.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*