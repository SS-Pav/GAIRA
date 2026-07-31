# guanine · Representation-hierarchy assessment (V3)
*Family: purine · expected theme: nucleic_purine*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.631 |
| 2 · MSS motif | mss cosine | 0.8745 |
| 3 · Theme (raw) | theme cosine raw | 0.8758 |
| 3 · Theme (identity) | baseline-subtracted | 0.9241 (null 0.7088, sep 0.2153) |
| 4 · Theme rank | Spearman ρ | 0.8727 (sep 0.0584) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.333 |
| 6 · Argmax agreement | dominant theme | nucleic_purine → nucleic_purine (agree) |
| 7 · Family | ΔPurine share | -0.2522 (0.5102 → 0.258) |

## Layer 8 — Interpretation
Identity-specific preservation across the hierarchy: both the latent fingerprint and the distinctive biochemical abstraction transfer to silver.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.92 / rank separation +0.058.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*