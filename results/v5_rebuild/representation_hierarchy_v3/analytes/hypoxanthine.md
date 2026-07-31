# hypoxanthine · Representation-hierarchy assessment (V3)
*Family: purine · expected theme: nucleic_purine*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.8443 |
| 2 · MSS motif | mss cosine | 0.9642 |
| 3 · Theme (raw) | theme cosine raw | 0.9914 |
| 3 · Theme (identity) | baseline-subtracted | 0.8269 (null 0.475, sep 0.3518) |
| 4 · Theme rank | Spearman ρ | 0.9545 (sep 0.0727) |
| 5 · Top-k overlap | top-2 / top-3 | 1.0 / 0.667 |
| 6 · Argmax agreement | dominant theme | nucleic_purine → nucleic_purine (agree) |
| 7 · Family | ΔPurine share | 0.0039 (0.2229 → 0.2268) |

## Layer 8 — Interpretation
Identity-specific preservation across the hierarchy: both the latent fingerprint and the distinctive biochemical abstraction transfer to silver.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.83 / rank separation +0.073.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*