# citrate · Representation-hierarchy assessment (V3)
*Family: organic_acid · expected theme: organic_acid_metabolism*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.6183 |
| 2 · MSS motif | mss cosine | 0.6956 |
| 3 · Theme (raw) | theme cosine raw | 0.8426 |
| 3 · Theme (identity) | baseline-subtracted | 0.8517 (null 0.7608, sep 0.0908) |
| 4 · Theme rank | Spearman ρ | 0.8091 (sep 0.0073) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.667 |
| 6 · Argmax agreement | dominant theme | organic_acid_metabolism → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | -0.0933 (0.2798 → 0.1865) |

## Layer 8 — Interpretation
Identity-specific preservation across the hierarchy: both the latent fingerprint and the distinctive biochemical abstraction transfer to silver.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.85 / rank separation +0.007.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*