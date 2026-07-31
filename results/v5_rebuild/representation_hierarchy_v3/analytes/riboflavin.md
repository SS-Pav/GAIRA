# riboflavin · Representation-hierarchy assessment (V3)
*Family: cofactor · expected theme: redox_broad*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.3394 |
| 2 · MSS motif | mss cosine | 0.6387 |
| 3 · Theme (raw) | theme cosine raw | 0.972 |
| 3 · Theme (identity) | baseline-subtracted | 0.7086 (null 0.6597, sep 0.0489) |
| 4 · Theme rank | Spearman ρ | 0.9182 (sep 0.0115) |
| 5 · Top-k overlap | top-2 / top-3 | 1.0 / 0.667 |
| 6 · Argmax agreement | dominant theme | nucleic_purine → nucleic_purine (agree) |
| 7 · Family | ΔPurine share | -0.0254 (0.2342 → 0.2087) |

## Layer 8 — Interpretation
Latent redistribution with retained biochemical abstraction: the 24-coordinate fingerprint is reshaped by adsorption, yet the higher-level theme identity survives.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.71 / rank separation +0.011.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*