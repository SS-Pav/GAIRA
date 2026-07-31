# acetoacetate · Representation-hierarchy assessment (V3)
*Family: organic_acid · expected theme: organic_acid_metabolism*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.423 |
| 2 · MSS motif | mss cosine | 0.7716 |
| 3 · Theme (raw) | theme cosine raw | 0.9169 |
| 3 · Theme (identity) | baseline-subtracted | 0.446 (null 0.4032, sep 0.0428) |
| 4 · Theme rank | Spearman ρ | 0.7636 (sep -0.0867) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.667 |
| 6 · Argmax agreement | dominant theme | organic_acid_metabolism → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.0383 (0.1624 → 0.2007) |

## Layer 8 — Interpretation
Latent redistribution with retained biochemical abstraction: the 24-coordinate fingerprint is reshaped by adsorption, yet the higher-level theme identity survives.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.45 / rank separation -0.087.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*