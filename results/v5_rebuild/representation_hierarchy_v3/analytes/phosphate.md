# phosphate · Representation-hierarchy assessment (V3)
*Family: organic_acid · expected theme: organic_acid_metabolism*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.361 |
| 2 · MSS motif | mss cosine | 0.6495 |
| 3 · Theme (raw) | theme cosine raw | 0.9183 |
| 3 · Theme (identity) | baseline-subtracted | 0.603 (null 0.54, sep 0.063) |
| 4 · Theme rank | Spearman ρ | 0.9545 (sep 0.0102) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 1.0 |
| 6 · Argmax agreement | dominant theme | organic_acid_metabolism → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.0021 (0.1989 → 0.201) |

## Layer 8 — Interpretation
Latent redistribution with retained biochemical abstraction: the 24-coordinate fingerprint is reshaped by adsorption, yet the higher-level theme identity survives.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.60 / rank separation +0.010.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).
- Weak serum-matrix recoverability (displacement 0.013): competition on colloid suppresses recovery.

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*