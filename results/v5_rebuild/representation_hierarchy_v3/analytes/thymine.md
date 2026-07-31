# thymine · Representation-hierarchy assessment (V3)
*Family: pyrimidine · expected theme: nucleic_pyrimidine*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.2704 |
| 2 · MSS motif | mss cosine | 0.5425 |
| 3 · Theme (raw) | theme cosine raw | 0.7403 |
| 3 · Theme (identity) | baseline-subtracted | 0.5558 (null 0.424, sep 0.1317) |
| 4 · Theme rank | Spearman ρ | 0.2182 (sep -0.0062) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.333 |
| 6 · Argmax agreement | dominant theme | nucleic_purine → nucleic_purine (agree) |
| 7 · Family | ΔPurine share | -0.0852 (0.281 → 0.1959) |

## Layer 8 — Interpretation
Latent redistribution with retained biochemical abstraction: the 24-coordinate fingerprint is reshaped by adsorption, yet the higher-level theme identity survives.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.56 / rank separation -0.006.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).
- Weak serum-matrix recoverability (displacement 0.018): competition on colloid suppresses recovery.

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*