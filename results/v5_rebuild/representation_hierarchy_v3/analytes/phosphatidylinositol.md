# phosphatidylinositol · Representation-hierarchy assessment (V3)
*Family: lipid · expected theme: lipid_acyl|sterol_membrane*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.8339 |
| 2 · MSS motif | mss cosine | 0.9558 |
| 3 · Theme (raw) | theme cosine raw | 0.9888 |
| 3 · Theme (identity) | baseline-subtracted | 0.4695 (null 0.5101, sep -0.0407) |
| 4 · Theme rank | Spearman ρ | 0.9909 (sep 0.0104) |
| 5 · Top-k overlap | top-2 / top-3 | 1.0 / 1.0 |
| 6 · Argmax agreement | dominant theme | nucleic_purine → nucleic_purine (agree) |
| 7 · Family | ΔPurine share | -0.0208 (0.2151 → 0.1943) |

## Layer 8 — Interpretation
Partial mid-level transfer: motif structure carries over but the identity-specific theme abstraction is weak — a surface-physics limit, not a representation error.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.47 / rank separation +0.010.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).
- Weak serum-matrix recoverability (displacement 0.017): competition on colloid suppresses recovery.

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*