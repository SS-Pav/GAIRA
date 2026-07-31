# cholesterol · Representation-hierarchy assessment (V3)
*Family: lipid · expected theme: lipid_acyl|sterol_membrane*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.3357 |
| 2 · MSS motif | mss cosine | 0.814 |
| 3 · Theme (raw) | theme cosine raw | 0.9181 |
| 3 · Theme (identity) | baseline-subtracted | 0.1114 (null -0.255, sep 0.3664) |
| 4 · Theme rank | Spearman ρ | 0.8818 (sep 0.0904) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.667 |
| 6 · Argmax agreement | dominant theme | lipid_acyl → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.0622 (0.1533 → 0.2155) |

## Layer 8 — Interpretation
Partial mid-level transfer: motif structure carries over but the identity-specific theme abstraction is weak — a surface-physics limit, not a representation error.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.11 / rank separation +0.090.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).
- Weak serum-matrix recoverability (displacement 0.016): competition on colloid suppresses recovery.

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*