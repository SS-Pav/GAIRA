# uracil · Representation-hierarchy assessment (V3)
*Family: pyrimidine · expected theme: nucleic_pyrimidine*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.0562 |
| 2 · MSS motif | mss cosine | 0.4858 |
| 3 · Theme (raw) | theme cosine raw | 0.45 |
| 3 · Theme (identity) | baseline-subtracted | 0.1591 (null 0.0649, sep 0.0942) |
| 4 · Theme rank | Spearman ρ | 0.6455 (sep -0.0225) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.333 |
| 6 · Argmax agreement | dominant theme | nucleic_pyrimidine → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.1088 (0.0961 → 0.2049) |

## Layer 8 — Interpretation
Partial mid-level transfer: motif structure carries over but the identity-specific theme abstraction is weak — a surface-physics limit, not a representation error.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.16 / rank separation -0.022.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*