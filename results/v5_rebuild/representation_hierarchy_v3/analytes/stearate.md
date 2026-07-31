# stearate · Representation-hierarchy assessment (V3)
*Family: lipid · expected theme: lipid_acyl|sterol_membrane*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.3209 |
| 2 · MSS motif | mss cosine | 0.3722 |
| 3 · Theme (raw) | theme cosine raw | 0.5662 |
| 3 · Theme (identity) | baseline-subtracted | 0.0776 (null -0.2541, sep 0.3318) |
| 4 · Theme rank | Spearman ρ | 0.8545 (sep 0.0827) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.667 |
| 6 · Argmax agreement | dominant theme | lipid_acyl → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.0692 (0.1328 → 0.202) |

## Layer 8 — Interpretation
Partial mid-level transfer: motif structure carries over but the identity-specific theme abstraction is weak — a surface-physics limit, not a representation error.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.08 / rank separation +0.083.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*