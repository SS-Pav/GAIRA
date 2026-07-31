# glucose · Representation-hierarchy assessment (V3)
*Family: saccharide · expected theme: saccharide_glycan*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.2555 |
| 2 · MSS motif | mss cosine | 0.5358 |
| 3 · Theme (raw) | theme cosine raw | 0.8458 |
| 3 · Theme (identity) | baseline-subtracted | -0.4605 (null -0.4806, sep 0.0201) |
| 4 · Theme rank | Spearman ρ | 0.8545 (sep -0.0096) |
| 5 · Top-k overlap | top-2 / top-3 | 1.0 / 0.667 |
| 6 · Argmax agreement | dominant theme | saccharide_glycan → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.024 (0.1717 → 0.1958) |

## Layer 8 — Interpretation
Partial mid-level transfer: motif structure carries over but the identity-specific theme abstraction is weak — a surface-physics limit, not a representation error.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine -0.46 / rank separation -0.010.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).
- Weak serum-matrix recoverability (displacement 0.014): competition on colloid suppresses recovery.

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*