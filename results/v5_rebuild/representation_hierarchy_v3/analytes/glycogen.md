# glycogen · Representation-hierarchy assessment (V3)
*Family: polysaccharide · expected theme: saccharide_glycan*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.6837 |
| 2 · MSS motif | mss cosine | 0.9023 |
| 3 · Theme (raw) | theme cosine raw | 0.9572 |
| 3 · Theme (identity) | baseline-subtracted | 0.0647 (null 0.0727, sep -0.0081) |
| 4 · Theme rank | Spearman ρ | 0.9545 (sep -0.0011) |
| 5 · Top-k overlap | top-2 / top-3 | 1.0 / 1.0 |
| 6 · Argmax agreement | dominant theme | saccharide_glycan → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.0454 (0.1648 → 0.2102) |

## Layer 8 — Interpretation
Partial mid-level transfer: motif structure carries over but the identity-specific theme abstraction is weak — a surface-physics limit, not a representation error.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.06 / rank separation -0.001.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*