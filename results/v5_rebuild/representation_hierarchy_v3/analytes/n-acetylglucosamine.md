# n-acetylglucosamine · Representation-hierarchy assessment (V3)
*Family: saccharide · expected theme: saccharide_glycan*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.5987 |
| 2 · MSS motif | mss cosine | 0.8527 |
| 3 · Theme (raw) | theme cosine raw | 0.9014 |
| 3 · Theme (identity) | baseline-subtracted | -0.6277 (null -0.5965, sep -0.0313) |
| 4 · Theme rank | Spearman ρ | 0.9 (sep 0.0164) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.667 |
| 6 · Argmax agreement | dominant theme | saccharide_glycan → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.1051 (0.0981 → 0.2033) |

## Layer 8 — Interpretation
Adsorption-driven observation bias: on silver this analyte is pulled toward the nucleic_purine attractor, so its distinctive Raman abstraction is not recovered even where surface-level structure partially transfers.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine -0.63 / rank separation +0.016.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).
- Weak serum-matrix recoverability (displacement 0.014): competition on colloid suppresses recovery.

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*