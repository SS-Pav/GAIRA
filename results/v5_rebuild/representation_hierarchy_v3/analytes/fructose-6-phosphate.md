# fructose-6-phosphate · Representation-hierarchy assessment (V3)
*Family: saccharide · expected theme: saccharide_glycan*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.3473 |
| 2 · MSS motif | mss cosine | 0.6931 |
| 3 · Theme (raw) | theme cosine raw | 0.8287 |
| 3 · Theme (identity) | baseline-subtracted | -0.7297 (null -0.7217, sep -0.008) |
| 4 · Theme rank | Spearman ρ | 0.8364 (sep 0.0165) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.667 |
| 6 · Argmax agreement | dominant theme | saccharide_glycan → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.1247 (0.0773 → 0.202) |

## Layer 8 — Interpretation
Adsorption-driven observation bias: on silver this analyte is pulled toward the nucleic_purine attractor, so its distinctive Raman abstraction is not recovered even where surface-level structure partially transfers.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine -0.73 / rank separation +0.017.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).
- Weak serum-matrix recoverability (displacement 0.016): competition on colloid suppresses recovery.

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*