# triolein · Representation-hierarchy assessment (V3)
*Family: lipid · expected theme: lipid_acyl|sterol_membrane*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.3138 |
| 2 · MSS motif | mss cosine | 0.3694 |
| 3 · Theme (raw) | theme cosine raw | 0.5219 |
| 3 · Theme (identity) | baseline-subtracted | -0.3572 (null -0.4846, sep 0.1274) |
| 4 · Theme rank | Spearman ρ | 0.6364 (sep 0.0262) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.333 |
| 6 · Argmax agreement | dominant theme | lipid_acyl → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.163 (0.0386 → 0.2016) |

## Layer 8 — Interpretation
Adsorption-driven observation bias: on silver this analyte is pulled toward the nucleic_purine attractor, so its distinctive Raman abstraction is not recovered even where surface-level structure partially transfers.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine -0.36 / rank separation +0.026.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).
- Weak serum-matrix recoverability (displacement 0.015): competition on colloid suppresses recovery.

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*