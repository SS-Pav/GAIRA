# ascorbate · Representation-hierarchy assessment (V3)
*Family: organic_acid · expected theme: organic_acid_metabolism*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.3917 |
| 2 · MSS motif | mss cosine | 0.674 |
| 3 · Theme (raw) | theme cosine raw | 0.8605 |
| 3 · Theme (identity) | baseline-subtracted | -0.5823 (null -0.5302, sep -0.0521) |
| 4 · Theme rank | Spearman ρ | 0.7636 (sep -0.0211) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.667 |
| 6 · Argmax agreement | dominant theme | lipid_acyl → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.0798 (0.1359 → 0.2157) |

## Layer 8 — Interpretation
Adsorption-driven observation bias: on silver this analyte is pulled toward the nucleic_purine attractor, so its distinctive Raman abstraction is not recovered even where surface-level structure partially transfers.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine -0.58 / rank separation -0.021.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).
- Weak serum-matrix recoverability (displacement 0.015): competition on colloid suppresses recovery.

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*