# lactate · Representation-hierarchy assessment (V3)
*Family: organic_acid · expected theme: organic_acid_metabolism*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.402 |
| 2 · MSS motif | mss cosine | 0.5653 |
| 3 · Theme (raw) | theme cosine raw | 0.7821 |
| 3 · Theme (identity) | baseline-subtracted | -0.6338 (null -0.538, sep -0.0958) |
| 4 · Theme rank | Spearman ρ | 0.8364 (sep -0.0056) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.333 |
| 6 · Argmax agreement | dominant theme | saccharide_glycan → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.1214 (0.0858 → 0.2072) |

## Layer 8 — Interpretation
Adsorption-driven observation bias: on silver this analyte is pulled toward the nucleic_purine attractor, so its distinctive Raman abstraction is not recovered even where surface-level structure partially transfers.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine -0.63 / rank separation -0.006.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*