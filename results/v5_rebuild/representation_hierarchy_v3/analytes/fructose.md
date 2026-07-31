# fructose · Representation-hierarchy assessment (V3)
*Family: saccharide · expected theme: saccharide_glycan*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.3411 |
| 2 · MSS motif | mss cosine | 0.5823 |
| 3 · Theme (raw) | theme cosine raw | 0.7526 |
| 3 · Theme (identity) | baseline-subtracted | -0.6003 (null -0.6484, sep 0.0481) |
| 4 · Theme rank | Spearman ρ | 0.8182 (sep -0.0002) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.667 |
| 6 · Argmax agreement | dominant theme | saccharide_glycan → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.1196 (0.0762 → 0.1958) |

## Layer 8 — Interpretation
Adsorption-driven observation bias: on silver this analyte is pulled toward the nucleic_purine attractor, so its distinctive Raman abstraction is not recovered even where surface-level structure partially transfers.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine -0.60 / rank separation -0.000.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*