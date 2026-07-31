# urate · Representation-hierarchy assessment (V3)
*Family: purine · expected theme: nucleic_purine*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.644 |
| 2 · MSS motif | mss cosine | 0.7488 |
| 3 · Theme (raw) | theme cosine raw | 0.9127 |
| 3 · Theme (identity) | baseline-subtracted | -0.5662 (null -0.1396, sep -0.4266) |
| 4 · Theme rank | Spearman ρ | 0.8273 (sep -0.0575) |
| 5 · Top-k overlap | top-2 / top-3 | 1.0 / 0.667 |
| 6 · Argmax agreement | dominant theme | saccharide_glycan → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.1368 (0.127 → 0.2638) |

## Layer 8 — Interpretation
Adsorption-driven observation bias: on silver this analyte is pulled toward the nucleic_purine attractor, so its distinctive Raman abstraction is not recovered even where surface-level structure partially transfers. directional perturbation validation — enzymatic (uricase) removal drops the oxopurine-carbonyl motif sharply (Δ=−0.060); validates response DIRECTION at the motif layer, not a dose magnitude.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine -0.57 / rank separation -0.058.

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*