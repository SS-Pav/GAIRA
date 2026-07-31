# oleate · Representation-hierarchy assessment (V3)
*Family: lipid · expected theme: lipid_acyl|sterol_membrane*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.264 |
| 2 · MSS motif | mss cosine | 0.3648 |
| 3 · Theme (raw) | theme cosine raw | 0.5072 |
| 3 · Theme (identity) | baseline-subtracted | -0.4675 (null -0.4814, sep 0.0139) |
| 4 · Theme rank | Spearman ρ | 0.7 (sep 0.0382) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.333 |
| 6 · Argmax agreement | dominant theme | lipid_acyl → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.1757 (0.0369 → 0.2126) |

## Layer 8 — Interpretation
Adsorption-driven observation bias: on silver this analyte is pulled toward the nucleic_purine attractor, so its distinctive Raman abstraction is not recovered even where surface-level structure partially transfers.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine -0.47 / rank separation +0.038.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*