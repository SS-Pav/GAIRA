# adenine · Representation-hierarchy assessment (V3)
*Family: purine · expected theme: nucleic_purine*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.3592 |
| 2 · MSS motif | mss cosine | 0.7405 |
| 3 · Theme (raw) | theme cosine raw | 0.9466 |
| 3 · Theme (identity) | baseline-subtracted | 0.733 (null 0.7069, sep 0.0261) |
| 4 · Theme rank | Spearman ρ | 0.8636 (sep 0.0331) |
| 5 · Top-k overlap | top-2 / top-3 | 1.0 / 0.667 |
| 6 · Argmax agreement | dominant theme | nucleic_purine → nucleic_purine (agree) |
| 7 · Family | ΔPurine share | -0.1157 (0.3206 → 0.205) |

## Layer 8 — Interpretation
Latent redistribution with retained biochemical abstraction: the 24-coordinate fingerprint is reshaped by adsorption, yet the higher-level theme identity survives. functional perturbation validation — a 14-point concentration series drives the nucleic_purine theme monotonically (ρ=0.996) along a saturating Langmuir law (K=0.89 µM); the biochemical abstraction is dynamically confirmed, not merely static.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.73 / rank separation +0.033.

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*