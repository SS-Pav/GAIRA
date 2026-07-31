# ergothioneine · Representation-hierarchy assessment (V3)
*Family: cofactor · expected theme: sulfur_antioxidant*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.2852 |
| 2 · MSS motif | mss cosine | 0.7263 |
| 3 · Theme (raw) | theme cosine raw | 0.9582 |
| 3 · Theme (identity) | baseline-subtracted | 0.2507 (null 0.3447, sep -0.0941) |
| 4 · Theme rank | Spearman ρ | 0.8545 (sep -0.0895) |
| 5 · Top-k overlap | top-2 / top-3 | 1.0 / 0.667 |
| 6 · Argmax agreement | dominant theme | nucleic_purine → nucleic_purine (agree) |
| 7 · Family | ΔPurine share | -0.0252 (0.1709 → 0.1456) |

## Layer 8 — Interpretation
Partial mid-level transfer: motif structure carries over but the identity-specific theme abstraction is weak — a surface-physics limit, not a representation error. functional perturbation validation — the sulfur_antioxidant theme rises monotonically and saturates with dose (ρ=0.927, K=1.52 µM).

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.25 / rank separation -0.089.

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*