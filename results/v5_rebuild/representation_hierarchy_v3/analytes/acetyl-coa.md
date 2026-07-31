# acetyl-coa · Representation-hierarchy assessment (V3)
*Family: cofactor · expected theme: sulfur_antioxidant|redox_broad*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.6495 |
| 2 · MSS motif | mss cosine | 0.9006 |
| 3 · Theme (raw) | theme cosine raw | 0.9795 |
| 3 · Theme (identity) | baseline-subtracted | 0.5083 (null 0.5332, sep -0.0249) |
| 4 · Theme rank | Spearman ρ | 0.8727 (sep -0.0124) |
| 5 · Top-k overlap | top-2 / top-3 | 1.0 / 0.667 |
| 6 · Argmax agreement | dominant theme | nucleic_purine → nucleic_purine (agree) |
| 7 · Family | ΔPurine share | -0.0364 (0.2308 → 0.1944) |

## Layer 8 — Interpretation
Identity-specific preservation across the hierarchy: both the latent fingerprint and the distinctive biochemical abstraction transfer to silver.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.51 / rank separation -0.012.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).
- Weak serum-matrix recoverability (displacement 0.016): competition on colloid suppresses recovery.

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*