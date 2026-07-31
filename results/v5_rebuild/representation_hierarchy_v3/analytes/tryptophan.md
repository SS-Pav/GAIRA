# tryptophan · Representation-hierarchy assessment (V3)
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.3394 |
| 2 · MSS motif | mss cosine | 0.6793 |
| 3 · Theme (raw) | theme cosine raw | 0.9305 |
| 3 · Theme (identity) | baseline-subtracted | 0.2139 (null 0.3667, sep -0.1528) |
| 4 · Theme rank | Spearman ρ | 0.8818 (sep -0.0007) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.667 |
| 6 · Argmax agreement | dominant theme | nucleic_purine → nucleic_purine (agree) |
| 7 · Family | ΔPurine share | 0.0638 (0.1517 → 0.2155) |

## Layer 8 — Interpretation
Partial mid-level transfer: motif structure carries over but the identity-specific theme abstraction is weak — a surface-physics limit, not a representation error.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.21 / rank separation -0.001.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*