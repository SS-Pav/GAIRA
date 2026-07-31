# tyrosine · Representation-hierarchy assessment (V3)
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.236 |
| 2 · MSS motif | mss cosine | 0.6292 |
| 3 · Theme (raw) | theme cosine raw | 0.9473 |
| 3 · Theme (identity) | baseline-subtracted | 0.1874 (null 0.1409, sep 0.0465) |
| 4 · Theme rank | Spearman ρ | 0.8727 (sep 0.0198) |
| 5 · Top-k overlap | top-2 / top-3 | 1.0 / 0.667 |
| 6 · Argmax agreement | dominant theme | nucleic_purine → nucleic_purine (agree) |
| 7 · Family | ΔPurine share | 0.0205 (0.1802 → 0.2008) |

## Layer 8 — Interpretation
Latent redistribution with retained biochemical abstraction: the 24-coordinate fingerprint is reshaped by adsorption, yet the higher-level theme identity survives.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine +0.19 / rank separation +0.020.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*