# serine · Representation-hierarchy assessment (V3)
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.3536 |
| 2 · MSS motif | mss cosine | 0.6897 |
| 3 · Theme (raw) | theme cosine raw | 0.8496 |
| 3 · Theme (identity) | baseline-subtracted | -0.6139 (null -0.5968, sep -0.0171) |
| 4 · Theme rank | Spearman ρ | 0.8 (sep 0.0184) |
| 5 · Top-k overlap | top-2 / top-3 | 0.5 / 0.333 |
| 6 · Argmax agreement | dominant theme | saccharide_glycan → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.1332 (0.0726 → 0.2058) |

## Layer 8 — Interpretation
Adsorption-driven observation bias: on silver this analyte is pulled toward the nucleic_purine attractor, so its distinctive Raman abstraction is not recovered even where surface-level structure partially transfers.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine -0.61 / rank separation +0.018.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*