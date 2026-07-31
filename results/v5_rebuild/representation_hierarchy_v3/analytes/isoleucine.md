# isoleucine · Representation-hierarchy assessment (V3)
*Family: amino_acid · expected theme: aromatic_amino_acid|protein_peptide*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.5932 |
| 2 · MSS motif | mss cosine | 0.8301 |
| 3 · Theme (raw) | theme cosine raw | 0.9379 |
| 3 · Theme (identity) | baseline-subtracted | -0.9043 (null -0.8225, sep -0.0818) |
| 4 · Theme rank | Spearman ρ | 0.8727 (sep -0.0071) |
| 5 · Top-k overlap | top-2 / top-3 | 1.0 / 0.667 |
| 6 · Argmax agreement | dominant theme | saccharide_glycan → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.0804 (0.1347 → 0.2152) |

## Layer 8 — Interpretation
Adsorption-driven observation bias: on silver this analyte is pulled toward the nucleic_purine attractor, so its distinctive Raman abstraction is not recovered even where surface-level structure partially transfers.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine -0.90 / rank separation -0.007.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*