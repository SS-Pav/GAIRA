# albumin · Representation-hierarchy assessment (V3)
*Family: protein · expected theme: protein_peptide*

| Layer | Metric | Value |
|---|---|---|
| 1 · Latent fingerprint | component cosine | 0.6025 |
| 2 · MSS motif | mss cosine | 0.879 |
| 3 · Theme (raw) | theme cosine raw | 0.9286 |
| 3 · Theme (identity) | baseline-subtracted | -0.1934 (null -0.1117, sep -0.0817) |
| 4 · Theme rank | Spearman ρ | 0.8182 (sep 0.0738) |
| 5 · Top-k overlap | top-2 / top-3 | 0.0 / 0.333 |
| 6 · Argmax agreement | dominant theme | lipid_acyl → nucleic_purine (differ) |
| 7 · Family | ΔPurine share | 0.0757 (0.1352 → 0.2109) |

## Layer 8 — Interpretation
Adsorption-driven observation bias: on silver this analyte is pulled toward the nucleic_purine attractor, so its distinctive Raman abstraction is not recovered even where surface-level structure partially transfers.

## Layer 9 — Limitations
- Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; this analyte's identity-specific signal is cosine -0.19 / rank separation +0.074.
- No dynamic perturbation validation exists for this analyte (Level 4 not measured).

*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*