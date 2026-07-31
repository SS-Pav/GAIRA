# GAIRA Foundation Explorer V2 — Cross-Modal Transfer

An interactive walkthrough of GAIRA's **four-level validation framework** for the
Raman → Ag-SERS jump: latent fingerprint preservation, biochemical theme preservation (with
the null control that keeps it honest), perturbation sensitivity, and matrix recoverability.

```bash
streamlit run gaira_foundation_explorer_v2/app.py
```

## What it is (and is not)

This is **additive**. The original **Foundation Explorer** (`gaira_foundation_explorer/`) is
untouched and still runs — it documents the whole frozen model. V2 focuses on one thing the
original compresses into a single cosine: **what actually survives when a spectrum moves from
Raman to silver**, separated into four distinct, non-interchangeable levels.

It reads **only** committed artifacts from
`results/v5_rebuild/pure_ag_sers_theme_preservation/` (tables, the nine audited figures, the
per-analyte cards, and the framework/spec/assessment docs) plus the frozen atlas fingerprint.
Nothing is retrained; every dataset is projected through the fixed basis
`09ed804a40836f4a05a91ba10900cded`, verified at load.

## The 11 pages

| # | Page | What it shows |
|--:|---|---|
| 1 | Overview | the four levels, headline numbers |
| 2 | The Metric Problem | why raw theme cosine (0.92) is a baseline artifact, not preservation |
| 3 | **Cross-Modal Validation ★** | the centerpiece: latent vs theme, naive-vs-honest toggle, quadrants |
| 4 | The Purine Attractor | 50/51 analytes become purine-dominant; what the 35% "match" really means |
| 5 | Theme Redistribution | where each analyte's composition moves on silver |
| 6 | MSS Motif Preservation | the mid-level layer between coordinates and themes |
| 7 | Perturbation Validation | adenine/ergothioneine dose + uricase directional (3 analytes only) |
| 8 | Matrix Recoverability | serum competition linkage |
| 9 | Per-Analyte Cards | drill into any of the 51 analytes, all four levels; "Not tested" where honest |
| 10 | Framework & Methods | the full framework + metric specification |
| 11 | Verdict | the honest assessment |

## The one idea

Transfer is not one number. A single Raman→SERS cosine calls **adenine** a failure (component
0.36); the framework shows adenine is one of GAIRA's best-validated analytes (purine theme
dominant, dose ρ = 0.996, strong in serum). And a naive theme cosine of 0.92 looks like
universal theme survival — until you subtract the shared compositional baseline and find it is
selective. V2 shows both, honestly.

## Requirements

`streamlit`, `plotly`, `pandas`, `numpy`, and the `gaira` package (`pip install -e .`) for the
fingerprint check. No SSD_Rad, no raw data, no recomputation. If the fingerprint check can't
import the engine (e.g. minimal CI), it falls back to the recorded summary — the app still runs.

## Regenerate the underlying artifacts

```bash
python results/v5_rebuild/pure_ag_sers_theme_preservation/code/theme_preservation.py
python results/v5_rebuild/pure_ag_sers_theme_preservation/code/make_cards_and_layers.py
python results/v5_rebuild/pure_ag_sers_theme_preservation/code/make_figures.py
```
