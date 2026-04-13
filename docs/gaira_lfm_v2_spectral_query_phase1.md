# GAIRA LFM v2 — Spectral Query (Phase 1)

## What v2 Is

A dataset-grounded biochemical composition analysis tool. Select a spectral dataset, inspect its cohort-level BSV structure. All composition is derived from measured spectra, not literature priors.

## What v2 Is NOT

- Not a classifier
- Not a disease prior alignment tool
- Not a transfer scoring engine
- Not a prediction system
- No "HCC prior", no "alignment to literature"

## What "Spectral Query" Means

```
selected dataset
→ spectra (from embedding NPZ)
→ preprocessing (L2 normalization; baseline/smoothing pre-applied)
→ 22 broad-band windows (450-1600 cm⁻¹)
→ motifs (window → molecular feature mapping)
→ themes (motif → biochemical category)
→ BSV composition (8 components)
→ cohort comparison (radar, heatmap, delta, trust graph)
```

## Datasets Currently Supported

Datasets auto-discovered from the embedding NPZ. Those with 2+ named cohorts appear in the selector:

| Dataset | Spectra | Cohorts |
|---|---|---|
| CCA / HCC / LM Serum SERS | 354 | cca (96), hcc (89), healthy_control (88), lm (81) |
| SHINE EV SERS | 366 | 12 day/concentration groups |
| Small 2023 EV | 367 | 11 cell line + concentration groups |
| Diabetes Plasma EV SERS | 352 | impact (222), strong_d (130) |
| + grounding/reference sets | various | metabolites, amino acids |

## Cohort Detection

Cohorts are parsed from the `semantic_groups` array in the NPZ: `class::dataset_id::cohort_name`. The app dynamically detects all cohorts for the selected dataset.

## Pipeline Modules

| Module | Purpose |
|---|---|
| `src/gaira/spectral/dataset_registry.py` | Discovers datasets with cohort structure |
| `src/gaira/spectral/dataset_loader.py` | Loads spectra and labels from NPZ |
| `src/gaira/spectral/preprocessing.py` | L2 vector normalization |
| `src/gaira/spectral/window_panel.py` | 22-window scheme + feature extraction |
| `src/gaira/spectral/bsv_projection.py` | Window → BSV projection + cohort stats + deltas |
| `src/gaira/spectral/plots.py` | Radar, heatmaps, mean spectra overlays |
| `src/gaira/spectral/trust_graph.py` | Per-cohort spectral traversal graphs |

## Outputs

1. **Dataset summary** — spectra count, cohorts, preprocessing
2. **Mean spectra overlay** — one trace per cohort
3. **BSV radar** — overlaid polygons per cohort
4. **BSV heatmap** — rows = cohorts, columns = 8 BSV components
5. **Delta-vs-reference heatmap** — deviation from healthy_control (or user-selected reference)
6. **Cohort traversal graphs** — separate graph per cohort: Cohort → Windows → Motifs → Themes → BSV

## How to Run

```bash
cd /Users/suraj/projects/GAIRA

# v2 spectral query
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v2_spectral_query.py

# v1 text query (still works independently)
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v1_text_query.py
```

## What Is Deliberately Excluded

- Disease priors / HCC prior
- Cross-dataset transfer scoring
- Cosine alignment to literature vectors
- Classifier / prediction framing
- Embedding-based retrieval
- Gemini / LLM dependence

## What Should Come Next (Phase 2)

- Refined-band panel mode (top-N discriminative windows)
- Window importance analysis
- PCA in BSV space
- Inter-cohort statistical testing
- Multi-dataset comparison view
