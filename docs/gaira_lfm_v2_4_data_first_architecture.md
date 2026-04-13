# GAIRA Spectral Query v2.4 — Replication-First Build

## Pipeline Replication Audit

### What Matched Original v1/v3

| Component | Original | v2.4 | Match? |
|---|---|---|---|
| HCC holdout AsLS baseline | λ=1e5, p=0.001, 10 iter | λ=1e5, p=0.001, 10 iter | EXACT |
| SG smoothing | window=11, order=3 | window=11, order=3 | EXACT |
| Normalization | L2 vector norm | L2 vector norm | EXACT |
| Window panel | 22 windows, 450-1600 cm⁻¹ | 22 windows, 450-1600 cm⁻¹ | EXACT |
| BSV projection | mean of mapped windows per component | mean of mapped windows per component | EXACT |
| CCA preprocessing | NPZ v2 (poly3 baseline) + L2 | NPZ v2 (poly3 baseline) + L2 | EXACT |
| Cohort means | per-sample BSV → mean | per-sample BSV → mean | EXACT |
| Delta computation | cohort mean − reference mean | cohort mean − reference mean | EXACT |
| Cosine formula | dot(a,b)/(‖a‖·‖b‖) | dot(a,b)/(‖a‖·‖b‖) | EXACT |

### What Was Added (Not in Original)

| Feature | Status | Rationale |
|---|---|---|
| Sample-level similarity distributions | NEW | Was present in v1 outputs but not in v2.x apps |
| PCA in BSV space | NEW | Standard visualization, does not alter BSV |
| Band drivers (effect size) | NEW | Data-driven window ranking, annotation-only |
| Expected BSV comparator | FROM v2.2 | Post-hoc only, does not affect observed BSV |
| Delta shift cosine | FROM v2.3 | Directional shift comparison |

### What Was Removed vs v2.3

| Feature | Removed | Why |
|---|---|---|
| Motif-based trust graphs for spectra | YES | Motifs do not compute observed BSV |
| Motif/theme nodes in spectral traversal | YES | Architecturally misleading |
| Z-score/rank normalization as primary mode | YES | Distorts raw biological signal |
| Shared min-max scaling | YES | Added complexity without clarity |

## Key Results

### HCC Holdout (Au SERS)

| Metric | HCC | Healthy |
|---|---|---|
| Cohort-level margin | **+0.054** | −0.052 |
| Sample-level separation | **+0.054** | −0.052 |
| Delta cosine (shift direction) | **+0.083** | — |

**HCC aligns preferentially with expected HCC** at both cohort and sample level. The observed HCC-vs-healthy spectral shift partially tracks the expected literature shift direction (delta cosine +0.083).

Healthy does NOT align preferentially with expected healthy — it aligns more with expected HCC. This is the honest result: both cohorts share strong serum SERS features that are closer to HCC's literature profile shape.

### CCA Dataset (AgNP SERS)

Only CCA shows positive margin (+0.036). HCC, healthy, and LM do not align preferentially with their expected profiles. Substrate sensitivity (AgNP vs mixed-substrate literature) is the likely explanation.

### Diabetes EV

BMI>25 shows positive margin (+0.166) and sample-level separation (+0.098). BMI≤25 does not. Comparators are approximate.

## Architecture

```
SECTION 1 — Observed (data-first, no motifs)
  spectra → preprocessing → 22 windows → 8 BSV components
  Outputs: radar, heatmaps, deltas, distributions, PCA, band drivers

SECTION 2 — Expected (literature-grounded)
  GAIRA landscape → condition profiles
  Outputs: comparator summary, provenance

SECTION 3 — Comparison (post-hoc)
  observed vs expected cosine matrix
  sample-level alignment distributions
  delta shift comparison
  per-axis agreement
  alignment summary
```

## How to Run

```bash
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v2_4_spectral_query.py
```
