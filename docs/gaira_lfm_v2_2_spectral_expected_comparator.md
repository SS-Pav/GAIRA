# GAIRA LFM v2.2 — Spectral Query + Expected BSV Comparator

## What v2.2 Adds

A post-hoc literature-expected BSV comparator layer on top of the spectral query pipeline. After computing observed spectral BSV from measured data, each cohort is matched to its closest literature-grounded BSV profile for comparison.

**Critical rule:** The expected comparator does NOT influence the spectral BSV projection. It is applied after the fact.

## Datasets Supported

| Dataset | Cohorts | Notes |
|---|---|---|
| CCA / HCC / LM Serum SERS | cca (96), hcc (89), healthy_control (88), lm (81) | Liver cancer comparison |
| Diabetes Plasma EV SERS | impact → BMI>25 (222), strong_d → BMI≤25 (130) | Metabolic cohorts, relabeled |
| SHINE EV, Small 2023 EV, etc. | Various | No specific expected comparators yet |

## Cohort Semantic Corrections

### Diabetes EV dataset
- `impact` → **BMI > 25** (adiposity cohort, NOT "disease severity")
- `strong_d` → **BMI ≤ 25 / Normal** (NOT "stronger diabetes")

These corrections are displayed in the sidebar and all labels.

## How Expected Comparator Selection Works

For each spectral cohort, the system:
1. Checks dataset-specific mapping first (`_DATASET_COHORT_MAP`)
2. Falls back to generic cohort name mapping
3. Looks up the matched condition in GAIRA's landscape v4 BSV compositional matrix
4. Returns the profile with match type, confidence, and provenance

### Match types
| Type | Meaning |
|---|---|
| direct | Exact condition match in literature corpus |
| approximate | Closest justified approximation (clearly labeled) |
| unavailable | No defensible comparator exists |

### CCA dataset mappings
| Cohort | Comparator | Match | Confidence |
|---|---|---|---|
| hcc | HCC | direct | high |
| healthy_control | healthy_control | direct | high |
| cca | cholangiocarcinoma | direct | high |
| lm | liver_cancer_unspecified | approximate | high |

### Diabetes EV mappings
| Cohort | Comparator | Match | Confidence |
|---|---|---|---|
| impact (BMI>25) | diabetic_nephropathy | approximate | moderate |
| strong_d (BMI≤25) | healthy_control | approximate | moderate |

## Observed vs Expected Analysis

For each cohort with an available comparator:
- **Cosine similarity** between observed spectral BSV and expected literature BSV
- **Delta** (observed − expected) per BSV component
- **Radar overlay** showing both profiles
- **Delta bar chart** showing per-component differences

## What Is Deliberately NOT Done

- Literature BSV does NOT modify spectral BSV computation
- No "HCC prior" is injected into any spectral projection
- No transfer scoring or alignment optimization
- No classifier framing
- Cosine similarity is descriptive, not a score to be maximized

## How to Run

```bash
cd /Users/suraj/projects/GAIRA
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v2_spectral_query.py
```

## What Should Come Next

- Normalize BSV scales before cosine comparison (literature vs spectral are on different scales)
- Add more dataset-specific cohort mappings as new datasets are ingested
- Expand literature coverage for EV-specific expected profiles
- Add statistical confidence intervals on observed BSV
