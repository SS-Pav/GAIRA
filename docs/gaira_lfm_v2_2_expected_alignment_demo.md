# GAIRA LFM v2.2 — Expected Alignment Demo

## What This Demo Shows

For each spectral dataset, the app now answers the discriminative question:

> Does each observed cohort align more strongly with its own expected literature profile than with alternative expected profiles?

This is the "so what" analysis — not just "high cosine" but "preferential alignment."

## Target Datasets

| Dataset | Cohorts | Expected Comparators |
|---|---|---|
| HCC Holdout (Vornoli 2020, Au SERS) | hcc (72), healthy_control (72) | HCC (direct), healthy_control (direct) |
| CCA / HCC / LM (Lin et al., AgNP SERS) | cca (96), hcc (89), healthy_control (88), lm (81) | cholangiocarcinoma (direct), HCC (direct), healthy_control (direct), liver_cancer_unspecified (approx) |
| Diabetes Plasma EV SERS | BMI>25 (222), BMI≤25 (130) | diabetic_nephropathy (approx), healthy_control (approx) |

## Analysis Pipeline

```
Dataset → Preprocess → Window Features → BSV Projection
                                              ↓
                              Observed Spectral BSV per Cohort
                                              ↓
                              Expected Literature BSV per Cohort
                                              ↓
                              Cross-Similarity Matrix (all pairs)
                                              ↓
                              Alignment Margin Analysis
                                              ↓
                              Scientific Interpretation
```

## Cross-Similarity Matrix

For a dataset with N cohorts and M expected profiles, the app computes an N×M cosine similarity matrix. Each cell = cosine(observed cohort BSV, expected literature BSV).

**Discriminative alignment** means the diagonal is dominant — each cohort aligns best with its own expected profile.

## Alignment Summary

For each cohort:
- **Own cosine**: similarity to its expected comparator
- **Best alternative cosine**: highest similarity to any other expected profile
- **Margin**: own_cosine − best_alt_cosine (positive = preferential alignment)

## Current Results

### HCC Holdout
| Cohort | Own Expected | Own Cosine | Best Alt | Alt Cosine | Margin |
|---|---|---|---|---|---|
| HCC | HCC | +0.889 | healthy_control | +0.855 | +0.034 |
| healthy_control | healthy_control | +0.858 | HCC | +0.891 | −0.033 |

HCC aligns preferentially with HCC literature. Healthy does not — both profiles are close in cosine space, reflecting shared serum biochemistry.

### CCA / HCC / LM
The AgNP substrate produces negative cosines with the literature BSV (which was built from mixed-substrate literature). CCA shows the only positive margin. This confirms the substrate sensitivity findings from earlier spectral query work.

### Diabetes EV
BMI>25 shows positive margin (+0.166) against diabetic_nephropathy, which is the closest available metabolic comparator. BMI≤25 does not align well with healthy_control, likely because the EV literature coverage is sparse.

## Known Limitations

- **Scale mismatch**: Literature BSV values (0-1 support weights) and spectral BSV values (raw normalized intensities ~0.001-0.03) are on different scales. Cosine similarity is scale-invariant for direction but magnitudes differ. Future work should explore rank-based comparison.
- **Substrate sensitivity**: The CCA dataset (AgNP) shows poor alignment with literature profiles (mixed substrate sources). This is expected from v2.1b substrate sensitivity analysis.
- **EV comparators are approximate**: No direct adiposity-specific EV profile exists in the GAIRA corpus.

## How to Run

```bash
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v2_spectral_query.py
```

## What Comes Next

- BSV scale normalization (rank-based or z-score) before cosine comparison
- Substrate-stratified literature BSV profiles
- Per-component alignment analysis (which axes agree, which diverge)
- Statistical confidence intervals via bootstrap
