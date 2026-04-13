# GAIRA LFM v2.3 — Normalized Alignment + Per-Axis Breakdown

## What v2.3 Adds Over v2.2

| Feature | v2.2 | v2.3 |
|---|---|---|
| Comparison modes | Raw cosine only | Raw + Z-score + Rank |
| Per-axis breakdown | No | Full per-axis agreement table + bar charts |
| Substrate context | Implicit | Explicit substrate, compatibility flag, note |
| Axis-level alignment info | No | Aligned/partial/divergent/weak per axis |
| Interpretation | Basic margin summary | Axis-aware, substrate-aware, mode-aware |

## Normalization Modes

| Mode | What It Does | When to Use |
|---|---|---|
| Raw | No normalization. Cosine on raw BSV values. | Baseline comparison. Affected by scale mismatch. |
| Z-score | Standardize each profile to zero mean, unit variance. | Compare relative prominence of axes. Best for mixed-scale comparison. |
| Rank | Convert to rank order (1-8). | Compare BSV profile shape only, ignoring all magnitude. |

**Default: Z-score**, which most effectively handles the scale mismatch between spectral BSV (~0.001-0.03) and literature BSV (0-1.0).

## Key Results

### HCC Holdout (Au SERS)

| Mode | HCC margin | healthy margin |
|---|---|---|
| Raw | +0.034 | −0.033 |
| **Z-score** | **+0.268** | −0.228 |
| Rank | +0.114 | −0.080 |

Z-score normalization reveals that HCC spectra strongly align with the expected HCC literature profile (margin +0.268). The improvement from raw (+0.034) to z-score (+0.268) confirms the scale mismatch was obscuring real discriminative structure.

HCC alignment is driven by: membrane_lipid, aromatic_amino_acid.
Divergence on: protein_backbone, pyrimidine_nucleotide.

### CCA / HCC / LM (AgNP SERS)

Under z-score: only CCA shows positive margin (+0.070). HCC, healthy, and LM do not align preferentially. This is consistent with the known substrate sensitivity — AgNP literature coverage is weaker than Au.

### Diabetes EV

BMI>25 shows positive margin under all modes (+0.166 raw, +0.275 z-score). This is notable given the approximate comparator (diabetic_nephropathy). BMI≤25 does not align well with healthy_control, likely reflecting sparse EV literature coverage.

## Per-Axis Agreement

For each cohort × expected pair, each BSV axis is classified:

| Category | Rule (under normalization) |
|---|---|
| aligned | Same direction, delta < 30% of max delta |
| partial | Same direction, delta 30-60% of max |
| divergent | Opposite direction or delta > 60% of max |
| weak | Both values near zero |

## Substrate Context

| Dataset | Substrate | Compatibility | Note |
|---|---|---|---|
| HCC Holdout | Au nanoparticles | favorable | Strong Au-SERS literature coverage |
| CCA / HCC / LM | AgNP colloids | mixed | Some axes substrate-sensitive |
| Diabetes EV | Au (EV SERS) | uncertain | Limited EV-specific literature |

## How to Run

```bash
# v2.3 (new)
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v2_3_spectral_query.py

# v2.2 (preserved)
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v2_spectral_query.py

# v1 text query (preserved)
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v1_text_query.py
```

## What Comes Next

- Bootstrap confidence intervals on cosine similarities
- Substrate-stratified literature BSV profiles (Au-only, AgNP-only)
- Window-level contribution to alignment/divergence
- Multi-dataset comparison view
