# GAIRA LFM v2.3 — Corrected Preprocessing + Raw-Primary Comparator

## Preprocessing Stack

### HCC Holdout (raw CSV)
Full local preprocessing at query time, matching spectral query v1:
1. **AsLS baseline correction** (lambda=1e5, p=0.001, 10 iterations)
2. **Savitzky-Golay smoothing** (window=11, order=3)
3. **L2 vector normalization**

This produces spectral BSV values with real cohort structure (range ~0.010-0.031, vs the previous flat ~0.025-0.030 from the ingestion pipeline).

### CCA / Diabetes EV (NPZ)
Data was preprocessed during ingestion (`v2_crop400_1800_interp1_poly3_vector`). Only L2 normalization is applied at query time.

## What Changed

- **Preprocessing now explicit and dataset-aware**: HCC holdout gets full AsLS + SG + L2; NPZ datasets get L2 only
- **Raw spectral BSV is biologically meaningful again**: cohort radar plots and trust graphs reflect real biochemical structure
- **Expected comparator remains post-hoc only**: shared min-max scaling for display, delta cosine for discriminative analysis

## Section Structure

**Section 1 — Measured Spectral Structure** (raw BSV, no literature influence):
- Preprocessing summary (pipeline type, parameters)
- Mean spectra by cohort
- Raw spectral BSV radar
- BSV heatmap
- Delta-vs-reference heatmap
- Cohort trust graphs (Cohort → Windows → Motifs → Themes → BSV)

**Section 2 — Literature-Expected Comparator** (post-hoc):
- Expected comparator definitions
- Shared min-max overlay radar
- Disease-vs-reference shift comparison (delta cosine)
- Per-axis agreement tables
- Cross-similarity matrix
- Alignment summary + interpretation

## Key Results After Corrected Preprocessing

### HCC Holdout
| Metric | Before (flat) | After (AsLS) |
|---|---|---|
| BSV range | 0.025-0.030 | 0.010-0.031 |
| Top HCC component | redox_metabolite | aromatic_amino_acid |
| HCC delta top axis | nucleic_acid_backbone (-0.010) | purine_nucleotide (-0.004) |
| Delta cosine (HCC shift) | +0.264 (z-score) | +0.083 (raw, honest) |

The raw delta cosine of +0.083 means the observed HCC-vs-healthy spectral shift partially tracks the expected literature shift direction — a modest but real signal without normalization tricks.

### CCA Dataset
Unchanged (already preprocessed in NPZ). Delta cosines remain negative for most cohorts, consistent with substrate sensitivity.

### Diabetes EV
Unchanged (NPZ). Delta cosine for BMI>25 shift: +0.210.

## Trust Graphs
Preserved. Each cohort gets: Cohort → Windows → Motifs → Themes → BSV. Derived from the corrected preprocessing output, so windows/motifs/themes now reflect the real spectral structure.

## How to Run

```bash
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v2_3_spectral_query.py
```
