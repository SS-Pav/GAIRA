# GAIRA Landscape v3 — Summary

## What Was Fixed

### BSV Bug
**Root cause**: Motif IDs (e.g., `motif_L3_0026`) were compared directly against BSV component subfamilies (e.g., `tryptophan`). No resolution step existed to translate motif IDs to their subfamily names.

**Fix**: Added motif metadata resolution via `phaseO3_batch1_refreshed_motifs.csv`. Now 44/95 motifs map to BSV components.

**Result**: BSV landscape went from **globally zero** to **32/37 conditions with nonzero BSV profiles**. HCC shows protein_backbone=1.0, aromatic_AA=0.75, purine=0.75 — biologically plausible.

### Condition Classification
59 conditions split into:
- 23 biological conditions
- 2 healthy/control
- 4 method/review
- 15 context/task
- 15 unknown/misc

Default view now shows **biological + control only** (25 conditions).

### Trust Labels
Each condition now has an explicit trust label:
- 2 strong (HCC, liver_cancer_unspecified)
- 2 moderate (hepatitis, NAFLD_NASH after filter adjustments)
- 14 weak
- 4 insufficient
- 36 absent

14 conditions flagged as single-source risk.

## Validation

### HCC BSV Profile (post-fix)
| Component | Score |
|---|---|
| protein_backbone | 1.000 |
| aromatic_amino_acid | 0.750 |
| purine_nucleotide | 0.750 |
| pyrimidine_nucleotide | 0.500 |
| glycan_carbohydrate | 0.500 |
| membrane_lipid | 0.500 |

### NAFLD BSV Profile
| Component | Score |
|---|---|
| protein_backbone | 1.000 |
| purine_nucleotide | 1.000 |
| aromatic_amino_acid | 0.667 |
| glycan_carbohydrate | 0.667 |

### Key Difference: NAFLD shows higher purine and glycan relative to protein, while HCC has higher aromatic amino acid signal. This is consistent with metabolic vs oncologic liver disease distinctions.

## Remaining Gaps
1. **51/95 motifs unmapped to BSV** — mostly chemistry_support and unresolved_support (amide, ring, C-C, etc.). These carry chemistry information but lack biological theme assignment.
2. **5 conditions with zero BSV** despite having evidence — their motif profiles are entirely chemistry/unresolved.
3. **Cirrhosis still absent** — the biggest content gap.
4. **14 single-source conditions** — cross-validation needed.

## App
```bash
streamlit run app/gaira_landscape_v3.py
```
Default: biological-only, sources >= 2, evidence >= 5, exclude single-source.
