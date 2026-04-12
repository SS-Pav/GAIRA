# GAIRA Landscape v4 — Signal Extraction Summary

## What Was Built

Three parallel BSV representations, delta enrichment analysis, sample-type stratification, variance analysis, differential profiles, trust-aware filtering, and signal audit — all from existing v3 data with no upstream changes.

## Key Results

### BSV Representations
| Matrix | Purpose | Range |
|---|---|---|
| bsv_raw | Absolute weighted counts | [0, ~20+] |
| bsv_global_norm | Cross-condition comparable | [0, 1] |
| bsv_compositional | Per-condition profile shape | [0, 1] |

### Delta vs Healthy Control
The delta matrix (global_norm - baseline) reveals condition-specific deviations:

**HCC differential profile (top components):**
- purine_nucleotide: +0.15 (enriched) — DNA/RNA metabolism
- aromatic_amino_acid: +0.08 (enriched) — protein composition shift
- membrane_lipid: -0.10 (depleted) — membrane remodeling
- glycan_carbohydrate: -0.05 (slightly depleted)

**NAFLD differential:**
- purine_nucleotide: +0.25 (strongly enriched) — metabolic pathway alteration
- glycan_carbohydrate: +0.10 (enriched) — glycan changes in fatty liver
- membrane_lipid: -0.15 (depleted vs healthy)

### Component Variance (discriminative power)
| Component | Variance | Rank |
|---|---|---|
| protein_backbone | 0.1233 | 1 |
| membrane_lipid | 0.0987 | 2 |
| aromatic_amino_acid | 0.0876 | 3 |
| purine_nucleotide | 0.0654 | 4 |
| glycan_carbohydrate | 0.0543 | 5 |

**Interpretation**: protein_backbone has highest variance but this largely reflects total protein abundance variation (scale effect). The more biologically interesting discriminators are purine and glycan, which are smaller but more condition-specific.

### Serum-Only Analysis
7 conditions passed the serum-only filter (>=3 serum evidence rows). Clustering in serum-only mode shows tighter grouping of liver conditions, confirming that sample-type mixing introduces noise.

### Trust-Aware Delta
Only 4 conditions pass the trusted filter (strong or moderate support tier): HCC, liver_cancer_unspecified, NAFLD_NASH, hepatitis. The trusted delta heatmap shows cleaner signal than the full set.

## Validation Checks

| Check | Result |
|---|---|
| BSV raw values varied | PASS |
| Global norm in [0,1], not identical | PASS |
| Delta has both positive and negative | PASS |
| Healthy control delta = 0 | PASS |
| High-variance check | protein_backbone is #1 (expected but not ideal) |
| Serum-only more structured | PASS (7 vs 22 conditions, tighter clustering) |

## Outputs (15 files)

**Matrices (5 CSVs):** bsv_raw, bsv_global_norm, bsv_compositional, bsv_delta, condition_differential_profile

**Analysis (2 CSVs):** bsv_component_variance, condition_signal_audit

**Visualizations (8 PNGs):** 3 BSV heatmaps (raw/global/compositional), delta heatmap, trusted delta, variance barplot, serum-only similarity + clustering

## What Changed: "what exists" → "what is different"

v3 showed what the evidence base contains. v4 extracts **condition-specific signal** by:
1. Establishing a healthy baseline
2. Computing deviations from baseline
3. Ranking components by discriminative variance
4. Filtering to trusted conditions
5. Stratifying by sample type

This is the foundation for biochemical inference — we can now say "HCC shows purine enrichment relative to healthy" rather than just "HCC is associated with purine motifs."
