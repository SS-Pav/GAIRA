# Biological Analysis Audit (Parts 7–10)

## Old analysis weaknesses

- Led with a radar and an overlapping PCA that emphasised (weak) separation.
- Absolute composition radars looked nearly identical across cohorts (compositional
  closure) — read as a bug, actually a visualization choice.
- SHINE EV-SERS and small2023 EV were not connected; SSD_Rad data under-used.
- No sample-level heterogeneity view; no paired/longitudinal analysis; no distinction
  between absolute BSV and ΔBSV.

## New analysis (per dataset, standardized)

Reordered hierarchy (radar/PCA demoted to an expander):
`data-quality/OOD → THEME effect sizes → MSS drivers + sample BSV heatmap → component
provenance → sample heterogeneity (distance) → dataset-specific view → summary +
cautious interpretation`.

- **Effect sizes are primary**: per-theme Mann-Whitney U, BH-FDR q, Cliff's delta, and
  2000× bootstrap CIs — a forest plot with **effect size emphasised over p** (so small
  effects at large n are called near-null, not "significant biology").
- **Sample-level BSV heatmap** (z-scored for display only; never for inference).
- **Distance analysis**: between-group vs within-group BSV distance and their **ratio**
  — answers whether the group difference exceeds biological heterogeneity.
- **MSS heatmap** where broad themes suppress structure.

## Normalization strategy

- Canonical BSV is shown in the **frozen reference frame**; never per-sample min-max,
  never re-centred against the same cohort.
- Z-scoring appears **only** in heatmaps and the diabetes "balanced view", each labelled
  visualization-only. Inference always uses the unmodified engine output.

## Effect-size strategy

Non-parametric throughout (small, non-normal cohorts): MWU + Cliff's delta + bootstrap
CIs + BH-FDR. p-values are reported but effect size and the heterogeneity ratio carry
the interpretation.

## Paired / longitudinal analysis (Part 8E / 9B)

**SHINE** uses a paired Day0→Day2 slope plot per dose cohort. It surfaces a **dose×time
interaction** (organic-acid / lipid move oppositely at control vs high dose) that the
pooled contrast (near-null) hides — exactly the structure that must not be buried in an
unpaired PCA. Pairing is cohort-level (cell-culture EV), stated explicitly.

## Overlap / generalization findings (honest)

Heterogeneity ratios (between/within BSV distance):

| dataset | contrast | leading theme (δ) | ratio | reading |
|---|---|---|---|---|
| diabetes | Impact vs Strong-D | nucleic_purine (−0.88) | **1.79** | difference exceeds heterogeneity — robust, patient-level |
| hcc | HCC vs control | saccharide_glycan (+0.46) | 0.62 | moderate, spectrum-level exploratory |
| covid | COVID vs Healthy | protein_peptide (−0.30) | 0.26 | near-null; groups overlap (shown honestly) |
| shine | D2 vs D0 (pooled) | organic_acid (−0.17) | 0.23 | near-null pooled; real structure is paired-by-dose |
| small2023 | c100 vs c00 | nucleic_purine (−1.00) | 2.80 | probe-loading effect — characterization, not biology |

Cross-study centroids sit in one BSV space with PC1 ≈ 95% reflecting **domain / matrix /
modality**, not biology — presented as an overview, not cross-domain equivalence.

## Dataset-specific (Part 9)

- **Diabetes** (9A): canonical absolute BSV + an exploratory standardized *balanced
  view* (redox no longer visually dominates); the canonical BSV is unchanged. Patient
  level; the purine/sulfur contrast is the robust finding within this one cohort.
- **SHINE** (9B): Day0 present and used; paired-by-dose slopes; response direction is
  dose-dependent (heterogeneous), not a single trajectory.
- **small2023** (9C): EV state-space characterization (distribution + heterogeneity),
  no disease classifier forced.
- **HCC** (9D): patient/sample-level absolute + ΔBSV + MSS drivers with adsorption /
  matrix caveats; classification performance is never equated with molecular
  explanation.

## Unsupported prior claims removed / reframed

- No cohort is described with molecule-present / pathway-activated / diagnosis language;
  all interpretation is "consistent with / associated with / within this dataset".
- COVID is reported as a **near-null** result rather than a separation.
- Cross-domain absolute comparisons are avoided without domain framing.

## Language check

Every verdict uses cautious phrasing; significance is never presented as biological
importance; characterization datasets are never presented as disease contrasts.
