# GAIRA Demo Reasoning V3.1 — Completion Report

**Date:** 2026-07-17 · **Branch:** `gaira-v3-1-diabetes-equivalence-and-visualization`
**Scope:** equivalence, visualization, and provenance correction (NOT a new engine, NOT V4 training).

## Nine required answers
1. **Which exact pipeline produced the historical better-looking diabetes radar?**
   `results/diabetes_gaira_audit_20260701_1322` — `analysis/run_diabetes_gaira_audit.py` using `analysis/_diabetes_overrides.build_report_diabetes` on per-patient mean spectra, then the **cohort z-score** figure `publication_figures_v2/figure_1_biochemical_state_2group`.
2. **Normalization, different engine, or both?** **Primarily normalization.** The engine difference (1304→1322, and vs the demo) is confined to the **single G10 redox axis** (Pearson 1.0000 on all 10 other axes, max abs 0.0; redox max abs 0.027). The multiaxis "balance" comes from the axis-wise cohort z-score, reproduced exactly (≤1.6e-15).
3. **Are OWD/NWD identical to Impact/Strong-D?** **Yes — confirmed equivalent.** Direct code map Impact→OWD (n=39/40), Strong-D→NWD (n=24). A `bmi≥25` rule exists but was not used. They encode study-design cohorts, not per-patient BMI.
4. **Were negative V3 coordinates hidden or clipped?** **Yes** — signed robust-z coordinates were plotted on a zero-origin radar (`radial_max` range `[0, …]`), collapsing/clipping negatives. Values were correct; the radar hid their sign.
5. **New default diabetes disease-comparison view?** **Cohort-relative biochemical effect profile** (pooled z-score, exact reproduction of the audited 1322 figure), rendered as a signed diverging plot.
6. **New global-position view?** **GAIRA Frozen Biological Ag-SERS Reference Coordinates v1** (secondary), signed diverging plot; frozen values unchanged from V3.
7. **What happened to the SHINE radar?** The 11-axis radar was **removed** (SHINE spectra not reconstructable); replaced with a **Legacy reduced-dimensional SHINE response** (active-axis heatmap + dose×time trajectory).
8. **Did any scientific coordinate values change?** **No.** Raw BSV byte-identical to V3 (≤1e-9); frozen calibration content hash identical to V3. Only visualization, naming, and provenance changed.
9. **What limitations remain?** Redox engine difference is real (G10-only, not corrected in the demo engine — historical override kept only in the diabetes reproduction path); frozen scale is Ag-SERS-only; dataset identity still a moderate separator (V3 nuisance report); 6/11 axes not independently grounded; historical p-values recomputed on V3.1 spectra differ slightly from the .mat-based originals (exact historical stats bundled and used for the default view).

## Preservation
V1 (16/16 checksums), V2 (clean vs 1674c89), V3 (clean vs 3445dd3) — all unchanged. Historical result folders unmodified. V3.1 created by rsync from V3; 8 engine modules byte-identical to V3.

## Equivalence artifacts (`data/generated/diabetes_equivalence/`)
`path_comparison_per_sample.csv`, `historical_v1_vs_v2_analysis_comparison.csv`, `historical_vs_v3_raw_bsv_per_sample.csv`, `axiswise_correlations.csv`, `group_effect_comparison.csv`, `normalization_variants.csv`, `diabetes_2group_stats_pathA_v31.csv`, `historical_2group_stats_exact.csv`, `equivalence_summary.json`.

## Tests — 31/31 pass (`tests/run_all.py`)
New: historical reproduction (z-score ≤1e-9; sterol top effect), label mapping (Impact=OWD/Strong-D=NWD), normalization equivalence (engine=G10-only; redox rank raw 1 → robust-z 5), signed visualization (negatives visible, symmetric range, zero line), SHINE provenance (no 11-axis radar; collapsed), V3 raw regression (engine byte-identical; raw ≤1e-9; frozen calibration hash unchanged). Plus 16 inherited V3 tests.

## Launch
```bash
cd /Users/surajpg/projects/GAIRA/gaira_demo_reasoning_v3_1
python selfcheck.py && python tests/run_all.py && ./run_demo.sh
```
Rebuild equivalence artifacts (deterministic): `python tools/build_diabetes_equivalence.py`.

## Exact scientific description
GAIRA Reasoning V3.1 preserves the frozen V3 coordinate system but separates absolute reference-space position from within-cohort biochemical effect visualization. It reproduces and audits the historical EV-diabetes normalization (exactly), corrects signed-coordinate visualization, reconciles cohort labels (Impact≡OWD, Strong-D≡NWD) and upstream BSV differences (redox-axis only), and prevents reduced-dimensional SHINE outputs from being misrepresented as an independent 11-axis projection. The definitive finding: the historical diabetes interpretation improved because of **axis-wise cohort z-score normalization**, not a materially different BSV engine — the only engine difference is a single-axis (G10 redox) refinement.
