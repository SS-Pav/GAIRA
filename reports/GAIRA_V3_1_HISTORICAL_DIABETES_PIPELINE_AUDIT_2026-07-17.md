# GAIRA V3.1 — Historical Diabetes Pipeline Audit

**Date:** 2026-07-17 · Read-only audit of the two historical result folders (their directories were NOT modified).

## Sources
- **1304** (earlier): `results/diabetes_gaira_audit_20260701_1304`
- **1322** (later, "better-looking"): `results/diabetes_gaira_audit_20260701_1322`
- Generating code: `analysis/run_diabetes_gaira_audit.py` (+ `analysis/_diabetes_overrides.py`, `analysis/make_publication_figures.py`).

## Side-by-side

| Property | 1304 | 1322 | V3 current |
| --- | --- | --- | --- |
| Input spectra | RawDataImpact/Strong `.mat` → per-patient mean spectra | same `.mat` mean spectra | `pilot2_target_validation_v1/tables/sample_query_spectra.csv` (autoresearch export) |
| BSV recomputed? | yes (loaded .mat, recomputed) | yes | yes (V3 build) |
| Inference engine | `build_report_diabetes` **original G10** (thione 480–510, unconditional ×1.20 boost) — i.e. plain demo engine | `build_report_diabetes` **tightened G10** (thione 490–505, **co-band-gated** ×1.20 boost via 720 cm⁻¹ imidazole) | plain demo `build_report` (original G10) |
| Sample unit | per-patient mean spectrum | per-patient mean | per-sample |
| Cohort defn | `group_2` = direct map Impact→OWD, Strong-D→NWD | same | Impact/Strong-D verbatim |
| n | OWD 39 / NWD 24 (63) | 39 / 24 | 39 / 24 |
| Axes | 11 (G01–G11) | 11 | 11 |
| Normalization | raw radar + (some) z | **cohort z-score** `z=(cohort_mean−pool_mean)/pool_SD`, ddof=1, pooled over 63 | frozen robust global `(raw−median)/MAD` |
| Norm timing | after per-patient BSV | after per-patient BSV (on cohort means) | frozen, applied per sample |
| Center/scale | pooled mean / SD (per axis) | pooled mean / SD | frozen median / MAD (Ag-SERS pop.) |
| Stats | Cohen's d | Cohen's d + Mann–Whitney U + BH q (`group_summary_2group.csv`) | Cohen's d |
| Figures | `fig_radar_2group`, `fig_radar_4subgroup` | `publication_figures_v2/figure_1..4`, forest, heatmap, `supplementary_table_s1` | 11-axis radar (Mode 3) |
| Radial geometry | zero-origin radar | zero-origin radar + normalized radar | zero-origin radar (**clips signed coords** — fixed in V3.1) |

## Key findings
1. **1304 and 1322 differ ONLY on the G10 redox axis** (all 10 other axes identical to 0.00000; G10 max abs 0.060, mean 0.108→0.095). The sole change is the G10 thione-window tightening + co-band gating in `_diabetes_overrides.build_report_diabetes`.
2. The 1322 engine (`build_report_diabetes`) differs from the current demo `build_report` **only on G10** as well (Pearson 1.0000 on all 10 other axes; redox Pearson 0.9999, max abs 0.027 on identical spectra).
3. The "better balanced" 1322 radar is the **cohort z-score normalized** figure (`figure_1_biochemical_state_2group`), reproduced exactly in V3.1 (≤1e-9).
4. Label `group_2` is a **direct** Impact→OWD / Strong-D→NWD relabel (line 205), not the `bmi≥25` rule (which exists as `_map_bmi_group` but is unused for `group_2`).
