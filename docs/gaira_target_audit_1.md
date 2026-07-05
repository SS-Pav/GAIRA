# gaira_target_audit_1 report

Audit-first, no implementation. Scope: HCC holdout — Gurian et al. 2020, Bonifacio group.

---

## 1. Executive verdict

**GO — spectrum-level only.**

The locked GAIRA BSV / ΔBSV pipeline (`gaira.spectral.*`) already supports the HCC holdout end-to-end and was verified running on the live file in **3.8 s** (144 spectra → preprocess → 22 windows → 8-axis BSV → ΔBSV per axis). No scorer or preprocessing changes are needed.

Sample-level aggregation is **not available** because the public release carries exactly one spectrum per sample and no patient identifier. Per-sample and per-spectrum are therefore the **same unit** for this dataset; the correct framing is **spectrum-level distributional analysis against a healthy-control reference cohort**.

Two parallel HCC analysis branches coexist in the repo. Only one (the BSV branch) is aligned with Phase-1 GAIRA doctrine; the other (theme-layer branch) is a separate older analysis and should not be conflated.

---

## 2. Repo / pipeline audit

### 2.1 Two existing HCC analysis branches (verified facts)

**Branch A — BSV / ΔBSV branch (locked, Phase-1 aligned).**
End-to-end wiring is complete and was exercised during this audit.

| Stage | File | Behavior |
|---|---|---|
| Dataset registry | [src/gaira/spectral/dataset_registry.py](src/gaira/spectral/dataset_registry.py) | `TARGET_DATASETS` contains `hcc_holdout_vornoli2020` (display text mis-attributes the paper — see §4.5). |
| Loader | [src/gaira/spectral/dataset_loader.py:32](src/gaira/spectral/dataset_loader.py#L32) `_load_hcc_holdout` | Reads `/Volumes/SSD_Rad/GAIRA_DATA/raw/hcc_serum/data.csv`, filters to master-axis fingerprint window, interpolates to common grid, maps `H0T→hcc`, `CTR→healthy_control`. |
| Preprocess | [src/gaira/spectral/preprocessing.py:53](src/gaira/spectral/preprocessing.py#L53) `preprocess(ds)` | For `hcc_holdout_vornoli2020` specifically: AsLS (λ=1e5, p=0.001, 10 iter) + Savitzky-Golay (win=11, order=3) + L2 norm. Pipeline tag: `raw_asls_sg_l2`. |
| Window panel | [src/gaira/spectral/window_panel.py](src/gaira/spectral/window_panel.py) `extract_window_features` | 22 canonical windows. |
| BSV projection | [src/gaira/spectral/bsv_projection.py:24](src/gaira/spectral/bsv_projection.py#L24) `project_to_bsv` | 22-window feature → 8-axis BSV. |
| Cohort aggregation | [src/gaira/spectral/bsv_projection.py:48](src/gaira/spectral/bsv_projection.py#L48) `compute_cohort_bsvs` | Returns `CohortBSV` per cohort with `mean_bsv`, `std_bsv`, **and the per-spectrum `sample_bsv` matrix preserved**. |
| ΔBSV | [src/gaira/spectral/bsv_projection.py:79](src/gaira/spectral/bsv_projection.py#L79) `compute_deltas` | Cohort-mean Δ vs a reference cohort; returns `{cohort: {component: Δ}}`. |

**Dry-run result** (verified just now, no changes to any file):

```
loaded: id=hcc_holdout_vornoli2020 X=(144, 1401) wn_range=(400.0, 1800.0)
        cohorts={'hcc': 72, 'healthy_control': 72}
preprocessed: raw_asls_sg_l2 (AsLS + SG + L2)
window_features=(144, 22)  bsv_matrix=(144, 8)
Δ(hcc − healthy_control):
  purine_nucleotide     −0.00393
  nucleic_acid_backbone +0.00277
  glycan_carbohydrate   +0.00162
  membrane_lipid        −0.00119
  protein_backbone      −0.00074
  redox_metabolite      −0.00073
  pyrimidine_nucleotide −0.00072
  aromatic_amino_acid   +0.00012
```

Magnitudes are consistent with the ergothioneine titration’s committed-axis threshold (|Δ|>0.005 for "commit"); purine axis is closest to commit.

**Branch B — Theme-layer branch (separate, not Phase-1).**
Different preprocessing and different analysis object. Do not conflate.

| Stage | File | Behavior |
|---|---|---|
| Preprocessing config | [scripts/process_biosample_dataset.py:30](scripts/process_biosample_dataset.py#L30) | `v1_crop430_1730_interp1_minmax`: crop 430–1730 cm⁻¹, 1 cm grid, **baseline_method=none**, minmax norm. |
| Runner | [scripts/run_hcc_holdout_evaluation.py](scripts/run_hcc_holdout_evaluation.py) (957 lines) | Ingests into an isolated DuckDB evaluation DB, runs `biochemical_theme_layer_v3` scoring via `ThemeEvaluationRunner`. |
| Differential calibration | [src/gaira/serum_differential_calibration.py](src/gaira/serum_differential_calibration.py) + [scripts/run_hcc_holdout_calibration.py](scripts/run_hcc_holdout_calibration.py) | Shared-background correction on theme scores (not BSV). |
| Outputs | `/Volumes/SSD_Rad/GAIRA_DATA/processed/hcc_holdout_evaluation/` and `.../hcc_holdout_calibration/` | Theme scores per spectrum (`hcc_holdout_theme_outputs_long.csv`), group summaries, effect sizes, before/after calibration metrics, representative cases. **No BSV / ΔBSV artifacts anywhere.** |

This branch produces theme-level outputs with named positive themes (`lipid_membrane_associated`, `protein_peptide_associated`, `nucleic_acid_purine_associated`, `carbohydrate_glycan_associated`, `oxidative_metabolic_stress_associated`) and caution themes. It is **not** the BSV/ΔBSV pipeline and should not be used for Pilot 1.

### 2.2 Parser / ingest layer

[src/gaira/parsers/biosample/hcc_serum_parser.py](src/gaira/parsers/biosample/hcc_serum_parser.py) — verified source of truth for sample-level metadata:

```python
sample_id       = f"{sample_code:03d}"         # zero-padded per-class
patient_id      = None                         # explicitly no patient mapping
replicate_id    = source_row_id                # == filename stem
class_label     = "CTR" | "H0T"                # binary
substrate_batch = "A" | "B" | "C"
acquisition_date = yyyymmdd
source_file     = "dataset.zip::<filename>"
```

Comment in parser (line 136): “Current GAIRA framing treats this as a binary HCC-versus-control serum SERS dataset on Ag plasmonic paper substrates, **without inventing patient-level mappings beyond the released sample-level metadata**.”

### 2.3 Reusable components for Pilot 1

- `gaira.spectral.dataset_loader.load_dataset("hcc_holdout_vornoli2020")` — already returns a 144-spectrum `SpectralDataset` with cohorts.
- `gaira.spectral.preprocessing.preprocess(ds)` — dataset-aware dispatch; already applies AsLS+SG+L2 for HCC.
- `gaira.spectral.window_panel.extract_window_features` — 22 windows.
- `gaira.spectral.bsv_projection.project_to_bsv` / `compute_cohort_bsvs` / `compute_deltas` — BSV + cohort + Δ.
- `gaira.spectral.band_drivers.compute_per_cohort_window_importance` — per-cohort driver windows.
- `gaira.spectral.expected_bsv.build_expected_comparators` — literature-expected comparator.

### 2.4 Blockers / risks

- **Mis-attribution** in `dataset_registry.py` display text: calls the dataset `Vornoli 2020, Au`. The actual source paper (`R_code.R` header) is **Gurian et al. 2020** (Bonifacio group, Trieste). The parser comment asserts **Ag plasmonic paper** substrate. Substrate chemistry is not declared in `data.csv` itself — must be reconciled against the paper before publishing any substrate-coupled interpretation.
- **One-spectrum-per-sample** release: no technical replicates, no patient IDs. Any "sample-level aggregation" language in reports must be written to acknowledge this.
- **`sample_code` is not globally unique**: 53 `sample_code` values appear in both classes (e.g. code `003` exists in CTR and H0T). The stable per-row unique ID is `biosample_id` (`hcc_serum_<filename_stem>`). Do not group by `sample_code` alone.

---

## 3. Streamlit demo audit

### 3.1 Current `gaira_demo_v3` (the "active demo")

No target-dataset pathway. `grep -r hcc|HCC|cohort|biosample|class_label` in [streamlit_apps/gaira_demo_v3/](streamlit_apps/gaira_demo_v3/) returns only unrelated helper-code matches ("target_x" = annotation coordinate in `pipeline_diagram_figure`). Tabs are:

1. Methods / Pipeline
2. Grounding (pure molecules)
3. Calibration (5 controlled contrasts)
4. Regression / Dose-response (Ergothioneine titration only)

The v3 `regression_registry.csv` explicitly lists other datasets as unsupported. HCC holdout is **not** offered anywhere in v1 / v2 / v3 demos.

### 3.2 Older app family (reusable reference, not to be edited for v3)

[streamlit_apps/gaira_v4/pages/2_🔬_Spectral_Query.py](streamlit_apps/gaira_v4/pages/2_🔬_Spectral_Query.py) is a fully wired Spectral Query page that:

- selects a target dataset (HCC holdout, CCA/HCC/LM, Diabetes plasma EV),
- runs `load_dataset → preprocess → extract_window_features → project_to_bsv → compute_cohort_bsvs → compute_deltas`,
- renders radar overlay, BSV heatmap, Δ heatmap, mean-spectra overlay, PCA in BSV space, per-spectrum cosine-similarity distributions vs expected comparators.

Plots available in [src/gaira/spectral/plots.py](src/gaira/spectral/plots.py):
- `radar_plot` (cohort-mean overlay)
- `bsv_heatmap`, `delta_heatmap`
- `mean_spectra_plot`
- PCA scatter in BSV space (done inline in the page)

### 3.3 What is missing for a scientifically correct target pilot

Relative to `gaira_v4` Spectral Query:

- **Spectrum-level distributional views** are present only as cosine-similarity histograms. A proper per-axis distribution (boxplot / violin of per-spectrum BSV or ΔBSV for each of the 8 axes, split by cohort) is **not surfaced** today.
- **Per-spectrum ΔBSV vs the healthy-control centroid** is not computed (the page only computes cohort-mean Δ).
- **Batch-effect surfacing** (substrate_batch A/B/C, acquisition_date) is not exposed in any view.
- **Sample-level heatmap** (144 × 8 BSV heatmap, with class and batch sidebars) is not implemented.

These are the *only* additions needed to satisfy doctrine — “sample or spectrum first, cohort summary later”.

---

## 4. HCC holdout dataset audit

### 4.1 File inventory (verified)

```
/Volumes/SSD_Rad/GAIRA_DATA/raw/hcc_serum/
  data.csv      1.3 MB   144 rows × (4 metadata + 2047 wavenumber cols)
  dataset.zip   2.7 MB   144 TXT spectra (one per row in data.csv)
  R_code.R      34 KB    original author preprocessing + PCA-LDA
```

Processed:
```
/Volumes/SSD_Rad/GAIRA_DATA/processed/
  hcc_holdout_evaluation/     theme-branch outputs (long/wide theme scores, group summaries, cases)
  hcc_holdout_calibration/    theme-branch serum-differential calibration outputs
# No BSV / ΔBSV artifacts persisted anywhere on disk.
```

### 4.2 Metadata inventory (verified)

Fields in `data.csv` (exact names):

| Field | Type | Range / values |
|---|---|---|
| `acquisition_date` | int → str `yyyymmdd` | 5 dates: 20181010 (n=5), 20181011 (24), 20181012 (16), 20181015 (40), 20181016 (59) |
| `substrate_batch` | str | `A` (n=51), `B` (n=47), `C` (n=46) |
| `class` | str | `CTR` (n=72), `H0T` (n=72) |
| `sample_code` | int | per-class numbering; **not globally unique** |
| `<−307.4 … 3270.7>` | float | 2047 wavenumber columns, Δλ ≈ 2.57 cm⁻¹, **includes negative values** (retain fingerprint region only) |

No `patient_id`, `spot_id`, `operator`, or replicate-count field in the release.

### 4.3 Grouping structure — verified by pandas

```
df.groupby(['class','sample_code']).size().describe()  →  min=1, max=1, mean=1
df.groupby('sample_code')['class'].nunique() > 1       →  53 sample_codes appear in both CTR and H0T
df.duplicated(subset=['class','sample_code','acquisition_date','substrate_batch']).sum()  →  0
```

**Conclusion: exactly one spectrum per (class, sample_code) combination. No technical replicates. `sample_code` alone is not a cross-class identifier.** The only globally unique identifier is `biosample_id` / `source_row_id` (= filename stem).

### 4.4 Preprocessing state of released data

- Spectra are **raw intensity** values on the **native axis** (−307.4 … 3270.7 cm⁻¹).
- No baseline, no smoothing, no normalization applied in the release.
- `R_code.R` shows the author preprocessing (`spc.loess` interpolation onto 400..1800 step 2, `modpolyfit` baseline degree 4, crop 430–1730, row-wise L2), but **those operations are not reflected in the released CSV/TXT**.
- GAIRA's `gaira.spectral.preprocess()` applies AsLS + SG + L2 (pipeline `raw_asls_sg_l2`), which is a defensible and GAIRA-consistent alternative and is already the one tied to the calibration v3 results.

### 4.5 Integrity checks (verified)

- No NaNs in spectral columns.
- No duplicated metadata rows.
- Class balance exact: 72 / 72.
- Substrate-batch × class × date are not orthogonal — example: date `20181010` has 5 rows spread across batches A, B, C and across both classes. A full confound check (χ²(class, batch), χ²(class, date)) has **not** been run; it should be run in Pilot 1 as a batch-effect caveat.
- Paper attribution in `dataset_registry.py` ("Vornoli 2020, Au") **conflicts** with `R_code.R` header ("Gurian et al. 2020"). The parser comment says "Ag plasmonic paper". Substrate chemistry is not stated in any file; must verify against the published paper before interpretation is finalized.

### 4.6 Compatibility with GAIRA

- Can run through the locked pipeline **as-is**. No adapter code needed.
- The loader's CSV→master-axis interpolation is already in place.
- The AsLS+SG+L2 stack is the same one used in `calibration v3` — ΔBSV magnitudes are directly comparable to the calibration-eval thresholds.
- `compute_deltas(cb, reference="healthy_control")` produces cohort-mean ΔBSV. For spectrum-level ΔBSV vs the healthy centroid, `CohortBSV.sample_bsv` (72 × 8 per cohort) minus `cohort_bsvs["healthy_control"].mean_bsv` gives it trivially — **one line, no scorer change**.

### 4.7 Valid analysis unit for Phase 1

- **Primary unit: per-spectrum (= per-sample here, by construction).**
- Sample-level aggregation collapses to a no-op (n=1 per sample).
- Patient-level aggregation is **not possible** from the release.

---

## 5. Pilot 1 recommendation

Given the audit result (only spectrum-level labels, one spectrum per sample):

**Analysis plan: spectrum-level first, cohort summary second.**

1. Load HCC holdout via `load_dataset("hcc_holdout_vornoli2020")` (144 × 1401).
2. Preprocess via `preprocess(ds)` (raw_asls_sg_l2 — unchanged).
3. Compute per-spectrum BSV matrix via `project_to_bsv(extract_window_features(Xn, ds.wavenumbers))`.
4. Healthy-control reference = **centroid of the 72 healthy_control per-spectrum BSV vectors** (mean; median as sensitivity check).
5. **Per-spectrum ΔBSV** = `sample_bsv − healthy_centroid` for all 144 spectra.
6. Deliverables (per GAIRA doctrine):
   - per-spectrum **8-axis BSV heatmap** (144 × 8), sorted by class and batch, with class & batch sidebars;
   - **per-axis ΔBSV distribution** (boxplot or violin) split by class — the primary interpretation object;
   - cohort-mean radar + cohort-mean Δ radar (summary only, not the primary object);
   - **batch-effect sanity panel**: per-axis ΔBSV distribution split by `substrate_batch` within healthy_control, to check that apparent class shifts aren't batch-driven;
   - **R7c / SAEL scoring** of the cohort-mean ΔBSV against the HCC literature-expected comparator (`build_expected_comparators`) — result tagged with `sael_status` exactly as calibration contrasts are;
   - explicit Phase-1 caveat block: "one spectrum per sample, no patient ID, no technical replicates; sample-level biology cannot be inferred from this release."

What Pilot 1 must **not** claim:
- no classifier metrics (AUC, accuracy) — that is the paper's scope, not GAIRA's;
- no molecule-level identity claims;
- no patient-level biology;
- no cohort-mean-only interpretation without spectrum-level distribution underneath.

---

## 6. Minimal implementation plan

Only what is strictly needed to run Pilot 1 cleanly.

1. **New analysis script** `scripts/run_gaira_target_pilot1_hcc_holdout_bsv.py` (≈120 lines), that:
   - calls the existing `gaira.spectral.*` functions (no new preprocessing / scorer code),
   - computes per-spectrum ΔBSV vs the healthy centroid,
   - writes tidy tables to a new output folder `/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_target_pilot1_hcc_holdout_bsv/tables/`:
     - `per_spectrum_bsv.csv` (144 × {biosample_id, class, batch, date, 8 BSV axes}),
     - `per_spectrum_delta_bsv_vs_healthy.csv`,
     - `cohort_mean_bsv.csv`, `cohort_delta_bsv.csv`,
     - `batch_effect_panel.csv` (per-axis distribution stats by batch within CTR),
     - `sael_score_hcc_vs_healthy.csv` (one row; reuse `gaira.calibration.eval_v3` scoring machinery only to the extent of `testable_axes_for` + `summarize_v3` on the cohort-mean Δ — scorer itself unchanged).
2. **Optional thin Streamlit view** (can wait until pilot report is validated): a single-page reusable module derived from [streamlit_apps/gaira_v4/pages/2_🔬_Spectral_Query.py](streamlit_apps/gaira_v4/pages/2_🔬_Spectral_Query.py) showing:
   - the spectrum-level BSV heatmap,
   - per-axis ΔBSV distribution (box / violin),
   - cohort-mean radar + Δ radar (summary),
   - batch-effect panel.
   Do **not** integrate into `gaira_demo_v3` unless the Pilot-1 outputs validate.
3. **One-line fix** in `dataset_registry.py`: update display name to reflect **Gurian et al. 2020** (Trieste), and resolve the Ag-vs-Au substrate statement against the paper.
4. **No changes** to: the scorer, the atlas, the calibration pipeline, the theme-layer branch, or the v3 demo app.

---

## 7. Final go / no-go

**GO — spectrum-level only.**

- The locked GAIRA BSV / ΔBSV pipeline runs cleanly on the HCC holdout today (verified).
- No scorer / atlas / preprocessing changes are required.
- Sample-level aggregation is structurally unavailable in this release — this is a fact of the dataset, not a pipeline limitation.
- Pilot 1 must be written with explicit scope: **distributional spectrum-level interpretation against a healthy-control reference**, with honest caveats about single-spectrum-per-sample and no patient IDs.
