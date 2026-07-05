# GAIRA demo v3 — data audit

**App:** `streamlit_apps/gaira_demo_v3/gaira_demo_v3.py`
**v1 / v2 status:** untouched.
**Date:** 2026-04-20

## Goal of this audit

Before implementing v3 changes, identify what is actually supported end-to-end
so the app stops exposing options that look real but are not.

## 1. Grounding datasets in use

| Dataset | Source | Rows | Role |
|---|---|---|---|
| RamanBioLib (`metadata_db.csv`, `raman_spectra_db.csv`, `raman_peaks_db.csv`) | `/Volumes/SSD_Rad/.../raw/ramanbiolib/` | 202 molecules, 9 families | Pure-molecule reference layer. Preprocessed + normalized upstream. Used for BSV projection. |
| Amino-acid Raman grounding (`aa.xlsx`) | `/Volumes/SSD_Rad/.../raw/amino_acid_raman_grounding/` | 1606 × 21 (wavenumbers × 20 AAs) | Available but **not currently wired** through the GAIRA demo preprocessing + projection pipeline. Surfaced in the layer description only. |

The Streamlit demo reuses v1's `grounding_molecule_*` tables; v3 adds a
layer-summary table rather than re-deriving spectra.

## 2. Literature-linked molecular evidence

- Parsed from `config/spectral_anchor_windows_v1.csv` · `supporting_source_ids`.
- **23 unique sources** back the 64 atlas bands:
  - 18 literature papers (`src_paper_XXXX_phaseB` / `phaseB2`)
  - 5 core references (`src_001` … `src_005`) — RamanBioLib-adjacent anchors
- Derived table: `streamlit_apps/gaira_demo_v3/data/literature_evidence_layer.csv`
  (per source: `kind`, `n_bands_supported`, `n_axes_touched`, `anchor_bands`, `ambiguous_bands`).

## 3. Atlas summary tables available

- Master: `config/spectral_anchor_windows_v1.csv` (66 rows; v1 trimmed/summarized to 64 actionable bands).
- Derived for demo: `streamlit_apps/gaira_demo/data/atlas_explorer.csv` (+ `atlas_axis_coverage.csv`).
- No second source-breakdown table needed — band-level hover already surfaces per-band source counts.

## 4. Calibration condition label mapping

`streamlit_apps/gaira_demo_v3/data/calibration_metadata_v3.csv` maps each
`contrast_id` to:

| Field | Purpose |
|---|---|
| `rich_label` | Human-readable condition name (replaces opaque IDs) |
| `baseline_label`, `perturbed_label` | Explicit comparison framing |
| `analyte`, `matrix`, `substrate` | Surface measurement context |
| `perturbation_type`, `concentration_info` | Explain what the comparison is |
| `behavior_class` | Pass/moderate/inconsistent summary |
| `caveat` | Any honest limitation |

Rich labels used:

| `contrast_id` | `rich_label` |
|---|---|
| `cspp_fig7_hypoxanthine_spike` | Serum baseline vs Hypoxanthine spike |
| `cspp_fig7_ergothioneine_spike` | Serum baseline vs Ergothioneine spike |
| `uricase_sigma_depletion` | Commercial serum — Uricase untreated vs treated |
| `uricase_spiked_hypoxanthine_serum` | Uricase-treated serum — baseline vs Hypoxanthine spike |
| `ergothioneine_titration_top_vs_zero` | Ergothioneine titration — 0.0 µM vs 2.0 µM endpoints |

## 5. Regression-ready datasets — what is actually supported

Supported in v3's regression tab **only if** the dataset is (a) a real ordered
series (≥ 3 levels) and (b) wired through the GAIRA BSV pipeline.

| `dataset_id` | #levels | Supported? | Reason |
|---|---|---|---|
| `ergothioneine_titration` | **11** (0.0 → 2.0 µM, step 0.2 µM; 5 reps each) | **YES** | Real ladder. Preprocessing + BSV projection already in v1's derived tables. |
| `uricase_sigma_depletion` | 2 | NO | Endpoint comparison, not a titration. Stays in Calibration. |
| `cspp_fig7_hypoxanthine_spike` | 2 (Bkg + Hyp) | NO | Verified via metadata: `conc.nunique() == 1` per cohort. Single-level spike. |
| `cspp_fig7_ergothioneine_spike` | 2 (Bkg + Erg) | NO | Same as above. |
| `adenine_sers_ladder` | 6 raw CSVs (1 ng/mL … 10 µM) | NO | Raw CSVs present but not yet wired through GAIRA's preprocessing + BSV pipeline. Candidate for v4. |

Audit script for the conc check:

```python
cspp = pd.read_csv('…/cspp_serum/Figure-7_all-spectra-and-metadata.csv')
cspp.groupby('metabolite')['conc'].nunique()
# → Bkg 1, Erg 1, Hyp 1
```

## 6. Derived tables created for v3

All under `streamlit_apps/gaira_demo_v3/data/`:

| File | Purpose |
|---|---|
| `grounding_layer_summary.csv` | One row per evidence layer (pure_molecule, literature_linked, atlas), with headline metrics, dataset sources, and an honest note. |
| `literature_evidence_layer.csv` | Per-source breakdown: kind, bands supported, axes touched, anchor vs ambiguous counts. |
| `calibration_metadata_v3.csv` | Rich labels + analyte + matrix + substrate + perturbation + concentration + behavior class + caveat. |
| `regression_registry.csv` | Supported/unsupported regression datasets with explicit reasons. |

Rebuild:

```bash
cd /Users/suraj/projects/GAIRA
PYTHONPATH=src .venv/bin/python streamlit_apps/gaira_demo_v3/build_v3_assets.py
```

## 7. What's intentionally still excluded

- Amino-acid dataset ingestion into BSV projection (needs a dedicated loader + validation pass).
- Adenine SERS ladder regression (raw CSVs exist, not wired).
- Any two-point comparisons in the regression tab (belongs in Calibration).
