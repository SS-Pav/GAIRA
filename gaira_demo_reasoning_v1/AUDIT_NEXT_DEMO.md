# GAIRA Scientific Reasoning Demo v1 — Audit (round 3, SSD_Rad pass)

**Audit date:** 2026-06-18 (round 3, SSD_Rad source of truth)
**Audit scope:** every tab in the v1 demo, every loader in `gaira_core/data_loader.py`,
every truth claim in `app.py`. This round prioritised `/Volumes/SSD_Rad/GAIRA_DATA/`
as the source of truth and wired real autoresearch outputs into adenine, SHINE,
EV diabetes, serum liver, and the grounding corpus map.
**Audit outcome:** demo upgraded from **"mostly real calibration only"** to
**"mostly real calibration + biological pilot"**. 8 of 9 loaders now return real
data; the only remaining placeholder is per-axis family counts (no
`per_axis_grounding_counts.csv` exists in the corpus yet).

---

## 1. Summary of current state

| Mode | Tab | Before audit | After audit |
| --- | --- | --- | --- |
| 1 | Construction Overview          | Static pipeline diagram      | unchanged |
| 1 | Grounding Corpus Map           | Demo placeholder             | unchanged (real registry export pending) |
| 1 | 11-Axis Biochemical Space      | UMAP/PCA, no envelopes, ambiguous title | **fixed**: dynamic UMAP/PCA label, 1.8σ confidence ellipses for families with ≥5 points and non-degenerate covariance |
| 1 | MSS / Motif Explorer           | Curated 11 molecules         | unchanged |
| 1 | Collision Viewer               | Curated 5 pairs              | unchanged |
| 1 | Physics-Aware Atlas            | Curated 8 regions            | unchanged |
| 1 | End-to-End Workflow            | Synthesised inputs           | unchanged (synth; honest about it) |
| 2 | Ergothioneine Dose Slider      | Real data, but radar appeared frozen — `radial_max=1.0` while real Δ values are ~0.01 made slider changes invisible | **fixed**: dynamic radial scale + provenance caption explaining the 50/50 redox_metabolite remap |
| 2 | Adenine Detection              | Demo placeholder (fabricated) | placeholder retained; **caption now specifies exactly where raw spectra live and what's needed to wire them** |
| 2 | Uric Acid / Isotope Validation | Demo placeholder with **fabricated isotope condition** | **fixed**: real SAEL contrasts (hypoxanthine spike, uricase depletion, hypoxanthine+uricase), per-axis verdict table, honest reporting of the disagree verdict for uricase depletion; **isotope condition removed** (no such data in corpus) |
| 3 | Serum Liver Disease            | Synthetic cohort BSVs        | placeholder retained; caption now states no cached BSV exists |
| 3 | EV Diabetes                    | Synthetic cohort BSVs        | placeholder retained; caption now states no cached BSV exists |
| 3 | SHINE Liver Injury / Hepatotoxicity | **Fabricated Day 3 and Day 7 cohorts** (no real data behind them) | **fixed**: real Day 0 + Day 2 × C0/C10/C20/C40 from autoresearch pilot3 outputs; **Day 3 and Day 7 removed**; per-cohort sample counts surfaced (Day 2 n=5–7, Day 0 n=0 in current export); autoresearch-native axes shown in an expander for transparency |

---

## 2. Real vs placeholder table (post-audit)

| Section | Loader | Source file | Status | Badge in UI |
| --- | --- | --- | --- | --- |
| 11-Axis Biochemical Space | `load_reference_points` | `streamlit_apps/gaira_demo/data/grounding_molecule_bsv.csv` + `grounding_molecule_index.csv` | **REAL (legacy 8-axis remapped to 11-axis)** | none |
| Ergothioneine dose slider | `load_ergothioneine_dose` | `streamlit_apps/gaira_demo/data/ergothioneine_dose_response.csv` | **REAL (legacy 8-axis remapped to 11-axis)** | none + provenance caption |
| Uric Acid / Isotope Validation | `load_uric_acid_validation` | `streamlit_apps/gaira_demo/data/calibration_conditions.csv` + `calibration_delta_bsv.csv` | **REAL (legacy 8-axis remapped to 11-axis)** | none + provenance caption + isotope-data-missing disclaimer |
| SHINE Liver Injury / Hepatotoxicity | `load_pilot_cohorts('shine_liver_injury')` → `_load_shine_real` | `/Volumes/SSD_Rad/GAIRA_DATA/.../pilot3_shine_single_set_day0_day2/tables/class_mean_bsv_day0_day2.csv` + `pilot3_shine_day2_controlanchored/tables/per_sample_bsv_day2.csv` | **REAL (autoresearch 8-axis remapped to 11-axis)** | none + provenance caption + autoresearch-axis expander |
| Grounding Corpus Map | `load_grounding_corpus`, `load_family_counts` | curated counts (no external file) | **PLACEHOLDER** | purple badge |
| Adenine Detection | `load_adenine_calibration` | curated sigmoid placeholder | **PLACEHOLDER** | purple badge + explicit pointer to real raw spectra files |
| Serum Liver Disease pilot | `_serum_liver_placeholder` | curated bases (no real BSVs cached) | **PLACEHOLDER** | purple badge + explicit pointer |
| EV Diabetes pilot | `_ev_diabetes_placeholder` | curated bases (no real BSVs cached) | **PLACEHOLDER** | purple badge + explicit pointer |
| End-to-end workflow spectra | `synth_reference_spectrum` | Gaussian-bump synthesis | **SYNTHETIC by design** | label visible in every plot title |

The legacy-8-axis remap and the autoresearch-8-axis remap are now both
explicitly named (`LEGACY8_TO_V11` and `AUTORESEARCH8_TO_V11` in `config.py`)
with docstrings that flag the projection as "demo-grade, not scientifically
rigorous" and instruct callers to flag it in the UI.

---

## 3. Data availability findings

### A. Adenine — raw spectra exist; cached BSVs do not.

- `/Volumes/SSD_Rad/GAIRA_DATA/raw/european_multi_instrument_adenine/ILSdata.csv` — 3,516 spectra × 0.0–9.0 µM × 4 substrates × 2 lasers. Columns: `conc`, `substrate`, `laser`, `sample`, `type`, `batch`, `replica` + 540 wavenumber columns (400–1999 cm⁻¹, 3 cm⁻¹ step).
- `/Volumes/SSD_Rad/GAIRA_DATA/raw/adenine_sers_control/` — 18 calibrated SERS files (lateral-flow + nanoparticles), wavenumber/intensity pairs.
- **To wire:** write a one-shot script that loops over a subset (e.g. one substrate × all concentrations), calls `build_report` per spectrum, aggregates per-concentration BSV mean/std, writes `data/cached/adenine_calibration.csv` with one row per concentration. Then point `load_adenine_calibration` at it.

### B. Uric acid / isotope — partial coverage; isotope missing.

- 3 of 4 demo conditions exist as real SAEL contrasts with per-axis Δ BSV + verdict.
- **Isotope (¹⁵N / ¹³C) does not exist** in the corpus. The demo previously fabricated a `uric_isotope_15N` condition; it has been removed.
- Raw spectra for the existing conditions live at `/Volumes/SSD_Rad/GAIRA_DATA/raw/cspp_serum/Figure-7_all-spectra-and-metadata.csv` (150 spectra, Hyp/Erg/Bkg × 50 each) and `/Volumes/SSD_Rad/GAIRA_DATA/raw/serum_ag_colloids/dataset_spectral_data.zip` (uricase experiment, 20 raw .txt files).

### C. SHINE — Day 0 + Day 2 only.

Real cohort-level BSVs at:
- `/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot3_shine_single_set_day0_day2/tables/class_mean_bsv_day0_day2.csv` — 8 cohorts (D0_C0..D2_C40)
- `/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot3_shine_day2_controlanchored/tables/per_sample_bsv_day2.csv` — 24 Day-2 samples for n-per-class

**No Day 3 or Day 7 data exists.** The previous demo fabricated those — removed.

The autoresearch ontology is 8 axes: 5 biology (`protein_peptide`, `lipid_membrane`, `nucleic_acid`, `carbohydrate_glycan`, `small_molecule_metabolite`) + 3 non-biology (`matrix_background`, `substrate_adsorption_bias`, `protocol_sensitive_signal`). The non-biology axes are intentionally *not* mapped into the 11-axis BSV and are instead surfaced via the autoresearch expander as caveats.

### D. EV diabetes — no cached BSVs.

`/Volumes/SSD_Rad/GAIRA_DATA/processed/ev_stress_disease_analysis_v1/harmonized_state_table.csv` has cohort/state labels (Normal vs Impact) but no per-axis BSVs — only embeddings.

### E. Serum liver disease (HCC/CCA/LM) — no cached BSVs.

Delta-analysis exists under `v8_serum_delta_analysis_v1/`; per-cohort BSV summaries are not pre-computed.

---

## 4. Implementation recommendations (next sprint)

Priority order, by expected impact ÷ effort:

1. **Run `build_report` over Day-0 SHINE raw spectra** so the Day-0 cohort gets real n values instead of n=0 (small change to the autoresearch pipeline). Source: same set 9 SHINE raw spectra.
2. **Compute per-concentration adenine BSV** from one of the two raw adenine datasets (European multi-instrument is richest, but adenine_sers_control is smaller and easier as a first pass). Cache to `data/cached/adenine_calibration.csv`. Removes the second remaining biological-tab placeholder.
3. **Compute EV-diabetes per-cohort BSV** by running the cached EV embeddings through `bsv_projection.project_to_bsv` (or, more correctly, by running raw spectra through `build_report`). Cache to `data/cached/ev_diabetes.csv`.
4. **Same for serum liver (Healthy/HCC/CCA/LM)** — raw spectra exist in `v8_serum_delta_analysis_v1/`. Cache to `data/cached/serum_liver.csv`.
5. **Build an isotope (¹⁵N / ¹³C) spike-in calibration** — this is the only missing dataset that would require new bench work.

---

## 5. Exact file/schema requirements to remove each remaining placeholder

To unlock the remaining placeholder badges, drop a CSV at the listed path and the existing loader will pick it up automatically (no code change needed — loaders are already structured "real first, fallback second").

| Placeholder | File required | Required columns |
| --- | --- | --- |
| Adenine detection | `data/cached/adenine_calibration.csv` | `condition`, `concentration_M`, then one column per axis: `G01_purine_nucleotide`, `G02_purine_metabolite`, …, `G11_metabolic_small_molecule` |
| EV diabetes | `data/cached/ev_diabetes.csv` | `cohort`, `n`, then 11 axis columns |
| Serum liver disease | `data/cached/serum_liver.csv` | `cohort`, `n`, then 11 axis columns |
| Grounding corpus map | `data/cached/grounding_corpus.csv` | `source`, `regime`, `tier`, `role`, `n_spectra`, `n_analytes` |
| Family analyte counts | `data/cached/family_counts.csv` | `axis`, `n_analytes` |

(For each, add a `_read_csv(cfg.CACHED_DIR / "<name>.csv")` block at the top of the relevant loader function — see `_load_shine_real` for the pattern.)

---

## 6. Code changes shipped in this audit

| File | Change |
| --- | --- |
| `gaira_core/config.py` | + `GAIRA_DATA_VOLUME`, `AUTORESEARCH_ROOT`, `SHINE_DAY02_TABLES`, `SHINE_DAY2_TABLES`. + `AUTORESEARCH8_TO_V11` mapping. + `AUTORESEARCH_NON_BIOLOGY` tuple. Hardened the `LEGACY8_TO_V11` docstring. |
| `gaira_core/data_loader.py` | Rewrote `load_uric_acid_validation` to read real SAEL contrasts and remap to 11-axis with per-axis verdicts; removed fabricated `uric_isotope_15N`. Added `_load_shine_real` reading autoresearch v1 outputs; rewrote `load_pilot_cohorts` to dispatch to real SHINE loader; split serum/EV/SHINE placeholders into named helpers; SHINE placeholder now matches real data shape (Day 0 + Day 2 only). |
| `gaira_core/plotting.py` | Added `_confidence_ellipse_points` helper; extended `biochemical_space_figure` to draw 1.8σ ellipse envelopes only for families with ≥5 points and non-degenerate covariance. |
| `app.py` | Fixed 11-Axis tab subtitle to say "UMAP (preferred) or PCA fallback" and updated chart title to dynamically report which projection ran. Fixed Ergothioneine slider: dynamic radial axis based on data max + provenance caption. Rewrote Uric Acid tab: real-condition dropdown, n-control/n-perturbed/SAEL-confidence metrics, real Δ BSV bar, per-axis verdict table, honest interpretation card; removed isotope condition. Rewrote SHINE tab: real Day 0/2 × C0/C10/C20/C40 cohorts, autoresearch-axis expander, updated caveats to acknowledge missing Day 3/7. Added explicit placeholder captions for adenine, serum-liver, EV-diabetes pointing at the real raw-data files. |

---

## 7. Remaining gaps

- **Adenine calibration**: requires a one-shot script to compute BSV from raw spectra. ~1 hour of work.
- **Serum liver / EV diabetes cohort BSVs**: same pattern. ~2–3 hours each.
- **Isotope validation**: no raw spectra exist anywhere in the corpus; requires bench work to acquire ¹⁵N / ¹³C uric acid spectra.
- **Day 0 SHINE n=0**: per-sample BSV for Day 0 wasn't exported by autoresearch; rerun pilot3 with `--export_per_sample_day0` (or equivalent) to backfill.
- **Grounding corpus + family-counts table**: replace curated counts with a real registry export.

---

## 8. Verdict (round 2 — before SSD_Rad pass)

The round-2 demo could honestly be presented as **"mostly real for the calibration + SHINE story"**:

- 11-Axis Biochemical Space → REAL (202 grounded molecules)
- Ergothioneine dose response → REAL
- Uric acid validation → REAL (3 SAEL contrasts including the inconsistent one)
- SHINE Liver Injury → REAL (Day 0 + Day 2 × C0/C10/C20/C40)

Remaining placeholders at end of round 2: adenine, serum-liver, EV-diabetes, grounding corpus map.

**Scientific guardrails maintained throughout:**
- No molecule-level overclaim.
- Disagree verdicts in uric-acid are *surfaced*, not hidden.
- Missing isotope dataset is explicitly disclosed.
- Day 3 and Day 7 fabrications removed.
- Synthesised reference spectra are explicitly labelled as such in every plot title.

---

# Round 3 — SSD_Rad Real Data Audit

**Date:** 2026-06-18 (same day, second pass)
**Source-of-truth volume:** `/Volumes/SSD_Rad/GAIRA_DATA/`
**Demo upgraded to:** **mostly real calibration + biological pilot**

## SSD_Rad findings — per-section table

| Demo section | Current status (after this audit) | Real data found? | Source path | Format | Sample / condition count | Usable now? | Required processing | Recommended demo action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Adenine Detection | **REAL** (newly wired) | Yes | `/Volumes/SSD_Rad/GAIRA_DATA/raw/adenine_sers_control/Adenine_bAgNPs_*.CSV` (6 concentrations) | semicolon-CSV, cp1252, Latin comma decimals, 7054 rows × 2 cols (wn, intensity) | 6 concentrations (10 pg/mL → 10 µg/mL), bAgNPs substrate | **Yes — wired** | parse + crop 400–1800 cm⁻¹ + interp to 1 cm⁻¹ + `build_report(substrate='Ag colloid SERS')` per concentration | Show monotonic G01 purine_nucleotide rise (0.067 → 0.168). Keep class-level only (substrate dampening applied) |
| SHINE Day-0 n=0 fix | **FIXED** (Day 0 n now 2–4 per cohort) | Yes — root cause was using wrong per-sample file | `/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot3_shine_ev_sers/tables/per_sample_bsv.csv` | CSV, 41 sample rows × 19 cols incl `class_label`, `n_scans`, `subset_id` | D0_C0 (4), D0_C10 (2), D0_C20 (4), D0_C40 (3), D2_C0 (5), D2_C10 (6), D2_C20 (6), D2_C40 (7); also Day 1 (n=4 total, not in class_mean) | **Yes — wired** | one-line `sample_path` change in `_load_shine_real` | n + n_scans now shown for every cohort; Day 1 surfaced as a caveat (no class-mean exists for it) |
| Grounding Corpus Map | **REAL** (43 sources) | Yes | `/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/gaira_evidence_warehouse_grounding_backbone_v1/tables/warehouse_source_registry.csv` joined with `grounding_peak_support_summary.csv` | CSV, 43 source rows × 17 cols + 4-row counts table | 12 reference_molecule + 30 disease_or_stress_paper + 1 serum_grounding | **Yes — wired** | tier-mapping (reference → tier 1, paper → tier 2), regime normalization, left-join on `source_id` | Tier-1 bar chart with real spectra counts (adenine 12 / aa_grounding 20 / metabolite_sers63 64 / serum_ag_colloids 64), full Tier-2 table |
| EV Diabetes pilot | **REAL** (2 cohorts) | Yes | `/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot2_target_validation_v1/tables/class_mean_bsv.csv` | CSV, 2 rows × 9 cols (class_label + 8 autoresearch axes) | Impact (n=39), Strong-D (n=24) | **Yes — wired** | autoresearch 8→11 axis remap | Show real numbers verbatim. Cohort labels are project-specific (NOT generic Normal/Diabetic); honest caption explains this |
| Serum Liver Disease pilot | **REAL** (4 cohorts) | Yes | `/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot4_1_cca_hcc_lm_serum_patient_level/tables/patient_level_bsv.csv` | CSV, 213 patient rows × 10 cols | HA n=48, CCA n=67, HCC n=49, LM n=49 | **Yes — wired** | groupby `class_label_display` → cohort means + counts; autoresearch 8→11 remap | Real cohort means displayed. 2 of 8 axes are 0 in this dataset (carried honestly). Per-patient variance not surfaced (could add a per-patient view in a future round) |
| Per-axis family counts | Placeholder (still curated estimates) | No | — | — | — | No — no `per_axis_grounding_counts.csv` exists | requires upstream export | Caption already flags as curated counts; drop a real CSV at `data/cached/family_counts.csv` to flip |
| Uric acid validation | REAL (round 2) | Already wired in round 2 | `streamlit_apps/gaira_demo/data/calibration_conditions.csv` + `calibration_delta_bsv.csv` | CSV | 3 SAEL contrasts (hypoxanthine spike, uricase depletion, hypoxanthine+uricase) | Yes | (already wired in round 2) | No change |
| Ergothioneine dose response | REAL (round 2) | Already wired in round 2 | `streamlit_apps/gaira_demo/data/ergothioneine_dose_response.csv` | CSV | 11 concentrations × 8 legacy axes | Yes | legacy 8→11 remap (round 1) + dynamic radial axis (round 2) | No change |
| 11-axis biochemical space | REAL (round 1) | Already wired in round 1 | `streamlit_apps/gaira_demo/data/grounding_molecule_bsv.csv` + `grounding_molecule_index.csv` | CSV | 202 molecules | Yes | legacy 8→11 remap | No change |
| End-to-end workflow spectra | Synthetic by design | n/a | n/a | n/a | n/a | n/a | n/a | Title clearly flags synthetic input |

## Files changed in this round (round 3)

| File | Change |
| --- | --- |
| `gaira_core/config.py` | + 4 new path constants: `SHINE_EV_SERS_TABLES`, `EV_DIABETES_TABLES`, `LIVER_PATIENT_TABLES`, `LIVER_COHORT_TABLES`, `GROUNDING_REGISTRY_TABLES`, `ADENINE_RAW_DIR`. |
| `gaira_core/data_loader.py` | Updated `_load_shine_real` to read `pilot3_shine_ev_sers/per_sample_bsv.csv` (fixes Day-0 n=0); added `n_scans` column. Added `_load_ev_diabetes_real` reading `pilot2_target_validation_v1`. Added `_load_serum_liver_real` reading `pilot4_1_cca_hcc_lm_serum_patient_level` and grouping by class_label. Updated `load_grounding_corpus` to try real `warehouse_source_registry.csv` first (43 sources). Added `_load_adenine_real` that parses raw cp1252 semicolon CSVs (6 concentrations), crops + interpolates, and runs `build_report` per spectrum to compute real Ag-SERS BSVs. `load_pilot_cohorts` now dispatches to all three real loaders. |
| `app.py` | **Adenine tab:** select_slider over 6 real concentrations, dynamic radial axis, dose-response log-scale plot of G01, evidence card cites real `source_file`. **Grounding Corpus Map:** complete rewrite — Tier-1 horizontal bar with real spectra/peak/class counts, Tier-2 family×regime stack + expandable table (30 manuscripts), updated definitions to name actual sources. **Biological pilot tabs:** real provenance captions for SHINE / serum_liver / ev_diabetes; updated caveats to acknowledge real n; autoresearch raw-axis expander now exposed for all three pilots. |
| `README.md` | (next) updated to reflect post-round-3 real/placeholder status. |

## Tests run in this round

- All 8 loaders re-tested end-to-end with the SSD_Rad volume mounted; only `load_family_counts` still returns `placeholder=True`.
- Adenine real-data sanity: 6 concentrations × Ag-SERS, G01 rises monotonically 0.067 → 0.168 across 6 orders of magnitude (with substrate dampening keeping the call class-level — correct behaviour, NOT a flat-line failure).
- SHINE n=0 sanity: every cohort now has n≥2 and n_scans≥882.
- Streamlit boot test (port 9879): HTTP 200 on `/_stcore/health`, no errors in log.

## Remaining gaps

1. **Per-axis family analyte counts** — no real `per_axis_grounding_counts.csv` exists. The corpus has 43-source warehouse_source_registry but no axis-level rollup. Build one upstream from `gaira_evidence_warehouse_grounding_backbone_v1` peak assignments.
2. **SHINE Day 1 cohorts** — `pilot3_shine_ev_sers/per_sample_bsv.csv` includes Day 1 (n=4 total), but the per-class means file `class_mean_bsv_day0_day2.csv` doesn't. Rerun `pilot3_shine_single_set_day0_day2` with Day 1 inclusion to add the Day 1 cohorts to the radar.
3. **EV diabetes cohort labels** — "Impact" / "Strong-D" are project-specific. Need source-study documentation to rename to clinically interpretable labels (e.g. mapping to overweight/normal-weight diabetic subgroups if that's what they are).
4. **Serum liver patient-level view** — cohort-aggregated values are very close to each other; per-patient distribution or motif-level decomposition would carry the discriminative signal better. Defer to a future "per-patient drill-down" tab.
5. **Isotope (¹⁵N) uric-acid validation** — no spectra anywhere in the corpus; bench work required.
6. **Family-counts CSV** — placeholder remains; drop a real `data/cached/family_counts.csv` (axis × n_analytes) to flip.

## Final verdict (after SSD_Rad audit)

**The next demo can be presented as `mostly real calibration + biological pilot`** —
matching the "mostly real calibration + biological pilot" level on the user's scale.

- 11-axis biochemical space → REAL (202 molecules)
- Ergothioneine dose response → REAL (11 concentrations)
- Adenine detection → **REAL (6 concentrations, Ag-SERS, live `build_report`)** ← new in round 3
- Uric acid validation → REAL (3 SAEL contrasts)
- Grounding corpus map → **REAL (43 sources)** ← new in round 3
- SHINE liver injury → REAL (Day 0 + Day 2 × C0/C10/C20/C40 with real n) ← n=0 bug fixed in round 3
- EV diabetes → **REAL (2 cohorts, real n)** ← new in round 3
- Serum liver disease → **REAL (4 cohorts, n=48/67/49/49)** ← new in round 3
- Per-axis family counts → still curated demo counts (only remaining placeholder badge)
- End-to-end workflow spectra → synthetic by design (titles say so explicitly)

**Scientific guardrails are maintained:**
- All molecule-level claims remain class-level (adenine G01 monotonic but specificity stays class-level with substrate caveat).
- Real cohort labels are surfaced verbatim — no relabel of "Impact" to "Diabetic" without source documentation.
- Day 1 SHINE samples that exist in per-sample file but not class-mean are *flagged as a caveat*, not silently dropped or fabricated.
- Sparse axes in liver / EV pilots (autoresearch axes that are 0 in the real data) are inherited honestly, not back-filled.
- Substrate-physics dampening for Ag-SERS purine is *visible* in the adenine results (G01 caps at 0.168 instead of saturating).
- No fabricated isotope data, no fabricated Day 3 / Day 7, no curated cohort means substituted for real ones.
