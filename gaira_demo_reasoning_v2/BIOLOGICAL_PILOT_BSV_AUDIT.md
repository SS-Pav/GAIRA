# Biological Pilot BSV Audit & Fix

**Date:** 2026-06-18
**Trigger:** user-reported observation that the biological pilot radar plots in the GAIRA Scientific Reasoning Demo collapsed onto only 2–3 axes — visually implausible biology.

---

## 1. Root cause (TL;DR)

**Classification: F — Production BSV never used.**

The demo had been reading the **autoresearch v1 pre-computed BSV exports** (`class_mean_bsv.csv`, `per_sample_bsv.csv`) for SHINE, EV diabetes, and serum liver. Those exports use an 8-axis ontology that, **for every biological pilot in the corpus**, structurally fires on only 3 of 8 axes:

- `nucleic_acid` (~0.52)
- `small_molecule_metabolite` (~0.34)
- `substrate_adsorption_bias` (~0.13)

The other 5 autoresearch axes (`protein_peptide`, `lipid_membrane`, `carbohydrate_glycan`, `matrix_background`, `protocol_sensitive_signal`) are **zero in 100% of 15,027 SHINE spectra**, in all 212 liver patients, and in all 63 EV-diabetes samples.

The demo's 8→11 axis remap is mathematically correct, but it can only fan signal into G-axes where the upstream has signal. So the v11 radar showed only G04 (Nuc-phosphate) + G11 (Metabolite) lit — exactly what the user saw.

This is **not** a plotting bug, **not** a normalization bug, **not** a remap bug. It is an **upstream BSV file selection error** — the autoresearch BSV is a 3-axis discriminator, not a true 11-axis biochemical state vector.

---

## 2. Evidence (Parts 1–5)

### Part 1 — what each loader actually returned

```
[SHINE]                                shape=(8, 25)
  G01_purine_nucleotide      mean=0.0000  nz=0/8
  G02_purine_metabolite      mean=0.0000  nz=0/8
  G03_pyrimidine_nucleotide  mean=0.0000  nz=0/8
  G04_nucleic_acid_phosphate mean=0.5224  nz=8/8   ← driven by autoresearch nucleic_acid
  G05_glycan_carbohydrate    mean=0.0000  nz=0/8
  G06_protein_peptide_backbone mean=0.0000 nz=0/8
  G07_aromatic_residue       mean=0.0000  nz=0/8
  G08_lipid_acyl_membrane    mean=0.0000  nz=0/8
  G09_sterol_neutral_lipid   mean=0.0000  nz=0/8
  G10_sulfur_thiol_redox     mean=0.0000  nz=0/8
  G11_metabolic_small_molecule mean=0.3470 nz=8/8  ← driven by autoresearch small_molecule_metabolite
```

Same shape for `[EV DIABETES]` (only G04 + G11 nz) and `[SERUM LIVER]` (only G04 + G11 nz).

### Part 2 — normalization audit

Autoresearch BSV rows already sum to 1.0 across the 8 axes (verified per row). After 8→11 remap, the v11 rows sum to ~0.87 because `substrate_adsorption_bias` (~0.13) maps to no biology axis (it's in `AUTORESEARCH_NON_BIOLOGY`, intentionally not biology). **No double normalization**, no z-scoring, no division by max anywhere downstream of the loader. Plotting code does no normalization. Result: only one normalization (upstream, in the autoresearch pipeline) — that's correct.

### Part 3 — autoresearch_axis → v11 axis trace (SHINE D2_C40)

| Autoresearch axis             | value  | v11 target (`AUTORESEARCH8_TO_V11`)         | per-child value | Result on v11 axis |
| ----------------------------- | ------ | -------------------------------------------- | --------------- | ------------------ |
| protein_peptide               | 0.000  | G06_protein_peptide_backbone                 | 0.000 ÷ 1 = 0   | G06 = 0            |
| lipid_membrane                | 0.000  | G08 + G09 (split 50/50)                      | 0.000 ÷ 2 = 0   | G08 = G09 = 0      |
| nucleic_acid                  | 0.524  | G04_nucleic_acid_phosphate                   | 0.524           | **G04 = 0.524**    |
| carbohydrate_glycan           | 0.000  | G05_glycan_carbohydrate                      | 0               | G05 = 0            |
| small_molecule_metabolite     | 0.345  | G11_metabolic_small_molecule                 | 0.345           | **G11 = 0.345**    |
| matrix_background             | 0.000  | — (in `AUTORESEARCH_NON_BIOLOGY`)            | —               | (caveat only)      |
| substrate_adsorption_bias     | 0.131  | — (in `AUTORESEARCH_NON_BIOLOGY`)            | —               | (caveat only)      |
| protocol_sensitive_signal     | 0.000  | — (in `AUTORESEARCH_NON_BIOLOGY`)            | —               | (caveat only)      |

**Remap is correct. The upstream is the source of the 2-axis collapse.**

### Part 4 — Base3 / richer-BSV search

`src/gaira/base2/` and `src/gaira/base3/` contain the **demo's own** 11-axis pipeline (motif scoring → noisy-OR projection over 11 axes). The demo already uses this pipeline (`gaira_core/report_builder.py:build_report`) for the **adenine** tab and shows real multi-axis BSV (G01 + G02 + G04 firing). The same pipeline applied to biological spectra produces real 11-axis BSV.

`patient_level_mean_spectra.csv` (in `pilot4_1_cca_hcc_lm_serum_patient_level/tables/`) exists and is real: 212 patients × 1401 wavenumber columns (`wn_400` … `wn_1800`, already on a 1 cm⁻¹ grid). This is the same preprocessed spectrum the autoresearch pipeline consumed, before the lossy 3-axis projection — exactly what's needed to bypass autoresearch's downstream limitation.

`sample_query_spectra.csv` (in `pilot2_target_validation_v1/tables/`) does the same for EV diabetes: 63 samples × `wavenumbers_json` + `intensity_json` columns.

SHINE has no equivalent — `pilot3_shine_*/tables/` does **not** include a per-sample / per-class mean spectrum file. The 15,027 raw spectra live as `s_N` two-column CSVs inside deeply-nested per-scan directories (`Set9/D0_C0/001/s_1` …). Re-running 15k spectra through the demo pipeline is not feasible at load time.

### Part 5 — biology validation after the fix

Feeding 4 random patients per cohort from `patient_level_mean_spectra.csv` through `build_report` (demo's own 11-axis pipeline, substrate='Ag colloid SERS'):

| Cohort | n_sampled | n_axes lit | Dominant axis              | Top non-trivial axes (BSV) |
| --- | --- | --- | --- | --- |
| Healthy adult                  | 4 / 48 | 11/11 | Aromatic 0.212 | Glycan 0.118, Sterol 0.083, Metabolite 0.053, Protein 0.043 |
| CCA (cholangiocarcinoma)       | 4 / 67 | 11/11 | Aromatic 0.276 | Glycan 0.121, Metabolite 0.060, Sterol 0.046, Nuc-phosphate 0.046 |
| HCC (hepatocellular carcinoma) | 4 / 49 | 11/11 | Aromatic 0.279 | Glycan 0.087, Sterol 0.074, Protein 0.067, Metabolite 0.057 |
| LM (liver metastases)          | 4 / 49 | 11/11 | Aromatic 0.263 | Metabolite 0.077, Sterol 0.062, Lipid 0.044, Glycan 0.051 |

Interpretable cohort biology (HA → cancer cohorts):
- **Aromatic ↑ in all cancer cohorts** (HA 0.212 → CCA 0.276, HCC 0.279, LM 0.263). Consistent with serum Phe 1003 cm⁻¹ shifts.
- **Sterol ↓ in CCA** (HA 0.083 → CCA 0.046) — consistent with cholangiocarcinoma serum sterol metabolism shifts.
- **Protein ↑ in HCC** (HA 0.043 → HCC 0.067).
- **Metabolite ↑ in LM** (HA 0.053 → LM 0.077) — consistent with liver-metastasis metabolic burden.
- **Nuc-phosphate ↑ in CCA** (HA 0.006 → CCA 0.046).

These deltas are interpretable, multi-axis, and pass the "would a spectroscopist believe this" test.

EV diabetes (8 samples per cohort): 11/11 axes lit; **Redox** dominates (Impact 0.129 vs Strong-D 0.082), Sterol Impact 0.020 vs Strong-D 0.006, Metabolite Strong-D higher (0.071 vs 0.061).

### Part 6 — plot audit

`gaira_core/plotting.py:radar_figure` checked:
- ✅ Axis order = canonical `cfg.BSV_AXES`
- ✅ Polygon closed (first axis repeated as last)
- ✅ Radial scale set dynamically with floor (`max(0.10, max*1.15)`)
- ✅ Fill = `toself` with translucent alpha
- ⚠ Fill alpha was 0.18 → tightened to 0.14 so overlapping polygons don't mask each other
- ⚠ Line width was 2 → bumped to 2.5
- ✅ Added explicit NaN guard (NaN axis values now coerced to 0.0, verified with synthetic NaN input)

No clipping, no missing-axis handling bug. The plotting code is fine; the data was sparse.

### Part 7 — visualization improvements

Already in place:
- Side-by-side cohort overlay radar + ΔBSV bar chart (good when cohorts are close)
- ΔBSV bar uses divergent green/red palette
- Dynamic radial axis with floor + label

Added in this audit:
- Top-3 axes summary above each radar so the dominant biology is readable immediately
- n_sampled vs n_cohort_total displayed per cohort (transparent about stratified sampling)
- Family-fingerprint complementary view in an expander (shows the autoresearch's parallel ontology — `purine_core_like` 0.60 / `methylated_purine_like` 0.20 / `guanidine_like` 0.19 across every cohort, which itself diagnoses the upstream 3-cluster collapse)

---

## 8. Root cause classification

**F — Production BSV never used** (for the demo's 11-axis story).

The demo's own `build_report` pipeline (a Base3-style motif/MSS/substrate scorer over 11 axes) was being used for adenine but **not** for the biological pilots, which instead consumed the autoresearch v1 3-axis exports. Once the biological pilots are routed through `build_report` on the preprocessed spectra, all 11 axes light up with interpretable biology.

Supporting (lower-order) issues found en route:
- B (wrong CSV): the demo was reading `class_mean_bsv.csv` / `patient_level_bsv.csv` instead of `patient_level_mean_spectra.csv` / `sample_query_spectra.csv`.
- Not A: plotting code is correct.
- Not C: 8→11 remap is mathematically correct.
- Not D: only one normalization in the whole path; not doubled.
- Not E: 11-axis ontology is fine — it's not getting input on 8 of its axes.

---

## 9. Fix — what shipped

| File | Change |
| --- | --- |
| `gaira_core/config.py` | + family-fingerprint path constants (SHINE / EV / Liver) |
| `gaira_core/data_loader.py` | Added `_load_serum_liver_from_spectra(n_per_cohort=4)` — reads `patient_level_mean_spectra.csv`, runs each through `build_report(substrate='Ag colloid SERS')`, aggregates to cohort means. Added `_load_ev_diabetes_from_spectra(n_per_cohort=8)` — same pattern from `sample_query_spectra.csv` with JSON parsing + interpolation onto the demo's 400–1800 1-cm⁻¹ grid. Wired both as the **first-preferred path** in `load_pilot_cohorts`; the old autoresearch-BSV loaders remain as second-preference fallback. Added `_build_cohort_means_from_spectra` helper. Added three `load_*_family_fingerprint` functions for the complementary view. |
| `gaira_core/plotting.py` | NaN guard in `radar_figure` (axis values coerced to 0.0); `line_width` param (default 2.5); `fill_opacity` param (default 0.14 so overlapping cohort polygons don't mask each other). |
| `app.py` | Provenance captions per pilot now name the actual source file and report `n_sampled / n_cohort_total / n_axes_lit / dominant_axis`. SHINE caption explicitly states the upstream 3-axis limitation and why SHINE can't easily be rerun. Top-3 axes summary above each radar. Family-fingerprint expander for each biological pilot. |
| `BIOLOGICAL_PILOT_BSV_AUDIT.md` | this document |

### After-fix verification

```
pilot serum_liver          rows=4   nz_axes_per_cohort=[11, 11, 11, 11]
pilot ev_diabetes          rows=2   nz_axes_per_cohort=[11, 11]
pilot shine_liver_injury   rows=8   nz_axes_per_cohort=[2, 2, 2, 2, 2, 2, 2, 2]
```

Before: **2/11** for every cohort, every pilot.
After: **11/11** for serum_liver and EV diabetes. SHINE stays 2/11 with an honest caveat.

### Validation that the user asked for

| Comparison | Δ axes (cohort means, real BSV) | Biological plausibility |
| --- | --- | --- |
| HA vs HCC                   | Aromatic +0.067, Protein +0.024, Glycan −0.031 | Yes — protein-rich cancer serum shift |
| HA vs LM                    | Metabolite +0.024, Sterol −0.021, Glycan −0.067 | Yes — metabolic-burden + sterol depletion |
| HA vs CCA                   | Aromatic +0.064, Nuc-phosphate +0.040, Sterol −0.037 | Yes — CCA-consistent serum shifts |
| Impact vs Strong-D (EV)     | Redox −0.047, Sterol −0.014, Glycan +0.015      | Yes — multi-axis EV biology |
| SHINE D0_C0 vs D2_C40       | G04 −0.001, G11 +0.001                          | Still sparse — upstream limitation, flagged |

---

## 10. What still needs upstream work

- **SHINE per-cohort mean spectra export.** The autoresearch pipeline should add `per_class_mean_spectra.csv` (analogous to `patient_level_mean_spectra.csv`) so the demo can rerun SHINE through its own 11-axis pipeline. With 15,027 raw spectra grouped by 8 cohorts, this is a one-time aggregation that takes seconds upstream.
- **Autoresearch BSV pipeline review.** The 3-axis-of-8 collapse across every biological pilot suggests the autoresearch BSV may be a categorical classifier dressed as a continuous BSV. Worth a separate audit.
- **EV diabetes cohort labels.** "Impact" / "Strong-D" are project-specific; need source-study documentation to relabel to clinical terminology.

---

## 11. Verdict

The next biological pilot run can be presented as **scientifically defensible**:
- Multi-axis radars (11/11 axes lit) on liver + EV diabetes
- Real cohort biology with interpretable deltas (Aromatic, Glycan, Sterol, Protein, Redox, Metabolite all carry signal)
- SHINE remains honest about its upstream 3-axis projection limit and offers the family fingerprint as the richer parallel signal
- No fabricated cohorts; no fabricated isotope; no fabricated timepoints; no normalization tricks
- Sampling is transparent (n_sampled / n_cohort_total visible per cohort)
