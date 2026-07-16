# Serum-Liver Provenance Reconciliation (212 / 213 / 214)

**Date:** 2026-07-15
**Trigger:** the full-state audit flagged a count spread across serum-liver tables (212 vs 213, with a "214" cited in an earlier subagent report).
**Data inspected (read-only, not modified):**
`/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot4_1_cca_hcc_lm_serum_patient_level/tables/`

---

## Canonical answer

| Quantity | Canonical value |
| --- | --- |
| **Unique patients** | **212** (CCA 66 · HA 48 · HCC 49 · LM 49) |
| **Patient-level BSV rows** (`patient_level_bsv.csv`) | **213** (212 patients + 1 duplicate measurement row) |
| **Mean spectra** (`patient_level_mean_spectra.csv`) | **212** (one mean spectrum per unique patient) |
| **"214"** | **Does not exist** — no serum-liver table has 214 rows; earlier report was an off-by-one miscount |

---

## Evidence

### Row / unique counts

| File | Rows | Unique raw `sample_id` | Unique patient (suffix-normalized) | Class distribution |
| --- | --- | --- | --- | --- |
| `patient_level_bsv.csv` | 213 | 213 | **212** | CCA 67, HCC 49, LM 49, HA 48 |
| `patient_level_mean_spectra.csv` | 212 | 212 | **212** | CCA 66, HCC 49, LM 49, HA 48 |
| `patient_level_delta_bsv.csv` | 213 | — | — | (mirrors bsv) |
| `patient_level_family.csv` | 212 | — | — | (mirrors mean_spectra) |

No table anywhere in `pilot4_*` has 214 rows (largest is 213).

### Identifier columns
- Both files key on `sample_id`. Class label column: `class_label_display` (CCA / HCC / LM / HA).
- **ID-format inconsistency between files:** the BSV table uses a measurement suffix (`SER-CCA-58_01`, `SER-LM-11-1_01`), while the mean-spectra table uses the bare patient id (`SER-CCA-58`, `SER-LM-11`). A naïve `sample_id` join therefore shows ~193–194 "mismatches" that are purely cosmetic; after normalizing the trailing `_NN` suffix the two files share **212** patients.

### The exact records responsible for the mismatch

1. **The 213 vs 212 difference — one duplicated CCA patient.**
   `patient_level_bsv.csv` contains **two rows** for CCA patient **`SER-CCA-58`**:
   `SER-CCA-58` and `SER-CCA-58_01` (two measurements of the same patient). This is
   the *only* patient with >1 BSV row, so BSV rows = 213 while unique patients = 212,
   and CCA = 67 rows but 66 unique patients. The mean-spectra table collapses this to
   one mean spectrum → CCA 66, total 212.

2. **A cosmetic ID inconsistency (does not change counts) — one LM patient.**
   The same LM patient appears as `SER-LM-11-1_01` in `patient_level_bsv.csv` and as
   `SER-LM-11` in `patient_level_mean_spectra.csv`. Both files still count LM = 49; this
   is an identifier-formatting inconsistency, not a missing or extra patient.

### Interpretation of the discrepancy
- The mismatch is **not** duplicate *patients*, **not** filtering, **not** malformed rows, and **not** different counting *units* in the biological sense. It is:
  - one **duplicated measurement row** (`SER-CCA-58` appears twice in the BSV table), and
  - one **inconsistent identifier string** (`SER-LM-11` vs `SER-LM-11-1_01`).
- Every patient in the mean-spectra file has patient metadata (class label); every patient has a BSV row. There is **no** patient lacking a spectrum and **no** spectrum lacking metadata — the two files describe the same 212-patient cohort at different granularities (BSV keeps a second measurement for one CCA patient).

---

## What v2 should display

| Statement / tab | Value v2 should use |
| --- | --- |
| Serum-liver cohort sizes | **212 unique patients** — HA 48 / CCA 66 / HCC 49 / LM 49 |
| Radar provenance | computed from `patient_level_mean_spectra.csv` (212 rows, one mean per patient) |
| "n=" in caveats | 212 total (not 213) |
| Any "213 rows" mention | qualify as "213 BSV rows = 212 patients + duplicate measurement of `SER-CCA-58`" |
| Any "214" claim | remove — no such table exists |

**Applied in v2:** the serum-liver caveat in `gaira_demo_reasoning_v2/app.py` now states
"212 unique patients — HA 48 / CCA 66 / HCC 49 / LM 49", explains the `SER-CCA-58`
duplicate, and states no table contains 214. **v1 was not modified.** No raw or
processed source data was modified.

> Note: the demo samples only 4 patients per cohort for the live radar, so this
> reconciliation does not change any displayed BSV value — only the provenance/caveat
> text. Confirmed by the numerical regression (serum loader table identical v1↔v2).
