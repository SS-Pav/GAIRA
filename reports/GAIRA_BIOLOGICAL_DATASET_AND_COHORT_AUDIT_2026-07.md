# GAIRA Biological Dataset & Cohort Audit

**Date:** 2026-07 · Read-only (archives extracted only to audit temp; `.rar` recovered via `bsdtar`). Full tables: `data_audit/biological_dataset_registry.csv`, `biological_cohort_registry.csv`.

## 13 biological datasets — exact counts
| Dataset | Modality | Independent samples | Raw spectra | Nature of the raw count |
| --- | --- | --- | --- | --- |
| diabetes_plasma_ev_sers | SERS | **63** (Impact 39 raw/40 meta, Strong-D 24) | **31,834** | technical map scans (441–1089/patient) |
| shine_ev_sers | SERS | ~41 wells (cell-line) | **23,646** | single-point map scans; D0/D1/D2 (no D3/D7) × C0/10/20/40 |
| small2023_ev | SERS | 5–6 cell lines/fractions | **105,140** (P1 19,557 + P2 85,583) | **AUGMENTED/synthetic mixtures** (identical 14,884 counts) |
| cca_hcc_lm_serum_sers | SERS | **218** (NC 49/CCA 69/HCC 50/LM 50) | 234 (226 bio + 8 dilution) | per-sample averaged; SER-CCA-58 ×2 |
| hcc_serum | SERS | **91** (53 paired CTR/H0T) | 144 | registered but **SKIPPED** in pipeline |
| covid_serum_raman | Raman | ~157 (~103 patients) | 465 (+12 tube) | 3 experimenter-averaged/patient |
| ovarian_plasma_raman_sers | Raman+SERS | **55** | 770 (385+385) | paired modality, ~7 tech/patient |
| ucla_saliva_sev_gc | Raman | **18** (17 GC + 1 healthy) | 2,231 | single-vesicle scans |
| coeliac_faecal_sers | SERS | **27** (CD9/CTR8/GFD10) | 27 (+3 ref) | 1/subject |
| stroke_urine_sers | SERS | **132** | 482 / 210 wdf | paired 532/633 nm |
| mycoplasma_na_sers | SERS | 44 strains (not human) | 521 | ~10/strain |
| single_vesicle_ev_raman | Raman | 6 preps (cell-line) | 525 | single-vesicle |
| LAB_DATA/Cracked_Au (in-house) | SERS | 8 wells | 5,252 (2,500 4-MBA non-bio + 2,752 EV/RNA/Blank) | map scans; **wired nowhere** |

## Required exact accounting
**EV diabetes:** 63 patients with raw spectra (Impact **40 metadata / 39 raw** — one Impact patient has no `.mat` cell; Strong-D 24). Total raw = 31,834 map scans (441 = 21×21 map, 1089 = 33×33); **1 sample/patient (technical map points, not biological replicates)**. Canonical demo cohort: OWD (Impact) 39, NWD (Strong-D) 24 → **63**. Per-patient mean spectra used in V3/V3.1, not the 31,834 scans. **39/24 is canonical** (40 is the metadata count including the patient lacking spectra). OWD=Impact, NWD=Strong-D (proven in the V3.1 label audit).

**Serum liver:** **212 canonical unique patients** (HA 48 / CCA 66-with-mean / HCC 49 / LM 49); `patient_level_bsv.csv` = 213 rows because **SER-CCA-58 appears twice** (69 dirs / 70 txt in raw); `patient_level_mean_spectra.csv` = 212. **SER-LM-11** stored as `SER-LM-11-1_01` in bsv vs `SER-LM-11` in mean (folder-name artifact, breaks the join). Raw independent serum samples = 218 (226 − dilution); after QC → 212–213 used. ~1 averaged spectrum/patient.

**SHINE:** cell-line EV, doses C0/C10/C20/C40 × days **D0, D1, D2 only (NO Day3/Day7)**; ~810–2284 technical map scans per (set,day,dose) cell = **23,646 total**. **No per-sample/per-condition mean spectrum exists** (only individual `s_N` scans) → cannot be recomputed through the 11-axis engine → the demo shows an **8-axis reduced-dimensional** view (3-axis collapse). Reconstruction is not defensible at demo time.

**small2023 EV:** cell-line EV **two-component mixtures**; NormedProbe1 19,557 + NormedProbe2 85,583 = 105,140. The pipeline counts 9,467 (pilot1 cell-line) and 85,583 (pilot1b mixture). The 85,583 (with 5 fractions at identical 14,884) are **augmented/synthetic mixtures, not independent acquisitions**.

## The ">180,000 biological spectra" — reconciliation (the key finding)
- Total raw biological ≈ **168,000–180,000**.
- **Three datasets supply ~89%:** small2023 105,140 (**augmented synthetic**), diabetes 31,834 (**technical map scans**), SHINE 23,646 (**technical scans**). None are independent biological samples.
- **Independent human biological samples across ALL datasets ≈ 760** (diabetes 63, cca 218, hcc 91, covid ~157, ovarian 55, ucla 18, coeliac 27, stroke 132) + a few dozen cell-line/microbial units.
- **>99% of the ~180k are technical scans, single-vesicle map points, experimenter-averaged replicates, or augmented synthetic mixtures.** Treating the spectrum count as biological sample size overstates n by **~200×**.

**Replicate-class separation:** independent patient/sample ≈ 760 · technical map scans ≈ 60k (diabetes + shine + ucla + cracked-au) · augmented/synthetic ≈ 105k (small2023) · experimenter/paired replicates (covid 3×, hcc pairs, ovarian modality, stroke excitation) · processed-duplicate/aggregation artifacts (pilot5 covid 309 spectrum-as-patient; SER-CCA-58) · non-biological controls (covid tube 12, mycoplasma bkg/media 50, cracked-au 4-MBA 2,500 + blank 704, cca dilution 8).
