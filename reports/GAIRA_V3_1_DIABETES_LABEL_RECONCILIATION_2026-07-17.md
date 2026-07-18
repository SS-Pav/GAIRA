# GAIRA V3.1 — Diabetes Label Reconciliation

**Date:** 2026-07-17 · Evidence: `diabetes_label_audit.csv` + `analysis/run_diabetes_gaira_audit.py`.

## Mapping table

| Original label | group_2 | Biological meaning | Metadata/code rule | n (patients) | n (with spectra) | Source | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Impact | OWD | Overweight/obese diabetic (clinical-trial cohort) | `df["group_2"]=df["Group"].map({"Impact":"OWD","Strong-D":"NWD"})` (run_diabetes_gaira_audit.py:205) | 40 | 39 | RawDataImpact.mat / patient_data.csv | **confirmed equivalent** (direct relabel) |
| Strong-D | NWD | Normal-weight diabetic (observational cohort) | same direct map | 24 | 24 | RawDataStrong.mat | **confirmed equivalent** |

Crosstab (proof): Impact → OWD 40 (only), Strong-D → NWD 24 (only). No cross-assignment.

## Important nuance
- OWD/NWD encode the **study-design cohort identity** (Impact = clinical-trial overweight/obese diabetic; Strong-D = observational normal-weight diabetic), per the script docstring.
- A per-patient `bmi≥25 → OWD` function (`_map_bmi_group`, line 176) **exists but is NOT used** to set `group_2`. Some Impact patients have bmi near/below 25 (e.g. 2151-0005 bmi 26.0), so OWD/NWD is a **cohort label, not a per-patient BMI split**.
- **Verdict: Impact ≡ OWD and Strong-D ≡ NWD (confirmed equivalent by direct code map).** The V3.1 UI may use them interchangeably but labels them "OWD (Impact)" / "NWD (Strong-D)" and states they are study-design cohorts.
