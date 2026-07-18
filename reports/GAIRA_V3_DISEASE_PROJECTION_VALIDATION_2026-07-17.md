# GAIRA V3 — Disease-Dataset Projection Validation

**Date:** 2026-07-17
**Critical rule:** disease labels were **NOT** used to fit the calibration (centers, scales, transforms, ontology). The global coordinates were **frozen before** any label comparison. Labels are used only for the post-hoc effect-size comparisons below. Proof: `tests/test_no_label_leakage.py` (label-free fit reproduces the stored calibration exactly; permuting the label column cannot change the fit).

Effect sizes are Cohen's d on **global coordinates** (per-sample), computed after freezing.

---

## EV diabetes — per-sample projection
- **63 real sample-level spectra** projected individually (not cohort means): Impact n=39, Strong-D n=24 (project-specific labels, kept verbatim — not a generic Normal vs Diabetic split).
- Coordinate systems available in the UI: raw / global / cohort-relative.

**Impact vs Strong-D (global coords), top axes:**

| Axis | Cohen's d |
| --- | --- |
| Redox (G10) | **+2.22** |
| Purine-nuc (G01) | +1.45 |
| Aromatic (G07) | +1.40 |

The previous diabetes analysis used cohort-wise axis standardization to stop the redox axis dominating the radar. V3 reproduces the redox signal **without** cohort-dependent coordinates: the frozen calibration already prevents raw-scale redox dominance (redox global variance rank 2, not 1), yet the Impact/Strong-D redox contrast remains the largest effect (d=+2.22) — a genuine, cohort-invariant biological difference. Agreement between the old cohort-relative view and the new global view on the redox direction is confirmed; the global view additionally makes the value comparable across datasets.

## Serum liver disease — per-patient projection
- **212 canonical unique patients** projected individually (HA 48, CCA 66, HCC 49, LM 49). (The 213th `patient_level_bsv` row is a duplicate measurement of `SER-CCA-58`; the mean-spectra file the projection reads has 212.)
- Substrate/matrix caveat: serum Ag-SERS; class-level only.

**Cancer vs HA (global coords), top axes:**

| Comparison | Top effects (Cohen's d) |
| --- | --- |
| CCA vs HA | Protein −2.03, Sterol −0.89, Glycan −0.86 |
| HCC vs HA | Glycan −0.64, Purine-met +0.53, Aromatic +0.41 |
| LM vs HA | Protein −1.40, Sterol −1.18, Purine-met +0.92 |

Interpretable, multi-axis, cohort-invariant shifts (protein/sterol depletion in CCA/LM serum; purine-metabolite elevation in HCC/LM). These are **exploratory biochemical-state deltas**, not diagnoses; the underlying BSV is a heuristic band-evidence measure.

## SHINE EV-SERS
- Day 0 + Day 2 × C0/C10/C20/C40 (8 cohorts) projected.
- **Distinguished from serum/EV:** SHINE is projected from the **legacy cached autoresearch BSV remap**, NOT a recomputed raw-spectrum projection (SHINE has no per-sample mean-spectra file). Its upstream **3-axis collapse is preserved** — raw nonzero axes per cohort = [2,2,2,2,2,2,2,2]. The UI states this and does not imply 11 independent measured axes.
- Global coordinates are applied to the sparse remapped BSV for comparability, but the dimensional limitation is inherited and flagged.

## Additional biological datasets — evaluated, EXCLUDED
| Dataset | Decision | Reason |
| --- | --- | --- |
| small2023 EV | excluded | no compatible per-sample mean-spectra table wired; would require new parsing/preprocessing validation |
| COVID serum Raman | excluded | Raman regime (calibration fit is Ag-SERS only) → would project off-distribution without a Raman reference; provenance/axis offset needs review |
| ovarian plasma Raman/SERS, saliva EV, others | excluded | not wired; substrate/laser metadata and identifier reliability not yet validated for projection |

Inclusion criteria (reliable IDs, compatible preprocessing, substrate/laser metadata, technically valid projection, improves cross-domain evaluation) were not met without additional work. Documented rather than silently dropped.

---

## Agreement / change vs prior views
- **Raw BSV** (V2): unchanged (≤1e-9).
- **Cohort-relative** (V2-style within-dataset z): retained as a labelled exploratory view; changes with comparison set by construction.
- **Global** (V3): new default; cohort-invariant; agrees with cohort-relative on directions (e.g. EV redox, serum protein) while adding cross-dataset comparability. Where the old radars over-emphasized a single axis by raw scale, the global view rebalances (see redox).
