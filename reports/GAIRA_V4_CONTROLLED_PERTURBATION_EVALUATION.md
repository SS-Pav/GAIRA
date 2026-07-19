# GAIRA V4 — Controlled Perturbation Evaluation

**Date:** 2026-07-18 · Registry: `data_audit/v4_controlled_perturbation_evaluation_registry.csv`.

## The key correction
> The serum spike-in, uricase, isotope, and dose-response datasets are **NOT for calibrating GAIRA**. They are **held-out controlled perturbation tests** of whether an **independently grounded** GAIRA inference responds in the expected biochemical direction. They must **never** define or fit axes, weights, centers, or scales.

Renamed hierarchy: **Molecular grounding** (defines evidence) → **Controlled Perturbation Evaluation** / *Grounded Perturbation Tests* (test response) → **Biological challenge sets** (unknown mixtures). Retire the term "calibration" (no parameters are fitted) and "validation" (evidence too limited for formal validation).

## Evaluation registry (all model-frozen-before-eval; all excluded from axis/weight fitting)
| Evaluation | Challenge type | Expected | Observed | Verdict |
| --- | --- | --- | --- | --- |
| adenine concentration | dose-response challenge | G01 ↑ with conc | Spearman 0.83 | **supportive** |
| ergothioneine dose | dose-response challenge | G10 ↑ with conc | Spearman 0.94 | **supportive** |
| hypoxanthine spike (serum) | spike challenge | G02 ↑ | agree (small) | supportive |
| hypoxanthine + uricase | enzyme-depletion challenge | hypox ↑, UA ↓ | agree | supportive |
| **uric-acid uricase depletion** | enzyme-depletion challenge | UA/G02 ↓ | **INCONSISTENT (6/11 axes wrong)** | **inconsistent (preserved)** |
| ¹⁵N-uric-acid | isotope challenge | band shift confirms UA | mechanistic | context |
| 53 serum metabolite spikes | analytical challenge set | per-metabolite response | **not evaluated** (available, unwired) | not_evaluated |
| European inter-instrument adenine | cross-platform challenge | stable purine top-1 | top-1 across cAg/cAu/sAg/sAu; G01 CV~0.14 | partially_supportive |
| serum protocol comparison | cross-platform challenge | protocol stability | not evaluated | not_evaluated |

## Rules enforced
- Model frozen before evaluation: **YES** for all.
- Grounding sources allowed: pure-analyte references only.
- Evaluation data excluded from axis/weight/center/scale fitting: **YES** (V3 frozen calibration is fit on biological range, not on these perturbations).
- Do not turn the inconsistent uricase contrast into support (preserved).

## Migration plan (safe, no bulk renames now)
Introduce the term **"Grounded Perturbation Tests"** in UI/reports; keep file names but add a `role` column mapping each dataset to `controlled_perturbation_evaluation` / `analytical_challenge_set` / `dose_response_challenge` / `enzyme_depletion_challenge` / `cross_platform_challenge`. Rename files only after this registry is adopted.
