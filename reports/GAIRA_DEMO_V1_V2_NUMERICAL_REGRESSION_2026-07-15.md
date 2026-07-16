# GAIRA Demo v1 → v2 — Numerical Regression

**Date:** 2026-07-15
**Claim under test:** v2's migration hardening changed **no scientific inference behavior**. v1 and v2 must produce numerically identical outputs for identical inputs.
**Result:** **PASS** — all output classes identical to floating-point tolerance (max abs diff `0.00e+00`, `atol = 1e-9`).

---

## Method

v1 and v2 both expose a package named `gaira_core`, so they cannot be imported
into one process. Each build was therefore run in its **own subprocess**
(`regress_worker.py <demo_dir> <out.json>`), producing a JSON of scientific
outputs; a comparator (`regress_compare.py`) then diffed the two JSONs per
output class. (Harness scripts live under the session scratchpad; they read
only, and are reproducible.)

For the "real" full-decomposition cases, the **input spectra are parsed by the
harness itself** (identical code for both builds) and fed to each build's
`build_report`, so any difference isolates to the engine — not to data loading.

`PYTHONHASHSEED=0` was fixed for both runs (see "Synthetic-noise nondeterminism"
below).

**Tolerance:** absolute, `atol = 1e-9`. Structural fields (confidence strings,
evidence ordering, caveats, substrate-rule identity) require exact equality.

---

## Cases tested (9 full reports + 4 loader tables = 20 loader rows)

| Case | Type | Substrate | Source |
| --- | --- | --- | --- |
| All six adenine concentrations | loader table (`adenine_6conc`, 6 rows) | Ag colloid SERS | `raw/adenine_sers_control/Adenine_bAgNPs_*.CSV` |
| Serum-liver, all 4 cohorts | loader table (`serum_liver`, 4 rows) | Ag colloid SERS | `patient_level_mean_spectra.csv` |
| EV-diabetes, both cohorts | loader table (`ev_diabetes`, 2 rows) | Ag colloid SERS | `sample_query_spectra.csv` |
| SHINE Day 0 + Day 2 × C0/C10/C20/C40 | loader table (`shine`, 8 rows) | — | autoresearch `class_mean_bsv_day0_day2.csv` |
| Real adenine 10 µg/mL | full report | Ag colloid SERS | `Adenine_bAgNPs_10micro.CSV` |
| Serum HA / CCA / HCC / LM (first patient each) | full report ×4 | Ag colloid SERS | `patient_level_mean_spectra.csv` |
| EV Impact / Strong-D (first sample each) | full report ×2 | Ag colloid SERS | `sample_query_spectra.csv` |
| Synthetic adenine | full report | Ag colloid SERS | `synth_reference_spectrum` |
| Synthetic ergothioneine (end-to-end example) | full report | Raman | `synth_reference_spectrum` |

This covers the required set: all six adenine concentrations; ≥1 serum-liver
input from every displayed cohort; both EV cohorts; SHINE Day 0 and Day 2; one
synthetic end-to-end example.

---

## Output classes compared and result

| Output class | What is compared | Max abs diff | Struct match | Verdict |
| --- | --- | --- | --- | --- |
| preprocessed_spectra | full 1401-pt processed intensity vector per case | **0.00e+00** | — | PASS |
| primitive_counts | `n_peaks` per case | **0** | — | PASS |
| motif_scores | raw + adjusted motif fire scores | **0.00e+00** | — | PASS |
| mss_scores | per-molecule fire / anchor / support / anti | **0.00e+00** | — | PASS |
| substrate_corrections | before / after / multiplier + rule identity | **0.00e+00** | rules identical | PASS |
| bsv_values | 11-axis BSV per case | **0.00e+00** | — | PASS |
| confidence_output | overall / substrate_sensitivity / specificity | — | identical | PASS |
| evidence_axis_ordering | axis order + evidence types + confidences | — | identical | PASS |
| caveat_generation | caveat type + summary list | — | identical | PASS |
| loader_bsv_tables | adenine(6) / serum(4) / ev(2) / shine(8) BSV rows | **0.00e+00** | labels identical | PASS |

**Tolerance:** `atol = 1e-9`. **OVERALL: PASS — v1 and v2 are numerically identical.**

The regression was re-run against the **final edited v2** (after the grounding-table
and provenance-text changes) with the same result: 0.00e+00 across all classes.

---

## Synthetic-noise nondeterminism (why `PYTHONHASHSEED=0`)

A first run **without** a fixed hash seed showed ~`1.5e-3` differences — but
**only** in the two synthetic cases (`synth_adenine`, `synth_ergothioneine`); all
seven real-file cases were already `0.00e+00`.

Root cause: `data_loader.synth_reference_spectrum` seeds its Gaussian noise with
`hash(mol_id)` when `seed=0`, and CPython randomizes `hash()` of strings **per
process** (`PYTHONHASHSEED`). The two subprocesses therefore drew different noise
realizations. This is a property of the **shared, unchanged** code (identical in
v1 and v2), not a divergence between the builds, and it affects only the
synthesized reference spectra, which the UI labels "synthesised, illustrative
only." Fixing `PYTHONHASHSEED=0` makes `hash()` identical across processes and
drives all differences to exactly 0.

---

## Intentional textual differences (NOT scientific, not in this comparison)

The regression compares scientific numeric/structural outputs of `build_report`
only. v2's intentional **text** changes live in `app.py` captions / `README` /
docstrings and do not enter `build_report`:

- provenance banner (REAL/DEGRADED/PLACEHOLDER),
- EV-diabetes caveat now naming the loader that actually ran,
- serum-liver caveat now stating canonical counts (212 patients; `SER-CCA-58`),
- per-axis grounding evidence caption / table,
- launch-path and title strings ("v2", "migration-hardened").

None of these alter preprocessing, primitives, motif/MSS scoring, substrate
corrections, BSV projection, thresholds, cohort selection, or inference.

---

## Conclusion

The expected scientific numerical difference — **zero / floating-point-equivalent**
— is confirmed for every tested output class. v2 is a behavior-preserving,
migration-hardened successor to the frozen v1 engine.
