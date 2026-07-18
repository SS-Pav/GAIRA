# GAIRA Demo Reasoning V3 — Completion Report

**Date:** 2026-07-17 · **Branch:** `gaira-v3-global-coordinate-prototype`
**Release:** GAIRA Demo Reasoning V3 — Global Coordinate Prototype

> V3 introduces a frozen global biochemical coordinate calibration built on GAIRA's transparent heuristic spectral evidence engine. It is a prototype universal coordinate system, **not** yet a trained Raman foundation model, generative model, validated clinical measurement, learned latent model, or molecular quantification system.

---

## 1. What V3 adds (over frozen V2)
- **Frozen, versioned global coordinates**: `raw_bsv → (raw − center)/scale → global`, robust per-axis, fit label-free on 275 biological Ag-SERS spectra, floored at 0.02, clip ±4. Loaded from an artifact; **never refit at runtime**.
- **Biochemical Ontology v1** (`config/biochemical_ontology_v1.yaml`) — disease-label-independent, with per-axis grounding status.
- **Three explicit coordinate systems**, never overwriting each other: `raw_bsv`, `global_biochemical_coordinates`, `cohort_relative_coordinates`.
- **4-mode UI**: How GAIRA Works (+ Ontology, + Raw→Global tabs) · Calibration Evidence (+ Axis Coverage) · Global Biological Projection (3 view-modes, cross-dataset map, nuisance) · Coordinate Validation.
- **6 reports, 6 test files (16 tests), 3 build tools, 4 generated artifacts.**

## 2. Preservation
- **V1** verified against `reports/gaira_demo_reasoning_v1_sha256_2026-07-15.txt` → all 16 files OK; unchanged.
- **V2** verified clean vs commit `1674c89` → no diff, no untracked. Unchanged.
- V3 was created by `rsync` from V2; its 8 scientific engine modules remain **byte-identical** to V2 (asserted in `test_v2_raw_bsv_regression.py`).

## 3. Raw V2→V3 regression
All raw BSV outputs identical to V2 across 14 cases (6 adenine, 4 serum, 2 EV, 2 synthetic): **max abs diff ≤ 1e-9**. PASS.

## 4. Global calibration method
Robust per-axis standardization (median center, 1.4826·MAD scale, floor 0.02, clip ±4), fit **label-free** on the biological reference population (serum 212 + EV 63); calibration titrations (adenine, ergothioneine) projected but excluded from the fit. Deterministic (identical content hash across rebuilds; timestamp excluded).

## 5. Ontology axis-status summary
Independently grounded: **G04, G05, G06**. Partially grounded: **G03, G07**. Derived split: **G01, G02, G08, G09**. Insufficiently grounded: **G10, G11**. → 11 axes inherited from **8 independent legacy dimensions** (3 split families). NA preserved for split-axis analyte counts.

## 6. Calibration results
- **Adenine:** Spearman(logC,G01)=0.83, target/off-target=6.6×, 6/6 loaded live. Supportive.
- **Ergothioneine:** Spearman(C,G10)=0.94 (55 live spectra); redox routes to G10 (range 0.067) over sibling G11 (0.009); global redox rises 0.68→3.9 and exceeds the biological range. Supportive.
- **Hypoxanthine/uricase/uric-acid (3 separate contrasts):** hypoxanthine spike = supportive; hypoxanthine+uricase = supportive; **uricase depletion = inconsistent (6/11 axes disagree, n=5/5) — preserved, not laundered.**

## 7. Disease projection
- **EV diabetes:** 63 per-sample; Impact vs Strong-D global effects — Redox d=+2.22, Purine-nuc +1.45, Aromatic +1.40. Redox signal reproduced without cohort-dependent coordinates.
- **Serum liver:** 212 per-patient; CCA vs HA Protein d=−2.03; LM vs HA Protein −1.40 / Sterol −1.18; HCC vs HA Glycan −0.64 / Purine-met +0.53.
- **SHINE:** Day0/Day2 × dose projected from legacy 3-axis remap (collapse preserved; not recomputed).

## 8. Cohort-invariance
Global coordinates identical across comparison sets (alone / own cohort / different disease cohort / mixed EV+serum): **max deviation ≤ 1e-9**. Cohort-relative coordinates change as expected. PASS.

## 9. Redox dominance
Raw variance rank **2 → global rank 2** (comparability, not cosmetic equality). Global max|z|≈6.5; ergothioneine extremes exceed the biological reference range [−2.15, 6.53]. Calibration removes raw-scale dominance while preserving genuine extreme redox.

## 10. Nuisance / domain
Mean dataset-identity η² = **0.49 (moderate)**; Purine-nuc 0.92, Purine-met 0.86, Aromatic 0.85. Matrix (serum vs EV) η²: Aromatic 0.77, Lipid 0.63. **Dataset identity remains a moderate-to-strong separator — full cross-domain invariance is NOT achieved** (prototype). Fit population is 100% Ag-SERS; Raman untested. No batch correction applied (diagnostic only).

## 11. Datasets included / excluded
- **Included (projected):** serum-liver (212), EV-diabetes (63), adenine (6), ergothioneine (55), SHINE (8 cohorts, legacy remap).
- **Excluded:** small2023 EV, COVID serum Raman, ovarian plasma, saliva EV, others — not wired / Raman off-distribution / identifier-preprocessing-metadata not validated. Documented in the disease-projection report.

## 12. Determinism, missing-data, no-leakage
- Determinism: identical calibration content hash on rebuild (timestamp excluded). PASS.
- REAL mode: 8/8 sections real. DEGRADED mode (empty `GAIRA_DATA_ROOT`): frozen calibration + reference samples remain available (bundled); live data tabs degrade honestly. **Missing calibration → "GLOBAL COORDINATE UNAVAILABLE", raw BSV retained, never refit.** PASS.
- No leakage: label-free fit reproduces the stored calibration; label permutation cannot change it. PASS.

## 13. Tests (16/16 pass, `tests/run_all.py`)
`test_v2_raw_bsv_regression` (2) · `test_global_coordinate_invariance` (2) · `test_calibration_behavior` (3) · `test_reference_space_coverage` (3) · `test_global_coordinate_determinism` (3) · `test_no_label_leakage` (3).

## 14. Known scientific limitations
1. Global coordinates are a deterministic **prototype**, not a learned/clinical measurement.
2. Fit population 100% Ag-SERS; **Raman generalization untested**.
3. **6 of 11 axes not independently grounded** (three legacy split families).
4. **Dataset identity is a moderate separator** — not yet cross-domain invariant.
5. The underlying BSV is a heuristic band-evidence measure (inherited from V2), not a validated molecular model.
6. SHINE remains a 3-axis upstream projection.
7. Uricase-depletion calibration is inconsistent (preserved honestly).

## 15. Launch
```bash
cd /Users/surajpg/projects/GAIRA/gaira_demo_reasoning_v3
python selfcheck.py          # data resolution
python tests/run_all.py      # 16 tests
./run_demo.sh                # or ../.venv/bin/streamlit run app.py
```
Rebuild artifacts (optional, deterministic): `python tools/build_global_coordinate_reference.py && python tools/build_axis_coverage.py`.

## 16. Exact statement of what V3 is / is not
GAIRA Reasoning V3 is a **deterministic global-coordinate prototype**. It preserves the transparent V2 band-evidence inference engine (raw BSV unchanged to ≤1e-9), adds a frozen versioned biochemical ontology and robust global axis calibration, and projects calibration and biological spectra into cohort-invariant coordinates. It is **not** yet a trained Raman foundation model or a validated molecular quantification system.
