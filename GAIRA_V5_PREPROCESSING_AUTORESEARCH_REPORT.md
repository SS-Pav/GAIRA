# GAIRA V5 — Stage B0: Preprocessing AutoResearch for Raman↔Ag-SERS Comparability

**Date:** 2026-07-19 · Branch `gaira-v5-rebuild-plan` · **Read-only controlled study.** No GAIRA implementation, Stage A/B result, historical pipeline, demo, or governing document was modified. Nothing was pushed.

> **DECISION: Outcome P4 — apparent improvement is caused by overprocessing. No preprocessing pipeline is frozen.**
>
> Cross-modal retrieval **can** be pushed past the pre-declared success bar (held-out MRR +0.097 against a +0.08 threshold; top-1 +0.108 against +0.05) — but **only** by pipelines that strip the broad shared Ag-SERS component. Those pipelines collapse Ag-SERS replicate agreement (0.946 → 0.620) and **do not improve matched-vs-mismatched band specificity, which falls**. Of 67 candidates that improved MRR, **zero** also improved peak specificity. The gain is contrast geometry, not recovered shared chemistry.

A retrieval-only study would have declared success here. The multi-objective design is what caught it.

---

## 1. Question and design

**Question.** Stage A/B found weak Raman↔Ag-SERS correspondence; the matched-analyte spectral audit showed the Ag-SERS arm is dominated by a shared colloid background with the analyte surviving as a ~5% residual. Can physically reasonable, leakage-safe preprocessing and explicit Ag-SERS background modelling recover cross-modal correspondence **without destroying real analyte information**?

**Corpus (frozen, unchanged).** 479 spectra — 214 Raman / 265 Ag-SERS — 87 analytes, **51 matched**; 785 nm; adenine perturbation series excluded; grid 520–1750 cm⁻¹ @ 2 cm⁻¹ (fixed, not optimised).

**Nested, leakage-safe evaluation.**
- **Outer:** 5 folds of held-out matched analytes; **both modalities** of a held-out analyte are test-only. Consumed **exactly once** (recorded in `configs/study_manifest.json`).
- **Inner:** 4-fold analyte-grouped CV within each outer development set — all selection happened here.
- Ag-SERS background models, and every fold-dependent quantity, are fitted on **training spectra only**. No Raman spectrum ever influences an Ag-SERS spectrum (verified by test). No analyte labels build spectral vectors.
- Acceptance thresholds and rejection rules were **frozen before** the outer test ran (`configs/acceptance_thresholds.json`, `configs/rejection_rules.json`).

**Search.** 120 complete candidate pipelines across seven controlled arms (A baseline×normalization, B smoothing, C replicate aggregation, D Ag-SERS background models, E derivatives, F modality-specific vs global, G rational combinations) plus the frozen prior baselines reproduced unchanged. Structured arm-by-arm, not a Cartesian sweep; deterministic under seed 0.

---

## 2. Headline results

### Outer test (held-out matched analytes, used once)

| Pipeline | MRR | 95% CI | top-1 | peak specificity | Ag-SERS replicate cos | Raman 1-NN | Ag-SERS 1-NN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw + L2 | 0.359 | [0.30, 0.42] | 0.176 | **+0.035** | **0.998** | 0.978 | 0.823 |
| **ASLS+SG+L2 (reference)** | 0.366 | [0.31, 0.43] | 0.176 | +0.022 | 0.946 | 0.967 | 0.896 |
| ASLS+SG+SNV (control) | 0.422 | [0.36, 0.49] | 0.235 | +0.022 | 0.515 | 0.967 | 0.881 |
| ASLS+SG+L2 + mean subtraction | 0.351 | [0.29, 0.42] | 0.167 | −0.002 | 0.810 | 0.967 | 0.896 |
| *B_0_savgol6* (best MRR, **ineligible**) | **0.464** | [0.40, 0.54] | **0.284** | +0.014 | 0.620 | 0.984 | 0.861 |
| *D_scaled_mean2* (best background, **ineligible**) | 0.434 | [0.37, 0.51] | 0.255 | +0.020 | 0.579 | 0.973 | 0.916 |

Retrieval rises exactly where replicate preservation collapses, and band specificity never improves — the **raw** baseline has the highest peak specificity of all.

### The rule-independent finding

Of the 67 candidates that improved cross-modal MRR over the reference baseline, **19 survived the integrity rules, and 0 of those also improved matched-vs-mismatched peak specificity**. This does not depend on any threshold choice.

### Why candidates were rejected (66 of 120)

| Reason | n |
| --- | --- |
| Ag-SERS replicate destruction | 58 |
| peak loss | 7 |
| Ag-SERS analyte-discrimination drop | 6 |
| analyte collapse (near-duplicate spectra) | 5 |
| peak invention | 5 |
| peak broadening | 5 |
| effective-rank collapse | 3 |
| Raman replicate destruction | 3 |

---

## 3. Arm-by-arm findings

- **A — baseline × normalization.** Aggressive broad-structure removal (poly-3, morphological) maximises MRR (0.460) but drives Ag-SERS replicate cosine to 0.46. **SNV — included only as a declared negative control — scores well on retrieval**, exactly the failure mode this design was built to catch.
- **B — smoothing.** Retrieval rises monotonically as peak retention falls; the top candidates broaden bands (width ratio 1.25–1.29) and lose 11–15% of peaks.
- **C — replicate aggregation.** Robust (Huber) aggregation was mildly best; effect small and not decisive.
- **D — Ag-SERS background models (primary).** **Arm D's own winner was `background = none`.** Mean subtraction made held-out retrieval *worse* (0.351 vs 0.366). Scaled-mean and low-rank removal raised inner MRR (0.385 → 0.445) but every variant failed the replicate-integrity rule.
- **E — derivatives.** Derivative order 0 won; first/second derivatives did not help.
- **F — modality-specific pipelines.** Allowing Raman and Ag-SERS to differ gave no advantage over one global pipeline.
- **G — combinations.** No rational combination exceeded the arm winners.

---

## 4. Scientific controls

- **Control 1–2 (mismatched-analyte and random-peak nulls).** Reported for every candidate (`tables/null_control_results.csv`); matched-minus-mismatched peak correspondence is the specificity metric used throughout.
- **Control 3 (label permutation).** Implemented in the retrieval objective; available per candidate.
- **Control 4 (background variance explained).** Low-rank removal accounts for 53–84% of Ag-SERS variance; scaled-mean 27–38%.
- **Control 5 (analyte residual retention) — the important positive result.** Background removal does **not** destroy analyte information: held-out Ag-SERS analyte 1-NN stays at **0.877–0.916** (baseline 0.896) even when 84% of Ag-SERS variance is removed, and **improves to 0.916** under scaled-mean subtraction. The analyte residual is real and retained — it simply does not align with the Raman spectra.
- **Controls 6–7 (clean vs noisy Ag-SERS exemplars)** and **Control 8 (Raman source sensitivity)** are carried in `tables/per_analyte_before_after.csv` (`raman_multi_source` flag); 27 of 51 analytes draw Raman replicates from two sources, which inflates within-Raman spread but does not change the conclusion.

**Family-stratified (Figure 6).** Ranks "improve" for purines, cofactors and proteins under the best-MRR candidate — but matched cosine **falls** for essentially every family (e.g. polysaccharide 0.89 → 0.32, purine 0.55 → 0.36). Ranks improve only because mismatched similarity falls faster once the common component is removed.

---

## 5. Decision

**Outcome P4 — apparent improvement is caused by overprocessing.**

- Retrieval rises (ΔMRR +0.097, Δtop-1 +0.108 — both past the frozen thresholds), **but**
- Ag-SERS replicate cosine falls to 0.62 (integrity floor 0.90× baseline = 0.85) — **fails**,
- matched-peak specificity **falls** (−0.007 for the best-MRR candidate; required +50%) — **fails**,
- matched-pair cosine falls for nearly every analyte, so the ranking gain is a contrast-geometry effect.

Not P1 (integrity and specificity gates fail). Not P2 (P2 requires peak specificity to improve; it does not). Not P3 (improvement is measurable, not absent — it is simply an artifact). Not P5 (the pattern is uniform across families and folds, not heterogeneous).

**Frozen preprocessing artifact: none.** `artifacts/selected_pipeline.json` records `frozen: false` with the reason.

---

## 6. What this means

The bottleneck is **acquisition contrast in the Ag-colloid measurement, not the preprocessing pipeline**. The analyte signal in Ag-SERS is present, reproducible, and survives aggressive background removal (Control 5) — but it does not carry analyte-specific correspondence with powder Raman at usable strength. No amount of baseline, smoothing, normalization, aggregation, derivative or background modelling recovers correspondence that the measurement does not contain; the operations that appear to help do so by degrading the spectra.

This is consistent with, and mechanistically explains, Stage B's Outcome B4 and the spectral audit's finding.

**Next authorized action (recommendation only — not executed):** targeted Ag-SERS re-acquisition in which the analyte, not the colloid, dominates the spectrum (higher effective surface coverage, blank-colloid difference measurement, or Au-SERS references), then re-run **this same frozen study design** on the enlarged corpus. No representation, encoder, ontology, BSV or MSS work is authorized by this study.

---

## 7. Proposed documentation updates (NOT applied — awaiting approval)

Per instruction, governing documents were **not** modified. If approved, the changes would be:
- **Hypothesis register:** add a Stage-B0 entry recording that preprocessing/background correction does not rescue cross-modal comparability (P4), and strengthen H7 (corpus/acquisition is the binding constraint) with this evidence.
- **Rebuild plan:** insert Stage B0 before any Stage-B repeat, with outcome P4 and the re-acquisition recommendation; note that Stage C's data-acquisition re-scoping is now supported by direct experimental evidence, not only by inference.
- **Architecture context:** record that SNV and aggressive baseline removal inflate cross-modal retrieval metrics without improving band specificity — a standing methodological caution for any future comparability metric.

---

## 8. Artifacts

`results/v5_rebuild/preprocessing_autoresearch/`
- `configs/` — `nested_splits.json` (fingerprint `708564c4…`), `acceptance_thresholds.json`, `rejection_rules.json`, `study_manifest.json` (`outer_test_used: true`)
- `tables/` — `pipeline_catalog.csv`, `search_results.csv`, `search_results_judged.csv`, `pareto_front.csv`, `outer_test_results.csv`, `per_analyte_before_after.csv`, `peak_integrity_results.csv`, `background_model_results.csv`, `background_variance_vs_retention.csv`, `replicate_preservation.csv`, `cross_modal_retrieval.csv`, `null_control_results.csv`, `family_results.csv`, `final_decision.json`
- `artifacts/` — `selected_pipeline.json` (frozen: false), `preprocessing_manifest.json`, per-candidate JSON
- `figures/` — 7 publication figures · `report.md` · PDF: `GAIRA_V5_PREPROCESSING_AUTORESEARCH_REPORT.pdf`
- Code: `src/gaira/preprocessing_autoresearch/` (reuses `gaira.preprocessing`, `gaira.representation`, `gaira.evidence`) · tests: `tests/test_v5_preprocessing_autoresearch.py`
