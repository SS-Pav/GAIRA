# GAIRA V5 — Phase 1 Preprocessing & Comparability Report (decision gate)

**Date:** 2026-07-18 · Branch `gaira-v5-rebuild-plan` · Governs: `GAIRA_V5_REBUILD_PLAN.md` Phase 1 (V5.1). Lab notebook: `results/v5_rebuild/phase1/` (code, figures, tables, logs, report.md). Hypotheses: `GAIRA_V5_HYPOTHESIS_REGISTER.md`.

> **Decision: STOP at the Phase-1 gate. Do NOT begin Phase 2 (observation-layer fitting).** Adopt modality-stratified analysis; the matched Raman↔Ag-SERS overlap (7 analytes) is too thin to estimate a cross-mode transform. Next required step is a data step (load the Gobbato pure-metabolite corpus), not a modeling step.

## What was built (canonical, in `src/gaira`)
- `src/gaira/data/` — spectrum contract (`SpectrumRecord`), the ONE canonical loader, admission gate. Phase 0 registries emitted (`results/v5_rebuild/phase0/tables/`): 295 observations, **271 admitted**, **11 acquisition domains** (9 Raman excitations + Ag-SERS colloid 633/785).
- `src/gaira/preprocessing/` — 6 deterministic candidate pipelines (baseline/smoothing/normalization) on a common 520–1750 cm⁻¹ / 2 cm⁻¹ grid.
No scoring engine, ontology, PCA-for-axes, BSV, or MSS was built (correctly deferred).

## Evidence (271 admitted spectra: 202 Raman + 69 Ag-SERS; 7 matched analytes)
| Pipeline | cov R | cov S | matched x-mod cosine | null cosine | modality leakage CV acc |
| --- | --- | --- | --- | --- | --- |
| raw+L2 | 1.00 | 0.997 | 0.527 | 0.410 | 0.823 |
| asls+L2 | 1.00 | 0.997 | 0.483 | 0.332 | 0.841 |
| asls+savgol+L2 | 1.00 | 0.997 | 0.484 | 0.333 | 0.841 |
| asls+savgol+SNV | 1.00 | 0.997 | 0.253 | 0.022 | 0.856 |
| poly+savgol+L2 | 1.00 | 0.997 | 0.248 | 0.037 | 0.860 |
| asls+savgol+area | 1.00 | 0.997 | 0.484 | 0.333 | 0.745 |

Majority-class baseline (all-Raman) = 0.745. Matched analytes: adenine, arginine, asparagine, cytochrome c, glutathione, histidine, tryptophan.

## The nine gate questions
1. **Can Raman and Ag-SERS share one preprocessing pipeline?** They share a common **window and pipeline family** (ASLS + conservative Savitzky–Golay), but **normalization should be modality-specific** — a single identical pipeline is not clearly best (SNV vs L2 vs area change cross-modality similarity from 0.25 to 0.53).
2. **Are spectra comparable after preprocessing?** **Within a modality, yes; across modalities, only weakly** (same-analyte Raman↔Ag-SERS cosine 0.25–0.53). SERS surface-selection physics reshapes the fingerprint.
3. **Preprocessing choices retained:** common window 520–1750 @2 cm⁻¹; ASLS baseline; conservative Savitzky–Golay; **L2 per modality** as default (SNV kept as an inspected alternative).
4. **Preprocessing choices rejected:** one universal cross-modality normalization; treating Raman and Ag-SERS intensities as directly comparable; combining peak-only (ORC-Ag) with full-spectrum data.
5. **Matched analytes across modalities:** 7 (listed above).
6. **Metadata gaps remaining:** Raman spans **9 excitation domains** (nuisance axis); Gobbato pure-metabolite spectra + amino-acid xlsx not yet loaded; ORC-Ag excitation/concentration unknown.
7. **Proceed to observation-layer phase?** **No.**
8. **If yes, why?** N/A.
9. **If not, what must be fixed first?** Expand matched Raman↔Ag-SERS analytes from 7 toward ~50 by loading the **Gobbato corpus** (53 metabolites measured as BOTH pure Raman powder AND pure Ag-SERS); resolve Raman excitation stratification. Only then is a Phase-2 observation model estimable. If matched overlap stays small, keep modality-stratified analysis and treat cross-mode transfer as a **data-acquisition gap** (including the confirmed absence of any Au-SERS grounding).

## Hypothesis outcomes
- **H1** (shared observation space): *insufficient evidence, leaning against direct sharing.*
- **H1a** (enough matched analytes to fit a transform): *rejected* (7).
- **H1b** (matched > others across modalities): *partially supported* (matched > null, weak).
- **H1c** (preprocessing preserves bands): *supported per modality.*
- **H6** (structure reflects chemistry not acquisition): *partially rejected* (modality leaks into PCA).

## Recommended next sprint (data, not modeling)
Parse and load the Gobbato pure-metabolite corpus (Ag-SERS 265 + Raman powders 153) and the amino-acid Raman panel into the canonical loader; re-run the Phase-1 matched-analyte comparability with ~50 matched pairs; then re-evaluate whether a Phase-2 observation model (reliability weighting / matched-analyte alignment) is estimable. Do not build the observation model until matched overlap is sufficient.
