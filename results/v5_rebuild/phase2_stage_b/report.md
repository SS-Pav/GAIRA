# Phase 2 Stage B — notebook summary

Reproduce:
```bash
python results/v5_rebuild/phase2_stage_b/code/run_stage_b.py   # ~40 min (per-fold encoder training)
python -m pytest tests/test_v5_evidence.py -q                  # 26 tests
```

**Corpus:** frozen Phase-2 manifest — 479 spectra (214 Raman + 265 Ag-SERS), 87 analytes, 51 matched. Splits A/B/C/D predeclared, all leakage checks pass; Split D infeasible for single-source Ag-SERS. Augmentation band retention 94%.

**Primary metric — held-out matched-analyte cross-modal retrieval (Split B, chance top-1 ≈ 0.098):**

| representation | top-1 | MRR | MRR CI | modality leak | cross-analyte dup |
| --- | --- | --- | --- | --- | --- |
| I1 adaptive regions | 0.294 | 0.460 | [0.36,0.56] | 0.79 | 0.00 |
| direct_SNV (baseline) | 0.275 | 0.452 | [0.35,0.54] | 0.86 | 0.00 |
| E1 shared encoder | 0.176 | 0.338 | [0.25,0.43] | 0.91 | **1.00** |
| E2 dual (primary) | 0.137 | 0.312 | [0.24,0.40] | 1.00 | **0.96** |

Full table in `GAIRA_V5_PHASE2_STAGE_B_REPRESENTATION_STRATEGY_REPORT.md`.

**Findings:** direct + interpretable occupy the top; every encoder is below direct and shows near-total embedding collapse (cross-analyte dup 0.96–1.00). Dual < shared encoder. Family retrieval (held-out analytes): direct 0.74 vs encoders 0.48–0.50. Within-modality Raman ARI: direct/I1 0.94 vs encoder 0.54–0.77.

**Decision: Outcome B4 — modality-stratified representations retained.** No shared biochemical representation supported; encoders collapse and add no value at this corpus scale (H1c/H1d rejected, H7 confirmed high-risk). Frozen shared representation: none. Working representation: modality-stratified direct SNV + I1 regions (auditable companion).

**Next:** Stage C re-scoped as targeted data acquisition (multi-source Ag-SERS, Au-SERS, external matched analytes) + interpretable refinement, then re-run this benchmark. Stage D (ontology) gated.

**STOP after Stage B.** No ontology / BSV / MSS / DART / perturbation / biological / scaling / pretraining started.
