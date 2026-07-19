# Stage B0 — Preprocessing AutoResearch — notebook summary

**Outcome P4 — apparent improvement is caused by overprocessing. No pipeline frozen.** Read-only study; nothing pushed; Stage A/B and historical systems untouched.

## Reproduce (in order)

```bash
cd results/v5_rebuild/preprocessing_autoresearch/code
python freeze_study.py      # freeze nested splits + acceptance thresholds (BEFORE the search)
python run_search.py        # 120 candidates, arms A-G, inner folds only   (~4 min)
python run_outer_test.py    # the ONE-TIME outer test (refuses to rerun)
python make_outputs.py      # derived tables, controls, artifacts
python make_report.py       # figures + PDF
python -m pytest ../../../../tests/test_v5_preprocessing_autoresearch.py -q
```

## Design

- Corpus: frozen Stage-B manifest — 479 spectra (214 Raman / 265 Ag-SERS), 87 analytes, **51 matched**, 785 nm, grid 520–1750 @ 2 cm⁻¹ (fixed).
- Nested: **5 outer × 4 inner**, analyte-grouped; both modalities of a held-out analyte are test-only; outer test consumed **once** (`configs/study_manifest.json`).
- Background models and all fold-dependent fitting use **training spectra only**; no Raman spectrum influences any Ag-SERS spectrum (unit-tested).
- Thresholds + rejection rules frozen before the outer test.

## Result

Outer test (held-out), reference baseline = ASLS+SG+L2 (MRR 0.366, top-1 0.176, peak specificity +0.022, Ag-SERS replicate cos 0.946):

| pipeline | MRR | top-1 | peak spec. | Ag-SERS repl. cos |
| --- | --- | --- | --- | --- |
| raw + L2 | 0.359 | 0.176 | **+0.035** | **0.998** |
| ASLS+SG+L2 (ref) | 0.366 | 0.176 | +0.022 | 0.946 |
| ASLS+SG+SNV (control) | 0.422 | 0.235 | +0.022 | 0.515 |
| + mean subtraction | 0.351 | 0.167 | −0.002 | 0.810 |
| best MRR (ineligible) | **0.464** | **0.284** | +0.014 | 0.620 |
| best background (ineligible) | 0.434 | 0.255 | +0.020 | 0.579 |

- 120 candidates, **66 rejected** by integrity rules (58 for Ag-SERS replicate destruction).
- **Of 67 candidates improving MRR, 0 also improved peak specificity** (threshold-independent).
- Arm D's own winner was **`background = none`**; mean subtraction made held-out retrieval worse.
- **Control 5 (positive):** background removal does *not* destroy analyte information — held-out Ag-SERS analyte 1-NN stays 0.877–0.916 (baseline 0.896) even after removing 84% of Ag-SERS variance, improving to 0.916 for scaled-mean.
- Matched-pair cosine **falls** for nearly every analyte; ranks improve only because mismatched similarity falls faster.

## Interpretation

The binding constraint is **acquisition contrast in the Ag-colloid measurement**, not preprocessing. The Ag-SERS analyte residual is real and reproducible but carries no usable analyte-specific correspondence with powder Raman. Operations that appear to help do so by degrading spectra.

**Next (recommendation only):** targeted Ag-SERS re-acquisition where the analyte dominates the colloid (higher surface coverage, blank-colloid difference spectra, or Au-SERS references), then re-run this same frozen design. No encoder/representation/ontology/BSV/MSS work authorized.

Full write-up: `GAIRA_V5_PREPROCESSING_AUTORESEARCH_REPORT.md` · PDF: `GAIRA_V5_PREPROCESSING_AUTORESEARCH_REPORT.pdf` (11 pages, 7 figures).
