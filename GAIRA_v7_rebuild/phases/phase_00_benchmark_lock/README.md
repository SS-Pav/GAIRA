# Phase 00 — Benchmark lock and reproducibility baseline

**Status:** ✅ **COMPLETE** — 12/12 gates PASS · validation 84 PASS / 3 WARN / 0 FAIL · 63 tests

Outputs: `results/v7_rebuild/phase00/` · report: [`PHASE_00_REPORT.md`](../../../results/v7_rebuild/phase00/reports/PHASE_00_REPORT.md)

---

## Purpose

Freeze everything the rest of V7 is measured against: canonical molecule identities, replicate groups, quality metadata, the chemical-family partition, analyte-grouped CV splits, evaluation metrics, the V5 control baseline, and the final success criteria.

## Why this phase exists

Everything downstream is measured against what is frozen here. V6.3 demonstrated the cost of measuring the wrong thing carefully: a full revalidation established that the fine-family ceiling was not a labelling artefact, which was valuable — but only because the harness was sound. Phase 00 is where V7's harness is made sound.

## Inputs

- raw Raman grounding corpus (375 spectra, 167 analytes, via `GAIRA_DATA_ROOT`)
- `assets/foundation/` — the V5 control atlas (read-only)
- `results/v6_rebuild/v63_ontology_revalidation/` — the V6.3 ontology (currently uncommitted on disk; commit-or-rederive decision required)
- `src/gaira/preprocessing/pipeline.py` — canonical preprocessing

## Outputs

- `canonical_analytes_v1.csv` (C-00)
- `replicate_groups_v1.csv` (C-01)
- `spectrum_quality_v1.csv` (C-02)
- `chemical_partition_v1.yaml` with per-class chemical rationale
- `evaluation_ontology_v7.csv`
- `cv_splits_v1.json` (C-03)
- `results/tables/phase00_baseline_metrics.csv` — V5 control under the V7 harness
- `results/manifests/dataset_role_map_v7.csv`
- `SUCCESS_CRITERIA.md` marked final
- `reports/PHASE_00_REPORT.md`

## Gate — all must pass before the next phase begins

All twelve gates PASS. See `results/v7_rebuild/phase00/PHASE_STATE.json`.

- [x] No alias leakage — every surface form maps to exactly one canonical ID
- [x] No replicate leakage — no canonical ID or replicate crosses a fold boundary
- [x] All three `cv_splits_v1.json` leakage checks read `false`
- [x] V5 baseline reproduced; fingerprint `09ed804a40836f4a05a91ba10900cded` verified
- [x] Atlas rebuilt bit-exactly from raw (max abs difference 0.0)
- [x] All inputs versioned and hashed
- [x] Splits deterministic under re-cut
- [x] Every class has a written chemical rationale
- [x] `unknown` class resolved (dissolved by the V6.3 fine ontology)
- [x] Every canonical ID covered by the frozen ontology
- [x] Quality score `q` frozen before Phase 01
- [x] Success criteria frozen

### Key results

| | |
|---|---:|
| benchmark lock level | **3** (basis refitted from raw) |
| max \|H_rebuilt − H_frozen\| | **0.0** |
| surface forms → canonical molecules | 167 → **154** |
| cross-source merges (leakage removed) | **11** (49 spectra) |
| fine / broad classes | 16 / 6 |
| CV folds (grouped by canonical_id) | 5 |
| V5 control, MSS fine retrieval@1 | **0.6707** |
| S-01 frozen replacement threshold | **≥ 0.7507** |

## Primary risks

- R-09 alias/replicate leakage (**Critical**)
- R-10 in-sample evaluation (**Critical**)
- R-15 SERS contamination (**Critical**)
- R-01 class-prior bias
- R-16 source/excitation confounding

### Known work items

**Alias hazards to resolve** (all observed in existing tables):

| Variants | Issue |
|---|---|
| `riboflavin` / `riboﬂavin` | U+FB02 ligature — currently two separate entries |
| `acetyl coenzyme a` / `acetyl-coa` | same molecule, **two different family assignments** |
| `urea` / `ure` | truncation |
| `13-methylmyristicacid` / `12-methyltetradecanoic acid` | inconsistent spacing convention |
| `(+)-arabinose` / `(-)-arabinose` | enantiomers — **must NOT be merged** |

**Partition problems to resolve:**

1. `unknown` (6 analytes) is not a chemistry — assign or exclude from partitioning.
2. `lipid` (5) overlaps `fatty_acid` (12) and `triglyceride` (15).
3. `polysaccharide` (5) vs `saccharide` (27) — V6 gave them distinct motifs; keeping them
   separate is defensible, but the decision needs a written rationale either way.

**Replicate group key.** Recommended: `(canonical_id, excitation)`, with analyte balancing
applied at `canonical_id` level across groups — preserving excitation as a tracked nuisance
factor without letting the 41 multi-excitation analytes buy extra weight. To be ratified here.

**Preprocessing gotcha.** `pipeline.py::common_grid()` defaults to the legacy
Ag-SERS-constrained 520–1750 cm⁻¹ window. The atlas window is 450–1800. V7 must always pass
the window explicitly.

---

## What belongs in this directory

Phase-00 code, configs, notebooks, per-phase tables, and the phase report. Artefacts that
later phases consume belong in `../../results/` (tables, figures, manifests, checkpoints,
phase_outputs) so the provenance chain stays in one place.

**Do not store here:** raw spectra · large regenerable intermediates · anything with a
hard-coded absolute path · outputs from other phases.

## Reference documents

- `../../context/GAIRA_V7_CONTEXT.md` — canonical scientific context
- `../../plan/GAIRA_V7_REBUILD_PLAN.md` — Phase 00 in the full sequence
- `../../plan/VALIDATION_AND_DECISION_RULES.md` — pre-registered selection rules
- `../../plan/RISK_REGISTER.md` — full risk detail
- `../../architecture/DATA_CONTRACTS.md` — artefact schemas
