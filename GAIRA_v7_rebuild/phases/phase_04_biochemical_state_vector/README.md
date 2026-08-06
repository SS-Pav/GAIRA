> ## ARCHIVED — decision A-14
>
> Built and measured as canonical **Phase 04** (`results/v7_rebuild/phase04/`), which also
> absorbed the original Phase 05 engine work. The BSV was defined as `Sᵀc` over the accepted
> themes and inherits their information loss exactly; effective rank 2.40 of a nominal K = 4
> (risk R-12 realised). Phase 04 ended `GATE_FAILED` on a SERS out-of-domain gate that is now
> out of scope (A-09/A-17). All outputs preserved. Superseded by **BSV2** (Phase 07), which is
> derived from Chemistry Evidence rather than from themes. See
> `context/GAIRA_V7_ARCHITECTURE_STATUS_AFTER_PHASE05.md` §3 F-02, §5.2 A-14.

# Phase 04 — Frozen projection engine and Biochemical State Vector

**Status:** COMPLETE — 10 of 11 gates. Outputs at `results/v7_rebuild/phase04/`.

> **Scope.** This phase merges the plan's Phase 04 (BSV construction, contract C-09) and
> Phase 05 (engine integration, contract C-10), as commissioned. Both gates are carried.

---

## What was built

A projection-only inference engine. Given a new Raman spectrum it produces LSM activations,
CSM activations, theme activations, a BSV, latent-geometry coordinates, nearest neighbours,
uncertainty at every level and a resolvable provenance chain — **without refitting anything**.

Every stage was selected by benchmark, not assumed:

| stage | selected | over |
|---|---|---|
| projection | elastic net | NNLS, LASSO, ridge, ARD, OMP |
| LSM → CSM | direct CSM projection | four aggregation schemes |
| theme mode | confidence-weighted | softmax **rejected** for activating zero-evidence themes |
| BSV | theme-only, 4 axes | three wider variants, all of which separate molecules worse |
| geometry | landmark barycentric | Nyström, graph interpolation, kNN |

## Held-out results

| level | molecule top-1 (split A) | class top-1, unseen molecule (split B) | replicate consistency |
|---|---:|---:|---:|
| raw spectrum | 0.790 | 0.608 | 0.904 |
| LSM | **0.806** | 0.850 | 0.891 |
| CSM | 0.799 | **0.855** | 0.893 |
| theme / BSV | 0.553 | 0.405 | **0.979** |
| geometry | 0.495 | 0.541 | 0.946 |

**Dictionary-level leakage measured at +0.055 top-1** by refitting the class-local NMF per
fold in a scratch control.

## Gate

- [x] BSV deterministic and bit-identical
- [x] Absolute and derived forms never conflated (`bsv` vs `bsv_elevation`)
- [x] Every axis interpretable, with named supporting CSMs and molecules
- [x] Uncertainty propagated at every level
- [x] Effective rank reported alongside K — **2.40 of 4**
- [x] No fitting during inference (static AST check)
- [x] Batch independence
- [x] All assets fingerprinted and verified on load
- [ ] **OOD detection — FAILS on real Ag-SERS (AUROC 0.548). Reported, not compensated.**

## What Phase 05 consumes

`artifacts/engine_config_v1.json` · `artifacts/bsv_reference_v1.json` (C-09) ·
`artifacts/inference_v1.npz` · the `SpectrumState` object.

Phase 05 should apply the +0.055 leakage correction to any in-sample benchmark and must not
claim an out-of-domain capability until the SERS failure is addressed.

## Reference documents

- `../../../results/v7_rebuild/phase04/reports/PHASE_04_REPORT.md`
- `../../../results/v7_rebuild/phase04/reports/PHASE_04_SCIENTIFIC_AUDIT.md` — **not approved
  for any out-of-domain or cross-modality claim**
- `../../../results/v7_rebuild/phase04/reports/PHASE_04_FIGURES.pdf` — 14 figures
