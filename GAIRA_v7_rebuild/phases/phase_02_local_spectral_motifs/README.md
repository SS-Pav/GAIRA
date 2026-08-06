# Phase 02 — Local Spectral Motif construction

**Status:** Not started — blocked by Phase 01

---

## Purpose

Fit an independent non-negative decomposition within each chemical class and retain only stability-selected Local Spectral Motifs.

## Why this phase exists

This is Strategy D, the change most directly targeted at limitations L-02 and L-07. Reweighting a global fit would still leave every class competing for the same slots; partitioning gives rare chemistry its own capacity. A 32-analyte protein family and a 9-analyte sterol family can no longer trade capacity because they no longer share an objective.

## Inputs

- `balanced_references_v1.npz` from Phase 01
- `chemical_partition_v1.yaml` from Phase 00
- frozen CV splits

## Outputs

- one LSM dictionary per class; `lsm_registry_v1.json` (C-05)
- per-class `k_c` selection curves, stability, redundancy, residual analyses
- all basis spectra plotted
- discarded-LSM record with reasons
- per-class source and excitation composition tables
- class-prior bias analysis
- `reports/PHASE_02_REPORT.md`

## Gate — all must pass before the next phase begins

- [ ] Every retained LSM meets the pre-registered stability threshold
- [ ] Every class has documented motif coverage, or a documented reason for none
- [ ] `k_c` selected per class by the pre-registered rule — no arbitrary fixed `k`
- [ ] `k_c ≤ ⌊n_analytes/2⌋` for every class
- [ ] Rare classes handled explicitly via anchors, never by duplication (P-11)
- [ ] Per-class source/excitation composition reported
- [ ] Class-prior bias tested and reported

## Primary risks

- R-01 class-prior bias (High)
- R-02 rare classes too small (High)
- R-04 motif proliferation
- R-16 source/excitation confounding

### Procedure per class

1. sweep `k_c` in `[1, ⌊n_analytes/2⌋]`
2. `R` repeated fits — seed schedule + **analyte-level** bootstrap
   (never resample replicates: it leaks within-analyte structure and inflates stability)
3. Hungarian alignment across runs on cosine
4. recurrence-based stability score; retain above threshold
5. redundancy within class
6. type each retained LSM: class-shared | subfamily | molecule-discriminating
7. sparse-NMF vs plain NMF swept, not assumed

### Class viability

| Analytes | Classes | Treatment |
|---|---|---|
| 12–32 | protein, saccharide, amino_acid, triglyceride, organic_acid, fatty_acid | full decomposition |
| 5–9 | sterol, cofactor, purine, polysaccharide, lipid, (unknown) | low `k_c` |
| 3 | nucleic_acid, pyrimidine | `k_c ≤ 1` |
| 1–2 | phospholipid, small_nitrogenous, carotenoid, polyol | **anchor route only** |

### The LSM typing matters downstream

Class-shared LSMs from different classes describing the same chemistry are exactly what
Phase 03 must merge. Molecule-discriminating LSMs are exactly what it must not merge away.

---

## What belongs in this directory

Phase-02 code, configs, notebooks, per-phase tables, and the phase report. Artefacts that
later phases consume belong in `../../results/` (tables, figures, manifests, checkpoints,
phase_outputs) so the provenance chain stays in one place.

**Do not store here:** raw spectra · large regenerable intermediates · anything with a
hard-coded absolute path · outputs from other phases.

## Reference documents

- `../../context/GAIRA_V7_CONTEXT.md` — canonical scientific context
- `../../plan/GAIRA_V7_REBUILD_PLAN.md` — Phase 02 in the full sequence
- `../../plan/VALIDATION_AND_DECISION_RULES.md` — pre-registered selection rules
- `../../plan/RISK_REGISTER.md` — full risk detail
- `../../architecture/DATA_CONTRACTS.md` — artefact schemas
