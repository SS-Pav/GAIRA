# Phase 01 — Balanced references and Local Spectral Motifs

**Status:** COMPLETE — gate passed; corpus re-audited before Phase 02 and reproduced
bit-identically (registry fingerprint `208482d6f7178b5b8f16cace91be55b0`).

> **Numbering.** This directory is the merge of the original plan's Phase 01 (balanced
> reference construction) and Phase 02 (Local Spectral Motif construction). They are one
> pipeline — balanced references exist only to be split by class and fitted — and were merged
> under the renumbering adopted 2026-08-06 (`../../plan/GAIRA_V7_REBUILD_PLAN.md`, top).
> Every gate from both original phases is carried below; none was dropped.

---

## Purpose

**Stage 1.** Decide what a row of the reference matrix is: all spectra, analyte-balanced
weighted spectra, or one robust prototype per analyte.

**Stage 2.** Fit an independent non-negative decomposition *within each chemical class* and
retain only stability-selected Local Spectral Motifs.

## Why this phase exists

Global NMF minimises a sum over rows, so an analyte with three replicates outvotes one with a
single spectrum (limitation L-01). Stage 1 tests whether changing the row definition materially
changes the representation. Stage 2 is Strategy D, targeted at L-02 and L-07: reweighting a
global fit would still leave every class competing for the same slots, whereas partitioning
gives rare chemistry its own capacity. A 30-molecule protein family and a 10-molecule sterol
family can no longer trade capacity because they no longer share an objective.

## Inputs

- balanced-reference inputs from Phase 00 (canonical IDs, replicate groups, quality `q`)
- `chemical_partition_v1` from Phase 00
- frozen analyte-grouped CV splits

## Outputs

- selected strategy + written rationale; `balanced_references_v1.npz` (C-04)
- one LSM dictionary per class; `lsm_registry_v1.csv` / `lsm_dictionary_v1.npz` (C-05)
- per-class `k_c` selection curves, stability, redundancy, residual analyses
- discarded-LSM record with reasons; per-class source and excitation composition
- `results/v7_rebuild/phase01/reports/PHASE_01_REPORT.md`

## Gate — carried from both original phases

**Stage 1**

- [x] Selection rule stated in `VALIDATION_AND_DECISION_RULES.md` **before** the sweep ran
- [x] No label supervision anywhere in the construction
- [x] Selected method improves class balance without materially damaging spectral fidelity
- [x] Replicated-analyte and multi-excitation stratifications reported
- [x] `B-uniform` sensitivity arm reported; control arm A reported honestly

**Stage 2**

- [x] Every retained LSM meets the pre-registered stability threshold
- [x] Every class has documented motif coverage, or a documented reason for none
- [x] `k_c` selected per class by the pre-registered rule — no arbitrary fixed `k`
- [x] `k_c ≤ ⌊n_analytes/2⌋` for every class
- [x] Rare classes handled explicitly via anchors, never by duplication (P-11)
- [x] Per-class source/excitation composition reported; class-prior bias tested

## Primary risks

R-01 class-prior bias · R-02 rare classes too small · R-04 motif proliferation ·
R-10 in-sample evaluation · R-15 SERS contamination · R-16 source/excitation confounding

### Stage 1 arms

| Arm | Strategy |
|---|---|
| A | all spectra, equal row weight — **control, = V5 behaviour** |
| B | analyte-balanced quality-weighted — **selected** |
| B-uniform | B with uniform `q` — isolates balancing from quality weighting |
| C-mean / C-median / C-trimmed / C-medoid / C-quality | prototype variants |

### Stage 2 procedure per class

1. sweep `k_c` in `[1, ⌊n_analytes/2⌋]`
2. `R = 12` repeated fits — seed schedule + **analyte-level** bootstrap
   (never resample replicates: it leaks within-analyte structure and inflates stability)
3. Hungarian alignment across runs on cosine
4. recurrence-based stability score; retain above threshold
5. redundancy within class
6. type each retained LSM: class-shared | subfamily | molecule-discriminating

### Result

50 LSMs across 16 classes, `k_c ∈ {1, 2, 3, 5, 6, 7, 10}`; types 21 class-shared /
26 subfamily / 3 molecule-discriminating; mean stability 0.967; 0 rejected; 0 anchors needed.

### The LSM typing matters downstream

Class-shared LSMs from different classes describing the same chemistry are exactly what
**Phase 02** must merge. Molecule-discriminating LSMs are exactly what it must not merge away.

---

## What belongs in this directory

Phase-01 code, configs, notebooks, per-phase tables, and the phase report. Artefacts that
later phases consume belong in `../../../results/v7_rebuild/phase01/` so the provenance chain
stays in one place.

**Do not store here:** raw spectra · large regenerable intermediates · anything with a
hard-coded absolute path · outputs from other phases.

## Reference documents

- `../../context/GAIRA_V7_CONTEXT.md` — canonical scientific context
- `../../plan/GAIRA_V7_REBUILD_PLAN.md` — Phase 01 in the full sequence
- `../../plan/VALIDATION_AND_DECISION_RULES.md` — pre-registered selection rules
- `../../plan/RISK_REGISTER.md` — full risk detail
- `../../architecture/DATA_CONTRACTS.md` — artefact schemas
