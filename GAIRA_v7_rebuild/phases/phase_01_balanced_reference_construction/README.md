# Phase 01 — Balanced reference construction

**Status:** Not started — blocked by Phase 00

---

## Purpose

Decide what a row of the reference matrix is: all spectra, analyte-balanced weighted spectra, or one robust prototype per analyte.

## Why this phase exists

Global NMF minimises a sum over rows, so an analyte with three replicates outvotes one with a single spectrum (limitation L-01). This phase tests whether changing the row definition materially changes the representation — at row level. Class-level rebalancing is Strategy D, tested in Phase 02.

## Inputs

- balanced-reference inputs from Phase 00 (canonical IDs, replicate groups, quality `q`)
- frozen CV splits

## Outputs

- selected strategy + written rationale
- `balanced_references_v1.npz` / `.csv` (C-04)
- arm comparison tables and figures
- discarded-variance asset if a prototype strategy is selected
- `reports/PHASE_01_REPORT.md`

## Gate — all must pass before the next phase begins

- [ ] Selection rule stated in `VALIDATION_AND_DECISION_RULES.md` **before** the sweep ran
- [ ] No label supervision anywhere in the construction
- [ ] Selected method improves class balance without materially damaging spectral fidelity
- [ ] Replicated-analyte stratification reported
- [ ] Multi-excitation stratification reported
- [ ] `B-uniform` sensitivity arm reported
- [ ] Control arm A reported honestly

## Primary risks

- R-10 in-sample evaluation (**Critical**)
- R-15 SERS contamination (**Critical**)

### Arms

| Arm | Strategy |
|---|---|
| A | all spectra, equal row weight — **control, = V5 behaviour** |
| B | analyte-balanced quality-weighted |
| B-uniform | B with uniform `q` — isolates balancing from quality weighting |
| C-mean / C-median / C-trimmed / C-medoid / C-quality | prototype variants |

### Two mandatory stratifications

1. **Restricted to the 87 replicated analytes**, as well as corpus-wide. 80 of 167 analytes
   are singletons, for which every arm is identical — corpus-wide numbers will be diluted
   toward zero and make the arms look falsely equivalent.
2. **Single-excitation vs multi-excitation.** 41 analytes span excitations. Per-bin mean and
   median can distort band *shape* across excitations (peak positions are excitation-invariant,
   relative intensities are not); the medoid cannot, because it is always a real measured
   spectrum. Per-excitation prototypes must be evaluated as an alternative to collapsing.

### If the control wins

That is the finding. Report it, proceed with A, and note that Strategy D remains a separate
and still-untested bet.

---

## What belongs in this directory

Phase-01 code, configs, notebooks, per-phase tables, and the phase report. Artefacts that
later phases consume belong in `../../results/` (tables, figures, manifests, checkpoints,
phase_outputs) so the provenance chain stays in one place.

**Do not store here:** raw spectra · large regenerable intermediates · anything with a
hard-coded absolute path · outputs from other phases.

## Reference documents

- `../../context/GAIRA_V7_CONTEXT.md` — canonical scientific context
- `../../plan/GAIRA_V7_REBUILD_PLAN.md` — Phase 01 in the full sequence
- `../../plan/VALIDATION_AND_DECISION_RULES.md` — pre-registered selection rules
- `../../plan/RISK_REGISTER.md` — full risk detail
- `../../architecture/DATA_CONTRACTS.md` — artefact schemas
