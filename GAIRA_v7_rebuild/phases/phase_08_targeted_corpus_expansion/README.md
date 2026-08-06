# Phase 09 — Targeted corpus expansion

> ## SUPERSEDED — renumbered
>
> Now **Phase 11** (deferred). Retained unmodified for provenance.


**Status:** **Deferred** — driven by Phase 07 residual analysis

---

## Purpose

Acquire new reference spectra for spectral directions V7 measurably cannot span.

## Why this phase exists

Balancing changes how existing evidence is weighted; it does not create evidence. Phospholipid chemistry has 2 analytes before balancing and 2 after. Sphingolipids are absent entirely. Some gaps are only closable by acquisition.

## Inputs

- V7 residual analysis from Phase 07
- current corpus coverage

## Outputs

- prioritised acquisition list, each entry traced to a measured residual direction
- expected effect per acquisition, stated before ingestion
- post-ingestion re-measurement
- `reports/PHASE_09_REPORT.md`

## Gate — all must pass before the next phase begins

- [ ] Every addition traced to a measured residual direction
- [ ] Expected effect stated before ingestion
- [ ] Effect re-measured after ingestion
- [ ] Corpus rebalance re-run; class partition revisited if the addition changes it

## Primary risks

- R-17 corpus is the binding constraint (**High**)

### Candidate priorities and current support

| Chemistry | Current support |
|---|---|
| sterols / steroids | 9 analytes, 7 uncovered (77.8%); motif AUC 0.683 |
| porphyrins / heme | motif coverage 4 analytes (2.4%) |
| flavins | motif coverage **2 analytes (1.2%)** |
| phosphate chemistry | no dedicated v1 motif |
| phospholipids | **2 analytes, 100% uncovered** |
| sphingolipids | **absent entirely** |
| organic acids | 15 analytes, 8 uncovered (53.3%) |
| sulfur / redox cofactors | cofactor 6 analytes, 2 uncovered |
| carotenoid | **2 analytes, 100% uncovered** |
| nucleic acid | **3 analytes, 100% uncovered** |

### The rule

> **Do not add datasets merely because they are available.**

Each addition must address a *measured* missing spectral direction identified by V7 residual
analysis, with a stated expected effect — and the effect must be re-measured after ingestion.
Acquisition driven by availability is how a corpus grows without getting better.

---

## What belongs in this directory

Phase-09 code, configs, notebooks, per-phase tables, and the phase report. Artefacts that
later phases consume belong in `../../results/` (tables, figures, manifests, checkpoints,
phase_outputs) so the provenance chain stays in one place.

**Do not store here:** raw spectra · large regenerable intermediates · anything with a
hard-coded absolute path · outputs from other phases.

## Reference documents

- `../../context/GAIRA_V7_CONTEXT.md` — canonical scientific context
- `../../plan/GAIRA_V7_REBUILD_PLAN.md` — Phase 09 in the full sequence
- `../../plan/VALIDATION_AND_DECISION_RULES.md` — pre-registered selection rules
- `../../plan/RISK_REGISTER.md` — full risk detail
- `../../architecture/DATA_CONTRACTS.md` — artefact schemas
