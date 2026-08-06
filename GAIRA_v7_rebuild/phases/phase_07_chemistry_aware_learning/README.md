> ## SUPERSEDED — renumbered
>
> Now **Phase 10** (deferred). The canonical Phase 08 is
> `phase_08_hierarchical_molecular_retrieval/`. Retained unmodified for provenance.

# Phase 08 — Chemistry-aware representation learning

**Status:** **Deferred** — begins only after a passing Phase 07 and a frozen unsupervised candidate

---

## Purpose

Test whether chemistry-aware learning improves on the best unsupervised V7 architecture, with the architecture held fixed so gains are attributable.

## Why this phase exists

Learning is separated from architecture deliberately. Mixing them makes it impossible to say whether an improvement came from the hierarchy or from the objective — and the whole point of Phases 01–07 is to establish what learning-free architectural change buys on its own.

## Inputs

- frozen unsupervised V7 candidate from Phase 07
- frozen analyte-grouped CV splits

## Outputs

- learning-gain attribution
- comparison with unsupervised V7
- overfitting analysis
- calibration
- `reports/PHASE_08_REPORT.md`

## Gate — all must pass before the next phase begins

- [ ] Held-out gains demonstrated
- [ ] Interpretability preserved — a gain that costs provenance or band-level explanation is not accepted
- [ ] No label or domain leakage

## Primary risks

- R-10 in-sample evaluation (**Critical**)
- R-15 SERS contamination (**Critical**)

### Candidates

graph-regularised NMF · discriminative NMF · fixed-dictionary metric learning ·
hybrid spectroscopy-prior CSM mapping · weak chemical supervision.

### Hard constraints

- **No disease labels.**
- **No SERS training.**
- All learning nested inside analyte-grouped CV — no held-out analyte information may enter
  model selection.

### Why "deferred" and not "optional"

Phase 08 is a real phase with real value; it is deferred because running it before the
unsupervised architecture is frozen would confound the two sources of gain.

---

## What belongs in this directory

Phase-08 code, configs, notebooks, per-phase tables, and the phase report. Artefacts that
later phases consume belong in `../../results/` (tables, figures, manifests, checkpoints,
phase_outputs) so the provenance chain stays in one place.

**Do not store here:** raw spectra · large regenerable intermediates · anything with a
hard-coded absolute path · outputs from other phases.

## Reference documents

- `../../context/GAIRA_V7_CONTEXT.md` — canonical scientific context
- `../../plan/GAIRA_V7_REBUILD_PLAN.md` — Phase 08 in the full sequence
- `../../plan/VALIDATION_AND_DECISION_RULES.md` — pre-registered selection rules
- `../../plan/RISK_REGISTER.md` — full risk detail
- `../../architecture/DATA_CONTRACTS.md` — artefact schemas
