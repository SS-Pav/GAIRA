# Phase 05 — Biochemical State Vector construction and normalisation

**Status:** Not started — blocked by Phase 04

---

## Purpose

Define the absolute BSV, its reference frame, its uncertainty model, its OOD support, and every derived form — keeping absolute and derived rigorously distinct.

## Why this phase exists

The BSV is GAIRA's output coordinate. Its absoluteness is what makes ΔBSV, cohort comparison, and DART trajectories meaningful; conflating an absolute vector with a difference is a correctness bug, not a naming quibble.

## Inputs

- CSM dictionary and `S` from Phases 03–04
- balanced references (for reference statistics)

## Outputs

- `bsv_reference_v1.json` (C-09)
- OOD support
- frozen visualisation transform (`P`, `μ`)
- reference distributions per axis
- worked end-to-end examples
- `reports/PHASE_05_REPORT.md`

## Gate — all must pass before the next phase begins

- [ ] BSV deterministic
- [ ] Absolute and delta forms not conflated anywhere in code, artefacts, or prose
- [ ] Every axis interpretable, with named supporting CSMs and chemistry
- [ ] Uncertainty propagated; singleton/anchor-supported axes carry inflated uncertainty
- [ ] Effective rank reported alongside `K`

## Primary risks

- R-12 BSV axes correlated

### The forms, kept distinct

| Form | Nature |
|---|---|
| **absolute BSV** `Sᵀc(x)` | the canonical coordinate |
| reference-normalised elevation | z-scored, signed, derived |
| **ΔBSV** `BSV₂ − BSV₁` | signed, derived |
| cohort-standardised view | visualisation only; cohort-dependent, not portable |
| DART trajectory `BSV(E,t)` | sequence of absolute BSVs |
| visualisation projection `y = Pᵀ(BSV−μ)` | **not the canonical BSV**; `P` applied, never fitted |

### Effective rank is mandatory

The V5 24-component space had participation ratio **15.2** and 16 components for 90% of
latent variance — a 38% gap between nominal and effective dimensionality, visible only
because it was measured. If V7's BSV shows a similar gap, `K` overstates the representation's
resolution and downstream users must be told. Report participation ratio, entropy rank, and
axes-for-90% alongside `K`.

### Support-aware uncertainty

An axis dominated by singleton or anchored CSMs must report **wider** uncertainty than one
supported by broad cross-class consensus. The V5 failure was that a motif with 1.2% corpus
coverage produced output indistinguishable in form from one with 7.2%.

---

## What belongs in this directory

Phase-05 code, configs, notebooks, per-phase tables, and the phase report. Artefacts that
later phases consume belong in `../../results/` (tables, figures, manifests, checkpoints,
phase_outputs) so the provenance chain stays in one place.

**Do not store here:** raw spectra · large regenerable intermediates · anything with a
hard-coded absolute path · outputs from other phases.

## Reference documents

- `../../context/GAIRA_V7_CONTEXT.md` — canonical scientific context
- `../../plan/GAIRA_V7_REBUILD_PLAN.md` — Phase 05 in the full sequence
- `../../plan/VALIDATION_AND_DECISION_RULES.md` — pre-registered selection rules
- `../../plan/RISK_REGISTER.md` — full risk detail
- `../../architecture/DATA_CONTRACTS.md` — artefact schemas
