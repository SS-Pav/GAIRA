# Phase 06 — End-to-end V7 engine integration

**Status:** Not started — blocked by Phase 05

---

## Purpose

Wire one canonical inference path, freeze the V7 atlas bundle, and verify every architectural invariant.

## Why this phase exists

The two-mode split — offline learning, live projection — is what makes spectra measured years apart comparable. This phase is where that property is either enforced or quietly lost.

## Inputs

- all frozen layers from Phases 00–05

## Outputs

- V7 engine (parallel to, not replacing, `src/gaira/engine/`)
- versioned output schema (C-10)
- API-ready interface
- frozen atlas bundle + `MANIFEST.json` with the multi-layer fingerprint
- reproducibility tests
- `reports/PHASE_06_REPORT.md`

## Gate — all must pass before the next phase begins

- [ ] **No fitting during inference** — static check for `fit`/`fit_transform`/`partial_fit`/RNG
- [ ] **Batch independence** — output identical alone vs in a batch of N
- [ ] Clean clone runs frozen inference with no lab volume and `GAIRA_DATA_ROOT` unset
- [ ] All assets fingerprinted; multi-layer atlas fingerprint verified on load
- [ ] Deterministic output, verified twice and on two machines
- [ ] Domain isolation — no domain object reachable from any pre-BSV module
- [ ] LSM layer retained in the bundle (needed by the future SERS observation model)

## Primary risks

- R-14 inference nondeterministic (**Critical**)
- R-13 runtime too complex

### The multi-layer fingerprint

V5 hashed only the NMF basis — adequate then, because the basis *was* the atlas. It is not
adequate for V7: an atlas with an identical CSM basis but a different `S` produces different
BSVs, and a basis-only fingerprint would make two behaviourally different atlases
indistinguishable. The V7 fingerprint covers seven layers in fixed order, with per-layer
hashes recorded individually so it is possible to say *which* layer changed.

### Do not slim the bundle by dropping the LSM layer

It is tempting to ship CSMs only. The future Raman→SERS observation model needs mode-level
structure, and the LSM dictionary is the finest structure available. It stays in the bundle
and in the fingerprint.

### The V5 property that must not regress

`assets/foundation/` is self-contained: inference runs on a clean clone with no raw data and
no lab volume. V7 must match this.

---

## What belongs in this directory

Phase-06 code, configs, notebooks, per-phase tables, and the phase report. Artefacts that
later phases consume belong in `../../results/` (tables, figures, manifests, checkpoints,
phase_outputs) so the provenance chain stays in one place.

**Do not store here:** raw spectra · large regenerable intermediates · anything with a
hard-coded absolute path · outputs from other phases.

## Reference documents

- `../../context/GAIRA_V7_CONTEXT.md` — canonical scientific context
- `../../plan/GAIRA_V7_REBUILD_PLAN.md` — Phase 06 in the full sequence
- `../../plan/VALIDATION_AND_DECISION_RULES.md` — pre-registered selection rules
- `../../plan/RISK_REGISTER.md` — full risk detail
- `../../architecture/DATA_CONTRACTS.md` — artefact schemas
