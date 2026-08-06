# Phase 03 — Consensus Spectral Motif construction

**Status:** Not started — blocked by Phase 02

---

## Purpose

Reintegrate the per-class LSM dictionaries into one comparable cross-class coordinate system of Consensus Spectral Motifs.

## Why this phase exists

**This phase carries V7's central bet.** Partitioning (Phase 02) buys fair capacity allocation at the cost of comparability. If LSMs from different classes cannot be reintegrated, Strategy D has traded one problem for a worse one (risk R-03). The CSM is the canonical spectroscopic evidence unit of V7 and replaces the legacy MSS overlay.

## Inputs

- stable LSMs from Phase 02
- balanced references (for activation co-occurrence)
- frozen CV splits

## Outputs

- `csm_dictionary_v1.npz`, `csm_registry_v1.json` (C-07)
- `lsm_graph_v1.json` (C-06) including the threshold sweep
- **integration-method comparison table — committed regardless of the winner**
- CSM reference manual: one page per CSM with spectrum, bands, provenance, uncertainty, flags
- `reports/PHASE_03_REPORT.md`

## Gate — all must pass before the next phase begins

- [ ] Every CSM has explicit, resolvable provenance (LSMs → classes → analytes → sources)
- [ ] CSMs meet the pre-registered stability threshold
- [ ] `M` quantitatively justified against the pre-registered composite
- [ ] Integration method chosen on evidence; full comparison table published
- [ ] Singletons and anchors flagged, counted, reported — never hidden
- [ ] Threshold sweep performed; selection sits in a stable region
- [ ] If meta-NMF selected: molecule-discriminating structure survival verified

## Primary risks

- R-03 local dictionaries incomparable (**High severity**)
- R-06 second NMF removes detail (**High**)
- R-07 communities threshold-artefactual (High)
- R-05 consensus clustering arbitrary
- R-08 anchors duplicate motifs

### Six edge features — not one

| # | Feature | What it catches |
|---|---|---|
| 1 | spectral cosine | overall shape |
| 2 | diagnostic-band overlap | agreement *where it matters* |
| 3 | peak-position agreement | position is excitation-invariant; intensity is not |
| 4 | bootstrap recurrence co-occurrence | shared stability regime |
| 5 | activation co-occurrence | response to the same molecules |
| 6 | provenance overlap (within-class discounted) | shared evidence |

**Why not cosine alone.** The V5 motif table shows the trap directly:
`purine_ring_breathing` and `sterol_ring_system` shared **0.679** component support while
their activations correlated only **0.243**. Under cosine alone they merge; under features
2, 3, and 5 they clearly should not.

### Method comparison — no method is presumed

consensus clustering · Leiden/Louvain · spectral clustering · sparse non-negative
meta-factorisation · hybrid graph + factorisation.

**The plan does not presuppose that NMF-on-NMF wins.** The stated prior favours graph or
hybrid routes: meta-NMF sees only one of six edge features, and its equal row weighting
reintroduces the spectrum-count bias V7 exists to remove. That is a hypothesis to test.
Publish the full table whichever way it goes.

### Anchors (Strategy F)

Admission requires **all** of: quality gate, novelty gate (residual after projection onto the
existing CSM set exceeds threshold), written chemical justification, and a permanent
`is_anchored` flag with `n_analytes = 1` and inflated downstream uncertainty.

---

## What belongs in this directory

Phase-03 code, configs, notebooks, per-phase tables, and the phase report. Artefacts that
later phases consume belong in `../../results/` (tables, figures, manifests, checkpoints,
phase_outputs) so the provenance chain stays in one place.

**Do not store here:** raw spectra · large regenerable intermediates · anything with a
hard-coded absolute path · outputs from other phases.

## Reference documents

- `../../context/GAIRA_V7_CONTEXT.md` — canonical scientific context
- `../../plan/GAIRA_V7_REBUILD_PLAN.md` — Phase 03 in the full sequence
- `../../plan/VALIDATION_AND_DECISION_RULES.md` — pre-registered selection rules
- `../../plan/RISK_REGISTER.md` — full risk detail
- `../../architecture/DATA_CONTRACTS.md` — artefact schemas
