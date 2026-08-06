# Phase 07 — Full in-domain Raman validation

**Status:** Not started — blocked by Phase 06

---

## Purpose

Evaluate the complete V7 stack against the V5 control under one harness, and deliver a replacement recommendation.

## Why this phase exists

**The decision phase.** Everything before this is construction. Here V7 either clears the criteria frozen in Phase 00 or it does not — and if it does not, the V5 atlas is retained and the negative result is documented in full.

## Inputs

- V7 engine and frozen atlas from Phase 06
- V5 control atlas
- frozen splits, metrics, and criteria from Phase 00

## Outputs

- unified V7 Raman validation report
- per-analyte appendix
- head-to-head comparison with the current atlas under one harness
- **replacement recommendation**

## Gate — all must pass before the next phase begins

- [ ] V7 meets the Tier-1 success criteria frozen in Phase 00, **or** the V5 atlas is retained and the negative result is documented (P-13)

## Primary risks

- R-10 in-sample evaluation (**Critical**)
- R-17 corpus is the binding constraint (**High**)

### Report at every layer

LSM retrieval · CSM top-1/top-3 · fine-family retrieval · theme top-1/top-3 ·
broad-superclass retrieval · system-level (if retained) · MRR · balanced accuracy · macro-F1 ·
permutation null · bootstrap CIs · calibration · reconstruction · diagnostic-band fidelity.

### The failure waterfall

The most informative diagnostic this project has produced. Four categories:

| Category | Meaning |
|---|---|
| **true projection failures** | the representation genuinely cannot separate them — **the number that matters** |
| semantic rescues | V5 failures that V7 resolves |
| semantic degradations | V5 successes that V7 breaks — **reported with equal prominence** |
| stable recoveries | correct under both |

Baseline to beat (V5, MSS layer, n=167): 54 failures, of which **31 (57.4%)** were true
representation errors.

### Diagnosing R-17

If Phase 07 fails while Phase 02/03 diagnostics look healthy — stable LSMs, coherent CSMs,
complete provenance, but flat retrieval — the binding constraint is the corpus, not the
architecture. That is a legitimate and useful finding, and it routes to Phase 09 rather than
to further architectural work.

---

## What belongs in this directory

Phase-07 code, configs, notebooks, per-phase tables, and the phase report. Artefacts that
later phases consume belong in `../../results/` (tables, figures, manifests, checkpoints,
phase_outputs) so the provenance chain stays in one place.

**Do not store here:** raw spectra · large regenerable intermediates · anything with a
hard-coded absolute path · outputs from other phases.

## Reference documents

- `../../context/GAIRA_V7_CONTEXT.md` — canonical scientific context
- `../../plan/GAIRA_V7_REBUILD_PLAN.md` — Phase 07 in the full sequence
- `../../plan/VALIDATION_AND_DECISION_RULES.md` — pre-registered selection rules
- `../../plan/RISK_REGISTER.md` — full risk detail
- `../../architecture/DATA_CONTRACTS.md` — artefact schemas
