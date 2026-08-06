# `results/v7_rebuild/`

Generated artefacts for the GAIRA V7 rebuild. One directory per phase.

| Phase | Directory | Status |
|---|---|---|
| 00 Benchmark lock and canonical data foundation | `phase00/` | **COMPLETE** |
| 01 Balanced references + class-local NMF → LSMs | `phase01/` | **COMPLETE** — 18/18 architecture compliance |
| 02 Consensus Spectral Motifs | — | not started |
| 03 Biochemical themes | — | not started |
| 04 Biochemical State Vector | — | not started |
| 05 Engine integration | — | not started |
| 06 In-domain Raman validation | — | not started |
| 07 Chemistry-aware learning | — | deferred |
| 08 Targeted corpus expansion | — | deferred |
| — Control experiment: frozen-atlas decomposition | `control_experiments/frozen_atlas_decomposition/` | **COMPLETE — NOT canonical V7** |

## Per-phase layout

```
phaseNN/
├── code/         implementation + orchestrator + validation + figure scripts
├── tables/       CSV outputs
├── figures/      SVG (vector) + PNG (preview)
├── reports/      the phase report
├── logs/         run logs
├── artifacts/    binary intermediates (gitignored by extension where regenerable)
├── validation/   PASS/FAIL/WARN validation results
├── manifests/    build manifest, data cards, split manifests
└── PHASE_STATE.json
```

## What must not be stored here

- **Raw spectra.** `data/`, `GAIRA_DATA/`, `/Volumes/`, `*.mat` are gitignored.
- **Anything belonging to V5 or V6.** Those trees are separate scientific records and are
  read-only for V7.
- **Absolute local paths.** Raw data resolves through `GAIRA_DATA_ROOT`.
- **PDFs.** Repo policy gitignores `*.pdf`; Markdown + SVG are tracked.

## Rules

1. Every artefact is referenced by its phase manifest, with a SHA-256.
2. Every phase report is committed with the code that produced it.
3. Phases run in order; each phase's frozen output is the next phase's input.
