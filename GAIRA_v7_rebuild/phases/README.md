# `GAIRA_v7_rebuild/phases/`

One directory per rebuild phase. Each holds that phase's objectives, outputs, gates, and — once
implementation begins — its code, configs, and phase report.

## Layout

| Directory | Phase | Status |
|---|---|---|
| `phase_00_benchmark_lock/` | 00 Benchmark lock and reproducibility baseline | Not started — **next approved step** |
| `phase_01_balanced_reference_construction/` | 01 Balanced reference construction | Not started |
| `phase_02_local_spectral_motifs/` | 02 Local Spectral Motif construction | Not started |
| `phase_03_consensus_spectral_motifs/` | 03 Consensus Spectral Motif construction | Not started |
| `phase_04_biochemical_themes/` | 04 Biochemical theme construction | Not started |
| `phase_05_biochemical_state_vector/` | 05 BSV construction and normalisation | Not started |
| `phase_06_end_to_end_integration/` | 06 End-to-end engine integration | Not started |
| `phase_07_in_domain_raman_validation/` | 07 Full in-domain Raman validation | Not started |
| `phase_08_chemistry_aware_learning/` | 08 Chemistry-aware learning | Deferred |
| `phase_09_targeted_corpus_expansion/` | 09 Targeted corpus expansion | Deferred |

## What belongs in a phase directory

Phase-specific code, configs, and the phase README. Once a phase runs, its report goes to
`../reports/` and its artefacts to `../results/` — so the provenance chain stays in one place
rather than being scattered across ten phase directories.

## What must not be stored here

- **Raw spectra.**
- **Artefacts other phases consume.** Those belong in `../results/`, referenced by a manifest.
- **Outputs from a different phase.**
- **Anything with a hard-coded absolute path.**

## Rules

1. **Phases run in order.** Each phase's frozen output is the next phase's input. Starting a
   phase on provisional upstream output silently breaks the provenance chain.
2. **Gates are binding.** A failed gate stops the phase; it is not waived because the next
   phase is more interesting.
3. **Decision rules are pre-registered** in `../plan/VALIDATION_AND_DECISION_RULES.md` before
   the sweep they govern is run.

See `../plan/GAIRA_V7_REBUILD_PLAN.md` for the full sequence and
`../plan/PHASE_DEPENDENCY_MAP.md` for dependencies and backtrack paths.
