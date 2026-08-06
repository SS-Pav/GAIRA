# `GAIRA_v7_rebuild/phases/`

One directory per rebuild phase. Each holds that phase's objectives, outputs, gates, and — once
implementation begins — its code, configs, and phase report.

## Layout

> **Directory names use the ORIGINAL numbering.** They are not renamed, because committed
> reports, tests and artefacts link to them by path. The **canonical** phase number is in the
> second column and is authoritative. Mapping: original 01 + 02 → canonical 01; original 03 →
> canonical 02; everything after shifts down by one.

| Directory | Canonical phase | Status |
|---|---|---|
| `phase_00_benchmark_lock/` | **00** Benchmark lock and reproducibility baseline | ✔ COMPLETE |
| `phase_01_balanced_reference_construction/` | **01** stage 1 — balanced references | ✔ COMPLETE |
| `phase_02_local_spectral_motifs/` | **01** stage 2 — Local Spectral Motifs (50) | ✔ COMPLETE |
| `phase_03_consensus_spectral_motifs/` | **02** Consensus Spectral Motifs (49) | ✔ COMPLETE |
| — (no plan directory; specified in-flight) | **02.5** Latent geometry of motif space | ✔ COMPLETE — analysis only |
| `phase_04_biochemical_themes/` | **03** Biochemical themes | **ARCHIVED (A-13)** |
| `phase_05_biochemical_state_vector/` | **04** Biochemical State Vector | **ARCHIVED (A-14)** |
| — (no plan directory; specified in-flight) | **04.5** Meta Components / hierarchical NMF | **ARCHIVED (A-15) — discarded** |
| `phase_06_end_to_end_integration/` | **05** Canonical CSM inference engine | ✔ COMPLETE — re-scoped |
| **`phase_06_chemistry_evidence_layer/`** | **06** Chemistry Evidence Layer | **Not started — next approved step** |
| **`phase_07_bsv2_discovery/`** | **07** BSV2 Discovery | Not started |
| **`phase_08_hierarchical_molecular_retrieval/`** | **08** Hierarchical Molecular Retrieval | Not started |
| `phase_06_in_domain_raman_validation/` | **09** V5 head-to-head replacement decision | Retained — SUPERSEDED spec, to be rewritten |
| `phase_07_chemistry_aware_learning/` | **10** Chemistry-aware learning | Deferred |
| `phase_08_targeted_corpus_expansion/` | **11** Targeted corpus expansion | Deferred |

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
