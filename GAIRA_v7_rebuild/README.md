# GAIRA V7 — Hierarchical Foundation Rebuild

**Status: Phase 00 complete. No V7 model has been fitted.** The benchmark is locked, canonical
molecule identities are frozen, and the V5 control baseline is measured. No latent component
has been generated. No existing scientific asset has been modified.

| Field | Value |
|---|---|
| Branch | `gaira-v7-rebuild` |
| Source branch | `gaira-v5-rebuild-plan` |
| Source commit | `ddbb3945d670eee58f5ad99f868fb3c36b2a2c06` |
| Frozen V5 atlas (unchanged, still in production) | `09ed804a40836f4a05a91ba10900cded` |

---

## What V7 is

A rebuild of GAIRA's spectral representation from a **flat global NMF** to a **hierarchical,
analyte-balanced** decomposition:

```
Raw Raman grounding spectra
        ↓  canonical preprocessing (unchanged)
analyte-balanced reference construction        one molecule = one reference unit
        ↓
class-specific local latent decompositions     each chemistry gets its own capacity
        ↓
stable Local Spectral Motifs (LSMs)
        ↓
cross-class motif integration
        ↓
Consensus Spectral Motifs (CSMs)               the canonical evidence unit
        ↓
soft biochemical themes
        ↓
absolute continuous Biochemical State Vector (BSV)
        ↓
context-aware interpretation
```

---

## Why it exists

The current architecture works well at coarse chemistry and has plateaued at fine chemistry.
Every number below is from a committed or on-disk table (sources in
`context/PRIOR_ARCHITECTURE_LIMITATIONS.md`):

| Finding | Evidence |
|---|---|
| Broad-superclass retrieval is strong | **0.820** coord / **0.808** MSS, vs a size-matched random-ontology control of ~0.10 |
| Fine-family retrieval has plateaued | **0.647–0.677**, flat across three different ontologies |
| Ontology cleanup was not the fix | one significant fine-level gain out of four levels (system, +0.060, p=0.041); coord and MSS changed by −0.012 and −0.006, both non-significant |
| Most failures are representation failures | **57.4%** of MSS failures (31 of 54) survived ontology cleanup as true representation errors |
| Components are stable but impure | **3 of 24** components reach purity ≥ 0.5; median purity 0.328, median stability 0.799 |
| Motifs borrow foreign mass | `sterol_ring_system` AUC 0.683, top-activated by **fatty acids**; `porphyrin` top-4 includes **thymine** |
| Coverage is severely unbalanced | protein 32 analytes … polyol 1; **107 of 167** analytes uncovered by any v1 motif |

**Root cause.** A flat global decomposition asks one basis to represent broad shared structure
*and* rare molecule-specific structure under a single reconstruction loss, with capacity
allocated by spectrum count. The dense classes win.

**The V7 change.** The statistical unit moves from *one spectrum = one vote* to
**one canonical molecule = one scientific reference unit**, and decomposition capacity is
allocated per chemical class rather than globally.

---

## What remains frozen

V7 **reads** these. V7 never writes them.

- `assets/foundation/` — the frozen 24-component atlas, `09ed804a40836f4a05a91ba10900cded`
- canonical preprocessing (`src/gaira/preprocessing/`)
- the NNLS projection used by the existing engine (`src/gaira/engine/`)
- all V5, V6, V6.2, and V6.3 reports and artefacts
- existing Streamlit applications and inference engines

The V5 atlas stays in production for the entire duration of V7 development. It is the control
arm. It is replaced only if Phase 07 clears the criteria frozen in Phase 00 — and if it does
not, the V5 atlas simply stays.

---

## Folder layout

```
GAIRA_v7_rebuild/
├── README.md                  ← you are here
├── context/                   scientific context: why V7, what the terms mean, what the data is
├── plan/                      the rebuild plan, decision rules, criteria, risks, git policy
├── architecture/              target architecture, learning/inference modes, contracts, manifests
├── phases/                    one directory per phase, each with objectives, outputs, gates
├── code/                      V7 implementation (Phase-00 code lives in results/v7_rebuild/phase00/code/)
├── data_contracts/            machine-readable schemas (empty)
├── results/                   tables · figures · manifests · checkpoints · phase_outputs
├── reports/                   phase reports (Phase-00 report in results/v7_rebuild/phase00/reports/)
├── tests/                     V7 tests (empty; scaffold test at repo-root `tests/`)
└── archive/                   superseded V7 material (empty)
```

---

## Phase status

| Phase | Name | Status | Output | Gate |
|---|---|---|---|---|
| 00 | Benchmark lock | ✅ **COMPLETE** | [`results/v7_rebuild/phase00/`](../results/v7_rebuild/phase00/) | 12/12 PASS |
| 01 | **Balanced references + class-local NMF → LSMs** | ✅ **COMPLETE** | [`results/v7_rebuild/phase01/`](../results/v7_rebuild/phase01/) | 8/8 · compliance 18/18 |
| 02 | LSM construction | Not started | — | — |
| 03 | CSM construction | Not started | — | — |
| 04 | Themes | Not started | — | — |
| 05 | BSV | Not started | — | — |
| 06 | Engine wiring | Not started | — | — |
| 07 | Raman validation | Not started | — | — |
| 08 | Chemistry-aware learning | Deferred | — | — |
| 09 | Corpus expansion | Deferred | — | — |

Phases run in order. Each gate is binding: a failed gate stops the phase rather than being
waived because the next phase is more interesting.

---

## What is implemented

**Phase 00** — benchmark lock (frozen basis reproduced from raw, bit-exactly), canonical
molecule identities (167 surface forms → 154 molecules), the chemical partition (16 fine /
6 broad), frozen analyte-grouped CV splits, the frozen quality score, the frozen evaluation
harness, and the V5 control baseline measured under it.

**Phase 01** — `src/gaira/v7/lsm/`: balanced reference construction (8 arms), independent
class-local NMF with adaptive `k_c`, 50 Local Spectral Motifs, a class-indexed registry, LSM
typing, the Strategy-F anchor route, and R-01/R-16 risk checks. The frozen atlas is a control
only and is never an input (P-15).

## What is not yet implemented

No balanced reference matrix, no class-local decomposition, no CSM, no theme mapping, no BSV,
no V7 engine, no V7 atlas, no end-to-end V7 validation. No downstream benefit of the motif
layer has been demonstrated against the Phase-00 harness.

---

## How to work on this

1. **Read `context/GAIRA_V7_CONTEXT.md` first.** It is the canonical scientific context and is
   sufficient background for any phase prompt.
2. **Read the phase README** in `phases/phase_NN_*/README.md`.
3. **Check the pre-registered rules** in `plan/VALIDATION_AND_DECISION_RULES.md` *before*
   running a sweep. A rule stated after seeing the curve is a post-hoc choice and must be
   labelled as one.
4. **Write a manifest** for every artefact: inputs and hashes, config, seeds, code SHA,
   environment, outputs, gate results, decisions.
5. **Commit the report with the code that produced it.**
6. **Do not touch anything outside `GAIRA_v7_rebuild/`** except the V7 scaffold test.

### Non-negotiables

| | |
|---|---|
| **Raman only** | SERS, serum, and biological material are projected through a frozen atlas, never fitted to it |
| **Non-negative** | every basis, activation, membership, and BSV |
| **Deterministic** | fixed seeds; byte-reproducible; identical output on a second machine |
| **No inference-time fitting** | no NMF, PCA, UMAP, clustering, or community detection at runtime |
| **Absolute BSV** | ΔBSV, elevation, and cohort views are derived and separately named |
| **Themes are chemistry** | never a disease, pathway, process, or phenotype |
| **No spectrum duplication** | rare classes get anchors, not synthetic multiplicity |

---

## Expected eventual inference contract

```
input   wavenumbers, intensities, metadata {excitation, instrument, domain?}
        ↓
        canonical preprocessing → NNLS against the frozen CSM basis
        → theme mapping via frozen S → absolute BSV
        → reference comparison, QC, uncertainty → domain interpretation
        ↓
output  { atlas_version, atlas_fingerprint,
          csm_activations ℝ₊^M, theme_activations ℝ₊^K,
          bsv ℝ₊^K            [ABSOLUTE],
          bsv_elevation ℝ^K   [derived, signed],
          uncertainty, qc, confidence_tier, evidence, provenance }
```

No fitting. Batch-independent. Runs from a frozen bundle on a clean clone with no lab volume.
Full schema in `architecture/DATA_CONTRACTS.md` (C-10).

---

## Document index

**Context** — `context/`
`GAIRA_V7_CONTEXT.md` · `PRIOR_ARCHITECTURE_LIMITATIONS.md` · `SCIENTIFIC_DESIGN_PRINCIPLES.md`
· `TERMINOLOGY_AND_DEFINITIONS.md` · `DATASET_AND_PROVENANCE_CONTEXT.md` ·
`REPOSITORY_BASELINE.md` · `CONSISTENCY_AUDIT.md`

**Plan** — `plan/`
`GAIRA_V7_REBUILD_PLAN.md` · `PHASE_DEPENDENCY_MAP.md` · `VALIDATION_AND_DECISION_RULES.md` ·
`SUCCESS_CRITERIA.md` · `RISK_REGISTER.md` · `GIT_AND_VERSIONING_PLAN.md`

**Architecture** — `architecture/`
`GAIRA_V7_TARGET_ARCHITECTURE.md` · `LEARNING_MODE_ARCHITECTURE.md` ·
`INFERENCE_MODE_ARCHITECTURE.md` · `DATA_CONTRACTS.md` · `ARTIFACT_AND_MANIFEST_SPEC.md` ·
`LIVE_DART_COMPATIBILITY.md`

**Figures** — `results/figures/planning/` (10 diagrams, SVG + PNG)

---

## Terminology note

**MSS is legacy terminology.** The canonical V7 term for the cross-class evidence unit is the
**Consensus Spectral Motif (CSM)**. The mapping is `legacy MSS → V7 CSM`, and the two are not
the same object: MSS was a curated overlay laid over a fixed basis; a CSM is derived
bottom-up from cross-class consensus of independent local fits, with mandatory provenance.
Full definitions in `context/TERMINOLOGY_AND_DEFINITIONS.md`.

---

**Phase 00 is complete** — benchmark locked (basis reproduced from raw, bit-exactly), canonical
molecule identities frozen (167 surface forms → 154 molecules), the chemical partition and
cross-validation splits frozen, and the V5 control baseline measured under the frozen harness.
Report: [`PHASE_00_REPORT.md`](../results/v7_rebuild/phase00/reports/PHASE_00_REPORT.md).

**Phase 01 is complete and architecture-compliant (18/18)** — 8 balanced-reference arms
compared, 16 chemistry classes fitted by independent class-local NMF with adaptive
`k_c ∈ {1,2,3,5}`, yielding 50 Local Spectral Motifs. Rare chemistry now receives 2.5× the
decomposition capacity per molecule that dense chemistry does; under V5 both received the same.
Report: [`PHASE_01_REPORT.md`](../results/v7_rebuild/phase01/reports/PHASE_01_REPORT.md).

> **Architecture note.** An earlier implementation of Phase 01 decomposed the *frozen atlas*
> rather than fitting class-local NMF over balanced references. It was audited, found
> non-compliant, and reclassified as a **control experiment** — preserved in full at
> `results/v7_rebuild/control_experiments/frozen_atlas_decomposition/`, with its objects
> renamed Atlas Component Substructures so the term LSM is reserved for the specification's
> object. See [`ARCHITECTURE_COMPLIANCE_AUDIT.md`](context/ARCHITECTURE_COMPLIANCE_AUDIT.md).
> Principles **P-15** (the frozen atlas is a control, never a foundation), **P-16**
> (architecture check before implementation) and **P-17** (redraw the pipeline every phase)
> were added as a result.

**Next step: awaiting explicit approval. Phase 02 NOT STARTED.**
