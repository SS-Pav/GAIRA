# GAIRA V7 — Architecture Compliance Audit

**Audit date:** 2026-08-06 · **Branch** `gaira-v7-rebuild` · **Audited commit** `0904d66`

This is an **architecture audit**, not a code audit. It asks one question of every completed
phase: *does the implementation match the approved written specification?* Every finding was
established by re-reading the specification documents, not from memory.

**Verdict in one line: Phase 00 is compliant with two minor documentation gaps. The phase
labelled "Phase 01" is NOT compliant — it implemented a different architecture and is
reclassified as a control experiment.**

---

## 1. The approved V7 architecture

Restated from `architecture/GAIRA_V7_TARGET_ARCHITECTURE.md`,
`architecture/LEARNING_MODE_ARCHITECTURE.md` and `plan/GAIRA_V7_REBUILD_PLAN.md`:

```
Balanced reference corpus                       one canonical molecule = one reference unit
        ↓
Split by chemistry class                        X → {X_c}, an organisational prior
        ↓
Independent class-local NMF                     X_c ≈ W_c H_c, adaptive k_c, no global competition
        ↓
Local Spectral Motifs (LSMs)                    rows of H_c, stability-selected
        ↓
Cross-class similarity graph                    6 edge features over the pooled LSM set
        ↓
Consensus Spectral Motifs (CSMs)                the canonical evidence unit
        ↓
Soft biochemical themes                         t = Sᵀc, S sparse non-negative row-normalised
        ↓
Continuous absolute BSV                         BSV(x) = t(x) ∈ ℝ₊^K
        ↓
Inference engine                                fixed-dictionary projection, no fitting
```

**Anything else is not V7.**

### The frozen-asset rule (newly made explicit; see §7)

> The V5/V6 frozen atlas may be used ONLY as a **baseline control**, a **benchmark
> comparator**, or a **reproducibility reference**. It must NOT become the foundation of the
> V7 learning architecture unless the specification explicitly says so.

This rule was implicit in the specification — `LEARNING_MODE_ARCHITECTURE.md` derives LSMs
from `X_c`, the balanced reference blocks, and never from `H` — but it was never written as a
single prohibition. Its absence is the proximate cause of the drift documented below.

---

## 2. Phase-by-phase compliance

### Phase 00 — Benchmark lock and reproducibility baseline

| # | Specification | Implemented | Match | Scientific consequence |
|---|---|---|---|---|
| 00.1 | Freeze canonical preprocessing (450–1800, 2 cm⁻¹, asls/savgol/L2) | Frozen, verified against the atlas manifest | ✅ | none |
| 00.2 | Freeze the V6.3 evaluation ontology | Adopted (16 fine / 6 broad), decision recorded | ✅ | none |
| 00.3 | Canonicalise molecule IDs | 167 surface forms → 154 canonical IDs, 13 merges audited | ✅ | none |
| 00.4 | Define replicate groups | `(canonical_id, excitation)` ratified with a comparison table | ✅ | none |
| 00.5 | Define + freeze quality metadata | `v7_q_v2` frozen before Phase 01 | ✅ | none |
| 00.6 | Freeze the chemical-family partition, resolving 3 named problems | All three resolved; rationale per class | ✅ | none |
| 00.7 | Freeze analyte-grouped CV splits | 5 folds, grouped by `canonical_id`, 3 leakage checks false | ✅ | none |
| 00.8 | Freeze evaluation metrics | `v7_harness_v1`, adopted from V6.3 | ✅ | none |
| 00.9 | Reproduce the atlas control | **Level 3** — basis refitted from raw, max abs diff 0.0 | ✅ | none |
| 00.10 | Freeze provisional success criteria | Frozen, S-01 ≥ 0.7507 pinned | ✅ | none |
| 00.11 | Output: **dataset role map** (`dataset_role_map_v7.csv`) | **NOT PRODUCED** | ❌ | **Low.** Provenance of which dataset plays which role is recoverable from the corpus card and `frozen_dependency_graph_v1.csv`, but not in the specified single artefact. |
| 00.12 | Output: **ontology table** (`evaluation_ontology_v7.csv`) | Produced under a different name (`chemical_partition_v1.csv` + `partition_rationale_v1.json`) | ⚠ | **None scientifically.** Content is complete; only the artefact name deviates. |

**Phase 00 verdict: COMPLIANT.** Two documentation-level gaps (00.11, 00.12), neither
affecting any scientific result. Notably, the Phase-00 manifest does **not** claim either
artefact — no false claim was made; the specified output list was simply not fully honoured.

**Required corrections:** produce `dataset_role_map_v7.csv`; either emit
`evaluation_ontology_v7.csv` or amend the plan to name the artefacts that exist. Both are
non-blocking and are carried into the Phase-01 rebuild.

---

### Phase 01 — as implemented at commit `f14e05b` / `0904d66`

| # | Specification (plan Phase 01 + Phase 02 = the user's "true Phase 01") | Implemented | Match | Scientific consequence |
|---|---|---|---|---|
| 01.1 | **Input: balanced canonical reference molecules** | **Input was the frozen 24-component atlas `H`** | ❌ **FAIL** | **Critical — see §3** |
| 01.2 | Compare 8 reference-construction arms (A, B, B-uniform, C×5) | **Not attempted** | ❌ FAIL | The class-imbalance correction that motivates V7 was never applied |
| 01.3 | Split the balanced references into chemistry classes | **Not attempted** — components were split, not references | ❌ FAIL | No class-local dataset ever existed |
| 01.4 | **Independent class-local NMF** `X_c ≈ W_c H_c` per class | **No NMF was fitted at all** | ❌ FAIL | Capacity was never reallocated; the V5 allocation was inherited wholesale |
| 01.5 | Adaptive `k_c` per class, `1 ≤ k_c ≤ ⌊n_c/2⌋`, smallest-on-Pareto-plateau | `n_motifs` chosen per **component** by silhouette | ❌ FAIL | The quantity swept was not `k_c`; the constraint is not even well-defined over components |
| 01.6 | Repeated fits + Hungarian alignment + recurrence stability | Jackknife co-assignment stability | ⚠ PARTIAL | A defensible stability estimate, but of cluster membership, not of a fitted basis |
| 01.7 | LSM typing: class-shared / subfamily / molecule-discriminating | **Not implemented** | ❌ FAIL | Phase 03 needs this typing to know what may be merged and what must not |
| 01.8 | Anchor route for classes with `n_analytes < 2` | **Not implemented** | ❌ FAIL | Rare chemistry has no path into the representation |
| 01.9 | Per-class source/excitation composition report | Measured only indirectly, via cross-source ARI | ⚠ PARTIAL | R-16 was observed but not reported per class as specified |
| 01.10 | Class-prior bias test (R-01) | **Not implemented** | ❌ FAIL | Untested |
| 01.11 | Output: one LSM dictionary **per class** | One motif set **per atlas component** | ❌ FAIL | Wrong indexing set; downstream contracts break |
| 01.12 | Determinism, provenance, manifests, no frozen-asset modification | Fully satisfied | ✅ | none |

**Phase 01 verdict: NON-COMPLIANT.** Nine hard failures, two partials, against the approved
architecture.

---

## 3. Where the drift happened, and what it costs mathematically

### The substitution

| | Specification | What was built |
|---|---|---|
| Object decomposed | balanced reference matrix `X_c ∈ ℝ₊^{n_c×676}` | frozen basis row `h_k ∈ ℝ₊^{676}` |
| Operation | `X_c ≈ W_c H_c` — a **fit** | band-profile clustering — a **partition of an existing vector's support** |
| Indexing set | chemical class `c` (16 of them) | atlas component `k` (24 of them) |
| Degrees of freedom | `k_c × 676` newly learned per class | **zero** — no parameter is learned |
| Capacity allocation | independent per class | inherited from the V5 global fit |

### Why this is not a small difference

**1. The V5 objective is still in force.** The frozen basis `H` was obtained by
`min ‖X − WH‖²_F` over all 375 spectra with equal row weight. That is exactly **Strategy A**,
the control arm, and it is the specific mechanism limitation L-01 identifies as the root
cause: capacity allocated by spectrum count, so dense classes crowd out rare ones. Building
Phase 01 on `H` means the entire V7 correction — one molecule, one reference unit; capacity
per class — was **never applied**. The implemented layer inherits the bias it was designed to
remove.

**2. The motif layer is bounded by its parent, by construction.** Every motif satisfies
`0 ≤ m ≤ h_k` pointwise, so
`span{motifs} ⊆ span{H}`. Formally: the decomposition can only redistribute mass the frozen
atlas already placed. The Phase-01 conservation check makes this precise and was reported as a
virtue — attributed evidence equals atlas activation to 2.2 × 10⁻¹⁶ — but it is equally the
proof of the ceiling. **A chemistry absent from all 24 components cannot be recovered by any
decomposition of those 24 components.** A class-local NMF has no such bound: it fits new basis
vectors in `ℝ₊^676` from the class's own data.

**3. The observed failure mode is the predicted one.** Purines resolved to one motif at 0.33
purity; pyrimidines to none. The control experiment attributed this to corpus scarcity (5
purines, 3 pyrimidines). That is *a* cause, but it is confounded with the architectural one:
purine chemistry receives no dedicated capacity in `H` because the V5 objective never gave it
any. **The control experiment cannot distinguish "the corpus is too thin" from "the frozen
basis never modelled this chemistry".** Only a class-local fit can separate those, because it
gives purine chemistry its own `k_c` regardless of how many protein spectra exist.

**4. Downstream contracts break.** `DATA_CONTRACTS.md` C-05 specifies
`lsm_registry_v1.json` keyed by **class**, with `k_c` and per-class source composition.
The implemented registry is keyed by **component**. Phase 03 pools LSMs across classes and
builds a similarity graph whose sixth edge feature is provenance overlap *with within-class
overlap discounted* — undefined when motifs have no class.

### How it happened

The Phase-01 brief specified "deterministic decompositions of individual atlas components".
That is unambiguous, and it contradicts the architecture documents. The correct response was
to **stop and raise a deviation report before writing code**. Instead the divergence was noted
in a docstring and in §2 of the report, and the implementation proceeded — which is how a
brief silently overrode an approved design. The report's §9 asked for the collision to be
resolved *after* the fact; by then the work was done and pushed.

**This is the failure mode the new standing invariant (§7) exists to prevent.**

---

## 4. Status decision for the implemented Phase 01

**Classification: (B) — exploratory control experiment built on the frozen atlas.**

It is not the canonical V7 Phase 01 and must not be treated as one. It is, however, a
legitimate and useful experiment: it establishes how much chemical resolution is recoverable
from the *existing* atlas with **no fitting at all**, which is precisely the control a
class-local rebuild must beat.

**Action: preserve everything, relabel, and move out of the phase sequence.**

| Item | From | To |
|---|---|---|
| Results tree | `results/v7_rebuild/phase01/` | `results/v7_rebuild/control_experiments/frozen_atlas_decomposition/` |
| Report | `PHASE_01_REPORT.md` | `CONTROL_EXPERIMENT_frozen_atlas_decomposition.md` |
| Package | `src/gaira/v7/lsm/` | `src/gaira/v7/atlas_decomposition/` |
| Tests | `tests/test_v7_phase01.py` | `tests/test_v7_control_atlas_decomposition.py` |
| Term used for its outputs | "Local Spectral Motif (LSM)" | **"Atlas Component Substructure (ACS)"** |

The rename of the *object* matters as much as the rename of the files: `LSM` is reserved by
`TERMINOLOGY_AND_DEFINITIONS.md` for a row of `H_c`, and leaving two different objects sharing
that name would corrupt every downstream document.

**Nothing is deleted.** All 98 substructures, 9 figures, validation tables and the report are
retained, with a banner stating what they are and are not.

---

## 5. Phase renumbering

The user's canonical architecture merges what the plan called Phase 01 (balanced references)
and Phase 02 (class-local NMF → LSMs) into a single **true Phase 01**. Adopted:

| Plan (original) | Canonical (adopted) |
|---|---|
| Phase 01 — balanced reference construction | **Phase 01, Stage 1** |
| Phase 02 — Local Spectral Motif construction | **Phase 01, Stage 2** |
| Phase 03 — Consensus Spectral Motifs | **Phase 02** |
| Phase 04 → 09 | shift down by one |

`plan/GAIRA_V7_REBUILD_PLAN.md` and `plan/PHASE_DEPENDENCY_MAP.md` are updated to match. All
gates from both original phases are carried into the merged phase; none is dropped.

---

## 6. Required corrections — checklist

| # | Correction | Status |
|---|---|---|
| C-1 | Reclassify the implemented Phase 01 as a control experiment; relabel files, report, package and object name | ✅ done |
| C-2 | Free the term `LSM` for the specification's object | ✅ done |
| C-3 | Implement the true Phase 01: balanced references → class split → independent class-local NMF → LSMs | ✅ done |
| C-4 | Adaptive `k_c` per class by the pre-registered smallest-on-Pareto-plateau rule | ✅ done |
| C-5 | LSM typing (class-shared / subfamily / molecule-discriminating) | ✅ done |
| C-6 | Anchor route for classes with too few analytes | ✅ done |
| C-7 | Per-class source/excitation composition (R-16) | ✅ done |
| C-8 | Class-prior bias test (R-01) | ✅ done |
| C-9 | Produce the missing `dataset_role_map_v7.csv` (00.11) | ✅ done |
| C-10 | Emit `evaluation_ontology_v7.csv` (00.12) | ✅ done |
| C-11 | Correct the residual `R-14` → `R-16` citation in the rebuild plan | ✅ done |
| C-12 | Adopt the phase renumbering in the plan and dependency map | ✅ done |
| C-13 | Add the frozen-asset rule and the pre-implementation architecture invariant to the governing documents | ✅ done |

---

## 7. Standing rules added as a result of this audit

Both are now written into `context/SCIENTIFIC_DESIGN_PRINCIPLES.md` as binding principles and
are enforced by `tests/test_v7_architecture_compliance.py`.

### P-15 — The frozen atlas is a control, never a foundation

> The V5/V6 frozen atlas may be used ONLY as a baseline control, a benchmark comparator, or a
> reproducibility reference. It must NOT become the foundation of the V7 learning
> architecture unless the specification explicitly says so.

### P-16 — Architecture check before implementation

> Before implementing any phase, re-read every V7 architecture document and verify that the
> phase to be implemented matches the approved architecture. If any discrepancy exists between
> the implementation brief and the approved architecture, **stop immediately**, generate an
> Architecture Deviation Report, and do not proceed until the discrepancy is resolved.
>
> A discrepancy noted in a docstring is not a resolution. Implementation must not begin.

### P-17 — Redraw the pipeline at the end of every phase

> Every phase report ends with a **Current V7 Pipeline** section: completed stages, remaining
> stages, the next phase's inputs and its outputs — drawn, not described. Had this been done
> after the control experiment, `24 atlas components → motifs` would have been visibly
> different from `class-local NMF → LSMs` on the page.

### Compliance table requirement

> Every phase report must end with an **Architecture Compliance** table
> (specification item · implemented? · evidence · PASS/FAIL). The phase gate opens only if
> every row is PASS. Otherwise the phase repeats.

---

## 8. Audit summary

| Phase | Specification followed? | Verdict |
|---|---|---|
| **00** Benchmark lock | 10 of 12 outputs exactly; 2 documentation gaps | ✅ **COMPLIANT** (corrections C-9, C-10 applied) |
| **01** (as implemented `f14e05b`) | 1 of 12 items | ❌ **NON-COMPLIANT** → reclassified as a control experiment |
| **01** (rebuilt from specification) | see `PHASE_01_REPORT.md` §Architecture Compliance | ✅ **COMPLIANT** |

**No scientific result has been lost.** The control experiment is preserved in full and now
serves the purpose it is actually fit for: the no-fitting baseline that the class-local
rebuild must beat.
