# GAIRA V7 — Phase Dependency Map

What blocks what, what can run in parallel, and where a failure forces a return.

---

## 1. The critical path

> **Rewritten 2026-08-06 after Phase 05.** Canonical numbering. Archived phases are shown
> because they are still on the *provenance* path even though they are off the *inference* path.

```
     ┌──────────────────────────────────────────────────────────────┐
     │  ✔ PHASE 00 — Benchmark lock                                 │
     │  154 canonical molecules · 16 fine classes · 5 grouped folds │
     │  V5 control · metrics · FROZEN success criteria              │
     └───────────────────────────┬──────────────────────────────────┘
                                 ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  ✔ PHASE 01 — Balanced references + class-local LSMs         │
     │  16 independent fits · adaptive k_c · 50 stable LSMs         │
     └───────────────────────────┬──────────────────────────────────┘
                                 ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  ✔ PHASE 02 — Consensus Spectral Motifs                      │
     │  49 CSMs · 1 merge of 4 proposed · full provenance           │
     │  ★ THE CANONICAL REPRESENTATION (A-08)                       │
     └───────────┬──────────────────────────────────┬───────────────┘
                 │                                  │
                 ▼                                  ▼
     ┌───────────────────────────┐    ┌──────────────────────────────────┐
     │ ✔ PHASE 02.5 — geometry   │    │  ✔ PHASE 05 — CSM inference      │
     │ visualisation + prior     │    │  engine · retrieval · calibration│
     │ NEVER an inference path   │    │  rejection · provenance          │
     └───────────┬───────────────┘    └──────────────┬───────────────────┘
                 ▼                                   │
     ┌───────────────────────────────────┐           │
     │  ARCHIVED ON EVIDENCE             │           │
     │  03 themes  · 04 BSV · 04.5 Meta  │           │
     │  outputs preserved, off the path  │           │
     └───────────────────────────────────┘           │
                                                     ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  ▶ PHASE 06 — Chemistry Evidence Layer  (16-d)               │
     │  frozen map + calibrator · unassigned mass · R-01 control    │
     └───────────────────────────┬──────────────────────────────────┘
                                 │ DG-06 — must clearly exceed the
                                 │ archived 11-axis profile (0.664)
                                 ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  ▶ PHASE 07 — BSV2 Discovery                                 │
     │  hierarchical NMF over Chemistry Evidence ONLY · K sweep     │
     └───────────────────────────┬──────────────────────────────────┘
                                 │ DG-07 — pre-registered informativeness
                                 │ floor; discard is an expected outcome
                                 ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  ▶ PHASE 08 — Hierarchical Molecular Retrieval               │
     │  CSM + soft chemistry prior · prototype + residual           │
     └───────────────────────────┬──────────────────────────────────┘
                                 │ DG-08 — must beat direct cosine 0.605
                                 ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  ▶ PHASE 09 — V5 head-to-head              [DECISION]        │
     │  frozen Tier-1 criteria under v7_harness_v1, unadjusted      │
     └───────────────────────────┬──────────────────────────────────┘
                    ┌────────────┴────────────┐
                    ▼                         ▼
        ┌───────────────────────┐  ┌────────────────────────┐
        │ PHASE 10 — Chemistry- │  │ PHASE 11 — Targeted    │
        │ aware learning        │  │ corpus expansion       │
        │ (deferred)            │  │ (deferred)             │
        └───────────────────────┘  └────────────────────────┘
```

**Note the branch after Phase 02.** Phase 05 depends on Phase 02 directly, *not* on Phases 03–04.
That is the structural consequence of archiving the theme/BSV path: the inference engine reads
the CSM dictionary and nothing above it. Phase 02.5's geometry remains a dependency of the
archived Phase 03 only, plus visualisation.

---

## 2. Dependency table

| Phase | Hard prerequisites | Consumes | Produces | Status |
|---|---|---|---|---|
| 00 | — | raw corpus, V5 assets, V6.3 ontology | canonical IDs, partition, folds, metrics, baseline, frozen criteria | ✔ |
| 01 | 00 | canonical IDs, folds, quality | balanced references + 50 LSMs + registry | ✔ |
| 02 | 01 | LSMs, balanced references | 49 CSMs + registry + graph | ✔ |
| 02.5 | 02 | CSM dictionary | geometry, priors — analysis only | ✔ |
| ~~03~~ | ~~02, 02.5~~ | ~~CSMs~~ | ~~themes + `S`~~ | **ARCHIVED** |
| ~~04~~ | ~~03~~ | ~~CSMs, `S`~~ | ~~BSV frame; the six-level hierarchy measurement~~ | **ARCHIVED** |
| ~~04.5~~ | ~~02, 02.5~~ | ~~CSM activations~~ | ~~Meta Components~~ | **ARCHIVED — discarded** |
| 05 | **02** (not 03/04) | CSM dictionary + registry, folds | inference engine, reference bank, calibrator, rejection, 11 evidence axes | ✔ |
| **06** | **02, 05, 00** | CSM activations, folds, frozen criteria | chemistry-evidence map `E` + calibrator + R-01 control | ▶ next |
| **07** | **06** | Chemistry Evidence matrix **only** | BSV2 programme dictionary `P` | ▶ |
| **08** | **06**, 07 *(if DG-07 passed)* | CSM activations + Chemistry Evidence | hierarchical retrieval + prior weight `λ` | ▶ |
| **09** | **08, 00** | full V7 stack, V5 control, frozen criteria | replacement recommendation | ▶ |
| 10 | 09 pass | frozen V7 architecture | learning-gain attribution | deferred |
| 11 | 09 residuals | residual directions | targeted acquisitions | deferred |

Phase 09's double dependency on Phase 08 **and** Phase 00 is deliberate: the yardstick was fixed
before anything was built with it, and it is not adjusted afterwards (P-13).

Phase 08's dependency on Phase 07 is **conditional**. If DG-07 rejects BSV2, Phase 08 proceeds
from Chemistry Evidence directly and BSV2 is simply absent from the retrieval score.

---

## 3. What can run in parallel

| Parallel work | Within | Note |
|---|---|---|
| Per-class local decompositions | Phase 02 | classes are independent by construction — this is the main parallelism opportunity |
| `k_c` sweeps within a class | Phase 02 | independent fits |
| Reference-construction arms | Phase 01 | eight independent arms |
| Integration-method candidates | Phase 03 | all five evaluated on the same graph |
| `K` sweep | Phase 04 | independent per `K` |
| Figure and report generation | any | after that phase's artefacts exist |

**Not parallelisable across phases.** Each phase's output is the next phase's frozen input.
Starting Phase 03 on provisional Phase 02 output produces CSMs that will not match the final
LSMs, and the provenance chain — the thing V7 exists to get right — breaks silently.

---

## 4. Failure and backtrack paths

| Failure | Detected in | Return to | Action |
|---|---|---|---|
| Alias or replicate leakage | 00 | 00 | fix canonicalisation; re-cut splits |
| `unknown` class unresolved | 00 | 00 | assign chemistry or exclude from partitioning |
| No reference strategy beats the control | 01 | 01 → 02 | proceed with control A, document that the balancing hypothesis was not supported at row level; Strategy D (class partitioning) is still tested in 02 |
| A class yields no stable LSM | 02 | 02 | route to anchor mechanism; **never** duplicate spectra (P-11) |
| Class-prior bias dominates a local fit | 02 | 00 | revisit the partition for that class |
| A class is source-confounded | 02 | 02 | document; consider excluding the class from CSM integration |
| Motif proliferation (too many LSMs) | 02 | 02 | tighten stability threshold; lower `k_c` ceiling |
| Local dictionaries do not integrate | 03 | 02 | revisit `k_c` and stability; if it persists, Strategy D has failed and 01's control arm becomes the fallback architecture |
| Community structure is threshold-artefactual | 03 | 03 | widen the sweep; switch to a threshold-free method |
| Meta-NMF erases discriminating structure | 03 | 03 | reject meta-NMF; use graph or hybrid |
| Excessive singletons | 03 | 02 | LSM stability threshold was too loose |
| Themes not chemically coherent at any `K` | 04 | 03 | CSMs are not chemically coherent; revisit integration |
| Theme layer adds nothing over CSMs | 04 | 04 | report honestly; consider shipping CSM-level BSV with `K = M` |
| BSV axes highly correlated | 05 | 04 | reduce `K`; effective rank must be reported either way |
| Non-determinism at inference | 06 | 06 | trace and remove; this is a blocking bug |
| Fitting found in the inference path | 06 | 06 | blocking bug |
| Success criteria not met | 07 | — | **stop. Document the negative result. Retain the V5 atlas.** (P-13) |

The last row is the one that matters most. The plan has a defined path for V7 failing, and
that path does not include adjusting the criteria.

---

## 5. Gate summary

| Phase | Gate in one line |
|---|---|
| 00 | No leakage; baseline reproduced; splits deterministic; criteria frozen |
| 01 | Rule pre-registered; no supervision; balance improved without fidelity loss |
| 02 | Every LSM stable; `k_c` justified per class; rare classes routed explicitly |
| 03 | Provenance complete; `M` justified; method chosen on evidence; singletons flagged |
| 04 | Chemistry-only themes; soft membership; `K` justified; value over CSM shown |
| 05 | Deterministic; absolute ≠ delta; uncertainty propagated; effective rank reported |
| 06 | No inference-time fitting; batch independent; clean clone works; fingerprint verified |
| 07 | **Pre-registered success criteria met, or V5 is retained** |
| 08 | Held-out gains without loss of interpretability |
| 09 | Every addition traced to a measured residual direction |

---

## 6. Estimated sequencing

Ordering and relative effort only — no calendar commitment.

| Phase | Relative effort | Main risk to the schedule |
|---|---|---|
| 00 | **high** | canonicalisation and partition decisions are fiddly and consequential |
| 01 | medium | eight arms × stratified reporting |
| 02 | **high** | per-class sweeps × repeated fits — the compute-heaviest phase |
| 03 | **high** | five integration methods × threshold sweeps; the hardest *methodological* phase |
| 04 | medium | `K` sweep + admissibility review |
| 05 | medium | uncertainty propagation is the fiddly part |
| 06 | medium | integration and invariant testing |
| 07 | **high** | full evaluation + waterfall + head-to-head |
| 08 | deferred | — |
| 09 | deferred | — |

Phases 00 and 03 deserve the most care: 00 because every later number depends on it, and 03
because it is where Strategy D's central bet — that locally-fitted dictionaries can be
reintegrated into one comparable coordinate system — is either vindicated or not.
