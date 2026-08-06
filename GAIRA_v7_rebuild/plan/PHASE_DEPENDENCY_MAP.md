# GAIRA V7 — Phase Dependency Map

What blocks what, what can run in parallel, and where a failure forces a return.

---

## 1. The critical path

```
     ┌──────────────────────────────────────────────────────────────┐
     │  PHASE 00 — Benchmark lock                                   │
     │  canonical IDs · replicate groups · quality · class partition│
     │  CV splits · metrics · V5 control · frozen success criteria  │
     └───────────────────────────┬──────────────────────────────────┘
                                 │ gates: no leakage, baseline reproduced,
                                 │        splits deterministic, criteria frozen
                                 ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  PHASE 01 — Balanced reference construction                  │
     │  select from {A control, B weighted, C prototypes}           │
     └───────────────────────────┬──────────────────────────────────┘
                                 │ gates: rule pre-registered, no supervision,
                                 │        balance improved without fidelity loss
                                 ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  PHASE 02 — Local Spectral Motifs                            │
     │  per-class decomposition · adaptive k_c · stability selection │
     └───────────────────────────┬──────────────────────────────────┘
                                 │ gates: stability met, k_c justified,
                                 │        rare classes routed, class bias tested
                                 ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  PHASE 03 — Consensus Spectral Motifs                        │
     │  similarity graph · method comparison · CSM derivation        │
     └───────────────────────────┬──────────────────────────────────┘
                                 │ gates: provenance complete, M justified,
                                 │        method chosen on evidence, singletons flagged
                                 ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  PHASE 04 — Biochemical themes                               │
     │  soft sparse membership S · K selection                       │
     └───────────────────────────┬──────────────────────────────────┘
                                 │ gates: chemistry only, soft membership,
                                 │        K justified, value over CSM shown
                                 ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  PHASE 05 — Biochemical State Vector                         │
     │  absolute BSV · normalisation · uncertainty · OOD · eff. rank │
     └───────────────────────────┬──────────────────────────────────┘
                                 │ gates: deterministic, absolute/delta distinct,
                                 │        uncertainty propagated, eff. rank reported
                                 ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  PHASE 06 — End-to-end engine integration                    │
     │  one inference path · frozen atlas bundle · invariant checks  │
     └───────────────────────────┬──────────────────────────────────┘
                                 │ gates: no inference fitting, batch independent,
                                 │        clean clone works, fingerprint verified
                                 ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  PHASE 07 — Full in-domain Raman validation   [DECISION]     │
     │  every layer · waterfall · head-to-head vs V5                 │
     └───────────────────────────┬──────────────────────────────────┘
                                 │ gate: pre-registered success criteria met
                    ┌────────────┴────────────┐
                    ▼                         ▼
        ┌───────────────────────┐  ┌────────────────────────┐
        │ PHASE 08 — Chemistry- │  │ PHASE 09 — Targeted    │
        │ aware learning        │  │ corpus expansion       │
        │ (deferred)            │  │ (deferred)             │
        └───────────────────────┘  └────────────────────────┘
```

---

## 2. Dependency table

| Phase | Hard prerequisites | Consumes | Produces |
|---|---|---|---|
| 00 | — | raw corpus, V5 assets, V6.3 ontology | canonical IDs, replicate groups, quality, partition, splits, metrics, baseline, frozen criteria |
| 01 | 00 | canonical IDs, replicate groups, quality, splits | balanced reference matrix + selected strategy |
| 02 | 01 | balanced references, class partition, splits | LSM dictionaries + registry |
| 03 | 02 | LSMs, balanced references (for activations) | CSM dictionary + registry + graph |
| 04 | 03 | CSMs, CSM registry, splits | theme registry + membership `S` |
| 05 | 04 | CSMs, `S`, balanced references | BSV reference frame, OOD, uncertainty, vis transform |
| 06 | 05 | all frozen layers | V7 engine + atlas bundle + manifest |
| 07 | 06, **00** (splits + metrics + criteria) | V7 engine, V5 control | validation report + replacement recommendation |
| 08 | 07 pass, frozen candidate | frozen V7 architecture | learning-gain attribution |
| 09 | 07 residual analysis | residual directions | targeted acquisitions |

Note Phase 07's double dependency: it needs Phase 06's engine **and** Phase 00's frozen
splits, metrics, and criteria. That is deliberate — the yardstick is fixed before anything is
built with it.

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
