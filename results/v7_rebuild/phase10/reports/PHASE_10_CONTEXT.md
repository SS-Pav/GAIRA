# GAIRA V7 — Phase 10 Continuity Note

Written before any Phase 10 implementation, from the governing documents and the committed phase
outputs — not from memory. Every number below is quoted from a committed artefact.

**Sources read.** `context/GAIRA_V7_CONTEXT.md`, `context/GAIRA_V7_ARCHITECTURE_STATUS_AFTER_PHASE05.md`,
`context/TERMINOLOGY_AND_DEFINITIONS.md`, `context/SCIENTIFIC_DESIGN_PRINCIPLES.md`,
`architecture/GAIRA_V7_TARGET_ARCHITECTURE.md`, `architecture/DATA_CONTRACTS.md`,
`architecture/ARTIFACT_AND_MANIFEST_SPEC.md`, `architecture/INFERENCE_MODE_ARCHITECTURE.md`,
`plan/GAIRA_V7_REBUILD_PLAN.md`, `plan/SUCCESS_CRITERIA.md`,
`plan/VALIDATION_AND_DECISION_RULES.md`, `plan/RISK_REGISTER.md`, and the reports, audits and
decision gates of Phases 00–09.

---

## 1. The canonical inference path, as frozen after Phase 09

```
raw Raman spectrum
    ↓  canonical preprocessing        450–1800 cm⁻¹ · 2.0 step · 676 bins
                                      asLS → Savitzky–Golay(9,3) → L2
    ↓  frozen LSM projection          50 motifs, NNLS, non-negative      (diagnostic)
    ↓  frozen CSM projection          49 motifs, NNLS                    (CANONICAL)
    ↓  direct CSM grounded retrieval  cosine over 154 reference molecules
    ↓  Chemistry Evidence             16 axes, model D:A_max_idf λ=0.5
    ↓  calibration                    temperature, T = 0.4538
    ↓  radar / ordered evidence view
    ↓  confidence · audit · provenance
    ↓  interpretation report
```

Phase 10 changes none of this. It wraps it.

## 2. What is NOT on the canonical path, and where each exclusion was measured

| excluded | measured outcome | source |
|---|---|---|
| Themes (A-13) | chemistry on unseen molecules 0.855 → **0.405** | Phase 03/04, F-01 |
| BSV as canonical coordinate (A-14) | inherits F-01; effective rank 2.40 of K=4 | Phase 04, F-02 |
| Meta Components (A-15) | information retained 0.185; class top-1 **0.392** | Phase 04.5, F-03 |
| 11 declared grounded axes (A-16) | **0.664** vs CSM 0.845 | Phase 05, F-04 |
| Latent geometry / continuous coordinates | molecule Δ+0.016, McNemar **p = 0.180**, CI crosses zero | Phase 06.5 §9 |
| Clustering / cluster identity | 0 of 7 internal indices has an interior optimum across K=2…30 | Phase 06.5 A1 |
| Chemistry-aware reranking of retrieval | Δ collapsed to non-significance once both arms used the same 154-molecule bank (gate G7b) | Phase 08, decision **A** |
| BSV2 on the inference path | adopted as a **derived** description only; reads Chemistry Evidence, never feeds it | Phase 07, A-20 |
| SERS out-of-domain detection | AUROC 0.548 on real Ag-SERS; a non-negative Raman dictionary reconstructs SERS comfortably | Phase 04, F-05 → A-09 |

**The load-bearing result.** Chemistry-class accuracy on *unseen molecules*:
raw 0.608 → LSM 0.850 → **CSM 0.855** → 11 axes 0.664 → themes 0.405 → Meta 0.392.
Four independent attempts to build a layer above the CSM each lost information. Phase 10 ships
the layer where the information is.

## 3. Frozen numbers Phase 10 must not move

| quantity | value | source |
|---|---|---|
| molecule retrieval top-1 / top-3 / top-5 / top-10 | 0.6053 / 0.7627 / 0.7947 / 0.8107 | Phase 05 → 08 → 09 |
| MRR / nDCG@5 | 0.6870 / 0.7112 | Phase 09 |
| chemistry top-1, **held out** | **0.8507** (top-3 0.9760, macro-F1 0.8110) | Phase 09 V4 |
| chemistry top-1, in-sample | 0.9547 — a sanity check, never a performance claim | Phase 09 V4 |
| CSM mean explained variance | 0.8232 | Phase 09 V2 |
| replicate consistency (CSM) | 0.8927 | Phase 09 V2 |
| radar reproducibility | 0.9596 | Phase 09 V4 |
| robustness (35 conditions) | radar 0.9648 > chemistry 0.8890 > molecule 0.8106 | Phase 09 |

## 4. Frozen artefacts

| fingerprint | value |
|---|---|
| atlas | `09ed804a40836f4a05a91ba10900cded` |
| LSM registry | `208482d6f7178b5b8f16cace91be55b0` |
| CSM registry | `0b4aa550ccefed3edabdbde5bae11c8d` |
| Phase 05 engine | `20d8bd99ce71f45a125c6a2b1d719e51` |
| derived atlas content hash | `2e43ddcca7d3be41c5f9da016fb8277f` |

Phase 10 adds a second, independent layer: content digests of all **ten files** the engine reads,
recomputed and pinned in `gaira.v7.runtime.freeze`. See §6.

## 5. Scope, restated

- **Raman only.** No SERS, serum, plasma, EV, bacteria or tissue claim is licensed by any V7
  number. Those are extension points, not capabilities.
- **The radar is relative biochemical evidence** — not a concentration, not an abundance, not a
  mixture fraction. L2 normalisation destroys absolute scale in the first stage.
- **Retrieval returns reference analogues**, not identifications. Top-1 is 0.6053.
- **The 16 classes are a curated cut through a continuum** (Phase 06.5 A1), not a natural kind.
- **The engine cannot tell that the true molecule is absent from its bank.** The `unknown` and
  `outlier` warnings detect unexplained *spectra*, not unknown *molecules*. Phase 09 measured
  this: white noise reconstructs at CSM EV ≈ 0.61, above the 0.50 warning floor, and the flag
  fires on only 1 of 20 random spectra (audit C5b). **This limitation is carried into Phase 10
  verbatim and must be surfaced in every client.**

## 6. What Phase 10 adds, and what it must not

**Adds:** typed public schemas, a runtime service, input adapters, scientific input validation,
a FastAPI service, a Python SDK, a CLI, an MCP tool server, deterministic report generation, a
thin Streamlit client, and plugin *contracts* (not implementations) for future modalities and
sample contexts.

**Must not:** reimplement preprocessing, projection, retrieval, chemistry evidence, calibration,
confidence or audit metrics anywhere outside `gaira.v7.canonical.engine`; alter any frozen
artefact; introduce an LLM, a cloud dependency, or a network call; or let an unsupported
modality run silently as Raman.

## 7. Principles Phase 10 inherits

P-02 non-negativity · P-04 provenance is first-class · P-09 learning offline, inference is
projection only · P-10 Raman-only core · P-11 one canonical molecule = one reference unit ·
P-13 no threshold adjustment after seeing results · P-15 the V5 atlas is a control, never a
foundation · **P-18 stability without informativeness is not evidence.**

Two additions Phase 10 proposes for the runtime layer specifically:

- **P-19 (proposed) — one implementation of every scientific quantity.** If a number appears in
  two places it will eventually disagree in two places. Enforced by static tests over the API,
  MCP and Streamlit sources.
- **P-20 (proposed) — a surface may narrow what it shows, never what it computes.** Every
  client renders a subset of one `InferenceResult`; none may compute a value the engine did not
  return.
