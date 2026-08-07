# GAIRA V7 — Phase 10 Validation Report

**17 of 17 gates PASS. Cross-surface parity: max |Δ| = 0.0. Phase 09 science: max deviation 0.0.**

---

## 1. Engine freeze audit (9 gates)

Full detail in `PHASE_10_ENGINE_FREEZE_AUDIT.md`.

| | |
|---|---|
| frozen assets pinned and verified | 10 / 10 |
| declared fingerprints recomputed | 4 / 4 match |
| canonical ontology ordering | unchanged |
| deterministic on repeat | yes |
| golden fixtures | 6 cases stored |
| Phase 09 retrieval reproduced | deviation **0.00e+00** |
| Phase 09 chemistry reproduced | deviation **0.00e+00** |

## 2. Cross-surface parity (Step 13)

Twelve locked spectra — a high-confidence hit, a chemistry-right/molecule-wrong case, the
lowest-EV spectrum in the corpus, an ambiguous class, four chemistry classes, three source
libraries, and a synthetic noise control — through **six** surfaces.

| comparison | identical |
|---|---|
| engine → runtime service | 12 / 12 |
| runtime service → Python SDK | 12 / 12 |
| Python SDK → FastAPI | 12 / 12 |
| Python SDK → Streamlit backend path | 12 / 12 |
| MCP → matched SDK call | 12 / 12 |

**60 comparisons, 0 divergent, maximum absolute difference 0.0** at a tolerance of 1e-12. Result
digests agree across the service, SDK, API and Streamlit paths for every case.

Compared per case: CSM activation (49 values), CSM explained variance, chemistry evidence (16),
calibrated probabilities (16), predicted class, all ten retrieval ranks and similarities, overall
confidence, the unknown flag, grid coverage, and score reconciliation.

## 3. Malformed input and scope (Step 13)

| | API | SDK | MCP |
|---|---|---|---|
| two-point spectrum rejected | ✓ 422 | ✓ `SpectrumRejected` | ✓ `can_run: false` |
| unsupported modality blocked | ✓ | ✓ | ✓ |

Also verified: length mismatch, unknown request field, empty body, over-length metadata, oversized
body (413), non-finite values, duplicate wavenumbers, descending axes, binary payloads, and
declared-but-unimplemented file formats.

## 4. Concurrency

16 requests over 8 threads reproduce the serial digests **exactly**. The engine holds no mutable
state and draws no random numbers, so this is structural rather than lucky.

## 5. Performance (Step 14) — laptop, measured

| | |
|---|---|
| engine load (cold) | 0.28 s |
| single-spectrum inference, median | **2.34 ms** (p95 3.11 ms) |
| 10 sequential | 0.021 s |
| 100 sequential | 0.218 s |
| API median / **overhead** | 3.95 ms / **1.61 ms** |
| MCP median / **overhead** | 5.30 ms / **2.97 ms** |
| report — PDF / JSON | 1.42 s / 0.002 s |
| peak RSS | 596 MB |

**Live Raman use is feasible.** At 2.3 ms per spectrum the engine sustains ~430 spectra/second
single-threaded; acquisition, not inference, is the bottleneck. No latency gate was invented
before measuring; the gates below were set afterwards at values these numbers clear comfortably.

## 6. Scientific validation (Step 17)

All 375 corpus spectra through the **Python SDK**, then the frozen Phase 08 retrieval modules.

| metric | Phase 10 runtime | Phase 09 | Δ |
|---|---|---|---|
| molecule top-1 | 0.605333 | 0.605333 | **0.00e+00** |
| molecule top-3 | 0.762667 | 0.762667 | **0.00e+00** |
| molecule top-5 | 0.794667 | 0.794667 | **0.00e+00** |
| molecule top-10 | 0.810667 | 0.810667 | **0.00e+00** |
| MRR | 0.687003 | 0.687003 | **0.00e+00** |
| chemistry top-1 (in-sample) | 0.954667 | 0.954667 | **0.00e+00** |
| CSM mean explained variance | 0.823196 | 0.823196 | **0.00e+00** |

Held-out chemistry (0.8507), robustness and calibration figures are quoted unchanged from Phase
09; Phase 10 re-ran the retrieval and representation arms rather than re-deriving the cross-
validated ones, because those depend on frozen fold assignments the runtime does not touch.

## 7. Test suite (Step 16)

**1436 passed, 1 skipped.** Phase 10 contributes 124 tests across three files.

| file | tests |
|---|---|
| `test_v7_phase10_runtime.py` | 39 — freeze ledger, adapters, validation, service, golden regression, reporting |
| `test_v7_phase10_surfaces.py` | 57 — API, MCP, SDK, CLI, plugin contracts, security |
| `test_v7_phase10_parity.py` | 28 — six-surface parity and the static architecture rules |

The single skip is pre-existing (`test_reproduce_foundation.py`, needs a raw V5 data root).

### Static architecture tests

Every surface is parsed with `ast` — not grepped — and must reference no scientific primitive
(`nnls`, `NMF`, `PCA`, `savgol_filter`, `cosine_similarity`, …) and import no scientific module
(`scipy.*`, `sklearn.*`, `gaira.v7.{lsm,csm,chemistry,retrieval,…}`). The Streamlit app
additionally may not import `gaira.v7.canonical`: it must go through the runtime, not reach past
it. `runtime/interpret.py` must import nothing at all.

## 8. Gates

| gate | status |
|---|---|
| FA1–FA9 engine freeze audit | **9 PASS** |
| P1 six surfaces produce identical scientific output | **PASS** (0 divergent of 60) |
| P2 result digests agree across surfaces | **PASS** |
| P3 malformed input rejected on every surface | **PASS** |
| P4 unsupported modality blocked on every surface | **PASS** |
| P5 concurrent API requests reproduce serial results | **PASS** |
| P6 Phase 09 science reproduced through the runtime | **PASS** (Δ = 0.0) |
| P7 single-spectrum inference is interactive (< 250 ms) | **PASS** (2.34 ms) |
| P8 API overhead under 100 ms | **PASS** (1.61 ms) |
