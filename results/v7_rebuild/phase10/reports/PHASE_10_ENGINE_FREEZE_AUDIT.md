# GAIRA V7 — Phase 10 Engine Freeze Audit

**Result: 9 of 9 gates PASS. Maximum deviation from the Phase 09 validation: exactly 0.0.**

Every fingerprint below was **recomputed** from the committed tree by
`results/v7_rebuild/phase10/code/run_engine_freeze_audit.py`. None was copied from
documentation. No Phase 10 wrapper was implemented until this passed.

---

## 1. Frozen assets — content digests

`GAIRAEngine.load()` was instrumented to record every file it opens. It opens exactly ten, all
inside the repository, all git-tracked. Their MD5 content digests are now pinned in
`gaira.v7.runtime.freeze.EXPECTED_DIGESTS` and verified before the runtime serves anything.

| file (under `results/v7_rebuild/`) | digest |
|---|---|
| `phase00/tables/canonical_analytes_v1.csv` | `dabd2834db31804fa948f5d30ff0fd44` |
| `phase00/tables/chemical_partition_v1.csv` | `0285392b5a70f55f4938344462486d45` |
| `phase01/PHASE_STATE.json` | `c66f7304b08aa6dce8415ca09c8a600b` |
| `phase01/artifacts/balanced_references_v1.npz` | `06fb6b7f2f58746023c77473c54f04d0` |
| `phase01/artifacts/lsm_dictionary_v1.npz` | `9d4bafe596e390d1ed0cd4eeecb50b6b` |
| `phase02/artifacts/csm_dictionary_v1.npz` | `3692ad772d661273c183fb23cf587c72` |
| `phase02/artifacts/csm_registry_v1.json` | `f75bce02c75747507034cd235ef2e9eb` |
| `phase05/PHASE_STATE.json` | `395e9abb425eab6118bdc8c89031827b` |
| `phase06/artifacts/chemistry_evidence_calibrator_v1.json` | `c9c6e8068d6116cbd22306addea24ac2` |
| `phase06/artifacts/chemistry_evidence_model_v1.json` | `0b387f2b26a16710e2436cb9e4d7865b` |

### Why this layer was added

Phase 09 verifies four **declared** fingerprints: values written *inside* `PHASE_STATE.json` and
`csm_registry_v1.json` by the phases that produced them. That check answers *"did the producing
phase claim this artefact?"* It does **not** answer *"is the file on disk still the file that
phase wrote."* A dictionary could be edited in place and the declared fingerprint would not move,
because the declared fingerprint lives in a different file.

Phase 10 pins the content of every file instead. Both layers now run: the declared fingerprints
(Phase 09, on engine load) and the content digests (Phase 10, before the runtime serves). The two
`PHASE_STATE.json` files are pinned deliberately — re-running an upstream phase rewrites them, and
that *should* invalidate the freeze.

## 2. Declared fingerprints — recomputed and matched

| term | recomputed | expected | |
|---|---|---|---|
| **Scientific Atlas Fingerprint** | `09ed804a40836f4a05a91ba10900cded` | same | ✓ |
| LSM registry | `208482d6f7178b5b8f16cace91be55b0` | same | ✓ |
| CSM registry | `0b4aa550ccefed3edabdbde5bae11c8d` | same | ✓ |
| Phase 05 engine | `20d8bd99ce71f45a125c6a2b1d719e51` | same | ✓ |
| **Frozen Runtime Content Hash** | `2e43ddcca7d3be41c5f9da016fb8277f` | — | recomputed at load |

The first four are the *declared* fingerprints the engine verifies on load. The fifth is derived
at load time from those four together with the reference-molecule set, the chemistry model
configuration and the calibrator, so it identifies the runtime as a whole. Terminology is defined
once in `PHASE_10_ARCHITECTURE.md` §4 and used identically in every Phase 10 document.

## 3. Engine version and shape

```
GAIRAEngine(atlas=2e43ddcca7d3…, 50 LSMs, 49 CSMs, 154 molecules, 16 chemistry axes)
```

| | |
|---|---|
| load time | 1.03 s (cold, laptop) |
| mean single-spectrum inference | 1.9 ms |
| canonical ontology ordering | matches the frozen `CLASS_ORDER` exactly |
| deterministic on repeat | yes, byte-identical `to_dict()` |
| chemistry model | `D:A_max_idf:lam0.5` (Phase 06 selection) |
| calibrator | temperature, `T = 0.4538` |

## 4. Numerical regression against Phase 09

Reproduced across **all 375 corpus spectra** through the engine's own public API.

| metric | Phase 10 | Phase 09 baseline | deviation |
|---|---|---|---|
| molecule top-1 | 0.6053 | 0.6053 | **0.00e+00** |
| molecule top-3 | 0.7627 | 0.7627 | **0.00e+00** |
| molecule top-5 | 0.7947 | 0.7947 | **0.00e+00** |
| molecule top-10 | 0.8107 | 0.8107 | **0.00e+00** |
| MRR | 0.6870 | 0.6870 | **0.00e+00** |
| chemistry top-1 (in-sample) | 0.9547 | 0.9547 | **0.00e+00** |
| CSM mean explained variance | 0.8232 | 0.8232 | **0.00e+00** |

Not "within tolerance" — identical.

## 5. Golden fixtures

Six cases stored at `tests/fixtures/v7_phase10/golden_inference_v1.json`, each with a digest over
the canonical JSON of the full `InferenceReport`. These are the regression anchors for every
Phase 10 surface.

| case | molecule | CSM EV | predicted class | digest |
|---|---|---|---|---|
| `high_confidence` | tubulin | 0.9987 | peptide_protein | `4f3e722278bc…` |
| `molecule_wrong_chemistry_right` | lactalbumin | 0.9779 | peptide_protein | `f291528e95e1…` |
| `low_explained_variance` | pyruvate | 0.2087 | carboxylic_acid_metabolite | `9bc5ea4f96b5…` |
| `ambiguous_chemistry` | vaccenic acid | 0.9683 | phospholipid_sphingolipid | `2c6841b8cb0a…` |
| `large_class_exemplar` | albumin | 0.9306 | peptide_protein | `acf4c856c186…` |
| `synthetic_noise_control` | — | 0.6083 | — | confidence 0.3433 |

The noise control is the most informative of the six. It reconstructs at **CSM EV 0.6083** —
above the 0.50 `unknown` floor — and its `unknown_warning` fires only because its retrieval
margin is below 0.01, not because the atlas recognised it as structureless. Confidence separates
it correctly (0.3433 against a corpus mean of 0.803). This is Phase 09 audit item C5b reproduced
as a permanent fixture.

## 6. A defect found in this audit, and fixed

The first draft of this script **hand-rolled the leave-one-out retrieval loop** rather than
calling the frozen Phase 08 modules. It dropped every spectrum of the query molecule instead of
only the query spectrum, so the true answer was never in the bank and it reported **top-1 of
0.0000** — a catastrophic apparent regression that was entirely an artefact of the audit.

Reimplementing scientific logic is exactly what Phase 10 forbids, and the freeze audit was not
exempt from its own rule. It now calls `gaira.v7.retrieval.models.build_bank` / `score_B` and
`gaira.v7.retrieval.evaluation.split_a_metrics`, and deviation dropped to exactly 0.0.

**This is why P-19 exists.** The very first thing Phase 10 wrote reproduced the failure mode the
phase was created to prevent, within twenty minutes of starting. The static tests in
`tests/test_v7_phase10_parity.py` now enforce it across the API, MCP and Streamlit sources.

## 7. The public scientific contract

What the frozen engine guarantees, exactly:

| guarantee | enforcement |
|---|---|
| Fingerprints verified on load; `FrozenArtifactError` on mismatch | `GAIRAEngine.load()` |
| Content digests verified before serving; `FrozenAssetError` on mismatch | `runtime.freeze.verify()` |
| No mutable state; frozen dataclasses out | `@dataclass(frozen=True)` |
| Deterministic — identical alone or in a batch, identical on repeat | verified, gate FA4 |
| Every retrieval score reconciles to its components within 1e-9 | asserted per candidate |
| Non-negative at every layer (P-02) | NNLS throughout |
| No random number drawn at inference | static test |
| Reads only committed repository assets | instrumented, gate FA9 |

## 8. Known limitations carried forward unchanged

1. **No validated open-set detection.** The engine cannot know the true molecule is absent from
   its 154-molecule bank. White noise reconstructs at EV ≈ 0.61. Confidence is the usable signal;
   the flag is not.
2. **In-sample chemistry figures describe the shipped fit**, not expected performance. Quote
   0.8507.
3. **Molecule top-1 is capped at 0.819** by corpus structure — 66 of 154 molecules have a single
   spectrum, so 68 of 375 queries are unretrievable by construction.
4. **R-01 (class-prior bias) remains OPEN.** Class sizes range from 3 to 80 spectra.
5. **Pure Raman reference spectra only.** Every applied regime is unmeasured in V7.
6. **The 16 classes are a reporting convention**, not a discovered structure (Phase 06.5 A1).

## 9. Gates

| gate | status |
|---|---|
| FA1 every frozen asset present and content-pinned | **PASS** |
| FA2 declared fingerprints recomputed and match | **PASS** |
| FA3 canonical ontology ordering unchanged | **PASS** |
| FA4 engine deterministic on repeat | **PASS** |
| FA5 golden fixtures stored for six representative cases | **PASS** |
| FA6 Phase 09 retrieval reproduced within 1e-6 | **PASS** (0.0) |
| FA7 Phase 09 chemistry reproduced within 1e-6 | **PASS** (0.0) |
| FA8 no frozen artefact written or modified | **PASS** |
| FA9 engine reads only committed repository assets | **PASS** |

Phase 10 implementation is cleared to proceed.
