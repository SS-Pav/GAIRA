# GAIRA V7 — Target Architecture

The complete specification of what V7 is and what it is becoming. Companion documents give the
detail: `LEARNING_MODE_ARCHITECTURE.md` (how each learning stage works),
`INFERENCE_MODE_ARCHITECTURE.md` (the runtime path), `DATA_CONTRACTS.md` (schemas),
`ARTIFACT_AND_MANIFEST_SPEC.md` (what is frozen), `LIVE_DART_COMPATIBILITY.md` (trajectories).

> **Revised 2026-08-06, after Phase 05.** Phases 00, 01, 02, 02.5, 03, 04, 04.5 and 05 are
> complete. The architecture below is the one the accumulated evidence supports; the one this
> document previously specified is preserved in §1.1 as the **legacy architecture**, labelled
> and not deleted. Every change is evidenced in
> [`context/GAIRA_V7_ARCHITECTURE_STATUS_AFTER_PHASE05.md`](../context/GAIRA_V7_ARCHITECTURE_STATUS_AFTER_PHASE05.md),
> which is the authority for *why*; this document is the authority for *what*.

Numbering throughout is the **canonical** numbering adopted 2026-08-06 (Phase 01 merges the
plan's original Phases 01 and 02; everything after shifts down by one). See
`plan/GAIRA_V7_REBUILD_PLAN.md` for the mapping table.

---

## 1. Three architectures

V7 now carries three, and the distinction is deliberate. A path that was tried and did not work
is evidence, not embarrassment; deleting it would destroy the reason the current path is what it
is.

| | Status | Meaning |
|---|---|---|
| **Legacy** | archived | built, measured, retired on evidence. Outputs preserved and reproducible. Not on the inference path. |
| **Current** | active | implemented, validated, on the inference path today. |
| **Future** | planned | specified, not implemented. Each stage has a decision gate that can reject it. |

### 1.1 Legacy architecture — ARCHIVED

```
spectrum
   ↓
CSM  (49)
   ↓
soft biochemical themes           ← Phase 03 · archived A-13
   ↓
BSV = Sᵀc  (4 accepted themes)    ← Phase 04 · archived A-14
   ↓
[ Meta Components, K = 3 ]        ← Phase 04.5 · archived A-15 (discarded, never on the path)
   ↓
11 declared evidence axes         ← Phase 05 · archived A-16
```

**Why it was retired.** Class top-1 on molecules the atlas has never seen, measured on identical
frozen splits (Phase 04 `hierarchy_retrieval_v1.csv`, Phase 05 `robustness_summary_v1.csv`):

| layer | dim | class top-1, unseen molecule |
|---|---:|---:|
| raw spectrum | 676 | 0.608 |
| LSM | 50 | 0.850 |
| **CSM** | **49** | **0.855** |
| theme / BSV | 4 | 0.405 |
| Meta Components | 3 | 0.392 |
| 11 evidence axes | 11 | 0.664 |

Every layer above the CSM lost information. Three independent constructions — discovered by
archetypal analysis, discovered by non-negative factorisation, and declared from band
assignments with no fitting — reached the same result.

**What is preserved.** All of it: `results/v7_rebuild/phase03/`, `phase04/`, `phase04_5/`,
`phase05/` retain their reports, audits, figures, tables, artefacts and tests exactly as
committed. The theme registry, membership matrix `S`, BSV reference frame and evidence-axis map
remain readable, fingerprinted artefacts. They are simply not consulted at inference.

### 1.2 Current architecture — ACTIVE

What runs today, validated through Phase 05.

```
new Raman spectrum
   ↓  canonical preprocessing            450–1800 cm⁻¹ · 2.0 step · 676 bins · asLS → SG → L2
   ↓  NNLS onto the frozen CSM basis     nothing fitted
49-dimensional CSM activation vector     ← THE CANONICAL REPRESENTATION (A-08)
   ↓
   ├── analyte retrieval                 154 reference vectors · cosine · calibrated
   ├── chemistry-class inference         16 fine classes · 0.845 top-1 on unseen molecules
   ├── evidence profile                  11 declared axes (A-16, to be superseded by A-19)
   ├── provenance                        axis → CSM → LSM → molecule → spectra · 0 broken
   └── uncertainty                       residual · margin · entropy · open-set rejection
```

### 1.3 Future architecture — PLANNED

The target the evidence now points at. **Nothing below §1.3 is implemented.** Each arrow is
gated and each gate can send the phase back.

```
new Raman spectrum
   ↓  canonical preprocessing                                          [ACTIVE]
   ↓  NNLS onto the frozen CSM basis                                   [ACTIVE]
49-dimensional CSM representation                                      [ACTIVE, A-08]
   ↓  frozen chemistry-evidence map · learned offline in Phase 06
16-dimensional Chemistry Evidence                                      [PLANNED, A-19]
   │   probabilistic · non-negative · sums to ≤ 1 with an explicit
   │   unassigned mass · one coordinate per frozen fine-16 class
   ↓  hierarchical NMF over Chemistry Evidence ONLY · Phase 07
BSV2 — learned biochemical programmes                                  [PLANNED, A-20]
   │   K selected on a Pareto frontier · never on reconstruction alone
   ↓
hierarchical molecular retrieval                                       [PLANNED, A-21]
   │   inputs: CSM activation  +  Chemistry Evidence (as a soft prior)
   │   soft chemistry prior → class-conditioned retrieval →
   │   prototype + residual scoring → hierarchical ranking →
   │   top-k + confidence (+ conformal sets if justified)
   ↓
domain-context interpretation                                          [downstream, unchanged]
```

**Two structural rules distinguish this from the legacy path**, and both exist because of what
the legacy path did wrong:

1. **Chemistry Evidence sits beside CSM inference, never instead of it.** Molecular retrieval
   reads the CSM activation vector *and* uses Chemistry Evidence as a prior. The legacy path
   replaced the representation at each level and paid for it every time.
2. **BSV2 is derived from Chemistry Evidence only, not from CSM.** This is what makes BSV2 a
   different object from the archived Meta Components (A-15), which factorised motif usage.
   BSV2 factorises *chemistry co-occurrence*. Whether that difference is enough is exactly what
   Phase 07 tests, and DG-07 can reject it.

---

## 2. The two modes, and why the split is absolute

| | Learning mode | Inference mode |
|---|---|---|
| When | offline, during a build | live, per spectrum |
| Input | the pure-Raman grounding corpus | one new spectrum |
| Output | a frozen, versioned atlas | representation + evidence + retrieval + QC |
| Fitting | yes — this is where all fitting happens | **never** |
| Determinism | deterministic given the seed schedule | deterministic, full stop |
| Runs where | a build machine with the corpus | anywhere, from a clean clone, no lab volume |

The reason the line is absolute: **comparability**. Two spectra measured in different labs years
apart are comparable only if projected onto the *same fixed axes*. Any fitting at inference —
even re-fitting a PCA for a plot — makes the coordinate system depend on the batch, and the
comparison silently becomes meaningless.

---

## 3. Learning mode

```
Raw Raman grounding corpus
   375 spectra · 154 canonical molecules · 16 fine chemistry classes · pure Raman only
        │
        ▼
canonical preprocessing                                          ── UNCHANGED from V5
   asLS baseline → Savitzky–Golay → L2
   450–1800 cm⁻¹ · 2.0 cm⁻¹ step · 676 bins
        │
        ▼
canonical molecule IDs · replicate groups · quality · CV folds    ── PHASE 00 ✔ COMPLETE
   alias resolution · q per spectrum · 5 folds grouped by canonical_id
   frozen V5 control baseline · frozen success criteria
        │
        ▼
balanced reference construction                                   ── PHASE 01 ✔ COMPLETE
   one canonical molecule = one reference unit
        │
        ▼
chemical-family partition into 16 classes                         ── PHASE 00 freeze / PHASE 01 use
   an organisational prior, never supervision inside a fit
        │
        ▼
class-local non-negative decomposition · adaptive k_c             ── PHASE 01 ✔ COMPLETE
   16 independent fits · repeated under resampling and seed variation
        │
        ▼
50 stable Local Spectral Motifs (LSMs)                            ── PHASE 01 ✔ COMPLETE
   Hungarian alignment across runs → recurrence → stability threshold
   registry 208482d6f7178b5b8f16cace91be55b0
        │
        ▼
seven-feature Consensus Spectral Graph · null calibration          ── PHASE 02 ✔ COMPLETE
   significance sweep → threshold consensus → merge proposals → falsification
        │
        ▼
49 Consensus Spectral Motifs (CSMs)                               ── PHASE 02 ✔ COMPLETE
   1 merge accepted of 4 proposed · 48 singletons · full provenance
   dictionary 0b4aa550ccefed3edabdbde5bae11c8d
        │
        ├──▶ latent geometry of motif space                        ── PHASE 02.5 ✔ COMPLETE
        │       continuum, not islands · K = 2 hydrophobic/polar
        │       VISUALISATION AND PRIOR ONLY — never an inference path (A-10)
        │
        ├──▶ [ soft themes → BSV → Meta Components ]               ── ARCHIVED (A-13/14/15)
        │
        ▼
frozen projection engine                                          ── PHASE 05 ✔ COMPLETE
   reference bank (154) · similarity metric · calibrator ·
   rejection channels · evidence axes (A-16) · provenance
   engine 20d8bd99ce71f45a125c6a2b1d719e51
        │
        ▼
Chemistry Evidence map                                            ── PHASE 06 ▶ PLANNED
   frozen map  CSM ℝ₊^49 → Chemistry Evidence ℝ₊^16
   learned offline, molecule-grouped CV, calibrated, provenanced
        │
        ▼
BSV2 — biochemical programmes                                     ── PHASE 07 ▶ PLANNED
   hierarchical NMF over Chemistry Evidence only · K on a Pareto frontier
        │
        ▼
hierarchical molecular retrieval                                  ── PHASE 08 ▶ PLANNED
   CSM + Chemistry Evidence prior → class-conditioned ranking
        │
        ▼
frozen GAIRA V7 Atlas                                             ── bundle, one fingerprint
   preprocessing spec · LSM dictionaries · CSM basis · chemistry-evidence map ·
   BSV2 programmes · reference bank · calibrator · rejection thresholds ·
   provenance · manifest
```

### Where each earlier limitation is addressed

| Prior limitation | Addressed by | Status |
|---|---|---|
| L-01 objective counts spectra | Phase 01 balanced reference construction | ✔ resolved |
| L-02 stable but impure components | Phase 01 class-local decomposition + adaptive `k_c` | ✔ resolved |
| L-03 motifs borrow foreign mass | Phase 02 bottom-up CSMs with mandatory provenance | ✔ resolved |
| L-04 fine retrieval plateau | measured, not assumed — **Phase 06 under `v7_harness_v1`** | ▶ open (U-06) |
| L-05 ontology cleanup insufficient | the CSM layer replaced the overlay entirely | ✔ resolved |
| L-06 true representation failures | Phase 06 failure waterfall | ▶ open |
| L-07 thin chemistry | adaptive `k_c` + Phase 09 targeted expansion | ▶ partial |

---

## 4. Inference mode

### 4.1 Current (ACTIVE)

```
new Raman spectrum (wavenumbers + intensities + metadata)
        │
        ▼  canonical preprocessing — deterministic, same spec as the build
        ▼  fixed-dictionary NNLS:  c(x) = argmin_{c ≥ 0} ‖x − cᵀ·CSM‖²
CSM activation c(x) ∈ ℝ₊^49
        │
        ├─▶ analyte retrieval        cosine vs 154 reference vectors → top-k
        ├─▶ chemistry-class          argmax over reference classes → top-3
        ├─▶ evidence profile         11 declared axes (A-16)
        ├─▶ provenance               additive decomposition back to spectra
        └─▶ uncertainty              residual · margin · entropy · rejection score
        │
        ▼
domain-context interpretation — downstream of everything; NEVER feeds back upstream
```

### 4.2 Future (PLANNED)

```
CSM activation c(x) ∈ ℝ₊^49
        │
        ▼  frozen chemistry-evidence map (matrix multiply + frozen calibration)
Chemistry Evidence  e(x) ∈ ℝ₊^16,  Σ e_k ≤ 1,  unassigned mass reported
        │
        ├─▶ BSV2  b(x) = frozen NNLS of e(x) onto the frozen programme dictionary
        │
        └─▶ hierarchical molecular retrieval
                 soft chemistry prior from e(x)
                 → class-conditioned candidate set
                 → prototype + residual scoring against c(x)
                 → ranked top-k + calibrated confidence
```

Note what does **not** change: the projection is still NNLS onto a frozen dictionary, everything
after it is a frozen matrix multiply or a distance calculation, and no step depends on which
other spectra are in the batch.

### 4.3 Prohibited at inference — a closed list

- NMF fitting · PCA fitting · UMAP / t-SNE / any manifold learning
- clustering of any kind · graph community detection · ontology optimisation
- threshold tuning against the incoming batch
- **any operation whose result depends on which other spectra are in the batch**

That last item is the general principle; the others are instances of it.

### 4.4 Permitted at inference — a closed list

- canonical preprocessing (deterministic, per-spectrum)
- NNLS against a frozen dictionary
- multiplication by frozen matrices (chemistry-evidence map, BSV2 programmes, normalisation)
- application of a **frozen** linear transform for visualisation
- distance and similarity calculations against frozen reference objects
- application of a **frozen** calibrator
- uncertainty propagation through frozen linear maps
- trajectory updates (appending a new absolute vector to a sequence)

### 4.5 The PCA/UMAP boundary

A frozen PCA transform applied for plotting is **visualisation**, not inference. Its output is
never a canonical coordinate, never an input to interpretation, and never used for retrieval or
scoring. UMAP is not shipped in the atlas at all: it has no out-of-sample transform stable
enough to freeze. **No V7 document may describe PCA or UMAP as inference.**

---

## 5. Layer contracts

| Layer | Object | Space | Frozen at | Live operation | Status |
|---|---|---|---|---|---|
| Preprocessing | spec | — | Phase 00 | apply | ACTIVE |
| Balanced reference | rows | `ℝ₊^{375×676}` | Phase 01 | not used live | ACTIVE |
| LSM dictionary | `H_c` per class | `ℝ₊^{k_c×676}` | Phase 01 | optional evidence | ACTIVE |
| **CSM basis** | `CSM` | `ℝ₊^{49×676}` | Phase 02 | **NNLS projection** | **ACTIVE — canonical** |
| Motif geometry | `D`, embedding | — | Phase 02.5 | visualisation only | ACTIVE |
| Reference bank | `R` | `ℝ₊^{154×49}` | Phase 05 | similarity | ACTIVE |
| Calibrator | frozen params | — | Phase 05 | apply | ACTIVE |
| Rejection thresholds | scalars + channel stats | — | Phase 05 | compare | ACTIVE |
| ~~Membership `S`~~ | `S` | `ℝ₊^{49×4}` | Phase 03 | — | **ARCHIVED** |
| ~~BSV reference frame~~ | `μ`, `σ` | `ℝ^4` | Phase 04 | — | **ARCHIVED** |
| ~~Meta programmes `H`~~ | `H` | `ℝ₊^{3×49}` | Phase 04.5 | — | **ARCHIVED (discarded)** |
| ~~Evidence axis map `M`~~ | `M` | `ℝ₊^{49×11}` | Phase 05 | matrix multiply | **ACTIVE → to be superseded** |
| Chemistry-evidence map | `E` | `ℝ₊^{49×16}` | Phase 06 | matrix multiply + calibrate | PLANNED |
| BSV2 programmes | `P` | `ℝ₊^{K×16}` | Phase 07 | frozen NNLS | PLANNED |
| Retrieval prior | prior model | — | Phase 08 | apply | PLANNED |
| Domain context | rules | — | separate version | applied last | unchanged |

`D = 676`. `M = 49` was selected in Phase 02. The Chemistry Evidence dimension is **16**, fixed
by the frozen `v7_fine_16` evaluation ontology — it is not a free parameter. `K` for BSV2 is
selected in Phase 07 on a Pareto frontier.

---

## 6. Invariants — must hold at every layer

| Invariant | Check | Currently |
|---|---|---|
| **Non-negativity** | every basis, activation, evidence coordinate and programme ≥ 0 | ✔ |
| **Determinism** | identical input → byte-identical output, twice, two machines | ✔ Phase 05 G12 |
| **Provenance completeness** | every CSM resolves to LSMs → classes → analytes → sources | ✔ 3,133 chains, 0 broken |
| **No inference-time fitting** | static check: no `fit`, `fit_transform`, or RNG in the inference path | ✔ |
| **Batch independence** | a spectrum's output is identical alone and in a batch of 1000 | ✔ |
| **Frozen-package sufficiency** | inference runs on a clean clone with no lab volume | ✔ |
| **Fingerprint integrity** | the atlas fingerprint covers every layer and is verified on load | ✔ |
| **Domain isolation** | no domain object reachable from any pre-interpretation module | ✔ |
| **Absolute coordinates** | no canonical coordinate is ever computed as a difference | ✔ |
| **Informativeness before stability** (P-18) | no layer selected on a stability metric without clearing a pre-registered informativeness floor | ✔ enforced from Phase 04.5 onward |

---

## 7. Relationship to the V5 engine

The V5 engine (`src/gaira/engine/`) stays in production, unmodified, until Phase 06 delivers a
head-to-head result against the frozen Tier-1 criteria. V7's engine
(`src/gaira/v7/inference/`) is a **parallel implementation**, not an edit.

| V5 | V7 current | Change |
|---|---|---|
| `manifold_components.npz` (24×676 global NMF) | CSM basis (49×676) | class-local then consensus; same role and interface |
| `component_registry_v1.json` | LSM registry + CSM registry | two levels, richer provenance |
| `mss_motifs_v1.yaml` (curated overlay) | *no equivalent* | the overlay is replaced by the CSM layer itself |
| `component_theme_weights_v1.json` (24×13) | ~~membership `S`~~ → chemistry-evidence map `E` | archived, then replaced by a calibrated 16-d map |
| `biochemical_ontology_v2.yaml` (13 themes) | ~~theme registry~~ → frozen `v7_fine_16` | archived; the label space is now the frozen evaluation ontology |
| `reference_normalization_v1.json` | reference bank + calibrator | same role, calibrated |
| `reference_support.npz` | rejection channels | same role, multi-channel |
| NNLS projection | NNLS projection | **unchanged in kind** |

The interface a caller sees — preprocess, project, read evidence, get retrieval and QC — is
deliberately the same shape. That is what makes the Phase-06 head-to-head comparison possible
and the replacement decision clean.

**Note on the disappearing MSS layer.** In V5 there were two objects between components and
themes: the components (learned) and the MSS motifs (curated overlay). In V7 there is one: the
CSM, which is learned and carries the interpretive role the overlay used to carry. L-03 showed
that a curated overlay over a mixed basis borrows foreign mass. The LSM layer sits *below* the
CSM layer as evidence, not as a second overlay.

**Note on the disappearing theme layer.** V7 originally planned a second interpretive object
above the CSM — themes, then the BSV. Three attempts to build it all lost information (§1.1).
The current architecture has no interpretive layer between the representation and retrieval;
the planned architecture puts one there again, with a different label space and a gate that can
reject it.
