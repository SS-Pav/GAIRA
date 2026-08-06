# GAIRA V7 — Target Architecture

The complete specification of what V7 will be. Companion documents give the detail:
`LEARNING_MODE_ARCHITECTURE.md` (how each learning stage works),
`INFERENCE_MODE_ARCHITECTURE.md` (the runtime path), `DATA_CONTRACTS.md` (schemas),
`ARTIFACT_AND_MANIFEST_SPEC.md` (what is frozen), `LIVE_DART_COMPATIBILITY.md` (trajectories).

**Nothing described here is implemented.**

---

## 1. The two modes, and why the split is absolute

GAIRA has exactly two modes, and the boundary between them is the most important
architectural line in the system.

| | Learning mode | Inference mode |
|---|---|---|
| When | offline, during a build | live, per spectrum |
| Input | the pure-Raman grounding corpus | one new spectrum |
| Output | a frozen, versioned atlas | a BSV + evidence + QC |
| Fitting | yes — this is where all fitting happens | **never** |
| Determinism | deterministic given the seed schedule | deterministic, full stop |
| Runs where | a build machine with the corpus | anywhere, from a clean clone, no lab volume |

The reason this line is absolute: **comparability**. Two spectra measured in different labs
years apart are comparable only if they were projected onto the *same fixed axes*. Any fitting
at inference — even something as innocuous as re-fitting a PCA for a plot — makes the
coordinate system depend on the batch, and the comparison silently becomes meaningless.

---

## 2. Learning mode

```
Raw Raman grounding corpus
   375 spectra, 167 canonical analytes, pure Raman only
        │
        ▼
canonical preprocessing                                          ── UNCHANGED from V5
   asls baseline → savgol smoothing → L2 norm
   450–1800 cm⁻¹, 2.0 cm⁻¹ step, 676 bins
        │
        ▼
canonical molecule IDs and replicate groups                      ── PHASE 00
   alias resolution (NFKC + whitespace + case)
   replicate group = (canonical_id, excitation)
   quality score q per spectrum
   frozen analyte-grouped CV splits
        │
        ▼
balanced reference construction                                  ── PHASE 01
   one canonical molecule = one reference unit
   strategy selected from {A all-spectra, B analyte-weighted, C robust prototype}
        │
        ▼
chemical-family partition                                        ── PHASE 00 freeze / PHASE 01 use
   X → {X_c}, one block per curated chemical class
   an organisational prior, never a target
        │
        ▼
class-specific repeated non-negative decomposition               ── PHASE 01
   for each class c:  X_c ≈ W_c H_c,  W_c, H_c ≥ 0
   k_c chosen per class (adaptive, Pareto-plateau rule)
   R repeated fits under resampling + seed variation
        │
        ▼
stable Local Spectral Motifs (LSMs)                              ── PHASE 01
   Hungarian alignment across runs → recurrence score
   retain only LSMs above the pre-registered stability threshold
   label each: class-shared | subfamily | molecule-discriminating
        │
        ▼
LSM alignment and consensus clustering                           ── PHASE 02
   pool all stable LSMs across all classes
   full-space similarity graph on 6 edge features
        │
        ▼
cross-class motif graph                                          ── PHASE 02
   nodes = LSMs, edges = multi-feature similarity
        │
        ▼
Consensus Spectral Motifs (CSMs)                                 ── PHASE 02
   integration method SELECTED ON EVIDENCE from:
     consensus clustering | graph communities |
     sparse non-negative meta-factorisation | hybrid
   each CSM: consensus spectrum + full provenance + uncertainty
   optional anchored atoms for rare chemistry (Strategy F)
        │
        ▼
CSM → theme soft mapping                                         ── PHASE 03 ✔ COMPLETE
   S ∈ ℝ₊^{49×5}, sparse, non-negative, row-normalised
   K = 5 by archetypal analysis; 4 themes accepted, 1 rejected
   15 bridge CSMs keep split membership; 9 poorly-explained CSMs left unplaced
        │
        ▼
continuous BSV reference space                                   ── PHASE 04 ✔ COMPLETE
   BSV = Sᵀc over 4 accepted themes; absolute, non-negative
   effective rank 2.40 of nominal K = 4 (risk R-12)
   frozen projection engine: elastic-net → direct CSM → confidence-weighted themes
   held out: molecule top-1 0.799, class top-1 on UNSEEN molecules 0.855 (raw 0.608)
   OPEN: OOD cannot separate real Ag-SERS (AUROC 0.548); confidence ECE 0.486
   reference distributions per theme axis
   normalisation frame (μ, σ), OOD support, uncertainty model
        │
        ▼
frozen GAIRA V7 Atlas                                            ── PHASE 05
   preprocessing spec + LSM dictionaries + CSM basis + S +
   theme registry + BSV reference stats + provenance + manifest
   ONE fingerprint over ALL layers
```

### Where each earlier failure is addressed

| Prior limitation | Addressed by |
|---|---|
| L-01 objective counts spectra | Phase 01 balanced reference construction |
| L-02 stable but impure components | Phase 01 class-specific decomposition + adaptive `k_c` |
| L-03 motifs borrow foreign mass | Phase 02 bottom-up CSMs with mandatory provenance; singletons visible |
| L-04 fine retrieval plateau | the whole chain — measured, not assumed, in Phase 06 |
| L-05 ontology cleanup insufficient | Phase 03 themes derived *from CSMs*, not asserted over them |
| L-06 true representation failures | Phase 06 failure waterfall, retained as the primary success metric |
| L-07 thin chemistry | Phase 01 adaptive `k_c` + Phase 02 anchors + Phase 08 targeted expansion |

---

## 3. Inference mode

```
New Raman spectrum  (wavenumbers + intensities + metadata)
        │
        ▼
canonical preprocessing                        ── deterministic; same spec as the build
   crop → resample to the frozen grid → asls → savgol → L2
        │
        ▼
fixed-dictionary non-negative projection       ── NNLS against the frozen CSM basis
   c(x) = argmin_{c ≥ 0} ‖x − c ᵀ CSM‖²
        │
        ▼
LSM activation evidence                        ── optional, for explanation
   per-LSM contribution within each contributing CSM
        │
        ▼
CSM activation evidence                        ── c(x) ∈ ℝ₊^M, with bands and provenance
        │
        ▼
soft biochemical themes                        ── t(x) = Sᵀ c(x)
        │
        ▼
absolute BSV                                   ── BSV(x) = t(x) ∈ ℝ₊^K
        │
        ▼
reference comparison + QC + uncertainty
   z-scored elevation against the frozen reference frame
   OOD score against the reference support
   reconstruction residual, band-fidelity check
   uncertainty propagated from projection through S
        │
        ▼
domain-context interpretation                  ── serum / EV / plasma / tissue / pathogen
   downstream of the BSV; NEVER feeds back upstream
```

### Prohibited at inference — a closed list

Live inference must **not** perform:

- NMF fitting
- PCA fitting
- UMAP fitting (or t-SNE, or any manifold learning)
- clustering of any kind
- graph community detection
- ontology optimisation
- threshold tuning against the incoming batch
- any operation whose result depends on *which other spectra* are in the batch

That last item is the general principle the others are instances of: **the output for a
spectrum must not depend on its batch-mates.** If it does, the coordinate system is not fixed.

### Permitted at inference — a closed list

- canonical preprocessing (deterministic, per-spectrum)
- NNLS against a frozen dictionary
- matrix multiplication by frozen matrices (`S`, normalisation)
- application of a **frozen** linear transform for visualisation (a PCA whose `P` and `μ`
  were fitted offline and shipped in the atlas — the transform is applied, never fitted)
- distance and similarity calculations against frozen reference objects
- uncertainty propagation through frozen linear maps
- trajectory updates (appending a new absolute BSV to a sequence)

### The PCA/UMAP boundary, stated explicitly

A frozen PCA transform applied for plotting is **visualisation**, not inference. Its output
`y = Pᵀ(BSV − μ)` is never the canonical BSV, is never an input to interpretation, and is
never used for retrieval or scoring. UMAP is not shipped in the atlas at all: it has no
out-of-sample transform that is stable enough to freeze, and using it would make the picture
depend on the batch.

**No V7 document may describe PCA or UMAP as inference.**

---

## 4. Layer contracts

| Layer | Object | Space | Frozen at | Live operation |
|---|---|---|---|---|
| Preprocessing | spec | — | Phase 00 | apply |
| Balanced reference | rows | `ℝ₊^{N×D}` | Phase 01 | not used live |
| LSM dictionary | `H_c` per class | `ℝ₊^{k_c×D}` | Phase 01 | optional evidence projection |
| CSM basis | `CSM` | `ℝ₊^{M×D}` | Phase 02 | **NNLS projection** |
| Membership | `S` | `ℝ₊^{M×K}` | Phase 03 | matrix multiply |
| Theme registry | names, definitions | — | Phase 03 | lookup |
| BSV reference | `μ`, `σ`, support | `ℝ^K`, `ℝ₊^K` | Phase 04 | z-score, OOD |
| Visualisation | `P`, `μ_P` | `ℝ^{K×2}` | Phase 04 | frozen transform |
| Domain context | rules | — | separate version | applied post-BSV |

`D = 676`. `M` (CSM count) is selected in Phase 02. `K` (theme count, = BSV dimension) is
selected in Phase 03.

---

## 5. Invariants — must hold at every layer, checked in Phase 05

| Invariant | Check |
|---|---|
| **Non-negativity** | every basis, activation, membership, and BSV component ≥ 0 |
| **Determinism** | identical input → byte-identical output, twice, on two machines |
| **Provenance completeness** | every CSM resolves to LSMs → classes → analytes → sources |
| **No inference-time fitting** | static check: no `fit`, `fit_transform`, or RNG use in the inference path |
| **Batch independence** | a spectrum's output is identical alone and in a batch of 1000 |
| **Frozen-package sufficiency** | inference runs on a clean clone with no lab volume |
| **Fingerprint integrity** | the atlas fingerprint covers every layer and is verified on load |
| **Domain isolation** | no domain object is reachable from any pre-BSV module |
| **Absolute BSV** | the BSV is never computed as a difference anywhere in the codebase |

---

## 6. Relationship to the V5 engine

The V5 engine (`src/gaira/engine/`) stays in production, unmodified, throughout V7
development. V7's engine is a **parallel implementation**, not an edit.

| V5 | V7 | Change |
|---|---|---|
| `manifold_components.npz` (24×676 NMF basis) | CSM basis (M×676) | different construction, same role and interface |
| `component_registry_v1.json` | CSM registry + LSM registry | richer provenance; two levels instead of one |
| `component_theme_weights_v1.json` (24×13) | membership `S` (M×K) | derived from CSMs rather than asserted onto components |
| `mss_motifs_v1.yaml` (curated overlay) | *no equivalent* | the overlay is replaced by the CSM layer itself |
| `biochemical_ontology_v2.yaml` (13 themes) | theme registry (K themes) | K selected on evidence |
| `reference_normalization_v1.json` | BSV reference stats | same role |
| `reference_support.npz` | OOD support | same role |
| NNLS projection | NNLS projection | **unchanged in kind** |

The interface a caller sees — preprocess, project, read a BSV, get evidence and QC — is
deliberately the same shape. That is what makes a Phase-06 head-to-head comparison possible
and a Phase-06 replacement decision clean.

**Note on the disappearing MSS layer.** In V5 there were two objects between components and
themes: the components (learned) and the MSS motifs (curated overlay). In V7 there is one:
the CSM, which is learned and carries the interpretive role the MSS overlay used to carry.
This is deliberate — L-03 showed that a curated overlay over a mixed basis borrows foreign
mass. The LSM layer sits *below* the CSM layer as evidence and explanation, not as a second
overlay.
