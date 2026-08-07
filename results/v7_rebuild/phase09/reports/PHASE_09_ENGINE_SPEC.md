# GAIRA V7 — Canonical Engine Specification

`gaira.v7.canonical.GAIRAEngine` · frozen at Phase 09 · Raman-only

This document specifies the engine precisely enough to reimplement it. It describes what the
engine *is*, not what it achieves; for measurements see `PHASE_09_REPORT.md`, and for the
equations see `PHASE_09_MATHEMATICAL_APPENDIX.md`.

---

## 1. Contract

```python
from gaira.v7.canonical import GAIRAEngine

engine = GAIRAEngine.load()                              # verifies fingerprints, then frozen
report = engine.infer(intensities, wavenumbers)          # one spectrum in, one report out
d      = report.to_dict()                                # fully JSON-serialisable
```

Three properties hold by construction and are asserted by gates:

- **Frozen.** `load()` verifies four fingerprints and raises `FrozenArtifactError` on any
  mismatch. The engine will not run on a changed atlas. It never writes to, regenerates or
  refits any upstream artefact.
- **Stateless.** No attribute is mutated after `__init__`. `InferenceReport` and
  `PreprocessingSummary` are `@dataclass(frozen=True)`. A spectrum's result is identical whether
  it is inferred alone or inside a batch, and identical on repeat (gate G5, verified).
- **Reconciling.** Every score the engine reports decomposes exactly into the components it
  displays. `retrieve()` sets `reconciles = |Σ contributions − similarity| < 1e-9` per candidate;
  it was true for all 375 × 10 candidates (gate G7).

## 2. Frozen artefacts and their fingerprints

| artefact | source | fingerprint |
|---|---|---|
| grounded atlas | phase01 | `09ed804a40836f4a05a91ba10900cded` |
| LSM dictionary (50 × 676) | phase01 | `208482d6f7178b5b8f16cace91be55b0` |
| CSM dictionary (49 × 676) | phase02 | `0b4aa550ccefed3edabdbde5bae11c8d` |
| retrieval engine | phase05 | `20d8bd99ce71f45a125c6a2b1d719e51` |
| atlas content hash | derived | `2e43ddcca7d3be41c5f9da016fb8277f` |

Also loaded, all frozen: the 154-molecule reference bank in CSM coordinates (`ref_A`), its
chemistry evidence (`ref_E`), molecule labels, class labels, the Phase 06 chemistry model, the
Phase 06 calibrator, the 16 axis names, and the CSM registry records that carry each motif's
diagnostic bands and contributing LSMs.

`load(frozen_root=None, verify=True)` resolves `frozen_root` through `gaira.v7.io.frozen_root()`.
No output path is hardcoded. `verify=False` exists for offline testing against a deliberately
perturbed copy and must never be used in production; the test suite exercises both branches.

## 3. Canonical constants

```python
GRID_LO, GRID_HI, GRID_STEP, N_BINS = 450.0, 1800.0, 2.0, 676
```

Frozen in Phase 00, unchanged since V5. Any spectrum entering the engine is expressed on this
grid before anything else happens.

## 4. Stage 1 — `preprocess(wavenumbers, intensities) -> (x, PreprocessingSummary)`

In order:

1. **Resample** onto the canonical grid by linear interpolation, `left=0, right=0`. Regions the
   input does not cover are **zero-filled, never extrapolated**, and a warning is emitted.
2. **Baseline** — asymmetric least squares, `lam = 1e5`, `p = 0.01`, 10 iterations. Subtract and
   clip at zero.
3. **Smooth** — Savitzky–Golay, window 9, polynomial order 3. Clip at zero.
4. **Normalise** — L2 to unit length.

The summary records the input range, peak count (`find_peaks`, prominence 0.02 of maximum), an
SNR estimate (`max / median|Δx|`), a bounded signal-quality score, and warnings. Three conditions
warn: input range not covering the canonical window, fewer than 100 input points, and SNR < 20.

`infer(..., already_preprocessed=True)` skips stages 1–3 for a spectrum already on the canonical
grid, re-applying only the L2 normalisation. This exists for corpus validation, not convenience:
running asLS a second time on an already-corrected spectrum would subtract a second small
baseline and quietly change the numbers.

## 5. Stage 2 — `project_lsm(x) -> dict`

Non-negative least squares of `x` onto the 50-motif LSM dictionary:

```
a_lsm = argmin_{a ≥ 0} ‖ x − a Hᴸ ‖₂
```

Returns the activation, the reconstruction, the residual norm, explained variance, the count of
active components, and the top eight motifs with their weights and shares. Diagnostic only — the
LSM activation is **not** consumed by any later stage. It is reported because it is the layer at
which "which local patterns are present" is legible.

## 6. Stage 3 — `project_csm(x) -> dict`

Projection onto the 49-motif CSM dictionary via `gaira.v7.inference.projection.project`, the
frozen Phase 05 routine (non-negative, P-02). **This activation is the canonical representation.**
Everything downstream reads it and only it.

Returns the activation, explained variance, residual fraction, sparsity, activation entropy,
active count, and the top eight CSMs — each with its weight, share, diagnostic band positions,
band assignment string, and contributing LSM ids. Those fields are what make the answer
inspectable: a user can follow a chemistry conclusion down to specific wavenumbers.

## 7. Stage 4 — `retrieve(a_csm, top_k=10) -> dict`

Cosine similarity between the L2-normalised query activation and the L2-normalised reference
bank, clipped to [0, 1], sorted descending. This is **Phase 08 Model A**, adopted unchanged after
chemistry-aware reranking failed to beat it once both arms used the same 154-molecule bank.

For each of the top *k* candidates the engine returns the molecule, its chemistry class, the
similarity, the five CSMs contributing most to that similarity — with each contribution, its
share, its LSMs and its diagnostic bands — the contribution sum, and `reconciles`. Because the
similarity is an inner product of unit vectors, the per-CSM contributions sum to it exactly.
There is no hidden term.

`margin` is the difference between the best and second-best similarity across the whole bank.

## 8. Stage 5 — `chemistry(a_csm) -> dict`

The frozen Phase 06 chemistry model maps the CSM activation to 16-dimensional **Chemistry
Evidence**, then the frozen Phase 06 calibrator maps evidence to calibrated probabilities.

Returns the axis names, the raw evidence vector, the calibrated probabilities, the L1-normalised
evidence (the radar radii), the top five axes, the predicted class, the evidence margin and the
normalised entropy.

The 16 axes, in canonical order:

```
acylglycerol · carboxylic_acid_metabolite · chromophore_pigment · fatty_acid ·
free_amino_acid · mono_oligosaccharide · nucleic_acid_polymer · peptide_protein ·
phosphate_metabolite · phospholipid_sphingolipid · polysaccharide · purine ·
pyrimidine · small_nitrogenous · sterol_steroid · sulfur_thiol_cofactor
```

> **The radar shows RELATIVE BIOCHEMICAL EVIDENCE.** It is not a concentration, not an abundance,
> and not a mixture decomposition. This label is mandatory on every rendering (gate G12).

## 9. Confidence and warnings — `_confidence(...)`

```
overall = clip(csm_explained_variance, 0, 1) × top1_similarity
```

Deliberately pessimistic and deliberately multiplicative: a spectrum the atlas cannot explain must
not produce a confident answer no matter how well its residual happens to match some reference.

Two warnings, with thresholds inherited from Phase 05 and **not tuned in Phase 09** (P-13):

| warning | condition |
|---|---|
| `unknown` | CSM explained variance < 0.50, **or** retrieval margin < 0.01 |
| `outlier` | CSM residual fraction > 0.50, **or** ≤ 1 active CSM |

Also reported: evidence coverage, top-1 and top-3 confidence, retrieval margin, chemistry
confidence (the maximum calibrated probability), and human-readable notes explaining each warning
that fired.

## 10. Provenance — `_provenance(...)`

A complete tree for every spectrum: **spectrum → LSM layer → CSM layer → chemistry layer →
molecule layer**, with the atlas fingerprint stamped at the root. Each molecule records the CSMs
it was retrieved through; each CSM records its LSMs, its diagnostic bands and its band
assignment. Every conclusion can be walked back to wavenumbers (gate G13).

## 11. Output object

```python
@dataclass(frozen=True)
class InferenceReport:
    preprocessing: PreprocessingSummary
    lsm: dict
    csm: dict
    retrieval: dict
    chemistry: dict
    confidence: dict
    provenance: dict
    atlas_fingerprint: str
```

`to_dict()` returns a plain JSON-serialisable structure with all NumPy types converted. 48
examples are shipped in `reports/examples/`.

## 12. Introspection

`engine.grid` (a copy), `engine.reference_molecules` (154), `engine.chemistry_axes` (16),
`engine.fingerprints`, `engine.atlas_fingerprint`, and a `__repr__` that states the shape of the
loaded atlas.

---

## 13. What the engine does not contain

Not present anywhere on the inference path, verified by gate G3:

**BSV2 · latent geometry · continuous manifold coordinates · UMAP · PCA · clustering · cluster
identifiers · themes · Meta Components · grounded axes · SERS handling · serum or EV handling ·
any tunable parameter set in Phase 09 · any random number drawn at inference time.**

BSV2 exists and is validated (Phase 07), but it is a **derived description** that reads Chemistry
Evidence downstream of the engine. It never feeds inference. UMAP appears in exactly one figure,
labelled `VISUALISATION ONLY`.

## 14. Failure modes and how the engine behaves

| situation | behaviour |
|---|---|
| fingerprint mismatch | `FrozenArtifactError` at `load()`; the engine does not run |
| spectrum shorter than the canonical window | zero-filled with a warning; never extrapolated |
| spectrum empty after preprocessing | warning; the L2 normaliser is epsilon-guarded |
| length mismatch between spectrum and wavenumbers | `ValueError` |
| `already_preprocessed=True` with the wrong bin count | `ValueError` |
| the atlas cannot express the spectrum | low explained variance → low confidence → `unknown` |
| the true molecule is not in the bank | the engine still returns a chemistry answer; it cannot know the molecule is absent |
| two candidates are indistinguishable | margin < 0.01 → `unknown`, with the margin quoted |

The last row is the important one. The engine has no mechanism for detecting that the true
molecule is absent from the 154-molecule bank; it will return its nearest neighbours with whatever
similarity they earn. The `unknown` and `outlier` warnings detect *unexplained spectra*, not
*unknown molecules*. Treating them as an open-set detector would be a misuse.

The warnings are also weaker than they look against **structureless** input. White noise
reconstructs at CSM explained variance ≈ 0.61 — above the 0.50 floor — so a warning fires on only
1 of 20 random spectra. Confidence still separates it cleanly (noise maximum 0.495 against a
corpus mean of 0.803), so **an operator should read the confidence rather than the flag**. This
is measured in `tests/test_v7_phase09.py` and recorded in the audit as C5b.
