# GAIRA V7 — Phase 01 Report
## Local Spectral Motif discovery (Strategy A)

**Branch** `gaira-v7-rebuild` · **Status** COMPLETE · **Gates** 9/9 PASS · **Tests** 47 passed

**Atlas fingerprint before and after: `09ed804a40836f4a05a91ba10900cded` — unchanged, max
element-wise difference 0.0.**

Reproduce:

```bash
export GAIRA_DATA_ROOT=/path/to/GAIRA_DATA/raw
python results/v7_rebuild/phase01/code/run_phase01.py
python results/v7_rebuild/phase01/code/make_figures.py
pytest tests/test_v7_phase01.py
```

---

## 1. Executive summary

The frozen atlas projects spectra onto 24 broad latent components, and Phase 00 confirmed
those components are stable but chemically impure — only 3 of 24 reach purity ≥ 0.5. Phase 01
asks whether that impurity is *resolvable without touching the atlas*: can each component be
decomposed, deterministically and without fitting anything, into reusable spectral
substructures that carry distinguishable chemistry?

**It largely can, but unevenly, and the exceptions matter.**

| | |
|---|---:|
| Retained Local Spectral Motifs | **98** (of 128 candidates; 30 rejected with reasons) |
| Components decomposed | **23 of 24** |
| Components irreducible | 1 (c12) |
| Components aligning with chemistry beyond a permutation null (p<0.05) | **23 of 23** |
| Components whose purity beats a **size-matched random partition** | **22 of 23** |
| Median purity gain beyond mechanical | **+0.145** |
| Median raw purity gain over the whole component | +0.223 |
| Molecule coverage | **100%** (154/154), 6.5 motifs per molecule |
| Max off-diagonal motif cosine | 0.844 (no near-duplicates) |
| Determinism | 3 independent runs → identical motif spectra |
| Attribution conservation error | **2.2 × 10⁻¹⁶** (machine epsilon) |

**The honest qualifier.** The layer resolves protein, saccharide and sterol chemistry well —
several motifs reach purity 1.00 with textbook band assignments. It **fails on the nucleic
chemistry the brief explicitly named**: purines yield one motif at purity 0.33 and pyrimidines
yield none at all. Acylglycerols (max purity 0.40) and organic acids (0.50) are also weak.
See §5.

---

## 2. Divergence from the architecture documents — read this first

The V7 architecture documents and this phase use the term "LSM" for **two different objects**,
and conflating them would corrupt the plan.

| | `LEARNING_MODE_ARCHITECTURE.md` (plan Phase 02) | This phase (Strategy A) |
|---|---|---|
| Construction | row of `H_c` from a **class-local NMF** over balanced references | **deterministic decomposition of a frozen atlas component** |
| Fits anything? | yes — NMF per chemical class | **no** — motifs are masked restrictions of existing components |
| Touches the atlas? | produces a replacement basis | **no** — atlas, projection and fingerprint unchanged |
| Depends on Phase 01 balanced references? | yes | no |

The brief for this phase is explicit and repeated — "LSMs are NOT new atlas components… they
are deterministic decompositions of individual atlas components" — so that is what was built.
It is a lower-risk, learning-free probe of the same underlying question (L-02: are the
components chemically resolvable?) and it answers that question without spending the rebuild.

**Two consequences that need a decision before Phase 02:**

1. The plan's Phase 01 was *balanced reference construction*; that has **not** been done, and
   the plan documents still describe it as the next step. The rebuild plan and the phase
   numbering are now out of step with what exists on disk.
2. If the class-local-NMF LSM is still wanted later, it needs a different name, or this
   layer does — otherwise `lsm_registry_v1.csv` and the architecture's `H_c` rows will collide
   in every downstream document.

Recorded here rather than silently reconciled. Recommended resolution in §9.

---

## 3. Methods

### The idea

An atlas component `h_k` is a single vector, so on its own it cannot tell you whether it is
chemically pure. The information lives in the **analytes that activate it**: if `h_k` carries
one substructure, every analyte activating it should show all of its bands in the same
proportions; if it is being asked to explain two chemistries, some analytes show one
sub-pattern of its bands and others another.

### The pipeline (every step deterministic)

| Step | Operation | Parameter |
|---|---|---|
| 1 Bands | peaks of `h_k` | prominence ≥ 5% of max; window ±4 bins (±8 cm⁻¹) |
| 2 Participants | canonical molecules whose share of their own total activation ≥ τ | τ = 0.03; ≥10 participants and ≥4 bands or the component is not analysable |
| 3 Profile | observed mass of each molecule inside each band, normalised to unit sum | — |
| 4 Cluster | hierarchical linkage; cut by silhouette on cosine distance | `n_motifs ∈ [2, min(9, n−1)]`; **1 is admissible** |
| 5 Score | stability (jackknife), purity, coverage, band fidelity, redundancy | — |
| 6 Reject | deterministic reasons | <3 molecules · <2 bands · stability <0.50 · cosine ≥0.98 |

Motif spectra are **masked restrictions of the parent**: exactly zero outside the motif's
bands, and never exceeding the parent anywhere. This is enforced by test, not by convention.

### No RNG on the discovery path

Stability is a **jackknife** (leave-one-molecule-out re-clustering), not a bootstrap,
specifically so discovery contains no random number generator at all. A test greps
`discovery.py` and `clustering.py` for `np.random`, `default_rng` and `RandomState` and fails
if any appears. The only randomness in Phase 01 is in the seeded evaluation permutation tests,
which never feed back into discovery.

### Three pre-registered selection rules, all published either way

**Chemistry is used only in evaluation.** No class label touches band definition, profile
construction, linkage choice or cut choice. If class labels had chosen the cut, "motifs align
with chemistry" would be circular and unfalsifiable.

| Choice | Candidates | Selected | Evidence |
|---|---|---|---|
| Profile | raw observed band mass · attribution-weighted | **raw** | mean AMI 0.324 vs 0.092; 23 vs 19 components decomposed |
| Linkage | average · ward · complete | **ward** | balance-constrained silhouette rule (below) |
| Motif spectrum | discriminative · representative | **discriminative** | 0 vs 25 motif pairs above cosine 0.9 |

**The linkage rule**, fixed before the run: *lowest mean size Gini among linkages within 0.05
mean silhouette of the best*. Silhouette differences of a few hundredths are not meaningful,
whereas a motif set in which one motif absorbs most participating molecules has peeled off
outliers rather than decomposed the component.

| Linkage | mean silhouette | mean size Gini | mean max-motif share |
|---|---:|---:|---:|
| average | **0.4552** | 0.4274 | 0.6027 |
| **ward** | 0.4262 | **0.2935** | **0.4512** |
| complete | 0.4472 | 0.3576 | 0.4860 |

Ward is within 0.03 of the best silhouette and is markedly more balanced. It also happens to
have the highest AMI — reported as *corroboration*, never as the selection criterion.

---

## 4. Implementation

`src/gaira/v7/lsm/` — a new subpackage. Nothing under `src/gaira/engine/`,
`src/gaira/preprocessing/` or any V5/V6 module was touched.

| Module | Responsibility |
|---|---|
| `motif.py` | the `LSM` and `Band` objects, structural invariants, motif-spectrum construction |
| `clustering.py` | deterministic linkage, cut selection, jackknife stability, the pre-registered rules |
| `discovery.py` | bands → participants → profiles → clusters → scores → rejection |
| `registry.py` | queryable index; keeps **rejected** motifs with their reasons |
| `matching.py` | interpretation path — splits an atlas activation across a component's motifs |
| `serialization.py` | canonicalised save/load + registry fingerprint |
| `validation.py` | chemical alignment, ambiguity resolution, nulls, redundancy, coverage, reproducibility, determinism |
| `visualization.py` | plotting primitives |

### The interpretation path conserves atlas evidence exactly

```
spectrum → canonical preprocessing → NNLS onto the FROZEN 24 components   [unchanged]
                                              ↓  w_k
                    motif attribution: which substructure of k is this w_k carrying?
```

Attribution **redistributes** an activation the atlas already produced; summed over a
component's motifs it equals `w_k`. Measured across all 375 spectra: max conservation error
**2.2 × 10⁻¹⁶**, unattributed evidence **0.0%**. Activation that matches no motif is returned
under a reserved `_unattributed` key rather than silently dropped.

---

## 5. Validation

### 5.1 Does the motif layer resolve chemical ambiguity? — the central question

Two nulls, because raw purity is not evidence on its own: **purity rises mechanically as a set
is cut into more pieces.**

- **Chemical alignment** — adjusted mutual information between motif membership and chemical
  class, against a 1000-draw label permutation. AMI is chance-corrected by construction.
- **Purity beyond mechanical** — observed size-weighted purity against a 500-draw
  **size-matched random partition**: the same motif size profile, membership shuffled. This is
  the direct analogue of Phase 00's size-matched random ontologies.

| Result | Value |
|---|---|
| Components aligned with chemistry (p < 0.05) | **23 of 23** decomposed |
| Components above the size-matched purity null (p < 0.05) | **22 of 23** |
| Median gain beyond mechanical | **+0.145** |
| AMI range | 0.128 (c06) → 0.631 (c02) |

The strongest decompositions:

| Component | classes present | purity whole → motifs | gain beyond mechanical | AMI |
|---:|---:|---|---:|---:|
| c17 | 13 | 0.22 → 0.68 | **+0.319** | 0.523 |
| c15 | 9 | 0.35 → 0.72 | +0.279 | 0.557 |
| c05 | 9 | 0.26 → 0.64 | +0.245 | 0.457 |
| c13 | 14 | 0.12 → 0.53 | +0.232 | 0.504 |
| c00 | 7 | 0.47 → 0.82 | +0.229 | 0.552 |

The weakest, and the one that does not clear the null:

| Component | purity whole → motifs | gain beyond mechanical | verdict |
|---:|---|---:|---|
| c04 | 0.25 → 0.30 | +0.041 | **not significant** |
| c06 | 0.50 → 0.54 | +0.037 | marginal |
| c12 | 0.28 → 0.29 | — | **IRREDUCIBLE** — no admissible cut survived rejection |

### 5.2 The chemistries the brief named

| Chemistry | motifs | best purity | verdict |
|---|---:|---:|---|
| peptide_protein | 20 | **1.00** | **resolved** |
| mono_oligosaccharide | 27 | **1.00** | **resolved** |
| sterol_steroid | 10 | **1.00** | **resolved** |
| free_amino_acid | 10 | 0.83 | mostly resolved |
| fatty_acid | 5 | 0.75 | partly resolved |
| carboxylic_acid_metabolite | 5 | 0.50 | **weak** |
| acylglycerol | 3 | 0.40 | **weak** |
| **purine** | 1 | **0.33** | **not resolved** |
| **pyrimidine** | **0** | — | **not resolved** |

Exemplar motifs, with their bands — these are chemically real, not statistical artefacts:

| Motif | n | purity | bands (cm⁻¹) | reading |
|---|---:|---:|---|---|
| `c02.m01` | 27 | 0.93 | 1000, 1240, 1336, 1456, 1676 | phenylalanine ring breathing (1000) + amide III (1240) + CH deformation (1456) + amide I (1676) — textbook protein Raman |
| `c00.m00` | 12 | 1.00 | 754, 1224, 1318, 1348, 1456 | amide III envelope + CH deformation |
| `c15.m00` | 11 | 1.00 | 1208, 1230, 1264, 1330, 1428, 1682 | amide III + amide I |
| `c17.m08` | 7 | 1.00 | 532, 556, 1036, 1102, 1236, 1280 | saccharide C–O / C–C skeletal modes |
| `c13.m05` | 4 | 1.00 | 702, 806, 1672 | 702 is a classic sterol/cholesterol ring band |

**Why purines and pyrimidines fail is structural, not incidental.** Phase 00 recorded 5
canonical purines and 3 pyrimidines in the whole corpus. A retained motif needs ≥3 molecules,
and those 8 molecules are spread across many components — so a pure nucleic motif needs 3 of
5 purines to co-cluster *within a single component*. This is the same rare-chemistry limit
Phase 00 measured (`k_c ceiling ≤ 2` for both classes), surfacing again in a different layer.
**No amount of decomposition creates evidence the corpus does not contain.**

### 5.3 Motif quality

| Score | mean | median | min | max |
|---|---:|---:|---:|---:|
| molecules per motif | 10.3 | 6 | 3 | 48 |
| bands per motif | 5.1 | 5 | 2 | 11 |
| purity | 0.564 | 0.527 | 0.167 | 1.000 |
| **stability** (jackknife) | **0.968** | 0.985 | 0.812 | 1.000 |
| coverage (share of component participants) | 0.226 | 0.155 | 0.049 | 0.960 |
| band fidelity vs parent | 0.717 | 0.709 | 0.375 | 0.979 |

Stability is high across the board (min 0.812, well above the 0.50 rejection floor), so no
retained motif is a fitting artefact.

### 5.4 Rejection

30 of 128 candidates rejected, every one with a deterministic reason:

| Reason | n |
|---|---:|
| `too_few_analytes` (<3 molecules) | 27 |
| `noise_single_band` (<2 bands) | 3 |
| `low_stability` (<0.50) | 0 |
| `redundant` (cosine ≥0.98) | 0 |

That no motif was rejected for instability or redundancy is itself a finding: the cut rule and
the band construction are not producing junk that has to be filtered out afterwards.

### 5.5 Redundancy and spectral overlap

| | |
|---|---:|
| max off-diagonal cosine | **0.844** |
| mean off-diagonal cosine | 0.081 |
| pairs above 0.90 | **0** |
| pairs above 0.95 | 0 |
| cross-component pairs above 0.90 | 0 |

The 98 motifs are genuinely distinct. This is the measurement that settled the
motif-construction choice (§3): the alternative "representative" construction produced 25
pairs above 0.9 and a maximum of 0.979 — motifs of a component becoming near-copies of one
another because they all inherited the parent's dominant peak.

### 5.6 Coverage

100% of the 154 canonical molecules and all 375 spectra participate in at least one retained
motif; median 7 motifs per molecule. No molecule is left uncovered.

### 5.7 Cross-source and replicate reproducibility

Discovery re-run on data subsets; agreement measured as adjusted Rand index against the
full-corpus run, restricted to shared molecules (chance-corrected, so a trivially
single-motif component does not score as perfect).

| Subset | median ARI | components compared |
|---|---:|---:|
| RamanBioLib only | **0.892** | 22 |
| Gobbato only | **0.559** | 10 |
| replicate half 0 | 0.788 | 23 |
| replicate half 1 | 0.678 | 23 |

**The Gobbato figure is the weak one and it is expected.** Phase 00 measured four classes as
~90–100% RamanBioLib-sourced; removing RamanBioLib removes most of the molecules that define
those components' motifs. The asymmetry (0.892 vs 0.559) is source confounding — risk **R-16**
— showing up exactly where Phase 00 predicted it would.

### 5.8 Determinism

Three independent discovery runs produced **identical motif spectra** (same 32-hex content
signature). No RNG appears anywhere in `discovery.py` or `clustering.py`, verified by a static
test.

---

## 6. Scientific findings

1. **Atlas components are decomposable — 23 of 24.** The impurity Phase 00 measured is not
   irreducible mixing; most components carry several band sub-patterns that separate cleanly.
2. **The separation tracks chemistry beyond chance.** 23/23 components significant on
   chance-corrected AMI; 22/23 exceed a size-matched random partition on purity, median gain
   +0.145. The finding is not the mechanical effect of cutting a set into more pieces.
3. **Protein, saccharide and sterol chemistry resolve to purity 1.00** with chemically
   readable band sets, including the classic phenylalanine-1000 + amide-I/III protein motif.
4. **Nucleic chemistry does not resolve at all** — 1 purine motif at 0.33 purity, 0 pyrimidine
   motifs. This is a corpus limit (5 purines, 3 pyrimidines total), not a method failure, and
   it is the same limit Phase 00 recorded.
5. **One component is genuinely irreducible** (c12), and roughly 20 of 24 components hold a
   single dominant motif alongside smaller ones — the decomposition is real but usually
   asymmetric.
6. **Source confounding is measurable at the motif level.** Cross-source ARI 0.892 vs 0.559
   is the first quantitative consequence of the confounding Phase 00 flagged.

---

## 7. Engineering findings

1. **The atlas is untouched and the layer is additive.** Fingerprint identical before and
   after; motif attribution conserves atlas activation to 2.2 × 10⁻¹⁶.
2. **Determinism holds without special effort** because the discovery path has no RNG. Choosing
   a jackknife over a bootstrap for stability was what made this achievable.
3. **A real bug was found by the unit tests, not by the pipeline.** `select_cut` allowed
   `n_motifs == n_participants`, which makes silhouette undefined and crashes. It never fired
   on the real corpus (every component has ≥24 participants) but would have on any small
   component. Fixed by bounding the sweep at `n − 1`.
4. **A figure led me to the wrong conclusion, and the measurement corrected it.** Figure 2
   suggested motifs were systematically de-emphasising their parent's dominant peak; the
   proposed "fix" was benchmarked and made redundancy dramatically worse (25 pairs above
   cosine 0.9 vs 0). The apparent problem was amplitude scaling in the plot. Figures now
   normalise so the comparison is on shape.
5. **`src/gaira/v7/lsm/` is a new namespace**, importable and testable without any data.
   47 tests: 30 unit tests on synthetic inputs, 17 contract tests on the committed artefacts.

---

## 8. Limitations

1. **This is not the architecture documents' LSM** (§2). It decomposes frozen components
   rather than fitting class-local bases, so it cannot deliver what a rebuilt basis could —
   it can only redistribute evidence the existing atlas already produces.
2. **Rare chemistry is untouched.** Purines, pyrimidines, phospholipids and carotenoids remain
   unresolved; the motif layer inherits the corpus limits it was given.
3. **Purity is measured against the Phase-00 fine ontology**, which is itself a curated prior.
   A motif "pure" in that ontology is pure with respect to a human classification, not to
   ground-truth chemistry.
4. **Participation threshold τ = 0.03 is a free parameter.** It was fixed before the run and
   not swept. A stricter τ would give fewer, purer participants per component; the sensitivity
   of the whole layer to it is unmeasured.
5. **Band windows are fixed at ±8 cm⁻¹** regardless of the band's actual width.
6. **Motifs overlap in participation** — a molecule belongs to a median of 7 motifs. The layer
   adds resolution, not a partition of the corpus.
7. **No downstream benefit has been demonstrated.** Phase 01 shows motifs separate chemistry
   *within* components; it does **not** show that using them improves retrieval, the BSV, or
   any Phase-00 benchmark. That test does not exist yet and must not be assumed.

---

## 9. Future dependencies and recommended decisions

**Consumed by later phases**

| Artefact | Used for |
|---|---|
| `artifacts/lsm_spectra_v1.npz` + `tables/lsm_registry_v1.csv` | the motif dictionary and its provenance |
| `matching.attribute_spectrum` | motif-level evidence at inference |
| `tables/chemical_alignment_v1.csv`, `purity_null_v1.csv` | the baseline any consensus layer must beat |

**Three decisions needed before Phase 02**

1. **Resolve the naming collision** (§2). Recommendation: keep "LSM" for this
   frozen-component layer, since it is now implemented, serialised and tested, and rename the
   architecture's class-local NMF rows — or drop that construction if this layer supersedes
   it. Either way the architecture documents and the rebuild plan need updating so the phase
   numbering matches what exists.
2. **Decide whether balanced reference construction still happens.** It was the plan's Phase
   01 and has not been done. If Strategy B is the class-local rebuild, it depends on it.
3. **Measure downstream benefit before building on this layer.** The natural next test is
   whether motif-level evidence improves fine-family retrieval against the frozen Phase-00
   harness and splits. Until that is measured, the motif layer is a demonstrated *internal*
   improvement with unknown external value.

---

## 10. Discussion

The result to take seriously is the asymmetry. The motif layer works where the corpus is
dense — proteins (30 molecules), saccharides (20), sterols (10) — and fails where it is thin —
purines (5), pyrimidines (3). That is the same boundary Phase 00 drew from a completely
different direction, and it is the boundary the whole V7 programme keeps running into: this
corpus supports coarse chemistry richly and fine chemistry sparsely.

That reframes what Phase 01 has shown. It has **not** shown that the frozen atlas is adequate;
it has shown that a substantial part of the atlas's chemical impurity is *recoverable without
refitting anything*. Twenty-three of twenty-four components hold real, stable, chemically
coherent substructure that the 24-dimensional projection discards. That is a genuine finding
about the existing atlas, obtained at a fraction of the cost of a rebuild.

But it should not be over-read, in two specific ways.

First, **no downstream benefit has been demonstrated**. Higher within-component purity is a
property of the motif layer, not evidence that any GAIRA output improves. The V6.2 experience
is the relevant caution: an added abstraction layer (`theme_posterior`) that was numerically
identical to the layer beneath it at every metric. Until motif evidence is measured against
the Phase-00 harness, this layer could be exactly that.

Second, **the layer cannot exceed its parent**. Every motif is a restriction of a frozen
component, so the motif layer can only redistribute evidence the atlas already produced — as
the conservation check makes precise. If a chemistry is absent from all 24 components, no
decomposition will find it. That is why the purine and pyrimidine results are not a bug to
fix but a boundary to respect, and why Phase 09 corpus expansion remains the only route to
that particular chemistry.

The most useful thing Phase 01 contributes to the wider rebuild is therefore a **cheap
diagnostic**: it says which components carry recoverable substructure and which do not, using
no fitting and no new assumptions. If Strategy B does rebuild the basis, this is the map of
where a rebuild has something to find.

---

## 11. Gates

| Gate | Result | Evidence |
|---|---|---|
| implementation_complete | ✅ PASS | 98 motifs across 23 decomposed components |
| atlas_unchanged | ✅ PASS | basis identical before/after, max abs difference 0.0 |
| fingerprint_unchanged | ✅ PASS | `09ed804a40836f4a05a91ba10900cded` recomputed |
| deterministic | ✅ PASS | 3 runs → identical motif spectra; no RNG on the discovery path |
| registry_generated | ✅ PASS | integrity checks pass; rejected motifs retained with reasons |
| projection_conserved | ✅ PASS | attribution error 2.2 × 10⁻¹⁶ |
| validation_passed | ✅ PASS | alignment, ambiguity, redundancy, coverage, reproducibility all measured |
| scientific_benefit_demonstrated | ✅ PASS | 23 components beat a permutation null; 22 beat a size-matched null |
| reproducibility_measured | ✅ PASS | cross-source and replicate ARI on 78 component-subset pairs |

**9 / 9 PASS.**
