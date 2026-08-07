# GAIRA V7 — Phase 09: The Canonical Inference Engine

**Status: COMPLETE · 16 of 16 gates PASS · architecture unchanged**

Phase 09 is a **packaging phase**. It introduces no new representation, no new optimisation, no
new retrieval strategy, no clustering, no dimensionality reduction, no thresholds and no
heuristics. Everything it executes was decided and measured in Phases 01–08. What it adds is a
single object — `GAIRAEngine` — that turns a raw Raman spectrum into a complete, explainable,
reproducible biochemical interpretation along exactly one path, and a validation of that object
across **every spectrum in the grounded corpus**.

| | |
|---|---|
| Engine | `GAIRAEngine(atlas=2e43ddcca7d3…, 50 LSMs, 49 CSMs, 154 molecules, 16 chemistry axes)` |
| Frozen fingerprints | atlas `09ed804a40836f4a05a91ba10900cded` · LSM `208482d6f7178b5b8f16cace91be55b0` · CSM `0b4aa550ccefed3edabdbde5bae11c8d` · Phase 05 engine `20d8bd99ce71f45a125c6a2b1d719e51` |
| Verified on load | yes — `FrozenArtifactError` on any mismatch, nothing upstream regenerated |
| Deterministic | yes — byte-identical outputs on repeat invocation |
| Scope | **pure Raman reference spectra only**. No SERS, no serum, no EV, no biofluid, no cross-modality claim anywhere in this phase. |
| Corpus | 375 spectra · 154 canonical molecules · 16 chemistry families · 3 source libraries |

---

## 1. What the engine is

```python
from gaira.v7.canonical import GAIRAEngine

engine = GAIRAEngine.load()                # verifies four fingerprints, then freezes
report = engine.infer(intensities, wavenumbers)
```

`infer()` runs five stages and nothing else:

1. **Preprocess** — crop to 450–1800 cm⁻¹, resample to 676 bins at 2.0 cm⁻¹, asymmetric
   least-squares baseline removal, Savitzky–Golay (window 9, order 3) smoothing, L2 normalisation.
2. **Project onto the LSM dictionary** (50 Local Spectral Motifs) — non-negative, no refitting.
3. **Project onto the CSM dictionary** (49 Consensus Spectral Motifs) — the canonical coordinates.
4. **Retrieve** the nearest reference molecules by cosine similarity in CSM space.
5. **Aggregate** the CSM activation into 16-dimensional **Chemistry Evidence**, calibrate it, and
   emit the radar together with confidence and warnings.

The returned `InferenceReport` carries the preprocessing summary, both projections, the ranked
molecular candidates with their per-term score decomposition, the Chemistry Evidence vector, the
calibrated confidence, the warnings, and the full provenance tree. There are no hidden scores:
every displayed similarity reconciles to its stated components, asserted per candidate for all
375 spectra (gate G7).

### What is deliberately *not* on the inference path

BSV2, latent geometry, UMAP, PCA, clustering, cluster identifiers, continuous manifold
coordinates, the legacy theme layer and Meta Components are all **absent**. Each exclusion is a
measured decision, not a preference:

| excluded | why, and where it was measured |
|---|---|
| Themes | chemistry generalisation to unseen molecules fell to 0.405 (Phase 03/04) |
| Meta Components | 0.392 (Phase 04.5) |
| 11 grounded axes | 0.664 (Phase 04) |
| Latent geometry / coordinates | molecule Δ+0.016, McNemar p = 0.180, CI crossing zero (Phase 06.5) |
| Chemistry-aware reranking | Δ collapsed to non-significance once both arms used the same 154-molecule bank (Phase 08, gate G7b) |
| BSV2 | a **derived, downstream** description; it reads Chemistry Evidence and never feeds it |

---

## 2. Validation 1 — the LSM layer

Every one of the 375 spectra was projected onto the frozen 50-motif LSM dictionary.

| metric | value |
|---|---|
| mean explained variance | **0.8237** |
| minimum explained variance | 0.2087 (pyruvate) |
| mean reconstruction error | 0.3732 |
| mean active components | 9.6 of 50 |
| mean sparsity | 0.8723 |
| replicate consistency | 0.8799 |

The dictionary reconstructs a typical spectrum from about ten of fifty motifs, which is the
behaviour a non-negative parts-based dictionary should show: it is describing composition, not
memorising curves. Replicates of the same molecule land in nearly the same place (0.880), so the
representation is dominated by the molecule rather than by the acquisition.

The minimum matters more than the mean. Pyruvate at 0.209, malic acid at 0.262 and thymine at
0.258–0.286 are the molecules the dictionary genuinely cannot express, and the engine flags them
(see §7).

## 3. Validation 2 — the CSM layer

| metric | value |
|---|---|
| chemistry-class top-1 | **0.8453** |
| chemistry-class top-3 | 0.9707 |
| macro F1 | 0.8068 |
| balanced accuracy | 0.7966 |
| mean explained variance | 0.8232 |
| mean active components | 9.56 of 49 |
| replicate consistency | 0.8927 |

Collapsing 50 local motifs to 49 consensus motifs costs essentially nothing in reconstruction
(0.8237 → 0.8232) while raising replicate consistency (0.8799 → 0.8927). That is the trade the CSM
layer exists to make, and it is the reason A-08 named the CSM the canonical representation.

Per-class F1 spans 1.000 (sterol_steroid, nucleic_acid_polymer) down to 0.400
(small_nitrogenous, n = 7) and 0.556 (phospholipid_sphingolipid, n = 8). The weak classes are the
small ones, and they are weak in the direction a spectroscopist would predict — they are confused
with the larger families whose chains and rings they share.

## 4. Validation 3 — molecular retrieval

Leave-one-spectrum-out over all 375 spectra against the full 154-molecule bank.

| metric | value | Phase 05/08 baseline |
|---|---|---|
| top-1 | **0.6053** | 0.6053 |
| top-3 | 0.7627 | 0.7627 |
| top-5 | 0.7947 | 0.7947 |
| top-10 | 0.8107 | 0.8107 |
| MRR | 0.6870 | 0.6870 |
| nDCG@5 | 0.7112 | 0.7112 |
| median rank | 1.0 | — |

**The frozen baseline is reproduced exactly** (gate G8). Phase 09 changed no retrieval behaviour,
and the numbers demonstrate it rather than asserting it.

The rank distribution explains the ceiling. 227 spectra (60.5%) rank their true molecule first,
a further 59 land at rank 2–3, and then **68 spectra (18.1%) are structurally unretrievable**:
they belong to the 66 molecules represented by a single spectrum, so when that spectrum is held
out the correct answer is not in the bank at all. Following the Phase 05/06.5 convention these are
counted as misses, which is conservative. Ignoring them entirely would report top-1 ≈ 0.74, and
that number would be misleading.

Retrieval confidence is the score margin, calibrated in-fold: ECE 0.1205, Brier 0.2260,
discrimination 0.6914. The risk–coverage curve is the practically useful object — abstaining
below a margin of 0.497 keeps 51% of spectra at 79% accuracy, and below 0.610 keeps 31% at 84%.

## 5. Validation 4 — Chemistry Evidence

> **Held-out is the performance number. In-sample is a sanity check.**

| metric | value |
|---|---|
| fine-class top-1, **molecule-grouped held-out** | **0.8507** |
| fine-class top-3, held-out | 0.9760 |
| macro F1, held-out | 0.8110 |
| fine-class top-1, in-sample | 0.9547 |
| fine-class top-3, in-sample | 1.0000 |
| macro AUC, in-sample | 0.9990 |
| macro average precision, in-sample | 0.9833 |
| calibration ECE | 0.0534 |
| Brier | 0.0540 |
| discrimination | 0.8529 |
| radar reproducibility across replicates | 0.9596 |

The shipped engine fits its chemistry map on all 375 spectra, as a shipped engine should. The
in-sample figures (0.9547, AUC 0.9990) describe that fitted object and are reported for
completeness with an explicit `IN_SAMPLE_WARNING` in the artifact. **The number to quote is
0.8507**, computed with the chemistry model refitted inside each of the five molecule-grouped
folds, so that no spectrum of a test molecule ever informs its own prediction. The gap between
0.955 and 0.851 is precisely the in-sample inflation that R-10 warns about, and it is stated here
rather than buried.

The radar is reproducible at cosine 0.960 across replicate spectra of the same molecule, which is
what makes it usable as a report to a human.

---

## 6. Robustness

Seven perturbations at five levels each, applied to the raw spectrum and run through the complete
engine (35 conditions × 375 spectra).

| output | mean over all conditions |
|---|---|
| molecule top-1 | 0.8106 |
| chemistry top-1 | 0.8890 |
| radar cosine to the clean radar | **0.9648** |

The ordering is the important result. Under every perturbation the radar degrades most slowly,
chemistry next, and molecule identity fastest. That is the correct ordering for a chemistry-level
answer: when the signal degrades, the engine should lose confidence in *which* molecule before it
loses its reading of *what kind of chemistry* is present. Baseline drift at level 0.8 is the
worst case (molecule 0.584, chemistry 0.723, radar 0.867), and even there the radar retains most
of its shape.

Shot noise is essentially free (radar cosine ≥ 0.998 at every level), which is consistent with
asLS + Savitzky–Golay absorbing high-frequency perturbations before projection.

## 7. Failure behaviour

The engine emitted an **unknown** warning on 86 spectra and an **outlier** warning on 18. Both are
inherited thresholds from Phase 05; neither was tuned in this phase (P-13).

The low-reconstruction tail is chemically coherent: pyruvate (0.209), thymine (0.258–0.286), malic
acid (0.262–0.312), urea (0.345). These are small, high-symmetry molecules with few strong bands
in 450–1800 cm⁻¹ — exactly the molecules a motif dictionary built from larger biomolecules should
struggle to express. The engine reports low explained variance and low confidence on them rather
than answering confidently, which is the behaviour the architecture was designed for.

Two failure shapes recur in the representative reports:

- **Singleton molecules** (rank 154): the answer is absent from the bank. The engine still returns
  the correct chemistry family with high confidence — trierucin, lauric acid and carotene are all
  classed correctly while being unretrievable as molecules.
- **Adjacent-class confusion**: stearate, a fatty acid, is predicted `acylglycerol` at CSM
  EV 0.945. The spectrum is well explained; the ontology boundary is the difficulty, not the
  representation.

One limit of the warnings was measured while writing the regression tests and is worth stating
plainly: **white noise is not reliably flagged.** It reconstructs at CSM explained variance ≈ 0.61,
above the 0.50 `unknown` floor, so a warning fires on only 1 of 20 random spectra. Confidence
still separates it cleanly — noise peaks at 0.495 against a corpus mean of 0.803 — so the correct
operator rule is *read the confidence, not the flag*. See `PHASE_09_SCIENTIFIC_AUDIT.md` C5b.

48 complete inference reports (best / median / worst × 16 families) are written to
`reports/examples/` and are the recommended starting point for anyone auditing engine behaviour
on a specific chemistry.

---

## 8. Gates

All 16 pass.

| gate | |
|---|---|
| G1 | frozen fingerprints verified on engine load |
| G2 | no new representation, optimisation or heuristic introduced |
| G3 | BSV2, PCA, UMAP, clustering and geometry absent from the inference path |
| G4 | engine holds no mutable state |
| G5 | engine is deterministic on repeat |
| G6 | every spectrum processed, no exceptions |
| G7 | every retrieval score reconciles |
| G8 | retrieval reproduces the frozen baseline exactly |
| G9 | LSM, CSM, retrieval and chemistry all validated |
| G10 | representative reports for every chemistry family |
| G11 | noise robustness measured end to end |
| G12 | radar labelled relative evidence, not concentration |
| G13 | provenance tree complete for every spectrum |
| G14 | Raman-only scope |
| G15 | chemistry accuracy reported held-out, not only in-sample |
| G16 | retrieval confidence temperature fitted in-fold |

G15 and G16 were added during the phase after two defects were found; see
`PHASE_09_SCIENTIFIC_AUDIT.md` §D.

---

## 9. The result the architecture rests on

Chemistry-class accuracy on **unseen molecules**, measured identically across the rebuild:

| representation | held-out fine-class top-1 |
|---|---|
| raw preprocessed spectrum | 0.608 |
| LSM activation | 0.850 |
| **CSM activation** | **0.855** |
| 11 grounded axes | 0.664 |
| themes | 0.405 |
| Meta Components | 0.392 |

Information rises from the raw spectrum to the CSM layer and **does not rise further**. Four
independent attempts to build a layer above the CSM — themes, Meta Components, grounded axes,
geometric coordinates — each lost information. Phase 09 ships the layer where the information
actually is, and treats everything above it as a *description* of that layer rather than a stage
of it.

---

## 10. Interpretation limits

These are constraints on what the engine's output means, not caveats about its quality.

1. **The radar is relative biochemical evidence.** It is not a concentration, not an abundance,
   and not a mixture decomposition. A tall spoke means "the spectrum contains evidence associated
   with this chemistry", not "there is a lot of this chemistry".
2. **Pure Raman reference spectra only.** Nothing here licenses a SERS, serum, EV or biofluid
   claim. Transfer to those regimes is unmeasured in V7 and must be established separately.
3. **A peak is not a molecule.** Retrieval returns ranked candidates with a score decomposition,
   deliberately, and top-1 at 0.605 is the honest figure for how often the single best guess is
   right.
4. **Sixteen classes are a curated cut through a continuum**, not a discovery. Phase 06.5 showed
   the CSM manifold has *no* preferred cluster count; the ontology is a reporting convention that
   generalises well (0.851), not a natural kind.
5. **In-sample chemistry figures describe the shipped fit, not expected performance.**
6. **R-01 (class-prior bias) remains OPEN.** The corpus is not balanced across the 16 families
   (n ranges from 3 to 80), and the chemistry layer inherits that prior.

---

## 11. Recommendation

**Freeze the V7 architecture here.** Phase 09 completes the arc the rebuild set out on: one
verified path, validated end to end on every spectrum, with every number reproduced from frozen
artifacts and every exclusion justified by a measurement. Four separate attempts to add a layer
above the CSM have now failed on evidence, and a fifth would be a re-run of the same experiment.

The productive next work is not architectural. It is **corpus** — 66 of 154 molecules have a
single spectrum, which alone caps molecule top-1 at 0.819 — and **transfer**, establishing what
survives the move from pure Raman to the applied regimes GAIRA ultimately targets. Both are
scientific questions. Neither requires changing the engine.

---

*Artifacts: `results/v7_rebuild/phase09/`. Companion documents:
`PHASE_09_ENGINE_SPEC.md`, `PHASE_09_MATHEMATICAL_APPENDIX.md`, `PHASE_09_DECISION_GATE.md`,
`PHASE_09_SCIENTIFIC_AUDIT.md`, `PHASE_09_FIGURES.pdf`.*
