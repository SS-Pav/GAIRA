# GAIRA V7 Phase 04 — Scientific Audit

Adversarial review of the frozen projection engine. The task is to falsify it, not to defend it.

**Verdict: the engine is sound as an internal foundation and is not ready to be described as a
Raman foundation model.** Three claims in the report need weakening, one capability is absent,
and the evidence base is one corpus.

---

## 1. Attempted falsifications, and what survived

| attack | result |
|---|---|
| *Is anything being fitted at inference?* | **No.** Static check over the inference path finds no `fit`/`fit_transform`/`partial_fit` and no RNG. Survives. |
| *Does a spectrum's answer depend on its batch?* | **No.** Identical alone and inside a batch, verified. Survives. |
| *Is it deterministic?* | **Yes**, bit-identical on re-run. Survives. |
| *Is the held-out evaluation actually held out?* | **Partly.** Spectrum-level leakage is removed by construction. Dictionary-level leakage is real, measured at **+0.055 top-1**, and remains in every split-A number. Survives with a stated correction. |
| *Are the activations degenerate?* | **Nearly.** Only ~2 of 49 CSMs activate per spectrum under NNLS on the CSM basis. Sparsity is desirable; this is at the edge of too sparse, and it is what made the softmax theme mode manufacture evidence. **Does not fully survive** — see W2. |
| *Has the theme layer collapsed?* | **It had.** The softmax mode gave all four themes activation on spectra with evidence for one. Caught by the zero-evidence veto and rejected. Survives only because the check was added. |
| *Does OOD detection work?* | **No.** AUROC 0.548 on real Ag-SERS. **Fails.** |
| *Is confidence usable?* | **No.** ECE 0.486. **Fails.** |
| *Do the geometry coordinates mean anything?* | **Yes** — neighbourhood purity 4.06× chance. Survives. |
| *Is the explanation chain real?* | **Yes**, every activated theme resolves to CSMs, LSMs and named molecules. Survives. |

## 2. Strengths

**S1. The leakage control is the most important thing in the phase.** Refitting the dictionary
per fold to measure what the frozen one is worth is rarely done and is the only way to know
whether a frozen-atlas benchmark means anything. +0.055 is a small and credible number.

**S2. Four defects were found that each changed a conclusion**, and each was demonstrated
before being changed: the impossible molecule-retrieval split, the theme-collapse mode, the
mis-indexed explanation, and the OOD score computed on the reconstruction. Two of them would
have produced a *better-looking* result if left in.

**S3. The central finding is a real generalisation result.** Class retrieval on unseen
molecules rises 0.608 → 0.855 through the frozen dictionaries. That is transfer, not recall.

**S4. Selection rules are hard constraints where they should be.** Physical admissibility
(no negative mass) and zero-evidence leakage are vetoes, not weighted terms.

**S5. The failing gate was left failing.**

## 3. Weaknesses

**W1. One corpus, in-domain, 375 spectra.** Every positive number is pure Raman on the corpus
the atlas was built from. There is no second dataset.

**W2. Extreme activation sparsity.** ~2 of 49 CSMs per spectrum means most of the dictionary is
never used by any single spectrum, and three of four themes routinely receive zero evidence.
Sparse is good; this may be *degenerate* sparse, and no test distinguishes the two.

**W3. The LSM layer is not on the inference path.** Direct CSM projection beat every LSM
aggregation. Phase 01's 50 motifs are, for inference purposes, an explanation device. That is
worth stating more prominently than it currently is.

**W4. The BSV has effective rank 2.40 of 4.** Calling it a four-dimensional biochemical state
is generous by 40%.

**W5. Six benchmark selections, each by a composite of two criteria**, with no sensitivity
analysis over the weighting.

**W6. The geometry extension is benchmarked on 49 self-referential points.**

**W7. Confidence is a product of four heuristic terms** with no calibration step, and its ECE
shows it.

## 4. Unsupported or over-stated claims

**U1. "The representation is genuinely reusable."** Supported for chemistry class on unseen
molecules *within this corpus*. Not supported across corpora, instruments or modality — and
the one cross-modality test failed. **Recommend: "reusable across unseen molecules of the same
corpus and modality".**

**U2. "Foundation-model style projection system."** A foundation model is expected to transfer
out of its training distribution. This one demonstrably does not detect when it has left it.
**Recommend: drop the phrase until an out-of-domain result exists.**

**U3. "Uncertainty propagates upward."** Four uncertainty channels are computed and carried,
which is true. But they are combined into a confidence with ECE 0.486, so propagation is
*implemented*, not *validated*. **Recommend: separate the two claims.**

**U4. Split-A molecule top-1 of 0.806** carries +0.055 of dictionary leakage. The honest
figure is ~0.75.

## 5. Likely reviewer criticisms

**R1.** *"Your OOD detector fails on the only real out-of-domain data you have. Why should I
trust any confidence this engine reports?"* — The strongest objection. Answer: the confidence
does not depend on the OOD term alone, but it is also badly calibrated, so the honest answer is
that today's confidence should be treated as a ranking, not a probability.

**R2.** *"You benchmarked six estimators and picked elastic net over NNLS on a 0.010 difference
in replicate consistency."* — Fair. The choice is defensible but not decisive, and NNLS would
have been an acceptable answer.

**R3.** *"Two of 49 CSMs activate. Is your dictionary 49 components or 2?"* — Effective rank of
the *dictionary* is 21.9 of 50; per-spectrum activation is much sparser. Both numbers should be
quoted together, and currently only one is prominent.

**R4.** *"If direct CSM projection beats every LSM aggregation, what is the LSM layer for?"* —
Explanation and the future SERS observation model. That is a design answer, not an empirical one.

**R5.** *"Themes retrieve worse than raw spectra on both axes. What is the theme layer for?"* —
Replicate consistency and a four-number summary for comparison and trajectory. A reviewer may
find that thin.

**R6.** *"Effective rank 2.40 — are two of your four themes redundant?"* — Not tested here.

## 6. Recommended experiments

**E1 (highest value).** An independent Raman corpus, different instrument. Nothing else
addresses W1 or U1.

**E2.** A calibration step for confidence — isotonic or Platt on held-out folds — and re-measure
ECE. Converts U3 into a result.

**E3.** A genuine OOD study: SERS, mineral, polymer and pure-noise probes, with a score that
uses the *un-modelled* structure of the residual rather than its magnitude alone. The residual
magnitude of a SERS spectrum is small precisely because the bands overlap; its residual
*structure* may not be.

**E4.** Sensitivity of all six benchmark selections to their criterion weights.

**E5.** Test whether the two low-variance BSV axes are redundant (W4, R6).

**E6.** Ablate the LSM layer entirely and re-measure. If nothing changes, say so.

## 7. Remaining risks

| risk | severity | status |
|---|---|---|
| no out-of-domain capability (U2, R1) | **high** | measured and failing; E3 |
| single corpus (W1, U1) | **high** | E1 |
| confidence uncalibrated (W7, U3) | **medium** | E2 |
| activation sparsity may be degenerate (W2, R3) | medium | untested |
| dictionary leakage in every in-sample number | medium | quantified at +0.055 |
| LSM layer not load-bearing (W3, R4) | low | design decision, now explicit |
| benchmark selections not weight-robust (W5) | low | E4 |

## 8. Verdict

The engine does what Phase 04 set out to build: a deterministic, batch-independent,
fully-explainable projection path with uncertainty carried at every level and leakage measured
rather than assumed. The hierarchy *does* support inference about unseen molecules, and the
CSM layer is where that inference lives.

It is **not** a foundation model in the sense the word implies, because it cannot yet tell when
a spectrum is outside its domain — and on the one real test of that, it was at chance.

**Approved as the foundation for Phase 05 in-domain validation. Not approved for any claim of
out-of-domain or cross-modality capability, and not ready for external submission without E1,
E2 and E3.**
