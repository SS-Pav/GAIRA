# GAIRA V7 — Phase 08: Hierarchical Molecular Retrieval

**Status** COMPLETE — 18 of 18 gates pass.
**Scope** Raman only. **BSV2 is not on the inference path** and is not imported anywhere in
`src/gaira/v7/retrieval/`.
**Frozen inputs verified** LSM `208482d6…` · CSM `0b4aa550…` · engine `20d8bd99…`
**Decision** **Outcome A — keep direct CSM retrieval.**

---

## 1. Executive summary

The question was whether chemistry-aware reranking beats direct CSM retrieval. **It does not,
and the reason is unusually clean: the nested weight search chose to use no chemistry at all.**

In every one of the five outer folds, the inner cross-validation selected

> α = 0.4, **β = 0, γ = 0, δ = 0**

Zero weight on chemistry similarity, zero on diagnostic band support, zero on the incompatibility
penalty. Model C therefore reduces to Model B exactly, and the difference between them is
**+0.0000, not a small number that happens to be non-significant**.

| model | top-1 | top-3 | top-5 | top-10 | MRR | nDCG@5 | median rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A** raw spectrum | **0.6347** | 0.7333 | 0.7653 | 0.7920 | 0.6919 | 0.7056 | 1 |
| **B** CSM *(canonical)* | 0.6053 | **0.7627** | **0.7947** | **0.8107** | 0.6870 | 0.7112 | 1 |
| **C** chemistry rerank | 0.6053 | 0.7627 | 0.7947 | 0.8107 | 0.6870 | 0.7112 | 1 |
| **D** probabilistic | 0.4027 | 0.5600 | 0.6187 | 0.7147 | 0.5136 | 0.5249 | 2 |
| **E** Bayesian fusion | 0.6160 | 0.7493 | 0.7920 | 0.8107 | **0.6911** | **0.7130** | 1 |

Paired against Model B, with molecule-level bootstrap CIs:

| model | Δ top-1 | 95% CI | McNemar p | significant |
|---|---:|---|---:|---|
| A raw spectrum | +0.0293 | [−0.0119, +0.0738] | 0.135 | no |
| **C chemistry rerank** | **+0.0000** | [+0.0000, +0.0000] | 1.000 | **no** |
| D probabilistic | −0.2027 | [−0.2857, −0.1137] | <0.001 | no (worse) |
| E Bayesian fusion | +0.0107 | [−0.0286, +0.0492] | 0.557 | no |

**Nothing beats CSM retrieval significantly.** Raw spectrum wins top-1 by 0.029 and loses top-5
by 0.029; the difference is not significant either way, and CSM is preferred on the metrics that
matter for a shortlist engine.

## 2. Two bugs that would have produced the opposite conclusion

Both were found by looking at results that were too clean, and both are recorded because the
first version of this phase reported **"outcome C — adopt hierarchical retrieval"**.

**Bug 1 — Models C, D and E were scored against a smaller candidate bank than Model B.** Model B
retrieved against all 154 molecules minus the held-out spectrum; C, D and E retrieved against
*training-fold molecules plus the query's own*, roughly 123. A smaller candidate set is a
strictly easier retrieval problem. The tell was in the failure analysis: **94 spectra improved
and exactly zero worsened.** A real model change moves ranks in both directions; only a bank
change moves them one way. Fixed so every model sees an identical bank; only the *chemistry
model* remains fold-restricted, which is correct.

**Bug 2 — confidence was derived from `1/rank`, and correctness is `rank ≤ 1`.** Calibrating a
function of the rank against the rank is circular, and it produced a discrimination of exactly
**1.000**. Fixed to the score margin (top-1 minus top-2), which is what an engine actually has at
inference. Honest calibration: ECE 0.121, discrimination 0.691.

## 3. Baseline reproduction

Reproduced **exactly**, to 1e-9, before any new model was measured:

| metric | Phase 05 | reproduced |
|---|---:|---:|
| CSM top-1 | 0.605333 | 0.605333 |
| CSM top-3 | 0.762667 | 0.762667 |
| CSM top-5 | 0.794667 | 0.794667 |

## 4. Model C, and why it chose no chemistry

$$S_{\text{total}} = \alpha \cdot s_{\text{CSM}} + \beta \cdot s_{\text{chem}}
+ \gamma \cdot s_{\text{band}} - \delta \cdot \text{penalty}$$

144 weight settings, searched **only in inner folds**. Every term is bounded in [0, 1] so the
score decomposes exactly, and **no candidate is ever filtered** — molecules outside the reranked
shortlist keep their CSM score and remain reachable, so a chemistry error is always recoverable.

The search returned β = γ = δ = 0 in all five folds. Two independent measurements confirm that
this is the data speaking and not a search failure:

- **Chemistry-axis permutation importance is exactly 0.0000 for all sixteen axes.** Shuffling any
  chemistry axis across spectra changes the mean reciprocal rank not at all.
- **Failure analysis: 0 helped, 0 hurt, 375 unchanged.**

The interpretation is straightforward and was foreseeable from Phase 06: **Chemistry Evidence is
computed *from* the CSM activations.** It is a 16-dimensional summary of the same 49-dimensional
vector already being used for retrieval. It carries no information about molecular identity that
the CSM layer does not already carry, so a linear combination of the two can do no better than
the CSM term alone — and the weight search discovered exactly that.

## 5. Split B — held-out molecules

Molecule top-1 is **undefined**, not zero: the correct answer is not among the candidates.

| model | chemistry top-1 | top-3 | macro-F1 | balanced acc. | nearest analogue class correct |
|---|---:|---:|---:|---:|---:|
| B CSM | 0.8453 | 0.9707 | 0.8053 | 0.8394 | 0.8453 |
| C chemistry rerank | 0.8453 | 0.9707 | 0.8053 | 0.8394 | 0.8453 |

Identical, for the same reason. When the molecule is absent, the top hit is the right *chemistry*
84.5% of the time — the engine's most useful behaviour on an unrepresented compound.

## 6. Calibration and abstention

| model | ECE | Brier | log loss | sharpness | discrimination |
|---|---:|---:|---:|---:|---:|
| B CSM | 0.1205 | 0.2260 | 0.6934 | 0.2062 | 0.6913 |
| C chemistry rerank | 0.1226 | 0.2263 | 0.6887 | 0.2005 | 0.6908 |

**When should GAIRA say "I don't know"?** The risk–coverage curve answers it: accuracy of 0.80
is reachable at **coverage 0.28**, and 0.90 is **not reachable at any coverage**. On a 154-way
problem with 66 single-spectrum molecules, an engine that answers only its most confident quarter
is right four times in five; one that answers everything is right three times in five. That is
the honest operating range, and it argues for shipping a shortlist with a confidence rather than
a single answer.

## 7. Noise robustness

Seven perturbations × five levels. Mean perturbed top-1: raw 0.926, CSM 0.828, chemistry rerank
0.828.

**Raw spectrum appears most robust, and the number is misleading.** This evaluation uses an
in-sample bank, so a perturbed spectrum still matches its own unperturbed reference — raw
retrieval is measuring self-similarity, not generalisation. The comparison that matters is Phase
06's held-out one, where the CSM layer beat the raw spectrum 0.845 to 0.592 on chemistry with
better retention. The figure states this on its face.

## 8. Explainability

**120 decompositions checked, 0 non-reconciling.** Every Model C score is four weighted terms
that sum to the displayed total, checked against the model's own output. For every retrieved
molecule the engine reports the similarity score, the supporting CSMs with their share of the
similarity, the supporting chemistry axes, the supporting diagnostic bands, the LSMs beneath
each CSM, and the full contribution table. There is no hidden term.

## 9. Limitations

1. **The null result is specific to this construction.** Chemistry Evidence is a deterministic
   function of the CSM activations; a chemistry channel derived *independently* — from band
   shape, from a second modality, from an external database — might behave differently. Nothing
   here bears on that.
2. **Model D underperforms badly** (top-1 0.403). A 154-way multinomial logistic on 375 spectra
   is over-parameterised; the result is about sample size, not about probabilistic retrieval.
3. **Model E's independence assumption is false** — chemistry is computed from CSM — so its
   +0.011 is an upper bound on naive fusion, not a probability.
4. **66 of 154 molecules have one spectrum** and leave the bank entirely under leave-one-out.
   Split A therefore has a structural ceiling below 1.0.
5. **Noise robustness is in-sample** (§7).
6. **Accuracy ≥ 0.90 is unreachable at any coverage.** If a downstream use needs that, this
   corpus does not support it.

## 10. Reproduction

```bash
PYTHONPATH=src python results/v7_rebuild/phase08/code/run_phase08.py   # ~9 min
PYTHONPATH=src python results/v7_rebuild/phase08/code/make_figures.py
PYTHONPATH=src python results/v7_rebuild/phase08/code/make_pdf.py
PYTHONPATH=src python -m pytest tests/test_v7_phase08.py -q
```
