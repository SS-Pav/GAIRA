# GAIRA V7 — Phase 08 Scientific Audit

Every conclusion attacked before it is allowed to stand. The brief asks specifically: *does
chemistry actually improve retrieval, or is CSM already sufficient?* — and requires that
improvements which are not statistically significant, and complexity that produces minimal gain,
both be rejected.

---

## A. The central claim

### "Chemistry-aware reranking does not improve molecular retrieval."

*Attack 1 — was the weight grid too coarse?* 144 settings over α ∈ {0.4, 0.6, 0.8, 1.0},
β ∈ {0, 0.1, 0.2, 0.4}, γ ∈ {0, 0.05, 0.1}, δ ∈ {0, 0.1, 0.3}. β = 0 was chosen over β = 0.1 in
every fold, so the boundary is interior, not at the edge of the grid. A finer grid would find
smaller non-zero weights, not better ones.

*Attack 2 — was the chemistry similarity badly constructed?* It is cosine between the query's
calibrated 16-axis evidence and the candidate molecule's mean evidence, both from the frozen
Phase 06 model. If a better chemistry similarity exists, this phase has not found it — but the
permutation-importance result is the stronger evidence: **shuffling any chemistry axis across
spectra changes MRR by exactly 0.0000.** That is not a statement about one similarity function;
it is a statement that the chemistry channel carries no rank-relevant signal at all.

*Attack 3 — is the null a power problem?* No. A power problem gives a small non-significant
effect. This gives **exactly zero**, in 375 of 375 spectra, with 0 helped and 0 hurt.

**Verdict: the claim survives, and it is stronger than "not significant". It is identically zero
by construction, because the weight search selected it away.**

### Why the result was foreseeable

Chemistry Evidence is a deterministic function of the CSM activations. Adding `f(x)` to `x` in a
linear score cannot add information about a target that `x` already determines. Phase 06 measured
this from the other direction — the 16-d layer costs 0.010 of class top-1 relative to the 49-d
layer it summarises — and this phase is the retrieval-side corollary.

**A referee would ask why the phase was run at all.** The answer is that "obvious" is not
"measured", the brief asked for it, and the measurement produced two things the argument alone
could not: the exact zero, and the discovery of two bugs that would have produced the opposite
answer.

---

## B. The bugs, and what they nearly cost

| # | defect | what it produced | how it was caught |
|---|---|---|---|
| 1 | **Models C/D/E scored against a ~123-molecule bank; Model B against 154.** | Δ top-1 = **+0.0213, p = 0.0078, "significant"** → decision outcome **C, adopt hierarchical retrieval**. | The failure analysis: **94 helped, 0 hurt.** A model change moves ranks both ways; only a bank change moves them one way. |
| 2 | **Confidence derived from `1/rank` while correctness is `rank ≤ 1`.** | discrimination exactly **1.000**, ECE 0.052 — a "perfectly discriminating" calibration. | The number itself. A discrimination of exactly 1.000 on a task with 60% accuracy is not a result. |

Bug 1 is the serious one. **Without the failure-analysis panel this phase would have recommended
an architecture change on an artefact.** Gate **G7b** — every model scored against an identical
candidate bank — now exists and is tested.

Both bugs share a shape worth naming: *a comparison in which the two arms were not given the same
problem*. It is the retrieval-side cousin of the leakage found in Phase 06's semantic comparator
and Phase 06.5's ensemble, which is now three phases in a row.

---

## C. Other claims, attacked

**"Raw spectrum retrieval is competitive."** Top-1 0.635 vs CSM 0.605, p = 0.135 — not
significant, and raw *loses* top-5 (0.765 vs 0.795) and top-10. On a shortlist engine top-5
matters more than top-1. **Supported as "not distinguishable", rejected as "better".**

**"Model E (Bayesian fusion) is the best MRR."** 0.6911 vs 0.6870, p = 0.557. **Not
significant.** And its conditional-independence assumption is false by construction — chemistry
is computed from CSM — so even a significant result would not have been interpretable as a
probability. Correctly labelled benchmark-only.

**"Model D shows probabilistic retrieval fails."** No. It shows a 154-class multinomial logistic
on 375 spectra fails, which is a sample-size result. **The claim must not be generalised.**

**"The engine is fully explainable."** 120 decompositions, 0 non-reconciling, each score four
weighted terms summing to the displayed total. **Supported** — with the caveat that explaining a
score is easy when three of its four terms carry zero weight. The decomposition machinery is
correct; it has not been stress-tested on a model that actually uses all four channels.

**"Noise robustness favours the raw spectrum."** Measured, and **misleading as stated**: the bank
is in-sample, so raw retrieval is matching a perturbed spectrum to its own unperturbed reference.
The report says so; the number should never travel without that sentence.

**"Accuracy 0.90 is unreachable at any coverage."** Supported, and it is the most operationally
important sentence in the phase.

---

## D. Complexity versus gain

The brief requires rejecting complexity that produces minimal gain. Model C adds four weights, a
band-support computation, an incompatibility penalty and a shortlist parameter — **for exactly
zero gain**. Rejected on both grounds simultaneously, which is the cleanest case the rule can
produce.

---

## E. Conclusions by strength

**Strongly supported** — chemistry-aware reranking gives exactly zero improvement; the weight
search selected zero chemistry weight in all five folds; permutation importance is zero for all
sixteen axes; baselines reproduce to 1e-9; every score decomposition reconciles; Split B
chemistry performance is unchanged at 0.845.

**Weakly supported** — that raw spectrum is *worse* than CSM (not significant either way); that
Model E's small MRR edge means anything.

**Unsupported** — any claim that probabilistic retrieval is unsuited to this problem (Model D's
failure is about n); any claim about noise robustness ordering from §7's in-sample numbers.

---

## F. What would overturn this

A chemistry channel **not derived from the CSM activations**. Band shape — width, asymmetry,
splitting — was flagged as the highest-value missing feature in Phase 05 and is independent of
the motif projection in a way that Chemistry Evidence is not. If such a channel existed, this
phase's machinery would test it in one run.

---

## G. Overall

The phase answers its question decisively in the negative, and the negative is exact rather than
marginal. Its main scientific contribution is not the null itself — which was foreseeable — but
the demonstration that the null survives a properly-constructed comparison, together with the
two bugs that a carelessly-constructed one would have hidden.

**Confidence in outcome A: 9 / 10.** The deduction is for §C's last point: the explainability
machinery has not been exercised on a model that uses all four channels, so its correctness under
non-zero chemistry weight is asserted by unit test rather than demonstrated on real data.
