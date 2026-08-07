# GAIRA V7 — Phase 08 Decision Gate

## Should hierarchical molecular retrieval replace direct CSM retrieval?

# **No. Outcome A — keep direct CSM retrieval.**

---

## The pre-declared rule

| outcome | condition | action |
|---|---|---|
| **A** | no improvement | keep CSM |
| B | small but significant improvement | optional reranking |
| C | consistent, statistically significant improvement | adopt hierarchical retrieval |

Only C permits an architecture change.

## What was measured

| | value |
|---|---|
| Δ top-1, chemistry rerank vs CSM | **+0.0000** |
| 95% CI (molecule bootstrap) | [+0.0000, +0.0000] |
| McNemar p | 1.000 |
| Δ top-5 | +0.0000 |
| Δ MRR | +0.0000 |
| chemistry-axis permutation importance | **0.0000 for all 16 axes** |
| spectra helped / hurt / unchanged | **0 / 0 / 375** |
| weights selected in inner folds | α = 0.4, **β = γ = δ = 0**, in **all five folds** |

**This is not a small improvement that failed a significance test. It is exactly zero**, because
the nested weight search — given 144 settings and free choice — assigned zero weight to every
chemistry term in every fold.

## Why

Chemistry Evidence is computed *from* the CSM activations. It is a 16-dimensional summary of the
49-dimensional vector already used for retrieval, so it carries no information about molecular
identity that the CSM layer does not already carry. A linear combination of a vector and a
function of that vector cannot beat the vector.

Two independent measurements agree: permutation importance is identically zero across all
sixteen chemistry axes, and no spectrum's rank moved in either direction.

## The other models

| model | Δ top-1 vs CSM | p | verdict |
|---|---:|---:|---|
| A raw spectrum | +0.0293 | 0.135 | not significant; loses top-5 by the same margin |
| D probabilistic | −0.2027 | <0.001 | significantly **worse** — over-parameterised at n = 375 |
| E Bayesian fusion | +0.0107 | 0.557 | not significant; its independence assumption is false |

**Nothing beats CSM retrieval significantly.**

## What this does not say

It does not say chemistry is useless. Phase 06 established that the Chemistry Evidence layer
predicts chemistry class on unseen molecules at 0.845 — the engine's strongest capability — and
this phase confirms it: when the molecule is absent, the top hit is the right chemistry 84.5% of
the time.

It says something narrower and more useful: **chemistry derived from CSM adds nothing to
molecular ranking**. A chemistry channel derived *independently* of the CSM activations is
untested and is the obvious next experiment if this direction is pursued.

## Architecture

Unchanged. The canonical path stays:

```
spectrum → preprocessing → LSM → CSM → Chemistry Evidence → molecular retrieval on CSM
```

with Chemistry Evidence retained as the interpretable output layer and the 16-axis radar, not as
a retrieval prior. Nothing is added to the inference path.

## Required before any future attempt

1. **Derive a chemistry channel that is not a function of the CSM activations.** Band shape,
   width and asymmetry are the obvious candidate and were flagged in Phase 05 §13.
2. **Ship the shortlist with its confidence.** Accuracy 0.80 at coverage 0.28; 0.90 unreachable.
3. **Keep the bank-identity check.** It is now gate G7b and it is the reason this phase reports A
   rather than C.

## Not done

Phase 09 has not been begun.
