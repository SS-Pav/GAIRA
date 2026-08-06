# Phase 08 — Hierarchical Molecular Retrieval

**Status:** Not started — blocked by Phase 07 / DG-07
**Architectural decision under test:** A-21.

---

## Purpose

Use chemistry as a **prior** on molecular identity, and find out whether it beats direct cosine
retrieval in CSM space.

## The baseline this phase must beat

Phase 05, Split A (the molecule is present in the reference bank), direct cosine over the
49-dimensional CSM activation vector:

| metric | value |
|---|---:|
| molecule top-1 | **0.605** |
| molecule top-3 | 0.763 |
| molecule top-5 | 0.795 |
| calibration ECE | 0.130 (discrimination 0.891) |

Chance on a 154-way problem is 0.006. Anything that does not beat 0.605 is not adopted.

## Inputs

- CSM activation vector `c(x) ∈ ℝ₊^49` — the canonical representation
- Chemistry Evidence `e(x) ∈ ℝ₊^16` — as a **soft prior**, never a hard filter
- BSV2, **if and only if** DG-07 passed
- the frozen 154-molecule reference bank and the frozen folds

## What to implement

1. **Soft chemistry prior.** `score(x, a) = f(sim(c(x), r_a)) + λ · log e_κ(a)(x)`, with `λ`
   fitted on training folds only, nested inside the CV so no test spectrum influences it.
2. **Class-conditioned retrieval.** Rank within chemistry, then across — with the class
   posterior carried, not thresholded.
3. **Prototype + residual scoring.** Match the class prototype first, then score the residual
   against within-class molecules. This is where fine discrimination should live.
4. **Hierarchical ranking.** Combine the above into one ranked list.
5. **Top-k and confidence**, calibrated as in Phase 05 and selected on Brier with the
   informativeness floor.
6. **Conformal prediction sets — only if justified.** A set with a coverage guarantee is worth
   more than a top-5 list, but only if the exchangeability assumption holds under
   molecule-grouped splits. If it does not, say so and do not ship it.

### The design constraint that matters

**A hard class filter makes a class error unrecoverable.** Chemistry Evidence is 0.845 accurate
on unseen molecules, so roughly one spectrum in six would have its correct molecule filtered out
before scoring began. The prior must be soft and must be overridable by strong spectral
evidence. Any implementation that cannot recover from a class error is rejected at the gate.

## Evaluation

| Axis | Measure |
|---|---|
| retrieval | top-1, top-3, top-5, MRR — against the 0.605 / 0.763 / 0.795 baseline |
| per-class | retrieval broken down by chemistry class; report where the prior helps and where it hurts |
| per-molecule | the failure list, by name — which molecules remain unretrievable and why |
| OOD behaviour | does the prior degrade rejection? Re-run the Phase 05 rejection channels |
| noise robustness | the same 7 × 5 grid; does the prior amplify or damp corruption? |
| calibration | ECE, Brier, sharpness, discrimination |
| ablation | prior off / prior on / prior hard-filtered — the third as a negative control |

## Decision Gate DG-08

| Check | Requirement |
|---|---|
| **Scientific** | molecule top-1 exceeds 0.605 with the improvement significant at α = 0.05 after correction (McNemar + permutation on the frozen folds) |
| **Scientific** | no chemistry class is made **worse** by the prior without that being reported by name |
| **Scientific** | open-set rejection is not degraded relative to Phase 05 (joint AUROC ≥ 0.921) |
| **Scientific** | calibration remains informative (P-18) |
| **Scientific** | the hard-filter negative control is reported, demonstrating the soft prior's necessity |
| **Engineering** | deterministic; `λ` frozen; no fitting at inference; batch-independent |
| **Architecture** | the prior is soft and overridable; a class error is recoverable |
| **Decision** | **Proceed** to Phase 09 (the V5 replacement decision) · **Repeat** with a corrected prior · **Redesign** — retrieval stays direct-cosine and the chemistry layer remains interpretive only |
