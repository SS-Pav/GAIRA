# Phase 07 — BSV2 Discovery

**Status:** Not started — blocked by Phase 06 / DG-06
**Architectural decision under test:** A-20.

---

## Purpose

Discover **biochemical programmes** — patterns of chemistry co-occurrence — by hierarchical NMF
over the Chemistry Evidence matrix, and determine whether they earn a place in the architecture.

## Input — one input, and the restriction is the point

**Chemistry Evidence only.** `Ev ∈ ℝ₊^{375×16}`, spectra × chemistry evidence.

**Not** CSM activations. This is the distinction from the archived Meta Components (A-15), which
factorised `A ∈ ℝ₊^{375×49}` — spectra × *motif* activations — and retained 0.185 of the CSM
layer's information before being discarded. BSV2 compresses a different object. Whether that is
enough is the open question (U-04), and this phase is allowed to answer *no*.

## Method

Hierarchical NMF, `Ev ≈ W P` with `W ∈ ℝ₊^{375×K}`, `P ∈ ℝ₊^{K×16}`, both non-negative.

**K sweep:** 2, 3, 4, 5, 6, 8, 10, 12, 14.

## Evaluation — eight axes, all measured, none skipped

| Axis | Measure |
|---|---|
| reconstruction | relative Frobenius error, explained variance |
| held-out chemistry prediction | class top-1/top-3 from BSV2 on molecule-grouped folds |
| programme stability | bootstrap component recovery (Hungarian-matched), consensus dispersion |
| interpretability | programme purity by chemistry against the base rate; is each programme nameable? |
| mutual information | I(BSV2; class) against I(Chemistry Evidence; class) |
| noise robustness | the same 7 × 5 perturbation grid, AURC |
| calibration | ECE, Brier, sharpness, discrimination |
| compression | dimensions used vs information retained |

## Selecting K

**On a Pareto frontier over all eight axes. Never on reconstruction alone** (R-12), and never on
raw accuracy alone. The weighting is pre-registered before the sweep runs. The frontier and the
rejected points are both reported — no cherry-picking (the Phase 04.5 rule, carried forward).

The K diagnostic must additionally report the **best achievable** held-out chemistry prediction
over every (variant, K) combination tested, as a diagnostic that is explicitly excluded from
selection. Phase 04.5 showed this catches the "a different K would have saved it" objection
before a reviewer raises it.

## Output

**Biochemical Programmes.** Not themes (LEGACY, A-13). Not manual mappings. Not Meta Components
(LEGACY, A-15). A frozen `P`, a projection rule, and a name for each programme derived from the
chemistry it loads on.

## Decision Gate DG-07

| Check | Requirement |
|---|---|
| **Scientific** | BSV2 retains **≥ 0.50** of Chemistry Evidence's information *and* **≥ 0.50** of its held-out class prediction — the informativeness floor, **pre-registered before the sweep** (P-18) |
| **Scientific** | any stability, robustness or calibration gain counts **only after** the floor is cleared |
| **Scientific** | every programme is nameable from the chemistry it loads on; an unnameable programme is reported as such |
| **Scientific** | K is justified on the declared Pareto frontier, with the frontier published |
| **Engineering** | inference is frozen projection; deterministic; batch-independent |
| **Architecture** | derived from Chemistry Evidence only — a static check must confirm no CSM path into `P` |
| **Decision** | **Proceed** to Phase 08 with BSV2 · **Repeat** with a corrected objective · **Redesign** — BSV2 is discarded, Chemistry Evidence remains the terminal interpretable layer, and Phase 08 proceeds without it |

**"BSV2 does not improve on Chemistry Evidence" is an expected and publishable outcome** and
must not be treated as a phase failure (P-13). Three prior attempts at this architectural
position were discarded on evidence; a fourth discard would be the strongest statement V7 has
made about the limits of abstraction over this corpus.
