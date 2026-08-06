# GAIRA V7 — Validation and Decision Rules

**Pre-registered.** Every rule here is committed *before* the sweep it governs is run
(principle P-12). A decision made under a rule not stated in advance is a post-hoc choice and
must be labelled as one in the phase manifest.

---

## 0. The governing principle

> **No single metric selects every layer.**

Each layer answers a different question, so each needs a different composite. Reconstruction
error is nearly meaningless for theme count. Retrieval accuracy is nearly meaningless for LSM
count within a 2-analyte class. Using one number everywhere is how a project optimises the
thing it can measure instead of the thing it wants.

Every rule below follows the same shape:

1. compute a **composite** of criteria that pull in different directions;
2. identify the **Pareto plateau** — the region where further increase buys little;
3. select the **smallest** value on the plateau;
4. record the curve, the plateau boundary, and the choice in the phase manifest.

**Smallest-on-plateau, not argmax.** Argmax on a noisy composite systematically over-selects:
noise at large `k` is rewarded, and every extra component costs interpretability, stability,
and live projection time. The plateau rule is defined concretely as: the smallest value whose
composite score is within a **pre-declared tolerance τ** of the maximum observed score. `τ` is
declared per layer *before* the sweep.

---

## 1. Reference-construction strategy — Phase 01

**Choose among:** A (all spectra, control) · B (analyte-weighted) · B-uniform · C-mean ·
C-median · C-trimmed · C-medoid · C-quality.

| Criterion | Direction | Weight class |
|---|---|---|
| Held-out reconstruction (analyte-grouped) | ↑ | primary |
| Diagnostic-band fidelity | ↑ | primary |
| Class-balance achieved (effective class weight distribution) | ↑ | primary |
| Replicate stability | ↑ | secondary |
| Downstream control retrieval (V5-style global NMF per arm) | ↑ | secondary |
| Information discarded | ↓ | secondary |

**Rule.** Select the arm that maximally improves class balance **subject to** held-out
reconstruction and diagnostic-band fidelity remaining within a pre-declared tolerance of the
control arm A. Balance is the objective; fidelity is the constraint. An arm that improves
balance by wrecking band fidelity has not solved the problem — it has moved it.

**Mandatory stratified reporting** (a violation invalidates the selection):

- corpus-wide **and** restricted to the 87 replicated analytes — 80 of 167 analytes are
  singletons for which all arms coincide, so corpus-wide numbers are diluted toward zero and
  make the arms look falsely equivalent;
- single-excitation **and** multi-excitation analytes separately — 41 analytes span
  excitations, where per-bin mean and median can distort band shape while the medoid cannot;
- the `B-uniform` arm, isolating the balancing effect from the quality-weighting effect.

**If control A wins**, that is the finding: the row-level balancing hypothesis is not supported
at this corpus size. Report it, proceed with A, and note that Strategy D (class partitioning,
Phase 02) is a separate and still-untested bet.

---

## 2. LSM count `k_c` — Phase 02

**Per class**, independently. There is no global `k`.

| Criterion | Direction | What it protects against |
|---|---|---|
| Held-out reconstruction (analyte-grouped) | ↑ | under-fitting |
| Diagnostic-band fidelity | ↑ | explaining variance without explaining chemistry |
| Stability across repeated fits | ↑ | fitting artefacts |
| Redundancy with retained LSMs | ↓ | duplicate motifs |
| Activation sparsity | ↑ | diffuse, non-selective mass |
| Within-class retrieval | ↑ | motifs that do not discriminate |
| Residual structure (band-shaped residual) | ↓ | unexplained chemistry |

**Rule.** Select the smallest `k_c` on the Pareto plateau of the composite, subject to:

- `1 ≤ k_c ≤ ⌊n_analytes(c) / 2⌋` — a class cannot have more motifs than half its analytes,
  or the "motifs" are memorised molecules;
- every retained LSM independently clears the stability threshold — so the effective `k_c` may
  end below the swept value, and that is a legitimate outcome;
- classes with `n_analytes < 2` get no local fit and route to the anchor mechanism.

**Report per class**, whatever the outcome: the full sweep curve, the plateau boundary, the
retained count, the discarded count with reasons, and the source/excitation composition.

---

## 3. Integration method and CSM count `M` — Phase 03

### 3a. Method selection

Candidates: hierarchical consensus clustering · Leiden/Louvain communities · spectral
clustering · sparse non-negative meta-factorisation · hybrid graph + factorisation.

| Criterion | Direction |
|---|---|
| Consensus stability across resamples | ↑ |
| Within-CSM spectral cohesion | ↑ |
| Between-CSM separation | ↑ |
| Chemical coherence of resulting groups | ↑ |
| Retained LSM information | ↑ |
| Downstream held-out recovery | ↑ |
| Sensitivity to hyperparameters (threshold, resolution) | ↓ |
| Singleton fraction | ↓ |
| Redundancy between CSMs | ↓ |

**Rule.** Select the method maximising the composite. **Publish the full comparison table
regardless of the winner** — the point of running five candidates is an auditable choice, and
a table showing why the winner won is the deliverable, not a footnote.

**No method is presumed.** In particular, **the plan does not presuppose that the second NMF
wins.** The stated prior (in `../architecture/LEARNING_MODE_ARCHITECTURE.md` Stage 4) is that
graph or hybrid routes look more promising, because meta-NMF sees only one of six edge
features and its equal row weighting reintroduces the spectrum-count bias V7 exists to remove.
That is a hypothesis to test, not a decision already made.

**Additional requirement if meta-NMF is selected:** explicitly verify that
molecule-discriminating LSMs survive into distinguishable CSMs. Compression can erase exactly
the structure Phase 02 worked to isolate (risk R-06).

### 3b. `M` selection

Same composite, with two explicit penalties:

- **singleton penalty** — a CSM built from one LSM is a local description no other
  decomposition confirmed;
- **redundancy penalty** — CSMs above a similarity threshold to one another are duplicates.
  Precedent: the V5 motif layer carried `porphyrin ↔ flavin` at 0.699 support cosine and
  `carboxylate ↔ colloid_matrix` at 0.687 — both should have been caught by a redundancy
  penalty and were not.

**Rule.** Smallest `M` on the Pareto plateau after applying both penalties.

### 3c. Graph threshold

**Rule.** Sweep the edge threshold across a pre-declared range. Report community stability at
each. Select from a **stable region**, never a single cut. If no stable region exists, the
graph construction is inadequate and must be revised — that is a finding, not a nuisance
(risk R-07).

---

## 4. Theme count `K` — Phase 04 — **LEGACY (A-13)**

> Retained for provenance. The rule was applied in canonical Phase 03 and selected K = 5
> (archetypal, 4 accepted). The theme layer is archived; the rule is superseded by §8.2.

| Criterion | Direction | Note |
|---|---|---|
| Useful information retained | ↑ | information *about chemistry*, not raw variance |
| Held-out superclass retrieval | ↑ | does the abstraction preserve coarse chemistry? |
| Stability | ↑ | across resamples |
| Chemical coherence / admissibility | ↑ | a chemist can name each theme |
| Interpretability | ↑ | |
| Compression | ↑ | `M/K` |
| Calibration (ECE) | ↓ | are confidences honest? |

**Rule.** Select the smallest `K` on the Pareto frontier that is **chemically admissible** —
every theme nameable as coherent chemistry. Admissibility is a hard constraint, not a
weighted criterion: an inadmissible `K` is rejected regardless of its score.

**Precedent worth heeding.** The V6.2 Pareto study
(`results/v6_rebuild/tables/v62_pareto.csv`) found chemical admissibility first satisfied at
`K = 13`, while information retained already reached 0.796 at `K = 6` and recoverability
*fell* monotonically with `K` (0.969 at K=2 → 0.503 at K=12). Compression and admissibility
pull hard in opposite directions in this data. V7 must expect the same tension and resolve it
by the stated rule rather than by whichever number looks nicer.

**Additional requirement.** Demonstrate that the theme layer adds value over the CSM layer, or
record that it does not. At V6.2, `theme_raw` and `theme_posterior` were numerically identical
at every metric on every ontology — added machinery that changed no decisions. A theme layer
that merely relabels CSMs is decorative (risk R-11).

---

## 5. BSV dimension — Phase 05 — **LEGACY (A-14)**

> Retained for provenance. Applied in canonical Phase 04; effective rank 2.40 of nominal
> K = 4 (R-12 realised). Superseded by §8.2.

**Rule.** BSV dimension **= `K`**, the selected number of biochemical themes. There is no
separate choice.

**But measure the effective rank separately**, and report both:

| Measure | Definition |
|---|---|
| participation ratio | `(Σλ)² / Σλ²` on the BSV covariance |
| effective entropy rank | `exp(H(λ/Σλ))` |
| axes for 90% variance | count |

**Precedent.** The V5 24-component space had participation ratio **15.2** and 16 components
for 90% of latent variance — a 38% gap between nominal and effective dimensionality, visible
only because someone measured it. If V7's BSV shows a similar gap, `K` overstates the
representation's actual resolution and downstream users must be told (risk R-12).

---

## 6. Explicit anti-patterns

Each of these has a specific failure mode, and several have already occurred in this project's
history.

| ✗ Anti-pattern | Why it fails |
|---|---|
| **Select by reconstruction alone** | reconstruction rewards modelling the dense classes — precisely the L-01 bias. The V5 basis reconstructs at 0.712 explained variance while only 3 of 24 components are chemically pure. |
| **Select by PCA appearance** | a 2-D projection of a K-dimensional space shows what the projection preserves, not what the space contains |
| **Select by UMAP clusters** | UMAP cluster structure is strongly hyperparameter-dependent and has no stable out-of-sample transform; visually crisp clusters are not evidence of anything |
| **Select by raw top-1 alone** | top-1 conflates representation quality with class-count difficulty. V6.3's twelve size-matched random ontologies exist precisely to separate the two — random 6-class grouping already scores 0.10, and coarsening 18→6 classes mechanically adds accuracy. |
| **Select a lower class count because accuracy rises** | the same mechanical effect. Any accuracy gain from coarsening must be reported against a size-matched random control, and the `gain_beyond_mechanical` figure is the one that counts. |
| **Let held-out analyte information into model selection** | inflates every downstream number invisibly. All sweeps, thresholds, and rules are fitted on training folds only. |
| **Tune the quality score `q` against Phase-01 outcomes** | `q` becomes a hidden hyperparameter; freeze it in Phase 00 |
| **Resample replicates for stability estimates** | leaks within-analyte structure and inflates apparent stability; bootstrap over canonical analytes only |
| **Duplicate spectra to balance rare classes** | adds no information; inflates apparent support (P-11) |
| **Choose the rule after seeing the curve** | the definition of post-hoc selection (P-12) |

---

## 7. Statistical procedures — frozen in Phase 00, used throughout

| Procedure | Specification |
|---|---|
| Cross-validation | analyte-grouped; no canonical ID or replicate crosses a fold |
| Permutation null | size-matched random ontologies (V6.3 used 12; V7 must use ≥ 12) |
| Confidence intervals | bootstrap over canonical analytes, 95% |
| Paired comparison | McNemar (exact) + permutation test, as in V6.3 |
| Effect size | Cohen's g and odds ratio for paired accuracy comparisons |
| Multiple comparisons | correction declared in Phase 00 and applied consistently |
| Calibration | expected calibration error (ECE) with a declared binning scheme |

**All of these are frozen before any V7 model is fitted.** The V6.3 revalidation is the
template — it is the strongest piece of methodology the project has produced, and V7 adopts
its harness wholesale rather than reinventing it.


---

## 8. Decision rules for the remaining phases — ADDED 2026-08-06

### 8.0 Every remaining gate has the same four parts

No phase from 06 onward is complete until all four are recorded in its `PHASE_STATE.json` and
its report:

| Part | What it asks | Failure means |
|---|---|---|
| **1. Scientific validation** | did the layer do what it claimed, on held-out data, against a pre-registered threshold? | the science does not support the layer |
| **2. Engineering validation** | deterministic, batch-independent, no inference-time fitting, fingerprints verified, tests pass | the layer cannot be shipped even if the science holds |
| **3. Architecture compliance** | non-negativity, provenance, layer isolation, no upstream artefact modified, P-18 respected | the layer violates a standing invariant |
| **4. Decision** | **Proceed** · **Repeat** · **Redesign** | — |

**Decision semantics, so the words mean the same thing every time:**

- **Proceed** — all three validations pass. The next phase begins.
- **Repeat** — a *defect* was found (an implementation error, a mis-specified metric, a
  contaminated split). The phase re-runs after the fix. A repeat is **not** licence to try a
  different threshold on the same result.
- **Redesign** — the layer is sound but the *evidence does not support it*. The architecture
  changes, the layer is archived with its outputs preserved, and the plan is rewritten. This is
  what happened to A-13, A-14, A-15 and A-16, and it is a normal outcome (P-13).

### 8.1 DG-06 — the Chemistry Evidence map

**Selected on** the pre-registered composite: held-out class top-1 (weight 0.40), calibration
quality as Brier (0.20), robustness retention (0.20), provenance completeness (0.10),
interpretability (0.10). **Never on reconstruction of the CSM activations** — the map is not
trying to reconstruct anything.

| Rule | Specification |
|---|---|
| **Gate** | S-21 … S-27 (`SUCCESS_CRITERIA.md` §7.1) |
| **Comparator** | the archived 11-axis profile, on identical folds, both re-measured in the same run |
| **Informativeness floor** | ≥ 0.50 of the CSM layer's held-out class information, declared before the run |
| **Mandatory control** | the R-01 class-agnostic decomposition control |
| **Redesign trigger** | S-21 not met → reinstate A-16, archive Chemistry Evidence |

### 8.2 DG-07 — BSV2 and its rank `K`

**`K` is selected on a Pareto frontier over eight axes**: reconstruction, held-out chemistry
prediction, programme stability, interpretability, mutual information, noise robustness,
calibration, compression. The weighting is declared before the sweep. **Reconstruction alone is
never sufficient** (R-12).

| Rule | Specification |
|---|---|
| **Sweep** | K ∈ {2, 3, 4, 5, 6, 8, 10, 12, 14} |
| **Gate** | S-28 … S-32 (`SUCCESS_CRITERIA.md` §7.2) |
| **Informativeness floor** | **pre-registered before the sweep runs**: ≥ 0.50 of Chemistry Evidence's information *and* ≥ 0.50 of its held-out class prediction |
| **Ordering rule** | stability, robustness and calibration gains are evaluated **only after** the floor is cleared. A layer that wins on stability and fails the floor is **discarded**, not "partially adopted" |
| **Mandatory diagnostic** | the best achievable held-out class prediction over *every* (variant, K) combination, reported and **excluded from selection** — it pre-empts "a different K would have worked" |
| **Publication rule** | the full frontier and every rejected point are published. No cherry-picking |
| **Redesign trigger** | floor not cleared → BSV2 discarded, Chemistry Evidence is terminal, Phase 08 proceeds without it |

### 8.3 DG-08 — hierarchical retrieval and the prior weight `λ`

`λ` is fitted on training folds only, **nested** inside the CV so no test spectrum influences it
— the Phase 05 metric-selection pattern, carried forward.

| Rule | Specification |
|---|---|
| **Gate** | S-33 … S-37 (`SUCCESS_CRITERIA.md` §7.3) |
| **Baseline** | direct cosine in CSM space: molecule top-1 0.605, top-3 0.763, top-5 0.795 |
| **Significance** | McNemar exact + permutation on the frozen folds, α = 0.05 after correction |
| **Mandatory ablation** | prior off / soft prior / **hard filter** — the third as a negative control demonstrating that softness is necessary |
| **Hard constraint** | a class error must be recoverable. Any design in which a wrong class posterior removes the correct molecule from the candidate set is rejected at the gate regardless of its mean accuracy |
| **Conformal sets** | shipped **only if** the exchangeability assumption is defensible under molecule-grouped splits. If it is not, say so and ship top-k |
| **Redesign trigger** | S-33 not met → retrieval stays direct-cosine; the chemistry layer remains interpretive |

### 8.4 DG-09 — the V5 replacement decision

| Rule | Specification |
|---|---|
| **Gate** | the frozen Tier-1 criteria S-01 … S-07, **unadjusted** (`SUCCESS_CRITERIA.md` §2, §6) |
| **Harness** | `v7_harness_v1`, the same harness that measured V5. Not a re-implementation |
| **Outcomes** | **replace** · **partial adoption** (requires its own written justification, §5.3) · **retain V5 and publish the negative result** |
| **Prohibited** | adjusting any threshold in either direction (P-13) |

### 8.5 The informativeness rule, stated once for all gates

> **P-18 — stability without informativeness is not evidence.**
> No representation, mode, calibrator or model may be selected on a reproducibility, stability
> or calibration metric unless it has first cleared a pre-registered informativeness floor.

Four independent instances motivated this rule, and all four were caught only after the fact:

| Phase | What was maximised | By what | Caught by |
|---|---|---|---|
| 03 | replicate consistency | a softmax theme mode assigning every spectrum the same flat vector | zero-evidence leakage test |
| 04 | evidence consistency | the same mode, promoted to the engine | leakage veto in the aggregation layer |
| 04.5 | every stability axis | Meta Components retaining 0.185 of CSM information | the 0.50 informativeness floor |
| 05 | expected calibration error | Platt scaling reporting the base rate 0.605 for every spectrum | reading a figure and noticing three molecules with identical confidence |

A fifth instance is expected. The floor is now declared before the measurement rather than
after it.
