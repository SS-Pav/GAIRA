# GAIRA V7 — Learning Mode Architecture

The offline build. This document specifies each stage mathematically and states what is
**decided by evidence** rather than assumed.

**The single most important disclaimer in this document:** V7 is **not** "NMF on NMF".
A second factorisation is *one candidate* among several for Stage 3, and the plan does not
presuppose it will win. Section 4 explains why.

---

## Why this is not simply "NMF on NMF"

A naive reading of "local decompositions, then integrate" is: fit NMF per class, stack the
activations, fit NMF again. That description is wrong about V7 in three ways.

**First, the object being integrated is a dictionary, not a data matrix.** After Stage 1 we
hold `Σ_c k_c` basis *spectra*, each living in the same `ℝ₊^676` space as the data. They are
already directly comparable as spectra — cosine, band overlap, peak position — without any
second factorisation. The integration problem is "which of these local descriptions are the
same chemistry?", which is a *correspondence* problem, not a *dimensionality reduction*
problem. Factorisation is one way to answer it; it is not the natural way.

**Second, the evidence for correspondence is multi-modal.** Two LSMs describe the same
chemistry if they look alike spectrally, *and* peak in the same places, *and* recur together
across bootstraps, *and* activate on overlapping analytes, *and* are supported by related
provenance. A second NMF over an activation matrix sees only one of those five signals. A
similarity graph can carry all five. Discarding four signals to fit a factorisation is a
methodological downgrade, not an upgrade.

**Third, stacking activations reintroduces the exact bias V7 exists to remove.** An LSM
activation matrix has one row per reference and one column per LSM. Refactorising it under a
squared-error loss weights each row equally — which is Strategy A, the thing L-01 identified
as the root problem. A second NMF would smuggle spectrum-count weighting back in at the very
stage meant to unify the balanced local fits.

So the honest structure is: **local decomposition → correspondence → consensus**, with
factorisation available as one correspondence method to be benchmarked. Stage 4 states the
benchmark.

---

## Stage 1 — Local class decomposition (Phase 01)

### Objective

For each chemical class `c ∈ {1, …, C}`, with balanced reference block
`X_c ∈ ℝ₊^{n_c × D}` (D = 676):

$$X_c \approx W_c H_c, \qquad W_c \in \mathbb{R}_+^{n_c \times k_c}, \; H_c \in \mathbb{R}_+^{k_c \times D}$$

`H_c` holds the **candidate LSMs** for class `c`. Under Strategy B the loss is row-weighted:

$$\min_{W_c, H_c \ge 0} \sum_{i=1}^{n_c} \omega_i \big\| x_{ci} - w_{ci} H_c \big\|^2 + \lambda \|H_c\|_1$$

with `ω_i` the analyte-balanced quality weight and the optional L1 term promoting sparse,
band-localised motifs. Whether sparse NMF beats plain NMF per class is itself a Phase-01
sweep, not an assumption.

### Repeated fits and stability selection

A single NMF fit gives components; it does not give evidence that those components are real.
Stage 1 therefore runs `R` fits per `(c, k_c)` under:

- different random initialisations (fixed seed schedule, so the *set* of runs is deterministic);
- analyte-level resampling (bootstrap over canonical analytes, never over replicate spectra —
  resampling replicates would leak within-analyte structure and inflate stability);

then aligns components across runs by the **Hungarian algorithm** on cosine similarity, and
scores each aligned component by **recurrence**: the fraction of runs producing a matching
component above a similarity threshold, and the mean similarity of the matches.

An LSM is **retained** only if its recurrence clears the pre-registered stability threshold.
Everything else is a fitting artefact and is discarded — but recorded, so "what did we throw
away" is answerable.

### Adaptive `k_c`

`k_c` is swept per class and selected by the smallest-on-Pareto-plateau rule over the
composite in `../context/SCIENTIFIC_DESIGN_PRINCIPLES.md` §E and
`../plan/VALIDATION_AND_DECISION_RULES.md`.

Constraints: `1 ≤ k_c ≤ ⌊n_analytes(c)/2⌋`. Classes with `n_analytes < 2` get no local fit and
are routed to the anchor mechanism (Strategy F).

### LSM typing

Each retained LSM is labelled by its within-class activation pattern:

| Type | Criterion | Example expectation |
|---|---|---|
| **class-shared** | activates above threshold on most analytes of the class | protein amide backbone |
| **subfamily** | activates on a coherent proper subset | the ester carbonyl separating triglycerides from free fatty acids |
| **molecule-discriminating residual** | activates on very few, with high selectivity | a specific ring substitution pattern |

This typing matters downstream: class-shared LSMs from *different* classes that describe the
same chemistry are exactly what Stage 3 must merge, while molecule-discriminating LSMs are
exactly what must *not* be merged away.

### Outputs

Per class: `H_c`; per LSM: stability, recurrence, type, redundancy with siblings, dominant
bands, activating analytes, source/excitation composition. Plus a per-class report including
the source/excitation composition check (risk R-16).

---

## Stage 2 — LSM integration: the motif similarity graph (Phase 02)

### Pooling

$$\mathcal{H} = \{H_1, H_2, \ldots, H_C\}, \qquad |\mathcal{H}| = \sum_c k_c^{\text{retained}}$$

All stable LSMs from all classes, pooled into one set. They already share the coordinate space
`ℝ₊^676` — the canonical grid — so they are directly comparable as spectra.

### Edge features

Build a weighted graph `G = (V, E)` with `V = H`. Each candidate edge carries **six**
features, deliberately capturing different and partly independent evidence:

| # | Feature | Definition | What it catches |
|---|---|---|---|
| 1 | **spectral cosine** | `⟨h_i, h_j⟩ / (‖h_i‖‖h_j‖)` on the full grid | overall shape agreement |
| 2 | **diagnostic-band overlap** | overlap restricted to each motif's dominant bands | agreement *where it matters* — two motifs can have high global cosine from shared broad structure while disagreeing on every diagnostic band |
| 3 | **peak-position agreement** | matched peak positions within a tolerance window | position is excitation-invariant; intensity is not |
| 4 | **bootstrap recurrence co-occurrence** | how often `h_i` and `h_j` both appear across resamples | shared stability regime |
| 5 | **activation co-occurrence** | correlation of their activations over the balanced references | do they respond to the same molecules? |
| 6 | **provenance overlap** | shared supporting analytes / classes / sources | are they describing the same evidence? |

**Why six and not one.** Feature 1 alone is the trap. Raman spectra of biological molecules
share a great deal of broad structure; two chemically distinct motifs routinely reach cosine
0.7+ on shared CH and ring modes. The V5 motif redundancy table already demonstrates this
failure mode: `purine_ring_breathing` and `sterol_ring_system` shared **0.679** component
support while their activations correlated only **0.243**. Under cosine alone they would
merge; under features 2, 3, and 5 they clearly should not.

**Feature 6 needs care.** Provenance overlap is partly *circular* — two LSMs from the same
class necessarily share provenance. It must be computed so that within-class overlap is
discounted, or it will simply re-encode the class partition and Stage 3 will rediscover the
classes it started from (risk R-01).

### Edge weighting and thresholding

Features are combined into an edge weight under a pre-registered scheme, and the graph is
sparsified. **Thresholds are a known arbitrariness** (risk R-07): community structure can be
an artefact of where the sparsification cut falls. Mitigation is mandatory: sweep the
threshold, report community stability across the sweep, and select only where structure is
stable over a range — not at a single lucky cut.

---

## Stage 3 — CSM derivation (Phase 02)

Clusters of LSMs become **Consensus Spectral Motifs (CSMs)** — the canonical V7 evidence
unit. For cluster `m`:

$$\text{csm}_m = \Pi\big(\{h : h \in \text{cluster}_m\}\big), \qquad \text{csm}_m \in \mathbb{R}_+^{D}$$

where Π is a non-negative consensus operator (candidates: weighted mean by LSM stability,
non-negative medoid, or the leading non-negative consensus direction), followed by
re-normalisation on the canonical grid.

### Mandatory CSM record

| Field | Content |
|---|---|
| `consensus_spectrum` | `ℝ₊^676` |
| `contributing_lsms` | LSM IDs + weights |
| `supporting_classes` | which class-local fits contributed |
| `supporting_analytes` | canonical IDs + counts |
| `dominant_bands` | positions, widths, band-fidelity score |
| `uncertainty` | spread of contributing LSMs about the consensus |
| `stability` | recurrence of the cluster across consensus resamples |
| `n_lsms`, `n_classes`, `n_analytes` | support breadth |
| `is_singleton` | `n_lsms == 1` |
| `is_anchored` | admitted via Strategy F rather than learned |
| `provenance` | sources, excitations, build manifest hash |

**Singletons and anchors are flagged, never hidden.** A singleton CSM is a local description
no other decomposition confirmed; an anchored CSM has one supporting reference by
construction. Both are legitimate — rare chemistry is real — but both must carry wider
uncertainty downstream and must be visible as such in every report. The V5 failure mode was
precisely that a motif with 1.2% corpus coverage (`flavin_redox_cofactor`) looked, in the
output, exactly like a motif with 7.2% coverage.

---

## Stage 4 — Optional second non-negative factorisation

**This stage is a candidate method, not a step.** It may or may not appear in the final build.

### What it would be

Form the LSM activation matrix `A ∈ ℝ₊^{N × |H|}` (balanced references × pooled LSMs) and
factorise:

$$A \approx U V, \qquad V \in \mathbb{R}_+^{M \times |\mathcal{H}|}$$

with rows of `V` inducing soft groupings of LSMs into `M` meta-motifs.

### The benchmark it must win

Stage 4 is adopted **only if** it beats all of the following on the pre-registered Phase-02
criteria:

| Candidate | Mechanism | Strength | Weakness |
|---|---|---|---|
| **graph communities alone** (Leiden / Louvain) | modularity on `G` | uses all six edge features; no `M` to choose in advance | resolution-parameter and threshold sensitivity |
| **hierarchical consensus clustering** | repeated clustering, consensus matrix, cut | well-understood stability semantics; gives a dendrogram | needs a cut rule; ignores graph topology |
| **spectral clustering** | eigen-decomposition of the graph Laplacian | handles non-convex structure | needs `M` in advance; sensitive to the affinity scale |
| **sparse non-negative meta-factorisation** (Stage 4) | NMF over `A` | soft, non-negative, overlapping membership — matches the chemistry, where one motif can genuinely belong to two groups | **sees only activation co-occurrence** — discards edge features 1, 2, 3, 6; and equal row weighting reintroduces the L-01 bias |
| **hierarchical dictionary models** | explicit two-level generative dictionary | principled hierarchy | more moving parts; harder to freeze and to explain |
| **hybrid graph + factorisation** | communities as initialisation/constraint for a non-negative refit | keeps multi-feature evidence *and* gets soft overlapping membership | most complex; risk of inheriting both methods' weaknesses |

### The stated prior, and why it is only a prior

On the reasoning in the preamble — the second factorisation sees one of six evidence
channels, and its equal row weighting reintroduces the spectrum-count bias — **the graph or
hybrid routes look more promising a priori.** That is a hypothesis to test, not a decision.
The hybrid route is genuinely attractive: it would use the graph to establish *which* LSMs
correspond (multi-feature evidence) and a constrained non-negative refit to establish *how
much* each contributes (soft overlapping membership).

**Phase 02 must report the full comparison table regardless of which method wins**, so the
choice is auditable.

### Additional risk if Stage 4 is selected

A second factorisation compresses; compression can erase exactly the molecule-discriminating
residual structure Stage 1 worked to isolate (risk R-06). If Stage 4 is selected, Phase 02
must explicitly verify that molecule-discriminating LSMs survive into distinguishable CSMs —
measured, not asserted.

---

## Stage 5 — Theme construction (Phase 03)

Themes are derived **from CSMs**, never asserted over them. This inverts the V5 direction and
is the direct response to L-05.

$$t = S^{\top} c, \qquad S \in \mathbb{R}_+^{M \times K}, \qquad \sum_{k} S_{mk} = 1 \ \ \forall m$$

`S` is **sparse** (a CSM belongs to few themes), **non-negative**, and **row-normalised** (a
CSM's membership is a distribution). Soft membership is retained from V6.2: shared
biochemical structure genuinely belongs to more than one theme, and forcing one parent
destroys information.

`K` is swept and selected on a pre-registered Pareto frontier over information retained,
mutual information with chemistry, chemical admissibility, calibration, held-out
superclass retrieval, stability, compression, and interpretability. Details in
`../plan/VALIDATION_AND_DECISION_RULES.md`.

**Themes name chemistry only.** No disease, pathway, process, or phenotype.

### A caution carried from V6.2

At the V6.2 theme layer, `theme_raw` and `theme_posterior` were numerically identical at
every metric on every ontology — the Bayesian posterior refinement changed no decisions. The
lesson: **abstraction layers added on top of an insufficient representation buy nothing.**
Phase 03 must demonstrate that the theme layer adds value over the CSM layer, or record that
it does not. A theme layer that merely relabels CSMs is decorative (risk R-11).

---

## Stage 6 — BSV construction (Phase 04)

$$\mathrm{BSV}(x) = t(x) = S^{\top} c(x) \in \mathbb{R}_+^{K}$$

**BSV dimension = K = number of biochemical themes.** The BSV is absolute: a position in a
fixed global coordinate system.

Phase 04 additionally freezes: normalisation frame `(μ, σ)` per axis, reference
distributions, the OOD support, the uncertainty propagation model, and — separately and
distinctly named — the derived forms: reference-normalised elevation, ΔBSV,
cohort-standardised views, and DART trajectories.

### Visualisation, and only visualisation

$$y = P^{\top}\big(\mathrm{BSV} - \mu\big)$$

`P` is fitted **offline**, frozen into the atlas, and thereafter only *applied*. `y` is a
picture. It is never the canonical BSV, never an input to interpretation, never used for
retrieval or scoring.

**The PCA representation is not the canonical BSV.** UMAP is not shipped at all — it has no
stable freezable out-of-sample transform, and using one would make the picture depend on the
batch.

### The correlated-axes risk

If themes end up highly correlated, `K` axes carry far fewer than `K` degrees of freedom and
the BSV over-states its own resolution (risk R-12). Phase 04 must measure the **effective
rank** of the BSV space separately from `K` and report both. Precedent: the V5 24-component
space had participation ratio 15.2 — a 38% gap between nominal and effective dimensionality
that was only visible because someone measured it.
