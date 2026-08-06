# GAIRA V7 — Scientific Design Principles

The rules V7 is built under. Part 1 states the inherited and new principles; Part 2 is the
coverage-imbalance strategy, which is the central methodological contribution of V7 and is
therefore specified in operational detail.

---

# Part 1 — Principles

## P-01 Spectroscopy first, then data architecture, then ML

Every design choice must have a spectroscopic justification before it has a statistical one.
If a method improves a metric but cannot be explained in terms of bands, chemistry, and
measurement physics, it is not adopted — it is recorded as an unexplained result and
investigated.

## P-02 Non-negativity is not optional

Every representation layer is non-negative: LSMs, CSMs, activations, memberships, themes,
and the BSV. A spectrum is a sum of contributions; a negative contribution has no physical
referent. Signed quantities appear only in explicitly derived readings (ΔBSV, z-scored
elevation) and are always named as such.

## P-03 Determinism over cleverness

Fixed seeds, fixed iteration order, fixed tie-breaking, fixed float handling. Every artefact
is byte-reproducible from its manifest on a clean clone with no lab volume. Where an
algorithm is inherently stochastic (NMF initialisation, consensus resampling), the
stochasticity is *inside* a stability procedure whose *output* is deterministic given the
seed schedule.

## P-04 Provenance is a first-class field, not a comment

Every LSM, CSM, theme, and BSV axis carries: contributing objects one level down, supporting
canonical analytes, supporting classes, dominant bands, source datasets, excitations, and the
build manifest hash. "Which chemistry supports this axis?" must be answerable by query, not
by archaeology. L-03 happened because that question was hard to ask.

## P-05 Peak ≠ molecule; region-based mapping only

No layer performs exact-peak-to-exact-molecule matching. Bands are regions with tolerance.
Assignments are multi-valued. Ambiguity is represented, never hidden.

## P-06 Chemistry organises; chemistry does not supervise

Chemical class partitions the decomposition. It is never a prediction target, never a term in
the local loss, and never an input to inference. The class prior's influence on the resulting
representation is itself measured and reported (Phase 02).

## P-07 Themes are chemistry; biology is downstream

Themes name chemistry (protein, lipid, nucleic, carbohydrate, organic-acid/energy,
sulfur/redox/cofactor). No theme names a disease, a pathway, a process, or a phenotype.
Domain context enters strictly after the BSV and never propagates upstream.

## P-08 The BSV is absolute

The BSV is a position in a fixed global coordinate system, not a change, not a label, not a
score. Differences, elevations, cohort standardisations, and low-dimensional views are all
derived and all named distinctly. Conflating a ΔBSV with a BSV is a correctness bug.

## P-09 Learning is offline; inference is projection

No fitting at inference: no NMF, no PCA fit, no UMAP, no clustering, no community detection,
no ontology optimisation. Inference is preprocessing, non-negative projection, matrix
multiplication, fixed linear transforms, distances, and uncertainty propagation. This is what
makes two spectra measured years apart comparable.

## P-10 Raman is the foundation; SERS is an observation channel

The universal representation is learned from **pure Raman** references only. SERS is a
measurement channel applied to that latent state, modelled later by an explicit observation
model. No SERS assumption — enhancement selectivity, surface affinity, orientation effects —
may shape the Raman foundation. The V5 corpus card already enforces this exclusion
(Ag-SERS, Au-SERS, DART, serum Ag-colloid, and the adenine SERS series are all excluded);
V7 inherits it unchanged.

## P-11 Balancing is not oversampling

Rare classes are never bootstrapped by duplicating identical spectra. A duplicate adds no
information; it only shifts the loss surface while inflating apparent support. A class with
two analytes has two analytes' worth of evidence, and every artefact derived from it must say
so in its provenance and its uncertainty.

## P-12 Pre-register the decision rule, then measure

For every model-selection decision, the rule is written down *before* the sweep is run and
committed with the code. Post-hoc rule selection is how a project talks itself into a result.
See `../plan/VALIDATION_AND_DECISION_RULES.md`.

## P-13 A negative result is a result

If V7 does not clear the pre-registered replacement criteria, the outcome is a documented
negative result and a retained V5 atlas. The bar is not lowered to fit the outcome.

## P-14 Modular addition over broad refactor

V7 lives entirely under `GAIRA_v7_rebuild/`. The V5 atlas, the V5/V6/V6.2/V6.3 artefacts, the
existing engines, and the existing Streamlit apps are untouched until Phase 07 authorises a
replacement — and even then, by a versioned addition, not an in-place edit.

## P-15 The frozen atlas is a control, never a foundation

The V5/V6 frozen atlas may be used ONLY as a **baseline control**, a **benchmark comparator**,
or a **reproducibility reference**. It must **NOT** become the foundation of the V7 learning
architecture unless the specification explicitly says so.

This was implicit — `LEARNING_MODE_ARCHITECTURE.md` derives every LSM from `X_c`, the balanced
reference blocks, and never from `H` — but it was never written as a single prohibition, and
its absence is the proximate cause of the drift documented in
`ARCHITECTURE_COMPLIANCE_AUDIT.md`. A layer built on the frozen atlas inherits the V5 global
objective, which is Strategy A, which is the exact bias V7 exists to remove. It is also
mathematically bounded by that atlas: it can only redistribute mass the atlas already placed.

Enforced by `tests/test_v7_phase01.py::test_frozen_atlas_is_not_an_input_to_the_lsm_package`.

## P-16 Architecture check before implementation

Before implementing any phase, re-read every V7 architecture document and verify that the
phase to be implemented matches the approved architecture. If any discrepancy exists between
the implementation brief and the approved architecture, **stop immediately**, generate an
Architecture Deviation Report, and do not proceed until the discrepancy is resolved.

**A discrepancy noted in a docstring is not a resolution.** Implementation must not begin.
Noting a divergence and building anyway is how a brief silently overrides an approved design —
which is exactly what happened once, and is why this principle exists.

## P-17 Redraw the pipeline at the end of every phase

Every phase report ends with a **Current V7 Pipeline** section: completed stages, remaining
stages, the next phase's inputs and its outputs — drawn, not described. Prose hides a
substitution that a diagram exposes: `24 atlas components → motifs` and `class-local NMF →
LSMs` are obviously different pictures and were not obviously different paragraphs.

## Architecture compliance is a gate, not a report

Every phase report ends with an **Architecture Compliance** table:
*specification item · implemented? · evidence · PASS/FAIL*. The phase gate opens only if every
row is PASS. Otherwise the phase repeats.

---

# Part 2 — The coverage-imbalance strategy

This is the core methodological change in V7. L-01 and L-02 showed that the global objective
allocates capacity by spectrum count, so dense chemistry crowds out rare chemistry. V7
attacks this at three levels: the **row** (which spectra count, and how much), the **block**
(which spectra are fitted together), and the **capacity** (how many motifs each block gets).

## The problem, restated quantitatively

| Level | Imbalance | Source |
|---|---|---|
| Replicate | 87/167 analytes have replicates; groups 1–3 spectra | `manifold.json → corpus_card.replicate_groups` |
| Analyte-per-class | protein 32 … polyol 1; top-5 families = 63% of analytes | `p2_family_census.csv` |
| Effective capacity | 24 components ≈ 15.2 participation ratio | `manifold.json → intrinsic_dimensionality` |
| Outcome | 3/24 components purity ≥ 0.5 | `component_registry_v1.json` |

## Strategies A–F: all six must be benchmarked

Phase 01 benchmarks A–C and E–F's prerequisites; D is structural and is implemented in
Phase 02. **No strategy is assumed to win.** The selection rule is pre-registered in
`../plan/VALIDATION_AND_DECISION_RULES.md` before the sweep runs.

### A. All spectra, equal row weight — *the control*

$$\min_{W,H \ge 0} \sum_{i=1}^{N} \big\| x_i - w_i H \big\|^2$$

Every spectrum contributes equally. **This is exactly what V5 does**, and it is retained as
the control arm so every V7 claim is measured against the real baseline rather than a
strawman. If A wins, V7's balancing hypothesis is falsified and that is reported as such.

### B. Analyte-balanced weighted fitting

Each canonical analyte contributes **total weight one**, distributed across its replicates in
proportion to quality:

$$w_{ai} = \frac{q_{ai}}{\sum_{j \in a} q_{aj}}, \qquad \sum_{i \in a} w_{ai} = 1$$

where `q_{ai}` is the quality score of replicate `i` of analyte `a` (SNR, baseline quality,
grid coverage, artefact flags — defined and frozen in Phase 00).

The weighted objective:

$$\min_{W,H \ge 0} \sum_{a} \sum_{i \in a} w_{ai} \big\| x_{ai} - w_{ai}^{\text{row}} H \big\|^2$$

**Properties.** Preserves all measurements (no information discarded), so within-analyte
variation still informs the fit — but a 3-replicate analyte no longer outvotes a 1-replicate
analyte. Quality-proportional weighting means a noisy replicate contributes less than a clean
one, which is the scientifically correct behaviour and is not available under A.

**Cost.** Requires a weighted NMF implementation and a frozen, defensible quality score.
A poorly specified `q` silently becomes a hidden hyperparameter — Phase 00 must freeze it,
and Phase 01 must include a sensitivity check with uniform `q` (which reduces B to simple
replicate averaging of weight).

### C. One robust analyte centroid

Replace each analyte's replicate set with a single prototype:

| Estimator | Definition | Property |
|---|---|---|
| **mean** | $\bar{x}_a = \frac{1}{n_a}\sum_i x_{ai}$ | efficient; not robust to one bad replicate |
| **median** | per-bin median | robust to outlier bins; can produce a spectrum that is not any real spectrum |
| **trimmed mean** | mean after dropping the extreme α fraction per bin | tunable robustness |
| **medoid** | $\arg\min_{x_{ai}} \sum_j d(x_{ai}, x_{aj})$ | **always a real measured spectrum**; preserves band shape exactly |
| **quality-weighted** | $\sum_i q_{ai} x_{ai} / \sum_i q_{ai}$ | uses QC metadata directly |

**Spectroscopic caution.** Per-bin median and mean can distort band *shape* when replicates
come from different excitations — and 41 of 167 analytes are multi-excitation. Peak positions
are excitation-invariant but relative intensities are not (resonance effects, instrument
response). The medoid avoids this by construction; the mean does not. **Phase 01 must
evaluate prototypes separately for single-excitation and multi-excitation analytes**, and must
consider building per-excitation prototypes rather than collapsing across excitation.

**Cost.** Discards within-analyte variance, which is the only direct estimate of measurement
uncertainty available. If C is selected, the discarded spread must be retained separately as
an uncertainty asset, not thrown away.

### D. Equal class contribution — *structural, via class-specific decomposition*

Rather than reweighting rows to equalise classes in one global fit, V7 **partitions**:

$$X \longrightarrow \{X_{c}\}_{c=1}^{C}, \qquad X_c \approx W_c H_c \ \ \text{fitted independently}$$

**Why partitioning beats global class-reweighting.** Reweighting a global fit to equalise
classes would give sterol chemistry the same *gradient weight* as protein chemistry — but
both would still be competing for the *same 24 slots*, and the solution that minimises total
error still favours structure shared across the reweighted whole. Partitioning gives sterol
chemistry its own slots. A 32-analyte protein family and a 9-analyte sterol family can no
longer trade capacity because they no longer share an objective.

**This is the single change most directly targeted at L-02 and L-07.**

**Cost, and it is real.** Local dictionaries are not automatically comparable — an LSM from
the sterol fit and an LSM from the fatty-acid fit are two local descriptions with no shared
coordinate. Restoring comparability is the entire job of Phase 03, and if Phase 03 fails, D
has traded one problem for a worse one. This is tracked as risk R-03.

### E. Adaptive motif count per class

**Do not force identical `k` for every class.** A 32-analyte protein family with
subfamilies (globular, fibrous, enzyme, transport) plausibly supports more motifs than a
2-analyte carotenoid family — which may support exactly one, or none that is stable.

`k_c` is selected per class from a pre-registered composite:

| Criterion | Direction | Why |
|---|---|---|
| held-out reconstruction (analyte-grouped) | ↑ | does the extra motif explain unseen molecules? |
| diagnostic-band fidelity | ↑ | does it explain the bands that matter chemically? |
| stability across repeated fits | ↑ | is it a real direction or a fitting artefact? |
| redundancy with existing LSMs | ↓ | is it a duplicate? |
| activation sparsity | ↑ | is it selective, or diffuse mass? |
| within-class retrieval | ↑ | does it help separate molecules in the class? |
| residual structure | ↓ | is there band-shaped structure still unexplained? |

**Rule (pre-registered): select the smallest `k_c` on the Pareto plateau**, where the plateau
begins at the smallest `k` whose composite score is within a stated tolerance of the maximum.
Smallest-on-plateau, not argmax — because argmax on a noisy composite systematically
over-selects.

**Floor and ceiling.** `k_c ≥ 1`. `k_c ≤ ⌊n_analytes(c) / 2⌋` — a class cannot have more
motifs than half its analytes, or the "motifs" are memorised molecules. For a 2-analyte class
this yields `k_c = 1`. For a 1-analyte class (polyol), local decomposition is not meaningful
at all: see F.

### F. Rare-chemistry anchors

Some chemistry is present in the corpus at 1–3 analytes: polyol (1), phospholipid (2),
carotenoid (2), nucleic_acid (3), pyrimidine (3). Local decomposition cannot learn a stable
motif from `n = 1`.

**Mechanism.** A high-quality canonical reference may be admitted as an **anchored atom** —
a fixed, non-learned basis vector entering the CSM dictionary directly — *only if* it passes
all of:

1. **Quality gate** — the reference meets the Phase-00 quality bar (SNR, baseline, coverage).
2. **Novelty gate** — the reference's residual after projection onto the existing stable CSM
   set exceeds a pre-registered threshold. That is: it represents a spectral direction the
   learned motifs genuinely do not span. An anchor that duplicates a learned motif is
   rejected (risk R-08).
3. **Chemical justification** — a written statement of which chemistry it represents and why
   the corpus cannot supply it otherwise.
4. **Provenance flag** — the anchor is marked `anchored: true` in the CSM registry, with
   `n_supporting_analytes = 1`, permanently. It is never presented as having consensus
   support it does not have.

**Anchors are declared honestly weak.** Downstream uncertainty for an anchored axis must be
wider than for a consensus axis. An anchor says "this direction exists in chemistry and we
have one clean reference for it", not "this direction is well-characterised".

**And, restating P-11:** rare classes are **never** handled by duplicating spectra. If
`n = 1`, the honest options are one anchored atom, exclusion with documentation, or Phase 09
corpus expansion. Not synthetic multiplicity.

## How the strategies compose

```
Strategy A/B/C  →  decides what a row is        (Phase 01)
Strategy D      →  decides which rows fit together (Phase 02)
Strategy E      →  decides how many motifs each block gets (Phase 02)
Strategy F      →  decides what to do when a block is too small to fit (Phase 02/03)
```

A/B/C are mutually exclusive; one is selected in Phase 01. D is structural and applies
regardless. E applies within D. F handles D's residue.

## Honest accounting of what balancing cannot do

Balancing changes **how existing evidence is weighted**. It does not create evidence.

- Phospholipid chemistry has 2 analytes. After balancing it still has 2 analytes.
- Carotenoid chemistry has 2 analytes. After balancing it still has 2.
- The corpus contains no sphingolipid reference. No weighting scheme conjures one.

What balancing buys is that those 2 analytes are no longer *drowned out* — their structure
gets a chance to appear as a stable local motif rather than being absorbed into a
protein-dominated global component. What it cannot buy is confidence in a 2-analyte motif.
Hence Phase 09: use V7's residual analysis to identify which spectral directions remain
genuinely unsupported, and acquire references **for those specific directions** —
not for whatever datasets happen to be available.
