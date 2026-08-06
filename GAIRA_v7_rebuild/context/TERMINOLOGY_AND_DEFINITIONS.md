# GAIRA V7 — Terminology and Definitions

Binding vocabulary for all V7 documents, code, artefacts, and reports. Where V7 diverges
from earlier naming, the legacy term is recorded and marked legacy.

> **Revised 2026-08-06, after Phase 05.** Three terms changed status. **Biochemical theme** and
> **BSV** became *legacy* — they were built, measured and archived (see
> `GAIRA_V7_ARCHITECTURE_STATUS_AFTER_PHASE05.md` A-13/A-14). **Chemistry Evidence**, **BSV2**
> and **Molecular Retrieval** are new. The definition of *chemical class* is amended: it is now
> admissible as an intermediate probabilistic coordinate, having previously been forbidden as
> any kind of output. §7.1 of the status document explains why that prohibition was internally
> inconsistent with the frozen success criteria from Phase 00 onward.
>
> Legacy definitions are **retained in full below**, marked LEGACY. Nothing is deleted.

Each term is defined by: **mathematical meaning**, **scientific meaning**, **input**,
**output**, and **provenance class** — one of

| Class | Meaning |
|---|---|
| **learned** | produced by fitting a model to data (offline only) |
| **derived** | computed deterministically from learned or curated objects |
| **curated** | asserted by a human from chemistry, with a written justification |
| **visualised** | a presentation of another object; never an input to inference |

---

## Canonical Raman reference

**Mathematical.** A vector `x ∈ ℝ₊^D` on the canonical grid (D = 676 bins,
450–1800 cm⁻¹, 2 cm⁻¹), or a robust prototype `x̄_a = ρ({x_{a,1}, …, x_{a,n_a}})` aggregating
the `n_a` quality-controlled replicates of analyte `a` under a robust estimator ρ.

**Scientific.** One quality-controlled Raman measurement, or one robust representative
spectrum, associated with exactly one **canonical molecule ID**. Aliases
(`riboflavin` / `riboﬂavin`, `dextrose` / `glucose`, `acetyl-coa` / `acetyl coenzyme a`)
resolve to a single canonical ID before anything else happens. The canonical ID — not the
spectrum, not the file — is the unit of scientific accounting in V7.

**Input.** Raw spectrum + instrument metadata (excitation, source dataset) + QC metadata.
**Output.** One row of the balanced reference matrix, carrying its canonical ID, replicate
group, class assignment, quality weight, and provenance.
**Provenance class.** **curated** (identity and QC) + **derived** (the prototype, if used).

---

## Class / chemical family

**Mathematical.** A partition `C = {c₁, …, c_C}` of the canonical analyte set, used to split
the reference matrix `X` into blocks `{X_c}` for independent local decomposition.

**Scientific.** A chemically curated organisational prior: *purine*, *saccharide*,
*triglyceride*, *sterol*, *fatty_acid*, *protein*, and so on.

**What it is not — three prohibitions (amended 2026-08-06):**

1. It is **not a disease label**. No class in V7 refers to a condition, phenotype, or process.
   *Unchanged.*
2. It is **not a terminal hard label**. V7 never emits "this spectrum is a saccharide" as a
   claim. Class may be reported only as a **probabilistic, uncertainty-carrying evidence
   coordinate** with its calibration and its unassigned mass alongside — see *Chemistry
   Evidence*. **Amended.** The previous wording was "It is not the inference output. V7 never
   predicts class", which contradicted the Phase-00 frozen success criteria: S-01 and S-03 are
   defined on `v7_fine_16` retrieval, i.e. exactly this task. The project has measured itself on
   fine-class retrieval since Phase 00. The amendment resolves the conflict in favour of the
   frozen criteria (P-13 forbids adjusting those) and narrows the prohibition to what it was
   always meant to prevent.
3. It is **not supervision inside the local fit**. The decomposition within a class is
   unsupervised; the class only decides *which spectra enter which fit*. *Unchanged, and now
   the load-bearing prohibition.*

> **Standing risk, sharpened.** Because the Chemistry Evidence layer predicts the same 16
> classes that partitioned the Phase-01 fits, part of its accuracy may be an imprint of the
> partition rather than a property of the representation (R-01, U-02). Phase 06 must run a
> class-agnostic decomposition control on the same folds before any Chemistry Evidence number
> is described as a property of the representation.

**Input.** Canonical analyte table + curated chemistry.
**Output.** Class assignment per analyte; the block structure `{X_c}`.
**Provenance class.** **curated**.

> **Standing risk.** Partitioning by a human prior can imprint that prior on the
> representation. Phase 02 must test this and report every class where the local
> decomposition is dominated by the partition rather than by spectroscopy.
> See `plan/RISK_REGISTER.md` R-01.

---

## Local Spectral Motif — LSM

**Mathematical.** A row of `H_c` from the class-local non-negative factorisation

$$X_c \approx W_c H_c, \qquad W_c \ge 0,\; H_c \ge 0,\; H_c \in \mathbb{R}^{k_c \times D}$$

retained only if it is **stable**: recurrent across repeated fits under resampling and seed
variation, above a pre-registered stability threshold, after Hungarian alignment across runs.

**Scientific.** A stable, non-negative basis spectrum learned *within one chemical class*.
An LSM captures one of three kinds of structure, and each retained LSM is labelled with
which:

- **class-shared** — present across most analytes of the class (e.g. the amide backbone
  pattern across proteins);
- **subfamily** — present across a coherent subset (e.g. the ester carbonyl shared by
  triglycerides but not free fatty acids);
- **molecule-discriminating residual** — the structure that separates near neighbours within
  the class.

**Why "local".** Its initial meaning is *conditional on the class-specific decomposition*.
An LSM from the sterol fit and an LSM from the fatty-acid fit are not yet comparable
coordinates — they are two local descriptions that may or may not describe the same
chemistry. Making them comparable is exactly the job of Phase 03.

**Input.** `X_c` (balanced, class-restricted), `k_c`, resampling and seed schedule.
**Output.** Per-class LSM dictionary `H_c`, plus stability, redundancy, band, and
provenance records per LSM.
**Provenance class.** **learned** (offline, Phase 02).

---

## Consensus Spectral Motif — CSM

**Mathematical.** Given the pooled stable LSM set `H = {H₁, …, H_C}` and a similarity graph
`G` over its members, a CSM is a consensus spectrum

$$\text{csm}_m = \Pi\big(\{h : h \in \text{cluster}_m\}\big) \in \mathbb{R}_+^D$$

where clusters come from the Phase-03 integration method (consensus clustering, graph
community detection, sparse non-negative meta-factorisation, or a hybrid — **selected on
evidence, not assumed**) and Π is a non-negative consensus operator with the resulting
spectrum re-normalised on the canonical grid.

**Scientific.** A stable **cross-class** spectral motif: a pattern that multiple independent
local decompositions converged on. A CSM is:

- a consensus spectral pattern, not a molecule and not a class;
- supported by multiple LSMs, analytes, bands, and (usually) classes;
- accompanied by full provenance;
- **the canonical spectroscopic evidence unit of V7.**

Every CSM carries, mandatorily:

| Field | Content |
|---|---|
| consensus basis spectrum | the `ℝ₊^D` vector |
| contributing LSMs | IDs, with per-LSM weight |
| supporting classes | which class-local fits contributed |
| supporting analytes | canonical IDs, with counts |
| dominant bands | positions and widths, with band-fidelity score |
| uncertainty | spread of contributing LSMs about the consensus |
| stability | recurrence of the cluster across consensus resamples |
| provenance | source datasets, excitations, build manifest hash |

A cluster of one LSM is a **singleton** and is penalised in model selection: it is a local
description that no other decomposition confirmed.

**Input.** Pooled stable LSMs + the similarity graph.
**Output.** CSM dictionary (the projection basis used at inference) + CSM registry.
**Provenance class.** **learned** (offline, Phase 03).

### Legacy mapping

> **Legacy MSS → V7 Consensus Spectral Motif (CSM).**

`MSS` ("Molecular Spectral Signature") is **legacy terminology**. It denoted a *curated*
band-and-exemplar overlay laid over a *pre-existing* global basis — 13 motifs in v1,
rebuilt to 18 in V6.

The CSM is not the same object, and the difference is the point:

| | Legacy MSS | V7 CSM |
|---|---|---|
| Origin | curated over a fixed basis | derived from many independent local fits |
| Direction | top-down (chemistry asserted onto components) | bottom-up (consensus emerges from data) |
| Failure mode observed | borrows unrelated component mass — `sterol_ring_system` top-activated by fatty acids, AUC 0.68 | a motif with no cross-class support is a singleton and is visible as one |
| Provenance | exemplar list | LSMs, classes, analytes, bands, stability |

**Do not use "MSS" as a primary V7 term.** It may appear only when (a) reporting a legacy
result, or (b) stating the mapping above. In V7 comparisons, the phrase is
"CSM/MSS-equivalent layer" so old and new numbers stay comparable.

---

## Biochemical theme — **LEGACY**

> **Status: LEGACY as of Phase 05 (archived decision A-13).** Themes were built (Phase 03,
> K = 5 archetypal, 4 accepted), measured (Phase 04), and retired: class top-1 on unseen
> molecules fell 0.855 at the CSM layer to 0.405 at the theme layer. The definition below is
> preserved verbatim because the Phase 03 artefacts remain on disk, fingerprinted and
> reproducible. **Do not use "theme" for new work.** The V7 term for the interpretable layer is
> **Chemistry Evidence**.

**Mathematical.** A column of the soft membership matrix `S ∈ ℝ₊^{M×K}` mapping `M` CSMs to
`K` themes. `S` is **sparse**, **non-negative**, and **row-normalised**
(`Σ_k S_{mk} = 1`), so a CSM distributes its membership across themes rather than being
forced to one parent.

**Scientific.** A broader chemistry, expressed as a soft non-negative combination of CSMs.
The intended set is chemical, e.g.:

- protein chemistry
- lipid chemistry
- nucleic chemistry
- carbohydrate chemistry
- organic-acid / energy chemistry
- sulfur / redox / cofactor chemistry

**Themes are chemistry, not biology.** A theme is never a disease, a pathway activity, a
phenotype, a process, or a clinical state. "Lipid chemistry" is admissible; "membrane
remodelling", "inflammation", or "tumour metabolism" are not. Those belong strictly to the
biological-interpretation layer, downstream of the BSV and outside the universal
representation.

**Soft, not hard.** No CSM is required to have exactly one parent. Shared biochemical
structure genuinely belongs to several themes, and forcing a single parent destroys
information — a lesson carried forward from V6.2, which already uses soft row-stochastic
membership with a temperature and a floor.

**Input.** CSM dictionary + CSM registry + curated chemical admissibility constraints.
**Output.** Theme registry + membership matrix `S` + per-CSM membership entropy.
**Provenance class.** **derived** from CSMs, with **curated** admissibility constraints.

---

## Biochemical State Vector — BSV — **LEGACY**

> **Status: LEGACY as of Phase 05 (archived decision A-14).** The BSV was defined as `Sᵀc` over
> the accepted themes and therefore inherits the theme layer's information loss exactly; its
> effective rank was 2.40 of a nominal K = 4 (risk R-12 realised). The definition is preserved
> because the Phase 04 reference frame remains a fingerprinted artefact and because **ΔBSV, the
> reference-normalised elevation and the DART trajectory definitions carry over unchanged to
> BSV2** — the algebra of "absolute coordinate, differences are derived" is what survives, not
> the particular basis. For new work use **BSV2**.

**Mathematical.** For a spectrum `x`, with CSM activation vector `c(x) ∈ ℝ₊^M` obtained by
fixed-dictionary non-negative projection,

$$t(x) = S^{\top} c(x), \qquad \mathrm{BSV}(x) = \big[t_1(x), t_2(x), \ldots, t_K(x)\big] \in \mathbb{R}_+^K$$

The BSV dimension **equals the number of biochemical themes, K**.

**Scientific.** The absolute continuous position of one spectrum in the fixed global
biochemical coordinate system.

**The BSV is:**

- **absolute** — a position, not a change;
- expressed in a **fixed global** coordinate system, identical for every spectrum ever
  projected under a given atlas version;
- **continuous** and **non-negative**;
- **deterministic** given the frozen atlas.

**The BSV is not:**

- a delta or difference;
- a hard label or a classification;
- a disease score, a risk score, or a diagnosis;
- a PCA/UMAP embedding.

### Derived quantities (all downstream of the BSV, none of them the BSV)

| Quantity | Definition | Note |
|---|---|---|
| **ΔBSV** | $\Delta\mathrm{BSV} = \mathrm{BSV}_2 - \mathrm{BSV}_1$ | signed; a *difference of two absolute vectors*. Never call a ΔBSV a BSV. |
| **Reference-normalised elevation** | $z_k = (t_k - \mu_k)/\sigma_k$ against the frozen reference frame | signed by construction; a *reading* of the BSV, not the BSV |
| **Cohort-standardised view** | standardisation within a study cohort | **visualised** only; cohort-dependent, therefore not portable |
| **DART trajectory** | $\mathrm{BSV}(E, t)$ — the BSV as a function of applied stimulus `E` and time `t` | a sequence of absolute BSVs in one fixed coordinate system |
| **Visualisation projection** | $y = P^{\top}(\mathrm{BSV} - \mu)$ for a *fixed, frozen* `P` | **visualised** only; `P` is never fitted at inference and `y` is never the canonical BSV |

**Input.** CSM activations + membership matrix `S`.
**Output.** `ℝ₊^K` vector + uncertainty + QC/OOD flags.
**Provenance class.** **derived**.

---

## Chemistry Evidence

**Status: CURRENT.** The V7 interpretable layer. Replaces *biochemical theme* (LEGACY) and
supersedes the eleven declared evidence axes of Phase 05 (archived A-16), subject to gate DG-06.

**Mathematical.** For a spectrum `x` with CSM activation `c(x) ∈ ℝ₊^49`, a frozen map
`E ∈ ℝ₊^{49×16}` and a frozen calibrator `Γ`,

$$e(x) = \Gamma\big(E^{\top} c(x)\big) \in \mathbb{R}_+^{16}, \qquad \sum_{k=1}^{16} e_k(x) \le 1$$

The shortfall `1 − Σ_k e_k(x)` is the **unassigned mass** and is reported, never redistributed
across the coordinates.

**Scientific.** A sixteen-dimensional probabilistic statement about *which chemistries the
evidence supports*, one coordinate per class of the frozen `v7_fine_16` ontology: acylglycerol,
carboxylic acid metabolite, chromophore pigment, fatty acid, free amino acid,
mono/oligosaccharide, nucleic acid polymer, peptide protein, phosphate metabolite,
phospholipid/sphingolipid, polysaccharide, purine, pyrimidine, small nitrogenous,
sterol/steroid, sulfur/thiol cofactor.

**Why sixteen.** Not a tuned hyperparameter. Sixteen is fixed by the evaluation ontology frozen
in Phase 00, which is also the label space of the frozen Tier-1 success criteria (S-01, S-03).
Choosing any other number would make V7 unmeasurable against its own bar.

**What it is and is not.**

| It is | It is not |
|---|---|
| a continuous, non-negative evidence coordinate | a hard classification |
| calibrated, with its ECE and discrimination reported | a probability anyone may threshold silently |
| accompanied by an explicit unassigned mass | a partition of unit mass across 16 bins |
| *beside* CSM retrieval, never replacing it | the input to molecular retrieval on its own |
| chemistry | biology — P-07 applies unchanged |

**Input.** CSM activation vector + frozen map + frozen calibrator.
**Output.** `ℝ₊^16` + unassigned mass + per-coordinate confidence + provenance.
**Provenance class.** **learned** (the map, offline in Phase 06) applied as **derived**.

---

## BSV2 — Biochemical Programmes

**Status: PLANNED (A-20).** Not implemented. Gate DG-07 can reject it, in which case Chemistry
Evidence remains the terminal interpretable layer.

**Mathematical.** A frozen non-negative programme dictionary `P ∈ ℝ₊^{K×16}` learned by
hierarchical NMF over the Chemistry Evidence matrix, with inference by frozen projection

$$\mathrm{BSV2}(x) = \arg\min_{b \ge 0} \; \lVert e(x) - b^{\top} P \rVert_2^2 \;\in\; \mathbb{R}_+^K$$

**Scientific.** A *biochemical programme* is a pattern of chemistry co-occurrence — which
chemistries tend to be evidenced together. `K` is selected on a Pareto frontier over
reconstruction, held-out chemistry prediction, programme stability, interpretability, mutual
information, noise robustness, calibration and compression. **Never on reconstruction alone**
(R-12), and never on accuracy alone.

**The critical distinction from the archived Meta Components (A-15).** Meta Components
factorised `A ∈ ℝ₊^{375×49}` — spectra × *motif* activations — and retained 0.185 of the CSM
layer's information. BSV2 factorises `Ev ∈ ℝ₊^{375×16}` — spectra × *chemistry* evidence. The
object being compressed is different. Whether that is enough is an open question (U-04), and it
is the reason DG-07 carries a pre-registered informativeness floor.

**Inherited unchanged from the legacy BSV.** BSV2 is **absolute**, non-negative, continuous, and
expressed in a fixed global coordinate system. ΔBSV2, reference-normalised elevation,
cohort-standardised views and DART trajectories are defined exactly as they were for the BSV,
and a difference is never called a BSV2.

**Input.** Chemistry Evidence **only** — never CSM activations directly.
**Output.** `ℝ₊^K` + uncertainty + programme provenance.
**Provenance class.** **learned** (offline, Phase 07) applied as **derived**.

---

## Molecular Retrieval

**Status: PLANNED (A-21).**

**Mathematical.** A ranking over the 154 canonical molecules combining the CSM-space similarity
with a soft chemistry prior:

$$\mathrm{score}(x, a) \;=\; f\big(\mathrm{sim}(c(x),\, r_a)\big) \;+\; \lambda \cdot \log \, e_{\kappa(a)}(x)$$

where `r_a` is molecule `a`'s reference activation vector, `κ(a)` its chemistry class, and `λ`
is fitted offline on training folds only. Prototype-plus-residual scoring and conformal
prediction sets are candidate refinements, each admitted only if it earns its place.

**Scientific.** Retrieval that uses chemistry as a *prior*, not a filter. A hard class filter
would make a class error unrecoverable; a soft prior re-weights and can be overruled by strong
spectral evidence.

**The baseline it must beat.** Direct cosine retrieval in CSM space: Split A molecule top-1
**0.605**, top-5 0.795 (Phase 05). Anything that does not beat that is not adopted.

**Input.** CSM activation + Chemistry Evidence.
**Output.** ranked top-k + calibrated confidence (+ conformal set, if justified).
**Provenance class.** **derived**, with a **learned** prior weight.

---

## Atlas

**V7 retains the term**, with a widened definition.

> **The GAIRA V7 Atlas** is the frozen, versioned collection of artefacts required for
> deterministic inference: the canonical preprocessing specification; the LSM dictionaries;
> the CSM definitions and projection basis; the CSM→theme membership matrix; the theme
> registry; the BSV reference statistics and normalisation frame; the OOD support; the full
> provenance chain; and the declared validation boundaries (domains, windows, and instrument
> conditions under which the atlas has been validated).

**The atlas is no longer only one global NMF basis.** In V5 the two were nearly synonymous —
`H` plus a thin overlay. In V7 the atlas is a layered bundle, and its fingerprint must cover
*every* layer, not just the projection basis. See
`architecture/ARTIFACT_AND_MANIFEST_SPEC.md`.

An atlas is identified by a version and a fingerprint, is immutable once frozen, and any
change to any layer requires a version bump. The current frozen V5 atlas is
`09ed804a40836f4a05a91ba10900cded` and remains in production throughout V7 development.

**Provenance class.** **learned + curated + derived**, frozen and versioned as a unit.

---

## Biological interpretation

**Mathematical.** A mapping `I: (BSV, domain context, priors) → interpretation`, applied
strictly *after* the BSV, with no path back into the representation.

**Scientific.** The domain-contextual reading of a biochemical state — what elevated lipid
chemistry might mean *in serum* versus *in an EV preparation* versus *in tissue*.

**Not part of the universal representation.** This separation is a hard architectural
boundary, and it is why the V5 engine keeps `domain.py` downstream of `bsv.py`. Domain
priors must never leak upstream into preprocessing, projection, CSMs, themes, or the BSV.
If they did, two labs measuring the same molecule under different assumed contexts would get
different coordinates, and the whole comparability argument collapses.

**Input.** BSV + domain context + uncertainty.
**Output.** Interpretation with confidence tiers, ambiguity, and multi-assignment support.
**Provenance class.** **curated** rules applied to **derived** quantities.

---

## Quick reference

| Term | Symbol | Space | Class | Introduced | Status |
|---|---|---|---|---|---|
| Canonical Raman reference | `x`, `x̄_a` | `ℝ₊^676` | curated / derived | input | ACTIVE |
| Chemical class | `κ` | partition of 16 | curated | organisational prior | ACTIVE |
| Local Spectral Motif | `h ∈ H_c` | `ℝ₊^676` | learned | Phase 01 | ACTIVE |
| Consensus Spectral Motif | `csm_m` | `ℝ₊^676` | learned | Phase 02 | ACTIVE |
| **CSM activation** | `c(x)` | `ℝ₊^49` | derived | inference | **ACTIVE — canonical** |
| **Chemistry Evidence** | `e(x)` | `ℝ₊^16` | derived | Phase 06 | PLANNED |
| **BSV2** | `b(x)` | `ℝ₊^K` | derived | Phase 07 | PLANNED |
| **Molecular Retrieval** | ranked list | — | derived | Phase 08 | PLANNED |
| ΔBSV2 | `b₂ − b₁` | `ℝ^K` | derived | analysis | PLANNED |
| Atlas | — | bundle | frozen | product | ACTIVE |
| Membership matrix | `S` | `ℝ₊^{49×4}` | derived + curated | Phase 03 | **LEGACY** |
| Biochemical theme | `t_k` | `ℝ₊` | derived | Phase 03 | **LEGACY** |
| Biochemical State Vector | `BSV(x)` | `ℝ₊^4` | derived | Phase 04 | **LEGACY** |
| Meta Component | `mc_j` | `ℝ₊^3` | learned | Phase 04.5 | **LEGACY — discarded** |
| Declared evidence axis | `a_j` | `ℝ₊^11` | curated + derived | Phase 05 | **LEGACY — superseded** |
| Legacy MSS | — | — | legacy | V5 | **use CSM** |
