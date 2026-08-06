# GAIRA V7 — Terminology and Definitions

Binding vocabulary for all V7 documents, code, artefacts, and reports. Where V7 diverges
from earlier naming, the legacy term is recorded and marked legacy.

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

**What it is not — three prohibitions:**

1. It is **not a disease label**. No class in V7 refers to a condition, phenotype, or process.
2. It is **not the inference output**. V7 never predicts class. Class exists only to
   allocate decomposition capacity fairly.
3. It is **not supervision inside the local fit**. The decomposition within a class is
   unsupervised; the class only decides *which spectra enter which fit*.

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

## Biochemical theme

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

## Biochemical State Vector — BSV

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

| Term | Symbol | Space | Class | Layer |
|---|---|---|---|---|
| Canonical Raman reference | `x`, `x̄_a` | `ℝ₊^D`, D=676 | curated / derived | input |
| Chemical class | `c` | partition | curated | organisational prior |
| Local Spectral Motif | `h ∈ H_c` | `ℝ₊^D` | learned | Phase 02 |
| Consensus Spectral Motif | `csm_m` | `ℝ₊^D` | learned | Phase 03 |
| CSM activation | `c(x)` | `ℝ₊^M` | derived | inference |
| Membership matrix | `S` | `ℝ₊^{M×K}` | derived + curated | Phase 04 |
| Biochemical theme | `t_k` | `ℝ₊` | derived | Phase 04 |
| Biochemical State Vector | `BSV(x)` | `ℝ₊^K` | derived | Phase 05 |
| ΔBSV | `BSV₂ − BSV₁` | `ℝ^K` | derived | analysis |
| Atlas | — | bundle | frozen | product |
| Legacy MSS | — | — | legacy | **use CSM** |
