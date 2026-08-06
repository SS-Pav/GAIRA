# GAIRA V7 — Rebuild Plan

The definitive implementation sequence. Ten phases, each with objectives, outputs, and a
**gate** that must pass before the next phase begins.

Companions: `PHASE_DEPENDENCY_MAP.md` (what blocks what),
`VALIDATION_AND_DECISION_RULES.md` (how each layer's size is chosen),
`SUCCESS_CRITERIA.md` (what replacement requires), `RISK_REGISTER.md` (what can go wrong).

**Status: nothing implemented. Phase 00 has not started.**

---

> ## ⚠ PHASE RENUMBERING — adopted 2026-08-06
>
> The canonical architecture merges what this plan called **Phase 01** (balanced reference
> construction) and **Phase 02** (Local Spectral Motif construction) into a single
> **Phase 01**, because they are one pipeline: balanced references exist only to be split by
> class and fitted. Every gate from both original phases is carried into the merged phase;
> none is dropped.
>
> | This plan (original) | Canonical (adopted) |
> |---|---|
> | Phase 01 — balanced reference construction | **Phase 01, Stage 1** |
> | Phase 02 — Local Spectral Motif construction | **Phase 01, Stage 2** |
> | Phase 03 — Consensus Spectral Motifs | **Phase 02** |
> | Phases 04–09 | shift down by one |
>
> See `context/ARCHITECTURE_COMPLIANCE_AUDIT.md` §5.

## Rules that apply to every phase

1. **Gates are binding.** A failed gate stops the phase. It does not get waived because the
   next phase is more interesting.
2. **Decision rules are pre-registered.** Every model-selection rule is written and committed
   *before* the sweep that it governs runs (P-12).
3. **Every phase ships a manifest** with inputs, config, seeds, code SHA, environment,
   outputs, gate results, and decisions. `code.dirty: true` invalidates the phase.
3b. **Every phase begins with an architecture check (P-16) and ends with an Architecture
   Compliance table (specification item · implemented? · evidence · PASS/FAIL) and a redrawn
   pipeline (P-17).** The gate opens only if every compliance row is PASS.
4. **Every phase ships a report** committed alongside the code that produced it.
5. **Nothing outside `GAIRA_v7_rebuild/` is modified** — with the single exception of the V7
   scaffold test under `tests/`.
6. **The V5 atlas is the control arm** in every comparison, at every phase where a comparison
   is meaningful.

---

# PHASE 00 — Benchmark lock and reproducibility baseline

**The most important phase in the plan.** Everything downstream is measured against what is
frozen here. Getting this wrong invalidates every later number, and — as V6.3 demonstrated at
some cost — measuring the wrong thing very carefully is still measuring the wrong thing.

### Objectives

1. **Freeze canonical preprocessing.** Window 450–1800 cm⁻¹, 2.0 cm⁻¹ step, 676 bins, asls →
   savgol → L2, with every parameter recorded. Note explicitly that
   `src/gaira/preprocessing/pipeline.py::common_grid()` defaults to the *legacy* 520–1750
   Ag-SERS window; V7 always passes the window explicitly.
2. **Freeze the V6.3 evaluation ontology.** Decide and record whether
   `results/v6_rebuild/v63_ontology_revalidation/` (currently uncommitted on disk) is
   committed as a versioned input or re-derived under a V7 manifest.
3. **Canonicalise molecule IDs.** NFKC + whitespace + case normalisation; resolve the known
   alias hazards (`riboflavin`/`riboﬂavin` with the U+FB02 ligature;
   `acetyl coenzyme a`/`acetyl-coa`, which is *also* a class-assignment inconsistency;
   `urea`/`ure`; missing-space fatty-acid names). Preserve enantiomers and anomers as
   distinct. Produce a manual review list of every merge and every deliberate non-merge.
4. **Define replicate groups.** Ratify `(canonical_id, excitation)` as the group key, with
   balancing applied at `canonical_id` level across groups.
5. **Define and freeze quality metadata** — the composite `q` used by Strategy B. Frozen
   *before* Phase 01 so it cannot become a tuned hyperparameter.
6. **Freeze the chemical-family partition**, resolving the three known partition problems:
   `unknown` (6 analytes — not a chemistry), `lipid` vs `fatty_acid` vs `triglyceride`
   overlap, and `polysaccharide` vs `saccharide`. Written chemical rationale per class.
7. **Freeze analyte-grouped CV splits** with three leakage checks.
8. **Freeze evaluation metrics** and the permutation-null / bootstrap-CI procedure.
9. **Reproduce the current atlas control** — confirm `09ed804a40836f4a05a91ba10900cded` and
   re-measure the V5 baseline under the frozen V7 evaluation harness, so the comparison in
   Phase 07 is like-for-like rather than against previously published numbers.
10. **Freeze the provisional success criteria** from `SUCCESS_CRITERIA.md` into final form.

### Outputs

| Output | Artefact |
|---|---|
| Dataset role map | `results/manifests/dataset_role_map_v7.csv` |
| Canonical analyte table | `canonical_analytes_v1.csv` (C-00) |
| Replicate groups | `replicate_groups_v1.csv` (C-01) |
| Quality metadata | `spectrum_quality_v1.csv` (C-02) |
| Class partition + rationale | `chemical_partition_v1.yaml` |
| Ontology table | `evaluation_ontology_v7.csv` |
| CV split manifest | `cv_splits_v1.json` (C-03) |
| Baseline metrics | `results/tables/phase00_baseline_metrics.csv` |
| Frozen control fingerprint | recorded + verified |
| Frozen success criteria | `SUCCESS_CRITERIA.md` marked final |
| Phase-00 report | `reports/PHASE_00_REPORT.md` |

### Gate

- [ ] No alias leakage: every surface form maps to exactly one canonical ID
- [ ] No replicate leakage: no canonical ID or replicate crosses a fold boundary
- [ ] All three `cv_splits_v1.json` leakage checks read `false`
- [ ] V5 baseline reproduced; fingerprint `09ed804a40836f4a05a91ba10900cded` verified
- [ ] All inputs versioned and hashed
- [ ] Splits deterministic (identical under re-run, on two machines)
- [ ] Class partition has a written chemical rationale for every class
- [ ] `unknown` class resolved (assigned or excluded from partitioning)
- [ ] Quality score `q` frozen and documented before Phase 01 begins
- [ ] Success criteria frozen

---

# PHASE 01 — Balanced reference construction

### Objectives

Compare, on the frozen splits:

| Arm | Strategy |
|---|---|
| **A** | all spectra, equal row weight — **the control (= V5 behaviour)** |
| **B** | analyte-balanced quality-weighted fitting |
| **C-mean** | mean prototype per analyte |
| **C-median** | per-bin median prototype |
| **C-trimmed** | trimmed-mean prototype |
| **C-medoid** | medoid prototype (a real measured spectrum) |
| **C-quality** | quality-weighted prototype |
| **B-uniform** | B with uniform `q` — sensitivity arm, isolates the balancing effect from the quality-weighting effect |

### Evaluation

- reconstruction (held-out, analyte-grouped)
- diagnostic-band fidelity
- replicate stability
- class balance achieved
- downstream control retrieval (V5-style global NMF on each arm, so arms are comparable
  *before* the architecture changes — this isolates the reference-construction effect)

### Two mandatory stratifications

1. **Report the effect restricted to the 87 replicated analytes, as well as corpus-wide.**
   80 of 167 analytes are singletons, for which A, B, and C are identical. Corpus-wide numbers
   will be diluted toward zero and will make the arms look falsely equivalent.
2. **Report single-excitation and multi-excitation analytes separately.** 41 analytes span
   multiple excitations; per-bin mean and median can distort band *shape* across excitations
   (peak positions are excitation-invariant, relative intensities are not). The medoid avoids
   this by construction. Phase 01 must also evaluate per-excitation prototypes as an
   alternative to collapsing across excitation.

### Outputs

- selected reference-construction strategy + rationale
- comparison tables and figures
- `balanced_references_v1.npz` / `.csv` (C-04) under the selected strategy
- discarded-variance asset if a prototype strategy is selected (within-analyte spread is the
  only direct measurement-uncertainty estimate available; it must be retained, not thrown away)
- `reports/PHASE_01_REPORT.md`

### Gate

- [ ] Selection rule stated in `VALIDATION_AND_DECISION_RULES.md` **before** the sweep ran
- [ ] No label supervision anywhere in the construction
- [ ] Selected method improves class balance without materially damaging spectral fidelity
- [ ] Replicated-analyte and multi-excitation stratifications both reported
- [ ] `B-uniform` sensitivity arm reported
- [ ] Control arm A included and reported honestly — **if A wins, that is the finding**

---

# PHASE 02 — Local Spectral Motif construction

### Objectives

For each chemical class:

1. sweep `k_c` within `1 ≤ k_c ≤ ⌊n_analytes(c)/2⌋`
2. run `R` repeated NMF / sparse-NMF fits (seed schedule + analyte-level bootstrap;
   **never** resample replicates — that leaks within-analyte structure and inflates stability)
3. align components across runs (Hungarian on cosine)
4. measure stability (recurrence + mean matched similarity)
5. quantify redundancy within the class
6. select stable LSMs against the pre-registered threshold
7. type each retained LSM: class-shared | subfamily | molecule-discriminating
8. sweep sparse-NMF vs plain NMF per class — not assumed
9. **report per-class source and excitation composition** (risk R-16: a class drawn
   overwhelmingly from one source may be modelling instrument response, not chemistry)
10. **test class-prior bias** (risk R-01): report every class whose local decomposition is
    driven by the partition rather than by spectroscopy

### Outputs

- one LSM dictionary per class; `lsm_registry_v1.json` (C-05)
- per-class reports incl. `k_c` selection curves, stability, redundancy, residual structure
- all basis spectra, plotted
- discarded-LSM record with reasons
- classes routed to the anchor mechanism (n < 2, or no LSM cleared stability)
- `reports/PHASE_02_REPORT.md`

### Gate

- [ ] Every retained LSM meets the pre-registered stability threshold
- [ ] Every class has documented motif coverage (or a documented reason for none)
- [ ] `k_c` selected per class by the pre-registered rule — **no arbitrary fixed `k`**
- [ ] `k_c ≤ ⌊n_analytes/2⌋` for every class
- [ ] Rare classes handled explicitly (anchor route, not duplication — P-11)
- [ ] Per-class source/excitation composition reported
- [ ] Class-prior bias tested and reported

---

# PHASE 03 — Consensus Spectral Motif construction

### Objectives

1. Pool all stable LSMs across classes.
2. Build the full-space similarity graph with **all six** edge features (spectral cosine,
   diagnostic-band overlap, peak-position agreement, bootstrap recurrence co-occurrence,
   activation co-occurrence, provenance overlap with within-class overlap discounted).
3. **Sweep the edge threshold** and report community stability across the sweep — a single
   unswept threshold is not acceptable evidence (risk R-07).
4. Compare integration methods **on evidence**:
   - hierarchical consensus clustering
   - Leiden / Louvain graph communities
   - spectral clustering
   - optional second sparse non-negative meta-factorisation
   - hybrid graph + factorisation
5. Select `M` on the pre-registered composite.
6. Derive CSMs with full provenance.
7. Admit rare-chemistry anchors (Strategy F) — quality gate, novelty gate, chemical
   justification, permanent `is_anchored` flag.
8. **If the second factorisation is selected**, explicitly verify that
   molecule-discriminating LSMs survive into distinguishable CSMs (risk R-06).

### Selection criteria for `M`

consensus stability · within-CSM spectral cohesion · between-CSM separation · chemical
coherence · retained LSM information · downstream held-out recovery · singleton penalty ·
redundancy penalty.

### Outputs

- `csm_dictionary_v1.npz`, `csm_registry_v1.json` (C-07)
- `lsm_graph_v1.json` (C-06) incl. the threshold sweep
- **integration-method comparison table — committed regardless of which method wins**
- CSM reference manual (one page per CSM: spectrum, bands, provenance, uncertainty, flags)
- `reports/PHASE_03_REPORT.md`

### Gate

- [ ] Every CSM has explicit, resolvable provenance (LSMs → classes → analytes → sources)
- [ ] CSMs meet the pre-registered stability threshold
- [ ] `M` quantitatively justified against the pre-registered composite
- [ ] Integration method chosen on evidence; **full comparison table published**
- [ ] Singletons and anchors flagged, counted, and reported — never hidden
- [ ] Threshold sweep performed; selection sits in a stable region, not at a single cut
- [ ] If meta-NMF selected: molecule-discriminating structure survival verified

---

# PHASE 04 — Biochemical theme construction

### Objectives

1. Derive themes **from CSMs** (not asserted over them — this is the direct response to L-05).
2. Sweep `K`.
3. Score each `K` on: information retained · mutual information with chemistry · chemical
   admissibility · calibration · held-out superclass retrieval · stability · compression ·
   interpretability.
4. Generate soft, sparse, row-normalised memberships `S`.
5. **Demonstrate that the theme layer adds value over the CSM layer** — or record that it does
   not. Precedent: at V6.2, `theme_posterior` and `theme_raw` were numerically identical at
   every metric on every ontology; the posterior machinery changed no decisions. A theme layer
   that merely relabels CSMs is decorative (risk R-11).

### Outputs

- `theme_registry_v1.yaml`, `theme_membership_v1.npz` (C-08)
- membership entropy per CSM
- theme graph
- `K` sweep with the Pareto frontier
- `reports/PHASE_04_REPORT.md`

### Gate

- [ ] Themes represent coherent chemistry
- [ ] **No disease, pathway, process, or phenotype labels** (P-07)
- [ ] No hard one-parent requirement; soft membership retained
- [ ] `K` justified on a Pareto frontier by the pre-registered rule
- [ ] Theme layer's value over the CSM layer measured and reported either way

---

# PHASE 05 — BSV construction and normalisation

### Objectives

Define the absolute BSV and every derived form, keeping them rigorously distinct:

| Form | Nature |
|---|---|
| **absolute BSV** | `Sᵀc(x)` — the canonical coordinate |
| reference-normalised elevation | z-scored against the frozen frame — signed, derived |
| **ΔBSV** | difference of two absolute BSVs — signed, derived |
| cohort-standardised view | visualisation only; cohort-dependent, not portable |
| DART trajectory | sequence of absolute BSVs |
| visualisation projection | frozen PCA applied — **not the canonical BSV** |

Also determine: raw activation handling, normalisation, reference centring, uncertainty
propagation, OOD scoring, confidence tiers, and **the effective rank of the BSV space measured
separately from `K`** (risk R-12; precedent: the V5 24-component space had participation ratio
15.2 — a 38% gap between nominal and effective dimensionality, visible only because it was
measured).

### Outputs

- `bsv_reference_v1.json` (C-09), OOD support, frozen visualisation transform
- reference distributions per axis
- worked examples end to end
- `reports/PHASE_05_REPORT.md`

### Gate

- [ ] BSV deterministic
- [ ] Absolute and delta forms not conflated anywhere in code, artefacts, or prose
- [ ] Every axis interpretable, with named supporting CSMs and chemistry
- [ ] Uncertainty propagated; singleton/anchor-supported axes carry inflated uncertainty
- [ ] Effective rank reported alongside `K`

---

# PHASE 06 — End-to-end V7 engine integration

### Objectives

Wire one canonical inference path: preprocessing → fixed LSM/CSM projection → CSM activation
→ themes → BSV → evidence → QC → uncertainty → provenance. Freeze the atlas bundle. Verify
every invariant.

### Outputs

- V7 engine (parallel to, not replacing, `src/gaira/engine/`)
- versioned output schema (C-10)
- API-ready interface
- frozen atlas bundle + `MANIFEST.json` with the multi-layer fingerprint
- reproducibility tests
- `reports/PHASE_06_REPORT.md`

### Gate

- [ ] **No fitting during inference** — static check for `fit`/`fit_transform`/`partial_fit`/RNG
- [ ] **Batch independence** — output identical alone vs in a batch of N
- [ ] Clean clone runs frozen inference with no lab volume and `GAIRA_DATA_ROOT` unset
- [ ] All assets fingerprinted; multi-layer atlas fingerprint verified on load
- [ ] Deterministic output, verified twice and on two machines
- [ ] Domain isolation: no domain object reachable from any pre-BSV module
- [ ] LSM layer retained in the bundle (needed by the future SERS observation model)

---

# PHASE 07 — Full in-domain Raman validation

**The decision phase.** Everything before it is construction; this is where V7 either earns
replacement or does not.

### Objectives

Evaluate the complete corpus under the frozen analyte-grouped holdouts, at every layer,
against the V5 control measured under the same harness.

### Required reporting, at every layer

LSM retrieval · CSM top-1 / top-3 · fine-family retrieval · theme top-1 / top-3 ·
broad-superclass retrieval · system-level performance (if retained) · MRR · balanced accuracy ·
macro-F1 · permutation null · bootstrap confidence intervals · calibration · reconstruction ·
diagnostic-band fidelity.

### Required failure analysis

The V6.3 waterfall structure, which is the most informative diagnostic the project has:

| Category | Meaning |
|---|---|
| **true projection failures** | the representation genuinely cannot separate them — the number that matters |
| **semantic rescues** | V5 failures that V7 resolves |
| **semantic degradations** | V5 successes that V7 breaks — reported with equal prominence |
| **stable recoveries** | correct under both |

Baseline to beat (V5, MSS layer, n=167): 54 failures, of which **31 (57.4%) were true
representation errors**.

### Outputs

- unified V7 Raman validation report
- per-analyte appendix
- head-to-head comparison with the current atlas under one harness
- **replacement recommendation**

### Gate

V7 must outperform or materially improve on the current atlas under the criteria frozen in
Phase 00. **If it does not, the outcome is a documented negative result and a retained V5
atlas** (P-13). The bar is not lowered to fit the result.

---

# PHASE 08 — Chemistry-aware learning *(deferred)*

Begins **only** after the best unsupervised V7 architecture is frozen as a candidate — so
that any gain is attributable to learning rather than to architecture.

### Compare

graph-regularised NMF · discriminative NMF · fixed-dictionary metric learning ·
hybrid spectroscopy-prior CSM mapping · weak chemical supervision.

### Constraints

- **No disease labels.**
- **No SERS training.**
- All learning nested inside analyte-grouped CV — no held-out analyte information may enter
  model selection.

### Outputs

learning-gain attribution · comparison with unsupervised V7 · overfitting analysis ·
calibration · `reports/PHASE_08_REPORT.md`

### Gate

- [ ] Held-out gains demonstrated
- [ ] Interpretability preserved — a gain that costs provenance or band-level explanation is
      not accepted
- [ ] No label or domain leakage

---

# PHASE 09 — Targeted corpus expansion *(deferred)*

Driven by V7's own residual analysis: which spectral directions remain genuinely unsupported.

### Candidate priorities

sterols/steroids · porphyrins/heme · flavins · phosphate chemistry ·
phospholipids/sphingolipids · organic acids · sulfur/redox cofactors.

Current support for reference: sterol 9 analytes (7 uncovered), flavin motif coverage 2
analytes (1.2%), phospholipid 2 (100% uncovered), carotenoid 2 (100% uncovered),
nucleic_acid 3 (100% uncovered), sphingolipid **absent entirely**.

### The rule

> **Do not add datasets merely because they are available.** Each addition must address a
> *measured* missing spectral direction, identified by V7 residual analysis, with a stated
> expected effect — and the effect must be re-measured after ingestion.

### Gate

- [ ] Every addition traced to a measured residual direction
- [ ] Expected effect stated before ingestion
- [ ] Effect re-measured after ingestion
- [ ] Corpus rebalance re-run; the class partition revisited if the addition changes it

---

## Phase sequence summary

```
00 Benchmark lock ──▶ 01 Balanced references + LSMs ──▶ 02 CSMs ──▶ 03 Themes
                                                                       │
                                                                       ▼
                                   06 Raman validation ◀── 05 Engine ◀── 04 BSV
                                            │
                                ┌───────────┴───────────┐
                                ▼                       ▼
                    07 Chemistry-aware learning   08 Corpus expansion
                          (deferred)                  (deferred)
```

Numbering is the one adopted 2026-08-06 (see the ⚠ block at the top of this document). The
per-phase sections below still carry their **original** headings — original Phase 01 + 02 are
canonical Phase 01, original Phase 03 is canonical Phase 02, and so on down. The mapping table
at the top is authoritative.
