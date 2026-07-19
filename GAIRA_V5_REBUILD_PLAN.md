# GAIRA V5 Rebuild Plan — evidence-dependent, sequential, adaptive

**Status:** planning only (no implementation in this pass). **Branch:** `gaira-v5-rebuild-plan`.
**Supersedes** the assumptions that the 11 axes emerged from UMAP, that 202 RamanBioLib rows = 202 unique molecules, that biological spectra are molecular grounding, that serum perturbation datasets calibrated the axes, that the substrate layer was validated, that the physics atlas numerically affects inference, and that the demo and production inference paths are equivalent. Basis: the forensic corpus/physics audit (branch `gaira-data-and-physics-audit-2026-07`) and the V4 architecture/evidence pass (`gaira-v4-...`), summarized in `GAIRA_CURRENT_STATE_AND_ARCHITECTURE_V4.md` and the `data_audit/` registries.

> **Governing principle:** Do not decide the representation, ontology, axis count, or the need for an encoder in advance. Build the pipeline so each decision is *earned* from the preceding evidence. The first implementation sprint stops after preprocessing + comparability; that result decides the next step.

---

## 1. Corrected current state (what GAIRA actually has)

### 1.1 Direct molecular grounding (provisional audited totals; traceable to V4 registries)
| Source | Modality / substrate | Excitation | Analytes | Full spectra | Notes |
| --- | --- | --- | --- | --- | --- |
| RamanBioLib | spontaneous **Raman** (CaF₂/glass/metal-ring) | 785/1064/532/488… | **141 unique** | **202** compound×substrate×laser rows | digitized (DOI 10.1002/jrs.1734); raw parquet 272,902 pts |
| amino_acid_raman_grounding | spontaneous **Raman** powder | — | 20 | 20 | 1/analyte |
| adenine_sers_control | **Ag-SERS** (bAgNPs) | 785 | 1 | ~16 | dose series |
| sers_metabolite_63 | **Ag citrate colloid SERS** (Lee–Meisel) | **633** | **63** | 63 | averaged, bg-subtracted, pure solutions (NOT Au) |
| Gobbato/Bonifacio serum Ag-colloid | pure **Ag-SERS** (265) + pure **Raman** powder (153) + serum perturbations + donor serum | 785 | 53 (+51 Raman) | 418 pure | mixed roles — must be split |
| ORC-roughened Ag metabolite set | **ORC-Ag SERS**, peak-level only | not in SI | **24** | **0** (454 peaks) | no reconstructable spectra |

**Provisional grounding summary (traceable, not marketing):** ~**228 analytes** across direct sources; ~**375 Raman full spectra**; ~**344 Ag-SERS full spectra**; **0 confirmed Au-SERS grounding spectra** (a real acquisition gap); **24 analytes with peak-level ORC-Ag evidence only**. All counts are provisional audited totals from `data_audit/v4_direct_grounding_sources.csv` and `v4_grounding_analyte_registry.csv`.

### 1.2 Controlled Perturbation Evaluation (held-out; NOT calibration)
These datasets are **held out** from: axis definition; observation-model fitting (except explicitly designated development-only pairs); motif-weight fitting; coordinate calibration; model training. Examples: adenine concentration response, ergothioneine spike-in, uricase depletion, hypoxanthine spike, ¹⁵N-uric-acid isotope, 53 serum metabolite spike-ins, inter-instrument adenine, other dose/enzyme challenges.
**Purpose:** *to test whether a model built from independent molecular grounding produces the expected biochemical response under known controlled perturbations.* (Registry: `data_audit/v4_controlled_perturbation_evaluation_registry.csv`.)

### 1.3 Biological challenge sets
EV, serum, plasma, disease, donor, patient datasets (13). **~180,000 spectra do NOT represent molecular grounding depth**; **~760 independent human samples** underpin much of that total (>99% technical scans / augmented / experimenter-averaged). **Biological mixtures must not define the core molecular coordinate system.** (Registries: `data_audit/biological_dataset_registry.csv`, `biological_cohort_registry.csv`.)

### 1.4 Current software state
- The **demo** (`gaira_demo_reasoning_v3_1/gaira_core`) and **production** (`src/gaira/base2`+`base3`) use **different deterministic inference paths** (top-axis agrees 5/6 on references; demo has 11 motifs, production 50).
- **Production `base2`/`base3` is richer and should become the canonical engine**; the demo should ultimately be a **presentation layer** over it (no duplicate scoring engine).
- The current **substrate layer is not empirically validated** (no cross-substrate benefit; blind to Au/planar/excitation).
- The production **42-effect substrate engine is dormant** (imported by nothing).
- The **physics atlas affects explanations only** (0 numeric BSV effect).
- The **11-axis ontology is curated**, not emergent (hard-coded `BIOLOGY_AXES_V11`; 6 of 11 are inherited splits).

---

## 2. V5 scientific objective
> Build one canonical, scientifically defensible GAIRA inference path in which heterogeneous Raman and SERS reference data are transformed into a comparable biochemical representation, the structure of the molecular grounding corpus is explored transparently, and biochemical coordinates are defined **only if** stable, interpretable, and independently testable structure emerges.

Intended high-level architecture:
```
Raw molecular reference spectra
  → canonical preprocessing & QC
  → mode/substrate-aware observation layer
  → canonical spectral representation
  → unsupervised structural analysis
  → interpretable biochemical ontology
  → frozen biochemical coordinate system
  → BSV projection
  → MSS analyte-support layer
  → domain-aware interpretation
  → controlled perturbation evaluation
  → biological challenge evaluation
```
**Rules:** each phase depends on evidence from the previous; no later phase is implemented if earlier assumptions fail; scientific conclusions drive implementation choices; everything remains deterministic, inspectable, and versioned.

---

## 3. Canonical inference path (target package layout)
V5 converges toward **one package under `src/gaira`**:
```
src/gaira/data/           canonical registries · loaders · provenance · metadata schemas
src/gaira/preprocessing/  spectral preprocessing · resampling · normalization · QC
src/gaira/observation/    modality/substrate observation models · reliability maps · transforms · uncertainty
src/gaira/representation/ direct spectral features · PCA/factorization · embedding adapters (if needed later)
src/gaira/ontology/       component interpretation · biochemical axes · hierarchy · evidence provenance
src/gaira/inference/      coordinate projection · BSV construction · confidence
src/gaira/mss/            analyte-level supporting evidence
src/gaira/physics/        ambiguity · collision · substrate reliability · OOD warnings
src/gaira/evaluation/     controlled perturbation tests · biological challenge tests · regression tests
```
The demo eventually calls this package and maintains no duplicate scoring engine. **Not implemented in this planning pass.** Existing `src/gaira/base2`/`base3`/`substrate`/`atlas` are the seed material for `representation`/`inference`/`observation`/`physics`.

---

## 4. Phase 0 — freeze inputs & data contracts (data governance FIRST)
Canonical registries: `grounding_spectrum_registry`, `grounding_analyte_registry`, `controlled_perturbation_registry`, `biological_dataset_registry`, `acquisition_domain_registry`, `substrate_registry`, `physics_evidence_registry` (extend the existing V4 `data_audit/` CSVs into `src/gaira/data/`).

**Per-spectrum required fields:** `spectrum_id, analyte_id, canonical_analyte_name, source_dataset, raw_path, modality, sers_or_raman, substrate_material, substrate_geometry, colloid_or_planar, excitation_nm, instrument, matrix, concentration, replicate, independent_measurement, raw_or_processed, wavenumber_min, wavenumber_max, point_count, quality_flags, intended_role, allowed_for_representation_learning, allowed_for_evaluation_only`.

**Explicit rules for:** duplicate handling (RamanBioLib's 61 duplicate compound×substrate rows); averaging (metabolite-63's 3-scan pre-averages; keep replicate provenance where it exists); technical replicates vs independent measurements; multiple lasers (RamanBioLib 785/1064/532/488); multiple substrates; **peak-only evidence** (ORC-Ag — flagged `allowed_for_representation_learning=false`, usable only for peak/collision); mixture spectra (excluded from grounding); missing metadata (excitation/conc absent for ORC-Ag); spectra that cannot safely enter joint analysis.

**Gate:** *No representation analysis begins until every included spectrum has sufficient provenance and acquisition-domain metadata.* Peak-only and biological/perturbation spectra are tagged out of the representation set at ingestion.

---

## 5. Phase 1 — canonical preprocessing & comparability audit (first scientific phase)
**Goal:** not merely to preprocess, but to determine whether spectra from different sources can be compared without destroying meaningful chemistry.

**Candidate steps:** raw validation · wavenumber-axis correction · cosmic-ray handling · baseline correction · optional smoothing · common-window selection · resampling to a shared grid · intensity normalization · QC metrics. **Do not assume one pipeline is valid for every modality.** Test at least: spontaneous Raman; Ag-colloid SERS; ORC-Ag peak evidence; any future Au-SERS.

**Alternatives to compare** — Baseline: airPLS · ASLS · rubber-band/polynomial (where appropriate). Smoothing: none · conservative Savitzky–Golay. Normalization: L2/vector · area · SNV · robust-percentile · internal-band (only where justified). Representation: full intensity · derivative · peak-presence · peak-weighted.

**Required analyses:** replicate consistency; analyte separability; source/substrate/excitation separability; preservation of known analyte bands; preprocessing sensitivity; over-smoothing detection; baseline residuals; outliers; missing-range effects.
**Required visualizations:** before/after overlays; per-source examples; replicate-correlation distributions; source/substrate PCA; analyte distance matrices; preprocessing-sensitivity heatmaps; QC dashboards.

**Decision gate — proceed if** replicates become internally consistent, known analyte features are preserved, spectra share a defensible common wavenumber space, and source effects are measurable/interpretable. **Stop or stratify if** preprocessing removes analyte-specific bands, source/instrument effects dominate irreducibly, modality-specific pipelines are required, or peak-only and full-spectrum data cannot combine directly.
**Output:** `reports/GAIRA_V5_PHASE1_PREPROCESSING_AND_COMPARABILITY_REPORT.md` — recommends **one global pipeline OR explicitly separate modality-specific pipelines.**

**Phase 1 outcome (2026-07-18):** completed. Spectra share a common window; but same-analyte cross-modality similarity is low (cosine 0.25–0.53), modality leaks into structure, and only **7 analytes were matched** across the loaded Raman/Ag-SERS sources — too thin to estimate any cross-mode model. **Root cause: an observation model was being attempted before the molecular grounding corpus was fully integrated (the Gobbato pure Raman + pure Ag-SERS corpus, which alone holds ~43 internally-matched analytes at 785 nm, had not been loaded).** Scientific sequencing error, corrected by inserting Phase 1.5.

---

## 5.5 Phase 1.5 — Canonical Grounding Corpus Completion (MANDATORY before any observation-model work)

**Rationale.** No representation or observation-model decision may be made on an incomplete grounding corpus. Phase 1.5 completes the corpus and re-quantifies matched analytes before anything else.

### V5 simplification — 785 nm only
The V5 canonical biochemical coordinate system is built **using 785 nm spectra only.** All other excitation wavelengths are **excluded from the V5 representation pipeline** (not deleted) and remain fully indexed with provenance so they can later support multi-excitation observation models. Purpose: remove excitation wavelength as a nuisance variable while we test whether biochemical structure emerges. RamanBioLib's 785-nm subset qualifies; its 532/1064/488/… subsets are indexed-but-excluded; **metabolite-63 is 633 nm → excluded from V5**.

### Objectives
1. **Re-audit every direct grounding source** and decide eligibility. **Include** iff: direct molecular reference · 785 nm · sufficient metadata · usable full spectra. **Exclude** biological mixtures, controlled-perturbation datasets, peak-only datasets, and non-785 spectra. Peak-only (ORC-Ag) stays available later for MSS.
2. **Fully integrate the Gobbato corpus** (highest priority): parse + load pure **Raman** metabolite powders and pure **Ag-SERS** metabolite spectra (B&WTek 785 nm) → analytes, spectra, matched analytes, replicates, concentrations, preprocessing needs; one canonical registry.
3. **Reconcile analyte synonyms** — synonyms, capitalization, salt/hydrate forms, common names, abbreviations → `canonical_analyte_registry_v5` (no duplicate analytes).
4. **Rebuild the grounding summary** — analytes; 785 Raman spectra; 785 Ag-SERS spectra; matched Raman/Ag analytes; spectra entering representation learning; spectra excluded + reasons.
5. **Re-run the overlap analysis** — analytes available in BOTH 785 Raman AND 785 Ag-SERS. **Only quantify overlap; do not estimate an observation model.**

### Prohibited in Phase 1.5
PCA, hierarchical clustering, NMF, embeddings, ontology construction, observation-model fitting, BSV, MSS.

### Output & gate
`GAIRA_V5_PHASE1_5_GROUNDING_COMPLETION_REPORT.md`. Proceed to Phase 2 only if the completed 785 nm corpus provides enough matched analytes + coverage that emergent-structure analysis is defensible; otherwise trigger a further data-acquisition phase.

---

## Representation philosophy (explicit staged hierarchy)
The post-Phase-1.5 question is **not** "can we build an observation model?" It is **"Does the completed 785 nm molecular grounding corpus contain stable biochemical structure?"** Stages, each attempted only if the prior fails: **Stage A — direct spectra · Stage B — chemically-constrained features · Stage C — learned embeddings** (grounding-only, nuisance-invariant).

## Observation model — a hypothesis, not an assumption
Hypothesis **H1**: *"A shared biochemical representation exists across Raman and Ag-SERS observations."* Observation-model development begins **only** if the completed corpus supports H1; otherwise maintain **modality-stratified representations**.

---

## 6. Phase 2 — Canonical Representation Discovery (renamed; was "observation-layer feasibility")
**Objective:** determine whether stable biochemical structure emerges from the completed 785 nm grounding corpus (Stage A→B→C). The matched-analyte observation-layer feasibility detail below is subsumed and pursued only if H1 is supported after Phase 1.5.

### Stage A — Direct Spectral Representations — COMPLETE (2026-07-18)
**Decision: Outcome B — modality-stratified representation defensible.** Evidence (479-spectrum audited corpus: 214 Raman + 265 Ag-SERS, 87 analytes, 51 matched; adenine concentration series excluded as controlled perturbation):
- Direct spectra recover chemistry **within** a modality (Raman-only clustering ARI vs analyte 0.49–0.52; PC1 loadings bootstrap-stable).
- They do **not** support a single shared Raman/Ag-SERS coordinate system: even under the best preprocessing (SNV), a modality classifier scores balanced accuracy 0.83 (chance 0.50) and cross-modal top-1 retrieval is only 0.16.
- A **weak but statistically-significant** residual cross-modal signal exists (matched > unmatched cosine, permutation p ≤ 0.02); peak positions do **not** align across modalities (SERS shifts/re-weights bands).
- **Preprocessing is decisive:** SNV suppresses modality (bal-acc 0.83, joint chemistry ARI 0.15 > nuisance 0.02); L2 lets modality/source dominate (bal-acc 0.94, nuisance ARI 0.26).
**Implication:** represent Raman and Ag-SERS separately; align at analyte/ontology level, not in raw spectral space (supports **H4-preliminary** over **H1**). **Next:** Stage B chemical features — test whether chemistry-aware, modality-invariant features strengthen the residual cross-modal signal before revisiting a shared space. Report: `GAIRA_V5_PHASE2_STAGE_A_DIRECT_REPRESENTATION_REPORT.md`; notebook `results/v5_rebuild/phase2_stage_a/`. **Stage B/C not started.**

The distinction Stage A forces: **observation** ≠ **biochemical representation**. The shared layer, if any, represents biochemical *evidence*, not a shared raw spectral space (see `GAIRA_V5_ARCHITECTURE_AND_SCIENTIFIC_CONTEXT.md`).

### Stage B — Biochemical Representation Strategy Benchmark — COMPLETE (2026-07-19)
**Decision: Outcome B4 — modality-stratified representations retained; no shared biochemical representation supported by the current corpus.** Under a leakage-safe framework (splits A held-out-analytes / B held-out-matched-pairs / C replicate-group / D source — D infeasible for single-source Ag-SERS), evaluated on held-out matched-analyte cross-modal retrieval (chance top-1 ≈ 0.098):
- **Best held-out MRR = 0.460 (I1 adaptive regions) ≈ direct_SNV 0.452** — CIs overlap, i.e. no material improvement over direct spectra by any interpretable or encoder representation.
- **Encoders underperform and collapse:** cross-modal top-1 0.08–0.18 (< direct 0.28); cross-analyte duplicate embedding 0.96–1.00; modality leakage driven to 0.91–1.00; held-out-analyte family retrieval 0.48–0.50 (< direct 0.74); within-modality Raman ARI 0.53–0.77 (< direct 0.94). Dual < shared encoder; VICReg avoids collapse only by destroying retrieval.
- **Candidates tested:** direct (SNV/L2/deriv reproduction), interpretable I1 regions / I2 multiscale / I3 sparse dictionary / I4 NMF, encoders E1 shared / E2 dual / E3 modality-specific / E2+triplet / E2+VICReg, and E4 hybrid (E1+I1). **Failed approaches:** all encoders (collapse + underperformance) and the hybrid (< direct).
- **Frozen representation:** **none as a shared representation.** Working representation for any interim downstream use = modality-stratified direct SNV (Raman held-out within-modality ARI 0.94), with I1 adaptive regions as an auditable interpretable companion.
- **Source-generalization caveat (unresolved):** Ag-SERS single-source → no observation-domain-invariance claim; cross-modal results are within the present 785 nm matched corpus.
- **Hypotheses:** H1 not supported by current corpus; H1c/H1d/H1e/H3 rejected; H2 not supported / H2a weakly supported; H4 reaffirmed; H7 rejected (encoder corpus-insufficiency confirmed).
Report: `GAIRA_V5_PHASE2_STAGE_B_REPRESENTATION_STRATEGY_REPORT.md`; notebook `results/v5_rebuild/phase2_stage_b/`; code `src/gaira/evidence/`. **Stage C/D not started.**

<details><summary>Original Stage B authorization spec (for reference)</summary>

### Stage B — Biochemical Representation Strategy Benchmark — (authorized 2026-07-19)
**Objective:** decide *how* to represent biochemical evidence so that shared chemistry is preserved across observation domains, by comparing two branches (plus an optional hybrid) under one leakage-safe evaluation framework. **This is an encoder feasibility study and representation benchmark, not foundation-model training.** Corpus is small (479 spectra, 87 analytes, 51 matched, largely single-source Ag-SERS) → small regularized models only; no transformer; no pretraining; training-set performance not interpreted; replicates are not independent semantic samples; no biological/perturbation/disease data.

- **Branch I — interpretable evidence representations:** I1 adaptive spectral regions · I2 multiscale/wavelet coefficients · I3 sparse dictionary codes · I4 non-negative basis activations. Fitted on **training data only**, mappable back to wavenumber space.
- **Branch II — encoder embeddings (small, regularized):** E1 shared 1D encoder · E2 dual encoder (primary hypothesis) · E3 modality-specific encoders (no cross-modal alignment) · E4 hybrid encoder + sparse evidence projection (only if E1–E3 are stable).
- **Objectives compared (predeclared, compact):** supervised contrastive (analyte identity), cross-modal InfoNCE, triplet/margin, VICReg-style regularization; reconstruction only as auxiliary; modality-adversarial only as a secondary experiment. Chemical-family labels are **evaluation-only** in the primary benchmark.
- **Evaluation (same framework for all reps):** held-out-analyte and held-out-matched-pair cross-modal retrieval (top-k, MRR, RNN, permutation nulls, analyte-bootstrap CIs); modality/source leakage; within-modality chemistry retention; seed/split stability; embedding-collapse & source-shortcut diagnostics; interpretability (attribution/sparse probes) and uncertainty.
- **Predeclared splits:** A held-out analytes · B held-out matched pairs · C replicate-group holdout · D source sensitivity (noted where impossible due to single-source Ag-SERS). Hyperparameters chosen on nested/validation splits, never on the final test set.
- **Decision gate (choose exactly one):** **B1** interpretable selected · **B2** encoder selected (state shared/dual/modality-specific) · **B3** hybrid selected · **B4** modality-stratified retained (no shared representation supported) · **B5** corpus insufficient for encoder conclusions (recommend exact data additions). Do not force a winner; do not present encoders as inherently superior.
**Outputs:** `GAIRA_V5_PHASE2_STAGE_B_REPRESENTATION_STRATEGY_REPORT.md`; notebook `results/v5_rebuild/phase2_stage_b/`; code under `src/gaira/evidence/`.
</details>

### Stage C — Targeted grounding-data acquisition & interpretable refinement — RE-SCOPED (post-B4)
Stage B selected **B4**, so Stage C is **NOT** encoder scaling. It is re-scoped to remove the corpus bottleneck that made every encoder collapse and made a shared representation unsupportable, then to re-run the Stage B benchmark:
- **Data (the operative gap):** multi-source Ag-SERS (break the single-source confound so Split D becomes feasible); Au-SERS references (currently zero); external matched analytes beyond the Gobbato instrument ecosystem; more matched analytes overall (51 is feasibility, not training, scale).
- **Interpretable refinement:** iterate I1 adaptive regions / I2 multiscale as the retained auditable representation.
- **Re-benchmark:** only after the above, re-run Stage B with the enlarged corpus; encoders are revisited **only** if the enlarged corpus first shows a shared signal that direct/interpretable can exploit. Do not scale an encoder on the current corpus.
**Encoder scaling / pretraining / transformers remain OUT of scope until a shared signal is demonstrated on a larger, multi-source corpus.**

### Stage D — Emergent biochemical ontology — GATED
Begins only after a representation is **frozen** (by Stage B, or Stage C if scaling was required). Then: stable latent-factor analysis, clustering, biochemical interpretation, ontology versioning. **Ontology construction is not pre-authorized** and starts only when a sufficiently stable representation exists.

**The representation/observation layer must never fabricate spectral bands.**

---

## 7. Phase 3 — canonical spectral representation study (after the observation decision)
Evaluate in sequence. **A — direct full-spectrum** (consistently preprocessed; domain-stratified or observation-aligned; one analyte-level centroid per valid condition; replicate uncertainty retained): PCA, sparse PCA, factor analysis, NMF, ICA, multiblock PCA. **B — chemically-constrained features** (peak presence/intensity, band families, derivatives, motif evidence, ORC-Ag peak evidence, substrate reliability). **C — learned embeddings** *only if A and B fail*: small 1D autoencoder, Siamese/contrastive on replicate/analyte identity, masked-spectrum reconstruction, supervised analyte-family encoder, or a pretrained Raman encoder if provenance/compatibility acceptable. **No disease labels; no perturbation test sets for training.**

Assess each: replicate compactness; analyte discrimination; family clustering; substrate/excitation/dataset leakage; seed & bootstrap stability; leave-source-out generalization; interpretability; unseen-analyte projection; suitability for a frozen coordinate system. Visuals: PCA score/loadings, dendrograms, analyte heatmaps, family/substrate/excitation-colored projections, bootstrap stability, kNN graphs, variance explained, source-leakage metrics.
**Decision hierarchy:** prefer direct spectra if stable chemistry; else chemically-constrained features; embeddings only if simpler fail; **reject any representation that clusters primarily by source/substrate/excitation.**
**Output:** `reports/GAIRA_V5_PHASE3_REPRESENTATION_SELECTION_REPORT.md` — selects **canonical representation v1** or concludes more data are needed.

---

## 8. Phase 4 — emergent structure analysis (after a representation is selected)
**Objective:** determine whether stable, continuous biochemical structure emerges — **not to force 11 axes.** Methods: PCA/sparse PCA, hierarchical clustering, consensus clustering, factor analysis, NMF, graph/community analysis. **Do not use UMAP to define axes** (UMAP = secondary visualization only).
Per candidate component/cluster inspect: spectral loadings; high/low-scoring analytes; chemical classes; functional groups; shared Raman/SERS bands; substrate/source dependence; bootstrap stability; replicate uncertainty.
**Define a candidate component only when:** stable spectral structure + coherent analyte composition + plausible biochemistry + independence from source/substrate artifacts + reproducibility under resampling + ability to project held-out analytes.
**Outcomes:** A stable continuous components; B only coarse families; C substrate-specific; D source-dominated; E no defensible ontology recoverable. **Do not preserve the existing 11 axes unless evidence supports them; the final ontology need not have 11 dimensions** (may be fewer parents, substrate-specific children, continuous factors + curated labels, or a hybrid).
**Output:** `reports/GAIRA_V5_PHASE4_EMERGENT_STRUCTURE_REPORT.md`.

---

## 9. Phase 5 — candidate ontology construction (only if Phase 4 supports structure)
Versioned ontology; each candidate axis records: `axis_id, axis_name, biochemical_interpretation, spectral_loadings, high/low-loading analytes, supporting families, modalities represented, substrates represented, known ambiguities, independence_score, bootstrap_stability, held-out_projection_stability, evidence_tier, confidence`. Prefer **hierarchical** structure (Purine → nucleotide-rich / small-metabolite-rich; Lipid → acyl / sterol / membrane; Protein/aromatic → aromatic-AA / peptide-backbone; Carbohydrate → saccharide / glycan). **Do not split a parent into children unless data support independent separation.** Distinguish: empirically-emergent axis vs curated semantic grouping vs inherited legacy axis vs unresolved mixed component. Freeze a version only after acceptance tests pass.
**Output:** `GAIRA_BIOCHEMICAL_ONTOLOGY_CANDIDATE_V1.md`.

---

## 10. Phase 6 — BSV construction (after the ontology is frozen)
BSV = projection into the frozen coordinate system. Methods: direct component scores · regularized projection · non-negative factor scores · distance-to-prototype · probabilistic membership · calibrated evidence scores. Must preserve direction, magnitude, uncertainty, acquisition-domain provenance, OOD status. **BSV is not a radar-specific construct** (radar is one visualization). Outputs per sample: BSV, uncertainty interval, axis evidence, observation-domain confidence, OOD score. Tests: replicate stability; held-out analyte projection; cross-mode matched-analyte consistency; concentration monotonicity; off-axis spillover; coordinate invariance; preprocessing sensitivity.
**Output:** `reports/GAIRA_V5_PHASE6_BSV_REPORT.md`.

---

## 11. Phase 7 — MSS integration (downstream of BSV)
Sequence: spectrum → representation → BSV → MSS analyte retrieval → evidence reconciliation → interpretation. MSS answers *which known analyte references support this biochemical-axis activation* — **it does not define axes.** MSS must be substrate-aware, excitation-aware, collision-aware, confidence-scored, and bounded by grounding coverage. Develop: top-k analyte support; family-level support; conflicting evidence; missing-reference caveats; substrate-mismatch penalties; physics-atlas ambiguity penalties.
**Output:** `reports/GAIRA_V5_PHASE7_MSS_INTEGRATION_REPORT.md`.

---

## 12. Phase 8 — controlled perturbation evaluation (representation/ontology/BSV/MSS all frozen)
Held-out tests: adenine concentration; ergothioneine dose; hypoxanthine spike; uricase depletion; ¹⁵N-uric-acid; serum metabolite spike-ins; cross-instrument adenine. Evaluate whether GAIRA recapitulates: correct axis activation + direction; concentration ordering; monotonicity; expected MSS support; low off-target; substrate-appropriate confidence; uncertainty at low concentration. Per test report: expected response, observed BSV, observed MSS, dose ordering, off-target spillover, confidence, failure mode, verdict. **Do not update parameters using test outcomes**; use failures to identify architectural deficiencies → new model version → re-evaluate with a clearly separated dev/test split. **Preserve the known uricase-depletion inconsistency; do not launder it.**
**Decision gate:** ready for biological challenge / requires ontology revision / requires observation-model revision / requires more grounding data / requires learned representation.
**Output:** `reports/GAIRA_V5_PHASE8_CONTROLLED_PERTURBATION_EVALUATION.md`.

---

## 13. Phase 9 — biological challenge datasets (only after Phase 8 acceptable)
Use EV diabetes, serum liver, SHINE, small2023 EV, others. Assess: reproducibility; cohort-level biochemical differences; domain dependence; longitudinal movement; OOD behavior; biological plausibility; stability under patient-level aggregation; separation of technical scans from biological samples. **Disease labels must not redefine the ontology.** Outputs: absolute coordinate position; cohort-relative effect profile; MSS support; domain context; technical caveats. (Carry forward V3.1's separation of absolute vs cohort-relative views and the SHINE reduced-dimensional honesty.)
**Output:** `reports/GAIRA_V5_PHASE9_BIOLOGICAL_CHALLENGE_REPORT.md`.

---

## 14. Phase 10 — physics atlas integration (ambiguity/confidence/collision/OOD first)
The atlas initially acts as an ambiguity/confidence/collision/OOD engine — **not a direct BSV multiplier.** Test: whether collision rules reduce false analyte calls; whether substrate reliability improves confidence; whether excitation mismatches are flagged; whether unsupported molecular certainty is reduced; whether known ambiguous bands get lower confidence. **Permit numerical influence only after demonstrated held-out benefit.** Seed the collision layer from the ORC-Ag exclusive-characteristic peak flags + the known shared-band map (720 purine, 1003 aromatic, 1440 lipid, 1517 carotenoid/UA).
**Output:** `reports/GAIRA_V5_PHASE10_PHYSICS_ATLAS_INFERENCE_REPORT.md`.

---

## 15. Adaptive iteration rules
- **Direct spectra → stable components:** continue with direct spectral representation.
- **Direct spectra cluster by substrate/source:** improve/stratify the observation layer before any ontology work.
- **Alignment improves matched-analyte agreement:** use aligned representation; validate on held-out analytes.
- **Alignment removes true biochemical information:** reject it; keep separate observation domains.
- **PCA components stable but chemically mixed:** use sparse PCA / factor analysis / NMF / hierarchical parents.
- **Clustering reveals families but PCA directions weak:** hybrid ontology (discrete family anchors + continuous within-family coordinates).
- **No stable structure emerges:** do not force an ontology; move to a carefully designed encoder using *only* grounding data.
- **Encoder clusters by source/substrate:** reject or retrain with nuisance-invariance objectives.
- **Controlled perturbation tests fail:** diagnose whether the failure is in preprocessing / observation model / representation / ontology / BSV / MSS / grounding coverage. **Never tune on the held-out test set.**

---

## 16. Quantitative acceptance criteria (balanced scorecard; thresholds provisional, refined after baseline)
`replicate_correlation` (target ≥0.9 within replicate family); `within_vs_between_analyte_distance` (between ≫ within); `bootstrap_component_stability` (loadings cosine ≥0.8 across ≥100 resamples); `cluster_ARI` (vs chemical family, meaningfully >0); `silhouette` (positive, family-level); `source_prediction_accuracy` (LOW — near chance = low leakage); `substrate_prediction_accuracy` (low after any alignment); `analyte_family_retrieval_topk`; `held_out_analyte_projection_stability`; `cross_mode_matched_analyte_distance` (shrinks after defensible alignment); `dose_response_spearman` (per perturbation, sign + magnitude); `off_target_BSV_spillover` (low); `OOD_detection` (Raman scored on SERS scale flagged); `confidence_calibration` (reliability curve). **No single metric decides; use the full scorecard.**

---

## 17. Required visual development artifacts (every phase; figures need matching metrics)
Preprocessing overlays; replicate-consistency plots; PCA score/loading plots; dendrograms; analyte×feature heatmaps; substrate/source leakage plots; matched-analyte cross-mode comparisons; component-stability plots; candidate-ontology maps; BSV projections; perturbation dose trajectories; MSS support panels; OOD/confidence diagrams. Save under versioned `results/gaira_v5/<milestone>/`. **No figure is evidence without a corresponding quantitative metric.**

---

## 18. Data splitting & leakage control
Partitions: grounding development set; grounding held-out analyte set; matched-substrate development pairs; matched-substrate held-out pairs; controlled-perturbation development set; controlled-perturbation **final** evaluation set; biological challenge sets. Where the corpus is too small for fixed splits: nested CV, leave-analyte-out, leave-source-out, leave-substrate-out, bootstrap stability. **Never allow spectra from the same analyte-condition-replicate family on both sides of a split** (critical given RamanBioLib duplicates and metabolite-63 pre-averaging).

---

## 19. Milestones (each: objective · inputs · tasks · analyses · visualizations · acceptance · failure · decision branches · deliverables · dependencies)
- **V5.0 — Canonical registries & data contracts** (Phase 0). *Accept:* every included spectrum has full provenance + domain metadata. *Deliverable:* `src/gaira/data/` registries + schema. *Dep:* V4 registries.
- **V5.1 — Preprocessing & comparability** (Phase 1). *Accept:* replicates consistent, analyte bands preserved, shared wavenumber space, measurable source effects. *Fail:* bands destroyed / irreducible source dominance → stratify. *Deliverable:* Phase-1 report + pipeline decision.
- **V5.2 — Observation-layer feasibility** (Phase 2). *Accept:* a chosen level (L0–L3) with held-out matched-analyte improvement, or decision D (more data). *Deliverable:* Phase-2 report.
- **V5.3 — Canonical representation selection** (Phase 3). *Accept:* a representation passing leakage + stability + interpretability. *Deliverable:* Phase-3 report + representation v1.
- **V5.4 — Emergent structure analysis** (Phase 4). *Accept:* ≥1 reproducible, chemically-coherent, artifact-independent component. *Deliverable:* Phase-4 report.
- **V5.5 — Candidate ontology** (Phase 5). *Accept:* versioned ontology passing independence/stability/projection tests. *Deliverable:* ontology candidate v1.
- **V5.6 — Frozen BSV** (Phase 6). *Accept:* invariance + replicate + monotonicity tests. *Deliverable:* BSV report + frozen artifact.
- **V5.7 — MSS integration** (Phase 7). *Accept:* MSS downstream, substrate/collision-aware, coverage-bounded. *Deliverable:* MSS report.
- **V5.8 — Controlled perturbation evaluation** (Phase 8). *Accept:* correct direction/ordering on the held-out suite; failures diagnosed not tuned. *Deliverable:* evaluation report + gate decision.
- **V5.9 — Biological challenge evaluation** (Phase 9). *Accept:* plausible cohort behavior, OOD honesty, technical/biological separation. *Deliverable:* challenge report.
- **V5.10 — Canonical demo integration** (presentation over the one engine; retire duplicate demo scoring). *Accept:* demo calls `src/gaira`; no duplicate engine. *Deliverable:* integrated demo + regression parity.

---

## 20. Immediate execution recommendation (first sprint = V5.0 + V5.1 only)
1. Reconcile all grounding registries (extend V4 `data_audit/` CSVs into `src/gaira/data/`).
2. Construct **one canonical spectrum loader** (reads raw per source with full metadata; no scoring).
3. Define acquisition-domain metadata (modality/substrate/geometry/excitation/instrument).
4. Implement deterministic **preprocessing candidates** (baseline/smoothing/normalization variants) — code only, no ontology.
5. Run a **preprocessing comparability benchmark** across Raman / Ag-SERS / ORC-Ag.
6. Generate analyte/source/substrate **QC visualizations**.
7. Identify **matched analytes** across Raman and Ag-SERS.
8. Determine whether **joint analysis is scientifically defensible**.
9. Produce `reports/GAIRA_V5_PHASE1_PREPROCESSING_AND_COMPARABILITY_REPORT.md`.
10. **STOP before observation-model fitting.** Do not begin PCA/ontology construction until this sprint is complete.

The Phase-1 result determines whether the next step is a shared observation model, modality-stratified analysis, or a data-acquisition gap (notably: **acquire Au-SERS references**) that must be resolved before PCA/clustering.

---

## 21. Major decision gates (summary)
- **G0 (Phase 0):** no representation analysis until provenance/metadata complete.
- **G1 (Phase 1):** one global pipeline vs modality-specific; proceed only if chemistry preserved & shared space defensible.
- **G2 (Phase 2):** cross-mode correction A/B/C/D; alignment only if held-out matched analytes improve.
- **G3 (Phase 3):** canonical representation; reject if it clusters by source/substrate/excitation.
- **G4 (Phase 4):** define components only if reproducible, coherent, artifact-independent; do not force 11 axes.
- **G5 (Phase 5):** freeze ontology only after acceptance tests.
- **G6 (Phase 6):** freeze BSV only after invariance/monotonicity/replicate tests.
- **G8 (Phase 8):** biological evaluation only after the perturbation suite passes.

**Conditions that trigger an encoder (Phase 3-C / adaptive rules):** direct full-spectrum AND chemically-constrained features both fail to reveal stable, artifact-independent biochemical structure — only then, an encoder trained on **grounding data only** (no disease labels, no perturbation sets), rejected/retrained if it clusters by source/substrate.
**Conditions before defining BSV axes:** a canonical representation selected (Phase 3) + reproducible emergent structure (Phase 4) + a frozen candidate ontology passing independence/stability/held-out-projection tests (Phase 5).
**Conditions before testing biological cohorts:** representation + ontology + BSV + MSS frozen AND the controlled perturbation suite (Phase 8) shows correct axis direction/ordering with low off-target and honest confidence.

---

## 22. Canonical conclusion
GAIRA V5 will not assume that its biochemical axes already exist. It will rebuild the coordinate system from a rigorously curated molecular grounding corpus. All spectra will first be made internally consistent and acquisition-aware. Matched analytes across Raman and SERS will then be used to determine whether a shared observation space can be constructed without erasing chemistry. Only after a canonical representation is selected will PCA, sparse factorization, hierarchical clustering, and related methods be used to test whether stable biochemical structure emerges. If direct spectra are insufficient, a learned embedding will be introduced only as a justified fallback. BSV axes will be frozen only after spectral, chemical, cross-source, and controlled-perturbation evidence support them. MSS will remain a downstream analyte-support layer, while the physics atlas will primarily govern ambiguity, confidence, and out-of-distribution reasoning. The demo will ultimately become a presentation layer over one canonical production inference engine.

> **Key principle:** Do not decide the representation, ontology, axis count, or need for an encoder in advance. Build the pipeline so each decision is earned from the preceding evidence. The first implementation stops after preprocessing + comparability.
