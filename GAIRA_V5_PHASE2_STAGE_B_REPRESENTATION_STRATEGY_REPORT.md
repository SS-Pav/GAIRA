# GAIRA V5 — Phase 2 Stage B: Biochemical Representation Strategy Benchmark

**Date:** 2026-07-19 · Branch `gaira-v5-rebuild-plan` · Governs: `GAIRA_V5_REBUILD_PLAN.md` (Phase 2 Stage B) · Context: `GAIRA_V5_ARCHITECTURE_AND_SCIENTIFIC_CONTEXT.md` · Hypotheses: `GAIRA_V5_HYPOTHESIS_REGISTER.md` · Notebook: `results/v5_rebuild/phase2_stage_b/`.

> **DECISION (§16): Outcome B4 — modality-stratified representations retained. No representation (interpretable or encoder) materially beats direct spectra on held-out cross-modal biochemical retrieval, so no shared biochemical representation is supported by the current corpus. A companion, equally important finding: small encoders do not just fail to help — they actively underperform direct spectra AND collapse (cross-analyte duplicate embedding fraction 0.96–1.00), so the encoder feasibility question is answered negatively at this corpus scale (H1c/H1d rejected; H7 confirmed high-risk). Represent Raman and Ag-SERS separately and align at the ontology level; before revisiting an encoder, acquire the data specified in §11.**
>
> **STOP after Stage B. Ontology (Stage D), BSV, MSS, DART, perturbation, biological cohorts, production, model scaling, and pretraining are NOT started.**

This was a benchmark, not an assumption. Encoders were given a fair, primary-hypothesis role (dual encoder E2) and evaluated identically to interpretable and direct representations under one leakage-safe framework. The result is a defensible negative for a shared representation and for encoders at this scale.

---

## 0. What was built (canonical, reusable)
`src/gaira/evidence/` (reuses `data`, `preprocessing`, `representation` — no duplication):
`datasets.py` (Stage B dataset + card), `families.py` (curated evaluation-only families), `splits.py` (leakage-safe A/B/C/D), `augmentations.py` (bounded + validity audit), `base.py`; interpretable `regions.py` (I1), `wavelets.py` (I2), `dictionary.py` (I3), `basis.py` (I4); encoder `encoders.py`, `losses.py`, `training.py`; `projection.py`, `hybrid.py` (E4), `evaluation.py`, `interpretability.py`, `uncertainty.py`, `serialization.py`.
Runners: `results/v5_rebuild/phase2_stage_b/code/{run_stage_b.py, stage_b_decide.py}`. Tests: `tests/test_v5_evidence.py` (26 passing). Benchmark runtime 2516 s.

---

## 1. Corpus, splits, and audit (Parts 4–5, 10)
- **Frozen Stage B corpus** (from the Phase-2 manifest): 479 spectra (214 Raman + 265 Ag-SERS), 87 analytes, **51 matched**; adenine concentration series excluded (perturbation); provenance/role preserved. Card: `tables/stage_b_dataset_card.json`.
- **Leakage-safe splits** (predeclared, all integrity checks PASS — `tables/stage_b_split_leakage_checks.json`): **A** held-out analytes, **B** held-out matched pairs (both modalities test-only), **C** replicate-group holdout, **D** source sensitivity — **explicitly infeasible for Ag-SERS (single source)**; leave-source-out built for Raman only.
- **Augmentation validity audit** (`tables/augmentation_audit.json`): bounded augmentations preserve **94.3%** of major bands and invent **0.8%** — identity is not altered.

---

## 2. Headline benchmark — held-out matched-analyte cross-modal retrieval (Split B, primary metric)
Chance top-1 ≈ 0.098 (per-fold ~10 held-out matched analytes). Baseline = **direct_SNV** (Stage A best preprocessing), evaluated held-out (apples-to-apples).

| Representation | branch | top-1 | MRR | MRR 95% CI | perm p | modality leak | cross-analyte dup | within-mod Raman ARI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **I1 adaptive regions** | interpretable | **0.294** | **0.460** | [0.36, 0.56] | 0.001 | 0.79 | 0.00 | 0.94 |
| direct_SNV (baseline) | direct | 0.275 | 0.452 | [0.35, 0.54] | 0.001 | 0.86 | 0.00 | 0.94 |
| direct_L2 | direct | 0.255 | 0.449 | [0.36, 0.54] | 0.001 | 0.92 | 0.00 | 0.94 |
| E4 hybrid (E1+I1) | hybrid | 0.255 | 0.438 | [0.35, 0.53] | 0.002 | — | — | 0.69 |
| I2 multiscale | interpretable | 0.235 | 0.432 | [0.34, 0.52] | 0.005 | 0.86 | 0.00 | — |
| I4 NMF basis | interpretable | 0.235 | 0.414 | [0.32, 0.51] | 0.003 | 0.92 | 0.01 | — |
| direct_deriv | direct | 0.235 | 0.406 | [0.31, 0.50] | 0.003 | 0.90 | 0.00 | — |
| I3 sparse dictionary | interpretable | 0.176 | 0.375 | [0.29, 0.45] | 0.063 | 0.86 | 0.00 | — |
| **E1 shared encoder** | encoder | 0.176 | 0.338 | [0.25, 0.43] | 0.018 | 0.91 | **1.00** | 0.53 |
| **E3 modality-specific** | encoder | 0.137 | 0.318 | [0.24, 0.41] | 0.119 | 1.00 | 0.97 | 0.57 |
| **E2 dual (primary)** | encoder | 0.137 | 0.312 | [0.24, 0.40] | 0.069 | 1.00 | 0.96 | 0.54 |
| E2 dual + triplet | encoder | 0.118 | 0.309 | [0.24, 0.39] | 0.183 | 1.00 | 0.99 | — |
| E2 dual + VICReg | encoder | 0.078 | 0.265 | [0.20, 0.33] | 0.829 | 0.93 | 0.43 | — |

**Reading:** the entire top half is direct + interpretable; the entire bottom half is encoders. I1 regions edges out direct_SNV but its CI overlaps the baseline almost entirely — **not a material or CI-separated improvement**. Every encoder is *below* direct, and the dual/modality-specific encoders show **near-total embedding collapse** (cross-analyte duplicate fraction 0.96–1.00) with modality leakage driven to 1.00 (separate encoders make modality trivially separable). VICReg reduces collapse (dup 0.43) only by destroying retrieval (top-1 0.078, p=0.83 ≈ null).

---

## 3. The fifteen Stage-B questions (Part 21)

**Q1. Did interpretable representations improve over direct spectra?** Marginally at best and not decisively. I1 adaptive regions gave the single highest held-out MRR (0.460 vs 0.452) but with fully overlapping bootstrap CIs — statistically indistinguishable from direct_SNV. I2/I3/I4 were ≤ direct. **No interpretable representation materially beats direct spectra.**

**Q2. Did encoder embeddings improve over direct spectra?** **No — they were worse on every held-out axis.** Held-out cross-modal top-1 0.08–0.18 (vs direct 0.28); held-out-analyte family retrieval 0.48–0.50 (vs direct 0.74); within-modality Raman ARI 0.53–0.77 (vs direct 0.94).

**Q3. Which encoder architecture performed best?** E1 (shared) had the best encoder MRR (0.338) — still below direct. The dual/modality-specific encoders were worse and collapsed.

**Q4. Did a dual encoder outperform a shared encoder?** **No.** E2 dual (0.312) < E1 shared (0.338). Contrary to the primary encoder hypothesis (H1e rejected for this corpus). Separate encoders make the two modalities *more* separable, not more aligned.

**Q5. Did any encoder reduce modality leakage without losing chemistry?** **No.** Encoders either raised modality leakage to 1.00 while collapsing (E1/E2/E3) or lowered it (VICReg 0.93) only by erasing analyte identity (retrieval → null). H1d rejected at this scale.

**Q6. Did performance hold for unseen matched analytes?** This *is* Split B (held-out matched analytes). Direct/interpretable retain a real, permutation-significant signal (p ≤ 0.005); encoders do not (E2 p=0.069, E3 p=0.119 — not significant).

**Q7. Did performance hold across seeds and splits?** Encoder MRR is stable in the *mean* across 3 seeds (E1 0.308±0.024, E2 0.292±0.015) — i.e. reproducibly mediocre, not unstably lucky. Direct/interpretable are deterministic.

**Q8. Any candidate showing source shortcuts?** In-sample signal-shortcut correlations are modest; but **source generalization is fundamentally unresolvable** because Ag-SERS is single-source (Split D infeasible for SERS). We therefore make no observation-domain-invariance claim.

**Q9. Which representation is most interpretable?** Interpretable I1/I2/I4 map directly to wavenumber space. Encoder attribution is *stable* across replicates (occlusion attribution replicate-correlation 0.95) but the embedding barely encodes interpretable evidence linearly (sparse probe embedding→regions R²=0.01) — the encoder's evidence path is weak.

**Q10. Which representation is most compatible with ontology construction?** Modality-stratified **direct spectra** (Raman especially: held-out within-modality ARI 0.94) and, secondarily, **I1 adaptive regions** (auditable, ARI 0.94, comparable cross-modal). A shared learned space is not.

**Q11. Which is most compatible with future DART trajectories?** None demonstrated (out of scope). Structurally, an interpretable region/basis representation is the more trajectory-friendly substrate (fixed, auditable axes) than a collapse-prone encoder; this is a hypothesis (H8), not a result.

**Q12. Is the corpus sufficient for encoder development?** **No.** Encoders collapse and underperform trivial baselines. H7 (corpus sufficient to train a generalizing encoder) is **rejected as high-risk confirmed**. 51 matched analytes, largely one instrument, single-source Ag-SERS, is below encoder scale.

**Q13. What representation should be frozen?** **None as a shared representation.** For downstream work, keep modality-stratified direct spectra (SNV) as the working representation; I1 adaptive regions is retained as an auditable interpretable companion. No encoder is frozen.

**Q14. Which Outcome B1–B5?** **B4** (modality-stratified retained). With an explicit B5-flavored caveat: the *encoder-specific* conclusion is corpus-insufficiency — encoders should not be revisited until the §11 data gaps are addressed.

**Q15. Next authorized phase?** Not Stage D (ontology) yet — B4 froze no shared representation. The authorized next step is **Stage C re-scoped as targeted grounding-data acquisition + interpretable refinement** (multi-source Ag-SERS, Au-SERS, external matched analytes), *then* re-run this benchmark. Ontology (Stage D) remains gated.

---

## 4. Diagnostics (Parts 11–13, 18–19)
- **Collapse (§11):** encoders E1/E2/E3 collapse (cross-analyte duplicate fraction 0.96–1.00; effective rank low). This is the decisive disqualifier — a low modality accuracy would have been meaningless here because the embedding erases analyte identity too. `figures/encoder_collapse_diagnostics.png`.
- **Within-modality retention (Split C, §12):** direct_SNV Raman ARI 0.94; I1 regions 0.94 (retained); E2 0.77 (degraded). Ag-SERS ARI low for all (0.21–0.55) — single-source, one-centroid-per-analyte limits SERS-only clustering (documented, not a finding).
- **Family neighborhood (held-out analytes, Split A, §13):** direct 0.71–0.74 purity (chance ≈ 0.25); interpretable 0.59–0.72; encoders 0.48–0.50. Even for chemical-family generalization to unseen analytes, direct spectra win and encoders hurt. Non-small-molecule refs (dna/rna/albumin/glycogen/CoA/acetyl-CoA) excluded from this analysis.
- **Interpretability (§18):** occlusion + input-gradient attribution implemented; occlusion attribution is replicate-stable (corr 0.95); sparse linear probe shows the encoder embedding does not linearly carry interpretable region evidence (R²=0.01).
- **Uncertainty (§19):** distance-to-support, neighbor agreement, OOD, and cross-modal agreement implemented; matched-reference cross-modal agreement mean 0.47 (not calibrated probabilities — reliability signals only).

---

## 5. Model-selection gates (Part 15) and decision (Part 16)
Pre-declared Pareto gates (`tables/stage_b_scorecard.json`): material cross-modal improvement over direct held-out (Δtop-1 ≥ 0.03 **and** MRR-CI above baseline), no collapse, within-modality retention, no source shortcut, seed stability, auditable evidence path. **No candidate passes the cross-modal gate** (none has an MRR CI above the direct baseline). Encoders additionally fail collapse and within-retention gates.

**Outcome B4 selected.** Not B1 (interpretable does not *materially* beat direct — CI overlap). Not B2/B3 (encoders/hybrid are worse and collapse). Not pure B5 (direct/interpretable give a stable, reproducible answer, so we *can* conclude — the corpus is sufficient to conclude "no shared representation and no encoder value", even though it is insufficient to *build* an encoder).

---

## 6. Hypothesis outcomes (see register)
H1 (canonical evidence representation exists): **not supported by current corpus** (no candidate beats modality-stratified direct). H1b: **partially supported** (weak significant cross-modal signal persists). **H1c (encoder improves retrieval): rejected** (encoders worse). **H1d (encoder reduces leakage while keeping identity): rejected** (collapse). **H1e (dual > shared): rejected** (dual worse). **H2 (interpretable recovers evidence space without encoders): not supported** (interpretable ≈ direct, not a new shared space). H2a (adaptive regions > isolated peaks): **weakly supported** (I1 best interpretable, beats I3). **H3 (hybrid best): rejected** (hybrid < direct). **H4 (modality-stratified): supported (reaffirmed).** **H6 (chemistry ≫ nuisance required): still not met by any shared candidate.** **H7 (corpus sufficient for a generalizing encoder): rejected — high risk confirmed.** H8 (DART trajectories): untested (future).

---

## 7. Limitations & source-generalization caveat
Single-source Ag-SERS → **Split D infeasible for SERS; no observation-domain-invariance claim is made**; all cross-modal results are *within the present 785 nm matched corpus*. 51 matched analytes / largely one instrument is a feasibility, not a training, scale. Encoders were deliberately small (≈11k params) and unaugmented-pretrained by design; this study does not claim a larger/pretrained encoder would also fail — only that at this corpus scale a small encoder collapses and adds no value.

---

## 8. Artifacts
Tables: `stage_b_dataset_card.json`, `stage_b_split_leakage_checks.json`, `augmentation_audit.json`, `stage_b_results.json`, `stage_b_scorecard.json`, `stage_b_decision.json`. Configs: `configs/stage_b_splits.json`. Models: serialized interpretable (.npz/.json) + encoder (.pt/.json). Figures (9): held-out MRR, Pareto leakage-vs-chemistry, primary-encoder PCA (modality/family/source), training curves, collapse diagnostics, dictionary atoms, NMF basis, adaptive regions, augmentation examples. Notebook: `results/v5_rebuild/phase2_stage_b/report.md`. Tests: `tests/test_v5_evidence.py`.
