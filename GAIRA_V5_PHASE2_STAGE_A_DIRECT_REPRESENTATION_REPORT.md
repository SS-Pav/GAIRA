# GAIRA V5 — Phase 2 Stage A: Direct Spectral Representation Report

**Date:** 2026-07-18 · Branch `gaira-v5-rebuild-plan` · Governs: `GAIRA_V5_REBUILD_PLAN.md` (Phase 2, Stage A) · Hypotheses: `GAIRA_V5_HYPOTHESIS_REGISTER.md` · Notebook: `results/v5_rebuild/phase2_stage_a/`.

> **DECISION (Stage A, §19): Outcome B — a modality-stratified representation is the defensible architecture. Direct 785 nm spectra do NOT support a single shared Raman/Ag-SERS coordinate system, but they DO carry real, recoverable chemistry within each modality plus a weak, statistically-significant residual cross-modal signal. Represent each modality separately and align at the analyte/ontology level; test Stage B chemical features to see whether the residual cross-modal signal can be strengthened before revisiting a shared space.**
>
> **STOP after Stage A (§24). Stage B / Stage C / observation model / ontology / BSV / MSS / perturbation evaluation are NOT started.**

This phase tested — it did not assume — whether direct (unlearned) spectral representations preserve a shared biochemical structure across Raman and Ag-SERS. H1 (shared space) and H4-preliminary (modality-stratified) were evaluated without preferring either.

---

## 0. What was built (canonical, reusable)
Code under `src/gaira/representation/` (reuses `src/gaira/data` and `src/gaira/preprocessing`; no duplication):
`datasets.py` (audited 785 nm input), `direct.py` (direct-representation entry point), `centroids.py` (analyte×modality×source centroids, no cross-modality averaging), `metrics.py`, `retrieval.py` (cross-modal retrieval + permutation nulls), `pca.py` (PCA + bootstrap loading stability by analyte), `factorization.py` (NMF/sparsePCA/FA with non-negativity guard), `clustering.py` (hierarchical + ARI vs nuisance), `leakage.py` (grouped-CV modality/source probes), `stability.py` (consensus clustering).
Runners under `results/v5_rebuild/phase2_stage_a/code/`: `run_input_audit.py`, `run_stage_a.py`. Tests: `tests/test_v5_representation.py` (16 passing).

---

## 1. §4 Input audit — corpus is internally consistent (with one correction)
All 9 invariant checks **pass** (`tables/phase2_input_audit_summary.json`):

| Check | Result |
| --- | --- |
| all entered spectra are 785 nm | PASS |
| all entered pass admission gate | PASS |
| all entered role = grounding | PASS |
| no controlled-perturbation spectra entered training | PASS |
| adenine concentration series excluded | PASS |
| no duplicate spectrum IDs | PASS |
| no raw/processed double-count | PASS |
| single common wavenumber grid (520–1750 @ 2 cm⁻¹, 616 pts) | PASS |
| entered count matches manifest | PASS |

**Correction applied (data-role leakage prevented, §3).** The 6 adenine bAgNPs Ag-SERS spectra were confirmed to be a **controlled concentration series** (10 pg → 10 µg/mL) — controlled-perturbation *evaluation* data, not independent grounding — and were **excluded from representation fitting**. Adenine remains a matched grounding analyte via Gobbato Raman + Ag-SERS, so the matched-analyte count is unchanged.

**Corrected corpus (immutable manifest `tables/phase2_input_manifest.csv`):** 485→**479 spectra** (214 Raman + 265 Ag-SERS; 271→265 Ag-SERS), **87 analytes**, **51 matched** (58.6%) — unchanged. Six analytes are flagged **non-small-molecule** (dna, rna, albumin, glycogen, coenzyme a, acetyl-coa): retained as grounding but not treated as "one peak = one molecule" references. Phase 1.5 report and hypothesis register were updated with the correction.

---

## 2. The twelve Stage-A questions

**Q1. Do direct spectra contain stable biochemical structure at all?**
Yes — within a modality. Direct Raman centroids recover chemistry: hierarchical-clustering **ARI vs analyte = 0.49–0.52** (across all three preprocessings), cophenetic corr 0.77–0.81, silhouette ~0.39. PC1 loadings are bootstrap-stable by analyte (stability 0.72–0.93). So unlearned spectra are not noise — chemistry is present and reproducible.

**Q2. Is that structure chemistry or acquisition nuisance?**
Preprocessing-dependent, and this is the central result. In the **joint** (Raman+SERS) space:
- **L2 (A1):** nuisance dominates — clustering ARI(source)=0.26, ARI(modality)=0.12 ≫ ARI(analyte)=0.02.
- **SNV (A2):** chemistry emerges — ARI(analyte)=0.15 > ARI(modality)=0.02, ARI(source)=0.02.
- **1st-derivative (A3):** similar to SNV (ARI analyte 0.15 > nuisance 0.02) but weakest cross-modal signal.
SNV/derivative suppress the multiplicative intensity-scale differences that separate powder-Raman from colloid-SERS; L2 does not.

**Q3. Can Raman and Ag-SERS share one representation (H1)?**
**No, not from direct spectra.** Even under the most favorable preprocessing (SNV), a modality classifier reads modality off the joint representation at **balanced accuracy 0.83** (chance 0.50; L2/derivative 0.93–0.94) — well above the 0.75 shared-space threshold. Modality is too separable for a single defensible coordinate system.

**Q4. Is the matched-analyte cross-modal identity preserved (H1d)?**
**Weakly, but real.** For the 51 matched analytes, an analyte's Raman centroid retrieves its own Ag-SERS centroid better than chance: matched cosine > unmatched (permutation **p ≤ 0.02** for L2/SNV; A3 n.s. at top-1), MRR permutation p ≤ 0.014 for all three. But absolute retrieval is weak: **top-1 = 0.08 (L2) / 0.16 (SNV) / 0.04 (deriv)** (chance ≈ 1/51 = 0.02), top-5 ≤ 0.31, reciprocal-NN ≤ 0.18. There is shared chemical information, but it is far too weak to identify an analyte across modalities from raw spectra.

**Q5. Do matched pairs share peak positions?**
**No.** Mean band-overlap for matched Raman↔SERS pairs (~0.40–0.62) is **not above** mismatched pairs (~0.42–0.62). SERS re-weights and shifts bands relative to Raman (surface selection rules, chemisorption), so the cross-modal signal is not carried by co-located peaks — consistent with why direct spectral overlap is weak.

**Q6. Which preprocessing best preserves structure (§6)?**
**SNV (A2).** It maximizes joint chemistry-over-nuisance and minimizes modality leakage; it is the only preprocessing under which the joint space clusters by chemistry rather than acquisition. L2 is dominated by intensity-scale nuisance; the first-derivative representation matches SNV for within/joint chemistry but has the weakest cross-modal retrieval. Conclusions are **robust in direction** across all three (matched > unmatched cosine in every case), but their strength is preprocessing-sensitive.

**Q7. PCA — how much structure, how stable (§9)?**
Joint PCA is diffuse: cumulative variance over 6 PCs is only 0.33–0.39, indicating no small set of dominant shared axes. Loadings are stable for PC1 (0.72–0.93 bootstrap cosine by analyte) and degrade by PC3–PC6 (0.29–0.58) — the leading axis is reproducible; deeper structure is not. Under L2 the leading axes align with modality, not chemistry (see `figures/joint_pca_by_modality_A1.png` vs the SNV contrast `..._A2.png`).

**Q8. NMF / sparse PCA / factor analysis (§10)?**
NMF was applied only to the non-negative L2 representation (non-negativity guard enforced; correctly skipped for signed SNV/derivative). Sparse PCA loadings are dense-ish (no strong sparse chemical parts emerged from direct spectra). No factorization produced clean, interpretable shared chemical components — consistent with the PCA and clustering evidence that a shared low-rank chemical basis is not present in direct spectra.

**Q9. Modality/source leakage under grouped CV (§12–13)?**
Grouped by analyte (no analyte or replicate crosses the split): modality is highly predictable (bal-acc 0.83–0.94 ≫ 0.50 chance). Source leakage within Raman is negligible (ARI(source)≈0). **H6 remains partially rejected:** nuisance (modality) structure is present and must be controlled before any ontology work — it cannot be assumed away.

**Q10. Two analysis levels agree (§5)?**
Yes. Centroid-level (primary) and spectrum-level tell the same story. **Caveat:** at centroid level the Ag-SERS side has exactly one centroid per analyte (single source, Gobbato), so SERS-only *clustering-vs-analyte* is not informative (degenerate ARI=0.00 — documented, not a finding). SERS structure is therefore assessed via cross-modal retrieval and joint analysis, not SERS-only clustering.

**Q11. Is the grounding sufficient to have decided this (rule out Outcome D)?**
Yes. 51 matched analytes with replication (Raman ~3, Ag-SERS ~5) and 479 spectra were enough to obtain statistically-significant cross-modal tests with permutation nulls. The limitation is not sample size but the intrinsic weakness of direct-spectral cross-modal overlap.

**Q12. What is the defensible architecture decision (§19)?**
**Outcome B — modality-stratified.** See below.

---

## 3. Scorecard (§14, weights fixed before results)
`tables/stage_a_scorecard.json` · the shared-space question is judged under its **most favorable** preprocessing (SNV) so an arbitrary preprocessing choice cannot decide the architecture; all three are reported.

| Dimension | Weight | Score | Basis |
| --- | --- | --- | --- |
| cross-modal identity | 0.35 | 0.31 | SNV top-1 0.16 (p<0.001) — significant but weak |
| chemistry over nuisance | 0.25 | 0.93 | SNV joint ARI analyte 0.15 > nuisance 0.02 |
| low modality leakage | 0.20 | 0.34 | SNV modality bal-acc 0.83 (still ≫ chance) |
| component stability | 0.10 | 0.55 | joint PC1–3 bootstrap stability |
| preprocessing robustness | 0.10 | 1.00 | matched > unmatched in all three; top-1 > chance in all three |
| **weighted total** | | **0.564** | |

The score is well above the "direct inadequate" floor (0.45): direct spectra are **not** inadequate (chemistry is recovered) — but the two shared-space gates both fail (top-1 0.16 < 0.30; modality bal-acc 0.83 > 0.75).

---

## 4. Decision (§19) — exactly one outcome

**Outcome B: modality-stratified representation is defensible.**

- **Outcome A (shared) rejected:** even under the best preprocessing, modality is too separable (bal-acc 0.83–0.94) and cross-modal retrieval too weak (top-1 ≤ 0.16) for a single shared coordinate system.
- **Outcome C (direct inadequate) rejected:** direct spectra *do* recover chemistry (Raman-only ARI 0.49–0.52; SNV joint chemistry > nuisance). Direct representation is adequate *within* a modality.
- **Outcome D (grounding insufficient) rejected:** 51 matched analytes gave significant, well-powered tests.
- **Outcome B selected:** represent Raman and Ag-SERS separately; align at the analyte/ontology level, not in raw spectral space. A weak (p ≤ 0.02) residual cross-modal signal exists, so a shared space is not impossible — it is simply not achievable from direct spectra. **Stage B (chemistry-aware, ideally modality-invariant features) is the recommended next step** to test whether that residual signal can be strengthened before a shared space is reconsidered.

This directly supports **H4-preliminary** over **H1** for the direct-representation regime, and is consistent with the governing principle: *do not force Raman and Ag-SERS into one coordinate system simply because they describe the same molecules.*

---

## 5. Hypothesis outcomes (see register for full text)
- **H1** (shared observation space): **not supported from direct spectra** — modality dominates; cross-modal retrieval weak. Still open for Stage B/observation-model regimes; do not assume.
- **H1a** (enough matched analytes): **Supported** (51, unaffected by the adenine correction).
- **H1d** (direct reps preserve cross-modal identity): **Partially supported / weak** — statistically significant (p ≤ 0.02) but low absolute strength (top-1 ≤ 0.16); peak positions do not align.
- **H4-preliminary** (modality-stratified preferable): **Supported for the direct-representation stage.**
- **H6** (structure = chemistry, not nuisance): **Partially rejected** — modality leakage bal-acc 0.83–0.94; must be controlled.
- **H5** (perturbation held out): **maintained** — adenine concentration series verified excluded from fitting.

---

## 6. Limitations (honest scope)
- **Single-source Ag-SERS** (Gobbato colloid): no cross-source SERS generalization can be established; SERS-only clustering is degenerate at centroid level (1 centroid/analyte). External generalization of SERS structure is **untested by construction**.
- Same-source matched pairs are strongest within one instrument — cross-modal results may be optimistic relative to a multi-instrument corpus.
- Zero Au-SERS grounding; no multi-excitation for these analytes (785-only by V5 design).
- Direct representation only — Stage B (chemical features) and Stage C (learned embeddings) not evaluated here.

---

## 7. Artifacts
- Manifest: `results/v5_rebuild/phase2_stage_a/tables/phase2_input_manifest.csv` (immutable input)
- Audit: `tables/phase2_input_audit_summary.json`
- Results: `tables/stage_a_results.json` (per-preprocessing, all blocks)
- Scorecard: `tables/stage_a_scorecard.{json,csv}` · Decision: `tables/stage_a_decision.json`
- Figures: `figures/joint_pca_by_modality_A1.png` (L2, modality dominates), `..._A2.png` (SNV contrast), `joint_pca_scree_A1.png`, `cross_modal_retrieval_vs_null_A1.png`
- Notebook summary: `results/v5_rebuild/phase2_stage_a/report.md`
- Tests: `tests/test_v5_representation.py` (16 passing)
