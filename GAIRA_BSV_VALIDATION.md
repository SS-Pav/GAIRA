# GAIRA Biochemical State Vector — Validation

A rigorous scientific validation of the **Biochemical State Vector (BSV v2)** produced by the frozen V6 Converged Reasoning Engine. The engine, Raman Reference Atlas, preprocessing, NMF, ontology, component registry, theme weights and BSV equations are **frozen and only measured here** (atlas fingerprint `09ed804a40836f4a05a91ba10900cded`, verified on every load). This study characterizes the BSV the way one would characterize a new imaging modality or assay: strengths, weaknesses, uncertainty — never optimizing, never modifying. Branch `gaira-v5-rebuild-plan`; nothing pushed.

**The question is not "does the software run?" but "is the BSV a stable, meaningful biochemical coordinate system?"**

---

## Reproduce

```bash
cd results/v5_rebuild/bsv_validation/code
python run_bsv_validation.py    # Parts 2-13, drives calibration data through the real V6 engine
python make_figures.py          # 6 figures + PDF
```

---

## Executive summary

| dimension | finding |
| --- | --- |
| **Implementation (Part 1)** | verified: frozen fingerprints intact, weights sum to 1, pipeline genuinely input-sensitive (suppressing c3 drops the purine theme). No hard-coded outputs. |
| **Monotonicity (Part 3)** | **yes** — every dose→target-theme relationship is monotonic and saturating (all permutation p = 0.002); target Spearman 0.34–0.97. |
| **Best calibrations** | ergothioneine→sulfur ρ **0.97**; colloidal adenine→purine ρ **0.91**. |
| **Specificity (Part 4)** | target theme moves most, but **leakage is high** (off-target/target 0.65–0.82) — substantial cross-talk. |
| **Theme coupling (Part 11)** | mean \|r\| 0.24, max 0.65 — and the coupling **encodes real biology** (protein↔sulfur, protein↔heme, aromatic↔lipid). |
| **Effective dimensionality (Part 12)** | **~4 of 11** themes (PC1 = 50% of variance) — the BSV is genuinely lower-dimensional than its nominal axis count. |
| **Inter-analyte geometry (Part 7)** | **purines cluster**; glucose isomers merge; geometry recovers chemistry. |
| **Confidence (Part 9)** | tracks **domain** OOD (ρ −0.57) but **not analyte-level recoverability** (strong ≈ weak Ag adsorbers) — a real gap. |
| **Reproducibility (Part 8)** | **substrate-dependent**: ICC 0.14 (paper) → 0.83 (colloid). |

**Verdict:** the BSV is a *stable and biochemically meaningful* coordinate system for well-adsorbed analytes on colloidal substrates, but it is **lower-dimensional and more coupled than its 11 nominal axes suggest**, and its confidence signal reports *spectrum quality*, not *analyte identifiability*. These are characterization results, not failures to fix.

---

## Part 1 — implementation verified

The frozen atlas fingerprint matches on every module load; the 24×13 theme-weight matrix rows sum to 1; the pipeline is genuinely input-sensitive (a 10× suppression of component c3 drops the purine theme 0.375→0.243). Every calibration dataset is run through the *actual* `GAIRAEngine.infer(...)`; no theme scores are computed by hand and no output is hard-coded. Versioning, OOD, reference normalization and confidence all execute in the pipeline.

---

## Part 2–3 — calibration and monotonicity

Every dose→target-theme relationship is **monotonic and best described by a saturating (Langmuir) model**, consistent with finite-site colloid adsorption. Permutation p = 0.002 in all seven series.

| series | target theme | Spearman | Pearson | Kendall | best model | effect size | OOD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adenine cAu@785 | purine | 0.91 | 0.90 | 0.77 | saturating | 5.0 | 0.30 |
| adenine cAg@785 | purine | 0.77 | 0.77 | 0.60 | saturating | 4.3 | 0.23 |
| adenine sAu@785 | purine | 0.62 | 0.63 | 0.45 | saturating | 5.7 | 0.26 |
| adenine sAg@785 | purine | 0.63 | 0.65 | 0.46 | saturating | 4.9 | 0.23 |
| adenine cAg@532 | purine | 0.45 | 0.40 | 0.33 | saturating | 3.8 | 0.22 |
| adenine sAg@532 | purine | 0.34 | 0.37 | 0.24 | saturating | 5.6 | 0.23 |
| **ergothioneine** | **sulfur** | **0.97** | **0.95** | **0.86** | saturating | 3.7 | 0.28 |

The *theme-composition* correlations (0.34–0.97) are lower than the raw *distance-from-control* correlations (0.95–1.00, Spike/Response audits) because composition is a **share** that competes with other rising themes — a direct consequence of the high cross-talk quantified in Part 4. Colloidal, 785 nm arms are cleanest; solid-substrate 532 nm arms are noisiest.

---

## Part 4 — theme specificity and cross-talk

The target theme always moves more than the average off-target theme (specificity margin 0.07–0.25), but **off-target themes move 65–82% as much as the target** (leakage ratio). The BSV is **not** a set of clean orthogonal channels: raising a single analyte's concentration moves many themes together, because increasing surface coverage changes the whole spectrum (and the OOD state) rather than only the target's bands. This is a real property of Ag-SERS projected into a Raman atlas, quantified here rather than hidden.

---

## Part 5 — component contributions: scaling vs redistribution

A key mechanistic distinction emerges:

- **Adenine trajectories are *redistributive*** (component turnover 0.6–0.8 from low→high dose; low/high profile correlation 0.13–0.45). Different concentrations activate *different* components — the driver shifts from c3 at low dose toward c13 at high dose.
- **Ergothioneine is *scaling*** (turnover 0.2, profile correlation 0.81). One component (c18) simply grows.

So the BSV does not merely rescale a fixed signature with concentration; for adenine it traces a genuinely curved, component-redistributing path. This matters for any future quantitation: a single-component linear model would be wrong for adenine.

---

## Part 6 — trajectories

Dose series are curved, not straight (mean step-to-step curvature 43–97°; straightness 0.54–0.93, best for colloidal 785 nm). The dominant net theme shift is **purine for 6/7 adenine arms**. One honest subtlety: ergothioneine's *largest* net theme displacement is toward purine even though its *most monotonic* theme is sulfur (ρ 0.97) — magnitude and monotonicity can disagree because ergothioneine's driving components carry mixed theme weight. Reported, not smoothed.

---

## Part 7 — inter-analyte geometry

BSV geometry recovers biochemistry:

| analyte | nearest neighbour (cosine) | second |
| --- | --- | --- |
| adenine | **xanthine (0.96)** | hypoxanthine |
| xanthine | **guanine (0.98)** | adenine |
| guanine | **xanthine (0.98)** | adenine |
| glucose | **(+)-glucose (1.00)** | urate |
| cholesterol | albumin (0.97) | hypoxanthine |

The **purines form a clear cluster** and the glucose stereoisomers merge — the BSV separates analytes by chemistry using themes alone. The cholesterol↔albumin proximity (both large, lipid+protein-rich serum molecules) is a plausible but weaker relationship.

---

## Part 8 — replicate stability

Reproducibility is **strongly substrate-dependent**: ICC(1) for the purine theme ranges from **0.14** (sAg@532, solid substrate) to **0.83** (cAu@785, colloid); within-dose CV 0.05–0.19. OOD CV 0.13–0.35, confidence CV 0.08–0.16. **Colloidal substrates give reliable BSVs; solid substrates do not.** This is the single most actionable acquisition finding.

---

## Part 9 — confidence system

| group | median OOD | median confidence |
| --- | --- | --- |
| pure Raman (in-domain) | **0.04** | **0.39** |
| pure Ag-SERS | 0.15 | 0.21 |
| serum baseline / spikes | 0.28 | 0.26 |
| serum spike — **strong** Ag adsorbers | 0.27 | 0.27 |
| serum spike — **weak** Ag adsorbers | 0.28 | 0.27 |

Confidence behaves sensibly at the **domain** level: it is highest and OOD lowest for in-domain pure Raman, and confidence correlates negatively with OOD across the corpus (ρ −0.57). But it has a real gap: **within serum, confidence does not distinguish strong from weak Ag adsorbers** (0.27 vs 0.27), even though the Spike Validation showed only strong adsorbers are recoverable. Confidence currently reports *spectrum quality / distance to the reference cloud*, not *analyte identifiability in this matrix*. (A minor inversion — pure Ag-SERS confidence 0.21 < serum 0.26 despite lower OOD — comes from the evidence-concentration and matrix-penalty terms, and confirms confidence is not a pure function of OOD.)

---

## Part 10 — V2 (old radar) vs V6 (BSV)

Reproduced qualitative behaviours (V6 recovers all three canonical V2 signals **without hand-curated axes**):

| behaviour | V2 (curated radar) | V6 (BSV) |
| --- | --- | --- |
| adenine → purine ↑ | by design | ρ 0.91 (colloidal), evidence-derived |
| ergothioneine → sulfur ↑ | by design | ρ 0.97, evidence-derived |
| uricase depletion → purine ↓ | by design | Δ −0.011 (correct sign) |

| dimension | V6 improvement | where V2 is still more intuitive |
| --- | --- | --- |
| evidence | full provenance (components→analytes→perturbation→literature) | — |
| validation | perturbation-validated, uncertainty-quantified | — |
| confidence / OOD | first-class, calibrated to domain | — |
| axes | 11 evidence-derived themes | V2's 11 curated axes are cleaner (orthogonal by construction); V6 themes are coupled (~4 effective dims) |
| single-spectrum readout | requires reading OOD + confidence + leakage | V2's radar is visually simpler to eyeball |

V6 is a scientific upgrade (grounded, validated, honest); V2's hand-curated axes remain more visually orthogonal and immediately legible for a single spectrum.

---

## Part 11–12 — orthogonality and state-space geometry

Themes are **not** independent coordinates: mean off-diagonal \|r\| 0.24, max 0.65. Crucially, the strongest couplings are **chemically real**, not artifacts:

- protein ↔ sulfur (+0.57) — proteins contain cysteine/methionine;
- protein ↔ heme-porphyrin (+0.53) — heme proteins;
- protein ↔ aromatic-amino-acid (+0.45) — aromatic residues;
- aromatic-amino-acid ↔ lipid (−0.65) — aromatic rings vs aliphatic chains.

The BSV state space is **effectively ~4-dimensional** (participation entropy 4.2; 90% of variance in 4 PCs; PC1 alone 50%). The 11 nominal themes carry ~4 independent biochemical directions in this reference corpus — partly because themes share biology (above) and partly because the 167-analyte reference set does not span all theme combinations.

---

## Part 13 — failure analysis (honest)

| failure | example | root cause |
| --- | --- | --- |
| **Weak Ag adsorbers mis-themed** | phenylalanine in serum → purine (0.27), not aromatic-AA | **physics/measurement**: phe barely adsorbs to Ag, so the serum background (purine-ish) dominates. Flagged: OOD 0.28, confidence 0.26. |
| **Cross-talk / leakage** | off-target themes move 65–82% as much as target | **physics**: concentration changes whole-spectrum contrast, not one band; amplified by OOD. |
| **Low effective dimensionality** | ~4 of 11 themes | **ontology + reference coverage**: coupled themes + a reference set that doesn't span all combinations. |
| **Substrate-dependent noise** | ICC 0.14 on solid substrates | **measurement**: solid-substrate SERS is far noisier than colloid. |
| **Confidence blind to recoverability** | strong ≈ weak adsorber confidence | **ontology/engine design**: confidence uses distance-to-reference, not per-analyte SERS adsorption priors. |
| **Serum background bias** | most serum spikes' top theme is purine | **reference coverage / physics**: the serum-colloid background projects onto purine-weighted components. |

Every failure is attributable to physics, reference coverage, or the ontology — **not** to a broken implementation. The engine surfaces each with OOD and confidence flags.

---

## Part 14 — recommendations (evidence-based; nothing implemented)

**Immediate (no atlas change):**
- *Ontology:* add a per-analyte / per-theme **matrix-recoverability prior** so confidence reflects Ag adsorption strength, closing the strong-vs-weak gap (Part 9).
- *Ontology:* down-weight or explicitly model the serum-colloid background component so it does not project onto the purine theme (Part 13).
- *Engine:* report a **leakage / specificity** number alongside each theme so cross-talk (Part 4) is visible per inference.

**Medium term:**
- *Reference corpus:* the BSV is ~4-D because the 167-analyte reference set does not span all theme combinations; adding orthogonal reference chemistries would raise effective dimensionality (Part 12).
- *Experimental:* prefer **colloidal over solid substrates** for any quantitative BSV use (ICC 0.14 vs 0.83, Part 8).
- *Ontology:* the recovered purine sub-structure and protein↔sulfur/heme couplings justify splitting/linking themes to match real biochemical structure.

**Long term:**
- *Architecture:* an **in-domain Raman dose-response** would separate "BSV cannot quantify" from "Ag-SERS is the limiting factor" — the single most informative missing measurement.
- *DART:* the redistributive-vs-scaling distinction (Part 5) and the trajectory-fingerprint schema are the right substrate for comparing future DART electrochemical trajectories against chemical reference trajectories.

---

## Scientific conclusions

1. **The BSV is a real, stable biochemical coordinate system** for well-adsorbed analytes on colloidal substrates: monotonic saturating dose response, purines clustering, biologically-meaningful theme couplings, and reproducibility ICC up to 0.83.
2. **It is lower-dimensional and more coupled than its 11 nominal axes** (~4 effective dimensions); the coupling encodes shared biology and is scientifically informative, not merely noise.
3. **Its confidence signal is a domain/quality gauge, not an identifiability gauge** — the most important limitation to communicate to any downstream user.
4. **Its failures are physical and honest** (weak adsorbers, serum background, substrate noise), all surfaced by OOD and confidence, none due to the implementation.
5. **It recovers every canonical V2 behaviour without hand-curated axes**, trading V2's visual orthogonality for evidence, validation and quantified uncertainty.

---

## Outputs

`GAIRA_BSV_VALIDATION.md` (this file) · `GAIRA_BSV_VALIDATION.pdf` (7 pages, 6 figures). Tables under `results/v5_rebuild/bsv_validation/tables/` (per-dataset BSVs, monotonicity, cross-talk, specificity, component contributions, nearest-neighbours, replicate stability, confidence system, theme correlation + mutual information); artifacts (trajectories, state-space geometry, validation manifest). All reproducible from the two scripts above; the frozen engine was measured, never modified.
