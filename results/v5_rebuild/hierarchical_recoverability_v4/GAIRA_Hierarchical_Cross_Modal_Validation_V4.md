# GAIRA — Hierarchical Cross-Modal Validation (V4)

### Null-calibrated recoverability across Raman, Ag-SERS, perturbation and biological matrix

*Additive analysis on the frozen atlas `09ed804a40836f4a05a91ba10900cded`. Every matched value
reproduces V3 bit-for-bit; V4 adds null calibration, evidence-based recovery definitions, blank
controls, and matrix prediction. Nothing retrained. Tables in `tables/`, figures in `figures/`,
per-analyte cards in `analytes/`, method in `METRICS_AND_DECISION_RULES.md`, audit in
`AUDIT_OF_V3_METRICS.md`.*

Each claim is tagged: **[obs]** observation · **[comp]** computation · **[interp]** interpretation ·
**[infer]** inference · **[spec]** speculation · **[lim]** limitation.

---

## 1 · Executive summary

Raman → Ag-SERS transfer cannot be described by one score, and — the central V4 result — **it must
not be described by one *level* either**. When every representation metric is calibrated against an
analyte-mismatched null, **analyte-specific cross-modal recovery is rare at every level**: latent
**7/51 (14%)**, MSS **3/51 (6%)**, theme **4/51 (8%)**. **[comp]** The impressively high raw cosines
(MSS 0.74, theme 0.92) are almost entirely shared background — matched barely exceeds mismatched
null (MSS +0.008, theme +0.002). **[interp]** The candidate hypothesis that **MSS is the primary
cross-modal metric is rejected**: MSS separates from its null *less* than the latent coordinates,
and the 3 MSS-recovered analytes are a **strict subset** of the 7 latent-recovered ones. **[infer]**
The strongest evidence is not a cosine at all but **functional perturbation** (3 analytes). The
purine attractor is present in the **unspiked-serum blank** (purine share 0.27, dominant theme)
*before any analyte* — it is substantially a background/substrate phenomenon. **[obs]**

## 2 · Scientific questions

1. How many of the 51 matched pure analytes retain **statistically specific** information at each
   representation level (latent, MSS, theme), against a null? **[comp]**
2. Does MSS earn the role of primary cross-modal metric? **[comp]**
3. Is "raw theme high" evidence of analyte identity? **[comp]**
4. Is the purine attractor an analyte-binding effect or a background/substrate effect? **[comp]**
5. Does any pure-transfer metric predict serum (matrix) recoverability? **[comp]**

## 3 · Dataset and provenance

- **Raman reference:** frozen corpus (`dataset.load_reference_corpus`), per-analyte mean coord.
- **Ag-SERS:** `spike_lib.load_pure_sers` — 265 spectra / 53 analytes / **5 replicates each**; 51
  match the Raman reference. Substrate Ag colloid, 785 nm, buffer. **[obs]**
- **Blank control:** `load_serum_baseline` (unspiked serum on Ag, 15 spectra) + uricase
  serum_reference (5). No pure Ag-colloid buffer blank exists in the dataset. **[lim]**
- **Perturbation:** `validation_results.json` (adenine, ergothioneine, uricase). **Matrix:**
  `phase7_serum_vs_pure.csv`.

## 4 · Frozen atlas & no-retraining confirmation

`z` = `atlas.coordinates` (NNLS onto the fixed 24-NMF basis); `b` = `eng.infer(z).bsv.composition`
(11 themes); `m` = `mss.activate(b)` (12 biochemical motifs). Identical calls to V3 ⇒ matched
values reproduce V3 (`reproducibility_vs_v3` max abs diff **0.0**). Fingerprint verified
`09ed804a…` at load. No NMF, preprocessing, ontology, registry, MSS, BSV, or normalization asset is
modified. **[comp]**

## 5 · Why transfer cannot be one metric

Different metrics answer different questions (§ METRICS_AND_DECISION_RULES). A single cosine
conflates surface fidelity, motif preservation, broad interpretation, analyte identity, functional
response and matrix visibility — which have different answers for the same analyte (adenine: weak
latent 0.36, but functional dose-response ρ=0.996). **[interp]**

## 6 · Level 1 — latent fingerprint preservation

`C_latent = cos(z_R, z_S)`; matched median **0.425**, mismatched-null median 0.380, per-analyte
separation **0.024**. **7/51 recovered** (rank-1 + jackknife-stable): *creatinine, glutathione,
hypoxanthine, thymine, urate, urea, xanthine* — strong Ag chemisorbers (oxopurines, thiol, N-rich
small molecules). **[comp]** A further 3–5 analytes are *supporting* (above null95, not uniquely
rank-1). The latent level carries the most analyte identity of any cosine — and it is still only
14%. **[interp]**

## 7 · Level 2 — MSS preservation

`C_MSS = cos(m_R, m_S)`; matched median **0.740** — but mismatched-null median **0.732**, so
per-analyte separation is only **0.0075**, *smaller than latent's*. **3/51 recovered** (creatinine,
urea, xanthine), a **strict subset** of the latent set (Figure 3, overlap matrix). **[comp]** The
high 0.74 is background, not motif-identity. **The MSS-is-primary hypothesis is rejected by the
null.** **[infer]** Expected-motif retention and top-k motif overlap are reported per analyte;
`mss_specificity_ranking.csv` ranks by null-adjusted specificity (Figure 6).

## 8 · Level 3 — broad biochemical-theme similarity

`C_theme_raw = cos(b_R, b_S)`; matched median **0.918**, null median **0.915** → separation
**0.002**. **[comp]** This is genuinely useful as *broad biochemical interpretation* — but it is
**not analyte identity** and must never be used to call an analyte "recovered". **[interp]**

## 9 · Theme baseline / common-mode issue

Every analyte's composition is dominated by the same high-share background themes plus ontology
cross-loading, so any two analytes — even unrelated ones — already agree at ≈0.9 (Figure 4, Figure
7). Raw theme cosine is a **common-mode** measurement. **[interp]**

## 10 · Analyte-specific theme alternatives

Four identity residuals were evaluated (Raman-centered, modality-centered, serum-blank-corrected,
whitened), each scored on **self-retrieval rank-1, jackknife stability, family-leave-out
robustness** — *not* on the highest score (`theme_variant_comparison.csv`). Selected: the
**Raman-centered identity residual** (best identity + stability). Whitened has the largest median
separation but poorer per-analyte identity — a higher score that is less trustworthy. **[comp]**
Theme-specific recovery (rank-1 + stable + expected-theme in Ag-SERS top-3): **4/51** (creatinine,
hypoxanthine, urea, xanthine). **[comp]**

## 11 · Spearman, top-k and argmax

- **Spearman ρ** matched median 0.87, null median ≈0.85 → not analyte-specific; gross-ordering
  descriptor only. **[comp]**
- **Top-k**: top-3 median 0.667; 28/51 exceed their chance-adjusted null (broad-theme retention).
- **Argmax**: 35% agree; **13/51 survive a top-two-margin > 0.02 filter** — brittle, purine-driven,
  supporting only. **[comp]**

## 12 · Null controls

Analyte-mismatched null (per-analyte percentiles 90/95/99), retrieval-rank permutation p (discrete
floor 1/51 = 0.0196), BH-FDR (reported; degenerate at N=51), chance-adjusted top-k/Spearman nulls,
and leave-one-replicate-out jackknife stability. **[comp]** Recovery uses **rank-1 + stable** as the
significance gate because the retrieval p floors at 1/51; FDR is reported for transparency. **[lim]**

## 13 · Recoverable-analyte definitions

An analyte is **specifically recovered at a level** iff its own Ag-SERS is the *uniquely nearest*
match among all 51 (rank-1 ⇒ matched > null95; p=0.0196) **and** jackknife-stable; theme recovery
also requires the expected theme in the Ag-SERS top-3. Independent flags (latent / MSS / theme /
perturbation / matrix), never one score. Weaker *supporting* tier = above null95 but not rank-1. **[comp]**

## 14 · Recoverable-analyte counts

| Level | n recovered | denominator | fraction (95% CI) |
|---|--:|--:|---|
| Latent-specific | **7** | 51 | 0.137 (0.06–0.25) |
| MSS-specific | **3** | 51 | 0.059 (0.02–0.14) |
| Theme-specific | **4** | 51 | 0.078 (0.02–0.16) |
| Top-3 > null | 28 | 51 | 0.549 (0.41–0.69) |
| Argmax robust | 13 | 51 | 0.255 (0.14–0.37) |
| Perturbation-validated | **3** | 51 | 0.059 (0–0.14) |
| Matrix-recovered | 9 | **serum-tested** | 0.176 (0.08–0.29) |

Threshold sensitivity (90/95/99 null percentiles) changes only the *supporting* tier (rank-1 is
percentile-independent): latent supported 10/10/7, MSS 6/6/3, theme 5/5/4. Family breakdown and
per-analyte lists: `recoverable_analytes_by_level.csv`, `per_analyte_evidence_profile.csv`.
**Evidence profiles:** 25 broad-theme-only, 15 no-analyte-specific-evidence, the remainder carrying
one or more specific/functional/matrix flags. **[comp]**

## 15 · Purine attractor and blank controls

The **unspiked-serum-on-Ag blank** projects to **purine share 0.266 with nucleic_purine dominant** —
the attractor is present in the background **before any analyte is added** (Figure 8). **[obs]**
Δpurine (Ag − Raman) is positive for 36/51; it **anti-correlates with latent (r=−0.38, p=0.006) and
MSS (r=−0.40, p=0.003)** — weaker preservation ⇒ stronger pull. **[comp]** Wording: the purine
attractor is a **phenomenological, substrate/background-associated** observation; a pure Ag-colloid
buffer blank is not in the dataset, so the mechanism (colloid vs serum constituents vs NMF/MSS
cross-loading) is not fully isolated. **[lim]** We do **not** claim Ag binding alone explains it.

## 16 · Perturbation validation (3 analytes)

- **Adenine** (dose→nucleic_purine): ρ=0.996, Langmuir K=0.89 µM, R²=0.993. **[obs]**
- **Ergothioneine** (dose→sulfur_antioxidant): ρ=0.927, K=1.52 µM. **[obs]**
- **Uricase/urate** (**directional depletion, not dose**): oxopurine-carbonyl motif Δ=−0.060, purine-
  ring motif ≈0, broad purine theme diffuse — correct sign at the motif layer. **[obs]**
Every other analyte: **perturbation sensitivity: not tested.** Functional response is the strongest
evidence in the hierarchy but is available for only 3/51. **[interp]**

## 17 · Matrix recoverability

Serum spike (`phase7`): 9 strong / (moderate/weak). **No pure metric significantly predicts serum
displacement** (latent r=0.17 p=0.24; MSS r=0.14 p=0.32; MSS-specificity r=0.21 p=0.14; theme_raw
r≈0). The only significant correlate is **overall confidence (r=0.71, p=5e-9)** — but this likely
reflects **signal strength**, not analyte identity, and is flagged as such. **[comp]** Matrix
recovery is a **separate property**; it is never inferred for untested analytes. **[interp]**

## 18 · Per-family results

Strong chemisorbers (oxopurines, thiol amino acids, N-rich small molecules) dominate every
recovered set; amino acids, sugars, lipids and pyrimidines are largely broad-theme-only. Δpurine
gain concentrates in weak adsorbers (`rank_by_family` in V3; Figure 8). **[comp]**

## 19 · Representative analytes (Figure 11)

Adenine (latent redistributed, perturbation+matrix✓), urate (latent+perturbation+matrix✓),
hypoxanthine (latent+theme+matrix✓), xanthine (all cosine levels✓), glucose/tyrosine/uracil
(broad/none). Real Raman vs Ag-SERS spectra shown. **[obs]**

## 20 · Metric-selection recommendations (`metric_decision_table.csv`)

Latent cosine → exact/substrate fidelity **and** the best cross-modal identity cosine (MSS
*supporting*). Raw theme → broad interpretation. Raman-centered identity residual → analyte-specific
theme. Spearman/argmax → descriptive. Perturbation → functional validation. Matrix → mixture
visibility. **MSS is not primary.** **[infer]**

## 21 · Limitations

Raman-trained atlas; no learned modality correction; the purine attractor's mechanism is not fully
isolated (no buffer blank); only 3 perturbation cases; recovery depends on null thresholds and the
discrete retrieval p-floor (FDR degenerate at N=51); matrix dependence; **confidence ≠ analyte
identifiability**; incomplete Au-SERS grounding; 5 replicates per analyte limits jackknife
resolution. **[lim]**

## 22 · Conclusions

Cross-modal biochemical recovery is **hierarchical and rare**: only the strong-chemisorber minority
retains analyte-specific latent structure (7/51), motif and theme identity are weaker still (3–4/51),
raw cosines are broad-interpretation background, the purine attractor is a background phenomenon, and
functional perturbation — though limited to 3 analytes — is the strongest evidence. GAIRA should
report the whole calibrated hierarchy and never call an analyte "detectable" from a raw cosine. **[interp]**

## 23 · Methods & formulas

See `METRICS_AND_DECISION_RULES.md` for every formula, null, and decision rule; `AUDIT_OF_V3_METRICS.md`
for the source audit. All computation via the frozen engine; deterministic (fixed seeds); reruns
yield identical tables (verified).

## 24 · Reproduction

```bash
python results/v5_rebuild/hierarchical_recoverability_v4/code/recoverability_analysis.py
python results/v5_rebuild/hierarchical_recoverability_v4/code/make_figures_v4.py
python results/v5_rebuild/hierarchical_recoverability_v4/code/make_cards_v4.py
python results/v5_rebuild/hierarchical_recoverability_v4/code/make_report_v4_pdf.py
```
Interactive: `streamlit run gaira_foundation_explorer_v4/app.py`. Frozen atlas unchanged.
