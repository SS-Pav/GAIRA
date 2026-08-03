# V4 Metrics & Decision Rules — null-calibrated hierarchical recoverability

*Exact formulas, null models, and the evidence-based recovery definitions. Every matched value
reproduces V3 bit-for-bit (`recoverability_summary.json → reproducibility_vs_v3`, max abs diff
0.0). Frozen atlas `09ed804a…` unchanged. Computed by `code/recoverability_analysis.py`.*

Notation: for analyte *i*, `z_R/z_S` = 24-D NMF coordinates (per-analyte mean over replicates,
Raman/Ag-SERS); `m_R/m_S` = 12-D MSS motif activations; `b_R/b_S` = 11-D theme composition.
`μ_R, μ_S` = grand-mean theme vectors across the 51 matched analytes. `b_blank` = theme vector
of the unspiked-serum-on-Ag background.

---

## The metrics (formula · range · interpretation · null)

### Level 1 — Latent fingerprint preservation
```
C_latent(i) = cosine(z_R(i), z_S(i))            ∈ [−1, 1]
```
*How strongly the exact Raman-derived latent fingerprint is preserved after Ag-SERS
observation.* Surface-physics-sensitive; **not** a detectability score. Matched median **0.425**.

### Level 2 — MSS motif preservation
```
C_MSS(i) = cosine(m_R(i), m_S(i))               ∈ [−1, 1]
```
Also: top-1 motif agreement, top-3 motif overlap, expected-motif rank/enrichment, strongest
gained/lost motif. Matched median **0.740**. *Candidate primary cross-modal metric — tested below.*

### Level 3A — Raw biochemical-theme similarity
```
C_theme_raw(i) = cosine(b_R(i), b_S(i))         ∈ [−1, 1]
```
**Broad biochemical-composition similarity only — never analyte identity or recoverability.**
Matched median **0.918**; mismatched-null median **0.915** → the score is almost entirely shared
ontology/background structure.

### Level 3B — Analyte-specific theme preservation (4 variants)
```
A Raman-centered:    r_R = b_R − μ_R ; r_S = b_S − μ_R ;  C = cosine(r_R, r_S)
B Modality-centered: r_R = b_R − μ_R ; r_S = b_S − μ_S ;  C = cosine(r_R, r_S)
C Ag-blank-corrected:r_R = b_R − μ_R ; r_S = b_S − b_blank ; C = cosine(r_R, r_S)
D Whitened:          w = Σ^(−1/2)(b − μ) per modality ;   C = cosine(w_R, w_S)
```
Each variant is scored on **self-retrieval rank-1 fraction, jackknife stability, and
family-leave-out robustness** — NOT on the highest raw score. The selected metric is the
**Raman-centered identity residual** (variant A): best identity + stability; see
`theme_variant_comparison.csv`. (Whitened has the largest median separation but poorer
per-analyte identity and stability — a higher score that is less trustworthy.)

### Level 3C — Theme-rank preservation
```
ρ_theme(i) = Spearman(b_R(i), b_S(i))           ∈ [−1, 1]
```
*Gross theme ordering.* Matched median **0.87**; its mismatched-null median is **≈0.85** — so it
is **not analyte-specific** (matched ≈ null). Descriptive only.

### Level 3D — Top-k theme overlap
```
O_k(i) = |Top_k(b_R) ∩ Top_k(b_S)| / k ,  k ∈ {2,3}
```
Reported raw and chance-adjusted against the mismatched top-k null. Top-3 median **0.667**.

### Level 3E — Dominant-theme agreement (argmax)
```
A_argmax(i) = 1[argmax(b_R) = argmax(b_S)]
margin(i)   = b_(1) − b_(2)   (top-two gap, each modality)
```
**Strict, brittle near ties, dominated by the Ag purine attractor.** 35% agree; **26% survive a
top-two-margin > 0.02 robustness filter** (`argmax_robust`). Supporting/diagnostic only.

### Level 4 — Controlled perturbation sensitivity *(3 analytes only)*
Adenine (dose→nucleic_purine): Spearman ρ, Langmuir K/R², dynamic range, off-target, trajectory
smoothness, replicate reproducibility. Ergothioneine (dose→sulfur_antioxidant): same. Uricase
(urate, **directional depletion, not dose**): oxopurine-motif Δ, purine-ring-motif Δ, broad
purine-theme Δ, expected-sign correctness, reproducibility, isotope control. **Every other analyte:
"Perturbation sensitivity: not tested."**

### Level 5 — Matrix recoverability *(serum-tested analytes only)*
Serum spike direction agreement, displacement/detectability, replicate reproducibility, tier
(strong/moderate/weak). Tested as a predictor from every pure metric (below).

---

## Null models

- **Analyte-mismatched null** (identity): for level with per-analyte vectors, `null_i = { sim(R_i,
  S_j) : j ≠ i }`. Percentiles 90/95/99 and median per analyte.
- **Retrieval-rank permutation p:** among all 51 candidates `sim(R_i, S_·)`, `rank(i)` = position
  of the true match (1 = uniquely nearest); `p_i = (1 + #{j≠i: sim(R_i,S_j) ≥ sim(R_i,S_i)}) / (1 +
  50)`. Discrete floor **1/51 = 0.0196**.
- **Benjamini-Hochberg FDR** across the 51 per-analyte p-values, per level. Reported for
  transparency; **degenerate at N=51** because the discrete p-floor makes tied minima give q≈0.05.
- **Chance-adjusted top-k / Spearman nulls:** mismatched distributions of the same statistic.
- **Jackknife stability:** leave out each of the 5 Ag-SERS replicates, recompute the per-analyte
  vector and the matched value; stable ⇔ all leave-one-out values still exceed `null95`.

---

## Recovery definitions (evidence-based — never a raw cosine threshold)

An analyte is **specifically recovered at a level** iff:

> its own Ag-SERS is the **uniquely nearest** match among all 51 (retrieval rank = 1 ⇒ matched >
> every mismatched ⇒ matched > null95; per-analyte p = 0.0196) **AND** it is **jackknife-stable**.

- **Latent-specific** — rank-1 + stable on `C_latent`.
- **MSS-specific** — rank-1 + stable on `C_MSS`; expected motif retained/enriched reported.
- **Theme-specific** — rank-1 + stable on the selected identity residual **AND** the expected theme
  in the Ag-SERS top-3 (so it is not explained only by the purine attractor). **Raw theme cosine
  alone never defines theme recovery.**
- **Perturbation-validated** — a controlled perturbation moves the expected motif/theme
  monotonically or in the correct direction with reproducibility. Only adenine, ergothioneine,
  urate.
- **Matrix-recovered** — serum spike strong tier only. Never inferred for untested analytes.

A weaker **supporting** tier = matched > null95 but not uniquely rank-1. Flags are **independent**
— there is no single opaque score. Transparent profiles: *spectral+motif preserved · latent
redistributed, motif retained · broad-theme only · perturbation-only · matrix-recovered · no
analyte-specific evidence.*

Rank-1 is used as the significance gate (not BH-FDR<0.05) because the retrieval p-value floors at
1/51; FDR q is still computed and reported. Threshold sensitivity at 90/95/99 percentiles is
reported for the supporting tier (rank-1 is percentile-independent).

---

## Decision table — which metric for which purpose (decided by the nulls, not preference)

| Purpose | Primary metric | Why (null result) |
|---|---|---|
| Exact spectral / substrate fidelity | **Latent cosine** | strongest cosine identity: 7/51 rank-1, separation 0.024 |
| Cross-modal chemical-motif preservation | **Latent cosine** (MSS *supporting*) | **MSS is NOT primary** — separation 0.0075 < latent's; 3/51 recovered are a strict subset of latent's 7 |
| Broad biochemical interpretation | Raw theme BSV / radar | high (0.92) but ≈ its null — interpretation, not identity |
| Analyte-specific theme retention | Raman-centered identity residual | null-adjusted; 4/51 rank-1 |
| Gross theme ordering | Spearman ρ | matched ≈ null → descriptive only |
| Strict dominant category | Argmax + top-two margin | brittle; 13/51 robust |
| Functional biochemical response | **Perturbation sensitivity** | strongest evidence; 3 analytes |
| Biological-mixture visibility | Matrix recoverability | separate property; 9/51 serum-strong |

**Verdict on the MSS hypothesis:** *rejected by the null analysis.* MSS motif cosine is dominated
by shared background (matched 0.740 vs null 0.732; separation 0.0075), recovers only 3/51 analytes
— **a strict subset of the 7 recovered at the latent level** — and adds no analyte beyond latent.
The latent fingerprint remains the most analyte-specific cross-modal cosine, and even it recovers
only ~14% of analytes. The strongest evidence is not a cosine at all but **functional perturbation
response**, available for three analytes.
