# Audit of the V3 cross-modal metrics (pre-V4)

*Verified from source before writing any V4 code. Confirms exactly what each existing metric
computes, its null (if any), and whether it can carry analyte identity. Frozen atlas
`09ed804a40836f4a05a91ba10900cded`; nothing here modifies it.*

## Representations (verified in `representation_hierarchy_v3/code/hierarchy_analysis.py`)

| Symbol | What | Source call | Aggregation |
|---|---|---|---|
| `z` (24) | latent NMF coordinates | `atlas.coordinates(V)` (NNLS onto frozen basis) | per-analyte **mean** over replicate spectra |
| `b` (11) | biochemical theme composition | `eng.infer(coordinates=z, domain="buffer").bsv.composition` | inferred from the per-analyte **mean coord** |
| `m` (12) | MSS biochemical motif activations | `mss.activate(bsv)` → biochemical motifs only (drops `colloid_matrix_background`) | from the same bsv |
| OOD, conf | `bsv.ood_score`, `bsv.overall_confidence` | | |

- **Raman side:** `dataset.load_reference_corpus()`, per-analyte mean coord over the frozen corpus.
- **Ag-SERS side:** `spike_lib.load_pure_sers()` → 265 spectra / 53 analytes / **exactly 5 replicates each**; per-analyte mean coord. 51 analytes match the Raman reference.
- **Replicate structure** enables jackknife (leave-one-replicate-out) stability — used in V4, not V3.

## Existing metrics, audited

| Metric | Formula | Range | Interpretation | Inflation / bias | V3 null? | Analyte-specific? | V4 role |
|---|---|---|---|---|---|---|---|
| **C_latent** | cos(z_R, z_S) | −1..1 | exact latent fingerprint preservation | surface-physics dominated | **none** | untested in V3 | headline (Level 1) — **add null in V4** |
| **C_MSS** | cos(m_R, m_S) | −1..1 | motif preservation | mid-level; surface-sensitive | **none** | untested in V3 | **candidate primary — add null in V4** |
| **C_theme_raw** | cos(b_R, b_S) | −1..1 | broad biochemical similarity | **compositional baseline + ontology cross-loading** | none (V2 had a distinct-only null) | **NO** (baseline-inflated) | supporting/diagnostic only |
| **C_theme_identity** (V3 "identity") | cos(b_R−μ_R, b_S−μ_R) | −1..1 | baseline-subtracted theme | one baseline choice only | vs-other-analytes (V2/V3) | weakly (median 0.11) | **re-derive as one of 4 variants** |
| **ρ_theme** | Spearman(b_R, b_S) | −1..1 | gross theme ordering | **baseline-inflated (raw ρ ≈ its null)** | rank_null (V3) | **NO** (sep +0.01) | supporting/diagnostic |
| **top-k** | \|topk_R ∩ topk_S\|/k | 0..1 | leading-theme retention | chance level not subtracted in V3 | none | partially | supporting — **add chance-adjusted null in V4** |
| **argmax** | 1[argmax b_R = argmax b_S] | 0/1 | dominant-theme agreement | **brittle; Ag purine attractor; ties** | none | NO (35% ≈ already-purine) | strict/diagnostic — **add top-two margin in V4** |
| **ΔPurine** | b_S[purine] − b_R[purine] | −1..1 | attractor pull | phenomenological | vs latent (V3, r=−0.38) | — | mechanism (needs blank control) |
| perturbation | dose ρ / motif Δ | — | functional validation | only 3 analytes | replicate reprod. | **yes (functional)** | strongest — Level 4 |
| matrix | serum displacement/direction | — | mixture visibility | matrix-dependent | — | separate | Level 5 |

## Key findings of the audit

1. **No V3 metric labelled "recoverability" is actually a null-calibrated identity test.** V3 reports
   matched values and, for the theme metrics, a mismatched null — but `C_latent` and `C_MSS` were
   **never** given matched-vs-mismatched nulls, permutation p-values, retrieval ranks, or FDR. V4 adds
   all of these, for every level.
2. **Raw theme cosine and raw Spearman ρ are broad-interpretation metrics, not identity metrics** — their
   matched values barely exceed their mismatched nulls (V2/V3). V4 keeps them but forbids using them alone
   to classify recovery.
3. **"Recoverable/detectable" has never been defined against a null.** V3 used descriptive tiers
   (0.80/0.65/…) with no statistical meaning. V4 defines recovery **only** as matched > analyte-mismatched
   null, with retrieval-rank significance (FDR-corrected) and leave-one-replicate-out stability.
4. **Blanks available for the purine control:** no pure Ag-colloid buffer blank exists in the SERS
   metabolites set, but **unspiked-serum-on-Ag blanks do** (`load_serum_baseline`, 15 spectra;
   `load_uricase` serum_reference, 5). V4 projects these to test whether the purine theme is present in
   the background before analyte addition.
5. **Reproducibility contract:** V4 recomputes `z`, `m`, `b` exactly as V3 (same calls, same aggregation),
   so `C_latent`, `C_MSS`, `C_theme_raw`, `ρ_theme` must reproduce V3 bit-for-bit; V4 only *adds* null
   calibration, recovery flags, blanks, and the Explorer. Nothing frozen is touched.

## What V4 must build (consequences)

- Matched-vs-mismatched null + retrieval-rank permutation p + BH-FDR for **every** level.
- Four theme identity-residual variants (Raman-centered, modality-centered, blank-corrected, whitened);
  select the most **stable and meaningful**, not the highest.
- Per-analyte recovery flags (latent / MSS / theme / perturbation / matrix) — independent, no single score.
- Actual counts + family breakdown + overlap + 90/95/99 threshold sensitivity + bootstrap CIs.
- Purine attractor blank/background controls + Δpurine correlations (latent, MSS, OOD, matrix, family).
- A decision table that lets the null results — not prior preference — decide whether MSS is primary.
