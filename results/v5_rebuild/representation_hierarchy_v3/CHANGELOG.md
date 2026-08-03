# Cross-Modal Transfer Analysis — Changelog

The Raman → Ag-SERS transfer analysis has three additive generations. **Each is preserved and
runnable; none overwrites another.** The frozen atlas fingerprint `09ed804a40836f4a05a91ba10900cded`
is identical across all three.

---

## V1 — Pure Ag-SERS validation (`foundation_audit/reports/PURE_AG_SERS_VALIDATION.md`)

- One metric per analyte: coordinate cosine (median 0.42), called *recoverability*.
- Established the modality-gap rung of the validation ladder and the recoverability tiers.
- **Kept verbatim.** V2/V3 reframe this number as Level 1 (latent fingerprint); they never
  change it.

## V2 — Theme preservation (`pure_ag_sers_theme_preservation/`, Foundation Explorer V2)

- Introduced the distinction between **latent fingerprint** and **biochemical theme** preservation.
- Showed raw theme cosine (0.92) is a **compositional-baseline artifact**; added the
  baseline-subtracted "distinctive" cosine + null (median 0.11, self-rank ≈ chance).
- Documented the **purine attractor** (50/51 → nucleic_purine) and the four-level framework
  (latent · theme · perturbation · matrix). 51 four-level cards; 9 figures; 30 tests.
- **Kept and runnable.** V3 reproduces every V2 number bit-for-bit
  (`hierarchy_summary.json → reproducibility_vs_v2`, max abs diff 0.0).

## V3 — Representation hierarchy (`representation_hierarchy_v3/`, Foundation Explorer V3) — THIS UPDATE

The interpretation is reorganised into an explicit **five-level representation hierarchy** and
the theme level is expanded from two metrics to a **ladder of five**, with new first-class
metrics and honest null controls throughout.

### What is new
- **Layer 4 · Theme RANK preservation (Spearman ρ)** — ordering of all 11 themes. Raw ρ median
  0.87, but *tested* against a null: rank_separation only +0.010 (baseline-inflated like cosine;
  a slim identity edge, positive 34/51).
- **Layer 5 · Top-k theme overlap** promoted to a first-class metric — top-2 0.50, top-3 0.67
  (the interpretable middle-ground; avoids argmax instability).
- **Representation Hierarchy** central figure + per-level distribution stats (median/variance/
  examples/limitations for all five levels).
- **Purine attractor, quantified** — Sankey flow (Raman→Ag dominant theme), per-analyte **ΔPurine**
  (36/51 increase), and **ΔPurine vs latent fingerprint: r=−0.38, p=0.006** (weaker adsorption ⇒
  stronger attractor pull).
- **Matrix robustness regression** — does pure transfer *predict* serum recovery? Only weakly:
  r=0.17, R²=0.028, p=0.24 (n.s.). An honest quantitative downgrade of V2's categorical claim.
- **9-layer per-analyte cards** (latent · MSS · theme cosine · rank ρ · top-3 · argmax · family ·
  interpretation · limitations) in physics-aware language.
- **Foundation Explorer V3** — a ~15-page app organised around the hierarchy.
- New docs: this changelog, `HIERARCHY_METRICS_SPECIFICATION.md`, `REPRESENTATION_HIERARCHY.md`,
  `INTERPRETATION_GUIDE.md`. New tests validating the hierarchy metrics + V3 pages.

### What is unchanged
- Every V1/V2 metric and number (reproduced exactly). The frozen atlas, NMF, preprocessing, MSS
  generation, ontology, registry, and theme weights. The fingerprint. V1 and V2 apps.

### Scientific bottom line (V3)
Raw theme cosine **and** raw rank ρ are baseline-inflated; the honest, null-corrected theme
preservation is selective and adsorption-tracking; the purine attractor is quantified and
significantly anti-correlated with adsorption fidelity; pure transfer is a weak predictor of
serum recovery; and dynamic perturbation — rare but decisive — is the strongest rung. The
framing changed from "does the theme survive?" to "**how far up the representation hierarchy does
agreement survive, and where does surface physics take over from biochemistry?**"

## V4 — Null-calibrated hierarchical recovery (`hierarchical_recoverability_v4/`, Foundation Explorer V4)

Recovery is redefined **statistically**: every representation metric is calibrated against an
analyte-mismatched null, and an analyte is "specifically recovered" at a level only if its own
Ag-SERS is the uniquely nearest match (retrieval rank-1) and jackknife-stable — never a raw cosine
above a threshold.

### What is new
- **Analyte-mismatched nulls** for every level (latent, MSS, raw theme, 4 theme-identity residual
  variants, Spearman, top-k, argmax), with retrieval-rank permutation p, BH-FDR (reported;
  degenerate at N=51), and leave-one-replicate-out jackknife stability.
- **Evidence-based recovery flags** (independent: latent / MSS / theme / perturbation / matrix) +
  transparent profiles; counts, bootstrap CIs, overlap matrix, 90/95/99 threshold sensitivity.
- **The MSS-primary hypothesis is tested and rejected:** MSS null-separation (0.0075) is smaller
  than latent's (0.024); MSS recovers 3/51, a strict subset of latent's 7/51.
- **Purine attractor blank control:** the unspiked-serum-on-Ag blank is already purine-dominant
  (0.27) before any analyte; Δpurine anti-correlates with latent (r=−0.38) and MSS (r=−0.40).
- **Matrix prediction:** no pure metric significantly predicts serum recovery; only confidence
  (r=0.71) — flagged as likely signal-strength, not identity.
- 11 figures, 51 nine-field cards, a 14-page PDF + Markdown report, Foundation Explorer V4 (15 pages).

### Headline counts (of 51 matched analytes)
latent-specific **7** · MSS-specific **3** · theme-specific **4** · perturbation **3** ·
matrix **9** (serum-tested). Raw cosines (MSS 0.74, theme 0.92) are shared background.

### Unchanged
Every V1/V2/V3 metric (V4 reproduces V3 matched values bit-for-bit, max abs diff 0.0). The frozen
atlas, NMF, preprocessing, MSS, ontology, registry, theme weights; the fingerprint; V1–V3 apps.
