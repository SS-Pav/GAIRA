# Interpretation Fixes — Round 2 (radar honesty, paper validation, biological visuals)

Triggered by a close read of the demo + the source Ag-SERS paper (PMC12680727). These
are interpretation/visualization fixes; **no frozen science was changed** (atlas, NMF,
ontology, MSS, BSV equations all byte-identical; fingerprint `09ed804a…` verified).

## FIX 1 — Calibration cascade is now Δ-vs-baseline (the weak-adsorber fix)

**Problem:** the flagship cascade's radar (and MSS/BSV panels) showed the **absolute**
biochemical composition. For a *weak* Ag adsorber (ergothioneine) the absolute state is
dominated by the **SERS background** (purine, from serum uric acid/hypoxanthine), so the
cascade showed "purine" as the top motif for ergothioneine — misleading. The user was
right that it looked wrong.

**Fix:** in calibration mode the biochemical half of the cascade (panels **3 ΔMSS**,
**4 ΔBSV**, **5 Δradar**, and the "top Δ motif" stat) now shows the **signed change vs
the zero-dose baseline**. Ergothioneine now correctly shows **sulfur up** and the purine
background **down**; adenine shows purine up. Panels 1–2 (spectrum, components) stay
absolute. `reasoning_cascade(..., baseline_coord=, delta_axes=, delta_scale=)`.

## FIX 2 — Serum validated against the source paper (PMC12680727)

Added §B+ "Validated against the source paper": GAIRA's serum tiers, computed only from
the frozen atlas, **independently reproduce the paper's finding** that only uric acid +
hypoxanthine are recoverable and that "concentration alone doesn't predict SERS
visibility" (adenine 0.4 µM, glucose mM → nothing). Also added the honest **urate
caveat**: GAIRA rates urate "poor" because it measures *incremental directional*
recoverability, and urate is the *saturated serum background* — the paper's "strongest
absolute signal" and GAIRA's "poor incremental spike" are both correct. See
PAPER_VALIDATION.md.

## FIX 3 — Biological near-null visuals (the COVID-overlap fix)

**Problem:** COVID absolute BSV radar + raw PCA overlapped completely — the difference is
tiny and the shared serum background dominates.

**Fix:** the **within-cohort BALANCED view** (standardize each theme's deviation) is now
shown for **every** cohort (previously diabetes-only). It makes a real-but-small group
difference legible without faking signal (labelled display-only). Combined with the
signed **ΔBSV group radar** (already added) beside the effect-size forest, and the
absolute radar + raw PCA demoted to an expander. COVID now shows its (small) pyrimidine/
sulfur-up, aromatic/lipid-down structure.

*On "local vs global":* the balanced view is a **local** within-cohort standardization
layered on the **global** frozen coordinates — the two are complementary (global gives
comparable axes; local makes the small difference visible), not either/or.

## FIX 4 — Per-cohort biochemical interpretation for all cohorts

Every study now prints a cautious per-cohort "Biochemical interpretation" naming its top
MSS motif drivers + domain note (previously only diabetes had bespoke depth).

## FIX 5 — "How to read the radars" explainer

A shared `components.radar_guide()` expander on Calibration and Biological explains
absolute vs Δ radar, compositional closure, and that the absolute radar reflects the
SERS background for weak adsorbers — so a first-time viewer doesn't hit the same
confusion.

## FIX 6 — "Why NMF, not PCA/UMAP/autoencoder" explainer

Reference Atlas now has an expander comparing NMF vs PCA/UMAP/autoencoder (non-negative,
additive/parts-based, a real readable spectrum, deterministic/freezable, chosen by
benchmark), and contrasting the **global fixed-coordinate** approach with the old
per-dataset PCA/clustering approach.

## Tests / QC

+2 regression tests: (a) the ergothioneine delta cascade shows sulfur up / purine
background down (analyte, not matrix); (b) the balanced view yields a legible group
difference for every cohort. Full V5+V6 suite passing; fresh-checkout verified; frozen
assets untouched; nothing pushed.
