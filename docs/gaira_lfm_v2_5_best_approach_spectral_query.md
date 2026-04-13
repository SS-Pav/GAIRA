# GAIRA Spectral Query v2.5 — Best-Approach Build

## Why v2.5 Exists

v2.4 faithfully replicated the original v1/v3 spectral pipeline but inherited an unclear visual hierarchy — expected comparators competed with measured spectral structure for attention. v2.5 enforces a strict scientific priority:

1. **Measured spectral structure is PRIMARY** — this is the main story
2. **Band drivers explain** which windows drive observed BSV differences
3. **Expected literature BSV validates** whether the spectral structure tracks known biology
4. **Delta-shift agreement is the main comparison metric**, not raw profile cosine

## Architecture

```
Section 1: MEASURED SPECTRAL STRUCTURE (primary output)
  spectra → preprocessing → 22 windows → BSV → cohort analysis
  Outputs: mean spectra, radar, heatmaps, distributions, PCA

Section 2: SPECTRAL BAND DRIVERS (explanation)
  observed BSV shifts → top windows → motif/theme annotations
  Flow: data → windows → annotations (NOT: data → motifs → BSV)

Section 3: EXPECTED LITERATURE COMPARATOR (secondary)
  GAIRA landscape → condition profiles → post-hoc comparison

Section 4: OBSERVED VS EXPECTED VALIDATION
  delta cosine (primary), similarity matrix, sample-level, per-axis agreement
```

## What Changed from v2.4

| Aspect | v2.4 | v2.5 |
|---|---|---|
| Visual hierarchy | Expected comparators shown alongside measured | Measured BSV is clearly primary, expected is a separate validation section |
| Band drivers | Present but flat | Full section with ranked importance + motif annotation tables |
| Pairwise deltas | Not shown | Multi-cohort pairwise delta heatmap added |
| Delta comparison position | Buried in expected section | Promoted to lead validation metric |
| Trust graphs | Per-cohort spectral traversals | Removed (replaced by cleaner band-driver tables) |
| Sample distributions | Box plots | Box plots for top BSV axes by delta magnitude |
| Motif role | Separate explanation module exists | Integrated as annotation within band-driver section |

## Why Direct Spectral BSV Is Primary

The BSV projection is: `mean(window_intensities_mapped_to_component)`. This is a direct arithmetic operation on measured spectral intensities — no motif mapping, no theme aggregation, no literature influence. The resulting BSV values are as close to the data as possible while still being interpretable through the BSV framework.

Motifs appear only as annotations for the top driving windows — they explain what molecular features each window region is commonly assigned to in Raman/SERS literature, but they do not alter the BSV values.

## Why Delta-Shift Is the Main Comparison Metric

Raw profile cosine (observed BSV vs expected BSV) is dominated by shared serum SERS baseline features. All cohorts have similar overall BSV shape, so raw cosines are high and margins are small.

Delta-shift comparison asks the better question: "Does the observed disease-vs-reference SHIFT move in the same DIRECTION as the expected literature shift?" This is more scientifically meaningful and less confounded by baseline similarity.

## How to Run

```bash
# v2.5 (best-approach)
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v2_5_spectral_query.py

# v2.4 (preserved)
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v2_4_spectral_query.py
```

## What Comes Next

- Per-window contribution to delta cosine (which windows drive agreement vs divergence)
- Bootstrap confidence intervals on sample-level separation
- Substrate-stratified expected profiles
- Multi-dataset comparative view
