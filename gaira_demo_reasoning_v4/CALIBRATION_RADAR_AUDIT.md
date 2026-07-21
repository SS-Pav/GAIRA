# Calibration Radar Audit (Parts 2 & 3)

## Original issue

The calibration radars appeared "nearly identical across concentrations and
perturbations," suggesting a stale cache, a wrong inference object, or normalization
suppressing the change.

## Root cause — traced, not assumed

Traced the full chain **raw spectrum → frozen NMF coefficients → MSS → BSV → radar**
for adenine, ergothioneine and uricase. The radar arrays **genuinely differ** across
doses:

- adenine purine composition **0.183 → 0.320** (max theme change 0.137, low→high dose)
- uricase purine **after − before = −0.011**
- biological cohorts also differ (diabetes maxΔtheme 0.052)

So it is **not** a cache bug and **not** the wrong object. The absolute composition
radar merely *looks* unchanged because the 11 theme shares sum to ~1 (compositional
closure): the large baseline structure dominates the polygon and a purine rise from
0.18→0.32 compresses the other axes only slightly. A test now asserts different doses
produce different radar arrays.

## Corrected computation

The frozen BSV/engine were **not** modified (no implementation bug was present). The
fix is additive visualization:

1. **Delta radar is now the default.** `ΔBSV(dose) = BSV(dose) − BSV(baseline)` on a
   **shared, centred** scale across the experiment (zero at the mid-ring, lobes out =
   increase, in = decrease). No per-sample min-max rescaling. For uricase,
   `ΔBSV = after − before`. The absolute composition radar is kept in an expander,
   clearly labelled compositional.
2. **Shared symmetric scale**: `±max|Δ|` over all doses in the experiment; no clipping.
3. **Debug table** of the numeric absolute + delta radar values at the selected dose.

## Before/after numerical example (adenine, top dose vs 0)

| theme | absolute (looks flat) | Δ vs baseline (delta radar) |
|---|---|---|
| nucleic_purine | 0.32 | **+0.137** |
| sterol_membrane | ~0.09 | +0.026 |
| lipid_acyl | ~0.06 | −0.026 |
| organic_acid | ~0.05 | −0.023 |

The delta radar makes the purine spike unmistakable; the absolute radar hides it.

## Trajectory / mechanism (Part 3)

Added deterministic mechanism metrics instead of relying on unsupervised PCA of the
11-theme BSV:

- **Redistribution index** `R(d) = 1 − cos(componentₖ, component₀)`. Adenine R(max) =
  **0.46**, cos-to-baseline falls to **0.54** ⇒ component REDISTRIBUTION.
- **Scaling metric** (cos-to-baseline, dominant share). Ergothioneine R(max) = **0.12**,
  cos-to-baseline stays **0.88** ⇒ single-motif SCALING.
- **Interpretable-axis trajectory**: target motif vs the top other-moving motif
  (chemically meaningful axes), not a PCA scatter.
- The BSV-space PCA trajectory is retained but explained: PC1 ≈ 99% for adenine (an
  essentially 1-D dose axis), consistent with BSV validation's low effective
  dimensionality; the PC2 wobble is captioned as noise, not smoothed away.

## Tests

`test_calibration_radars_differ_across_doses`, `..._delta_direction_and_mechanism`,
`test_uricase_delta_radar_purine_decreases` assert: radars differ across doses; the
delta radar is exactly zero at baseline and non-zero at top dose; the target theme
moves in the correct direction (adenine purine ↑, ergothioneine sulfur ↑, uricase
purine ↓); and adenine redistributes more than ergothioneine scales.
