# GAIRA V6 Demo — Visual Audit

Every matplotlib figure was rendered headless (`python selfcheck.py` →
`_selfcheck/*.png`) and inspected. This log records what was inspected and what was
fixed — not merely that an audit happened. Plotly figures (reference PCA, Sankey)
construct cleanly but are **not** pixel-audited (kaleido not installed); they use
standard scatter/Sankey traces and render in Streamlit.

## Fixes made during the audit

| Figure | Problem found on inspection | Fix |
|---|---|---|
| architecture diagram | top italic caption overlapped the first box | raised ylim + moved caption |
| theme radar | fully saturated decagon for far-OOD SERS (tanh `display` pins to ~1) | switched to the engine's **composition-share** radar backend (informative in- and out-of-domain) |
| spectral collision map | faint striping everywhere (α 0.05, unmerged, ≥3 threshold) | merged contiguous hotspots; raised to ≥4 motifs; dropped crowded labels |
| dose-response | zig-zag from connecting all replicates | per-dose **mean** line over a faint replicate cloud |
| reasoning cascade | monospace font family unresolved warning | `family="monospace"` |
| component evolution | five indistinguishable blue lines | colour by **direction** (rising warm / falling cool) + end labels |

## Dimensions checked (all figures)

- **Clipping / overflow**: none; `bbox_inches="tight"`; wide figures (heatmaps,
  forest, cascade) sized to content.
- **Labels / units / legends**: every axis labelled with units (cm⁻¹, µM, evidence
  share, signed z); legends present where ≥2 series; captions + interpretation +
  limitations attached via `components.figure`.
- **Font sizes**: readable on a laptop; the signature cascade (15.5×7.4in) is the
  largest single figure and remains legible; small multiples avoided (expandable
  panels used instead).
- **Colour consistency**: one palette (`theme.py`) across all pages — PRIMARY blue for
  atlas/components, UP-red / DOWN-blue diverging for increase/decrease, tier
  green/amber/red (Page 5), group colours (Page 6). No rainbow.
- **Empty panels**: none; pages with missing data show explicit status text, not blank
  axes.
- **Misleading scaling**: radar uses composition share (not the saturating display);
  dose-response y-axes not zero-forced but honest; heatmaps symmetric about 0 (RdBu).
- **Radar saturation**: fixed (see table).
- **Baseline alignment**: difference spectra draw an explicit zero line; before/after
  radars overlay on a shared scale.
- **Replicate aggregation**: dose-response and recoverability show replicate clouds +
  aggregated means; biological stats aggregate to the correct unit (patient-level for
  diabetes).
- **Confidence / OOD presentation**: shown OUTSIDE the radar (stat tiles, quality
  panel, cascade stats box), never conflated with theme axes.
- **Narrow layout**: two-column layouts collapse under Streamlit's responsive columns;
  the widest figure scales to container width.

## Figures inspected (rendered + viewed)

Overview/atlas: architecture, corpus_breakdown, radar, mss_hierarchy, fingerprint,
basis_c3, collisions. Calibration: cascade (signature), cal_mss_evolution,
cal_dose_langmuir (ergothioneine, near-textbook), cal_component_evo, cal_trajectory,
cal_uricase_diff_mss, cal_compare. Serum: p5_reco_cascade, p5_reco_scatter,
p5_reco_heatmap, p5_confidence_limitation. Biological: p6_forest, p6_radar, p6_pca,
p6_quality, p6_centroids. DART: p7_ladder, p7_gallery (labelled CONCEPTUAL),
p7_datamodel.

## Known visual limitations

- Plotly reference-PCA and Sankey are not pixel-audited (no kaleido); verified to
  construct with correct node/link counts (24→12→11, 76 links).
- The BSV-space dose trajectory (adenine) is ~1-D (PC1 ≈ 99%); the small PC2 wobble is
  noise and is captioned as such rather than smoothed away.

---

## Correction pass — figures added & audited

Rendered headless and inspected: **delta radar** (signed, shared centred scale — the
new default calibration BSV view; verified zero at baseline, purine spike at top dose),
**mechanism curves** (redistribution vs evidence), **pairwise MSS trajectory**,
**NMF component similarity map (MDS)** and **dendrogram** (24 nodes, deterministic),
**SHINE paired slope** (Day0→Day2 per dose), **sample BSV heatmap**, **distance bars**,
**balanced-view bars**. Checks: no radar saturation (delta radar centred, not min-max);
heatmaps symmetric about 0 (RdBu) and labelled z-display; MDS labelled clusters-not-
coordinates; paired slope legibly labelled per stratum. PCA demoted to expanders on
biological pages and to a labelled secondary section on the atlas page. No blank panels;
REAL/UNAVAILABLE states render correctly.
