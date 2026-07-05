# BUILD REPORT — v2 Tab 2 FIX · pass 3

**Date:** 2026-04-26
**Phase:** GAIRA_COMMAND_CENTER_V2_TAB2_FIX_PASS3
**Decision:** SHIPPED

---

## What this is

Pass 3 fixes the readability + scientific-flow issues raised after pass 2:
1. Artificial chemistry-grouped centroid map → real motif/family UMAP.
2. Unclear MSS clusters → annotated MSS UMAP with selectable colour modes
   and a dominant-class summary table.
3. Drilldown stuck on G05 → inline dropdown covering all G01–G11.
4. Confusing orange "most-frequent anchor" plot → demoted to an opt-in
   expander with explanatory caveat.
5. Sparse axis-overlap heatmap → confusion-matrix-style Blues heatmap
   with row-normalised + absolute toggle, network demoted to expander.

v1 is **untouched**.

## What was removed

| removed | replacement |
|---|---|
| `render_motif_family_cluster_map()` (chemistry-grouped centroids + Fibonacci jitter) | `render_normal_motif_family_umap()` — real motif UMAP coloured by `primary_group` |
| `render_axis_overlap_matrix()` (sparse Oranges heatmap) | `render_axis_overlap_confusion_style()` — Blues heatmap, row-normalised by default |
| Default-visible orange band-frequency bar chart in drilldown | Moved into `Band frequency within this family (advanced)` expander with caveat: *low counts reflect diverse analyte-specific anchors* |
| Sidebar `drilldown_family` dropdown that defaulted to G05 and lived in the wrong place | Inline dropdown inside section D, default G01, all 11 families selectable |

## What was restored / newly added

| section | function | change |
|---|---|---|
| B | `render_normal_motif_family_umap()` | NEW — full-width Plotly Scattergl using actual motif `umap_1`/`umap_2` columns; one trace per family for a clean 11-entry legend; centroid labels with manual stagger to avoid collisions; optional convex-hull toggle (default OFF) |
| C | `render_mss_analyte_cluster_map()` | three colour modes via sidebar radio (`biochemical class` / `BSV family` / `cluster id (precomputed)`); cluster annotations now use the precomputed `cluster_id` and label up to 10 clusters with ≥ 5 members using their **dominant biochemical class**; new MSS cluster summary table below with `cluster_id · n · dominant class · representative analytes · likely BSV family` |
| D | `render_mss_within_family_drilldown()` | inline dropdown covers ALL G01..G11; subset UMAP highlights selected family on grey background; analyte table with `analyte · biochemical class · top anchor bands · support bands · competitors · reliability`; band-frequency plot moved to opt-in expander |
| G | `render_axis_overlap_confusion_style()` | NEW Blues heatmap with two modes: row-normalised (default; diagonal = 1.0; off-diagonal = fraction of axis A bands also owned by axis B) or absolute count (diagonal = bands axis owns); cell annotations only above 0.15 (normalised) / 2 (absolute) to declutter; full hover with axis A/B own-band counts and shared count; curated top-10 axis-pair side table with risk tier; network demoted to `Show network view` expander with only top-8 strongest edges |

## Are all 11 families selectable in the drilldown?

**Yes.** The inline dropdown is built from `BSV_AXES_ORDER` (G01..G11) and uses `family_short_lookup()` for full labels. Even families with **0 mapped analytes** in the current corpus are selectable — selecting one shows a soft info card explaining the family is real but has no member analytes in the embedding (rather than crashing or hiding it).

Distribution in the current corpus (from `analyte_to_hybrid_group_map_v1.csv`):
G11=82 · G06=31 · G09=27 · G05=24 · G08=23 · G07=17 · G10=14 · G04=8 · G02=5 · G03=3 · G01=2 → all 11 are non-empty in this corpus.

## Clustering method for MSS annotation

No new clustering was run. The MSS cluster annotations use the **precomputed `cluster_id`** column from `gaira_representation_cluster_analysis_v1/tables/mss_analyte_embedding_v1.csv` (precomputed agglomerative clustering, 11 clusters). The summary table joins these with the precomputed per-cluster breakdown in `mss_cluster_breakdown_v1.csv` (dominant_broad_class, n_members, sample_members, purity, entropy_bits) and adds a derived `likely BSV family` column by mapping `dominant_broad_class → primary_group` via the analyte→group map.

The `cluster id (precomputed)` colour mode in the MSS map uses the same column. No HDBSCAN / KMeans was run inside the app — clarity was achieved by surfacing existing cluster IDs rather than re-clustering.

## Section order (final)

```
A · Representation hierarchy
B · Motif / family UMAP                    (real embedding · coloured by G01–G11)
C · MSS analyte cluster map                (annotated clusters + summary table)
D · MSS-within-family drilldown            (inline dropdown · all 11 families)
E · BSV saliency map                       (clean defaults preserved)
F · Shared-band ambiguity map              (traffic-light)
G · Axis overlap confusion-style matrix    (Blues; network optional)
H · Hierarchical clustering support        (expander)
I · Hybrid evidence flow                   (expander)
```

## Sidebar (default + advanced)

**Default:**
- Family UMAP: centroid-label toggle, hull toggle, opacity slider
- MSS map: colour mode (class / family / cluster), cluster-label toggle, legend toggle, opacity slider
- Saliency: band bin, ambiguity threshold
- Axis overlap matrix: row-normalised vs absolute count

**Advanced (collapsed):**
- Saliency canonical band labels
- Dendrogram cluster-summary mode (MOTIF / MSS)

## Files changed (v2 only)

| file | change |
|---|---|
| `streamlit_apps/gaira_command_center_v2/components/motif_mss_bsv_tab.py` | rewrite — replaced 4 functions, added `_stagger_label`, `_convex_hull_pts` helpers |
| `streamlit_apps/gaira_command_center_v2/BUILD_REPORT_v2_TAB2_FIX_PASS3.md` | NEW (this file) |

`utils/*` and `components/ui_blocks.py` did not need changes — pass-2's helpers (`axis_overlap_matrix`, `family_band_frequencies`, `BSV_FAMILY_COLORS`, `MAJOR_CLASS_LABELS`) were sufficient.

`v1/` was not touched.

## Acceptance check

| criterion | result |
|---|---|
| v1 unchanged | ✅ no edits in `gaira_command_center_v1/` |
| v2 launches | ✅ `streamlit run` HTTP 200 on `/` and `/_stcore/health` (port 8766); zero errors / tracebacks |
| Tab 2 begins with real motif/family UMAP | ✅ section B uses actual `umap_1`/`umap_2` from `motif_analyte_embedding_v1.csv` |
| MSS cluster map has readable labels + summary table | ✅ up to 10 dominant-class labels with collision-staggering; cluster summary table below |
| MSS-within-family dropdown works for all G01–G11 | ✅ inline `st.selectbox` over `BSV_AXES_ORDER`; default G01 (not G05) |
| No default G05-only plot | ✅ band-frequency plot is now opt-in expander; sidebar default family-selector removed |
| Axis overlap appears as confusion-style matrix by default | ✅ Blues, row-normalised, square 11×11 with diagonal = 1.0 |
| Network is optional expander only | ✅ `Show network view` expander, default closed |
| BSV saliency map remains intact | ✅ section E untouched from pass 2 |

## Remaining limitations

- The motif UMAP positions the user sees are exactly what the precomputed embedding produced. Some families (e.g. G01 with 2 analytes, G03 with 3) have so few members that their centroid label sits right on top of one of the points. The collision-stagger softens this but cannot fully fix tiny clusters.
- The MSS "likely BSV family" column in the cluster summary uses *dominant-class → most-common-family* (mode of `primary_group` over the dominant class). For low-purity clusters this can be slightly misleading; the `caveat` column on the dendrogram cluster table flags low-purity clusters explicitly.
- The confusion-matrix mode is not symmetric in row-normalised mode (M[i,j] = fraction of axis A's bands shared with axis B; M[j,i] differs whenever axes A and B own different total band counts). This is a feature, not a bug — it matches the asymmetric "confusion" reading.
- Dendrograms remain pre-rendered PNGs.

## Next step

Tab 3 — grounding molecule explorer. The drilldown table built in section D is a useful preview of the per-analyte data scope Tab 3 will own (full spectrum trace + anchor / support / anti-evidence overlays + BSV radar + per-analyte calibration check).

## Strict invariants preserved

- GAIRA core unchanged.
- No GAIRA scoring rerun.
- All loads cached via `@st.cache_data`.
- Missing-artifact tolerance: every loader returns `None` and the renderer surfaces a soft `gaira-warn` card.
- All visualisation-only computations (cluster annotation labels, overlap matrix, band frequencies) are explicitly marked.
