# BUILD REPORT — Tab 2 Visual Intelligence Upgrade v1

**Date:** 2026-04-26
**Phase:** GAIRA_STREAMLIT_TAB2_UPGRADE_V1
**Decision:** SHIPPED

---

## Goal

Convert Tab 2 from a static-figure dump into an **interactive, annotated, scientifically interpretable visual system**. Every panel must answer *"what did GAIRA actually learn from Raman data?"* — not "here are some embeddings."

## What was upgraded

### 1 — Interactive UMAP cluster view (`render_umap_cluster_view`)

| before | after |
|---|---|
| Static matplotlib UMAP, no interactivity | Plotly `Scattergl` UMAP with zoom / pan / hover |
| Two side-by-side `.png` images | MSS ↔ Motif toggle in sidebar, single full-width canvas |
| No cluster boundaries | Convex-hull polygons per cluster (translucent fill + border) |
| No cluster meaning | Centroid annotation = dominant biochemical class (majority vote) |
| Generic embedding | Per-point hover: `analyte_id`, `broad_class`, `regime`, `support_tier`, `n_spectra`, `cluster_id`, top-3 MSS anchor bands |
| Legend cluttered | One trace per colour group, dark-theme legend, constant-size markers |
| No per-cluster summary | Per-cluster cards with members / purity / entropy / Raman-vs-SERS / sample analytes |

Sidebar controls: embedding (MSS / MOTIF), colour-by (class / cluster), cluster-hulls toggle, cluster-labels toggle, marker-opacity slider.

Below every UMAP: a blue interpretation panel explaining what the clusters *mean* biochemically — that purity < 1.0 indicates **shared chemistry across families**, not a model failure.

### 2 — Side-by-side dual UMAP comparison (`render_dual_umap_comparison`)

- Real Plotly subplot (`make_subplots(1, 2)`) — independent zoom/pan per panel via Plotly defaults; shared colour map across both so the same `broad_class` is the same colour everywhere.
- Hover preserved on both panels; legend shown once on the left and grouped via `legendgroup` so toggling a class hides it on both panels.
- Interpretation panel below: *"MSS compresses motif space into analyte-level structure"* — the canonical reading.

### 3 — Dendrogram panel (`render_dendrogram_panel`)

- Pre-rendered MSS + motif dendrograms loaded side-by-side (the existing PNGs are publication-grade).
- New blue interpretation panel explains the four-tier hierarchy (root: lipid vs non-lipid; tier-2: nucleic vs protein; tier-3: within-family decomposition; leaves: 236 analytes).
- Sidebar toggle to show/hide the hierarchy explainer.

### 4 — BSV saliency map (`render_bsv_saliency_map`) — **NEW core figure**

The conceptual bridge that was missing.

- Plotly `Heatmap`, x = Raman shift (400–1800 cm⁻¹), y = 11 BSV axes (G01 … G11 with full names from `hybrid_bsv_group_registry_v1.csv`), z = per-axis max-normalised band weight.
- Source: parses `shared_core_anchors`, `raman_support_features`, and `sers_support_features` from `gaira_base_4_mss_core_build_v1/registry/grounding_molecular_signatures_v4.csv` (30 broad-class signatures × ~10 bands each), routes each band to the parent BSV group via `analyte_to_hybrid_group_map_v1.csv`, weights anchors 2× supports.
- 8 canonical anchor bands annotated as red vertical dotted lines (725 purine ring, 785 pyrimidine, 1003 Phe, 1080 phosphate/glycan, 1340 glycan/purine, 1450 amide-III/lipid, 1655 amide-I/C=C, 1745 ester C=O).
- Per-cell hover: band centre, weight, top-3 contributing biochemical classes.
- **Shared-band overlay** (sidebar-toggleable): a separate orange bar plot below the heatmap showing how many BSV axes draw evidence ≥ threshold from each band. Bars ≥ 2 mark **collision-prone regions** — exactly where MSS competitor logic earns its keep.
- Sidebar controls: band bin (5/10/20/25/50 cm⁻¹), shared-band overlay on/off, overlap threshold slider.
- Interpretation panel calls out G01↔G02 (purine ring), G04↔G05 (1080 cm⁻¹), G06↔G08 (amide-I / lipid C=C) by name.

### 5 — Hybrid BSV flow diagram (`render_hybrid_bsv_flow_diagram`)

- Replaced the existing static PNG with a **clean Plotly node-and-arrow diagram**: 6 nodes (Raw → Primitives → {Motif, MSS} → Hybrid → 11 BSV axes), 6 directed edges, edge weight labels (`0.25·motif`, `0.75·MSS`).
- Each node is a colour-coded rounded rectangle with a hover tooltip describing what passes through it.
- Auto-discovered confusion-heatmap + confidence-vs-accuracy figures kept below as supporting context (still published artifacts from `gaira_base_4_hybrid_bsv_build_v1/figures/`).

### 6 — Molecule explorer moved to Tab 3 (`render_tab3_link`)

- Old skeleton removed entirely.
- Replaced with a single disabled card: *"Open in Tab 3 — Grounding tests"*, briefly listing what Tab 3 will render (spectrum trace + anchor/support overlays + BSV radar + MSS row + per-analyte calibration check).

### 7 — UX polish

- Every major section is wrapped in a collapsible `st.expander` (UMAP / dual-UMAP / dendrogram / saliency / hybrid-flow); UMAP, dual-UMAP, and saliency open by default.
- Dark-theme palette consistent across heatmap, scatter, bar, and node-diagram (`plotly_dark` template + `#0d1117` paper/plot bg).
- New `ui_blocks.cluster_card()` and `ui_blocks.interpretation()` primitives keep the per-cluster summary and the explainer panels visually distinct from the body.
- Section dividers preserved between every block.

### 8 — Performance

- Every artifact load is `@st.cache_data`-wrapped (`_cached_embedding`, `_cached_breakdown`, `_cached_signatures`, `_cached_amap`, `_cached_bsv_registry`, `_cached_saliency`).
- All scatter plots use `Scattergl` (WebGL-backed) for 236-point sets and any future scaling.
- BSV saliency matrix is computed once per `(build_root, bin)` and cached.

## Final Tab-2 structure

```
1. Concept overview (3 cards)            — Levels 1·2·3 of GAIRA representation
2. MSS evolution (2 cards)               — v4.1 → v4.2 narrative
3. Interactive UMAP (Plotly)             — MSS / MOTIF toggle, hulls, labels, hover, per-cluster cards
4. Side-by-side · MSS vs Motif (Plotly)  — 1×2 subplot, shared colour map, joint legend
5. Hierarchical dendrograms              — pre-rendered images + interpretation panel
6. BSV saliency map (Plotly)             — band ⇒ axis heatmap + shared-band overlay
7. Hybrid BSV flow                       — Plotly node-and-arrow + supporting figures
8. Tab 3 explorer link                   — disabled card pointing to upcoming explorer
```

## Files changed / created

| file | change |
|---|---|
| `streamlit_apps/gaira_command_center/components/motif_mss_bsv_tab.py` | full rewrite — 6 render functions + sidebar controls |
| `streamlit_apps/gaira_command_center/components/ui_blocks.py` | added `cluster_card()`, `interpretation()`, two new CSS classes |
| `streamlit_apps/gaira_command_center/utils/embedding_loader.py` | NEW — embedding / cluster / signature / saliency loaders |
| `streamlit_apps/gaira_command_center/BUILD_REPORT_tab2_upgrade_v1.md` | NEW — this report |

GAIRA core untouched.

## Artifacts used

| artifact | role |
|---|---|
| `gaira_representation_cluster_analysis_v1/tables/mss_analyte_embedding_v1.csv` | MSS UMAP (236 analytes × 9 cols incl. cluster_id) |
| `gaira_representation_cluster_analysis_v1/tables/motif_analyte_embedding_v1.csv` | Motif UMAP (same shape) |
| `gaira_representation_cluster_analysis_v1/tables/mss_cluster_breakdown_v1.csv` | per-cluster purity / entropy / sample members |
| `gaira_representation_cluster_analysis_v1/tables/motif_cluster_breakdown_v1.csv` | same for motif clusters |
| `gaira_representation_cluster_analysis_v1/figures/fig_mss_dendrogram_v1.png` | static dendrogram (kept) |
| `gaira_representation_cluster_analysis_v1/figures/fig_motif_dendrogram_v1.png` | static dendrogram (kept) |
| `gaira_base_4_mss_core_build_v1/registry/grounding_molecular_signatures_v4.csv` | source of band positions for the saliency map |
| `gaira_base_4_hybrid_bsv_build_v1/tables/analyte_to_hybrid_group_map_v1.csv` | broad_class → BSV group routing |
| `gaira_base_4_hybrid_bsv_build_v1/tables/hybrid_bsv_group_registry_v1.csv` | full BSV axis names for y-axis labels |
| `gaira_base_4_hybrid_bsv_build_v1/figures/fig_hybrid_family_confusion_heatmap_v1.png` | supporting context under hybrid flow |
| `gaira_base_4_hybrid_bsv_build_v1/figures/fig_hybrid_confidence_vs_accuracy_v1.png` | supporting context under hybrid flow |

## What was approximated

- **Cluster boundaries** = convex hulls of each `cluster_id` group (the precomputed agglomerative cluster from `*_analyte_embedding_v1.csv`). HDBSCAN was *not* re-run because the precomputed `cluster_id` column already partitions the analytes; the `dbscan_cluster` column is also available but is binary (0/1) so less useful for visualisation. If `cluster_id` is missing for a future embedding the function falls back gracefully (no hulls, no labels — raw scatter still renders).
- **BSV saliency weights** = `|DR|` per band, anchors weighted ×2 vs supports, then per-axis max-normalised. This is a structural saliency derived from the MSS registry — it's faithful to which bands each axis is *defined* on, not a learned attribution. The interpretation panel makes that explicit.
- **Dendrograms** kept as pre-rendered PNGs (publication-grade); a Plotly dendrogram could be wired in later but the static images already convey the hierarchy clearly.

## Acceptance criteria

| criterion | result |
|---|---|
| UMAP is interactive (zoom / hover / toggle) | ✅ Plotly Scattergl + sidebar controls |
| Clusters are visually obvious | ✅ convex-hull polygons per cluster |
| Clusters are annotated with meaning | ✅ centroid label = dominant class + per-cluster card panel |
| BSV saliency map clearly shows band → axis mapping | ✅ heatmap + 8 canonical anchor lines |
| Overlap regions visible | ✅ shared-band overlay bar plot (toggleable) |
| No static matplotlib dependence for **core** visuals | ✅ all core panels are Plotly; only dendrograms are kept as PNGs |
| App still runs if clustering fails | ✅ `_build_umap_figure` operates on raw scatter even when `cluster_id` is degenerate; `_add_cluster_hulls` skips clusters with < 3 points |
| Modules import cleanly | ✅ `utils.embedding_loader`, `components.ui_blocks`, `components.motif_mss_bsv_tab` |
| Render path executes under stubbed Streamlit | ✅ 5 Plotly figures (112 traces total), 4 image fallbacks, 5 expanders, 30 columns |
| Live `streamlit run` boots | ✅ HTTP 200 on `/` and `/_stcore/health`; no errors / tracebacks in server log |

## Screenshots / paths

The app does not write images — it renders Plotly client-side. To capture
screenshots, run:

```bash
streamlit run streamlit_apps/gaira_command_center/app.py
```

then capture each panel from the browser. Source paths for the static images
that *are* loaded into the app:

- `/Volumes/SSD_Rad/GAIRA_BUILD/gaira_representation_cluster_analysis_v1/figures/fig_mss_dendrogram_v1.png`
- `/Volumes/SSD_Rad/GAIRA_BUILD/gaira_representation_cluster_analysis_v1/figures/fig_motif_dendrogram_v1.png`
- `/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_hybrid_bsv_build_v1/figures/fig_hybrid_family_confusion_heatmap_v1.png`
- `/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_hybrid_bsv_build_v1/figures/fig_hybrid_confidence_vs_accuracy_v1.png`

## Strict invariants preserved

- GAIRA core unchanged (no edits under `src/gaira/`).
- No GAIRA scoring rerun inside the app.
- Every load path is configurable.
- App runs even when artifacts are missing — every loader returns `None` and the renderer surfaces a soft `gaira-warn` card instead of crashing.
- Saliency derivation is annotation-only; it never overrides MSS scoring.

## Reading the new Tab 2

Pick any panel and the question to ask is:

> *Does this make me think GAIRA learned a structured biochemical representation, or does it look like another ML embedding?*

If the BSV saliency map's shared-band overlay (orange bars at 720, 1003, 1080, 1340, 1450, 1655) doesn't immediately make the collision story obvious, the upgrade has failed and we revisit. From the stubbed-render trace (112 Plotly traces across 5 figures), every panel is now carrying biochemical signal — not just point clouds.
