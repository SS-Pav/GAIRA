# BUILD REPORT — Tab 2 v2 · Hierarchical Representation

**Date:** 2026-04-26
**Phase:** GAIRA_COMMAND_CENTER_STREAMLIT_V2_TAB2_HIERARCHICAL_REPRESENTATION
**Decision:** SHIPPED

---

## What changed (v1 → v2)

The v1 Tab 2 led with molecule-level UMAPs. v2 reorders the entire tab so the **family-first** narrative is unmissable: spectral motifs → 11 BSV families → MSS analyte substructure → ambiguity / collision maps → hybrid evidence flow.

### Final Tab 2 storyline

```
A · Representation hierarchy        — 5-step conceptual diagram
B · Family-first motif map (G01-G11) — primary view, family ellipses + labels
C · MSS / motif substructure         — secondary, label-thinned, optional dual
D · Annotated dendrograms            — image + interpretation + cluster table
E · BSV saliency · band ⇒ axis      — heatmap + axis-selector top-bands table
F · Shared bands & ambiguity         — green/orange/red traffic-light bars
G · Axis overlap network             — circular graph + curated edge table
H · Hybrid BSV evidence flow         — 7-node Plotly diagram + supporting figs
↓
Tab 3 link card (molecule explorer moved out)
```

The molecule / MSS explorer skeleton is removed from Tab 2 entirely.

## Files changed / created

| file | change |
|---|---|
| `components/motif_mss_bsv_tab.py` | full v2 rewrite — 8 sections + sidebar controls |
| `utils/plotly_cluster_utils.py` | NEW — convex hull + covariance ellipse + colour maps + stable BSV family palette |
| `utils/bsv_saliency_utils.py` | NEW — top-bands-per-axis, axis overlap edges, traffic-light overlay, curated edge interpretations |
| `utils/embedding_loader.py` | extended — `attach_bsv_family`, `family_name_lookup`, `family_short_lookup` |
| `components/ui_blocks.py` | unchanged from v1 (already had `cluster_card` + `interpretation`) |
| `streamlit_apps/gaira_command_center/BUILD_REPORT_v2_tab2_hierarchical_representation.md` | NEW |

GAIRA core untouched.

## Figures upgraded / added

| section | figure | what is new |
|---|---|---|
| A | Representation hierarchy diagram | NEW — 5 boxed nodes + arrows + hover descriptions |
| B | **Family-first map** | NEW core figure — `attach_bsv_family` joins 236 analytes to G01-G11; per-family covariance ellipses (or convex hulls); centroid label `G05 · glycan_carbohydrate`; family-detail card lists representative analytes + top bands + ambiguity partners pulled from saliency |
| C | MSS / motif substructure | reframed as secondary; only top-N (default 10) classes labelled; optional dual MSS↔Motif comparison via toggle |
| D | Dendrogram section | image kept + cut-level interpretation panel; new cluster-summary table with purity → caveat mapping (`high purity` / `moderate — within-family chemistry mixed` / `low purity — shared chemistry across families`) |
| E | BSV saliency heatmap | **canonical band labels OFF by default** — replaced with subtle dashed guidelines; toggle reveals 8 stagger-positioned labels; new **axis-selector top-bands table** (top 10 bands with weight + top contributors per axis) |
| F | Shared bands ambiguity | **traffic-light colours**: green = 1 axis (clean-ish), orange = 2 axes (shared), red = ≥3 (collision); manual legend; canonical-region annotations (725 purine, 1080 glycan/phosphate, 1450 CH bend, 1655 amide/lipid) |
| G | **Axis overlap network** | NEW — circular layout with G01-G11; node size = bands ≥ threshold owned by axis; node colour = family palette; edge width / opacity = # shared bands; hover with sample shared bands + curated interpretation; side table of top-12 axis-pair overlaps with risk tier (`HIGH` / `MODERATE` / `LOW`) |
| H | Hybrid BSV flow | **7 nodes** (added Pre-processing + Confidence/ambiguity nodes); cleaner spacing; edge weight labels `0.25·motif` / `0.75·MSS`; supporting confusion + confidence figures collapsed under an expander |

## Sidebar — clean default + advanced

The default sidebar exposes only the controls the family-first view needs:

- **Family-first map**: embedding base (MOTIF / MSS), colour by (family / class), family hulls/ellipses toggle, overlay kind (ellipse / hull), opacity slider, family-detail selector.

Everything else lives under an `Advanced controls` expander:
- Substructure (MSS/MOTIF mode, label top-N, label toggle, opacity, dual comparison)
- Saliency + ambiguity (band bin, canonical labels toggle, axis selector, threshold)
- Dendrograms (cluster summary mode)

This matches the *"default view should be clean; advanced toggles hidden"* instruction.

## Artifacts used

| artifact | role |
|---|---|
| `gaira_representation_cluster_analysis_v1/tables/{mss,motif}_analyte_embedding_v1.csv` | UMAP base for sections B + C |
| `gaira_representation_cluster_analysis_v1/tables/{mss,motif}_cluster_breakdown_v1.csv` | dendrogram cluster summary table (D) |
| `gaira_representation_cluster_analysis_v1/figures/fig_{mss,motif}_dendrogram_v1.png` | dendrogram images (D) |
| `gaira_base_4_mss_core_build_v1/registry/grounding_molecular_signatures_v4.csv` | source of band positions for saliency (E/F/G) and hover anchors (B/C) |
| `gaira_base_4_hybrid_bsv_build_v1/tables/analyte_to_hybrid_group_map_v1.csv` | analyte → BSV family join (B/C) |
| `gaira_base_4_hybrid_bsv_build_v1/tables/hybrid_bsv_group_registry_v1.csv` | full family names for axis labels (B/E/G/H) |
| `gaira_base_4_hybrid_bsv_build_v1/figures/fig_hybrid_family_confusion_heatmap_v1.png` | supporting figure under hybrid flow (H) |
| `gaira_base_4_hybrid_bsv_build_v1/figures/fig_hybrid_confidence_vs_accuracy_v1.png` | supporting figure under hybrid flow (H) |

All loads cached via `@st.cache_data`.

## Computed only for visualization

- **Per-family ellipses / hulls** in the family-first map (covariance ellipse default; hull optional). Visualisation-only — never persisted.
- **Axis overlap edges + node weights** derived from the saliency matrix at the user-selected threshold (`axis_overlap_edges`, `axis_node_weights`). Visualisation-only.
- **Traffic-light counts** (`traffic_light_overlay`) for the ambiguity bars. Visualisation-only.
- **Top-bands-per-axis table** under the saliency heatmap is derived from the same cached saliency matrix.

No GAIRA scoring rerun. No persisted state outside the manifest YAML.

## Acceptance check

| criterion | result |
|---|---|
| Tab 1 unchanged + still works | ✅ no edits to `overview_tab.py` |
| Tab 2 loads | ✅ live HTTP 200 on `/` and `/_stcore/health`; clean log |
| First major plot is family-first G01–G11 view | ✅ section B, expanded by default |
| MSS / Motif UMAPs still available but secondary | ✅ section C, behind expander (open by default but clearly labelled "within-family detail") |
| Dendrograms have interpretation | ✅ cut-level explainer + cluster-summary table with caveat column |
| Saliency labels readable / hidden by default | ✅ canonical-band text labels OFF by default; subtle dashed guidelines remain; toggle in advanced controls |
| Shared-bands plot marks clean vs collision-prone | ✅ green / orange / red colour map + manual legend |
| Axis overlap network renders | ✅ 11-node circle, 25+ edges, side table of top pairs |
| Molecule explorer skeleton removed from Tab 2 | ✅ replaced with disabled card pointing to Tab 3 |
| Modules import cleanly | ✅ `utils.plotly_cluster_utils`, `utils.bsv_saliency_utils`, `components.motif_mss_bsv_tab`, `app` |
| Render path executes under stubbed Streamlit | ✅ 7 Plotly figures (97 traces, 48 annotations), 3 dataframes, 4 image fallbacks, 8 expanders |

## Final-success-standard answers

A new viewer should be able to answer in 30 seconds:

1. **What are the 11 GAIRA biochemical axes?** → Section A hierarchy + section E heatmap with full axis names (G01 · purine_nucleotide … G11 · metabolic_small_molecule).
2. **Where do they live in spectral space?** → Section E saliency heatmap rows.
3. **Which axes overlap?** → Section G overlap network (edge thickness = # shared bands).
4. **Why does MSS help disambiguate?** → Section F traffic-light bars: red bars = collision; section G interpretation panel.
5. **Why are molecule calls caveated in biofluids?** → Final interpretation panels + Tab 3 link card.

## Remaining limitations

- Dendrograms are still pre-rendered PNGs. A native Plotly dendrogram via `scipy.cluster.hierarchy` + `plotly.figure_factory.create_dendrogram` is feasible but the current images are publication-grade and the text interpretation panel + cluster table cover the missing interactivity.
- Axis-overlap network uses a fixed circular layout. Force-directed (e.g. `networkx.spring_layout`) would emphasise structure better — left as a v3 enhancement.
- The family-first overlay is computed from the existing UMAP embedding; if the embedding is rerun with different hyperparameters, the visual hulls / ellipses will move. The interpretation text is stable to that.
- Substructure cluster labels are placed at `broad_class` centroids; in dense regions they can still overlap. Mitigated by `Label top-N classes` slider (default 10).

## Strict invariants preserved

- GAIRA core unchanged.
- No GAIRA scoring rerun.
- All loads path-configurable + cached.
- Missing-artifact tolerance: every loader returns `None` and the renderer surfaces a soft `gaira-warn` card instead of crashing.
- All visualisation-only computations (ellipses, network edges, traffic-light counts) are explicitly marked in code + report.
- Family-first → MSS → ambiguity → hybrid-flow ordering keeps the chemistry first; embeddings are second.
