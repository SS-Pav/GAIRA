# BUILD REPORT — GAIRA Context Graph Explorer v1

**Date:** 2026-04-26
**Phase:** GAIRA_CONTEXT_GRAPH_EXPLORER_STREAMLIT_V1
**Decision:** SHIPPED

---

## Files created

```
streamlit_apps/gaira_context_graph_explorer/
├── BUILD_REPORT_context_graph_explorer_v1.md   # this file
├── README.md
├── app.py
├── assets/
│   └── README.md
├── components/
│   ├── __init__.py
│   ├── condition_axis_graph.py        # Tab 2
│   ├── context_embedding_tab.py       # Tab 5
│   ├── evidence_tables.py             # raw events expander
│   ├── hierarchical_context_graph.py  # Tab 3
│   ├── mss_transfer_graph.py          # Tab 4
│   ├── overview_tab.py                # Tab 1
│   └── ui_blocks.py                   # cards / metrics / interpretation
├── config/
│   └── app_config.yaml                # paths + colour palettes
└── utils/
    ├── __init__.py
    ├── graph_builders.py              # bipartite + Sankey edge builders
    ├── load_context_data.py           # safe CSV/text loaders + cache
    └── plotly_graph_utils.py          # Plotly figure renderers
```

15 files in total (3 utils + 7 components + 2 configs/assets + app + 2 docs).

## Data tables loaded

All 15 expected tables resolve successfully (verified at first launch):

| key | file | rows |
|---|---|---:|
| events | gaira_evidence_events_long.csv | 1928 |
| nodes | context_graph_nodes.csv | 53 |
| edges | context_graph_edges.csv | 197 |
| axis_transfer | axis_transfer_scores.csv | 11 |
| mss_transfer | mss_transfer_classification.csv | 18 |
| sample_axis | sample_type_axis_recurrence.csv | small |
| cf_axis | condition_axis_motif_recurrence.csv | 83 |
| emergent | emergent_behavior_metrics.csv | 11 |
| findings | top_emergent_findings.csv | 23 |
| caveats | caveat_recurrence.csv | 57-dataset spread across 6 categories |
| dataset_features | context_dataset_bsv_features.csv | 13 (× 11 axes + meta) |
| clusters | context_cluster_assignments.csv | 13 |
| axis_neighborhood | axis_neighborhood_summary.csv | 11 |
| ctx_dependence | context_dependence_scores.csv | small |
| inventory | context_graph_artifact_inventory.csv | 2168 |

The pre-rendered HTML figures (`context_graph_global.html`,
`context_embedding_dataset.html`) are surfaced as optional collapsed
embeds in their respective tabs.

## Graphs rendered

| tab | visual | type |
|---|---|---|
| 1 | Headline-metric cards + axis/MSS/caveat tables + REPORT preview | text · tables |
| 2 | **Condition → Axis bipartite network** | Plotly Scattergl bipartite (left = condition families · right = G01-G11) with weight-thresholded edges, colour-by-direction, hover tooltips |
| 3 | **5-layer Sankey** (sample_type → dataset → condition_family → BSV axis × direction → MSS candidate) | Plotly Sankey with `arrangement="snap"`, MSS-layer toggle, max-edges-per-layer cap |
| 4 | **MSS transfer bipartite** (MSS ↔ {dataset / condition_family / sample_type}) | Plotly Scattergl bipartite + classification filter + side classification table |
| 5 | **Context embedding** (UMAP / PCA over 11-dim BSV-effect vector per dataset) | Plotly Scattergl with colour modes (sample_type / condition_family / caveat burden) and size modes (fixed / evidence count / caveat burden) + embedded HTML preview |

Stubbed-render trace: 4 Plotly figures (the default panes — additional
figures appear when the user toggles tab options) · 6 dataframes ·
3 expanders · 0 warnings.

## Acceptance check

| criterion | result |
|---|---|
| All modules import cleanly | ✅ utils.* + components.* + app |
| Live `streamlit run` boots | ✅ HTTP 200 on `/` and `/_stcore/health` (port 8770); zero errors / tracebacks in log |
| All 5 tabs load | ✅ smoke render exercised every tab via stub |
| Missing files do not crash | ✅ `load_csv_safe` returns `None`; renderers fall back to soft `cge-warn` cards |
| Condition-axis graph renders | ✅ Tab 2 |
| Hierarchical Sankey renders | ✅ Tab 3 |
| MSS transfer graph renders | ✅ Tab 4 |
| Context embedding renders | ✅ Tab 5 |
| Filters work | ✅ multiselect + slider + radio controls wired through `build_*` helpers |
| Labels readable | ✅ left-side labels for bipartite charts, full names in hover |

## Missing artifacts

None. All 15 expected tables and both expected pre-rendered HTML figures
exist at the configured `context_root`.

## Strongest graph for demo

**Tab 3 — Hierarchical Sankey** is the clearest single answer to "what
does GAIRA do across pilots?" — it shows the full sample → dataset →
condition → BSV axis → MSS candidate flow on one canvas with edge widths
proportional to evidence count and edge colours encoding direction.

For a demo headline, **Tab 2 — Condition → Axis bipartite** is the next
strongest because it directly answers "which biochemical axes recur in
which clinical contexts?" and supports interactive filtering by
direction (up / down / stable).

## Recommended next improvements

1. **Sankey colour-by-confidence overlay** — use `confidence_tier` from
   the events table to colour edges by STRONG / MODERATE / WEAK, not just
   direction.
2. **Per-axis neighbourhood drilldown tab** — click a G-axis label and
   open a focused view of every dataset / MSS / caveat touching that
   axis. We have `axis_neighborhood_summary.csv` already; just needs a
   selector + small view.
3. **Caveat overlay on the Sankey** — annotate datasets with non-zero
   caveat-mention counts so substrate-sensitive / QC-flagged pilots are
   visible at a glance.
4. **Cross-pilot statistical panel** — for each TRANSFERABLE MSS
   candidate, run a tiny meta-analysis (mean direction-weighted effect
   across pilots) and surface as a card; would benefit from an
   inverse-variance weighting if per-pilot SDs are present.
5. **Snapshot export button** — write the current bipartite / Sankey
   figure to PNG so it can be dropped into the demo deck.

## Strict invariants preserved

- GAIRA core unchanged.
- No GAIRA scoring rerun inside the app.
- All loads are path-configurable via `config/app_config.yaml`.
- Missing-artifact tolerance verified (loader returns `None` + soft warning).
- All visualisation-only computations (bipartite positions, edge weights,
  Sankey layer bookkeeping) live in `utils/` and are explicitly named.
