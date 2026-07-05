# BUILD REPORT — GAIRA Context Graph Explorer · v2

**Date:** 2026-04-26
**Phase:** GAIRA_CONTEXT_GRAPH_EXPLORER_STREAMLIT_V2
**Decision:** SHIPPED · v1 untouched · all 7 tabs render

---

## Files created

```
streamlit_apps/gaira_context_graph_explorer_v2/
├── BUILD_REPORT_context_graph_explorer_v2.md    # this file
├── README.md
├── app.py
├── assets/README.md
├── cache/                                       # auto-generated on first run
├── components/
│   ├── __init__.py
│   ├── condition_axis_graph.py                  # Tab 2
│   ├── condition_specific_graphs.py             # Tab 3 (NEW)
│   ├── context_embedding_tab.py                 # Tab 7
│   ├── emergent_paths.py                        # shared helper
│   ├── hierarchical_context_graph.py            # Tab 4
│   ├── mss_transfer_graph.py                    # Tab 5 (3 sub-tabs)
│   ├── overview.py                              # Tab 1
│   ├── sample_type_comparison.py                # Tab 6 (NEW)
│   └── ui_blocks.py
├── config/
│   ├── app_config.yaml                          # paths + palettes + dataset short labels
│   └── condition_mapping.yaml                   # 27 condition rules
└── utils/
    ├── __init__.py
    ├── condition_mapper.py                      # NEW — condition_A → specific_condition
    ├── graph_builders.py                        # bipartite + Sankey edge builders
    ├── load_context_data.py                     # cache builder + cached loaders
    └── plotly_graph_utils.py                    # bipartite, Sankey, heatmap renderers
```

19 source files. v1 was not touched.

## Data sources loaded

Primary: **all 15 v1 discovery tables** under
`/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_context_graph_discovery_v1/tables/`.

Secondary (live, lazy reads):
- `gaira_base_4_shine_ev_gaira_pilot_v1/tables/shine_cohort_bsv_means_v1.csv`
  — used by Tab 3 to render the hepatotoxicity dose trajectory
- `gaira_base_4_small_ev_shared_structure_pass_v2/tables/axis_rank_comparison_v2.csv`
  — used by Tab 3 for small-EV probe-axis comparison

## Condition labels recovered (27 specific labels, 100% mapping)

| label | n events | sample type |
|---|---:|---|
| hepatotoxicity_mss_layer | 376 | EV |
| HCC | 115 | serum |
| Tube_QC_caution | 110 | serum |
| CCA | 108 | serum |
| P2_HCC_vs_NC | 99 | serum |
| P2_CCA_vs_NC | 99 | serum |
| P2_LM_vs_NC | 99 | serum |
| LM | 88 | serum |
| P1_HCC_vs_CTR | 88 | serum |
| P1_HCC_holdout | 88 | serum |
| diabetes_classifier | 80 | EV |
| COVID_vs_Healthy | 77 | serum |
| Suspected_vs_Healthy | 77 | serum |
| COVID_vs_Suspected | 55 | serum |
| P2_CCA_vs_LM | 55 | serum |
| P2_LM_vs_HCC | 55 | serum |
| P2_CCA_vs_HCC | 55 | serum |
| hepatotoxicity_classifier | 44 | EV |
| hepatotoxicity_D2_C40_vs_C0 | 44 | EV |
| cross_pilot_consensus | 31 | mixed |
| hepatotoxicity_D1 | 22 | EV |
| hepatotoxicity_D0 | 22 | EV |
| OWD | 17 | EV |
| calibration_self | 11 | mixed |
| NWD | 7 | EV |
| smallEV_Probe2 | 3 | EV |
| smallEV_Probe1 | 3 | EV |

**1928 / 1928 events mapped (100%).** No "unmapped" residue.

## New tables generated (auto-cached)

```
cache/condition_axis_edges_broad.csv         # 11 broad families × 11 axes
cache/condition_axis_edges_specific.csv      # 27 specific cohorts × 11 axes
cache/condition_mss_edges.csv                # specific × MSS
cache/condition_specific_events.csv          # events_long + specific_condition column (681 KB)
cache/context_embedding_points_v2.csv        # datasets + short_label + caveat_burden + evidence_count
cache/emergent_paths_ranked.csv              # top 50 paths sample→dataset→cond→axis
cache/sample_type_axis_summary.csv           # sample_type × axis recurrence + direction
cache/sample_type_mss_summary.csv            # sample_type × top-25 MSS
```

## Graphs implemented

| tab | graph | type |
|---|---|---|
| 1 | Headline metric cards + 27-row condition coverage table | text · table |
| 2 | Condition × axis bipartite (broad / specific toggle) | Plotly Scattergl bipartite |
| 3 | BSV bar / radar profile (with optional reference overlay) | Plotly Bar / Scatterpolar |
| 3 | Cohort-restricted hierarchical Sankey | Plotly Sankey |
| 3 | Hepatotoxicity dose trajectory (D0 / D1 / D2 selector) | Plotly line per axis × set |
| 4 | Full 5-layer hierarchical Sankey + Top emergent paths table | Plotly Sankey + dataframe |
| 5 | All / EV-only / Serum-only MSS bipartite (3 sub-tabs) | Plotly bipartite × 3 |
| 6 | Axis recurrence heatmap | Plotly Heatmap (Blues) |
| 6 | Direction consistency heatmap | Plotly Heatmap (Greens, 0–1) |
| 6 | MSS recurrence heatmap | Plotly Heatmap (Oranges) |
| 6 | Per-axis EV vs serum contrast table | dataframe |
| 6 | Paired EV vs serum mean-effect scatter (G01–G11 as points) | Plotly Scatter |
| 7 | UMAP / PCA dataset embedding with colour modes + size modes + hull/ellipse overlays + nearest-neighbour table | Plotly Scatter + dataframe |

Stubbed-render trace: **11 Plotly figures · 7 dataframes · 0 warnings · 10 tabs (7 main + 3 MSS sub-tabs).**

## Acceptance check

| criterion | result |
|---|---|
| v1 unchanged | ✅ no edits to `gaira_context_graph_explorer/` |
| v2 launches | ✅ `streamlit run` HTTP 200 on `/` and `/_stcore/health` (port 8771); zero errors / tracebacks |
| All 7 tabs load | ✅ smoke render exercised every tab + 3 MSS sub-tabs |
| Serum cohorts visible separately | ✅ HCC / CCA / LM / COVID_vs_Healthy / P1_HCC_vs_CTR / P2_HCC_vs_NC / Tube_QC_caution / etc. as first-class condition labels |
| EV cohorts visible separately | ✅ OWD / NWD / hepatotoxicity_D0 / D1 / D2_C40_vs_C0 / smallEV_Probe1 / Probe2 / hepatotoxicity_mss_layer / hepatotoxicity_classifier / diabetes_classifier |
| MSS transfer viewable EV-only and serum-only | ✅ Tab 5 has dedicated sub-tabs |
| Context embeddings have short labels | ✅ `dataset_short_labels` config maps 15 datasets → human-readable names; full names always available in hover |
| Sankey has top-paths table | ✅ Tab 4 ranks paths by `n_events × |effect| × consistency`, surfaces `confidence_tier` |
| No graph is unreadable by default | ✅ bipartite default top-N=40, Sankey default max_edges_per_layer=60, MSS default top_n=25; sliders to adjust |
| Missing-file tolerance | ✅ every loader returns `None` and the renderer surfaces a soft `cge-warn` card |
| Cache builds on first launch | ✅ 8 cache files written under `cache/` (verified after first boot) |

## Missing artifacts

None blocking. The two **live secondary reads** (`shine_cohort_bsv_means_v1.csv`,
`axis_rank_comparison_v2.csv`) both exist; if they go missing in the
future, Tab 3 falls back to a "(table not found)" caption rather than
crashing.

## Limitations

- The hepatotoxicity trajectory uses `mean_clr` from the cohort-BSV-means
  table directly. If the SHINE pilot is re-run with a different output
  schema, that read needs updating.
- The "Top emergent paths" ranking is a heuristic
  (`n_events × |effect| × direction-consistency`), not a statistical test.
- The condition mapper's regex order matters — labels are matched
  first-wins. The current order prioritises P1/P2 prefixed liver-cancer
  cohorts over generic HCC/CCA/LM, then falls back to dataset-level
  catch-all rules.
- Small-EV currently surfaces as "Probe1 / Probe2" — finer-grained
  HT-1080 / THP-1 mixture ratios would need an upstream parser change in
  the v1 discovery driver (the small-EV pilot's events use probe IDs as
  the comparison key).

## Recommended figures for the GAIRA main demo

1. **Tab 6 · Paired EV-vs-serum mean-effect scatter** — single canvas that
   answers "do EV and serum agree on which BSV axes shift?" with each
   axis as one point and quadrants encoding agreement.
2. **Tab 4 · Hierarchical Sankey** — the canonical
   sample → dataset → cohort → axis → MSS picture; pair with the
   Top-emergent-paths table for the demo voiceover.
3. **Tab 3 · Hepatotoxicity trajectory** — the only directly-quantitative
   dose-response figure available; Day-2 highlighted as paper-relevant.
4. **Tab 6 · Direction-consistency heatmap** — single most useful
   "which axes are *reproducibly* up vs down across pilots" snapshot.
5. **Tab 2 · Specific-granularity bipartite** — drives home that
   different cohorts run different biochemical programs (vs the
   averaged-out broad-family view).

## Strict invariants preserved

- GAIRA core unchanged.
- No GAIRA scoring rerun inside the app.
- v1 explorer untouched (verified).
- Missing-artifact tolerance throughout.
- All 27 condition labels are derived from `condition_A` patterns +
  dataset hints + comparison_type fallbacks; the mapping is open in
  `config/condition_mapping.yaml` for inspection / extension.
- All visualisation-only computations (bipartite positions, Sankey layer
  bookkeeping, hull/ellipse overlays) live in `utils/` and are
  explicitly named.
