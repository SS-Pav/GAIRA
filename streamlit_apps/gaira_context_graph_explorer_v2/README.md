# GAIRA Context Graph Explorer · v2

Specific cohort-aware extension of the v1 Context Graph Explorer.

v2 splits the v1 condition_family axis into **27 specific condition labels**
(HCC, CCA, LM, COVID_vs_Healthy, OWD, NWD, hepatotoxicity_D2_C40_vs_C0,
smallEV_Probe1/2, P2_HCC_vs_NC, ...) so we can inspect each cohort's
biochemical program on its own.

v1 is **untouched** — `streamlit_apps/gaira_context_graph_explorer/`.

## Run

```bash
streamlit run streamlit_apps/gaira_context_graph_explorer_v2/app.py
```

Default port 8501. The first launch derives 8 cache tables under
`cache/` from the v1 discovery output; subsequent launches reuse them
via Streamlit's `@st.cache_data`.

## Tabs

1. **Overview** — headline metrics + per-condition coverage table.
2. **Condition → Axis programs** — bipartite condition × BSV axis
   network with broad-vs-specific granularity toggle, sample-type
   filter, direction filter, edge-weight threshold, top-N control,
   auto-derived per-condition program bullets.
3. **Specific Condition Explorer** — pick one cohort and see:
   - BSV bar / radar profile
   - Optional reference-cohort overlay (e.g. OWD vs NWD, HCC vs LM)
   - Top recurrent MSS candidate panel
   - Cohort-restricted Sankey
   - Source-dataset table
   - **Hepatotoxicity dose trajectory** (live read of `shine_cohort_bsv_means_v1.csv`)
   - **Small-EV axis-rank comparison** (when applicable)
4. **Hierarchical context flow** — 5-layer Sankey (sample type → dataset →
   specific condition → axis × direction → MSS) with sample-type /
   condition / max-edge / MSS-layer / "show only top paths" controls,
   and a **Top emergent paths** table beside it (`path_score`,
   `confidence_tier`).
5. **MSS transfer · candidate layer** — three sub-tabs (All / EV-only /
   Serum-only) bipartite MSS ↔ {dataset / specific_condition /
   sample_type}, classification filter, plus the explicit "why MSS
   transfer is mostly EV" framing.
6. **EV vs Serum comparison** — five panels:
   axis recurrence heatmap · direction-consistency heatmap · MSS
   recurrence heatmap · per-axis EV-vs-serum contrast table · paired
   EV-vs-serum mean-effect scatter (G01–G11 as points, quadrants =
   directional agreement).
7. **Context embeddings** — UMAP / PCA over the 11-dim BSV-effect
   vector per dataset; colour by sample type / condition_family /
   specific_condition / caveat burden; size by evidence count or caveat
   burden; optional convex-hull or covariance-ellipse overlay per
   sample type; nearest-neighbour table.

## Cache (auto-generated on first launch)

```
cache/
├── condition_axis_edges_broad.csv
├── condition_axis_edges_specific.csv
├── condition_mss_edges.csv
├── condition_specific_events.csv          # events_long + specific_condition column
├── context_embedding_points_v2.csv
├── emergent_paths_ranked.csv
├── sample_type_axis_summary.csv
└── sample_type_mss_summary.csv
```

## Strict invariants

- GAIRA core unchanged.
- No GAIRA scoring rerun.
- `gaira_context_graph_explorer/` (v1) is **never** modified by v2.
- Missing-artifact tolerance: every loader returns `None` and the
  renderer falls back to a soft warning card.
