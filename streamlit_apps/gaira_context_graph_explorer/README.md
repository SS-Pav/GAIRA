# GAIRA Context Graph Explorer

Visualisation-only Streamlit app over the discovery output at
`/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_context_graph_discovery_v1/`.

This app does **not** rescore spectra or rerun GAIRA pipelines. It loads
the precomputed evidence-event tables and renders five interactive views
of the recurring biochemical structure across pilots.

## Run

```bash
streamlit run streamlit_apps/gaira_context_graph_explorer/app.py
```

Default port 8501. Use `--server.port=N` to pick another.

## Tabs

1. **Overview** — headline metrics, axis transfer table, MSS classification
   table, caveat summary, and an inline raw-events explorer.
2. **Condition → Axis network** — bipartite layout. Condition families on
   the left, BSV axes (G01–G11) on the right. Edge weight = recurrence
   × |effect|; colour = dominant direction (red up · blue down · grey
   stable). Threshold slider, weight kind, direction filter.
3. **Hierarchical context (Sankey)** — five layers: sample type → dataset
   → condition family → BSV axis × direction → MSS candidate. Filter by
   sample type / dataset, cap edges per layer, toggle MSS layer.
4. **MSS transfer** — MSS candidates on the left ↔ chosen grouping
   (dataset / condition_family / sample_type) on the right. Classification
   filter (TRANSFERABLE / SAMPLE_TYPE_SPECIFIC / CANDIDATE_ONLY).
5. **Context embedding** — datasets in BSV-effect space (UMAP / PCA),
   colour by sample type / condition family / caveat burden; size by
   evidence count or caveat burden.

## Requirements

`streamlit`, `pandas`, `plotly`, `pyyaml`. NetworkX is optional (not used
in v1; primary visuals are Plotly Sankey / bipartite scatter).

## Strict invariants

- Read-only over the discovery output.
- Missing-artifact tolerance: every loader returns `None` and the renderer
  shows a soft warning card.
- All visualisation-only computations (bipartite positions, Sankey layer
  bookkeeping) are explicit in the utility code.
