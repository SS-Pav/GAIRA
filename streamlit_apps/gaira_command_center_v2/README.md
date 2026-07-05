# GAIRA Command Center — **v2 (experimental)**

Redesigned Tab 2 with a concept-first storyline. The broken family-first hull
plot from the earlier rewrite has been removed; family overlays are now an
opt-in advanced toggle.

This is the **experimental** branch. The stable readable version lives in
`gaira_command_center_v1/`.

## Run

```bash
streamlit run streamlit_apps/gaira_command_center_v2/app.py
```

Then open http://localhost:8501.

## Tabs

1. **Overview / Evidence Stack** — same as v1.
2. **Motif · MSS · BSV — v2 redesign**
   - A · Representation hierarchy (concept diagram)
   - B · 11-axis BSV taxonomy (compact card grid — no huge dataframe)
   - C · BSV saliency map (canonical labels OFF by default; axis inspector collapsed)
   - D · Shared bands & ambiguity (green/orange/red traffic-light)
   - E · Axis overlap network (manual chemistry-grouped layout, labels outside nodes)
   - F · MSS / motif UMAP (clean — only ~9 major clusters labelled; legend collapsed)
   - G · Annotated dendrograms (numbered callouts + summary table)
   - H · Hybrid BSV evidence flow

## Key fixes vs the broken intermediate

- No more family-first hull plot at the top. Optional `experimental` toggle in advanced controls puts family ellipses on the UMAP only.
- No "Top 10 bands driving G05" table on first load. Collapsed under the saliency map as `Inspect one BSV axis`.
- Saliency canonical band labels OFF by default.
- Axis overlap network uses manual chemistry-grouped positions (no random force-layout); short G-id inside nodes, full family name outside, only top-60% edges drawn.
- MSS / motif UMAP labels only ~9 known major classes; legend collapsed (toggle in advanced).

## Versioning

- `gaira_command_center_v1/` — stable readable demo.
- `gaira_command_center_v2/` — this folder.
