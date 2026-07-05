# GAIRA Command Center — **v1 (stable)**

The first stable Streamlit demo. Tab 1 (Overview / Evidence Stack) and
Tab 2 (Motif · MSS · BSV construction) — interactive but readable.

This is the **stable readable** branch. The experimental hierarchical
family-first redesign lives in `gaira_command_center_v2/`.

## Run

```bash
streamlit run streamlit_apps/gaira_command_center_v1/app.py
```

Then open http://localhost:8501.

## Tabs

1. **Overview / Evidence Stack** — pipeline schematic, eight evidence layers,
   dataset coverage, core principles, demo roadmap.
2. **Motif · MSS · BSV construction**
   - Concept overview (3-level hierarchy)
   - MSS v4.1 → v4.2 evolution
   - Interactive UMAP (MSS / MOTIF toggle, hulls, hover)
   - Side-by-side MSS vs Motif comparison
   - Annotated dendrograms (image + interpretation)
   - BSV saliency heatmap + shared-band overlay
   - Hybrid BSV evidence flow + supporting confusion / confidence figures
   - Tab-3 link card

## Strict invariants

- GAIRA core unchanged.
- No GAIRA scoring rerun inside the app.
- Every artifact load is path-configurable + missing-tolerant.

## Versioning

- `gaira_command_center_v1/` — this folder, the stable readable demo.
- `gaira_command_center_v2/` — experimental hierarchical Tab 2 redesign.

To run v2: `streamlit run streamlit_apps/gaira_command_center_v2/app.py`
