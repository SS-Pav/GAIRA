# BSV v1 — Visualization Guide

## Radar Plot
Shows query vs comparator BSV as overlaid polygons on a shared polar axis.

- **Blue polygon**: query condition (e.g., HCC)
- **Red polygon**: comparator condition (e.g., healthy control)
- **Axes**: 8 BSV components, normalized 0-1
- **Components with zero coverage** are excluded from the radar

**Reading the radar**: Where blue extends beyond red, the query condition has stronger biochemical signal for that component. Where red extends beyond blue, the comparator is stronger.

## Delta Bar Plot
Shows the difference (query - comparator) for each component.

- **Blue bars** (positive): query has more support than comparator
- **Red bars** (negative): comparator has more support
- **Gray bars** (near zero): no meaningful difference

**Reading the delta**: Taller bars = bigger biochemical state differences. The bar chart is more precise than the radar for exact comparisons.

## When BSV Components Appear "Flat"
If all components are similar between query and comparator, this means:
1. The conditions share similar biochemistry (possible — many liver diseases have overlapping signatures)
2. The evidence base doesn't distinguish them at the motif level (more common with sparse evidence)
3. The BSV components are too broad (possible — component definitions may need refinement)

## Streamlit Integration
Both plots are rendered as matplotlib figures via `st.pyplot()`. They are side-by-side for comparative queries.
