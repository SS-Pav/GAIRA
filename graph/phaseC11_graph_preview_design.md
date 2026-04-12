# Phase C1.1 — Embedded Graph Preview Design

## Technology
PyVis (Python library wrapping vis.js) — generates interactive HTML graph embeds.

## Subgraph Strategy
The preview shows a **simplified interpretive core**, not the full query traversal:
- Max 60 nodes total
- Max 15 evidence rows (sampled)
- Top 6 themes, 8 motifs, 5 biomolecules, 4 FGs

## Node Styling

| Type | Color | Shape | Size |
|---|---|---|---|
| Condition | Yellow (#FDD835) | Diamond | 30 |
| Motif | Green (#66BB6A) | Circle | 22 |
| BiochemicalTheme | Teal (#80CBC4) | Star | 25 |
| Biomolecule | Magenta (#CE93D8) | Star | 20 |
| EvidenceRow | Light blue (#90CAF9) | Circle | 12 |
| FunctionalGroup | Orange (#FFCC80) | Triangle | 20 |
| Peak | Red (#EF5350) | Square | 18 |
| Paper/Source | Gray (#BDBDBD) | Circle | 10 |

## Edge Styling
- Solid lines = direct evidence
- Dashed lines = inferred from mapping table
- Color-coded by relationship type

## Interaction
- Hover to see node type and details
- Drag to rearrange layout
- Physics simulation auto-arranges
- Dark background (#1a1a2e) for visual clarity

## Limitations
- Simplified view — not all edges shown
- Use Neo4j Browser for full-depth inspection
- Legend provided below the graph
