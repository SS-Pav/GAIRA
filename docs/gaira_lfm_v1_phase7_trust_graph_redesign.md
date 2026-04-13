# GAIRA LFM v1 — Phase 7: Trust Graph Redesign

## Problem

The Phase 6 trust graph (pyvis) was informative but visually worse than earlier versions:
- Long text crammed into node labels and hover tooltips
- Too many edge types rendered simultaneously (dashed back-links from every section to every evidence item)
- Felt like a force-directed spaghetti diagram, not a staged reasoning flow

## What Changed

### Architecture: pyvis → plotly

Replaced pyvis (HTML-embedded, hard to control) with plotly (native Streamlit rendering, full position control). This enables:
- Fixed column positions for staged layout
- Batched edge rendering (7 traces vs 70)
- Dark theme with clean contrast
- No external HTML embedding

### Graph structure: 4-column staged flow

```
Column 0       Column 1              Column 2         Column 3
─────────    ─────────────────     ────────────     ──────────────
 Query    →   Evidence nodes   →   Theme nodes   →   Response
              (by tier)            (auto-detected)    sections
```

- **Column 0 — Query**: single node, user's question
- **Column 1 — Evidence**: retrieved items, grouped by tier (grounded first), short labels (≤25 chars)
- **Column 2 — Themes**: auto-extracted biochemical themes (nucleic acid, lipid, protein, etc.) detected by keyword presence across evidence
- **Column 3 — Response sections**: Summary, Themes, Strongest, Supporting, Caveats, Confidence

### Theme extraction (new)

8 biochemical theme categories are detected across retrieved evidence:
- nucleic acid, protein, aromatic AA, lipid/membrane, nucleotide bases, glycan, redox, substrate caveat

Each theme node connects to the evidence items that mention it, and to response sections that share evidence support. This creates a natural grouping layer between raw evidence and the synthesized answer.

### Long text: moved out of the graph

| Component | Where it lived (Phase 6) | Where it lives now |
|---|---|---|
| Evidence full text | pyvis hover tooltip | Node Inspector expander below graph |
| Evidence metadata | pyvis hover tooltip | Sidebar + Inspector |
| Section support details | Inline in graph | "Section support map" expander |
| Theme details | Not present | "Detected themes" expander |

Graph labels are now ≤25 characters. All detail is in the inspector panel.

### Edge design: two tiers only

| Edge type | Style | Connects |
|---|---|---|
| Main flow | Light lines (15% opacity) | Adjacent columns only |
| Cross-column | Very faint lines (6% opacity) | Theme → section links |

No dashed back-links. No per-section evidence links rendered in the graph itself — those are in the inspector.

### Visual design

- Dark background (#1a1a2e) for contrast
- Tier-colored nodes (green=grounded, orange=context, purple=benchmark, blue=spectral)
- Structural nodes (query, theme, section) in white/teal/red
- Node shapes: circle (evidence), square (query), hexagon (theme), diamond (section)
- Horizontal legend below graph
- Column headers above graph
- Compact hover tooltips (not paragraphs)

## UI structure

The Evidence Chain area now has 4 parts:

1. **Trust summary strip** — groundedness label, counts, conflicts, explanation
2. **Plotly trust graph** — staged 4-column visualization
3. **Node inspector** — expandable details for evidence, themes, section support
4. **Section support** — compact "Supported by" annotations under each response section

## How to Run

```bash
cd /Users/suraj/projects/GAIRA
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v1_text_query.py
```

## Performance

- 7 plotly traces (2 edge batches + 5 node type groups) vs 70 individual pyvis traces
- No external HTML embedding — native Streamlit plotly rendering
- Graph renders in <100ms

## What Is Deferred

- Full corpus graph
- Sentence-level attribution
- Embedding-based retrieval
- Graph database backend
- Spectral query integration
- Animated graph traversal
