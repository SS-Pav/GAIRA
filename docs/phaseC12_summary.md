# Phase C1.2 — UI Bugfixes + Graph Readability Summary

## Changes

### 1. Strikethrough Bug Fixed
**Root cause**: Theme names containing underscores (e.g., `nucleic_acid_(DNA)`) were being interpreted as Markdown emphasis markers, producing strikethrough or italic artifacts.

**Fix**: All user-facing names now pass through `_safe_name()` which replaces underscores with spaces and strips `~` and `*` characters. Applied in templates, motif labels, and the Streamlit app.

### 2. Top / Secondary Theme Split
Themes are now split into two sections:
- **Top Themes** (first 4): displayed prominently with full scoring columns
- **Secondary Themes** (4-8): in a collapsible expander, lighter formatting

No theme is struck through or visually deprecated. All are visible; priority is communicated through placement and score.

### 3. Graph Preview Improvements
- **Anchor node is larger and prominent** (size 35-40 vs 10-22 for others)
- **Cleaner layout**: tighter gravity, better spring constants, fewer max nodes (50)
- **Better labeling**: `_clean_label()` replaces underscores, truncates sensibly
- **Improved legend**: dark background matches graph, explains solid/dashed/thick-red edge types
- **Less clutter**: max 12 evidence nodes, 6 motifs, 5 themes, 4 biomolecules
- **Edge styling**: LINKED_TO edges are thick red (2.5px) to emphasize condition connections

### 4. Demo Readability Polish
- Functional groups moved to collapsible expander (less visual noise)
- Evidence table truncates meaning cleanly at 70 chars with underscore cleanup
- Motif enrichment labels use spaces not hyphens
- All `_` replaced with spaces in display text throughout
- Query entity displayed with spaces, not underscores
- Confidence shown as uppercase tags (HIGH, MED, LOW) for quick scanning

### 5. Cypher Panel
- Labeled "Inspect in Neo4j Browser" (clearer intent)
- Step-by-step numbered instructions
- Caption explains it's for full-depth exploration beyond the simplified preview

## Files Updated

| File | Changes |
|---|---|
| `graph/phaseC1_templates.py` | `_safe_name()`, Top/Secondary split, cleaner Markdown |
| `app/graph_preview.py` | Anchor emphasis, edge width, legend restyle, node caps |
| `app/gaira_query_demo.py` | Theme split UI, FG in expander, cleaner labels throughout |

## Files Created

| File | Purpose |
|---|---|
| `docs/phaseC12_summary.md` | This summary |

## What Did NOT Change
- Routing logic (phaseC1_query_router.py)
- Cypher templates (phaseC1_query_engine.py)
- Scoring logic (phaseC1_scoring.py — only the evidence selection bugfix from C1.1)
- Neo4j graph structure
- Deterministic reasoning approach
