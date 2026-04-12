# Phase C1.1 — Query/Explanation Refinement Summary

## What Changed from C1 to C1.1

### 1. Specificity-Aware Scoring
Themes are now scored on **support + specificity** with a geometric-mean blend, not raw support alone. Broad ubiquitous themes (protein, lipid) are explicitly penalized for low condition specificity. The UI exposes all three score components.

### 2. Enhanced Caveat Engine
7 caveat rules now fire based on:
- Low evidence / single source
- Broad themes dominating top ranks
- Generic chemistry hubs in top results
- Zero inferred support on chemistry queries
- Broadly-shared motifs dominating
- Low specificity in top themes

### 3. Evidence Quality Filtering
Sample evidence rows are now quality-ranked (length + assignment level + fragment/noise penalties) with source diversity preference. No more truncated fragments or classifier text in displayed evidence.

### 4. Motif Prominence
Motifs now appear as a dedicated table with subfamily, family, member count, and condition-enrichment assessment (condition-enriched / broadly-shared / sparse).

### 5. Embedded Graph Preview
PyVis-powered interactive subgraph embedded directly in Streamlit. Dark background, color-coded nodes, solid vs dashed edges (direct vs inferred), hover tooltips, drag-to-arrange. Limited to 60 nodes for readability.

### 6. Improved Cypher Panel
Copy-paste Cypher with rendered parameter values. Step-by-step Neo4j Browser instructions included.

### 7. Updated Explanation Format
Sections A-I (was A-F):
A. Query understanding
B. Grounding
C. Graph traversal
D. Ranked themes (with support + specificity)
E. Top motifs
F. Top biomolecules
G. Top functional groups
H. Sample evidence (quality-filtered)
I. Caveats (enhanced)

## Files Created/Updated

### Core Logic (updated):
- `graph/phaseC1_scoring.py` — specificity + caveats + evidence quality
- `graph/phaseC1_templates.py` — sections A-I + motif formatting

### New Components:
- `app/graph_preview.py` — PyVis embedded graph builder
- `app/gaira_query_demo.py` — full demo with graph preview

### Documentation (7 new):
- `graph/phaseC11_scoring_logic.md`
- `graph/phaseC11_caveat_rules.md`
- `graph/phaseC11_evidence_selection_rules.md`
- `graph/phaseC11_motif_summary_logic.md`
- `graph/phaseC11_graph_preview_design.md`
- `graph/phaseC11_neo4j_inspection_instructions.md`
- `graph/phaseC11_specificity_rules.csv`

## How to Run
```bash
streamlit run app/gaira_query_demo.py
```

## Next Phase (C2)
Add LLM presentation layer — Claude generates natural-language summaries from the deterministic C1.1 explanation output. All reasoning stays deterministic; only the final narration uses LLM.
