# Phase C1 — GAIRA Query & Explanation Engine Summary

## What Was Built

A working GAIRA query system with 4 components:

### 1. Query Router ([graph/phaseC1_query_router.py](graph/phaseC1_query_router.py))
- Rule-based keyword matching (no LLM)
- Supports: condition, peak, theme, chemistry query types
- Vocabulary: 30+ conditions, 12 themes, 15+ chemistry terms
- Returns `ParsedQuery` with type, entities, confidence

### 2. Graph Query Engine ([graph/phaseC1_query_engine.py](graph/phaseC1_query_engine.py))
- Generates and executes Cypher queries against Neo4j
- 4 Cypher templates (condition, peak, theme, chemistry)
- Returns structured `GraphResult` with motifs, themes, biomolecules, evidence sample
- Includes visualization Cypher for Neo4j Browser copy-paste

### 3. Scoring Layer ([graph/phaseC1_scoring.py](graph/phaseC1_scoring.py))
- Deterministic scoring (no LLM)
- Weights: direct support (+3), inferred (+1.5), motif linkage (+2), source diversity (+1)
- Generic hub downweighting (0.5x for "stretch", "vibration", "ring")
- Confidence levels: high/medium/low based on evidence depth + source diversity
- Caveat generation for low evidence, generic nodes, single-source

### 4. Explanation Engine ([graph/phaseC1_templates.py](graph/phaseC1_templates.py))
- TrustGraph-style structured output (sections A-F)
- Pure template formatting (no LLM)
- Text rendering for display

### 5. Streamlit Demo ([app/gaira_query_demo.py](app/gaira_query_demo.py))
- Text input for queries
- Metrics display (evidence rows, sources, motifs, themes)
- Ranked theme table with scores and confidence
- Biomolecule and functional group tables
- Sample evidence rows
- Caveats display
- Expandable: graph traversal details, Neo4j viz query, full text explanation

## How to Run

```bash
# Ensure Neo4j is running with the GAIRA graph imported
streamlit run app/gaira_query_demo.py
```

Configure Neo4j connection in the sidebar (default: bolt://localhost:7687).

## Architecture

```
User Query (text)
    ↓
Query Router (keyword matching)
    ↓
Graph Query Engine (Cypher → Neo4j)
    ↓
Scoring Layer (deterministic weights)
    ↓
Explanation Engine (template formatting)
    ↓
Streamlit UI (tables + text + Neo4j viz query)
```

All reasoning is deterministic. No LLM is involved. Every score is traceable to graph edges and evidence rows.

## Supported Query Types

| Type | Example | What It Traverses |
|---|---|---|
| Condition | "What about HCC?" | Condition → Motifs → Evidence → Themes |
| Peak | "Peak at 1005 cm-1?" | Peak → Evidence → Assignments → Themes/Conditions |
| Theme | "Lipid signal?" | Theme → Assignments → Evidence → FGs/Conditions |
| Chemistry | "Amide I to biology?" | FG → Inferred Themes + Direct Assignments |

## Next Phase (C2)

Add LLM presentation layer:
- Take the structured explanation output from C1
- Use Claude/GPT to generate natural-language summaries
- Keep all reasoning deterministic (C1); only format with LLM (C2)
- Add conversational follow-up capability
