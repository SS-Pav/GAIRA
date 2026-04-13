# GAIRA LFM v1 — Phase 5: Trust Graph + Evidence Chain

## What Phase 5 Adds

A query-local trust graph and evidence grounding summary that makes the answer pipeline transparent:

- **Trust graph**: visual directed graph showing Query → Evidence Nodes → Packet → Response
- **Groundedness assessment**: programmatic summary of how grounded the answer is
- **Evidence composition metrics**: counts of grounded evidence vs context vs summaries
- **Hover details**: each evidence node shows its title, tier, score, source, and text preview

## Trust Graph Concept

For each query, a small directed graph is built:

```
[Query] → [Evidence Node 1] → [Packet] → [Response]
         → [Evidence Node 2] ↗
         → [Evidence Node 3] ↗
         → ...
```

Each evidence node is color-coded by tier:
- Green: Grounded Evidence
- Orange: Domain Context
- Purple: Benchmark Summary
- Blue: Spectral Query
- Gray: Meta Summary
- Teal: Evidence Packet
- Red: GAIRA Response

## Node Schema

Each node carries:
| Field | Description |
|---|---|
| id | Unique identifier |
| label | Display name (truncated) |
| node_type | Evidence tier or structural role |
| color | Tier-based color |
| size | Proportional to score (evidence) or fixed (structural) |
| title | HTML hover tooltip with full metadata |
| source | Source file path (evidence nodes only) |
| score | Retrieval score (evidence nodes only) |

## Edge Schema

| From | To | Meaning |
|---|---|---|
| query | evidence node | "this evidence was retrieved for this query" |
| evidence node | packet | "this evidence was included in the packet" |
| packet | response | "the response was synthesized from this packet" |

## Groundedness Assessment

Programmatically computed from the evidence tier mix:

| Assessment | Condition |
|---|---|
| strongly grounded | >= 50% grounded_evidence |
| well grounded with context | >= 50% grounded + domain_context |
| mixed evidence base | no single tier dominates |
| summary-heavy | >= 50% benchmark/spectral/meta |

## How to Run

```bash
cd /Users/suraj/projects/GAIRA
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v1_text_query.py
```

## Rendering

The trust graph is rendered using **pyvis** (networkx-based) with hierarchical left-to-right layout. The HTML is embedded in Streamlit via `components.html()`. Physics is disabled for stable layout.

## What Is Deferred

- Full corpus graph (only query-local for now)
- Graph database integration
- Animated graph traversal
- Response section ↔ evidence node linking
- Spectral query graph integration
- Graph editing or manual node curation
