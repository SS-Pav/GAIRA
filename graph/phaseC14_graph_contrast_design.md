# Phase C1.4 — Graph Contrast Design

## Current State
The graph preview shows the query-side subgraph. For comparative queries, the contrast is reflected in the scoring tables (enriched/shared/depleted), not in the graph layout.

## Why Not a Dual-Anchor Layout
A dual-anchor graph (query on left, comparator on right, shared in middle) would require:
- Two separate subgraph retrievals rendered together
- Complex force-directed positioning to maintain L/R separation
- Significant PyVis customization

This is deferred to a future phase.

## Current Contrast Semantics
- The graph preview is labeled as "query-side subgraph"
- The note below the legend states: "Simplified preview. Use Neo4j Browser for full graph inspection."
- For comparative queries, the Cypher query includes both conditions so Neo4j Browser can show the full comparative structure

## What the User Should Know
1. The graph preview shows what's connected to the **query** condition
2. The **scoring tables** show the comparative enrichment (enriched/shared/depleted)
3. For true visual comparison, use the Neo4j Browser with the provided Cypher query
