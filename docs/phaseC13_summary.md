# Phase C1.3 — Comparative Query Modes Summary

## What Changed

### Query Router
- **New**: pairwise comparison (`Compare X vs Y`)
- **New**: one-vs-rest enrichment (`What is enriched in X vs rest?`)
- **New**: `ParsedQuery.query_mode` field (single / pairwise / one_vs_rest)
- **New**: `ParsedQuery.comparator` field for second condition
- Existing single/peak/theme/chemistry queries unchanged

### Scoring Layer
- **New**: comparative enrichment metrics (enrichment_ratio, contrast_score)
- **New**: theme interpretation labels (enriched / associated / shared / depleted)
- **New**: motif comparative interpretation (enriched / comparator-associated / shared)
- **New**: score boost for enriched themes (1.3x), penalty for depleted (0.5x)
- **New**: comparative caveats ("shared between query and comparator", "no enriched themes found")
- Existing single-condition scoring unchanged

### Templates
- **New**: comparative output sections (D_enriched, D2_associated, D3_shared, D4_depleted)
- **New**: enrichment ratio display in theme tables
- **New**: comparator member counts in motif tables
- Single-condition output structure unchanged (D_top, D2_secondary)

### Demo UI
- **New**: query mode badge (ASSOCIATIVE / COMPARATIVE / ENRICHMENT)
- **New**: comparative theme layout with enriched/shared/depleted sections
- **New**: motif table shows comparator members and interpretation
- **New**: Cypher panel labeled as associative or comparative subgraph
- **New**: sidebar shows comparative query examples
- Single-condition UI unchanged

### Graph Preview
- No structural changes to PyVis layout
- Graph shows query-side subgraph; enrichment is in tables

## Files Updated (5)
- `graph/phaseC1_query_router.py`
- `graph/phaseC1_query_engine.py`
- `graph/phaseC1_scoring.py`
- `graph/phaseC1_templates.py`
- `app/gaira_query_demo.py`

## Files Created (7)
- `docs/phaseC13_query_modes.md`
- `docs/phaseC13_example_outputs.md`
- `docs/phaseC13_summary.md`
- `graph/phaseC13_comparative_scoring_logic.md`
- `graph/phaseC13_discriminative_metrics.csv`
- `graph/phaseC13_graph_semantics.md`
- `graph/phaseC13_motif_contribution_logic.md`

## Design Principles Preserved
- All reasoning is deterministic (no LLM)
- Direct vs inferred edges remain distinct
- "Enriched" means graph-enriched, not clinically validated
- No hardcoded comparator panel — query determines comparison
- Existing single-condition behavior fully preserved
